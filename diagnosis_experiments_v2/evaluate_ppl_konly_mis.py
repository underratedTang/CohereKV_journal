#!/usr/bin/env python3
"""Bulk-prefix KV compression followed by teacher-forced continuation-only PPL.

A context prefix of P tokens is prefetched without compression. Immediately
at the P-token boundary, each method compresses once to
B = round(P * KV_budget_fraction) physical slots per KV head. Afterwards the
cache is never recompressed: the continuation is evaluated against the single
compressed prefix state, with Count-weighted attention and logical RoPE.

For strictly post-compression scoring, one ground-truth continuation token is
used as a warm-up/query and NOT scored. The N requested scored targets are the
following N continuation tokens, predicted by N queries that all run against
the compressed KV cache. All methods use identical token windows and masks.
The warm-up avoids mistakenly scoring the first target from a pre-compression
prefix logit. Continuation evaluation is teacher-forced, not free generation.

The native diagnosis E-E graph uses R=D=0. This mode performs a single
compression from an uncompressed cache (Count=1 everywhere) rather than
repeated multi-round compression.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[1]
TASK_ROOT=ROOT/'task_evaluation'
if str(TASK_ROOT) not in sys.path: sys.path.insert(0,str(TASK_ROOT))
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from evaluate_ppl_kvmerge import (KVCompressingDynamicCache, _replace_dynamic_cache_inplace,
    apply_runtime_patch_if_needed, force_attn_hidden_and_compress, get_eval_data)
from calibrate_median_reference import validate_payload, validate_value_payload, validate_query_bounds
from core_model_utils import parse_dtype
from cosine_mis_plan import select_split,route_top1
from ppl_sweep_utils import (parse_floats_csv,routing_specs,setting_key,
    computation_key,config_comparison_rows,atomic_write_csv,formatted_ppl_summary,
    finalize_ppl_group_deltas)
from k_only_mis_drop import ee_mis,fuse_weighted_slots
from mis_sweep_settings import sweep_settings,sweep_label
from method_registry import (selected_methods,selected_families,family_for_method,
    effective_method,is_sink_gated,graph_is_importance_aware,priority_for_family)
from routing_load_stats import routing_coverage,concentration
from models.common.mymethod_common import recent_window_importance




def fixed_cache_capacity(retention_ratio: float, context_length: int) -> int:
    """Physical KV slots after the ONE bulk compression, relative to prefix P."""
    ratio=float(retention_ratio)
    if not (math.isfinite(ratio) and 0.0 < ratio < 1.0):
        raise ValueError('KV budget ratio must be finite and strictly between 0 and 1')
    if int(context_length)<2:
        raise ValueError('Context prefix must contain at least two tokens')
    return max(1,int(round(ratio*int(context_length))))


@torch.no_grad()
def evaluate_bulk_perplexity(
        model: nn.Module, data: torch.Tensor, context_len: int,
        continuation_len: int, device: torch.device,
        amp_dtype: Optional[torch.dtype]=None, prefill_chunk_size: int=128,
        cache_factory: Optional[Callable[[], DynamicCache]]=None) -> float:
    """Only score N targets *after* a single P-token prefix KV compression.

    Window: [P prefix][1 unscored continuation warm-up][N scored targets].
    To predict the N targets, model inputs are tokens at positions P..P+N-1;
    targets are P+1..P+N. Every scored logit therefore comes from a
    post-compression forward, including the first one. No future token can
    enter the prefix's compression decision.
    """
    if context_len<2 or continuation_len<1 or prefill_chunk_size<1:
        raise ValueError('Require context_len>=2, continuation_len>=1, chunk>=1')
    window_length=context_len+1+continuation_len
    if data.ndim!=2 or data.shape[0]!=1 or data.shape[1]%window_length:
        raise ValueError('Expected [1, num_windows*(P+1+N)] token stream')
    n_windows=int(data.shape[1]//window_length)
    if n_windows<1:
        raise ValueError('No complete bulk-prefix evaluation window')
    nll_total=torch.zeros((),device=device,dtype=torch.float64)
    tokens_total=0
    autocast_enabled=amp_dtype is not None and device.type=='cuda'
    for sequence_idx in tqdm(range(n_windows),desc='Evaluating bulk-prefix continuation perplexity'):
        sample=data[:,sequence_idx*window_length:(sequence_idx+1)*window_length].to(device)
        cache=cache_factory() if cache_factory is not None else DynamicCache()
        with torch.amp.autocast('cuda',enabled=autocast_enabled,
                                dtype=amp_dtype or torch.float32):
            # Prefill: same original P tokens for all methods. The cache
            # subclass compresses on the LAST prefill forward only.
            for start in range(0,context_len,prefill_chunk_size):
                end=min(context_len,start+prefill_chunk_size)
                out=model(sample[:,start:end],use_cache=True,past_key_values=cache)
                if out.past_key_values is not cache:
                    raise RuntimeError('Prefill did not reuse the same KV cache object')
            if hasattr(cache,'bulk_compression_done') and not cache.bulk_compression_done:
                raise RuntimeError('The requested one-time prefix compression did not occur')
            # Position P is an unscored warm-up input; do NOT use the prefix's
            # last (pre-compression) logit to predict it. All N scored targets
            # are predicted using attention over the already compressed cache.
            for start in range(context_len,context_len+continuation_len,prefill_chunk_size):
                end=min(context_len+continuation_len,start+prefill_chunk_size)
                out=model(sample[:,start:end],use_cache=True,past_key_values=cache)
                if out.past_key_values is not cache:
                    raise RuntimeError('Continuation replaced the KV cache unexpectedly')
                labels=sample[:,start+1:end+1]
                logits=out.logits.float()
                nll=torch.nn.functional.cross_entropy(
                    logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='sum')
                nll_total+=nll.double()
                tokens_total+=labels.numel()
        if hasattr(cache,'bulk_compression_done') and not cache.bulk_compression_done:
            raise RuntimeError('Compression was undone in continuation')
    expected=n_windows*continuation_len
    if tokens_total!=expected:
        raise RuntimeError(f'Scored {tokens_total}, expected {expected} continuation tokens')
    return torch.exp(nll_total/tokens_total).item()


class JournalCompressingCache(KVCompressingDynamicCache):
    """One-time bulk K-only MIS, optional train-frozen sink-only KV-head gating."""
    def __init__(self, *, model, method, family, budget, routing, cutoff, ee_lambda,
                 reference, sink_gate, score_window, graph_backend, chunk_size, stats,
                 context_length):
        capacity = fixed_cache_capacity(budget, context_length)
        super().__init__(model=model,method='mymethod',family=family,
            compress_ratio=1.-float(budget),sim_thr=-1,method_sim_thrs=None,
            keepkv_config={},mymethod_config={},centroidkv_config={},
            min_kv_len=2,compress_every=1,stats=stats,task_eval_dir=TASK_ROOT,
            target_kv_len=capacity)
        self.strategy=method
        self.routing=routing
        self.cutoff=float(cutoff)
        self.ee_lambda=float(ee_lambda)
        self.reference=np.asarray(reference,dtype=np.float64)
        self.sink_gate=np.asarray(sink_gate,dtype=bool)
        if self.sink_gate.shape!=self.reference.shape:
            raise ValueError('Train-frozen sink mask and K reference shapes disagree')
        self.budget=float(budget)
        self.budget_tokens=capacity
        self.context_length=int(context_length)
        self.bulk_compression_done=False
        self.score_window=int(score_window)
        self.graph_backend=graph_backend
        self.chunk_size=int(chunk_size)
        self.counts=None
        self.slot_positions=None
        self.last_nkeep=0
        self._stats.setdefault('compress_rounds',0)
        self._stats.setdefault('sink_gated_mis_head_rounds',0)
        self._stats.setdefault('sink_gated_mergeall_head_rounds',0)
        self._stats.setdefault('routed_slots',0)
        self._stats.setdefault('routed_token_mass',0.0)
        self._stats.setdefault('rejected_slots',0)
        self._stats.setdefault('mis_rejected_token_mass',0.0)
        self._stats.setdefault('k_only_rejected_token_mass',0.0)
        self._stats.setdefault('importance_rejected_token_mass',0.0)
        self._stats.setdefault('mis_routed_importance_sum',0.0)
        self._stats.setdefault('mis_accepted_importance_sum',0.0)
        self._stats.setdefault('mis_rejected_importance_sum',0.0)
        self._stats.setdefault('mis_conflict_edges',0)
        self._stats.setdefault('mis_original_ee_conflict_edges',0)
        self._stats.setdefault('mis_ea_rejected_slots',0)
        self._stats.setdefault('represented_reject_count_mass',0.0)
        self._stats.setdefault('dropped_slots',0)
        self._stats.setdefault('dropped_token_mass',0.0)
        self._stats.setdefault('within_budget_calls',0)
        self._stats.setdefault('compression_boundary_calls',0)

    @torch.no_grad()
    def maybe_compress(self, attentions, hidden_states):
        self._forward_calls += 1
        if not self.key_cache:
            raise RuntimeError('Model did not create a KV cache')
        n=int(self.key_cache[0].shape[-2])
        logical=int(self._ppl_logical_seen_tokens)
        if self.bulk_compression_done:
            # Continuation is read-only with respect to the compression
            # decision: newly forwarded tokens append to the physical cache,
            # and are never routed/MIS-compressed again.
            self._stats['post_bulk_forward_calls']=self._stats.get('post_bulk_forward_calls',0)+1
            return
        if logical<self.context_length:
            self._stats['prefill_forward_calls']=self._stats.get('prefill_forward_calls',0)+1
            return
        if logical!=self.context_length or n!=self.context_length:
            raise RuntimeError(f'Bulk compression must occur exactly at prefix P={self.context_length}, '
                f'observed logical={logical}, physical={n}')
        if not all(int(k.shape[-2])==n for k in self.key_cache):
            raise RuntimeError('Inconsistent cached key length across layers')
        # One bulk compression at exactly P, never on an intermediate prefill
        # forward and never during the teacher-forced continuation.
        nk=self.budget_tokens
        if n<=nk or n<2:
            raise RuntimeError(f'Expected bulk compression: prefix={n} <= budget={nk}')
        if logical!=n:
            raise RuntimeError(f'At bulk compression physical ({n}) != logical ({logical})')
        if attentions is None or len(attentions)!=len(self.key_cache):
            raise RuntimeError('KeepKV attention score capture missing during PPL')
        before=int(n)
        self._stats.setdefault('first_compression_logical_tokens', logical)
        self._stats.setdefault('first_compression_before_len', before)
        legacy=[];new_counts=[];new_pos=[]
        for li,(key_layer,val_layer,scores) in enumerate(zip(self.key_cache,self.value_cache,attentions)):
            if scores is None or scores.ndim!=4 or tuple(scores.shape[:2])!=tuple(key_layer.shape[:2]) or scores.shape[-1]!=n:
                raise RuntimeError(f'Invalid KeepKV score capture at layer {li}: {None if scores is None else tuple(scores.shape)}')
            if key_layer.shape[0]!=1:
                raise ValueError('This diagnostic PPL supports batch=1 only')
            h=int(key_layer.shape[1])
            positions=(self.slot_positions[li] if self.slot_positions is not None else
                torch.arange(logical-n,logical,device=key_layer.device).expand(1,h,n))
            prev_n=(int(self.counts[li].shape[-1]) if self.counts is not None else 0)
            counts=(torch.cat((self.counts[li].to(key_layer.device),
                torch.ones((1,h,n-prev_n),device=key_layer.device)),dim=-1)
                if self.counts is not None else torch.ones((1,h,n),device=key_layer.device))
            if self.slot_positions is not None:
                newly=torch.arange(logical-(n-prev_n),logical,device=key_layer.device).view(1,1,-1).expand(1,h,-1)
                positions=torch.cat((self.slot_positions[li].to(key_layer.device),newly),dim=-1)
            assert positions.shape==counts.shape==(1,h,n)
            scores=scores.float()
            if scores.shape[-2]>self.score_window:scores=scores[:,:,-self.score_window:]
            importance=recent_window_importance(scores.reshape(h,scores.shape[-2],n),ptr=0,beta=1.0)
            # all KV heads in a layer have same physical count, but different keep indices.
            n_keep=min(n,max(1,nk))
            keep,evict=select_split(key_layer[0].float(),importance,positions[0],n_keep,0,0)
            kout=[];vout=[];cout=[];pout=[]
            for hi in range(h):
                keys=key_layer[0,hi];vals=val_layer[0,hi];cnt=counts[0,hi]
                kk=keep[hi];ee=evict[hi]
                one_keys=keys.float().unsqueeze(0)
                if self.routing=='cosine':
                    cutoff=self.cutoff
                elif self.routing=='l2':
                    cutoff=self.cutoff*float(self.reference[li,hi])
                else:raise ValueError(self.routing)
                best,cos,routed,dist=route_top1(one_keys,kk[None],ee[None],
                    self.routing,cutoff,self.chunk_size)
                # Sink-only ablations select the same native MIS or Merge All
                # at each physical KV head, from TRAIN calibration only.
                strategy=effective_method(self.strategy,bool(self.sink_gate[li,hi]))
                if is_sink_gated(self.strategy):
                    self._stats['sink_gated_mis_head_rounds' if strategy!='merge_all'
                                else 'sink_gated_mergeall_head_rounds']+=1
                cov=routing_coverage(int(routed[0].sum().item()),int(ee.numel()))
                gate_is_mis=strategy not in ('merge_all','importance_weighted_merge_all')
                # A per-head summary records routing and collision load even
                # when the gate uses IW-MergeAll (no MIS graph construction).
                hkey=(li,hi)
                hstat=self._stats.setdefault('head_diagnostics',{}).setdefault(hkey,dict(
                    n_compressions=0,n_mis_rounds=0,n_iw_rounds=0,
                    routed_slots=0,evicted_slots=0,rejected_slots=0,
                    rejected_count_mass=0.,routed_count_mass=0.,
                    routing_coverage_sum=0.,routing_group_hhi_sum=0.,
                    max_group_load_fraction_sum=0.,n_routed_groups_sum=0.,
                    mean_routed_group_size_sum=0.))
                hc=concentration(best[0].detach().cpu().numpy(),routed[0].detach().cpu().numpy(),len(kk))
                hstat['n_compressions']+=1
                hstat['n_mis_rounds' if gate_is_mis else 'n_iw_rounds']+=1
                hstat['evicted_slots']+=int(ee.numel())
                hstat['routed_slots']+=int(routed[0].sum().item())
                hstat['routed_count_mass']+=float(cnt[ee[routed[0]]].float().sum().item())
                hstat['routing_coverage_sum']+=cov
                for metric in ('routing_group_hhi','max_group_load_fraction','mean_routed_group_size','n_routed_groups'):
                    hstat[metric+'_sum']+=hc[metric]
                plan=None
                if strategy not in ('merge_all','importance_weighted_merge_all') and plan is None:
                    strategy_family=family_for_method(strategy)
                    plan=ee_mis(keys.float(),vals.float(),kk,ee,best[0],routed[0],dist[0],
                        float(self.reference[li,hi]),1.0,self.ee_lambda,
                        self.graph_backend,counts=cnt,importance=importance[hi],
                        priority=priority_for_family(strategy_family),
                        importance_aware=graph_is_importance_aware(strategy_family),
                        importance_mode='count_importance')
                if plan is not None:
                    self._stats['mis_head_rounds'] = self._stats.get('mis_head_rounds',0)+1
                    self._stats['mis_ea_rejected_slots']+=plan.ea_rejected
                    self._stats['mis_original_ee_conflict_edges']+=plan.original_ee_conflict_edges
                    self._stats['mis_conflict_edges']+=int(plan.conflict_edges.sum().item())
                    self._stats['mis_routed_importance_sum']+=plan.routed_importance_sum
                    self._stats['mis_accepted_importance_sum']+=plan.accepted_importance_sum
                    self._stats['mis_rejected_importance_sum']+=plan.rejected_importance_sum
                    rej=routed[0]&~plan.accepted
                    rejection_mass=float(cnt[ee[rej]].float().sum().item())
                    hstat['rejected_slots']+=int(rej.sum().item())
                    hstat['rejected_count_mass']+=rejection_mass
                    self._stats['rejected_slots']+=int(rej.sum().item())
                    self._stats['mis_rejected_token_mass']+=rejection_mass
                    self._stats['k_only_rejected_token_mass' if strategy.startswith(('k_only_mis_drop','k_only_degree_anchor_mis_drop','k_only_anchor_first_mis_drop'))
                                else 'importance_rejected_token_mass']+=rejection_mass
                    if strategy.endswith('_allcount'):
                        self._stats['represented_reject_count_mass']+=rejection_mass
                kf,vf,cf=fuse_weighted_slots(keys,vals,cnt,kk,ee,best[0],routed[0],
                    (plan.accepted if plan is not None else routed[0]),strategy,importance=importance[hi])
                if strategy not in ('merge_all','importance_weighted_merge_all') and self._stats['compress_rounds'] % 32 == 0:
                    # Auditable post-dtype K/V change relative to identical
                    # current physical cache with Merge All (same routing/Count).
                    k_ma,v_ma,_=fuse_weighted_slots(keys,vals,cnt,kk,ee,best[0],routed[0],routed[0],'merge_all')
                    kd=(kf.float()-k_ma.float()).abs();vd=(vf.float()-v_ma.float()).abs()
                    self._stats['key_probe_groups']=self._stats.get('key_probe_groups',0)+int(kf.shape[0])
                    self._stats['key_probe_changed_groups']=self._stats.get('key_probe_changed_groups',0)+int(kd.any(-1).sum().item())
                    self._stats['key_probe_max_abs_delta']=max(self._stats.get('key_probe_max_abs_delta',0.),float(kd.max().item()))
                    self._stats['value_probe_changed_groups']=self._stats.get('value_probe_changed_groups',0)+int(vd.any(-1).sum().item())
                    self._stats['value_probe_max_abs_delta']=max(self._stats.get('value_probe_max_abs_delta',0.),float(vd.max().item()))
                if plan is not None:
                    # Streaming diagnostics are BEFORE the next compression and
                    # require no future Queries. The IW control is computed on
                    # the same current physical slots, not a parallel run state.
                    k_iw,v_iw,c_iw=fuse_weighted_slots(keys,vals,cnt,kk,ee,best[0],routed[0],routed[0],
                        'importance_weighted_merge_all',importance=importance[hi])
                    group_n=(torch.bincount(best[0][routed[0]],minlength=len(kk))>0)
                    if bool(group_n.any()):
                        ix=group_n
                        measures={
                            'anchor_key_drift_sq_sum':(kf.float()[ix]-keys.float()[kk[ix]]).square().sum(-1).sum(),
                            'anchor_value_drift_sq_sum':(vf.float()[ix]-vals.float()[kk[ix]]).square().sum(-1).sum(),
                            'iw_anchor_key_drift_sq_sum':(k_iw.float()[ix]-keys.float()[kk[ix]]).square().sum(-1).sum(),
                            'iw_anchor_value_drift_sq_sum':(v_iw.float()[ix]-vals.float()[kk[ix]]).square().sum(-1).sum(),
                            'iw_to_mis_key_shift_sq_sum':(k_iw.float()[ix]-kf.float()[ix]).square().sum(-1).sum(),
                            'iw_to_mis_value_shift_sq_sum':(v_iw.float()[ix]-vf.float()[ix]).square().sum(-1).sum(),
                            'iw_to_mis_logcount_sum':(cf.float()[ix].clamp_min(1).log()-c_iw.float()[ix].clamp_min(1).log()).sum(),
                        }
                        self._stats['diagnosed_routed_groups']=self._stats.get('diagnosed_routed_groups',0)+int(ix.sum().item())
                        for field,value in measures.items():
                            self._stats[field]=self._stats.get(field,0.)+float(value.item())
                kout.append(kf);vout.append(vf);cout.append(cf);pout.append(positions[0,hi,kk])
                self._stats['routed_slots']+=int(routed[0].sum().item())
                self._stats['routed_token_mass']+=float(cnt[ee[routed[0]]].float().sum().item())
                not_routed=~routed[0]
                self._stats['dropped_slots']+=int(not_routed.sum().item())
                self._stats['dropped_token_mass']+=float(cnt[ee[not_routed]].float().sum().item())
            legacy.append((torch.stack(kout,0)[None],torch.stack(vout,0)[None]))
            new_counts.append(torch.stack(cout,0)[None]);new_pos.append(torch.stack(pout,0)[None])
        _replace_dynamic_cache_inplace(self,legacy)
        self.counts=new_counts;self.slot_positions=new_pos
        self.last_nkeep=n_keep
        after=int(self.key_cache[0].shape[-2])
        if after != nk or any(int(c.shape[-1])!=nk for c in new_counts):
            raise RuntimeError(f'KV budget violation: expected {nk}, got {after}')
        self._stats['compress_rounds']+=1
        self.bulk_compression_done=True
        self._stats['max_before_len']=max(self._stats.get('max_before_len',0),before)
        self._stats['max_after_len']=max(self._stats.get('max_after_len',0),after)
        self._stats['last_physical_kv_len']=after
        self._stats['last_logical_tokens']=logical
        for li,layer in enumerate(self._ppl_model.model.layers):
            attn=layer.self_attn
            attn._mymethod_counts_kv=new_counts[li]
            attn._mymethod_eager_enabled=True
            attn._mymethod_sdpa_enabled=False
            attn._keepkv_eager_enabled=False
            attn._mymethod_selector='keepkv'
            attn._mymethod_score_window=self.score_window
            attn._kvmerge_attention_chunk_size=self.chunk_size


def prepare_model_for_new_sequence(model,score_window,chunk_size):
    for layer in model.model.layers:
        attn=layer.self_attn
        attn._mymethod_counts_kv=None
        attn._mymethod_eager_enabled=False
        attn._mymethod_sdpa_enabled=False
        attn._keepkv_eager_enabled=False
        attn._mymethod_selector='keepkv'
        attn._mymethod_score_window=int(score_window)
        attn._kvmerge_attention_chunk_size=int(chunk_size)
        attn._kvmerge_capture_attention_scores=False
        for field in ('_kvmerge_captured_attention_score','_kvmerge_captured_score_log_scale'):
            if hasattr(attn,field):delattr(attn,field)


def load_frozen_calibration(path,model_name):
    info=json.loads(Path(path).read_text())
    scales=validate_payload(info,model=model_name,split='train')
    if info.get('sink_calibration_schema')!='streamingllm_last_query_first_token_v2':
        raise ValueError('PPL requires freshly recalibrated StreamingLLM-style first-token sink metadata')
    masses=np.asarray(info['sink_mass_mean'],dtype=np.float64)
    gate=np.asarray(info['sink_presence_by_kv_head'],dtype=bool)
    if masses.shape!=scales.shape or gate.shape!=scales.shape:
        raise ValueError('Bad head calibration shapes')
    if not np.array_equal(gate,masses>=float(info['sink_presence_threshold'])):
        raise ValueError('Gate inconsistent with train calibration')
    return info,scales,validate_value_payload(info,scales.shape),validate_query_bounds(info,scales.shape),gate


def load_args():
    p=argparse.ArgumentParser(description='One-time bulk-prefix KV compression; continuation-only teacher-forced PPL')
    p.add_argument('--model',required=True)
    p.add_argument('--calibration',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--dataset',default='wikitext2',choices=['wikitext2','c4'])
    p.add_argument('--context_len',type=int,default=4096,help='Uncompressed prefix tokens before ONE compression')
    p.add_argument('--continuation_len',type=int,default=256,help='Number of scored post-compression targets; one additional unscored warm-up token is used')
    p.add_argument('--prefill_chunk_size',type=int,default=128,help='Model forward chunk length (not a compression interval)')
    p.add_argument('--max_sequences',type=int,default=2,help='0 evaluates all full sequences')
    p.add_argument('--budgets',default='0.2,0.4,0.6',
        help='KV retention fractions relative to --context_len: B=round(context_len*ratio)')
    p.add_argument('--routing_modes',default='cosine,l2')
    p.add_argument('--cosine_thresholds',default='0.8',
        help='Comma-separated sweep for cosine routing; independent of L2 thresholds')
    p.add_argument('--l2_lambdas',default='0.7',
        help='Comma-separated sweep for calibrated L2 routing only')
    p.add_argument('--mis_families',default='k_only,k_only_importance',
        help='MIS families; Full Cache and Merge All remain mandatory')
    p.add_argument('--mis_methods',default=None,
        help='Explicit comma-separated independent MIS arms, optionally *_sink')
    p.add_argument('--ee_lambdas',default='0.3,0.4,0.5,0.6,0.7',
        help='Shared Key L2 threshold multiplier lambda for BOTH K-only priorities')
    # Legacy one-value CLI aliases; if provided they override the plural sweep.
    p.add_argument('--cosine_threshold',type=float,default=None,help=argparse.SUPPRESS)
    p.add_argument('--l2_lambda',type=float,default=None,help=argparse.SUPPRESS)
    p.add_argument('--ee_lambda',type=float,default=None,help=argparse.SUPPRESS)
    p.add_argument('--resume',type=int,choices=(0,1),default=1,
        help='Resume partial PPL sweep when the full run signature matches (default: 1)')
    p.add_argument('--score_window',type=int,default=128)
    p.add_argument('--distance_chunk_size',type=int,default=256)
    p.add_argument('--graph_backend',default='gpu_batched')
    p.add_argument('--torch_dtype',default='bfloat16')
    p.add_argument('--attn_implementation',default='eager')
    p.add_argument('--device',default='cuda')
    p.add_argument('--device_map',default='auto')
    p.add_argument('--trust_remote_code',action='store_true')
    p.add_argument('--use_fast_tokenizer',action='store_true')
    return p.parse_args()


def _read_completed_rows(csv_file):
    if not csv_file.exists():return []
    with csv_file.open('r',newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _write_metadata(path,meta):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    temp.replace(path)


def run():
    a=load_args()
    methods=selected_methods(a.mis_families,a.mis_methods)
    families=tuple(f for f in selected_families(a.mis_families)
                   if any(family_for_method(m)==f for m in methods))
    if a.context_len<2 or a.continuation_len<1 or a.prefill_chunk_size<1 or a.max_sequences<0:
        raise ValueError('Invalid bulk-prefix context, continuation, chunk or window settings')
    if a.score_window<1 or a.prefill_chunk_size<min(a.score_window,a.context_len):
        raise ValueError('prefill_chunk_size must cover the requested recent Query score window, '
                         'so the bulk compression uses the full historical Importance window')
    if a.device!='cuda' and a.torch_dtype in ('float16','bfloat16'):
        raise ValueError('For CPU PPL choose torch_dtype=float32')

    cos_str=str(a.cosine_threshold) if a.cosine_threshold is not None else a.cosine_thresholds
    l2_str=str(a.l2_lambda) if a.l2_lambda is not None else a.l2_lambdas
    ee_str=str(a.ee_lambda) if a.ee_lambda is not None else a.ee_lambdas
    cosine=parse_floats_csv(cos_str,'PPL cosine routing threshold',lower=-1.,upper=1.)
    l2s=parse_floats_csv(l2_str,'PPL L2 routing lambda',lower=0.)
    sweeps=sweep_settings(ee_str)
    ees=[s['ee_lambda'] for s in sweeps]
    sweeps_by_ee={s['ee_lambda']:s for s in sweeps}
    specs=routing_specs(a.routing_modes,cosine,l2s)
    budgets=parse_floats_csv(a.budgets,'PPL KV budget',lower=0.,upper=1.)
    capacities=[fixed_cache_capacity(b,a.context_len) for b in budgets]

    info,reference,_reference_v,_q_bounds,_gate=load_frozen_calibration(a.calibration,a.model)
    import transformers
    from core_model_utils import first_execution_device
    cfg=AutoConfig.from_pretrained(a.model,trust_remote_code=a.trust_remote_code)
    family={'qwen2':'qwen','llama':'llama'}.get(str(cfg.model_type).lower())
    if family is None:raise ValueError(f'PPL only supports Qwen2/Llama, got {cfg.model_type}')
    max_positions=getattr(cfg,'max_position_embeddings',None)
    if max_positions is not None and (a.context_len+a.continuation_len)>int(max_positions):
        raise ValueError(f'Context + continuation exceeds model context: '
                         f'{a.context_len}+{a.continuation_len}>{max_positions}')
    if reference.shape!=(cfg.num_hidden_layers,cfg.num_key_value_heads):
        raise ValueError('Calibration belongs to a different model architecture')

    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    dest=out/'ppl_konly_importance_mis.csv'
    comparison_dest=out/'ppl_comparison_by_config.csv'
    head_dest=out/'ppl_routing_load_diagnostics.csv'
    head_rows=[]
    meta_file=out/'ppl_run_config.json'
    signature_inputs=dict(model=a.model,calibration_sha256=hashlib.sha256(Path(a.calibration).read_bytes()).hexdigest(),
        dataset=a.dataset,context_len=a.context_len,continuation_len=a.continuation_len,
        prefill_chunk_size=a.prefill_chunk_size,max_sequences=a.max_sequences,
        budget_ratios=budgets,routing_specs=specs,comparison_sweeps=sweeps,
        score_window=a.score_window,distance_chunk_size=a.distance_chunk_size,
        graph_backend=a.graph_backend,torch_dtype=a.torch_dtype,attn_implementation=a.attn_implementation,
        device=a.device,device_map=a.device_map,
        mis_families=families,methods=methods,
        implementation='degree_anchor_bulk_prefix_continuation_v1')
    signature=hashlib.sha256(json.dumps(signature_inputs,sort_keys=True).encode()).hexdigest()
    old_rows=[]
    if a.resume and (dest.exists() or meta_file.exists()):
        if not meta_file.exists():
            raise RuntimeError('PPL CSV exists without run signature; set PPL_RESUME=0 or use a fresh output directory')
        prev=json.loads(meta_file.read_text(encoding='utf-8'))
        if prev.get('run_signature')!=signature:
            raise RuntimeError('PPL sweep differs from previous run in this directory. '
                'Use PPL_RESUME=0 to replace its results, or choose another experiment name. '
                f'Old signature={prev.get("run_signature")} New signature={signature}')
        old_rows=_read_completed_rows(dest)
        print(f'[PPL RESUME] loaded {len(old_rows)} completed rows; same experiment signature',flush=True)

    meta=dict(args=vars(a),status='running',run_signature=signature,
        signature_inputs=signature_inputs,
        model_type=cfg.model_type,transformers_version=transformers.__version__,
        pytorch=torch.__version__,python=platform.python_version(),
        sink_train_calibration=info['sink_calibration_schema'],
        sink_presence_threshold=info['sink_presence_threshold'],n_sink_kv_heads=int(_gate.sum()),
        sink_gate_policy='*_sink: train Sink KV head -> original MIS; non-Sink -> exact Merge All; frozen for all PPL rounds',
        n_kv_heads=int(_gate.size),
        scoring='Continuation-only teacher-forced CE: [P unscored prefix][1 unscored warm-up][N scored targets]; BF16-safe FP32 CE, FP64 accumulation',
        kv_compression='Exactly one bulk compression at P prefix tokens, to B=round(P*retention_ratio) physical KV slots; no continuation recompression; routed Counts preserved, log(C) attention bias',
        capacity_by_ratio={str(b):c for b,c in zip(budgets,capacities)},
        peak_during_forward='Prefill cache reaches P, then compresses exactly to B; continuation appends uncompressed teacher-forced tokens (physical cache grows after bulk compression)',
        limitation='One-time bulk compression from uncompressed P-token prefix (all Counts=1, R=D=0), NOT native multi-round CohereKV certificate propagation',
        sweep_semantics='All methods: same E-E geometric conflicts. Degree-first, degree-anchor product or E-A first (Count Mean/Drop); optional Count*I D-first/weighted fusion controls.',
        importance_source='Current-round count-decoupled KeepKV recent_window_importance beta=1.0; W=Count*I used by all importance D-first priorities and IW fusion. Selection still uses I. No persistent importance_mass state.',
        no_lookahead=True,
)
    _write_metadata(meta_file,meta)

    tokenizer=AutoTokenizer.from_pretrained(a.model,use_fast=a.use_fast_tokenizer,
        trust_remote_code=a.trust_remote_code)
    model_kwargs=dict(torch_dtype=parse_dtype(a.torch_dtype),
        attn_implementation=a.attn_implementation,trust_remote_code=a.trust_remote_code)
    if a.device_map.lower()!='none':model_kwargs['device_map']=a.device_map
    model=AutoModelForCausalLM.from_pretrained(a.model,**model_kwargs).eval()
    if a.device_map.lower()=='none':model.to(torch.device(a.device))
    device=first_execution_device(model,a.device)
    window_length=a.context_len+1+a.continuation_len
    ids=get_eval_data(a.dataset,window_length,tokenizer)
    if ids.ndim!=2 or ids.shape[0]!=1:raise ValueError('PPL expects a contiguous batch=1 token stream')
    total_full=int(ids.shape[1])//window_length
    nseq=total_full if a.max_sequences==0 else min(a.max_sequences,total_full)
    if nseq<1:raise ValueError('Not enough test tokens for requested bulk-prefix window')
    ids=ids[:,:nseq*window_length].contiguous()

    expected_count=len(specs)*len(budgets)*len(ees)*len(methods)
    ncontrols=sum(m in ('full_cache','merge_all','importance_weighted_merge_all')
                  for m in methods)
    estimated_unique=len(specs)*len(budgets)*(ncontrols-1+len(ees)*(len(methods)-ncontrols))+1
    print(f'[PPL SWEEP] model={a.model} data={a.dataset}/test windows={nseq} '
          f'protocol=bulk_once context={a.context_len} continuation_scored={a.continuation_len} '
          f'warmup=1 forward_chunk={a.prefill_chunk_size} GPU={a.device}',flush=True)
    print(f'[PPL GRID] routing_specs={specs} MIS_FAMILIES={families} METHODS={methods} budgets={budgets} '
          f'rows={expected_count} max_unique_model_evaluations={estimated_unique}',flush=True)
    for s in sweeps:print('[PPL SWEEP] '+sweep_label(s),flush=True)
    apply_runtime_patch_if_needed(family,TASK_ROOT,'mymethod')

    rows=[];existing={};cache={};start_all=time.time()
    planned_keys={setting_key(mode,param,b,ee,m)
                  for mode,param in specs for b in budgets for ee in ees for m in methods}
    for r in old_rows:
        key=setting_key(r['routing_mode'],r['routing_parameter'],r['budget'],r['ee_lambda'],r['method'])
        if key not in planned_keys:raise RuntimeError(f'Unexpected row in resumed PPL CSV: {key}')
        if key in existing:raise RuntimeError(f'Duplicate row in resumed PPL CSV: {key}')
        existing[key]=r
        cache_key=computation_key(*key)
        if cache_key in cache and abs(float(cache[cache_key]['ppl'])-float(r['ppl']))>1e-7:
            raise RuntimeError(f'Inconsistent PPL result for invariant calculation: {cache_key}')
        cache[cache_key]=r

    for mode,param in specs:
        for budget,budget_tokens in zip(budgets,capacities):
            for ee in ees:
                sweep=sweeps_by_ee[ee]
                group={}
                for method in methods:
                    key=setting_key(mode,param,budget,ee,method)
                    reuse_key=computation_key(mode,param,budget,ee,method)
                    if key in existing:
                        row=existing[key]
                        group[method]=row
                        rows.append(row)
                        continue
                    begin=time.time()
                    if reuse_key in cache:
                        src=cache[reuse_key]
                        ppl=float(src['ppl'])
                        stats={field:src[field] for field in (
                            'compress_rounds','first_compression_logical_tokens',
                            'first_compression_before_len','within_budget_calls',
                            'compression_boundary_calls','max_before_len','max_after_len',
                            'routed_slots','routed_token_mass','dropped_slots','dropped_token_mass',
                            'rejected_slots','mis_rejected_token_mass','k_only_rejected_token_mass','importance_rejected_token_mass',
                            'mis_conflict_edges','mis_original_ee_conflict_edges','mis_ea_rejected_slots','mis_routed_importance_sum','mis_accepted_importance_sum','mis_rejected_importance_sum',
                            'represented_reject_count_mass','last_physical_kv_len','key_probe_groups',
                            'key_probe_changed_groups','key_probe_max_abs_delta','value_probe_changed_groups','value_probe_max_abs_delta',
                            'sink_gated_mis_head_rounds','sink_gated_mergeall_head_rounds',
                            'mis_head_rounds',
                            'diagnosed_routed_groups','anchor_key_drift_sq_sum','anchor_value_drift_sq_sum',
                            'iw_anchor_key_drift_sq_sum','iw_anchor_value_drift_sq_sum',
                            'iw_to_mis_key_shift_sq_sum','iw_to_mis_value_shift_sq_sum',
                            'iw_to_mis_logcount_sum') if field in src}
                        reused=True
                    else:
                        reused=False
                        stats={}
                        print(f'[PPL RUN] B={budget:.0%} routing={mode}:{param:g} '
                              f'{sweep_label(sweep)} method={method}',flush=True)
                        def factory():
                            prepare_model_for_new_sequence(model,a.score_window,a.prefill_chunk_size)
                            if method=='full_cache':return DynamicCache()
                            return JournalCompressingCache(model=model,method=method,family=family,
                                budget=budget,routing=mode,cutoff=param,
                                ee_lambda=ee,reference=reference,sink_gate=_gate,
                                score_window=a.score_window,graph_backend=a.graph_backend,
                                chunk_size=a.distance_chunk_size,stats=stats,
                                context_length=a.context_len)
                        if method=='full_cache':
                            prepare_model_for_new_sequence(model,a.score_window,a.prefill_chunk_size)
                            ppl=evaluate_bulk_perplexity(model,ids,a.context_len,a.continuation_len,device,
                                amp_dtype=(parse_dtype(a.torch_dtype) if a.device=='cuda' else None),
                                prefill_chunk_size=a.prefill_chunk_size,cache_factory=None)
                        else:
                            with force_attn_hidden_and_compress(model):
                                ppl=evaluate_bulk_perplexity(model,ids,a.context_len,a.continuation_len,device,
                                    amp_dtype=(parse_dtype(a.torch_dtype) if a.device=='cuda' else None),
                                    prefill_chunk_size=a.prefill_chunk_size,cache_factory=factory)
                    if method!='full_cache' and not reused and stats.get('compress_rounds',0)!=nseq:
                        raise RuntimeError(f'{method}: expected exactly {nseq} bulk compressions, got {stats.get("compress_rounds",0)}')
                    if not reused:
                        for (layer_idx,head_idx),record in sorted(stats.get('head_diagnostics',{}).items()):
                            nr=record['n_compressions']
                            head_rows.append(dict(model=a.model,method=method,budget=budget,
                                ee_lambda=ee,routing_mode=mode,routing_parameter=param,
                                layer_idx=layer_idx,kv_head_idx=head_idx,
                                **record,mean_routing_coverage=record['routing_coverage_sum']/max(1,nr),
                                mean_group_hhi=record['routing_group_hhi_sum']/max(1,nr),
                                mean_max_group_load_fraction=record['max_group_load_fraction_sum']/max(1,nr),
                                mean_routed_group_size=record['mean_routed_group_size_sum']/max(1,nr),
                                mean_n_routed_groups=record['n_routed_groups_sum']/max(1,nr)))
                        if head_rows:atomic_write_csv(head_dest,head_rows)
                    row=dict(model=a.model,dataset=a.dataset,split=('validation' if a.dataset=='c4' else 'test'),
                        method=method,mis_family=family_for_method(method),routing_mode=mode,routing_parameter_type=('cosine_threshold' if mode=='cosine' else 'l2_lambda'),
                        routing_parameter=param,budget=budget,retention_ratio=budget,
                        fixed_kv_budget_tokens=budget_tokens,budget_policy='round_fraction_times_context_len_bulk_once',
                        cosine_threshold=(param if mode=='cosine' else ''),
                        l2_lambda=(param if mode=='l2' else ''),
                        ee_lambda=ee,sweep_id=sweep['sweep_id'],
                        score_window=a.score_window,
                        sink_gate_applied=int(is_sink_gated(method)),
                        n_train_sink_heads=int(_gate.sum()),
                        n_train_nonsink_heads=int(_gate.size-_gate.sum()),
                        sink_gated_mis_head_rounds=stats.get('sink_gated_mis_head_rounds',0),
                        sink_gated_mergeall_head_rounds=stats.get('sink_gated_mergeall_head_rounds',0),
                        context_len=a.context_len,continuation_len=a.continuation_len,
                        unscored_warmup_tokens_per_sequence=1,forward_chunk_size=a.prefill_chunk_size,
                        num_sequences=nseq,scored_tokens=nseq*a.continuation_len,ppl=ppl,
                        calibration_schema=info['aggregation'],
                        compress_rounds=stats.get('compress_rounds',0),
                        first_compression_logical_tokens=stats.get('first_compression_logical_tokens',''),
                        first_compression_before_len=stats.get('first_compression_before_len',''),
                        within_budget_calls=stats.get('within_budget_calls',0),
                        compression_boundary_calls=stats.get('compression_boundary_calls',0),
                        max_before_len=stats.get('max_before_len',''),
                        max_after_len=stats.get('max_after_len',''),
                        routed_slots=stats.get('routed_slots',0),
                        routed_token_mass=stats.get('routed_token_mass',0.0),
                        dropped_slots=stats.get('dropped_slots',0),
                        dropped_token_mass=stats.get('dropped_token_mass',0.0),
                        rejected_slots=stats.get('rejected_slots',0),
                        mis_rejected_token_mass=stats.get('mis_rejected_token_mass',0.0),
                        k_only_rejected_token_mass=stats.get('k_only_rejected_token_mass',0.0),
                        importance_rejected_token_mass=stats.get('importance_rejected_token_mass',0.0),
                        mis_conflict_edges=stats.get('mis_conflict_edges',0),
                        mis_original_ee_conflict_edges=stats.get('mis_original_ee_conflict_edges',0),
                        mis_ea_rejected_slots=stats.get('mis_ea_rejected_slots',0),
                        mis_routed_importance_sum=stats.get('mis_routed_importance_sum',0.0),
                        mis_accepted_importance_sum=stats.get('mis_accepted_importance_sum',0.0),
                        mis_rejected_importance_sum=stats.get('mis_rejected_importance_sum',0.0),
                        mis_accepted_importance_fraction=(stats.get('mis_accepted_importance_sum',0.0) /
                            max(1e-12,stats.get('mis_routed_importance_sum',0.0))),
                        represented_reject_count_mass=stats.get('represented_reject_count_mass',0.0),
                        last_physical_kv_len=stats.get('last_physical_kv_len',''),
                        key_probe_groups=stats.get('key_probe_groups',0),
                        key_probe_changed_groups=stats.get('key_probe_changed_groups',0),
                        key_probe_max_abs_delta=stats.get('key_probe_max_abs_delta',0.0),
                        value_probe_changed_groups=stats.get('value_probe_changed_groups',0),
                        value_probe_max_abs_delta=stats.get('value_probe_max_abs_delta',0.0),
                        mis_head_rounds=stats.get('mis_head_rounds',0),
                        diagnosed_routed_groups=stats.get('diagnosed_routed_groups',0),
                        anchor_key_drift_sq_sum=stats.get('anchor_key_drift_sq_sum',0.),
                        anchor_value_drift_sq_sum=stats.get('anchor_value_drift_sq_sum',0.),
                        iw_anchor_key_drift_sq_sum=stats.get('iw_anchor_key_drift_sq_sum',0.),
                        iw_anchor_value_drift_sq_sum=stats.get('iw_anchor_value_drift_sq_sum',0.),
                        iw_to_mis_key_shift_sq_sum=stats.get('iw_to_mis_key_shift_sq_sum',0.),
                        iw_to_mis_value_shift_sq_sum=stats.get('iw_to_mis_value_shift_sq_sum',0.),
                        iw_to_mis_logcount_sum=stats.get('iw_to_mis_logcount_sum',0.),
                        evaluation_reused=int(reused),elapsed_seconds=(0. if reused else time.time()-begin))
                    # Paired deltas are finalized only after IW-MergeAll exists.
                    # Count MergeAll precedes IW-MergeAll in the evaluation loop.
                    row['delta_ppl_from_merge_all']=''
                    row['relative_delta_ppl_pct']=''
                    row['delta_ppl_from_iw_merge_all']=''
                    row['relative_delta_ppl_from_iw_pct']=''
                    rows.append(row);group[method]=row;cache[reuse_key]=row
                    # Persist EVERY completed arm; interruption can resume safely.
                    atomic_write_csv(dest,rows)
                if len(group)==len(methods):
                    # Also repair existing rows when resuming a prior interrupted run.
                    finalize_ppl_group_deltas(group)
                    atomic_write_csv(dest,rows)
                    print(formatted_ppl_summary(group,mode,param,budget,ee,methods),flush=True)
                    atomic_write_csv(comparison_dest,config_comparison_rows(rows,methods))
    if not dest.exists():atomic_write_csv(dest,rows)
    meta.update(status='completed',new_unique_model_evaluations=sum(int(r['evaluation_reused'])==0 for r in rows if isinstance(r,dict)),
                recorded_rows=len(rows),elapsed_seconds=time.time()-start_all,
                num_sequences=nseq,scored_tokens=nseq*a.continuation_len)
    _write_metadata(meta_file,meta)
    print(f'[PPL COMPLETE] rows={len(rows)} summaries={len(config_comparison_rows(rows,methods))} '
          f'CSV={dest} ; comparison={comparison_dest}',flush=True)

if __name__=='__main__':run()
