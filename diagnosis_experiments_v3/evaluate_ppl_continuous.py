#!/usr/bin/env python3
"""Continuous, multi-round, p-certified CohereKV/K-only MIS perplexity.

For each cycle, feed U-1 unscored tokens; compress at that logical boundary;
feed the last unscored token AFTER compression (causal warm-up); then feed S
scored tokens, and score exactly those S targets. No reset between cycles.
All tokens are distinct positions in one contiguous WikiText-2 test stream.

Run from the CohereKV repository root, with its task_evaluation and models
installed.  Diagnostic source only; no pretrained weights shipped here.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,DynamicCache

ROOT=Path(__file__).resolve().parents[1]
TASK_ROOT=ROOT/'task_evaluation'
if str(TASK_ROOT) not in sys.path:sys.path.insert(0,str(TASK_ROOT))
from evaluate_ppl_kvmerge import (KVCompressingDynamicCache,_replace_dynamic_cache_inplace,
    apply_runtime_patch_if_needed,force_attn_hidden_and_compress,get_eval_data)
from core_model_utils import first_execution_device,parse_dtype
from continuous_model_support import prepare_model_for_new_sequence,load_frozen_calibration
from cosine_mis_plan import select_split
from k_only_mis_drop import ee_mis,fuse_weighted_slots
from method_registry import (selected_methods,selected_families,family_for_method,
    effective_method,is_wo_ea_certificate,graph_is_importance_aware,priority_for_family)
from models.common.mymethod_common import recent_window_importance,reference_scale_from_original_keys
from p_geometry import (certified_routes,physical_l2_routes,
    merge_cluster_tensors,merge_cluster_moments,parse_p,validate_geometry_mode,
    route_diagnostics_p2,route_diagnostics_moment,route_diagnostics_countaware,
    is_count_aware,physical_count_aware_ratio)
from fusion_fidelity import audit_fusion_head
from physical_ee_controls import (is_physical_method,physical_method_spec,
    physical_routes_for_method,nearest_count_matched_accept,physical_selected_ee_conflicts)
from necessity_controls import (HeadHistories,exact_history_selection,
    importance_count_matched,is_history_method)
from lambda_sweep_utils import (parse_lambda_pairs,build_sweep_summary,build_sweep_cycles,COMPARATORS)
from history_oracle_p4 import (extend_histories,merged_histories,audit_head,summarize_pairs)
from result_csv import save_csv as _save_csv
from group_d2_selection import (GROUP_METHODS, GROUP_D2, WO_HISTORY, WO_EE, MATCHED,
                                choose_group_d2,merge_group_d2_history)


def _csv_floats(s,name, *, p=False):
    values=[]
    for tok in str(s).split(','):
        tok=tok.strip()
        if not tok:continue
        val=parse_p(tok) if p else float(tok)
        if not p and not (math.isfinite(val) and val>0):raise ValueError(f'{name} must be positive finite')
        if val not in values:values.append(val)
    if not values:raise ValueError(f'Empty {name}')
    return values




def _steps(start,end,chunk,score_window=1):
    """Do not make a final chunk shorter than score_window at compression."""
    if start>=end:return []
    length=end-start
    if length<=chunk:return [(start,end)]
    rem=length%chunk
    if rem and rem<score_window and chunk>=score_window and length>=chunk+score_window:
        # Last two chunks total chunk+rem, split to (rem,chunk).
        # This need not preserve the target max chunk; split instead as
        # (chunk+rem-score_window,score_window), both <=chunk.
        first_cut=end-(chunk+rem)
        return _steps(start,first_cut,chunk,score_window)+[(first_cut,end-score_window),(end-score_window,end)]
    return [(i,min(end,i+chunk)) for i in range(start,end,chunk)]


@torch.no_grad()
def evaluate_continuous(model,data,cycles,unscored_len,scored_len,device,*,
                        cache_factory=None,prefill_chunk_size=128,score_window=128,
                        amp_dtype=None,on_cycle=None):
    """Returns (total_ppl, cycle_rows). No repeated documents/windows/cache reset.

    Every scored target is predicted by a query that ran AFTER that cycle's
    compression, including the first target (predicted from final warmup).
    """
    U=int(unscored_len);S=int(scored_len);P=int(cycles)
    if P<1 or U<2 or S<1 or prefill_chunk_size<1 or score_window<1:
        raise ValueError('cycles>=1, unscored_len>=2, scored_len>=1; chunks positive')
    total_length=P*(U+S)
    if data.ndim!=2 or data.shape[0]!=1 or data.shape[1]!=total_length:
        raise ValueError(f'Expected exactly [1,{total_length}] contiguous input IDs')
    cache=cache_factory() if cache_factory else DynamicCache()
    data=data.to(device)
    total_nll=0.;total_count=0;per_cycle=[]
    autocast=amp_dtype is not None and device.type=='cuda'
    with torch.amp.autocast('cuda',enabled=autocast,dtype=amp_dtype or torch.float32):
        for cycle in range(P):
            base=cycle*(U+S)
            stop=base+U-1
            # Full unscored prefix except the final warm-up. After the LAST
            # forward the cache compresses, BEFORE the last unscored query.
            for lo,hi in _steps(base,stop,prefill_chunk_size,score_window):
                out=model(data[:,lo:hi],use_cache=True,past_key_values=cache)
                if out.past_key_values is not cache:raise RuntimeError('KV cache object replaced')
                del out
            if hasattr(cache,'compress_rounds') and cache.compress_rounds!=cycle+1:
                raise RuntimeError(f'Expected compression round {cycle+1} at logical position {stop}')
            # One last unscored token, now with the compressed cache.
            out=model(data[:,stop:stop+1],use_cache=True,past_key_values=cache)
            if out.past_key_values is not cache:raise RuntimeError('KV cache object replaced')
            logits=out.logits[:,-1,:].float()
            target=data[:,stop+1]
            loss=torch.nn.functional.cross_entropy(logits,target,reduction='sum')
            nll=float(loss.double().item());n=1
            del out
            # All scored input tokens are forwarded, so the next cycle's
            # prefix begins at the next new position; none is replayed.
            # Their first S-1 logits predict the following S-1 scored labels.
            scored_start=base+U;scored_stop=scored_start+S
            for lo,hi in _steps(scored_start,scored_stop,prefill_chunk_size):
                out=model(data[:,lo:hi],use_cache=True,past_key_values=cache)
                if out.past_key_values is not cache:raise RuntimeError('KV cache object replaced')
                num=max(0,min(hi,scored_stop-1)-lo)
                if num:
                    ll=out.logits[:,:num,:].float()
                    labels=data[:,lo+1:lo+1+num]
                    val=torch.nn.functional.cross_entropy(ll.reshape(-1,ll.shape[-1]),labels.reshape(-1),reduction='sum')
                    nll+=float(val.double().item());n+=int(labels.numel())
                del out
            if n!=S:raise AssertionError(f'Cycle scored {n}, expected {S}')
            if not math.isfinite(nll):raise RuntimeError('Nonfinite NLL in cycle')
            total_nll+=nll;total_count+=n
            stats=cache._stats if hasattr(cache,'_stats') else {}
            row=dict(cycle=cycle+1,logical_tokens=(cycle+1)*(U+S),
                     scored_tokens=n,cycle_nll_sum=nll,cycle_nll_per_token=nll/n,
                     cycle_ppl=math.exp(nll/n),cumulative_nll_per_token=total_nll/total_count,
                     cumulative_ppl=math.exp(total_nll/total_count),
                     compression_rounds=int(stats.get('compress_rounds',0)),
                     routed_slots=int(stats.get('routed_slots',0)),
                     accepted_slots=int(stats.get('accepted_slots',0)),
                     rejected_slots=int(stats.get('rejected_slots',0)),
                     geometry_over_delta=int(stats.get('geometry_over_delta',0)),
                     physical_kv_len=int(cache.key_cache[0].shape[-2]) if len(cache.key_cache) else 0)
            per_cycle.append(row)
            if on_cycle:on_cycle(row)
    if total_count!=P*S:raise AssertionError('Incomplete scored tokens')
    if hasattr(cache,'_ppl_logical_seen_tokens') and cache._ppl_logical_seen_tokens!=total_length:
        raise RuntimeError('Logical KV position did not advance over full text')
    return math.exp(total_nll/total_count),per_cycle


class ContinuousPCache(KVCompressingDynamicCache):
    """Physical fixed-budget cache; represented-history p certificates are separate.

    All registered MIS priorities and fusion/count controls remain available.
    MergeAll uses physical Key L2 routing without historic certificates.
    CohereKV w/o MIS uses historical p-certified E-A routing but accepts all
    routed candidates and propagates certificates even if a merged group
    exceeds the p-diameter threshold. It is NOT certified p-compact.
    Every MIS arm has a paired *_wo_ea_certificate variant: physical-Key L2
    E-A routing, unchanged p-certified E-E MIS / priority / K/V fusion, and
    real historical-certificate propagation. Such variants are diagnostic,
    not certified p-compact, because the anchor bypasses E-A certification.
    Importance-weighted K means and AllCount do NOT corrupt the geometric
    history; independent arithmetic means and represented Counts are stored.
    """
    def __init__(self,*,model,method,budget,p,lambda_ea,lambda_ee,reference,sink_gate,
                 unscored_len,scored_len,reference_mode='online_first',score_window=128,
                 sink_size=4,recent_protect_size=128,recency_beta=1.0,chunk_size=256,
                 graph_backend='gpu_batched',geometry_mode='minkowski',
                 route_diagnostics=True,stats=None,oracle_enabled=False,
                 oracle_layers=(),oracle_heads=(),oracle_evictions_per_head=16,
                 oracle_anchors_per_eviction=4,lambda_history=0.75,group_d2_pair_window=8,
                 fusion_sample_topk=4):
        if not (0<float(budget)<1):raise ValueError('Budget ratio in (0,1)')
        self.budget_slots=max(1,round((int(unscored_len)-1)*float(budget)))
        super().__init__(model=model,method='mymethod',family='p_certificate',
            compress_ratio=1.-float(budget),sim_thr=-1.,method_sim_thrs=None,
            keepkv_config={},mymethod_config={},centroidkv_config={},
            min_kv_len=2,compress_every=1,stats=(stats if stats is not None else {}),
            task_eval_dir=TASK_ROOT,target_kv_len=self.budget_slots)
        self.strategy=method
        # Comparison under identical data/budget/scale: restore the conference
        # worst-case historical certificate WITHOUT changing Group-D2's p=2 path.
        # This is a conference-mechanism comparator, not a claim of identical
        # published calibration/threshold hyperparameters.
        self.p=math.inf if method=='coherekv_inf_certificate' else parse_p(p)
        if method in GROUP_METHODS and self.p!=2.:
            raise ValueError('Group-D2 is an EXACT p=2 implementation; other p are theoretical only')
        self.geometry_mode=validate_geometry_mode(
            'minkowski' if method=='coherekv_inf_certificate' else geometry_mode,self.p)
        self.route_diagnostics=bool(route_diagnostics)
        self.oracle_enabled=bool(oracle_enabled)
        if self.oracle_enabled and (self.p!=4. or self.geometry_mode not in ('moment_interp','fourth_moment','count_aware','count_aware_phys_075','count_aware_phys_100')):
            raise ValueError('Exact history oracle requires p=4 with moment_interp/fourth_moment carrying R2/Rinf')
        self.oracle_layers=set(map(int,oracle_layers))
        self.oracle_heads=set(map(int,oracle_heads))
        self.oracle_evictions_per_head=int(oracle_evictions_per_head)
        self.oracle_anchors_per_eviction=int(oracle_anchors_per_eviction)
        self.oracle_histories={}  # CPU originals for selected layer/head ONLY
        if self.oracle_enabled and (not self.oracle_layers or not self.oracle_heads
                    or self.oracle_evictions_per_head<=0 or self.oracle_anchors_per_eviction<=0):
            raise ValueError('Oracle needs nonempty selected layers/heads and positive sampling')
        self.lambda_ea=float(lambda_ea)
        self.lambda_ee=float(lambda_ee)
        self.lambda_history=float(lambda_history)
        self.group_d2_pair_window=int(group_d2_pair_window)
        if self.group_d2_pair_window<0 or self.group_d2_pair_window>128:
            raise ValueError('group_d2_pair_window must be between 0 and 128')
        self.fusion_sample_topk=int(fusion_sample_topk)
        if not 0<=self.fusion_sample_topk<=100:
            raise ValueError('fusion_sample_topk must be in 0..100')
        if not (math.isfinite(self.lambda_history) and self.lambda_history>0):
            raise ValueError('Exact historical lambda must be finite and positive')
        self.exact_histories={}  # only used by exact H1 / matched H2, full heads
        if is_count_aware(self.geometry_mode) and not math.isclose(self.lambda_ea,self.lambda_ee,rel_tol=1e-12):
            raise ValueError('count_aware requires lambda_ea == lambda_ee for its simultaneous-merge proof')
        self.reference=np.asarray(reference,dtype=np.float64)
        if reference_mode not in ('online_first','frozen_train'):
            raise ValueError('Unknown reference_mode')
        self.reference_mode=reference_mode
        self.online_ref=None
        self.sink_gate=np.asarray(sink_gate,dtype=bool)
        if self.sink_gate.shape!=self.reference.shape:raise ValueError('Mismatch sink/calib dimensions')
        if (not math.isfinite(self.lambda_ea) or not math.isfinite(self.lambda_ee)
                or self.lambda_ea<=0 or self.lambda_ee<=0):
            raise ValueError('Both lambda_ea and lambda_ee must be positive and finite')
        self.unscored_len=int(unscored_len);self.scored_len=int(scored_len)
        self.period=self.unscored_len+self.scored_len
        self.score_window=int(score_window);self.chunk_size=int(chunk_size)
        self.sink_size=int(sink_size)
        self.recent_protect_size=int(recent_protect_size)
        self.recency_beta=float(recency_beta)
        if self.sink_size<0 or self.recent_protect_size<0:
            raise ValueError('sink_size and recent_protect_size must be nonnegative')
        if not (math.isfinite(self.recency_beta) and self.recency_beta>=1.):
            raise ValueError('recency_beta must be finite and >=1')
        if self.budget_slots<self.sink_size:
            raise ValueError(f'Post-compression budget {self.budget_slots} cannot preserve '
                             f'{self.sink_size} mandatory original sink tokens; '
                             'increase budget or reduce sink_size')
        # Match original KeepKV semantics: sinks are mandatory; recent protection
        # uses remaining kept slots and can be capped by budget. Report its cap.
        self.effective_recent_protect_size=min(self.recent_protect_size,
            self.budget_slots-self.sink_size)
        if self.effective_recent_protect_size<self.recent_protect_size:
            print(f'[KEEPKV PROTECT] recent_protect_size={self.recent_protect_size} '
                  f'exceeds remaining kept budget; protecting '
                  f'{self.effective_recent_protect_size} most-recent slots',flush=True)
        self.graph_backend=graph_backend
        self.compress_rounds=0
        self.counts=None;self.slot_positions=None
        self.geom_counts=None;self.geom_means=None;self.geom_radii=None;self.geom_diameters=None
        self.geom_r2=None;self.geom_rinf=None;self.geom_dinf=None
        self.group_m2=None  # exact historical centered second moment for new Group-D2 arms
        self._stats.setdefault('compress_rounds',0)
        self._stats.setdefault('routed_slots',0)
        self._stats.setdefault('accepted_slots',0)
        self._stats.setdefault('rejected_slots',0)
        self._stats.setdefault('geometry_over_delta',0)

    @torch.no_grad()
    def maybe_compress(self,attentions,hidden_states):
        self._forward_calls+=1
        logical=int(self._ppl_logical_seen_tokens)
        # Compress exactly at the last-but-one token of the unscored interval,
        # leaving a post-compression warm-up before EVERY scored interval.
        boundary=self.compress_rounds*self.period+self.unscored_len-1
        if logical<boundary:return
        if logical!=boundary:
            raise RuntimeError(f'Forward skipped compression boundary {boundary}: got {logical}')
        if not self.key_cache:raise RuntimeError('No KV')
        n=int(self.key_cache[0].shape[-2]);B=int(self.budget_slots)
        if n<=B:raise RuntimeError('Budget is not smaller than physical cache')
        if attentions is None or len(attentions)!=len(self.key_cache):
            raise RuntimeError('Missing current-round KeepKV score capture')
        new_kv=[];new_count=[];new_positions=[]
        new_gc=[];new_gm=[];new_gr=[];new_gd=[]
        new_gr2=[];new_grinf=[];new_gdinf=[];new_group_m2=[]
        online_ref=[]
        for layer,(KL,VL,score) in enumerate(zip(self.key_cache,self.value_cache,attentions)):
            if KL.shape[0]!=1 or int(KL.shape[-2])!=n:raise ValueError('Expected uniform batch-1 cache')
            H=int(KL.shape[1]);D=int(KL.shape[-1]);dev=KL.device
            if self.reference_mode=='online_first':
                if self.online_ref is None:
                    # Exactly conference Eq.(7), measured before the FIRST
                    # compression from original, never-merged current keys.
                    online_ref.append(reference_scale_from_original_keys(KL).reshape(-1).double())
                scale=(online_ref[-1] if self.online_ref is None else self.online_ref[layer])
            else:
                scale=torch.as_tensor(self.reference[layer],device=dev,dtype=torch.float64)
            if score is None or score.ndim!=4 or score.shape[:2]!=KL.shape[:2] or score.shape[-1]!=n:
                raise RuntimeError(f'Invalid importance capture layer={layer}')
            prev=0 if self.counts is None else int(self.counts[layer].shape[-1])
            if prev>n:raise RuntimeError('History longer than cache')
            tail=n-prev
            prev_counts=(self.counts[layer].to(dev) if self.counts is not None
                         else KL.new_ones((1,H,0),dtype=torch.float32))
            attention_counts=torch.cat((prev_counts,torch.ones((1,H,tail),device=dev)),dim=-1)
            if self.slot_positions is None:
                prevpos=torch.empty((1,H,0),device=dev,dtype=torch.long)
            else:prevpos=self.slot_positions[layer].to(dev)
            pos_new=torch.arange(logical-tail,logical,device=dev).view(1,1,-1).expand(1,H,-1)
            positions=torch.cat((prevpos,pos_new),dim=-1)
            if self.geom_counts is None:
                gc_old=KL.new_ones((1,H,0),dtype=torch.float64)
                gm_old=KL.new_empty((1,H,0,D),dtype=torch.float64)
                gr_old=KL.new_zeros((1,H,0),dtype=torch.float64)
                gd_old=gr_old
                gr2_old=gr_old;grinf_old=gr_old;gdinf_old=gr_old
            else:
                gc_old=self.geom_counts[layer].to(dev)
                gm_old=self.geom_means[layer].to(dev)
                gr_old=self.geom_radii[layer].to(dev)
                gd_old=self.geom_diameters[layer].to(dev)
                gr2_old=self.geom_r2[layer].to(dev) if self.geom_r2 is not None else torch.zeros_like(gr_old)
                grinf_old=self.geom_rinf[layer].to(dev) if self.geom_rinf is not None else torch.zeros_like(gr_old)
                gdinf_old=self.geom_dinf[layer].to(dev) if self.geom_dinf is not None else torch.zeros_like(gr_old)
            gc=torch.cat((gc_old,torch.ones((1,H,tail),device=dev,dtype=torch.float64)),dim=-1)
            gm=torch.cat((gm_old,KL[:,:,prev:,:].double()),dim=-2)
            gr=torch.cat((gr_old,torch.zeros((1,H,tail),device=dev,dtype=torch.float64)),dim=-1)
            gd=torch.cat((gd_old,torch.zeros((1,H,tail),device=dev,dtype=torch.float64)),dim=-1)
            gr2=torch.cat((gr2_old,torch.zeros((1,H,tail),device=dev,dtype=torch.float64)),dim=-1)
            grinf=torch.cat((grinf_old,torch.zeros((1,H,tail),device=dev,dtype=torch.float64)),dim=-1)
            gdinf=torch.cat((gdinf_old,torch.zeros((1,H,tail),device=dev,dtype=torch.float64)),dim=-1)
            if self.group_m2 is None:
                group_m2_old=torch.zeros((1,H,0),device=dev,dtype=torch.float64)
            else:
                group_m2_old=self.group_m2[layer].to(dev)
            group_m2=torch.cat((group_m2_old,torch.zeros((1,H,tail),device=dev,dtype=torch.float64)),dim=-1)
            if not (gc.shape==gr.shape==gd.shape==gr2.shape==grinf.shape==gdinf.shape==attention_counts.shape==positions.shape==(1,H,n)):
                raise RuntimeError('Historical geometry and physical cache mismatch')
            if score.shape[-2]>self.score_window:score=score[:,:,-self.score_window:]
            imp=recent_window_importance(score.float().reshape(H,score.shape[-2],n),
                                         ptr=0,beta=self.recency_beta)
            # positions carry original logical token indices across compressions;
            # mandatory sinks and recency are selected by these indices, NOT
            # physical cache slot indices (which are sparse after compression).
            keep,evict=select_split(KL[0].float(),imp,positions[0],B,
                                    self.sink_size,self.recent_protect_size)
            if keep.shape!=(H,B):
                raise RuntimeError('KeepKV selector returned an unexpected retained shape')
            kout=[];vout=[];cout=[];pout=[];gcount=[];gmean=[];grad=[];gdiam=[]
            gR2=[];gRinf=[];gDinf=[];gM2=[]
            for h in range(H):
                base_scale=float(scale[h].item())
                delta_ea=self.lambda_ea*base_scale
                delta_ee=self.lambda_ee*base_scale
                delta_cert=max(delta_ea,delta_ee)
                k=KL[0,h];v=VL[0,h]
                cnt=attention_counts[0,h];importance=imp[h]
                centers=gm[0,h];rads=gr[0,h];diams=gd[0,h]
                r2s=gr2[0,h];rinfs=grinf[0,h];dinfs=gdinf[0,h]
                kk=keep[h];ee=evict[h]
                moments2=group_m2[0,h]
                oracle_current=None
                if (self.oracle_enabled and layer in self.oracle_layers and h in self.oracle_heads):
                    # Transfer ORIGINAL unmerged new Keys before CountMean/fusion.
                    # Earlier originals are carried in CPU history lists.
                    original_tail=k[prev:].detach().double().cpu().numpy()
                    original_positions=positions[0,h,prev:].detach().cpu().numpy()
                    oracle_current=extend_histories(
                        self.oracle_histories.get((layer,h),[]),
                        original_tail,original_positions,prev)
                    if len(oracle_current)!=n:
                        raise AssertionError('Oracle history physical slot mismatch')
                    represented=np.fromiter((len(member.keys) for member in oracle_current),
                                              dtype=np.int64,count=n)
                    counted=gc[0,h].detach().cpu().numpy()
                    if not np.array_equal(represented,counted):
                        raise AssertionError('Oracle original-member counts differ from geometric Count')
                strategy=effective_method(self.strategy,bool(self.sink_gate[layer,h]))
                exact_current=None
                if is_history_method(strategy):
                    original_tail=k[prev:].detach().cpu().float().numpy()
                    exact_current=HeadHistories.extend(
                        self.exact_histories.get((layer,h)),original_tail,prev)
                    actual_counts=gc[0,h].detach().cpu().long().numpy()
                    if not np.array_equal(exact_current.counts(),actual_counts):
                        raise AssertionError('Exact H history failed to track represented original Count')
                physical_method=is_physical_method(strategy)
                physical_kind=(physical_method_spec(strategy)[1] if physical_method else None)
                # The matched control computes a counterfactual MIS quota but
                # selects nearest physical candidates, NOT an independent set.
                has_mis=strategy not in ('eviction_only','merge_all','importance_weighted_merge_all',
                                         'coherekv_wo_mis','physical_gated_merge_all_075',
                                         *GROUP_METHODS)
                actual_mis=has_mis and physical_kind not in ('matched','importance_matched','h_matched')
                uses_history_ea=(not physical_method and strategy not in ('eviction_only','merge_all',
                    'importance_weighted_merge_all') and not is_wo_ea_certificate(strategy))
                tracks_history=(not physical_method and strategy not in ('eviction_only','merge_all',
                    'importance_weighted_merge_all'))
                has_full_certificate=(actual_mis and uses_history_ea)
                # Both diagnostic ablations retain honest historical certificates:
                # w/o MIS may violate E-E; w/o E-A certificate may violate E-A.
                # The bound is still propagated, not silently reset to zero.
                if strategy=='coherekv_wo_mis':
                    delta_cert=delta_ea
                gate_diagnostics={}
                if is_count_aware(self.geometry_mode) and has_mis and graph_is_importance_aware(family_for_method(strategy)):
                    raise ValueError('count_aware requires geometric (unweighted) E-E conflicts; IA graph is not certified')
                if is_count_aware(self.geometry_mode) and tracks_history and not has_mis and not math.isinf(self.p):
                    raise ValueError('count_aware requires MIS to certify simultaneous merging')
                if self.route_diagnostics and tracks_history:
                    if self.geometry_mode in ('moment_interp','fourth_moment','count_aware','count_aware_phys_075','count_aware_phys_100'):
                        gate_diagnostics=route_diagnostics_moment(
                            k,centers,rads,r2s,rinfs,diams,kk,ee,
                            delta_ea,delta_cert,self.p,self.chunk_size,
                            counts=gc[0,h] if self.geometry_mode in ('moment_interp','count_aware','count_aware_phys_075','count_aware_phys_100') else None)
                    elif self.p==2.:
                        gate_diagnostics=route_diagnostics_p2(
                            k,centers,rads,diams,kk,ee,delta_ea,delta_cert,self.chunk_size)
                    for metric,value in gate_diagnostics.items():
                        self._stats[metric]=int(self._stats.get(metric,0))+int(value)
                if uses_history_ea:
                    best,routed,gamma=certified_routes(
                        centers,rads,diams,kk,ee,delta_ea,self.chunk_size,
                        history_limit=delta_cert,geometry_mode=self.geometry_mode,
                        p=self.p,radius_2=r2s if self.geometry_mode in ('moment_interp','fourth_moment','count_aware','count_aware_phys_075','count_aware_phys_100') else None,
                        radius_inf=rinfs if self.geometry_mode in ('moment_interp','fourth_moment','count_aware','count_aware_phys_075','count_aware_phys_100') else None,
                        counts=gc[0,h] if is_count_aware(self.geometry_mode) else None,
                        physical_keys=k if physical_count_aware_ratio(self.geometry_mode) is not None else None)
                elif strategy=='eviction_only':
                    # No routing or merging: fixed KeepKV selection only.
                    best=torch.zeros(ee.numel(),device=ee.device,dtype=torch.long)
                    routed=torch.zeros(ee.numel(),device=ee.device,dtype=torch.bool)
                    gamma=torch.zeros(ee.numel(),device=ee.device,dtype=torch.float64)
                elif physical_method:
                    # All paired physical controls share the SAME E-A candidate
                    # and nearest-anchor rule for a given gate ratio.
                    best,routed,gamma=physical_routes_for_method(
                        k,kk,ee,delta_ea,strategy,self.chunk_size)
                else:
                    # Original MergeAll: nearest physical Key with DeltaEA gate.
                    best,routed,gamma=physical_l2_routes(
                        k,kk,ee,delta_ea,self.chunk_size)
                plan=None
                if has_mis and physical_kind not in ('h_exact','h_matched'):
                    fam=family_for_method(strategy)
                    # For physical controls E-E conflict is between CURRENT
                    # PHYSICAL evicted Keys (not historical geometric centers).
                    # The conference arm keeps its historical E-E certificate.
                    plan=ee_mis(k.float(),v.float(),kk,ee,best,routed,gamma,
                        base_scale,1.,self.lambda_ee,
                        self.graph_backend,counts=cnt,importance=importance,
                        priority=('importance_mass_delta' if physical_method else
                                  'degree' if strategy=='coherekv_inf_certificate'
                                  else priority_for_family(fam)),
                        importance_aware=(False if physical_method or strategy=='coherekv_inf_certificate'
                                          else graph_is_importance_aware(fam)),
                        certificate_centers=(None if physical_method else centers),
                        certificate_radii=(None if physical_method else rads),
                        geometry_mode=self.geometry_mode,p=self.p,
                        certificate_r2=(r2s if not physical_method and self.geometry_mode in ('moment_interp','fourth_moment','count_aware','count_aware_phys_075','count_aware_phys_100') else None),
                        certificate_rinf=(rinfs if not physical_method and self.geometry_mode in ('moment_interp','fourth_moment','count_aware','count_aware_phys_075','count_aware_phys_100') else None))
                history_selection_stats={}
                if strategy in GROUP_METHODS:
                    # Same physical E-A routing in every Group-D2 arm.
                    # Only the full selector certifies final historical D2.
                    accepted,history_selection_stats=choose_group_d2(
                        kk,ee,best,routed,gc[0,h],centers,moments2,
                        importance,self.lambda_history*base_scale,
                        mode=strategy,pair_window=self.group_d2_pair_window)
                    for metric,value in history_selection_stats.items():
                        if metric=='group_d2_max_real_ratio':
                            self._stats[metric]=max(float(self._stats.get(metric,0.)),float(value))
                        else:
                            self._stats[metric]=int(self._stats.get(metric,0))+int(value)
                    if strategy==GROUP_D2 and history_selection_stats['group_d2_real_violations']:
                        raise RuntimeError('Full Group-D2 failed its exact historical compactness invariant')
                elif physical_kind in ('h_exact','h_matched'):
                    accepted,history_selection_stats=exact_history_selection(
                        k,kk,ee,best,routed,gamma,cnt,importance,exact_current,
                        delta_ee,self.lambda_history*base_scale,
                        matched=(physical_kind=='h_matched'))
                    for metric,value in history_selection_stats.items():
                        if metric=='history_max_result_diameter_ratio':
                            self._stats[metric]=max(float(self._stats.get(metric,0.)),float(value))
                        else:
                            self._stats[metric]=int(self._stats.get(metric,0))+int(value)
                elif physical_kind=='matched':
                    accepted=nearest_count_matched_accept(best,routed,gamma,plan.accepted,ee)
                elif physical_kind=='importance_matched':
                    accepted=importance_count_matched(best,routed,plan.accepted,ee,cnt,importance)
                else:
                    accepted=plan.accepted if plan is not None else routed
                matched_mis_quota=(int(plan.accepted.sum().item()) if physical_kind in ('matched','importance_matched')
                                   else int(history_selection_stats.get('history_h1_selected_shadow',0)) if physical_kind=='h_matched' else 0)
                if physical_kind in ('matched','importance_matched','h_matched') and int(accepted.sum().item())!=matched_mis_quota:
                    raise AssertionError('Count-matched control failed per-anchor quota')
                if oracle_current is not None:
                    oracle_rows=audit_head(
                        oracle_current,kk.detach().cpu().numpy(),ee.detach().cpu().numpy(),
                        best.detach().cpu().numpy(),routed.detach().cpu().numpy(),
                        centers=centers.detach().cpu().numpy(),
                        radius_p=rads.detach().cpu().numpy(),
                        radius_2=r2s.detach().cpu().numpy(),
                        radius_inf=rinfs.detach().cpu().numpy(),
                        diameter_p=diams.detach().cpu().numpy(),
                        delta_ea=delta_ea,delta_cert=delta_cert,now=logical,
                        round_index=self.compress_rounds+1,layer=layer,head=h,
                        evictions_per_head=self.oracle_evictions_per_head,
                        anchors_per_eviction=self.oracle_anchors_per_eviction,
                        geometry_mode=self.geometry_mode,
                        physical_keys=k.detach().cpu().numpy())
                    self._stats.setdefault('oracle_pairs',[]).extend(oracle_rows)
                    summary=summarize_pairs(oracle_rows)
                    summary.update(round=self.compress_rounds+1,layer_idx=layer,kv_head_idx=h,
                                   logical_position=logical,delta_ea=delta_ea,delta_cert=delta_cert)
                    self._stats.setdefault('oracle_head_rounds',[]).append(summary)
                # Audit ACTUAL physical E-E conflicts for every compressed
                # method, including the historical conference control. A
                # missing audit must never be recorded as an artificial zero.
                physical_ee=physical_selected_ee_conflicts(k,ee,best,accepted,delta_ee)
                for metric,value in physical_ee.items():
                    self._stats[metric]=int(self._stats.get(metric,0))+int(value)
                if physical_kind=='matched':
                    self._stats['physical_mis_counterfactual_accepted']=int(
                        self._stats.get('physical_mis_counterfactual_accepted',0))+matched_mis_quota
                self._stats['routed_slots']+=int(routed.sum().item())
                self._stats['accepted_slots']+=int(accepted.sum().item())
                self._stats['rejected_slots']+=int((routed&~accepted).sum().item())
                dropped_history_count=int(gc[0,h,ee[~accepted]].sum().item())
                self._stats['dropped_historical_count']=int(self._stats.get('dropped_historical_count',0))+dropped_history_count
                accepted_history_count=int(gc[0,h,ee[accepted]].sum().item())
                repeated_accepted=int((gc[0,h,ee[accepted]]>1).sum().item())
                self._stats['accepted_previously_merged_slots']=int(
                    self._stats.get('accepted_previously_merged_slots',0))+repeated_accepted
                if plan is not None:self._stats['mis_conflict_edges']=self._stats.get('mis_conflict_edges',0)+int(plan.conflict_edges.sum().item())
                kf,vf,cf=fuse_weighted_slots(k,v,cnt,kk,ee,best,routed,accepted,
                    strategy,importance=importance)
                fidelity_stats,pair_samples,anchor_samples=audit_fusion_head(
                    k,v,cnt,importance,kk,ee,best,accepted,gamma,kf,vf,delta_ea,
                    topk=self.fusion_sample_topk)
                for sample in pair_samples:
                    sample.update(round=self.compress_rounds+1,layer_idx=layer,kv_head_idx=h,
                                  strategy=strategy,p=self.p,geometry_mode=self.geometry_mode,
                                  delta_ea=delta_ea)
                for sample in anchor_samples:
                    sample.update(round=self.compress_rounds+1,layer_idx=layer,kv_head_idx=h,
                                  strategy=strategy,p=self.p,geometry_mode=self.geometry_mode,
                                  delta_ea=delta_ea)
                self._stats.setdefault('fusion_pairs',[]).extend(pair_samples)
                self._stats.setdefault('fusion_anchors',[]).extend(anchor_samples)
                for key in ('fusion_accepted_pairs','fusion_touched_anchors',
                            'fusion_historical_pairs','fusion_countaware_excess_pairs',
                            'fusion_physical_over_delta_pairs'):
                    self._stats[key]=int(self._stats.get(key,0))+int(fidelity_stats[key])
                # Geometry tracks exactly represented original Keys. It does
                # NOT include AllCount-only rejected pseudo-count contributions.
                # Nor does an importance-weighted *attention key* alter the
                # arithmetic center of the represented historical Key vectors.
                members=(routed if not has_mis and strategy not in GROUP_METHODS else accepted)
                owners=[[] for _ in range(B)]
                owner_ids=best[members].tolist()
                physical_ids=ee[members].tolist()
                for owner_id,physical_id in zip(owner_ids,physical_ids):
                    owners[owner_id].append(physical_id)
                if exact_current is not None:
                    self.exact_histories[(layer,h)]=exact_current.merge(
                        kk.detach().cpu().numpy(),ee.detach().cpu().numpy(),
                        best.detach().cpu().numpy(),accepted.detach().cpu().numpy())
                    if len(self.exact_histories[(layer,h)].slots)!=B:
                        raise AssertionError('Exact historical oracle did not follow physical slots')
                if oracle_current is not None:
                    # Must follow ACTUAL accepted decisions, not counterfactual
                    # gates; otherwise next-round exact ground truth is wrong.
                    self.oracle_histories[(layer,h)]=merged_histories(
                        oracle_current,kk.detach().cpu().numpy(),
                        ee.detach().cpu().numpy(),best.detach().cpu().numpy(),
                        members.detach().cpu().numpy())
                    if len(self.oracle_histories[(layer,h)])!=B:
                        raise AssertionError('Oracle fusion not aligned to physical budget')
                if strategy in GROUP_METHODS:
                    c_next,m_next,m2_next=merge_group_d2_history(
                        kk,ee,best,accepted,gc[0,h],centers,moments2)
                    # Geometric invariant uses the double-precision original-Key
                    # moment ledger, NEVER the quantized physical K output.
                    d2_next=torch.where(c_next>1,
                        (2*m2_next/(c_next-1).clamp_min(1)).clamp_min(0).sqrt(),
                        torch.zeros_like(m2_next))
                    if strategy==GROUP_D2 and bool((d2_next>self.lambda_history*base_scale*(1+1e-8)).any()):
                        raise RuntimeError('Group-D2 selected merge exceeded exact history D2 threshold')
                    r_next=torch.zeros_like(c_next)
                    d_next=d2_next
                    r2_next=torch.zeros_like(c_next)
                    rinf_next=torch.zeros_like(c_next)
                    dinf_next=torch.zeros_like(c_next)
                elif tracks_history:
                    Hc=[];Hm=[];Hr=[];Hd=[];Hr2=[];Hrinf=[];Hdinf=[]
                    for ji,ki in enumerate(kk.tolist()):
                        indexes=[ki]+owners[ji]
                        ids=torch.tensor(indexes,device=dev,dtype=torch.long)
                        if self.geometry_mode in ('moment_interp','fourth_moment','count_aware','count_aware_phys_075','count_aware_phys_100'):
                            C_new,M_new,R_new,D_new,R2_new,Rinf_new,Dinf_new=merge_cluster_moments(
                                gc[0,h,ids],centers[ids],rads[ids],diams[ids],
                                r2s[ids],rinfs[ids],dinfs[ids],self.p,
                                geometry_mode=self.geometry_mode)
                        else:
                            C_new,M_new,R_new,D_new=merge_cluster_tensors(
                                gc[0,h,ids],centers[ids],rads[ids],diams[ids],self.p,
                                geometry_mode=self.geometry_mode)
                            R2_new=torch.zeros_like(R_new)
                            Rinf_new=torch.zeros_like(R_new)
                            Dinf_new=torch.zeros_like(D_new)
                        if bool(D_new>delta_cert+2e-9*max(1.,delta_cert)):
                            if has_full_certificate:
                                raise RuntimeError(f'p-compactness invariant failed mode={self.geometry_mode} '
                                                   f'method={strategy} p={self.p} round={self.compress_rounds+1} '
                                                   f'layer={layer} head={h} count={float(C_new.item()):.0f} '
                                                   f'D={float(D_new.item()):.12g} limit={delta_cert:.12g} '
                                                   f'overshoot={float(D_new.item())-delta_cert:.6g}')
                            # Expected possible outcome of disabling E-E MIS
                            # or disabling certified E-A while retaining E-E MIS.
                            self._stats['geometry_over_delta']+=1
                        Hc.append(C_new);Hm.append(M_new)
                        Hr.append(R_new);Hd.append(D_new)
                        Hr2.append(R2_new);Hrinf.append(Rinf_new);Hdinf.append(Dinf_new)
                    r2_next=torch.stack(Hr2,dim=0)
                    rinf_next=torch.stack(Hrinf,dim=0)
                    dinf_next=torch.stack(Hdinf,dim=0)
                    c_next=torch.stack(Hc,dim=0)
                    m_next=torch.stack(Hm,dim=0)
                    r_next=torch.stack(Hr,dim=0)
                    d_next=torch.stack(Hd,dim=0)
                else:
                    # Baseline does not claim historical p-compactness. Skip
                    # O(group^2) geometric propagation; zero radius/diameter
                    # are *unused placeholders*, not certified observations.
                    c_next=torch.zeros((B,),device=dev,dtype=torch.float64)
                    c_next.scatter_add_(0,best[members],gc[0,h,ee[members]])
                    c_next+=gc[0,h,kk]
                    m_next=kf.detach().double()
                    r_next=torch.zeros((B,),device=dev,dtype=torch.float64)
                    d_next=torch.zeros_like(r_next)
                    r2_next=torch.zeros_like(r_next)
                    rinf_next=torch.zeros_like(r_next)
                    dinf_next=torch.zeros_like(r_next)
                if strategy not in GROUP_METHODS:
                    m2_next=torch.zeros((B,),device=dev,dtype=torch.float64)
                kout.append(kf);vout.append(vf);cout.append(cf);pout.append(positions[0,h,kk])
                gcount.append(c_next);gmean.append(m_next)
                grad.append(r_next);gdiam.append(d_next)
                gR2.append(r2_next);gRinf.append(rinf_next);gDinf.append(dinf_next);gM2.append(m2_next)
                self._stats.setdefault('head_rounds',[]).append(dict(
                    round=self.compress_rounds+1,layer_idx=layer,kv_head_idx=h,
                    strategy=strategy,p=self.p,geometry_mode=self.geometry_mode,
                    lambda_ea=self.lambda_ea,lambda_ee=self.lambda_ee,
                    delta_ea=delta_ea,delta_ee=delta_ee,delta_cert=delta_cert,
                    history_certified=bool(has_full_certificate or physical_kind=='h_exact' or strategy==GROUP_D2),
                    history_tracked=bool(tracks_history or physical_kind in ('h_exact','h_matched') or strategy in GROUP_METHODS),
                    ea_certificate_used=bool(uses_history_ea),  # false for exact-history oracle; separate flag follows
                    ee_mis_used=bool(actual_mis),
                    historical_exact_oracle=bool(physical_kind=='h_exact'),
                    historical_shadow_matched=bool(physical_kind=='h_matched'),
                    lambda_history=self.lambda_history,
                    delta_history=self.lambda_history*base_scale,
                    **history_selection_stats,
                    physical_ea_gate_ratio=(physical_method_spec(strategy)[0] if physical_method else ''),
                    selection_rule=('eviction_only' if strategy=='eviction_only' else
                                    'conference_inf_certificate_mis' if strategy=='coherekv_inf_certificate' else
                                    'nearest_countmatched' if physical_kind=='matched' else
                                    'importance_countmatched' if physical_kind=='importance_matched' else
                                    'group_d2_exact' if strategy==GROUP_D2 else
                                    'group_d2_ignore_M2' if strategy==WO_HISTORY else
                                    'group_d2_independent_anchor' if strategy==WO_EE else
                                    'group_d2_quota_importance' if strategy==MATCHED else
                                    'exact_history_mis' if physical_kind=='h_exact' else
                                    'history_quota_physical_EE_nearest' if physical_kind=='h_matched' else
                                    'physical_mis' if physical_kind=='mis' else
                                    'physical_no_mis' if physical_method else 'historical'),
                    physical_mis_counterfactual_accepted=matched_mis_quota,
                    real_history_d2_over_delta_max=(float(d_next.max().item())/(self.lambda_history*base_scale) if strategy in GROUP_METHODS else ''),
                    group_d2_exact_history_m2_mean=(float(m2_next.mean().item()) if strategy in GROUP_METHODS else ''),
                    **physical_ee,
                    before_slots=n,after_slots=B,evicted_slots=int(ee.numel()),
                    routed_slots=int(routed.sum().item()),accepted_slots=int(accepted.sum().item()),
                    rejected_slots=int((routed&~accepted).sum().item()),
                    dropped_historical_count=dropped_history_count,
                    accepted_historical_count=accepted_history_count,
                    accepted_previously_merged_slots=repeated_accepted,
                    physical_gate_ratio=(physical_count_aware_ratio(self.geometry_mode) or ''),
                    **fidelity_stats,
                    ea_countaware_phys_075_possible=gate_diagnostics.get('ea_countaware_phys_075_possible',''),
                    ea_countaware_phys_100_possible=gate_diagnostics.get('ea_countaware_phys_100_possible',''),
                    ea_countaware_phys_075_rescue_moment=gate_diagnostics.get('ea_countaware_phys_075_rescue_moment',''),
                    ea_countaware_phys_100_rescue_moment=gate_diagnostics.get('ea_countaware_phys_100_rescue_moment',''),
                    ea_countaware_phys_075_blocks_countaware=gate_diagnostics.get('ea_countaware_phys_075_blocks_countaware',''),
                    ea_countaware_phys_100_blocks_countaware=gate_diagnostics.get('ea_countaware_phys_100_blocks_countaware',''),
                    ea_countaware_phys_075_merged_possible=gate_diagnostics.get('ea_countaware_phys_075_merged_possible',''),
                    ea_countaware_phys_100_merged_possible=gate_diagnostics.get('ea_countaware_phys_100_merged_possible',''),
                    ea_physical_possible=gate_diagnostics.get('ea_physical_possible',''),
                    ea_minkowski_possible=gate_diagnostics.get('ea_minkowski_possible',''),
                    ea_exact_p2_possible=gate_diagnostics.get('ea_exact_p2_possible',''),
                    ea_exact_rescues_minkowski=gate_diagnostics.get('ea_exact_rescues_minkowski',''),
                    ea_moment_possible=gate_diagnostics.get('ea_moment_possible',''),
                    ea_moment_rescues_minkowski=gate_diagnostics.get('ea_moment_rescues_minkowski',''),
                    ea_fourth_possible=gate_diagnostics.get('ea_fourth_possible',''),
                    ea_fourth_rescues_minkowski=gate_diagnostics.get('ea_fourth_rescues_minkowski',''),
                    ea_fourth_rescues_moment=gate_diagnostics.get('ea_fourth_rescues_moment',''),
                    ea_moment_only_not_fourth=gate_diagnostics.get('ea_moment_only_not_fourth',''),
                    ea_physical_but_not_exact=gate_diagnostics.get('ea_physical_but_not_exact',''),
                    ea_countaware_possible=gate_diagnostics.get('ea_countaware_possible',''),
                    ea_countaware_rescues_moment=gate_diagnostics.get('ea_countaware_rescues_moment',''),
                    ea_countaware_only_moment=gate_diagnostics.get('ea_countaware_only_moment',''),
                    ea_countaware_merged_evicted_possible=gate_diagnostics.get('ea_countaware_merged_evicted_possible',''),
                    ea_moment_merged_evicted_possible=gate_diagnostics.get('ea_moment_merged_evicted_possible',''),
                    ea_countaware_rescues_previously_merged=gate_diagnostics.get('ea_countaware_rescues_previously_merged',''),
                    ea_countaware_over_delta_possible=gate_diagnostics.get('ea_countaware_over_delta_possible',''),
                    ea_countaware_eligible_evicted=gate_diagnostics.get('ea_countaware_eligible_evicted',''),
                    ea_countaware_eligible_merged_evicted=gate_diagnostics.get('ea_countaware_eligible_merged_evicted',''),
                    ea_old_evicted_over_limit=gate_diagnostics.get('ea_old_evicted_over_limit',''),
                    ea_old_anchor_over_limit=gate_diagnostics.get('ea_old_anchor_over_limit',''),
                    max_radius_over_delta=(float(r_next.max().item())/delta_cert if tracks_history else ''),
                    max_diameter_over_delta=(float(d_next.max().item())/delta_cert if tracks_history else ''),
                    mean_radius_over_delta=(float(r_next.mean().item())/delta_cert if tracks_history else ''),
                    mean_diameter_over_delta=(float(d_next.mean().item())/delta_cert if tracks_history else ''),
                    mean_exact_r2_over_delta=(float(r2_next.mean().item())/delta_cert if tracks_history and self.geometry_mode in ('moment_interp','fourth_moment','count_aware','count_aware_phys_075','count_aware_phys_100') else ''),
                    max_rinf_over_delta=(float(rinf_next.max().item())/delta_cert if tracks_history and self.geometry_mode in ('moment_interp','fourth_moment','count_aware','count_aware_phys_075','count_aware_phys_100') else '')))
            new_kv.append((torch.stack(kout,0)[None],torch.stack(vout,0)[None]))
            new_count.append(torch.stack(cout,0)[None]);new_positions.append(torch.stack(pout,0)[None])
            new_gc.append(torch.stack(gcount,0)[None]);new_gm.append(torch.stack(gmean,0)[None])
            new_gr.append(torch.stack(grad,0)[None]);new_gd.append(torch.stack(gdiam,0)[None])
            new_gr2.append(torch.stack(gR2,0)[None]);new_grinf.append(torch.stack(gRinf,0)[None]);new_gdinf.append(torch.stack(gDinf,0)[None])
            new_group_m2.append(torch.stack(gM2,0)[None])
        _replace_dynamic_cache_inplace(self,new_kv)
        self.counts=new_count;self.slot_positions=new_positions
        self.geom_counts=new_gc;self.geom_means=new_gm
        self.geom_radii=new_gr;self.geom_diameters=new_gd
        self.geom_r2=new_gr2;self.geom_rinf=new_grinf;self.geom_dinf=new_gdinf
        self.group_m2=new_group_m2
        if self.reference_mode=='online_first' and self.online_ref is None:
            self.online_ref=online_ref
        self.compress_rounds+=1
        self._stats['compress_rounds']+=1
        self._stats['last_physical_kv_len']=B
        self._stats['last_logical_tokens']=logical
        self._stats['max_before_len']=max(self._stats.get('max_before_len',0),n)
        self._stats['max_after_len']=max(self._stats.get('max_after_len',0),B)
        if int(self.key_cache[0].shape[-2])!=B:
            raise RuntimeError('Wrong physical KV capacity after compression')
        for li,l in enumerate(self._ppl_model.model.layers):
            attn=l.self_attn
            attn._mymethod_counts_kv=new_count[li]
            attn._mymethod_eager_enabled=True
            attn._mymethod_sdpa_enabled=False
            attn._keepkv_eager_enabled=False
            attn._mymethod_selector='keepkv'
            attn._mymethod_score_window=self.score_window
            attn._kvmerge_attention_chunk_size=self.chunk_size


def args_parser():
    p=argparse.ArgumentParser(description='Multi-round continuous p-distance CohereKV/MIS PPL')
    p.add_argument('--model',required=True)
    p.add_argument('--calibration',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--dataset',default='wikitext2',choices=['wikitext2'])
    p.add_argument('--sequence_index',type=int,default=0,
                   help='Disjoint test-token window index; each window contains cycles*(U+S) tokens')
    p.add_argument('--cycles',type=int,default=4)
    p.add_argument('--unscored_len',type=int,default=16384)
    p.add_argument('--scored_len',type=int,default=256)
    p.add_argument('--p_values',default='2,4,8,16,inf')
    p.add_argument('--geometry_modes',default='minkowski',
                   help='Comma-separated: minkowski,moment_interp,count_aware,count_aware_phys_075,count_aware_phys_100,fourth_moment,exact_p2; physical variants cap physical Key distance at 0.75/1.00 times EA delta')
    p.add_argument('--route_diagnostics',type=int,choices=[0,1],default=1)
    p.add_argument('--oracle_enabled',type=int,choices=[0,1],default=0)
    p.add_argument('--oracle_methods',default='k_only_importance_mis_drop',
                   help='Comma-separated methods whose actual history will be sampled')
    p.add_argument('--oracle_layers',default='0,14,27')
    p.add_argument('--oracle_heads',default='0,1')
    p.add_argument('--oracle_evictions_per_head',type=int,default=16)
    p.add_argument('--oracle_anchors_per_eviction',type=int,default=4)
    p.add_argument('--lambda_pairs',default=None,
                   help='Explicit EA:EE pairs, comma-separated. Overrides lambda_ea/lambda_ee cartesian grid. Example 0.45:0.3375,0.60:0.60')
    p.add_argument('--lambda_history',type=float,default=0.75,
                   help='Exact final Group-D2 threshold divided by frozen reference')
    p.add_argument('--group_d2_pair_window',type=int,default=8,
                   help='Bounded lookahead candidates after failed singleton (0 disables joint recovery; range 0..128)')
    p.add_argument('--fusion_sample_topk',type=int,default=4,
                   help='Deterministic per-head fusion sample rows; zero suppresses huge raw CSV, preserving aggregate statistics')
    p.add_argument('--lambda_ea',default='0.5')
    p.add_argument('--lambda_ee',default='0.5')
    p.add_argument('--budgets',default='0.1')
    p.add_argument('--reference_mode',default='online_first',choices=['online_first','frozen_train'])
    p.add_argument('--mis_families',default='k_only,k_only_importance')
    p.add_argument('--mis_methods',default=None)
    p.add_argument('--score_window',type=int,default=128)
    p.add_argument('--sink_size',type=int,default=4)
    p.add_argument('--recent_protect_size',type=int,default=128)
    p.add_argument('--recency_beta',type=float,default=1.0)
    p.add_argument('--prefill_chunk_size',type=int,default=128)
    p.add_argument('--distance_chunk_size',type=int,default=256)
    p.add_argument('--graph_backend',default='streamed')
    p.add_argument('--torch_dtype',default='bfloat16')
    p.add_argument('--attn_implementation',default='eager')
    p.add_argument('--device',default='cuda')
    p.add_argument('--device_map',default='auto')
    p.add_argument('--trust_remote_code',action='store_true')
    p.add_argument('--use_fast_tokenizer',action='store_true')
    return p


def main():
    a=args_parser().parse_args()
    if a.sequence_index<0:raise ValueError('sequence_index must be nonnegative')
    if not 0<=a.group_d2_pair_window<=128:
        raise ValueError('group_d2_pair_window must be between 0 and 128')
    if not 0<=a.fusion_sample_topk<=100:
        raise ValueError('fusion_sample_topk must be in 0..100')
    if a.cycles<1 or a.unscored_len<2 or a.scored_len<1 or a.score_window<1 or a.prefill_chunk_size<a.score_window:
        raise ValueError('Invalid cycles/length/score window/chunk size')
    if a.sink_size<0 or a.recent_protect_size<0 or not(math.isfinite(a.recency_beta) and a.recency_beta>=1.0):
        raise ValueError('sink/recent protection must be nonnegative; recency_beta >=1 finite')
    if a.device!='cuda' and a.torch_dtype in ('float16','bfloat16'):
        raise ValueError('Use float32 for CPU')
    ps=_csv_floats(a.p_values,'p',p=True)
    geometry_modes=tuple(x.strip() for x in a.geometry_modes.split(',') if x.strip())
    if not geometry_modes or len(set(geometry_modes))!=len(geometry_modes) or \
       any(x not in ('minkowski','exact_p2','moment_interp','fourth_moment','count_aware','count_aware_phys_075','count_aware_phys_100') for x in geometry_modes):
        raise ValueError('geometry_modes must be unique supported geometry modes')
    if a.lambda_pairs:
        lambda_pairs=parse_lambda_pairs(a.lambda_pairs)
    else:
        lambda_pairs=tuple((ea,ee) for ea in _csv_floats(a.lambda_ea,'lambda_ea')
                                     for ee in _csv_floats(a.lambda_ee,'lambda_ee'))
    lambda_eas=list(dict.fromkeys(ea for ea,ee in lambda_pairs))
    lambda_ee_by_ea={ea:[ee for ea2,ee in lambda_pairs if ea2==ea] for ea in lambda_eas}
    budgets=_csv_floats(a.budgets,'budget')
    if any(not 0<b<1 for b in budgets):raise ValueError('Budget fractions must be strictly between 0 and 1')
    methods=selected_methods(a.mis_families,a.mis_methods)
    if any(x in GROUP_METHODS for x in methods) and (ps!=[2.] or geometry_modes!=('exact_p2',)):
        raise ValueError('Group-D2 experiment must use p=2 and exact_p2 geometry mode')
    total_length=a.cycles*(a.unscored_len+a.scored_len)
    oracle_methods={m.strip() for m in a.oracle_methods.split(',') if m.strip()}
    oracle_layers=[int(x.strip()) for x in a.oracle_layers.split(',') if x.strip()]
    oracle_heads=[int(x.strip()) for x in a.oracle_heads.split(',') if x.strip()]
    if a.oracle_enabled and (not oracle_methods or not oracle_layers or not oracle_heads
           or a.oracle_evictions_per_head<1 or a.oracle_anchors_per_eviction<1):
        raise ValueError('Oracle needs methods/layers/heads and positive sample counts')
    cfg=AutoConfig.from_pretrained(a.model,trust_remote_code=a.trust_remote_code)
    if a.oracle_enabled and (any(x<0 or x>=cfg.num_hidden_layers for x in oracle_layers)
          or any(x<0 or x>=cfg.num_key_value_heads for x in oracle_heads)):
        raise ValueError('Oracle layer/head indices outside model configuration')
    if a.oracle_enabled and not any(m in methods for m in oracle_methods):
        raise ValueError('Oracle methods do not intersect methods enabled in run script')
    if a.oracle_enabled and (4. not in ps or not any(g in geometry_modes for g in
            ('moment_interp','fourth_moment','count_aware','count_aware_phys_075','count_aware_phys_100'))):
        raise ValueError('Exact p4 oracle needs p=4 and moment_interp/fourth_moment geometry')
    model_type=str(cfg.model_type).lower()
    family={'qwen2':'qwen','llama':'llama'}.get(model_type)
    if family is None:raise ValueError('Only Qwen2/Llama patch is supported')
    max_ctx=getattr(cfg,'max_position_embeddings',None)
    if max_ctx is not None and total_length>int(max_ctx):
        raise ValueError(f'Continuous {total_length} logical tokens exceeds model max_position_embeddings={max_ctx}. '
                         'Reduce cycles/length or select a model with a validated longer-context configuration; '
                         'never wrap/reset RoPE position IDs')
    info,reference,_,_,gate=load_frozen_calibration(a.calibration,a.model)
    if reference.shape!=(cfg.num_hidden_layers,cfg.num_key_value_heads):
        raise ValueError('Frozen train reference shape does not match current model')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    signature=dict(vars(a),total_logical_tokens=total_length,methods=methods,
                   reference_sha256=hashlib.sha256(Path(a.calibration).read_bytes()).hexdigest(),
                   implementation='journal_group_d2_crossround_validation_v20',
                   token_start=a.sequence_index*total_length,
                   token_end=(a.sequence_index+1)*total_length,
                   resolved_lambda_pairs=[dict(lambda_ea=ea,lambda_ee=ee,ee_over_ea=ee/ea)
                                          for ea,ee in lambda_pairs])
    (out/'continuous_config.json').write_text(json.dumps(signature,indent=2),encoding='utf8')
    tokenizer=AutoTokenizer.from_pretrained(a.model,use_fast=a.use_fast_tokenizer,
        trust_remote_code=a.trust_remote_code)
    token_start=a.sequence_index*total_length
    token_end=token_start+total_length
    ids=get_eval_data(a.dataset,token_end,tokenizer)
    if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[-1]<token_end:
        raise ValueError(f'Insufficient contiguous test tokens through position {token_end}; '
                         f'only {ids.shape[-1]} available')
    ids=ids[:,token_start:token_end].contiguous()
    signature['scored_stream_sha256']=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
    (out/'continuous_config.json').write_text(json.dumps(signature,indent=2),encoding='utf8')
    print(f'[SEQUENCE] id={a.sequence_index} positions=[{token_start},{token_end}) '
          f'sha256={signature["scored_stream_sha256"][:16]}',flush=True)
    kwargs=dict(torch_dtype=parse_dtype(a.torch_dtype),
                attn_implementation=a.attn_implementation,trust_remote_code=a.trust_remote_code)
    if a.device_map.lower()!='none':kwargs['device_map']=a.device_map
    model=AutoModelForCausalLM.from_pretrained(a.model,**kwargs).eval()
    if a.device_map.lower()=='none':model.to(a.device)
    device=first_execution_device(model,a.device)
    apply_runtime_patch_if_needed(family,TASK_ROOT,'mymethod')
    all_results=[];all_cycles=[];per_heads=[];all_oracle_pairs=[];all_oracle_summaries=[];all_fusion_pairs=[];all_fusion_anchors=[]
    # FullCache ignores all geometry settings. Do NOT reuse MergeAll across
    # lambda_ee even when its K/V and PPL would be the same: accepted physical
    # E-E conflict auditing explicitly depends on lambda_ee. Reusing those
    # counts across thresholds would silently corrupt the sweep analysis.
    baseline_cache={}
    for budget in budgets:
        for p in ps:
            for geometry_mode in geometry_modes:
                if geometry_mode=='exact_p2' and p!=2.:
                    continue
                if geometry_mode in ('moment_interp','count_aware','count_aware_phys_075','count_aware_phys_100') and p<2.:
                    continue
                if geometry_mode=='fourth_moment' and p!=4.:
                    continue
                for lambda_ea in lambda_eas:
                    for lambda_ee in lambda_ee_by_ea[lambda_ea]:
                        for method in methods:
                            stats={};t0=time.time()
                            # Only a genuinely identical run may be reused.
                            # In particular lambda_ee changes the measured
                            # physical E-E conflicts even for no-MIS controls.
                            baseline_key=('full_cache',) if method=='full_cache' else None
                            reused_baseline=baseline_key is not None and baseline_key in baseline_cache
                            if reused_baseline:
                                ppl,cycles,stats=copy.deepcopy(baseline_cache[baseline_key])
                            else:
                                prepare_model_for_new_sequence(model,a.score_window,a.prefill_chunk_size)
                                def factory():
                                    return ContinuousPCache(model=model,method=method,budget=budget,
                                        p=p,lambda_ea=lambda_ea,lambda_ee=lambda_ee,
                                        lambda_history=a.lambda_history,
                                        group_d2_pair_window=a.group_d2_pair_window,
                                        fusion_sample_topk=a.fusion_sample_topk,
                                        reference=reference,sink_gate=gate,
                                        reference_mode=a.reference_mode,
                                        unscored_len=a.unscored_len,scored_len=a.scored_len,
                                        score_window=a.score_window,
                                        sink_size=a.sink_size,recent_protect_size=a.recent_protect_size,
                                        recency_beta=a.recency_beta,chunk_size=a.distance_chunk_size,
                                        graph_backend=a.graph_backend,geometry_mode=geometry_mode,
                                        route_diagnostics=bool(a.route_diagnostics),stats=stats,
                                        oracle_enabled=bool(a.oracle_enabled and p==4.
                                                and geometry_mode in ('moment_interp','fourth_moment','count_aware','count_aware_phys_075','count_aware_phys_100')
                                                and method in oracle_methods),
                                        oracle_layers=oracle_layers,oracle_heads=oracle_heads,
                                        oracle_evictions_per_head=a.oracle_evictions_per_head,
                                        oracle_anchors_per_eviction=a.oracle_anchors_per_eviction)
                                print(f'[CONTINUOUS] {method} p={p} lambda_ea={lambda_ea} lambda_ee={lambda_ee} B={budget} '
                                      f'mode={geometry_mode} cycles={a.cycles} U={a.unscored_len} S={a.scored_len}',flush=True)
                                if method=='full_cache':
                                    ppl,cycles=evaluate_continuous(model,ids,a.cycles,a.unscored_len,a.scored_len,device,
                                        cache_factory=None,prefill_chunk_size=a.prefill_chunk_size,
                                        score_window=a.score_window,amp_dtype=parse_dtype(a.torch_dtype) if device.type=='cuda' else None)
                                else:
                                    with force_attn_hidden_and_compress(model):
                                        ppl,cycles=evaluate_continuous(model,ids,a.cycles,a.unscored_len,a.scored_len,device,
                                            cache_factory=factory,prefill_chunk_size=a.prefill_chunk_size,
                                            score_window=a.score_window,amp_dtype=parse_dtype(a.torch_dtype) if device.type=='cuda' else None)
                                    if stats.get('compress_rounds')!=a.cycles:
                                        raise RuntimeError(f'Wrong number of compressions: {stats.get("compress_rounds")}')
                                if baseline_key is not None:
                                    baseline_cache[baseline_key]=copy.deepcopy((ppl,cycles,stats))
                            if reused_baseline:
                                print(f'[REUSED BASELINE] {method} p={p} lambda_ea={lambda_ea} lambda_ee={lambda_ee}',flush=True)
                            common=dict(model=a.model,dataset=a.dataset,
                                        sequence_index=a.sequence_index,
                                        token_start=token_start,token_end=token_end,
                                        stream_sha256=signature['scored_stream_sha256'],method=method,
                                        mis_family=family_for_method(method),
                                        p=(math.inf if method=='coherekv_inf_certificate' else p),
                                        geometry_mode=('minkowski' if method=='coherekv_inf_certificate' else geometry_mode),
                                        lambda_ea=lambda_ea,lambda_ee=lambda_ee,
                                        ee_over_ea=lambda_ee/lambda_ea,
                                        budget=budget,unscored_len=a.unscored_len,
                                        scored_len=a.scored_len,cycles=a.cycles)
                            for record in cycles:all_cycles.append(dict(**common,**record))
                            for record in stats.get('head_rounds',[]):
                                per_heads.append({**record,**common})
                            for record in stats.get('fusion_pairs',[]):
                                all_fusion_pairs.append({**record,**common})
                            for record in stats.get('fusion_anchors',[]):
                                all_fusion_anchors.append({**record,**common})
                            for record in stats.get('oracle_pairs',[]):
                                all_oracle_pairs.append({**common,**record})
                            for record in stats.get('oracle_head_rounds',[]):
                                all_oracle_summaries.append({**common,**record})
                            result=dict(**common,scored_tokens=a.cycles*a.scored_len,
                                        ppl=ppl,nll_per_token=math.log(ppl),
                                        compress_rounds=stats.get('compress_rounds',0),
                                        accepted_slots=stats.get('accepted_slots',0),
                                        routed_slots=stats.get('routed_slots',0),
                                        rejected_slots=stats.get('rejected_slots',0),
                                        mis_conflict_edges=stats.get('mis_conflict_edges',0),
                                        geometry_over_delta=stats.get('geometry_over_delta',0),
                                    dropped_historical_count=stats.get('dropped_historical_count',0),
                                    ea_exact_rescues_minkowski=stats.get('ea_exact_rescues_minkowski',0),
                                    ea_minkowski_possible=stats.get('ea_minkowski_possible',0),
                                    ea_moment_possible=stats.get('ea_moment_possible',0),
                                    ea_moment_rescues_minkowski=stats.get('ea_moment_rescues_minkowski',0),
                                    ea_fourth_possible=stats.get('ea_fourth_possible',0),
                                    ea_fourth_rescues_minkowski=stats.get('ea_fourth_rescues_minkowski',0),
                                    ea_fourth_rescues_moment=stats.get('ea_fourth_rescues_moment',0),
                                    ea_moment_only_not_fourth=stats.get('ea_moment_only_not_fourth',0),
                                    ea_exact_p2_possible=stats.get('ea_exact_p2_possible',0),
                                    ea_physical_possible=stats.get('ea_physical_possible',0),
                                    ea_countaware_possible=stats.get('ea_countaware_possible',0),
                                    ea_countaware_rescues_moment=stats.get('ea_countaware_rescues_moment',0),
                                    ea_countaware_only_moment=stats.get('ea_countaware_only_moment',0),
                                    ea_countaware_merged_evicted_possible=stats.get('ea_countaware_merged_evicted_possible',0),
                                    ea_moment_merged_evicted_possible=stats.get('ea_moment_merged_evicted_possible',0),
                                    ea_countaware_rescues_previously_merged=stats.get('ea_countaware_rescues_previously_merged',0),
                                    ea_countaware_over_delta_possible=stats.get('ea_countaware_over_delta_possible',0),
                                    ea_countaware_phys_075_possible=stats.get('ea_countaware_phys_075_possible',0),
                                    ea_countaware_phys_100_possible=stats.get('ea_countaware_phys_100_possible',0),
                                    ea_countaware_phys_075_blocks_countaware=stats.get('ea_countaware_phys_075_blocks_countaware',0),
                                    ea_countaware_phys_100_blocks_countaware=stats.get('ea_countaware_phys_100_blocks_countaware',0),
                                    fusion_accepted_pairs=stats.get('fusion_accepted_pairs',0),
                                    fusion_touched_anchors=stats.get('fusion_touched_anchors',0),
                                    fusion_countaware_excess_pairs=stats.get('fusion_countaware_excess_pairs',0),
                                    fusion_physical_over_delta_pairs=stats.get('fusion_physical_over_delta_pairs',0),
                                    physical_ee_accepted_pairs=stats.get('physical_ee_accepted_pairs',0),
                                    physical_ee_accepted_conflicts=stats.get('physical_ee_accepted_conflicts',0),
                                    physical_mis_counterfactual_accepted=stats.get('physical_mis_counterfactual_accepted',0),
                                    lambda_history=a.lambda_history,
                                    group_d2_pair_window=a.group_d2_pair_window,
                                    historical_exact_oracle=bool(method=='necessity_history_exact'),
                                    history_ea_candidates=stats.get('history_ea_candidates',0),
                                    history_ea_passed=stats.get('history_ea_passed',0),
                                    history_ea_rejected=stats.get('history_ea_rejected',0),
                                    history_ee_extra_edges=stats.get('history_ee_extra_edges',0),
                                    history_h1_selected_shadow=stats.get('history_h1_selected_shadow',0),
                                    history_max_result_diameter_ratio=stats.get('history_max_result_diameter_ratio',0.),
                                    history_matched_swaps=stats.get('history_matched_swaps',0),
                                    history_matched_groups=stats.get('history_matched_groups',0),
                                    history_matched_fallback_groups=stats.get('history_matched_fallback_groups',0),
                                    history_matched_quota_shortfall=stats.get('history_matched_quota_shortfall',0),
                                    accepted_previously_merged_slots=stats.get('accepted_previously_merged_slots',0),
                                    group_d2_anchors=stats.get('group_d2_anchors',0),
                                    group_d2_accepted=stats.get('group_d2_accepted',0),
                                    group_d2_pair_attempts=stats.get('group_d2_pair_attempts',0),
                                    group_d2_pair_success=stats.get('group_d2_pair_success',0),
                                    group_d2_real_violations=stats.get('group_d2_real_violations',0),
                                    group_d2_max_real_ratio=stats.get('group_d2_max_real_ratio',0.),
                                    group_d2_shadow_quota=stats.get('group_d2_shadow_quota',0),
                                    group_d2_matched_quota_shortfall=stats.get('group_d2_matched_quota_shortfall',0),
                                    group_d2_same_state_matched_swaps=stats.get('group_d2_same_state_matched_swaps',0),
                                    group_d2_same_state_matched_violating_anchors=stats.get('group_d2_same_state_matched_violating_anchors',0),
                                    group_d2_same_state_matched_anchors=stats.get('group_d2_same_state_matched_anchors',0),
                                        max_before_len=stats.get('max_before_len',0),
                                        max_after_len=stats.get('max_after_len',0),
                                        seconds=(0.0 if reused_baseline else time.time()-t0),
                                        reused_baseline=bool(reused_baseline))
                            all_results.append(result)
                            _save_csv(out/'ppl_continuous.csv',all_results)
                            _save_csv(out/'ppl_continuous_cycles.csv',all_cycles)
                            _save_csv(out/'ppl_continuous_head_rounds.csv',per_heads)
                            _save_csv(out/'ppl_continuous_fusion_pairs.csv',all_fusion_pairs)
                            _save_csv(out/'ppl_continuous_fusion_anchors.csv',all_fusion_anchors)
                            _save_csv(out/'ppl_continuous_oracle_pairs.csv',all_oracle_pairs)
                            _save_csv(out/'ppl_continuous_oracle_head_rounds.csv',all_oracle_summaries)
                            print(f'[PPL] method={method} p={p} lambda_ea={lambda_ea} lambda_ee={lambda_ee} ppl={ppl:.6f} '
                                  f'geometry_mode={geometry_mode} compressions={result["compress_rounds"]}',flush=True)
    if a.oracle_enabled and not all_oracle_pairs:
        raise RuntimeError('Oracle was enabled but no candidate pairs were recorded')
    completed_sweep=build_sweep_summary(all_results)
    if all(method in methods for method in COMPARATORS):
        valid_p_modes=sum(1 for pval in ps for mode in geometry_modes
                          if not (mode=='exact_p2' and pval!=2.)
                          and not (mode in ('moment_interp','count_aware','count_aware_phys_075',
                                            'count_aware_phys_100') and pval<2.)
                          and not (mode=='fourth_moment' and pval!=4.))
        expected=len(budgets)*valid_p_modes*len(lambda_pairs)
        if len(completed_sweep)!=expected:
            raise RuntimeError(f'Incomplete physical lambda comparison: {len(completed_sweep)} of {expected} paired triples')
    (out/'continuous_run_complete.json').write_text(json.dumps(dict(config=signature,
        n_results=len(all_results),n_cycles=len(all_cycles),
        n_lambda_sweep_pairs_completed=len(completed_sweep),
        n_group_d2_arms_completed=sum(m in methods for m in GROUP_METHODS)),indent=2),encoding='utf8')


if __name__=='__main__':main()
