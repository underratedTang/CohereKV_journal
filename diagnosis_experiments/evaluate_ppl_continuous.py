#!/usr/bin/env python3
"""Continuous multi-round journal Hard MIS vs Full Cache / Eviction / MergeAll.

For each cycle, feed U-1 unscored tokens; compress at that logical boundary;
feed the last unscored token AFTER compression (causal warm-up); then feed S
scored tokens, and score exactly those S targets. No reset between cycles.
All tokens are distinct positions in one contiguous WikiText-2 validation/test stream.

Run from the CohereKV repository root, with its task_evaluation and models
installed.  Diagnostic source only; no pretrained weights shipped here.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,DynamicCache

ROOT=Path(__file__).resolve().parents[1]
TASK_ROOT=ROOT/'task_evaluation'
if str(TASK_ROOT) not in sys.path:sys.path.insert(0,str(TASK_ROOT))
from evaluate_ppl_kvmerge import (KVCompressingDynamicCache,_replace_dynamic_cache_inplace,
    apply_runtime_patch_if_needed,force_attn_hidden_and_compress)
from core_model_utils import first_execution_device,parse_dtype,load_token_stream
from journal_mis import route,hard_mis,count_mean,propagate,merge_all_baseline,node_weights
from sweep_config import build_configs
from continuous_model_support import prepare_model_for_new_sequence,load_frozen_calibration
from keepkv_split import select_split
from method_registry import METHODS, PRIORITY_MODES
from models.common.mymethod_common import recent_window_importance,reference_scale_from_original_keys
from ppl_log import emit_sequence_header,emit_sequence_result


def _save_csv(path,records):
    if not records:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    tmp.replace(path)


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
                     dropped_slots=int(stats.get('dropped_slots',0)),
                     geometry_over_delta=int(stats.get('geometry_over_delta',0)),
                     mis_conflict_edges=int(stats.get('mis_conflict_edges',0)),
                     mis_parallel_rounds=int(stats.get('mis_parallel_rounds',0)),
                     evicted_mass=float(stats.get('evicted_mass',0)),merged_mass=float(stats.get('merged_mass',0)),dropped_mass=float(stats.get('dropped_mass',0)),
                     routed_importance_mass=float(stats.get('routed_importance_mass',0)),
                     contributed_importance_mass=float(stats.get('contributed_importance_mass',0)),
                     importance_retention_rate=(stats['contributed_importance_mass']/stats['routed_importance_mass'] if stats.get('routed_importance_mass',0)>0 else ''),
                     routed_raw_importance=float(stats.get('routed_raw_importance',0)),
                     contributed_raw_importance=float(stats.get('contributed_raw_importance',0)),
                     raw_importance_retention_rate=(stats['contributed_raw_importance']/stats['routed_raw_importance'] if stats.get('routed_raw_importance',0)>0 else ''),
                     physical_kv_len=int(cache.key_cache[0].shape[-2]) if len(cache.key_cache) else 0)
            per_cycle.append(row)
            if on_cycle:on_cycle(row)
    if total_count!=P*S:raise AssertionError('Incomplete scored tokens')
    if hasattr(cache,'_ppl_logical_seen_tokens') and cache._ppl_logical_seen_tokens!=total_length:
        raise RuntimeError('Logical KV position did not advance over full text')
    return math.exp(total_nll/total_count),per_cycle


class ContinuousPCache(KVCompressingDynamicCache):
    """CountMean with independent EA/EE gates and certified R,D history."""
    def __init__(self,*,model,method,budget,lambda_ea,lambda_ee,reference,
                 unscored_len,scored_len,alpha_ea=1.,alpha_ee=1.,
                 reference_mode='frozen_train',score_window=128,priority_mode='degree',
                 sink_size=4,recent_protect_size=128,recency_beta=1.0,chunk_size=256,
                 stats=None):
        if not (0<float(budget)<1):raise ValueError('Budget ratio in (0,1)')
        self.budget_slots=max(1,round((int(unscored_len)-1)*float(budget)))
        super().__init__(model=model,method='mymethod',family='p_certificate',
            compress_ratio=1.-float(budget),sim_thr=-1.,method_sim_thrs=None,
            keepkv_config={},mymethod_config={},centroidkv_config={},
            min_kv_len=2,compress_every=1,stats=(stats if stats is not None else {}),
            task_eval_dir=TASK_ROOT,target_kv_len=self.budget_slots)
        if method not in METHODS:raise ValueError(f"Unexpected method {method}")
        if method=='full_cache':raise ValueError('Full Cache uses native DynamicCache')
        self.strategy=method
        if priority_mode not in PRIORITY_MODES: raise ValueError(f'Unknown priority mode: {priority_mode}')
        self.priority_mode=priority_mode
        self.alpha_ea=float(alpha_ea);self.alpha_ee=float(alpha_ee)
        if not (0<=self.alpha_ea<=1 and 0<=self.alpha_ee<=1):
            raise ValueError('History weights must be in [0,1]')
        self.lambda_ea=float(lambda_ea);self.lambda_ee=float(lambda_ee)
        self.reference=np.asarray(reference,dtype=np.float64) if reference is not None else None
        if method!='eviction_only' and reference_mode=='frozen_train' and self.reference is None:raise ValueError('Missing frozen reference')
        if reference_mode not in ('online_first','frozen_train'):
            raise ValueError('Unknown reference_mode')
        self.reference_mode=reference_mode
        self.online_ref=None
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
        self.compress_rounds=0
        self.counts=None;self.contributions=None;self.slot_positions=None
        self.geom_means=None;self.geom_radii=None;self.geom_diameters=None
        self._stats.setdefault('compress_rounds',0)
        self._stats.setdefault('routed_slots',0)
        self._stats.setdefault('accepted_slots',0)
        self._stats.setdefault('rejected_slots',0)
        self._stats.setdefault('dropped_slots',0)
        self._stats.setdefault('geometry_over_delta',0)
        for field in ('routed_importance_mass','contributed_importance_mass','routed_raw_importance','contributed_raw_importance','mis_conflict_edges',
                      'mis_parallel_rounds','evicted_mass','merged_mass','dropped_mass'):
            self._stats.setdefault(field,0)

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
        new_kv=[];new_count=[];new_contribution=[];new_positions=[]
        new_gm=[];new_gr=[];new_gd=[]
        online_ref=[]
        for layer,(KL,VL,score) in enumerate(zip(self.key_cache,self.value_cache,attentions)):
            if KL.shape[0]!=1 or int(KL.shape[-2])!=n:raise ValueError('Expected uniform batch-1 cache')
            H=int(KL.shape[1]);D=int(KL.shape[-1]);dev=KL.device
            if self.strategy=='eviction_only':
                scale=torch.ones(H,device=dev,dtype=torch.float64)
            elif self.reference_mode=='online_first':
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
            actual_counts=torch.cat((prev_counts,torch.ones((1,H,tail),device=dev)),dim=-1)
            if self.slot_positions is None:
                prevpos=torch.empty((1,H,0),device=dev,dtype=torch.long)
            else:prevpos=self.slot_positions[layer].to(dev)
            pos_new=torch.arange(logical-tail,logical,device=dev).view(1,1,-1).expand(1,H,-1)
            positions=torch.cat((prevpos,pos_new),dim=-1)
            if self.geom_means is None:
                gm_old=KL.new_empty((1,H,0,D),dtype=torch.float64)
                gr_old=KL.new_zeros((1,H,0),dtype=torch.float64)
                gd_old=gr_old
            else:
                gm_old=self.geom_means[layer].to(dev)
                gr_old=self.geom_radii[layer].to(dev)
                gd_old=self.geom_diameters[layer].to(dev)
            gm=torch.cat((gm_old,KL[:,:,prev:,:].double()),dim=-2)
            gr=torch.cat((gr_old,torch.zeros((1,H,tail),device=dev,dtype=torch.float64)),dim=-1)
            gd=torch.cat((gd_old,torch.zeros((1,H,tail),device=dev,dtype=torch.float64)),dim=-1)
            if not (gr.shape==gd.shape==actual_counts.shape==positions.shape==(1,H,n)):
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
            kout=[];vout=[];cout=[];aout=[];pout=[];gmean=[];grad=[];gdiam=[]
            for h in range(H):
                base_scale=float(scale[h].item())
                delta_ea=self.lambda_ea*base_scale
                delta_ee=self.lambda_ee*base_scale
                delta_cert=max(delta_ea,delta_ee)
                k=KL[0,h];v=VL[0,h]
                cnt=actual_counts[0,h];importance=imp[h]
                centers=gm[0,h];rads=gr[0,h];diams=gd[0,h]
                kk=keep[h];ee=evict[h]
                strategy=self.strategy; journal=strategy=='coherekv_journal'
                best=torch.zeros(len(ee),device=dev,dtype=torch.long)
                routed=torch.zeros(len(ee),device=dev,dtype=torch.bool)
                accepted=routed.clone();edges=rounds=0
                if journal:
                    best,routed,gamma=route(k,centers,rads,diams,kk,ee,delta_ea,self.alpha_ea,chunk_size=self.chunk_size)
                    accepted,edges,rounds=hard_mis(k,centers,rads,diams,ee,best,routed,gamma,delta_ee,self.alpha_ee,weights=(None if self.priority_mode=='degree' else node_weights(self.priority_mode,cnt,importance)),chunk_size=self.chunk_size)
                    kf,vf,cf=count_mean(k,v,cnt,kk,ee,best,accepted)
                elif strategy=='merge_all':
                    kf,vf,cf,best,routed=merge_all_baseline(k,v,cnt,kk,ee,delta_ea,chunk_size=self.chunk_size)
                    accepted=routed.clone()
                else:kf,vf,cf=k[kk].clone(),v[kk].clone(),cnt[kk].clone()
                if journal:
                    owners=[[] for _ in range(B)]
                    for local_id in torch.nonzero(accepted).flatten().tolist():
                        owners[int(best[local_id].item())].append(local_id)
                    Hm=[];Hr=[];Hd=[]
                    for ji,ki in enumerate(kk.tolist()):
                        ids=torch.cat((kk[ji:ji+1],ee[owners[ji]])) if owners[ji] else kk[ji:ji+1]
                        M_new,R_new,D_new=propagate(cnt[ids],centers[ids],rads[ids],diams[ids],
                                                   chunk_size=self.chunk_size)
                        Hm.append(M_new);Hr.append(R_new);Hd.append(D_new)
                    m_next=torch.stack(Hm);r_next=torch.stack(Hr);d_next=torch.stack(Hd)
                else:
                    m_next=kf.double();r_next=torch.zeros(B,device=dev,dtype=torch.float64);d_next=torch.zeros_like(r_next)
                bounded=journal and self.alpha_ea==self.alpha_ee==1
                over=int((d_next>delta_cert+max(1e-5,1e-5*delta_cert)).sum().item()) if journal else 0
                if bounded and over:raise AssertionError('Historical diameter exceeded max threshold')
                raw_imp=importance[ee].double()
                q=cnt[ee].double()*raw_imp
                raw_routed=float(raw_imp[routed].sum().item());raw_merged=float(raw_imp[accepted].sum().item())
                q_routed=float(q[routed].sum().item());q_merged=float(q[accepted].sum().item())
                evicted_mass=float(cnt[ee].sum().item());merged_mass=float(cnt[ee[accepted]].sum().item())
                head=dict(round=self.compress_rounds+1,layer_idx=layer,kv_head_idx=h,
                    before_slots=n,after_slots=B,evicted_slots=len(ee),routed_slots=int(routed.sum()),
                    accepted_slots=int(accepted.sum()),rejected_slots=int((routed&~accepted).sum()),
                    dropped_slots=int((~accepted).sum()),evicted_mass=evicted_mass,merged_mass=merged_mass,
                    dropped_mass=evicted_mass-merged_mass,routed_importance_mass=q_routed,
                    contributed_importance_mass=q_merged,routed_raw_importance=raw_routed,
                    contributed_raw_importance=raw_merged,
                    importance_retention_rate=q_merged/q_routed if q_routed else '',
                    raw_importance_retention_rate=raw_merged/raw_routed if raw_routed else '',mis_conflict_edges=edges,mis_parallel_rounds=rounds,
                    geometry_over_delta=over,history_certified=journal,threshold_guaranteed=bounded,history_tracked=journal,
                    reference_scale=base_scale if strategy!='eviction_only' else '',delta_ea=delta_ea if strategy!='eviction_only' else '',
                    delta_ee=delta_ee if journal else '',delta_cert=delta_cert if journal else '',
                    max_certified_radius=float(r_next.max()) if journal else '',
                    mean_certified_radius=float(r_next.mean()) if journal else '',
                    max_certified_diameter=float(d_next.max()) if journal else '',
                    mean_certified_diameter=float(d_next.mean()) if journal else '',
                    represented_count=float(cf.sum()),merge_rate=float(accepted.sum())/len(ee) if len(ee) else 0.,
                    assignment_rate=float(routed.sum())/len(ee) if len(ee) else 0.,
                    mis_rejection_rate=float((routed&~accepted).sum())/int(routed.sum()) if routed.any() else 0.,
                    represented_mass_merge_rate=merged_mass/evicted_mass if evicted_mass else 0.)
                for field in ('routed_slots','accepted_slots','rejected_slots','dropped_slots','evicted_mass',
                              'merged_mass','dropped_mass','routed_importance_mass','contributed_importance_mass','routed_raw_importance','contributed_raw_importance',
                              'mis_conflict_edges','mis_parallel_rounds','geometry_over_delta'):
                    self._stats[field]+=head[field]
                self._stats.setdefault('head_rounds',[]).append(head)
                kout.append(kf);vout.append(vf);cout.append(cf);aout.append(cf);pout.append(positions[0,h,kk])
                gmean.append(m_next);grad.append(r_next);gdiam.append(d_next)
            new_kv.append((torch.stack(kout,0)[None],torch.stack(vout,0)[None]))
            new_count.append(torch.stack(cout,0)[None]);new_contribution.append(torch.stack(aout,0)[None]);new_positions.append(torch.stack(pout,0)[None])
            new_gm.append(torch.stack(gmean,0)[None])
            new_gr.append(torch.stack(grad,0)[None]);new_gd.append(torch.stack(gdiam,0)[None])
        _replace_dynamic_cache_inplace(self,new_kv)
        self.counts=new_count;self.contributions=new_contribution;self.slot_positions=new_positions
        self.geom_means=new_gm
        self.geom_radii=new_gr;self.geom_diameters=new_gd
        if self.strategy!='eviction_only' and self.reference_mode=='online_first' and self.online_ref is None:
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
            attn._mymethod_counts_kv=new_contribution[li]
            attn._mymethod_eager_enabled=True
            attn._mymethod_sdpa_enabled=False
            attn._keepkv_eager_enabled=False
            attn._mymethod_selector='keepkv'
            attn._mymethod_score_window=self.score_window
            attn._kvmerge_attention_chunk_size=self.chunk_size


def args_parser():
    p=argparse.ArgumentParser(description='Multi-round continuous CohereKV/MIS PPL')
    p.add_argument('--model',required=True)
    p.add_argument('--calibration')
    p.add_argument('--methods',default=','.join(METHODS),help='Comma-separated enabled method names')
    p.add_argument('--output_dir',required=True)
    p.add_argument('--dataset',default='wikitext2',choices=['wikitext2'])
    p.add_argument('--split',default='validation',choices=['validation','test'])
    p.add_argument('--sequence_index',type=int,default=0,
                   help='Non-overlapping Token window index, window length=cycles*(U+S)')
    p.add_argument('--cycles',type=int,default=4)
    p.add_argument('--unscored_len',type=int,default=16384)
    p.add_argument('--scored_len',type=int,default=256)
    p.add_argument('--alpha_ea',default='0')
    p.add_argument('--alpha_ee',default='1')
    p.add_argument('--priority_modes',default=','.join(PRIORITY_MODES))
    p.add_argument('--lambda_ea',default='0.5')
    p.add_argument('--lambda_ee',default='0.5')
    p.add_argument('--budgets',default='0.1')
    p.add_argument('--reference_modes','--reference_mode',dest='reference_modes',default='frozen_train',help='Comma-separated frozen_train,online_first')
    p.add_argument('--score_window',type=int,default=128)
    p.add_argument('--sink_size',type=int,default=4)
    p.add_argument('--recent_protect_size',type=int,default=128)
    p.add_argument('--recency_beta',type=float,default=1.0)
    p.add_argument('--prefill_chunk_size',type=int,default=128)
    p.add_argument('--distance_chunk_size',type=int,default=256)
    p.add_argument('--torch_dtype',default='bfloat16')
    p.add_argument('--attn_implementation',default='eager')
    p.add_argument('--device',default='cuda')
    p.add_argument('--device_map',default='auto')
    p.add_argument('--trust_remote_code',action='store_true')
    p.add_argument('--use_fast_tokenizer',action='store_true')
    return p


def main():
    a=args_parser().parse_args()
    if a.sequence_index<0:
        raise ValueError('sequence_index must be nonnegative')
    if a.cycles<1 or a.unscored_len<2 or a.scored_len<1 or a.score_window<1 or a.prefill_chunk_size<a.score_window:
        raise ValueError('Invalid cycles/length/score window/chunk size')
    if a.sink_size<0 or a.recent_protect_size<0 or not(math.isfinite(a.recency_beta) and a.recency_beta>=1.0):
        raise ValueError('sink/recent protection must be nonnegative; recency_beta >=1 finite')
    if a.device!='cuda' and a.torch_dtype in ('float16','bfloat16'):
        raise ValueError('Use float32 for CPU')
    configs=build_configs(a)
    methods=tuple(dict.fromkeys(c['method'] for c in configs))
    total_length=a.cycles*(a.unscored_len+a.scored_len)
    cfg=AutoConfig.from_pretrained(a.model,trust_remote_code=a.trust_remote_code)
    model_type=str(cfg.model_type).lower()
    family={'qwen2':'qwen','llama':'llama'}.get(model_type)
    if family is None:raise ValueError('Only Qwen2/Llama patch is supported')
    max_ctx=getattr(cfg,'max_position_embeddings',None)
    if max_ctx is not None and total_length>int(max_ctx):
        raise ValueError(f'Continuous {total_length} logical tokens exceeds model max_position_embeddings={max_ctx}. '
                         'Reduce cycles/length or select a model with a validated longer-context configuration; '
                         'never wrap/reset RoPE position IDs')
    reference=None
    needs_calibration=any(c['reference_mode']=='frozen_train' for c in configs)
    if needs_calibration:
        if not a.calibration:raise ValueError("Frozen train mode requires calibration")
        info,reference=load_frozen_calibration(a.calibration,a.model,a.dataset)
        if info.get('torch_dtype',a.torch_dtype)!=a.torch_dtype or info.get('attn_implementation',a.attn_implementation)!=a.attn_implementation:
            raise ValueError("Calibration dtype/attention backend mismatch")
        if info.get('use_fast_tokenizer',a.use_fast_tokenizer)!=a.use_fast_tokenizer:
            raise ValueError("Calibration tokenizer mismatch")
        if reference.shape!=(cfg.num_hidden_layers,cfg.num_key_value_heads):
            raise ValueError('Frozen train reference shape does not match current model')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'continuous_run_complete.json').unlink(missing_ok=True)
    (out/'ppl_continuous_head_rounds.csv').unlink(missing_ok=True)
    signature=dict(vars(a),total_logical_tokens=total_length,methods=methods,
                   reference_sha256=hashlib.sha256(Path(a.calibration).read_bytes()).hexdigest() if needs_calibration else None,
                   implementation='journal_weighted_mis_v4',
                   configurations=configs,
                   token_start=a.sequence_index*total_length,
                   token_end=(a.sequence_index+1)*total_length)
    (out/'continuous_config.json').write_text(json.dumps(signature,indent=2),encoding='utf8')
    tokenizer=AutoTokenizer.from_pretrained(a.model,use_fast=a.use_fast_tokenizer,
        trust_remote_code=a.trust_remote_code)
    token_start=a.sequence_index*total_length
    token_end=token_start+total_length
    ids=load_token_stream(tokenizer,a.dataset,split=a.split)
    if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[-1]<token_end:
        raise ValueError(f'Insufficient contiguous split tokens through {token_end}; '
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
    emit_sequence_header()
    all_results=[];all_cycles=[];patched=False
    for setting in configs:
        method=setting['method'];stats={};t0=time.time()
        prepare_model_for_new_sequence(model,a.score_window,a.prefill_chunk_size)
        kwargs_eval=dict(prefill_chunk_size=a.prefill_chunk_size,score_window=a.score_window,
            amp_dtype=parse_dtype(a.torch_dtype) if device.type=='cuda' else None)
        print(f'[CONTINUOUS] {setting}',flush=True)
        if method=='full_cache':
            ppl,cycles=evaluate_continuous(model,ids,a.cycles,a.unscored_len,a.scored_len,device,**kwargs_eval)
            if any(r['physical_kv_len']!=r['logical_tokens'] or r['compression_rounds']!=0 for r in cycles):
                raise RuntimeError('Native Full Cache did not retain all positions')
            stats=dict(max_before_len=total_length,max_after_len=total_length,compress_rounds=0)
        else:
            if not patched:apply_runtime_patch_if_needed(family,TASK_ROOT,'mymethod');patched=True
            def factory():
                return ContinuousPCache(model=model,method=method,budget=setting['budget'],
                    alpha_ea=setting['alpha_ea'] if setting['alpha_ea']!='' else 1.,
                    alpha_ee=setting['alpha_ee'] if setting['alpha_ee']!='' else 1.,
                    lambda_ea=setting['lambda_ea'] or .5,lambda_ee=setting['lambda_ee'] or .5,
                    reference=reference,reference_mode=setting['reference_mode'] or 'online_first',
                    priority_mode=setting['priority_mode'] or 'degree',
                    unscored_len=a.unscored_len,scored_len=a.scored_len,score_window=a.score_window,
                    sink_size=a.sink_size,recent_protect_size=a.recent_protect_size,
                    recency_beta=a.recency_beta,chunk_size=a.distance_chunk_size,stats=stats)
            with force_attn_hidden_and_compress(model):
                ppl,cycles=evaluate_continuous(model,ids,a.cycles,a.unscored_len,a.scored_len,device,
                    cache_factory=factory,**kwargs_eval)
            if stats.get('compress_rounds')!=a.cycles:raise RuntimeError('Wrong compression count')
        common=dict(**setting,model=a.model,dataset=a.dataset,split=a.split,sequence_index=a.sequence_index,
                    token_start=token_start,token_end=token_end,stream_sha256=signature['scored_stream_sha256'],
                    unscored_len=a.unscored_len,scored_len=a.scored_len,cycles=a.cycles)
        for record in cycles:all_cycles.append({**common,**record})
        heads=[{**common,**record} for record in stats.get('head_rounds',[])]
        if heads:
            path=out/'ppl_continuous_head_rounds.csv';first=not path.exists()
            with path.open('a',encoding='utf8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(heads[0]));
                if first:w.writeheader()
                w.writerows(heads)
        result=dict(**common,scored_tokens=a.cycles*a.scored_len,ppl=ppl,
                    nll_sum=sum(r['cycle_nll_sum'] for r in cycles),nll_per_token=math.log(ppl),
                    seconds=time.time()-t0)
        for field in ('compress_rounds','routed_slots','accepted_slots','rejected_slots','dropped_slots',
                      'mis_conflict_edges','mis_parallel_rounds','geometry_over_delta','evicted_mass','merged_mass',
                      'dropped_mass','routed_importance_mass','contributed_importance_mass','routed_raw_importance','contributed_raw_importance','max_before_len','max_after_len'):
            result[field]=stats.get(field,0)
        result['importance_retention_rate']=(result['contributed_importance_mass']/result['routed_importance_mass'] if result['routed_importance_mass'] else '')
        result['raw_importance_retention_rate']=(result['contributed_raw_importance']/result['routed_raw_importance'] if result['routed_raw_importance'] else '')
        all_results.append(result)
        _save_csv(out/'ppl_continuous.csv',all_results)
        _save_csv(out/'ppl_continuous_cycles.csv',all_cycles)
        emit_sequence_result(result)
    (out/'continuous_run_complete.json').write_text(json.dumps(dict(config=signature,
        n_results=len(all_results),n_cycles=len(all_cycles)),indent=2),encoding='utf8')


if __name__=='__main__':main()
