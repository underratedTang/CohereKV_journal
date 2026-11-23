#!/usr/bin/env python3
"""Read-only physical fusion diagnostics for one KV head.

This module NEVER changes routing, MIS, K/V, geometry certificates, or Counts.
Pair rows are deterministic top-k diagnostics, NOT an unbiased sample.
Head/round summaries cover ALL accepted pair events and changed anchors.
"""
from __future__ import annotations
import math
import torch


def _summary(x):
    if not x.numel():return 0.,0.,0.,0.
    z=x.detach().double()
    if not bool(torch.isfinite(z).all()):raise RuntimeError('Nonfinite physical fusion diagnostic')
    return (float(z.mean()),float(torch.quantile(z,.5)),
            float(torch.quantile(z,.9)),float(z.max()))


def _top_ids(score,k):
    """Sorted top-k deterministic indices; aggregate stats use all events."""
    n=int(score.numel())
    if not n or k<=0:return []
    values=score.detach().cpu().tolist()
    return sorted(range(n),key=lambda i:(-float(values[i]),i))[:min(k,n)]


@torch.no_grad()
def audit_fusion_head(keys,values,counts,importance,keep,evict,best,accepted,
                      gamma,merged_keys,merged_values,delta,*,topk=4):
    """Actual fusion statistics, collected *after* CountMean on one KV head.

    `counts` are attention/fusion multiplicities, not oracle geometric certificates.
    `best` is anchor-relative; gamma is the selected E-A history cross bound.
    Key distance and changed-anchor Key shift are divided by the geometry ΔEA.
    Value drift is relative to original anchor Value norm, with a unit floor.
    Importance is the current pre-fusion KeepKV score I (without log Count).
    """
    if not (math.isfinite(float(delta)) and float(delta)>0):
        raise ValueError('Positive physical diagnostic delta required')
    if keys.ndim!=2 or values.shape!=keys.shape or merged_keys.shape!=(keep.numel(),keys.shape[-1]):
        raise ValueError('Unexpected physical fusion shapes')
    if merged_values.shape!=merged_keys.shape:
        raise ValueError('Unexpected fused Value shape')
    src=torch.nonzero(accepted.bool().reshape(-1)).flatten()
    keep=keep.long().reshape(-1);evict=evict.long().reshape(-1)
    best=best.long().reshape(-1);gamma=gamma.double().reshape(-1)
    c=counts.double().reshape(-1);imp=importance.double().reshape(-1)
    stats={'fusion_accepted_pairs':int(src.numel()),'fusion_touched_anchors':0,
           'fusion_historical_pairs':0,'fusion_countaware_excess_pairs':0,
           'fusion_physical_over_delta_pairs':0}
    pair_samples=[];anchor_samples=[]
    for key in ('physical_dist_over_delta','pair_count_ratio','pair_importance_ratio',
                'pair_value_dist','pair_anchor_count_share'):
        for stat in ('mean','median','p90','max'):
            stats[f'fusion_{key}_{stat}']=0.
    for key in ('anchor_key_shift_over_delta','anchor_value_shift_relative',
                'anchor_incoming_count_ratio','anchor_incoming_countaware_excess'):
        for stat in ('mean','median','p90','max'):
            stats[f'fusion_{key}_{stat}']=0.
    if src.numel()==0:
        return stats,pair_samples,anchor_samples

    eidx=evict[src];a_pos=best[src];aidx=keep[a_pos]
    de=(keys[eidx].double()-keys[aidx].double()).norm(dim=-1)
    dv=(values[eidx].double()-values[aidx].double()).norm(dim=-1)
    phys=de/float(delta)
    counts_e=c[eidx];counts_a=c[aidx]
    ratio=counts_e/counts_a
    fraction=counts_e/(counts_e+counts_a)
    importance_e=imp[eidx];importance_a=imp[aidx]
    # Avoid treating zero anchor importance as a finite meaningful ratio.
    imp_ratio=torch.where(importance_a>1e-30,importance_e/importance_a.clamp_min(1e-30),
                          torch.zeros_like(importance_e))
    excess=gamma[src]>float(delta)+2e-9*max(1.,float(delta))
    stats.update(fusion_historical_pairs=int((counts_e>1).sum()),
        fusion_countaware_excess_pairs=int(excess.sum()),
        fusion_physical_over_delta_pairs=int((phys>1.+2e-9).sum()))
    pair_fields=dict(physical_dist_over_delta=phys,pair_count_ratio=ratio,
                     pair_importance_ratio=imp_ratio,pair_value_dist=dv,
                     pair_anchor_count_share=fraction)
    for name,z in pair_fields.items():
        v=_summary(z)
        for key,val in zip(('mean','median','p90','max'),v):
            stats[f'fusion_{name}_{key}']=val

    touched=a_pos.unique(sorted=True)
    stats['fusion_touched_anchors']=int(len(touched))
    keyshift=(merged_keys[touched].double()-keys[keep[touched]].double()).norm(dim=-1)/float(delta)
    original_val_norm=values[keep[touched]].double().norm(dim=-1).clamp_min(1.)
    valshift=(merged_values[touched].double()-values[keep[touched]].double()).norm(dim=-1)/original_val_norm
    in_c=torch.zeros(keep.numel(),device=keys.device,dtype=torch.float64)
    in_c.index_add_(0,a_pos,counts_e)
    in_excess=torch.zeros(keep.numel(),device=keys.device,dtype=torch.float64)
    in_excess.index_add_(0,a_pos,excess.double())
    incoming=in_c[touched]/c[keep[touched]]
    anchor_fields=dict(anchor_key_shift_over_delta=keyshift,
                       anchor_value_shift_relative=valshift,
                       anchor_incoming_count_ratio=incoming,
                       anchor_incoming_countaware_excess=in_excess[touched])
    for name,z in anchor_fields.items():
        v=_summary(z)
        for key,val in zip(('mean','median','p90','max'),v):
            stats[f'fusion_{name}_{key}']=val

    # deterministic TOP-K events for interpretability. Do not use these samples
    # to compute reported population rates or confidence intervals.
    selected={}
    for label,z in [('largest_physical_distance',phys),('largest_source_count',ratio),
                    ('largest_importance_ratio',imp_ratio),('largest_cross_excess',gamma[src])]:
        for rank,j in enumerate(_top_ids(z,topk),1):
            selected.setdefault(j,[]).append(f'{label}:{rank}')
    for j,labels in sorted(selected.items()):
        pair_samples.append(dict(sample_reason=';'.join(labels),
          evicted_slot=int(eidx[j]),anchor_slot=int(aidx[j]),
          anchor_relative_slot=int(a_pos[j]),
          physical_key_distance_over_delta=float(phys[j]),
          physical_value_distance=float(dv[j]),
          history_cross_bound_over_delta=float(gamma[src[j]]/float(delta)),
          cross_bound_exceeds_delta=bool(excess[j]),
          source_fusion_count=float(counts_e[j]),anchor_fusion_count=float(counts_a[j]),
          count_ratio=float(ratio[j]),source_historical=bool(counts_e[j]>1),
          source_importance=float(importance_e[j]),anchor_importance=float(importance_a[j]),
          importance_ratio=(float(importance_e[j]/importance_a[j])
                            if float(importance_a[j])>1e-30 else 'undefined_zero_anchor')))

    selected_anchor={}
    for label,z in [('largest_key_shift',keyshift),('largest_value_shift',valshift),
                    ('largest_incoming_count',incoming)]:
        for rank,j in enumerate(_top_ids(z,topk),1):
            selected_anchor.setdefault(j,[]).append(f'{label}:{rank}')
    for j,labels in sorted(selected_anchor.items()):
        apos=int(touched[j]);num=int((a_pos==apos).sum())
        anchor_samples.append(dict(sample_reason=';'.join(labels),
          anchor_relative_slot=apos,anchor_slot=int(keep[apos]),
          merged_evicted_states=num,
          anchor_before_count=float(c[keep[apos]]),
          incoming_evicted_count=float(in_c[apos]),
          incoming_to_anchor_count_ratio=float(incoming[j]),
          incoming_cross_excess_pairs=int(in_excess[apos]),
          anchor_importance=float(imp[keep[apos]]),
          fused_key_shift_over_delta=float(keyshift[j]),
          fused_value_shift_relative=float(valshift[j])))
    return stats,pair_samples,anchor_samples
