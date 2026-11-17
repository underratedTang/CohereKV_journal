#!/usr/bin/env python3
"""Single-round degree-first or importance-weighted K-only MIS replay.

One physical original token has C=1 in this diagnostic. K/V mean for both
MIS-Drop variants is *identical*; only the log(C) attention bias differs.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from method_registry import FUSION_METHODS,FAMILY_METHODS,fusion_is_importance_weighted
METHODS=FUSION_METHODS

@dataclass
class Fusion:
    keys: dict
    values: dict
    counts: dict
    n_routed: int
    n_mis_accepted: int
    n_mis_rejected: int
    group_rows: list
    rejected_rows: list
    sse: dict
    v_sse: dict


def fuse_plan(keys,values,keep_idx,evict_idx,best_idx,routed,accepted,
              pool_sizes=None,selected_sizes=None,conflict_edges=None,mis_family="k_only",importance=None,
              methods_to_compute=None):
    if mis_family not in FAMILY_METHODS:raise ValueError('Unknown MIS family')
    drop_method,allcount_method=FAMILY_METHODS[mis_family]
    possible=('merge_all','importance_weighted_merge_all',drop_method,allcount_method)
    family_methods=(possible if methods_to_compute is None else
                    tuple(x for x in possible if x in methods_to_compute))
    if 'merge_all' not in family_methods or not any(m in family_methods for m in (drop_method,allcount_method)):
        raise ValueError('Need Merge All and at least one active native MIS arm')
    keys=np.asarray(keys,dtype=np.float32);values=np.asarray(values,dtype=np.float32)
    keep=np.asarray(keep_idx,dtype=np.int64);evict=np.asarray(evict_idx,dtype=np.int64)
    best=np.asarray(best_idx,dtype=np.int64);routed=np.asarray(routed,dtype=bool)
    acc=np.asarray(accepted,dtype=bool)
    imp=np.asarray(importance,dtype=np.float64).reshape(-1) if importance is not None else np.zeros(len(keys))
    if len(imp)!=len(keys) or not np.isfinite(imp).all():raise ValueError('Bad KeepKV importance')
    if keys.ndim!=2 or values.shape!=keys.shape:raise ValueError('K/V shape mismatch')
    if len(keep)+len(evict)!=len(keys):raise ValueError('Partition mismatch')
    if any(len(x)!=len(evict) for x in (best,routed,acc)):
        raise ValueError('Bad routing/MIS array length')
    if np.any(acc&~routed):raise ValueError('MIS accepted outside routed pool')
    if np.any(best[routed]<0) or np.any(best[routed]>=len(keep)):
        raise ValueError('Bad routed anchor')
    k64=keys.astype(np.float64);v64=values.astype(np.float64)
    targets=np.where(routed)[0]
    included=np.where(routed&acc)[0]
    rejected=np.where(routed&~acc)[0]
    if np.any(imp<0):raise ValueError('Importance must be nonnegative')

    def make(src,iw,allcount):
        # Single-round Replay: C=1, hence W=C*I=I. When all observed
        # importance values in a group are zero, use count mean fallback.
        mass=imp if iw else np.ones(len(keys),dtype=np.float64)
        denominator=mass[keep].copy()
        knum=k64[keep]*denominator[:,None]
        vnum=v64[keep]*denominator[:,None]
        count=np.ones(len(keep),dtype=np.float64)
        np.add.at(denominator,best[src],mass[evict[src]])
        np.add.at(knum,best[src],k64[evict[src]]*mass[evict[src],None])
        np.add.at(vnum,best[src],v64[evict[src]]*mass[evict[src],None])
        np.add.at(count,best[src],1.)
        if iw:
            for anchor in np.flatnonzero(denominator<=0):
                a_src=src[best[src]==anchor]
                pts=np.r_[keep[anchor],evict[a_src]]
                denominator[anchor]=len(pts)
                knum[anchor]=np.sum(k64[pts],axis=0)
                vnum[anchor]=np.sum(v64[pts],axis=0)
        if allcount:
            np.add.at(count,best[rejected],1.)
        return (knum/denominator[:,None]).astype(np.float32), (vnum/denominator[:,None]).astype(np.float32),count.astype(np.float32)

    outk={};outv={};counts={}
    for method in family_methods:
        src=targets if method in ('merge_all','importance_weighted_merge_all') else included
        outk[method],outv[method],counts[method]=make(src,fusion_is_importance_weighted(method),method.endswith('_allcount'))
    if drop_method in outk and allcount_method in outk:
        if not np.array_equal(outk[drop_method],outk[allcount_method]):
            raise AssertionError('Drop and AllCount must have identical K')
        if not np.array_equal(outv[drop_method],outv[allcount_method]):
            raise AssertionError('Drop and AllCount must have identical V')
    k_all=outk['merge_all'];v_all=outv['merge_all']
    active_mis=(drop_method if drop_method in outk else allcount_method)
    k_drop=outk[active_mis];v_drop=outv[active_mis]
    group=[];reject_rows=[]
    sse={arm:0. for arm in family_methods};v_sse={arm:0. for arm in family_methods}
    pools=np.asarray(pool_sizes).reshape(-1) if pool_sizes is not None else None
    selected=np.asarray(selected_sizes).reshape(-1) if selected_sizes is not None else None
    edges=np.asarray(conflict_edges).reshape(-1) if conflict_edges is not None else None
    for anchor,idx in enumerate(keep):
        pos=targets[best[targets]==anchor]
        na=int(acc[pos].sum()); nr=int(len(pos)-na)
        if pools is not None and int(pools[anchor])!=len(pos):raise AssertionError('Pool mismatch')
        if selected is not None and int(selected[anchor])!=na:raise AssertionError('Selected mismatch')
        group_ids=np.r_[idx,evict[pos]]
        for arm in family_methods:
            sse[arm]+=float(((k64[group_ids]-outk[arm][anchor])**2).sum())
            v_sse[arm]+=float(((v64[group_ids]-outv[arm][anchor])**2).sum())
        if pos.size:
            rej=pos[~acc[pos]]
            d=np.linalg.norm(k64[evict[pos]]-k64[idx],axis=-1)
            rd=np.linalg.norm(k64[evict[rej]]-k64[idx],axis=-1)
            group.append(dict(anchor_idx=int(idx),n_routed=len(pos),n_mis_accepted=na,
                n_mis_rejected=nr,n_importance_mis_accepted=(na if mis_family!="k_only" else 0),
                n_importance_mis_rejected=(nr if mis_family!="k_only" else 0),
                routed_importance_sum=float(imp[evict[pos]].sum()),
                accepted_importance_sum=float(imp[evict[pos[acc[pos]]]].sum()),
                rejected_importance_sum=float(imp[evict[pos[~acc[pos]]]].sum()),
                accepted_importance_fraction=float(imp[evict[pos[acc[pos]]]].sum()/max(1e-12,float(imp[evict[pos]].sum()))),
                group_count=int(len(pos)+1),drop_count=int(na+1),
                reject_count_mass=nr,mis_reject_count_fraction=nr/max(1,len(pos)+1),
                native_conflict_edges=(int(edges[anchor]) if edges is not None else 0),
                mean_re_distance=float(d.mean()),
                mean_rejected_re_distance=(float(rd.mean()) if rd.size else 0.),
                merge_all_anchor_drift=float(np.linalg.norm(outk['merge_all'][anchor]-k64[idx])),
                mis_drop_anchor_drift=float(np.linalg.norm(k_drop[anchor]-k64[idx])),
                merge_all_to_mis_shift=float(np.linalg.norm(k_all[anchor]-k_drop[anchor])),
                merge_all_to_mis_value_shift=float(np.linalg.norm(v_all[anchor]-v_drop[anchor])),
                accepted_anchor_key_drift=float(np.linalg.norm(k_drop[anchor]-k64[idx])),
                accepted_importance_mass=float(imp[idx]+imp[evict[pos[acc[pos]]]].sum()),
                importance_weighted_key_sse=float(np.sum(
                    imp[np.r_[idx,evict[pos[acc[pos]]]],None] *
                    (k64[np.r_[idx,evict[pos[acc[pos]]]]]-k_drop[anchor])**2)),
                accepted_ea_distance_max=float(d[acc[pos]].max()) if na else 0.,
                accepted_ea_distance_mean=float(d[acc[pos]].mean()) if na else 0.))
            for relative in rej:
                reject_rows.append(dict(anchor_idx=int(idx),rejected_evicted_idx=int(evict[relative]),
                    original_re_distance=float(np.linalg.norm(k64[evict[relative]]-k64[idx])),
                    original_re_value_distance=float(np.linalg.norm(v64[evict[relative]]-v64[idx])),
                    n_routed_group=len(pos),n_selected_group=na,
                    rejected_count_mass=1.0,
                    rejected_importance=float(imp[evict[relative]])))
    return Fusion(outk,outv,counts,len(targets),len(included),len(rejected),
                  group,reject_rows,sse,v_sse)
