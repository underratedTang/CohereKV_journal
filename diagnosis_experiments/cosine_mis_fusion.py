#!/usr/bin/env python3
"""Pure, query-free, one-round (C=1, R=D=0) MIS-fusion triad.

All three variants operate on precisely the same R-E routing plan:
  merge_all         original routed evicted K; all original V and original count
  mis_to_anchor     MIS-rejected routed evicted K replaced by anchor K
  all_to_anchor     every routed evicted K replaced by anchor K
The original routed evicted V and integer count are identical in every arm.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

METHODS = ('merge_all', 'mis_to_anchor', 'all_to_anchor')

@dataclass
class Triad:
    keys: dict
    values: np.ndarray
    counts: np.ndarray
    n_routed: int
    n_mis_accepted: int
    n_mis_rejected: int
    group_rows: list
    rejected_rows: list
    sse: dict


def fuse_plan(keys, values, keep_idx, evict_idx, best_idx, routed, mis_selected,
              pool_sizes=None, selected_sizes=None, conflict_edges=None):
    """Fuse original per-token float32 K/V with fixed R-E routing and native E-E MIS.

    Inputs are local KV-head arrays, *not* projected 2D geometry.
    The selected routing mode fixes group membership; CohereKV MIS supplies selection.
    """
    keys=np.asarray(keys,dtype=np.float32)
    values=np.asarray(values,dtype=np.float32)
    keep=np.asarray(keep_idx,dtype=np.int64).ravel()
    evict=np.asarray(evict_idx,dtype=np.int64).ravel()
    anchors=np.asarray(best_idx,dtype=np.int64).ravel()
    routed=np.asarray(routed,dtype=bool).ravel()
    accepted=np.asarray(mis_selected,dtype=bool).ravel()
    if keys.ndim!=2 or values.ndim!=2 or keys.shape[0]!=values.shape[0]:
        raise ValueError('Expected per-prefix K/V arrays with matching sequence length')
    if len(keep)+len(evict)!=len(keys) or len(set(keep.tolist()+evict.tolist()))!=len(keys):
        raise ValueError('Retained/Evicted partition must cover original prefix exactly')
    if len(anchors)!=len(evict) or len(routed)!=len(evict) or len(accepted)!=len(evict):
        raise ValueError('Plan array lengths are inconsistent')
    if np.any(accepted & ~routed):
        raise AssertionError('E-E MIS must only select already routed candidates')
    if np.any(anchors[routed]<0) or np.any(anchors[routed]>=len(keep)):
        raise ValueError('Invalid anchor index on compatible route')
    m=len(keep)
    mapped=np.flatnonzero(routed)
    routed_anchors=anchors[mapped]
    rejected=routed & ~accepted
    nacc=int(accepted.sum());nrej=int(rejected.sum())
    # Use original float64 arithmetic for all methods, then cast to output FP32.
    kval=keys.astype(np.float64)
    vval=values.astype(np.float64)
    baseline_sum=kval[keep].copy()
    value_sum=vval[keep].copy()
    selected_sum=kval[keep].copy()
    counts=np.ones(m,dtype=np.int64)
    if len(mapped):
        np.add.at(baseline_sum,routed_anchors,kval[evict[mapped]])
        np.add.at(value_sum,routed_anchors,vval[evict[mapped]])
        np.add.at(counts,routed_anchors,1)
        selected_pos=mapped[accepted[mapped]]
        rejected_pos=mapped[rejected[mapped]]
        if len(selected_pos):
            np.add.at(selected_sum,anchors[selected_pos],kval[evict[selected_pos]])
        if len(rejected_pos):
            np.add.at(selected_sum,anchors[rejected_pos],kval[keep[anchors[rejected_pos]]])
    mk=(baseline_sum/counts[:,None]).astype(np.float32)
    ck=(selected_sum/counts[:,None]).astype(np.float32)
    ak=keys[keep].copy()  # all_to_anchor: (1 + #routed) copies of the same anchor.
    mv=(value_sum/counts[:,None]).astype(np.float32)
    cmap={'merge_all':mk,'mis_to_anchor':ck,'all_to_anchor':ak}
    stats=[];outs=[];sse={name:0. for name in METHODS}
    pool=np.asarray(pool_sizes).ravel() if pool_sizes is not None else None
    sel=np.asarray(selected_sizes).ravel() if selected_sizes is not None else None
    edge=np.asarray(conflict_edges).ravel() if conflict_edges is not None else None
    for a,token in enumerate(keep):
        local=mapped[routed_anchors==a]
        ids=np.r_[token,evict[local]]
        n_local=int(len(local))
        n_selected=int(accepted[local].sum())
        n_rej=n_local-n_selected
        if pool is not None and int(pool[a])!=n_local:
            raise AssertionError('E-E MIS pool count differs from routed members')
        if sel is not None and int(sel[a])!=n_selected:
            raise AssertionError('E-E MIS selected count differs from selected mask')
        gk=kval[ids]
        distances=np.linalg.norm(gk[1:]-gk[0],axis=-1) if n_local else np.empty(0)
        dcore=np.linalg.norm(gk[1:][accepted[local]]-gk[0],axis=-1) if n_selected else np.empty(0)
        drej=np.linalg.norm(gk[1:][~accepted[local]]-gk[0],axis=-1) if n_rej else np.empty(0)
        for name,avg in cmap.items():
            sse[name]+=float(np.sum((gk-avg[a].astype(np.float64))**2))
        if n_local:
            stats.append(dict(anchor_idx=int(token),n_routed=n_local,n_mis_accepted=n_selected,
                n_mis_rejected=n_rej,group_count=n_local+1,
                native_conflict_edges=int(edge[a]) if edge is not None else 0,
                mean_re_distance=float(distances.mean()),
                mean_selected_re_distance=float(dcore.mean()) if n_selected else '',
                mean_rejected_re_distance=float(drej.mean()) if n_rej else '',
                merge_all_anchor_drift=float(np.linalg.norm(mk[a].astype(np.float64)-gk[0])),
                mis_to_anchor_anchor_drift=float(np.linalg.norm(ck[a].astype(np.float64)-gk[0])),
                all_to_anchor_anchor_drift=0.,
                merge_all_to_mis_shift=float(np.linalg.norm(mk[a].astype(np.float64)-ck[a].astype(np.float64)))))
            for pos in local[~accepted[local]]:
                outs.append(dict(anchor_idx=int(token),rejected_evicted_idx=int(evict[pos]),
                    original_re_distance=float(np.linalg.norm(kval[evict[pos]]-kval[token])),
                    n_routed_group=n_local,n_selected_group=n_selected))
    assert int(counts.sum())==len(keep)+len(mapped)
    assert nacc+nrej==len(mapped)
    assert np.array_equal(ak,keys[keep])
    return Triad(cmap,mv,counts.astype(np.float32),len(mapped),nacc,nrej,stats,outs,sse)
