#!/usr/bin/env python3
"""Physical-Key E-A / E-E diagnostic arms, with no historical certificate.

All arms using the same E-A gate ratio route from identical physical
nearest-anchor candidates, given identical cache states.  The capacity-matched control takes the exact
per-anchor MIS cardinality, then selects nearest-by-physical-Key candidates
WITHOUT applying E-E conflict exclusion; it is not an independent algorithm
and is reported solely to isolate selection versus acceptance count.
"""
from __future__ import annotations
import math
import torch
from p_geometry import physical_l2_routes, _l2_cdist

PHYSICAL_GATE_MODES = {
    'physical_gated_merge_all_075': (0.75, 'all'),
    'physical_gated_importance_mis_075': (0.75, 'mis'),
    'physical_gated_nearest_matched_075': (0.75, 'matched'),
    'physical_gated_importance_mis_100': (1.00, 'mis'),
    'physical_gated_nearest_matched_100': (1.00, 'matched'),
    'necessity_history_exact': (1.00, 'h_exact'),
    'necessity_history_matched': (1.00, 'h_matched'),
    'necessity_ee_importance_matched': (1.00, 'importance_matched'),
    'group_d2': (1.00, 'group_d2'),
    'group_d2_wo_history': (1.00, 'group_d2_wo_history'),
    'group_d2_wo_ee': (1.00, 'group_d2_wo_ee'),
    'group_d2_importance_matched': (1.00, 'group_d2_importance_matched'),
}


def is_physical_method(method):
    return str(method) in PHYSICAL_GATE_MODES


def physical_method_spec(method):
    if not is_physical_method(method):
        raise ValueError('Not a physical E-A/E-E method: '+str(method))
    return PHYSICAL_GATE_MODES[str(method)]


@torch.no_grad()
def physical_routes_for_method(keys, keep, evict, delta_ea, method, chunk_size=256):
    """Identical physical E-A routing for the paired 0.75 methods.

    Uses the original physical_l2_routes (including numeric conventions),
    so 1.00 matches the original MergeAll E-A rule exactly.
    """
    ratio,_=physical_method_spec(method)
    return physical_l2_routes(keys,keep,evict,float(delta_ea)*ratio,chunk_size)


@torch.no_grad()
def nearest_count_matched_accept(best,routed,physical_ea_distance,mis_accept,evict):
    """No-E-E selection with *exactly* the MIS per-anchor accepted cardinalities.

    On the same routed pool, select the closest candidates per anchor. This
    is a controlled counterfactual, not an E-E MIS: selected pairs may conflict.
    """
    best=best.long().reshape(-1)
    routed=routed.bool().reshape(-1)
    mis_accept=mis_accept.bool().reshape(-1)
    physical_ea_distance=physical_ea_distance.double().reshape(-1)
    evict=evict.long().reshape(-1)
    n=len(best)
    if not all(len(x)==n for x in (routed,mis_accept,physical_ea_distance,evict)):
        raise ValueError('All routed arrays must have matching lengths')
    if bool((mis_accept & ~routed).any()):
        raise ValueError('MIS accepted an unrouted candidate')
    accepted=torch.zeros(n,dtype=torch.bool,device=best.device)
    for owner in torch.unique(best[routed],sorted=True).tolist():
        pool=torch.nonzero(routed & (best==owner)).flatten().tolist()
        wanted=int((mis_accept & (best==owner)).sum().item())
        if wanted>len(pool):raise AssertionError('Matched quota exceeds routed pool')
        ranked=sorted(pool,key=lambda i:(float(physical_ea_distance[i].item()),int(evict[i].item())))
        if wanted:
            accepted[torch.tensor(ranked[:wanted],device=best.device,dtype=torch.long)]=True
    if int(accepted.sum().item())!=int(mis_accept.sum().item()):
        raise AssertionError('Matched control failed to match MIS capacity')
    return accepted


@torch.no_grad()
def physical_selected_ee_conflicts(keys,evict,best,accepted,delta_ee):
    """Actual accepted E-E physical conflict count, irrespective of selection rule.

    Counts unordered pairs of evicted physical slots going to the same anchor;
    does NOT include anchor-evicted pairs. Chunking is over 128 sources to
    avoid a large dense matrix when no MIS accepts a large group.
    """
    if not math.isfinite(float(delta_ee)) or delta_ee<=0:
        raise ValueError('Positive E-E threshold required')
    evict=evict.long().reshape(-1)
    best=best.long().reshape(-1)
    accepted=accepted.bool().reshape(-1)
    if not (len(evict)==len(best)==len(accepted)):
        raise ValueError('Mismatched E-E input arrays')
    selected=torch.nonzero(accepted).flatten()
    conflicts=0
    pairs=0
    threshold=float(delta_ee)
    tol=2e-9*max(1.,threshold)
    for owner in torch.unique(best[selected],sorted=True).tolist():
        loc=selected[best[selected]==owner]
        m=int(loc.numel())
        pairs+=m*(m-1)//2
        if m<2:continue
        group=keys[evict[loc]].double()
        for i in range(0,m,128):
            x=group[i:i+128]
            dist=_l2_cdist(x,group)
            right=torch.arange(m,device=group.device)[None,:]>torch.arange(i,i+len(x),device=group.device)[:,None]
            conflicts+=int(((dist>threshold+tol)&right).sum().item())
    return dict(physical_ee_accepted_pairs=pairs,
                physical_ee_accepted_conflicts=conflicts)
