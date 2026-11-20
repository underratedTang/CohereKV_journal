#!/usr/bin/env python3
"""CPU regression: Count-aware D-first vs degree; Drop/AllCount parity."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch

from k_only_mis_drop import (ee_mis,_neighbor_importance_delta,k_only_ee_mis,k_only_importance_ee_mis,
                            fuse_weighted_slots)
from cosine_mis_fusion import fuse_plan
from method_registry import (METHODS,FAMILY_METHODS,FAMILY_ALL_METHODS,selected_families,
                             selected_methods, family_for_method,effective_method,is_sink_gated,
                             original_method)
from mis_sweep_settings import sweep_settings
from ppl_sweep_utils import computation_key,config_comparison_rows


def construct(*,high_importance=100.0,lambda_=1.,backend='cpu_batched'):
    # Three Evicted nodes form a STAR under a K-only L2 conflict threshold=1.
    # e0: degree 2, high importance; e1/e2: degree 1, low importance.
    keys=torch.tensor([[0.0],[0.0],[2.0],[2.1]],dtype=torch.float32)
    values=torch.tensor([[1.],[2.],[3.],[4.]],dtype=torch.float32)
    importance=torch.tensor([0.,high_importance,1.,1.],dtype=torch.float32)
    keep=torch.tensor([0]);evict=torch.tensor([1,2,3]);best=torch.zeros(3,dtype=torch.long)
    routed=torch.ones(3,dtype=torch.bool)
    dist=torch.tensor([0.,2.,2.1])
    args=(keys,values,keep,evict,best,routed,dist,1.,1.,lambda_,backend)
    original=k_only_ee_mis(*args,importance=importance)
    weighted=k_only_importance_ee_mis(*args,importance=importance)
    return keys,values,importance,keep,evict,best,routed,original,weighted


def main():
    assert selected_methods('k_only,k_only_importance')==('full_cache','merge_all',*FAMILY_ALL_METHODS['k_only'],*FAMILY_ALL_METHODS['k_only_importance'])
    assert selected_methods('k_only')==('full_cache','merge_all',*FAMILY_ALL_METHODS['k_only'])
    assert selected_methods('k_only_importance')==('full_cache','merge_all',*FAMILY_ALL_METHODS['k_only_importance'])
    assert selected_families('k_only_importance,k_only')==('k_only','k_only_importance')
    for x in ('', 'qub', 'joint','k_only,k_only','k_only,unknown'):
        try:selected_families(x)
        except ValueError:pass
        else:raise AssertionError('Bad family accepted: '+x)
    assert family_for_method('k_only_importance_mis_drop_allcount')=='k_only_importance'
    assert len(FAMILY_ALL_METHODS['k_only'])==8 and len(FAMILY_ALL_METHODS['k_only_importance'])==8
    assert selected_methods('k_only,k_only_importance',
        'k_only_importance_mis_drop_allcount_sink,k_only_mis_drop') == (
            'full_cache','merge_all','k_only_mis_drop','k_only_importance_mis_drop_allcount_sink')
    for bad in ('k_only_mis_drop,k_only_mis_drop','bogus','k_only_mis_drop_sink,unknown',''):
        try:selected_methods('k_only,k_only_importance',bad)
        except ValueError:pass
        else:raise AssertionError('Bad switch list accepted: '+repr(bad))
    for method in FAMILY_ALL_METHODS['k_only']+FAMILY_ALL_METHODS['k_only_importance']:
        assert family_for_method(method) in ('k_only','k_only_importance')
        if is_sink_gated(method):
            assert effective_method(method,False)=='merge_all'
            assert effective_method(method,True)==original_method(method)
        else:
            assert not is_sink_gated(method)
            assert effective_method(method,False)==method
            assert effective_method(method,True)==method
    assert len(sweep_settings('0.3,0.4,0.5'))==3
    for text in ('', '0,0.5','0.4,0.4','nan'):
        try:sweep_settings(text)
        except ValueError:pass
        else:raise AssertionError('Bad sweep accepted: '+text)

    keys,vals,imp,keep,evict,best,routed,old,new=construct()
    assert torch.equal(old.accepted,torch.tensor([False,True,True]))
    assert torch.equal(new.accepted,torch.tensor([True,False,False]))
    assert torch.equal(old.conflict_edges,new.conflict_edges)
    assert old.conflict_edges.sum().item()==2
    assert old.priority=='degree' and new.priority=='importance_mass_delta'
    assert old.accepted_importance_sum==2.
    assert new.accepted_importance_sum==100.
    assert new.rejected_importance_sum==2.
    assert abs(old.routed_importance_sum-102.)<1e-6
    for backend in ('cpu_batched','gpu_batched','torch'):
        *_,a,b=construct(backend=backend)
        assert torch.equal(a.accepted,old.accepted)
        assert torch.equal(b.accepted,new.accepted)
    # Deliberately distinguish D-first from the PREVIOUS neighbor ratio.
    # Center importance=145, leaves=100 each: P-center=145/345 > 100/245,
    # but D-center=145-200=-55 < D-leaves=100-145=-45. D accepts leaves.
    # This example also shows D is a HEURISTIC, not exact MWIS.
    amb_imp=torch.tensor([0.,145.,100.,100.])
    amb_args=(keys,vals,keep,evict,best,routed,torch.tensor([0.,2.,2.1]),1.,1.,1.,'cpu_batched')
    amb_old=k_only_ee_mis(*amb_args,importance=amb_imp)
    amb_new=k_only_importance_ee_mis(*amb_args,importance=amb_imp)
    assert torch.equal(amb_old.accepted,torch.tensor([False,True,True]))
    assert torch.equal(amb_new.accepted,torch.tensor([False,True,True]))
    # Confirm the exact STATIC neighbor difference, including masked padding,
    # isolated nodes, negative scores, and all-zero importance.
    graph=torch.tensor([[[False,True,True,False,False],
                         [True,False,False,False,False],
                         [True,False,False,False,False],
                         [False,False,False,False,False],
                         [False,False,False,False,False]]])
    vals_imp=torch.tensor([[145.,100.,100.,4.,999.]])
    valid_mask=torch.tensor([[True,True,True,True,False]])
    scores=_neighbor_importance_delta(vals_imp,graph,valid_mask)[0]
    expected=torch.tensor([-55.,-45.,-45.,4.,0.],dtype=torch.float64)
    assert torch.equal(scores,expected)
    zero_scores=_neighbor_importance_delta(torch.zeros((1,5)),graph,valid_mask)
    assert torch.equal(zero_scores,torch.zeros_like(zero_scores))
    # Differentiates D-first from ratio: 100 vs two 60 leaves still chooses
    # center (D=-20 vs -40), showing no global MWIS guarantee (100 < 120).
    not_optimal_imp=torch.tensor([0.,100.,60.,60.])
    not_optimal=k_only_importance_ee_mis(*amb_args,importance=not_optimal_imp)
    assert torch.equal(not_optimal.accepted,torch.tensor([True,False,False]))
    *_,old_zero,new_zero=construct(high_importance=0.)
    assert old_zero.accepted.any() and new_zero.accepted.any()
    # Original MIS cannot be changed by changing importance on the same graph.
    *_,old_low,new_low=construct(high_importance=.01)
    assert torch.equal(old.accepted,old_low.accepted)
    assert torch.equal(new_low.accepted,old.accepted)
    # Empty graph: both priorities must accept every routed candidate.
    *_,old_empty,new_empty=construct(lambda_=4.)
    assert old_empty.accepted.all() and new_empty.accepted.all()
    try:k_only_importance_ee_mis(keys,vals,keep,evict,best,routed,torch.zeros(3),1.,1.,1.)
    except ValueError:pass
    else:raise AssertionError('Missing importance accepted')
    try:k_only_importance_ee_mis(keys,vals,keep,evict,best,routed,torch.zeros(3),1.,1.,1.,
                                  importance=torch.tensor([0.,float('nan'),1.,1.]))
    except ValueError:pass
    else:raise AssertionError('NaN importance accepted')

    # Both Drop and AllCount must share IDENTICAL K/V for the same input.
    counts=torch.tensor([2.,3.,5.,7.])
    for family,plan in [('k_only',old),('k_only_importance',new)]:
        drop,allcount=FAMILY_METHODS[family]
        mk,mv,mc=fuse_weighted_slots(keys,vals,counts,keep,evict,best,routed,routed,'merge_all')
        kd,vd,cd=fuse_weighted_slots(keys,vals,counts,keep,evict,best,routed,plan.accepted,drop)
        ka,va,ca=fuse_weighted_slots(keys,vals,counts,keep,evict,best,routed,plan.accepted,allcount)
        assert torch.equal(kd,ka) and torch.equal(vd,va)
        assert torch.equal(ca,mc)
        assert float(ca.item())==17.
        assert float(cd.item())==2.+float(counts[evict[plan.accepted]].sum())
        assert not torch.equal(kd,mk)
        replay=fuse_plan(keys.numpy(),vals.numpy(),keep.numpy(),evict.numpy(),best.numpy(),
                         routed.numpy(),plan.accepted.numpy(),plan.pool_sizes.numpy(),
                         plan.selected_sizes.numpy(),plan.conflict_edges.numpy(),
                         mis_family=family,importance=imp.numpy())
        assert np.array_equal(replay.keys[drop],replay.keys[allcount])
        assert np.array_equal(replay.values[drop],replay.values[allcount])
        assert replay.n_mis_rejected==int((~plan.accepted).sum())
        group=replay.group_rows[0]
        assert abs(group['accepted_importance_sum']-plan.accepted_importance_sum)<1e-6
        assert abs(group['rejected_importance_sum']-plan.rejected_importance_sum)<1e-6
        assert abs(group['routed_importance_sum']-102.)<1e-6
    # No cross-λ reuse for experimental arms; controls deliberately reused.
    assert computation_key('cosine',.8,.2,.3,'k_only_mis_drop')!=computation_key('cosine',.8,.2,.4,'k_only_mis_drop')
    assert computation_key('cosine',.8,.2,.3,'merge_all')==computation_key('cosine',.8,.2,.4,'merge_all')
    # Same fixed pre-compression state: sink gate has exact tensor equivalence.
    # On non-sink heads the *_sink result equals Merge All in K, V and Count.
    # On sink heads it equals its native MIS, not a different priority or Count.
    for family,plan in [('k_only',old),('k_only_importance',new)]:
        base_keys,base_vals,base_counts=fuse_weighted_slots(
            keys,vals,counts,keep,evict,best,routed,routed,'merge_all')
        for method in FAMILY_METHODS[family]:
            variant=method+'_sink'
            for sink in (True,False):
                applied=effective_method(variant,sink)
                accepted=(plan.accepted if sink else routed)
                actual=fuse_weighted_slots(keys,vals,counts,keep,evict,best,routed,accepted,applied)
                target=(fuse_weighted_slots(keys,vals,counts,keep,evict,best,routed,plan.accepted,method)
                        if sink else (base_keys,base_vals,base_counts))
                assert all(torch.equal(x,y) for x,y in zip(actual,target))
    print('[TEST PASS] original degree and Count*I D-first, Count/AllCount and Sink parity')


if __name__=='__main__':main()
