#!/usr/bin/env python3
"""CPU regression of the two parameter-free degree/E-A MIS priority ablations."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from method_registry import (FAMILIES,FAMILY_METHODS,METHODS,priority_for_family,
    selected_methods,fusion_weight_mode)
from k_only_mis_drop import ee_mis,fuse_weighted_slots
from cosine_mis_fusion import fuse_plan
from ppl_sweep_utils import computation_key


def case(anchor):
    # The center Evicted has degree 2; leaves have degree 1 and don't conflict.
    # Anchor location changes only E-A priorities, NOT E-E graph geometry.
    k=torch.tensor([[anchor],[0.1],[4.1],[4.2]],dtype=torch.float32)
    v=torch.tensor([[10.],[20.],[30.],[40.]],dtype=torch.float32)
    keep=torch.tensor([0]);evict=torch.tensor([1,2,3])
    best=torch.zeros(3,dtype=torch.long)
    routed=torch.ones(3,dtype=torch.bool)
    dist=torch.abs(k[evict,0]-k[0,0])
    counts=torch.tensor([2.,3.,4.,5.])
    importance=torch.tensor([8.,1.,100.,1.])
    args=(k,v,keep,evict,best,routed,dist,1.,1.,1.,'cpu_batched')
    plans={}
    for family in ('k_only','k_only_degree_anchor','k_only_anchor_first'):
        plans[family]=ee_mis(*args,counts=counts,importance=importance,
                             priority=priority_for_family(family))
    # Exact same E-E graph for each priority; zero EA-gated evictions.
    assert all(int(x.conflict_edges.sum().item())==2 for x in plans.values())
    assert all(x.ea_rejected==0 for x in plans.values())
    for fam,pl in plans.items():
        method=FAMILY_METHODS[fam][0]
        assert fusion_weight_mode(method)=='count'
        assert pl.accepted.sum().item()>=1
        fused=fuse_weighted_slots(k,v,counts,keep,evict,best,routed,pl.accepted,method)
        included=torch.cat((keep,evict[pl.accepted]))
        expected_counts=counts[included]
        expected_key=(k[included]*expected_counts[:,None]).sum(0)/expected_counts.sum()
        expected_value=(v[included]*expected_counts[:,None]).sum(0)/expected_counts.sum()
        assert torch.allclose(fused[0][0].float(),expected_key)
        assert torch.allclose(fused[1][0].float(),expected_value)
        assert torch.allclose(fused[2][0],expected_counts.sum())
        out=fuse_plan(k.numpy(),v.numpy(),keep.numpy(),evict.numpy(),best.numpy(),
            routed.numpy(),pl.accepted.numpy(),pl.pool_sizes.numpy(),
            pl.selected_sizes.numpy(),pl.conflict_edges.numpy(),
            mis_family=fam,importance=importance.numpy())
        assert np.allclose(out.keys[method][0],((k[0]+k[evict[pl.accepted]].sum(0))/(1+pl.accepted.sum())).numpy(),atol=1e-6)
        assert out.n_mis_rejected==int((~pl.accepted).sum())
    return plans


def main():
    expected=('k_only_degree_anchor_mis_drop','k_only_anchor_first_mis_drop')
    assert all(x in METHODS for x in expected)
    assert all(len(FAMILY_METHODS[f])==2 for f in ('k_only_degree_anchor','k_only_anchor_first'))
    assert selected_methods('k_only,k_only_degree_anchor,k_only_anchor_first',
        ','.join(('k_only_mis_drop',*expected))) == ('full_cache','merge_all','k_only_mis_drop',*expected)
    for method in expected:
        assert computation_key('cosine',.8,.05,.1,method)!=computation_key('cosine',.8,.05,.3,method)
    # Anchor near center -> new priorities select center, old degree selects leaves.
    x=case(.15)
    assert torch.equal(x['k_only'].accepted,torch.tensor([False,True,True]))
    assert torch.equal(x['k_only_degree_anchor'].accepted,torch.tensor([True,False,False]))
    assert torch.equal(x['k_only_anchor_first'].accepted,torch.tensor([True,False,False]))
    # Anchor between center and leaves -> distance-first chooses center,
    # product trades E-A^2 against degree and keeps both leaves, as degree-first.
    y=case(2.0)
    assert torch.equal(y['k_only'].accepted,torch.tensor([False,True,True]))
    assert torch.equal(y['k_only_degree_anchor'].accepted,torch.tensor([False,True,True]))
    assert torch.equal(y['k_only_anchor_first'].accepted,torch.tensor([True,False,False]))
    # Same conflict graph regardless of priority and backend implementation.
    assert torch.equal(x['k_only'].conflict_edges,y['k_only'].conflict_edges)
    print('[TEST PASS] degree, degree*anchor, anchor-first: same E-E graph, different accepted sets, Count-Mean Drop parity')

if __name__=='__main__':main()
