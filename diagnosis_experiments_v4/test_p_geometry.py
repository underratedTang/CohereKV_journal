#!/usr/bin/env python3
"""CPU-only mathematical and MIS-regression tests for p certificates."""
import math
import torch
from p_geometry import Cluster,singleton,merge_clusters,certified_routes
from k_only_mis_drop import ee_mis


def truth(keys,p):
    n=len(keys)
    mu=keys.double().mean(0)
    r=torch.linalg.vector_norm(keys.double()-mu,dim=-1)
    dist=torch.pdist(keys.double())
    if n==1:return mu,0.,0.
    if math.isinf(p):return mu,float(r.max()),float(dist.max())
    return mu,float((r.pow(p).mean()).pow(1./p)),float((dist.pow(p).mean()).pow(1./p))


def test_certificates():
    torch.manual_seed(42)
    for p in (1,1.2,2,4,8,16,math.inf):
        raw=torch.randn(24,5)
        clusters=[singleton(x) for x in raw]
        blocks=[]
        # Multiple merges in different orders, including unequal cluster sizes.
        for i in range(0,24,3):blocks.append(merge_clusters(clusters[i:i+3],p))
        for i in range(0,24,6):
            merged=merge_clusters(blocks[i//3:i//3+2],p)
            mu,r,d=truth(raw[i:i+6],p)
            assert torch.allclose(merged.mean,mu,atol=1e-6)
            assert merged.radius+1e-8>=r,(p,merged.radius,r)
            assert merged.diameter+1e-8>=d,(p,merged.diameter,d)
        all_merged=merge_clusters(blocks,p)
        mu,r,d=truth(raw,p)
        assert all_merged.radius+1e-8>=r
        assert all_merged.diameter+1e-8>=d
        assert math.isfinite(all_merged.radius)
        assert math.isfinite(all_merged.diameter)
    # This is an exact counterexample to max-diameter being required by p=2.
    x=torch.tensor([[0.]]*9+[[2.]])
    _,_,d=truth(x,2)
    assert d<1 and (x.max()-x.min())>1


def test_routing_and_all_mis_priorities():
    device='cpu'
    keys=torch.tensor([[0.,0.],[0.4,0.],[0.6,0.],[3.,0.],[2.9,0.]])
    vals=keys.clone();counts=torch.ones(5)
    keep=torch.tensor([0,3]);evict=torch.tensor([1,2,4])
    radii=torch.zeros(5,dtype=torch.float64);diams=radii.clone()
    best,routed,gamma=certified_routes(keys.double(),radii,diams,keep,evict,.9)
    assert routed.tolist()==[True,True,True]
    assert best.tolist()==[0,0,1]
    for priority in ('degree','degree_anchor_product','anchor_first','importance_mass_delta'):
        plan=ee_mis(keys,vals,keep,evict,best,routed,gamma,
                    1.,1.,.9,'cpu_batched',counts=counts,
                    importance=torch.ones(5),priority=priority,
                    certificate_centers=keys.double(),certificate_radii=radii)
        assert plan.accepted.tolist()==[True,True,True],(priority,plan.accepted)
    # When two collapsed representatives are identical but their historical
    # radii exceed the threshold, p-cross certificate must create an edge.
    centers=torch.tensor([[0.,0.],[0.,0.],[0.,0.]])
    radius=torch.tensor([0.,.6,.6])
    plan=ee_mis(centers,centers,torch.tensor([0]),torch.tensor([1,2]),
        torch.tensor([0,0]),torch.tensor([True,True]),torch.tensor([.6,.6]),
        1.,1.,1.0,'cpu_batched',importance=torch.ones(3),
        certificate_centers=centers,certificate_radii=radius)
    assert int(plan.conflict_edges.sum())==1
    assert int(plan.accepted.sum())==1


if __name__=='__main__':
    test_certificates();test_routing_and_all_mis_priorities()
    print('[OK] p certificates; p MIS preserves all priorities')
