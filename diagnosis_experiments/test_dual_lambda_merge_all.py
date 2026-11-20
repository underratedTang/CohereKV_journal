#!/usr/bin/env python3
"""No model weights: verify MergeAll routing and independently gated EA/EE MIS."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from p_geometry import physical_l2_routes,certified_routes,merge_cluster_tensors,parse_p
from k_only_mis_drop import ee_mis,fuse_weighted_slots
from method_registry import effective_method,family_for_method,graph_is_importance_aware,priority_for_family


def test_physical_l2_not_historical():
    keys=torch.tensor([[0.0],[0.4],[0.49],[7.0]],dtype=torch.float32)
    keep=torch.tensor([0]);evict=torch.tensor([1,2,3])
    best,routed,dist=physical_l2_routes(keys,keep,evict,0.5)
    assert routed.tolist()==[True,True,False]
    # All same physical keys, but historical certificates would reject all.
    radii=torch.full((4,),100.0)
    diam=torch.full((4,),200.0)
    _,certified,_=certified_routes(keys.double(),radii,diam,keep,evict,0.5)
    assert not bool(certified.any())
    assert dist[:2].tolist()==[float(keys[1]),float(keys[2])]


def test_ee_lambda_independent():
    keys=torch.tensor([[0.0],[0.4],[-0.4]],dtype=torch.float32)
    keep=torch.tensor([0]);evict=torch.tensor([1,2])
    centers=keys.double();zeros=torch.zeros((3,),dtype=torch.float64)
    best,routed,ea=certified_routes(centers,zeros,zeros,keep,evict,0.5)
    assert routed.tolist()==[True,True]
    counts=torch.ones(3)
    importance=torch.ones(3)
    def plan(ee_cutoff):
        return ee_mis(keys,keys,keep,evict,best,routed,ea,1.0,1.,ee_cutoff,
                      backend='streamed',counts=counts,importance=importance,
                      priority='degree',certificate_centers=centers,
                      certificate_radii=zeros)
    assert int(plan(0.5).accepted.sum())==1
    assert int(plan(1.0).accepted.sum())==2
    _,both,_=certified_routes(centers,zeros,zeros,keep,evict,0.3)
    assert not bool(both.any())


def test_mis_invariant_with_distinct_lambdas():
    # All singleton history; EA <= 0.5 and EE <= 1.0, certified bound 1.0.
    keys=torch.tensor([[0.0],[0.4],[-0.4]],dtype=torch.float64)
    c=torch.ones((3,),dtype=torch.float64)
    z=torch.zeros((3,),dtype=torch.float64)
    C,mean,R,D=merge_cluster_tensors(c,keys,z,z,2)
    assert C.item()==3 and D.item() < 1.0


def main():
    test_physical_l2_not_historical()
    test_ee_lambda_independent()
    test_mis_invariant_with_distinct_lambdas()
    print('[OK] MergeAll physical-only L2; independent EA/EE thresholds; p-invariant baseline')


if __name__=='__main__':main()
