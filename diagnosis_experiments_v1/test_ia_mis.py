#!/usr/bin/env python3
"""Fast CPU smoke tests: IA E-A/E-E graph, unchanged D-first and IW fusion."""
import torch
import numpy as np
from k_only_mis_drop import ee_mis, fuse_weighted_slots
from cosine_mis_fusion import fuse_plan
from method_registry import (selected_methods, effective_method, family_for_method,
    graph_is_importance_aware, fusion_is_importance_weighted)
from ppl_sweep_utils import computation_key


def plan(keys,importance,counts=None,lam=1.0):
    keys=torch.tensor(keys,dtype=torch.float32)
    imp=torch.tensor(importance,dtype=torch.float32)
    cnt=torch.tensor(counts or [1.]*len(keys))
    vals=keys.clone()
    keep=torch.tensor([0]);evict=torch.arange(1,len(keys))
    best=torch.zeros(len(evict),dtype=torch.long)
    routed=torch.ones(len(evict),dtype=torch.bool)
    dist=(keys[evict]-keys[0]).norm(dim=-1)
    p=ee_mis(keys,vals,keep,evict,best,routed,dist,1.,1.,lam,
             'cpu_batched',counts=cnt,importance=imp,
             priority='importance_delta',importance_aware=True)
    return p,(keys,vals,cnt,keep,evict,best,routed,imp)


def check_close(actual,expected,atol=1e-6):
    assert np.isclose(float(actual),float(expected),atol=atol), (actual,expected)


def main():
    selected=selected_methods('k_only_importance,ia_mis_iwmean',
         'importance_weighted_merge_all,ia_mis_iwmean_drop,ia_mis_iwmean_drop_allcount_sink')
    assert selected==('full_cache','merge_all','importance_weighted_merge_all',
                      'ia_mis_iwmean_drop','ia_mis_iwmean_drop_allcount_sink')
    assert effective_method('ia_mis_iwmean_drop_allcount_sink',False)=='merge_all'
    assert effective_method('ia_mis_iwmean_drop_allcount_sink',True)=='ia_mis_iwmean_drop_allcount'
    assert graph_is_importance_aware(family_for_method('ia_mis_iwmean_drop'))
    assert fusion_is_importance_weighted('ia_mis_iwmean_drop')
    assert not fusion_is_importance_weighted('ia_mis_countmean_drop')
    assert computation_key('cosine',.8,.2,.3,'importance_weighted_merge_all') == computation_key('cosine',.8,.2,.4,'importance_weighted_merge_all')

    # E-A with rho=1 exactly reduces to raw L2 distance; 1.5 > threshold.
    p,x=plan([[0.,0.],[1.5,0.]], [1.,1.])
    assert p.ea_rejected==1 and not bool(p.accepted[0])
    # rho=.01 permits 1.5 Key separation; source still receives tiny weight.
    p,x=plan([[0.,0.],[1.5,0.]], [1.,.01])
    assert p.ea_rejected==0 and bool(p.accepted[0])
    # Contrast geometric 1.5 vs weighted E-A sqrt(2*.01/1.01)*1.5.
    check_close((2*.01/1.01)**.5*1.5,.2110816)

    # E-E: one high, one low -> relaxed pair; both high => conflict.
    keys=[[0.,0.],[.9,0.],[0.,.9]]
    ph,x=plan(keys,[1.,1.,1.])
    pl,x=plan(keys,[1.,1.,.01])
    assert int(ph.conflict_edges.sum())==1
    assert int(pl.conflict_edges.sum())==0
    assert ph.accepted.sum()==1 and pl.accepted.sum()==2
    assert pl.original_ee_conflict_edges==1

    # Multi-round rho can exceed one; count must not be clipped.
    p,x=plan([[0.,0.],[.2,0.]], [1.,.5],counts=[1.,10.])
    assert p.accepted.all() and p.ea_rejected==0
    keys,vals,cnt,keep,evict,best,routed,imp=x
    fused=fuse_weighted_slots(keys,vals,cnt,keep,evict,best,routed,p.accepted,
                              'ia_mis_iwmean_drop',importance=imp)
    # W0=1, W1=5 --> merged key=5*.2/6.
    check_close(fused[0][0,0],1./6)
    check_close(fused[2][0],11.)

    # Numeric replay and torch fusion must agree for C=1, including Drop/AllCount.
    p,x=plan([[0.,0.],[1.,0.],[0.,.2]], [1.,.8,.01],lam=.5)
    keys,vals,cnt,keep,evict,best,routed,imp=x
    kd,vd,cd=fuse_weighted_slots(keys,vals,cnt,keep,evict,best,routed,p.accepted,
                                  'ia_mis_iwmean_drop',importance=imp)
    ka,va,ca=fuse_weighted_slots(keys,vals,cnt,keep,evict,best,routed,p.accepted,
                                  'ia_mis_iwmean_drop_allcount',importance=imp)
    assert torch.equal(kd,ka) and torch.equal(vd,va)
    assert float(ca[0])==3.
    rp=fuse_plan(keys.numpy(),vals.numpy(),keep.numpy(),evict.numpy(),
                 best.numpy(),routed.numpy(),p.accepted.numpy(),
                 p.pool_sizes.numpy(),p.selected_sizes.numpy(),p.conflict_edges.numpy(),
                 mis_family='ia_mis_iwmean',importance=imp.numpy())
    assert np.allclose(kd.numpy(),rp.keys['ia_mis_iwmean_drop'])
    assert np.allclose(vd.numpy(),rp.values['ia_mis_iwmean_drop'])
    assert np.allclose(cd.numpy(),rp.counts['ia_mis_iwmean_drop'])
    assert np.allclose(ca.numpy(),rp.counts['ia_mis_iwmean_drop_allcount'])
    only_drop=fuse_plan(keys.numpy(),vals.numpy(),keep.numpy(),evict.numpy(),
              best.numpy(),routed.numpy(),p.accepted.numpy(),
              p.pool_sizes.numpy(),p.selected_sizes.numpy(),p.conflict_edges.numpy(),
              mis_family='ia_mis_iwmean',importance=imp.numpy(),
              methods_to_compute=('merge_all','ia_mis_iwmean_drop'))
    assert set(only_drop.keys)=={'merge_all','ia_mis_iwmean_drop'}

    # Degenerate all-zero current importance deterministically falls back to Count Mean.
    p,x=plan([[0.,0.],[.2,0.]], [0.,0.],counts=[2.,3.])
    keys,vals,cnt,keep,evict,best,routed,imp=x
    k,v,c=fuse_weighted_slots(keys,vals,cnt,keep,evict,best,routed,p.accepted,
                               'ia_mis_iwmean_drop',importance=imp)
    check_close(k[0,0],.12)
    assert torch.isfinite(k).all() and torch.isfinite(v).all()
    print('[TEST PASS] IA E-A/E-E; count-aware rho; IW K/V; Drop/AllCount; replay parity; zero fallback')


if __name__=='__main__':main()
