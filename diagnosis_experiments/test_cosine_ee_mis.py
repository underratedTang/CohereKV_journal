#!/usr/bin/env python3
"""CPU tests for isolated Cosine E-E MIS ablations, priorities, fusion, registry."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from k_only_mis_drop import ee_mis, fuse_weighted_slots
from cosine_mis_fusion import fuse_plan
from method_registry import (FAMILIES,FAMILY_METHODS,FAMILY_SINK_METHODS,
    selected_methods,ee_metric_for_method,canonical_method,effective_method,
    priority_for_family, fusion_weight_mode)
from ppl_sweep_utils import setting_key,computation_key,config_comparison_rows

def main():
    k=torch.tensor([[1.,0.],[1.,0.],[2.,0.],[0.,1.]],dtype=torch.float32)
    v=torch.tensor([[1.,1.],[2.,2.],[3.,3.],[4.,4.]],dtype=torch.float32)
    keep=torch.tensor([0]);evict=torch.tensor([1,2,3])
    best=torch.zeros(3,dtype=torch.long);routed=torch.ones(3,dtype=torch.bool)
    dist=torch.linalg.vector_norm(k[evict]-k[keep],dim=-1)
    imp=torch.tensor([100.,100.,100.,1.]);count=torch.ones(4)
    out={}
    for family in FAMILIES:
        for metric,threshold in (('cosine',.2),('l2',.2)):
            plan=ee_mis(k,v,keep,evict,best,routed,dist,1.,1.,threshold,
                        'cpu_batched',importance=imp,counts=count,
                        priority=priority_for_family(family),ee_metric=metric)
            out[(family,metric)]=plan
            for method in FAMILY_METHODS[family]:
                if ee_metric_for_method(method)!=metric:continue
                assert fusion_weight_mode(method)==fusion_weight_mode(canonical_method(method))
                fused=fuse_weighted_slots(k,v,count,keep,evict,best,routed,plan.accepted,method,importance=imp)
                assert fused[0].shape==(1,2) and fused[1].shape==(1,2)
                replay=fuse_plan(k.numpy(),v.numpy(),keep.numpy(),evict.numpy(),best.numpy(),
                    routed.numpy(),plan.accepted.numpy(),plan.pool_sizes.numpy(),
                    plan.selected_sizes.numpy(),plan.conflict_edges.numpy(),
                    mis_family=family,importance=imp.numpy(),methods_to_compute=('merge_all',method))
                assert method in replay.keys
            for method in FAMILY_SINK_METHODS[family]:
                if ee_metric_for_method(method)!=metric:continue
                assert effective_method(method,False)=='merge_all'
                assert effective_method(method,True)==method[:-5]
    # Same E-E graph for all priorities, regardless of importance.
    for metric in ('cosine','l2'):
        reference=out[('k_only',metric)].conflict_edges
        assert all(torch.equal(out[(f,metric)].conflict_edges,reference) for f in FAMILIES)
    # Two collinear, unequal-length keys are NOT cosine-incompatible;
    # they are L2-incompatible for small enough lambda*S.
    assert out[('k_only','cosine')].conflict_edges.sum().item()==2
    assert out[('k_only','l2')].conflict_edges.sum().item()==3
    # Cosine E-E edge graph does not depend on Key scale or frozen S.
    orig=out[('k_only','cosine')]
    scaled=ee_mis(k*9,v,keep,evict,best,routed,dist*9,100.,1.,.2,
                  'cpu_batched',importance=imp,counts=count,ee_metric='cosine')
    assert torch.equal(orig.conflict_edges,scaled.conflict_edges)
    assert torch.equal(orig.accepted,scaled.accepted)
    # L2 / cosine are separate, independently keyed sweeps.
    assert setting_key('cosine',.8,.1,.1,'k_only_mis_drop','l2')!=setting_key(
        'cosine',.8,.1,.1,'k_only_mis_drop','cosine')
    assert computation_key('cosine',.8,.1,.1,'merge_all','l2')==computation_key(
        'cosine',.8,.1,.5,'merge_all','cosine')
    assert computation_key('cosine',.8,.1,.1,'k_only_mis_drop','l2')!=computation_key(
        'cosine',.8,.1,.1,'k_only_mis_drop_cosee','cosine')
    methods=selected_methods('k_only_importance',
        'k_only_importance_mis_drop,k_only_importance_mis_drop_cosee')
    assert methods==('full_cache','merge_all','k_only_importance_mis_drop',
                     'k_only_importance_mis_drop_cosee')
    # Comparison schema stays rectangular for disjoint L2 and cosine grids.
    rows=[]
    for metric,m,t in [('l2','k_only_importance_mis_drop',.1),
                       ('cosine','k_only_importance_mis_drop_cosee',.02)]:
        for method,ppl in [('full_cache',5.),('merge_all',6.),(m,5.6)]:
            rows.append(dict(routing_mode='cosine',routing_parameter=.8,budget=.1,
                ee_metric=metric,ee_lambda=t,method=method,ppl=ppl,
                sweep_id=1,fixed_kv_budget_tokens=410))
    comp=config_comparison_rows(rows,methods)
    assert len(comp)==2 and len(set(tuple(sorted(r.keys())) for r in comp))==1
    by_metric={r['ee_metric']:r for r in comp}
    assert by_metric['l2']['k_only_importance_mis_drop_cosee']==''
    assert by_metric['cosine']['k_only_importance_mis_drop']==''
    for bad in (-.1,2.1):
        try:ee_mis(k,v,keep,evict,best,routed,dist,1.,1.,bad,'cpu_batched',importance=imp,ee_metric='cosine')
        except ValueError:pass
        else:raise AssertionError('Invalid cosine E-E threshold was accepted')
    print('[TEST PASS] cosine E-E graph, norm invariance, family priority parity, fusion, sink, metric-aware grid')

if __name__=='__main__':main()
