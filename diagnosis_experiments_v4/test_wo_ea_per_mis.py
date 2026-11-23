#!/usr/bin/env python3
"""CPU-only regression of E-A-certificate ablation using real ContinuousPCache.

The new method must route by physical Key L2, retain p-historical E-E MIS,
propagate real nonzero history even when the resulting p-diameter exceeds Delta.
No model weights or Transformers needed (reuses the AST-based fake cache).
"""
from types import SimpleNamespace
import numpy as np
import torch
from method_registry import (selected_methods, family_for_method,
    fusion_weight_mode, priority_for_family, FAMILY_NATIVE_METHODS,
    FAMILIES, METHODS, FUSION_METHODS, original_method, effective_method,
    is_sink_gated, is_wo_ea_certificate, without_ea_certificate)
from test_continuous_cache_rounds import load_class
from k_only_mis_drop import fuse_weighted_slots


def cache_for(method, *, lambda_ea=0.1, lambda_ee=0.1, p='inf'):
    model=SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace())]))
    return load_class()(model=model,method=method,budget=0.5,p=p,
        lambda_ea=lambda_ea,lambda_ee=lambda_ee,
        reference=np.array([[10.]],dtype=np.float64),
        sink_gate=np.array([[True]]),unscored_len=7,scored_len=2,
        reference_mode='frozen_train',sink_size=0,recent_protect_size=0,
        graph_backend='streamed')


def _forward_compress(cache, values, *, logical=6, score_rows=2):
    kk=torch.tensor(values,dtype=torch.float32).reshape(1,1,-1,1)
    cache.key_cache=[kk]
    cache.value_cache=[kk.clone()]
    cache._ppl_logical_seen_tokens=logical
    cache.maybe_compress((torch.ones((1,1,score_rows,len(values))),),None)
    return cache


def test_registry_and_degree_priority():
    assert family_for_method('k_only_mis_drop_wo_ea_certificate')=='k_only'
    assert priority_for_family(family_for_method('k_only_mis_drop_wo_ea_certificate'))=='degree'
    assert fusion_weight_mode('k_only_mis_drop_wo_ea_certificate')=='count'
    methods=selected_methods('k_only','k_only_mis_drop_wo_ea_certificate,k_only_mis_drop')
    assert methods==('full_cache','eviction_only','merge_all','k_only_mis_drop','k_only_mis_drop_wo_ea_certificate'),methods


def test_ee_mis_still_filters_conflicts():
    # First round: no prior history. Both .9 and -.9 route to anchor 0
    # under EA Delta=1, but their E-E L2 distance 1.8 exceeds EE Delta=1.
    values=[0.,4.,8.,-.9,.9,20.]
    wo=_forward_compress(cache_for('k_only_mis_drop_wo_ea_certificate'),values)
    full=_forward_compress(cache_for('k_only_mis_drop'),values)
    no_mis=_forward_compress(cache_for('merge_all'),values)
    assert wo._stats['routed_slots']==2,wo._stats
    assert wo._stats['accepted_slots']==1,wo._stats
    assert wo._stats['rejected_slots']==1,wo._stats
    assert torch.allclose(wo.key_cache[0],full.key_cache[0]),'At round 1 E-A is identical'
    assert no_mis._stats['accepted_slots']==2
    head=wo._stats['head_rounds'][0]
    assert head['ea_certificate_used'] is False
    assert head['ee_mis_used'] is True
    assert head['history_tracked'] is True
    assert head['history_certified'] is False
    assert wo.geom_counts[0].sum().item()==4
    assert wo.geom_radii[0].max().item()>0


def test_ea_is_physical_but_history_remains_real():
    # Pre-existing real cluster {-0.4,0.4} represented by its mean K=0.
    # Its radius=.4, diameter=.8, Count=2, all originally <= Delta=1.
    # The new token K=.7 passes physical E-A (.7<=1), but fails the
    # conference historical E-A bound (.4+.7=1.1>1).
    def initial(method):
        c=cache_for(method)
        c.counts=[torch.tensor([[[2.,1.,1.]]])]
        c.slot_positions=[torch.tensor([[[0,1,2]]],dtype=torch.long)]
        c.geom_counts=[torch.tensor([[[2.,1.,1.]]],dtype=torch.float64)]
        c.geom_means=[torch.tensor([[[[0.],[10.],[20.]]]],dtype=torch.float64)]
        c.geom_radii=[torch.tensor([[[.4,0.,0.]]],dtype=torch.float64)]
        c.geom_diameters=[torch.tensor([[[.8,0.,0.]]],dtype=torch.float64)]
        return _forward_compress(c,[0.,10.,20.,.7,50.,60.])
    no_ea=initial('k_only_mis_drop_wo_ea_certificate')
    full=initial('k_only_mis_drop')
    no_ee=initial('coherekv_wo_mis')
    merge=initial('merge_all')
    assert no_ea._stats['routed_slots']==1
    assert no_ea._stats['accepted_slots']==1
    assert full._stats['routed_slots']==0
    assert no_ee._stats['routed_slots']==0
    assert merge._stats['routed_slots']==1
    # The accepted group now represents {-0.4,0.4,.7}: true max=1.1,
    # so both its stored diameter and the violation count must reveal it.
    assert no_ea.geom_counts[0][0,0,0].item()==3
    assert no_ea.geom_diameters[0][0,0,0].item()>1.09
    assert no_ea._stats['geometry_over_delta']>=1
    assert no_ea._stats['head_rounds'][0]['history_tracked'] is True
    assert no_ea._stats['head_rounds'][0]['history_certified'] is False
    assert merge._stats['head_rounds'][0]['history_tracked'] is False
    assert no_ea._stats['head_rounds'][0]['max_diameter_over_delta']>1.
    # A second valid compression boundary must carry over the actual
    # (possibly over-limit) certificate, not reset it to zero.
    tail=torch.tensor([70.,80.,90.,100.,110.,120.,130.,140.,150.]).view(1,1,9,1)
    no_ea.key_cache=[torch.cat((no_ea.key_cache[0],tail),dim=-2)]
    no_ea.value_cache=[torch.cat((no_ea.value_cache[0],tail),dim=-2)]
    no_ea._ppl_logical_seen_tokens=15
    no_ea.maybe_compress((torch.ones((1,1,2,12)),),None)
    assert no_ea.compress_rounds==2
    assert no_ea.geom_diameters[0][0,0,0].item()>=1.09
    assert no_ea._stats['head_rounds'][1]['history_tracked'] is True



def test_ee_uses_historical_radius_not_only_physical_keys():
    # Both candidates are within E-A physical L2=0.4 and their physical
    # E-E L2=0.8 is <= Delta=1, but their historical cross-cluster
    # certificate is .4+.8+.4=1.6, so MIS MUST reject one candidate.
    cache=cache_for('k_only_mis_drop_wo_ea_certificate',p='inf')
    vals=[0.,10.,20.,-.4,.4,30.]
    cache.counts=[torch.tensor([[[1.,1.,1.,2.,2.,1.]]])]
    cache.slot_positions=[torch.tensor([[[0,1,2,3,4,5]]],dtype=torch.long)]
    cache.geom_counts=[cache.counts[0].double().clone()]
    cache.geom_means=[torch.tensor(vals,dtype=torch.float64).reshape(1,1,6,1)]
    cache.geom_radii=[torch.tensor([[[0.,0.,0.,.4,.4,0.]]],dtype=torch.float64)]
    cache.geom_diameters=[torch.tensor([[[0.,0.,0.,.8,.8,0.]]],dtype=torch.float64)]
    _forward_compress(cache,vals)
    assert cache._stats['routed_slots']==2,cache._stats
    assert cache._stats['accepted_slots']==1,cache._stats
    assert cache._stats['rejected_slots']==1,cache._stats
    assert cache._stats['mis_conflict_edges']>=1,cache._stats
    assert cache._stats['head_rounds'][0]['ee_mis_used'] is True


def test_every_mis_arm_has_a_matching_wo_ea_arm():
    native=[m for family in FAMILIES for m in FAMILY_NATIVE_METHODS[family]]
    assert len(native)==15
    for method in native:
        paired=method+'_wo_ea_certificate'
        assert method in METHODS and paired in METHODS,paired
        assert not is_wo_ea_certificate(method)
        assert is_wo_ea_certificate(paired)
        assert without_ea_certificate(paired)==method
        assert family_for_method(paired)==family_for_method(method)
        assert priority_for_family(family_for_method(paired))==priority_for_family(family_for_method(method))
        assert fusion_weight_mode(paired)==fusion_weight_mode(method)
        assert is_sink_gated(paired)==is_sink_gated(method)
        assert effective_method(paired,False)==('merge_all' if is_sink_gated(method) else paired)
        native_effective=effective_method(method,True)
        expected=native_effective+'_wo_ea_certificate'
        assert effective_method(paired,True)==expected
        if not is_sink_gated(method):
            assert paired in FUSION_METHODS
    assert 'coherekv_wo_ea_certificate' not in METHODS
    enabled=selected_methods(','.join(FAMILIES),','.join(native+[x+'_wo_ea_certificate' for x in native]))
    assert len(enabled)==3+2*len(native)


def test_each_pair_same_first_compression():
    # No history yet => certified and physical E-A both reduce to Key L2.
    # This checks that the chosen MIS priority, fusion weights, Count rules,
    # and sink-head gating are identical within each E-A ablation pair.
    native=[m for family in FAMILIES for m in FAMILY_NATIVE_METHODS[family]]
    for method in native:
        paired=method+'_wo_ea_certificate'
        original=_forward_compress(cache_for(method,lambda_ea=.1,lambda_ee=.1),
                                    [0.,4.,8.,-.9,.9,20.])
        ablation=_forward_compress(cache_for(paired,lambda_ea=.1,lambda_ee=.1),
                                    [0.,4.,8.,-.9,.9,20.])
        assert original._stats['routed_slots']==ablation._stats['routed_slots'],method
        assert original._stats['accepted_slots']==ablation._stats['accepted_slots'],method
        assert torch.allclose(original.key_cache[0],ablation.key_cache[0],atol=1e-6,rtol=0),method
        assert torch.allclose(original.value_cache[0],ablation.value_cache[0],atol=1e-6,rtol=0),method
        assert torch.allclose(original.counts[0],ablation.counts[0],atol=1e-6,rtol=0),method
        assert torch.allclose(original.geom_diameters[0],ablation.geom_diameters[0],atol=1e-8,rtol=0),method
        assert not ablation._stats['head_rounds'][0]['ea_certificate_used'],method
        assert ablation._stats['head_rounds'][0]['ee_mis_used'],method
        assert ablation._stats['head_rounds'][0]['history_tracked'],method
        assert not ablation._stats['head_rounds'][0]['history_certified'],method


def test_importance_pair_only_changes_ea():
    # Start from a nonzero historical radius to distinguish physical E-A
    # from certified E-A, without altering importance D-first ranking.
    def history(method):
        obj=cache_for(method)
        obj.counts=[torch.tensor([[[2.,1.,1.]]])]
        obj.slot_positions=[torch.tensor([[[0,1,2]]],dtype=torch.long)]
        obj.geom_counts=[torch.tensor([[[2.,1.,1.]]],dtype=torch.float64)]
        obj.geom_means=[torch.tensor([[[[0.],[10.],[20.]]]],dtype=torch.float64)]
        obj.geom_radii=[torch.tensor([[[.4,0.,0.]]],dtype=torch.float64)]
        obj.geom_diameters=[torch.tensor([[[.8,0.,0.]]],dtype=torch.float64)]
        return _forward_compress(obj,[0.,10.,20.,.7,50.,60.])
    original=history('k_only_importance_mis_drop')
    paired=history('k_only_importance_mis_drop_wo_ea_certificate')
    assert original._stats['routed_slots']==0
    assert paired._stats['routed_slots']==1
    assert paired._stats['accepted_slots']==1
    assert paired._stats['head_rounds'][0]['history_tracked']
    assert paired._stats['head_rounds'][0]['ee_mis_used']
    assert paired._stats['head_rounds'][0]['max_diameter_over_delta']>1


def test_fusion_variants_exactly_match_each_pair():
    # Nonuniform importance and Counts distinguish CountMean, IW-mean,
    # I-mean and AllCount (the latter adds rejected Count only).
    keys=torch.tensor([[0.],[10.],[20.],[30.]],dtype=torch.float32)
    vals=torch.tensor([[1.],[3.],[5.],[7.]],dtype=torch.float32)
    counts=torch.tensor([2.,3.,5.,7.])
    importance=torch.tensor([.25,.5,.75,.125])
    keep=torch.tensor([0]);evict=torch.tensor([1,2,3]);best=torch.zeros(3,dtype=torch.long)
    routed=torch.ones(3,dtype=torch.bool)
    accepted=torch.tensor([True,False,True])
    for family in FAMILIES:
        for method in FAMILY_NATIVE_METHODS[family]:
            if is_sink_gated(method):
                continue
            result=fuse_weighted_slots(keys,vals,counts,keep,evict,best,routed,
                accepted,method,importance=importance)
            paired=fuse_weighted_slots(keys,vals,counts,keep,evict,best,routed,
                accepted,method+'_wo_ea_certificate',importance=importance)
            assert all(torch.equal(x,y) for x,y in zip(result,paired)),method
            assert result[-1].item()==(17 if method.endswith('_allcount') else 12),method
    # Non-sink heads of a paired sink-gated method use the original MergeAll.
    for method in ('k_only_mis_drop_sink','k_only_importance_mis_drop_sink',
                   'k_only_importance_imean_drop_sink'):
        native=cache_for(method)
        paired=cache_for(method+'_wo_ea_certificate')
        native.sink_gate[:]=False;paired.sink_gate[:]=False
        _forward_compress(native,[0.,4.,8.,-.9,.9,20.])
        _forward_compress(paired,[0.,4.,8.,-.9,.9,20.])
        assert native._stats['head_rounds'][0]['strategy']=='merge_all',method
        assert paired._stats['head_rounds'][0]['strategy']=='merge_all',method
        assert torch.equal(native.key_cache[0],paired.key_cache[0]),method
        assert torch.equal(native.counts[0],paired.counts[0]),method


def main():
    test_registry_and_degree_priority()
    test_every_mis_arm_has_a_matching_wo_ea_arm()
    test_fusion_variants_exactly_match_each_pair()
    test_each_pair_same_first_compression()
    test_importance_pair_only_changes_ea()
    test_ee_mis_still_filters_conflicts()
    test_ee_uses_historical_radius_not_only_physical_keys()
    test_ea_is_physical_but_history_remains_real()
    print('[OK] paired E-A ablations for all MIS arms, matched E-E/priority/fusion, history across rounds')


if __name__=='__main__':main()
