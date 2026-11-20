#!/usr/bin/env python3
"""CPU-only regression: physical MergeAll != certified E-A without E-E MIS.

Uses the actual ContinuousPCache implementation with fake model dependencies;
no Transformers/model weights required. Verifies propagation of potentially
oversized historical p certificates after unfiltered merging.
"""
from types import SimpleNamespace
import numpy as np
import torch
from method_registry import selected_methods, family_for_method, fusion_weight_mode
from test_continuous_cache_rounds import load_class


def make_cache(method, p='inf', lambda_ee=.1):
    model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace())]))
    klass=load_class()
    return klass(model=model, method=method, budget=.5, p=p,
                 lambda_ea=.1,lambda_ee=lambda_ee,
                 reference=np.array([[10.0]]),sink_gate=np.array([[True]]),
                 unscored_len=7,scored_len=2,reference_mode='online_first',
                 sink_size=0,recent_protect_size=0,graph_backend='streamed')


def compress_first(method, p='inf', lambda_ee=.1):
    cache=make_cache(method,p,lambda_ee)
    # 3 kept anchor keys: 0, 3, 6. Two evicted keys -0.9 and +0.9
    # route to the SAME anchor 0, passing E-A bound Delta=1. Their
    # mutual distance=1.8 > Delta, so no-E-E ablation must create
    # an oversized history group while MIS must reject >=1 candidate.
    data=torch.tensor([0.,3.,6.,-.9,.9,6.1],dtype=torch.float32).view(1,1,6,1)
    cache.key_cache=[data];cache.value_cache=[data.clone()]
    cache._ppl_logical_seen_tokens=6
    cache.maybe_compress((torch.ones((1,1,2,6)),),None)
    return cache


def test_method_registry():
    assert family_for_method('coherekv_wo_mis')=='baseline'
    assert fusion_weight_mode('coherekv_wo_mis')=='count'
    methods=selected_methods('k_only','coherekv_wo_mis,k_only_mis_drop')
    assert methods==('full_cache','merge_all','coherekv_wo_mis','k_only_mis_drop'),methods


def test_unfiltered_certified_routing_and_real_certificates():
    wo=compress_first('coherekv_wo_mis')
    mis=compress_first('k_only_mis_drop')
    merge=compress_first('merge_all')
    assert wo._stats['rejected_slots']==0
    assert wo._stats['routed_slots']==wo._stats['accepted_slots']
    assert mis._stats['rejected_slots']>=1,mis._stats
    assert wo.geom_diameters[0][0,0,0].item()>1.7,wo.geom_diameters
    assert wo._stats['geometry_over_delta']>=1,wo._stats
    assert wo._stats['head_rounds'][0]['history_tracked'] is True
    assert wo._stats['head_rounds'][0]['history_certified'] is False
    assert mis._stats['head_rounds'][0]['history_certified'] is True
    assert merge._stats['head_rounds'][0]['history_tracked'] is False
    assert torch.count_nonzero(merge.geom_diameters[0])==0  # unused placeholders
    assert wo.geom_counts[0].sum().item()==6

    # lambda_ee does not affect a no-E-E ablation; lambda_ea alone gates E-A.
    wo_other_ee=compress_first('coherekv_wo_mis',lambda_ee=2.5)
    assert torch.allclose(wo.key_cache[0],wo_other_ee.key_cache[0])
    assert torch.allclose(wo.geom_diameters[0],wo_other_ee.geom_diameters[0])


def test_history_survives_second_round():
    cache=compress_first('coherekv_wo_mis')
    past=cache.geom_diameters[0][0,0,0].item()
    assert past>1.7
    tail=torch.arange(6,15,dtype=torch.float32).view(1,1,9,1)*0.1
    cache.key_cache=[torch.cat([cache.key_cache[0],tail],dim=-2)]
    cache.value_cache=[torch.cat([cache.value_cache[0],tail],dim=-2)]
    cache._ppl_logical_seen_tokens=15
    cache.maybe_compress((torch.ones((1,1,2,12)),),None)
    assert cache.compress_rounds==2
    # E-A is allowed to drop new unrouteable tokens. Historical count is
    # the number of represented originals, not the number ever observed.
    assert 3 <= cache.geom_counts[0].sum().item() <= 15
    assert cache._stats['head_rounds'][1]['history_tracked'] is True
    # An over-limit old cluster retains its real certificate, never reset to 0.
    assert cache.geom_diameters[0][0,0,0].item()>=past-1e-5
    physical=compress_first('merge_all')
    physical.key_cache=[torch.cat([physical.key_cache[0],tail],dim=-2)]
    physical.value_cache=[torch.cat([physical.value_cache[0],tail],dim=-2)]
    physical._ppl_logical_seen_tokens=15
    physical.maybe_compress((torch.ones((1,1,2,12)),),None)
    assert physical._stats['head_rounds'][1]['routed_slots']>cache._stats['head_rounds'][1]['routed_slots']


def main():
    test_method_registry()
    test_unfiltered_certified_routing_and_real_certificates()
    test_history_survives_second_round()
    print('[OK] CohereKV w/o MIS certified E-A, no E-E, real over-limit history')


if __name__=='__main__':main()
