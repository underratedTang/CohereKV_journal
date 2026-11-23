#!/usr/bin/env python3
"""CPU-only zero-merge KeepKV regression, no Transformers or model weights."""
from types import SimpleNamespace
import numpy as np
import torch
from method_registry import (BASE_METHODS, METHODS, FUSION_METHODS,
                             selected_methods, family_for_method)
from test_continuous_cache_rounds import load_class


def test_eviction_only():
    assert 'eviction_only' in BASE_METHODS and 'eviction_only' in METHODS
    assert 'eviction_only' not in FUSION_METHODS
    assert family_for_method('eviction_only')=='baseline'
    assert selected_methods('k_only','k_only_mis_drop')[:3] == (
        'full_cache','eviction_only','merge_all')
    model=SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace())]))
    c=load_class()(model=model,method='eviction_only',budget=.5,p=2,
        lambda_ea=.1,lambda_ee=.1,reference=np.array([[10.]]),
        sink_gate=np.array([[True]]),unscored_len=7,scored_len=2,
        reference_mode='frozen_train',sink_size=0,recent_protect_size=0)
    # The AST-based test uses a deterministic KeepKV selector: first B indices.
    # Prove routing/MIS/fusion never run, even when all distances are eligible.
    def forbidden(*args,**kwargs):
        raise AssertionError('Eviction-only must never route, run MIS or fuse')
    env=c.maybe_compress.__globals__
    for name in ('certified_routes','physical_l2_routes',
                 'ee_mis','fuse_weighted_slots'):
        env[name]=forbidden
    k0=torch.tensor([0.,1.,2.,3.,4.,5.]).view(1,1,-1,1)
    v0=torch.tensor([100.,111.,122.,133.,144.,155.]).view(1,1,-1,1)
    c.key_cache=[k0.clone()];c.value_cache=[v0.clone()]
    c._ppl_logical_seen_tokens=6
    c.maybe_compress((torch.ones(1,1,2,6),),None)
    assert torch.equal(c.key_cache[0],k0[:,:,:3,:])
    assert torch.equal(c.value_cache[0],v0[:,:,:3,:])
    assert c.counts[0].tolist()==[[[1.,1.,1.]]]
    assert c.slot_positions[0].tolist()==[[[0,1,2]]]
    assert c.geom_counts[0].sum().item()==3
    assert c._stats['dropped_slots']==3
    assert c._stats['routed_slots']==c._stats['accepted_slots']==0
    assert c._stats['rejected_slots']==0
    assert len(c._stats['head_rounds'])==1
    head=c._stats['head_rounds'][0]
    assert head['evicted_slots']==head['dropped_slots']==3
    assert head['ea_certificate_used']==head['ee_mis_used']==False
    assert head['history_tracked']==head['history_certified']==False
    assert head['max_diameter_over_delta']==''
    # Second compression: no ghosts of evicted values or represented counts.
    ktail=torch.arange(6.,15.).view(1,1,-1,1)
    vtail=ktail+200.
    c.key_cache=[torch.cat((c.key_cache[0],ktail),dim=-2)]
    c.value_cache=[torch.cat((c.value_cache[0],vtail),dim=-2)]
    c._ppl_logical_seen_tokens=15
    c.maybe_compress((torch.ones(1,1,2,12),),None)
    assert torch.equal(c.key_cache[0],k0[:,:,:3,:])
    assert torch.equal(c.value_cache[0],v0[:,:,:3,:])
    assert c.counts[0].sum().item()==3
    assert c.geom_counts[0].sum().item()==3
    assert c._stats['dropped_slots']==12  # 3 + 9
    assert c._stats['compress_rounds']==2
    assert len(c._stats['head_rounds'])==2
    assert c._stats['head_rounds'][-1]['dropped_slots']==9
    print('[OK] eviction_only: identical selection, exact surviving K/V, no routing/MIS/merge, 2 rounds')


if __name__=='__main__':
    test_eviction_only()
