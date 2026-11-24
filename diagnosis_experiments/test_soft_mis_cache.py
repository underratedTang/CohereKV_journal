#!/usr/bin/env python3
"""CPU-only integration test: execute the actual cache code for two rounds."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from conference_mis import conference_routes,conference_weights
from p_geometry import parse_p,physical_l2_routes
from method_registry import METHODS,priority_for_method,attention_for_method,methods_for_p
from soft_mis import (soft_anchor_routes,soft_mis_weights,fuse_soft_slots,
                      merge_weighted_geometry)

class FakeBase:
    def __init__(self,*,model,stats,**kwargs):
        self._ppl_model=model
        self._stats=stats
        self._forward_calls=0
        self._ppl_logical_seen_tokens=0
        self.key_cache=[]
        self.value_cache=[]

def replace(cache,kv):
    cache.key_cache=[pair[0] for pair in kv]
    cache.value_cache=[pair[1] for pair in kv]

def split(keys,importance,positions,B,sink,recent):
    H,L,D=keys.shape
    return (torch.arange(B).expand(H,-1),torch.arange(B,L).expand(H,-1))

def importance(score,ptr,beta):
    return torch.ones(score.shape[0],score.shape[-1],device=score.device)

def reference(keys):
    return torch.full((keys.shape[0],keys.shape[1],1),10.,device=keys.device)

def load_cache():
    tree=ast.parse(Path(__file__).with_name('evaluate_ppl_continuous.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ContinuousPCache')
    namespace=dict(KVCompressingDynamicCache=FakeBase,TASK_ROOT=Path('.'),
        np=np,torch=torch,math=math,parse_p=parse_p,METHODS=METHODS,
        priority_for_method=priority_for_method,attention_for_method=attention_for_method,
        reference_scale_from_original_keys=reference,
        recent_window_importance=importance,select_split=split,
        physical_l2_routes=physical_l2_routes,soft_anchor_routes=soft_anchor_routes,
        conference_routes=conference_routes,conference_weights=conference_weights,
        soft_mis_weights=soft_mis_weights,fuse_soft_slots=fuse_soft_slots,
        merge_weighted_geometry=merge_weighted_geometry,
        _replace_dynamic_cache_inplace=replace)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'cache','exec'),namespace)
    return namespace['ContinuousPCache']

class CacheTests(unittest.TestCase):
    def test_all_methods_two_rounds(self):
        klass=load_cache()
        schemas=[]
        for method in METHODS:
            if method=='full_cache':continue
            model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace())]))
            obj=klass(model=model,method=method,budget=.5,p=2,lambda_ea=10.,lambda_ee=10.,
                      reference=np.array([[10.]]),sink_gate=np.array([[True]]),
                      unscored_len=7,scored_len=2,reference_mode='online_first',
                      sink_size=0,recent_protect_size=0)
            raw=torch.arange(6,dtype=torch.float32).view(1,1,6,1)*.01
            obj.key_cache=[raw];obj.value_cache=[raw.clone()]
            obj._ppl_logical_seen_tokens=6
            obj.maybe_compress((torch.ones(1,1,2,6),),None)
            self.assertEqual(obj.compress_rounds,1)
            tail=torch.arange(6,15,dtype=torch.float32).view(1,1,9,1)*.01
            obj.key_cache=[torch.cat((obj.key_cache[0],tail),dim=-2)]
            obj.value_cache=[torch.cat((obj.value_cache[0],tail),dim=-2)]
            obj._ppl_logical_seen_tokens=15
            obj.maybe_compress((torch.ones(1,1,2,12),),None)
            self.assertEqual(obj.compress_rounds,2)
            self.assertEqual(len(obj._stats['head_rounds']),2)
            # The per-head CSV writer uses the first method's keys as schema.
            # Require all arms to emit exactly the same column set.
            schemas.append(set(obj._stats['head_rounds'][0]))
            self.assertEqual(set(obj._stats['head_rounds'][0]),
                             set(obj._stats['head_rounds'][1]))
            self.assertEqual(obj.key_cache[0].shape[-2],3)
            self.assertTrue(torch.isfinite(obj.key_cache[0]).all())
            self.assertTrue(torch.isfinite(obj.geom_radii[0]).all())
            if method.startswith('coherekv'):
                self.assertEqual(int(obj.counts[0].sum().item()),15)
                self.assertFalse(hasattr(obj,'geom_counts'))
                expected=priority_for_method(method)
                self.assertEqual(obj._stats['head_rounds'][0]['mis_priority'],expected)
            if method!='eviction_only':
                self.assertGreater(obj._stats['routed_importance_mass'],0.)
                self.assertAlmostEqual(obj._stats['contributed_importance_mass'],obj._stats['routed_importance_mass'])
            if method=='eviction_only':
                self.assertEqual(int(obj.counts[0].sum().item()),3)
        self.assertTrue(all(schema==schemas[0] for schema in schemas))

    def test_hard_limit_cache_two_rounds(self):
        klass=load_cache()
        model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace())]))
        obj=klass(model=model,method='coherekv_soft_mis',budget=.5,p=math.inf,
                  lambda_ea=10.,lambda_ee=10.,
                  reference=np.array([[10.]]),sink_gate=np.array([[True]]),
                  unscored_len=7,scored_len=2,reference_mode='online_first',
                  sink_size=0,recent_protect_size=0)
        for rnd in range(2):
            if rnd==0:
                block=torch.arange(6,dtype=torch.float32).view(1,1,6,1)*.01
                obj.key_cache=[block];obj.value_cache=[block.clone()]
                obj._ppl_logical_seen_tokens=6
                score=torch.ones(1,1,2,6)
            else:
                block=torch.arange(6,15,dtype=torch.float32).view(1,1,9,1)*.01
                obj.key_cache=[torch.cat((obj.key_cache[0],block),dim=-2)]
                obj.value_cache=[torch.cat((obj.value_cache[0],block),dim=-2)]
                obj._ppl_logical_seen_tokens=15
                score=torch.ones(1,1,2,12)
            obj.maybe_compress((score,),None)
        self.assertEqual(obj.compress_rounds,2)
        self.assertFalse(hasattr(obj,'geom_counts'))
        self.assertTrue(torch.isfinite(obj.geom_radii[0]).all())

    def test_fractional_history_persists(self):
        klass=load_cache()
        model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace())]))
        obj=klass(model=model,method='coherekv_soft_mis',budget=.5,p=2,
                  lambda_ea=.1,lambda_ee=.01,
                  reference=np.array([[10.]]),sink_gate=np.array([[True]]),
                  unscored_len=7,scored_len=2,reference_mode='online_first',
                  sink_size=0,recent_protect_size=0)
        first=torch.arange(6,dtype=torch.float32).view(1,1,6,1)*.1
        obj.key_cache=[first];obj.value_cache=[first.clone()]
        obj._ppl_logical_seen_tokens=6
        obj.maybe_compress((torch.ones(1,1,2,6),),None)
        self.assertGreater(obj._stats['soft_fractional_slots'],0)
        self.assertFalse(hasattr(obj,'geom_counts'))
        self.assertEqual(int(obj.counts[0].sum().item()),6)
        second=torch.arange(6,15,dtype=torch.float32).view(1,1,9,1)*.1
        obj.key_cache=[torch.cat((obj.key_cache[0],second),dim=-2)]
        obj.value_cache=[torch.cat((obj.value_cache[0],second),dim=-2)]
        obj._ppl_logical_seen_tokens=15
        obj.maybe_compress((torch.ones(1,1,2,12),),None)
        self.assertEqual(obj.compress_rounds,2)
        self.assertTrue(torch.isfinite(obj.geom_radii[0]).all())
        self.assertFalse(hasattr(obj,'geom_counts'))
        # In round two one Evict is outside the physical E-A cutoff.
        self.assertEqual(int(obj.counts[0].sum().item()),14)

    def test_scheme_a_preserve_and_recompute_from_real_count(self):
        klass=load_cache()
        model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace())]))
        obj=klass(model=model,method='coherekv_soft_mis_importance',budget=.5,p=2,
                  lambda_ea=.1,lambda_ee=.01,reference=np.array([[10.]]),sink_gate=np.array([[True]]),
                  unscored_len=7,scored_len=2,reference_mode='online_first',sink_size=0,recent_protect_size=0)
        def compress(raw,logical):
            obj.key_cache=[raw];obj.value_cache=[raw.clone()]
            obj._ppl_logical_seen_tokens=logical
            obj.maybe_compress((torch.ones(1,1,2,raw.shape[-2]),),None)
            torch.testing.assert_close(model.model.layers[0].self_attn._mymethod_counts_kv,obj.contributions[0])
        first=torch.arange(6,dtype=torch.float32).view(1,1,6,1)*.1
        compress(first,6)
        c=obj.counts[0].clone(); m=obj.contributions[0].clone()
        self.assertTrue(bool((c>m).any()))
        far=torch.full((1,1,9,1),100.)
        compress(torch.cat((obj.key_cache[0],far),dim=-2),15)
        torch.testing.assert_close(obj.counts[0],c,rtol=0,atol=0)
        torch.testing.assert_close(obj.contributions[0],m,rtol=0,atol=0)
        # One fresh singleton matches the previously merged anchor. Its
        # history is compatible, so g=1; other singleton evicts are far away.
        index=int((c-m).argmax())
        near=obj.key_cache[0][:,:,index:index+1,:].clone()
        compress(torch.cat((obj.key_cache[0],near,far[:,:,:8,:]),dim=-2),24)
        self.assertAlmostEqual(obj.counts[0][0,0,index].item(),c[0,0,index].item()+1.)
        self.assertAlmostEqual(obj.contributions[0][0,0,index].item(),c[0,0,index].item()+1.)
        self.assertGreater(obj.contributions[0][0,0,index].item(),m[0,0,index].item()+1.)

    def test_hard_importance_available_and_attention_equals_count(self):
        self.assertIn('coherekv_soft_mis_importance',methods_for_p(math.inf))
        self.assertNotIn('coherekv_soft_mis_importance_log_count',methods_for_p(math.inf))
        klass=load_cache()
        for method in ('coherekv_soft_mis','coherekv_soft_mis_importance'):
            model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace())]))
            obj=klass(model=model,method=method,budget=.5,p=math.inf,lambda_ea=.1,lambda_ee=.01,
                      reference=np.array([[10.]]),sink_gate=np.array([[True]]),unscored_len=7,scored_len=2,
                      reference_mode='online_first',sink_size=0,recent_protect_size=0)
            for rnd in range(2):
                block=torch.arange(6 if rnd==0 else 9,dtype=torch.float32).view(1,1,-1,1)*.1
                raw=block if rnd==0 else torch.cat((obj.key_cache[0],block),dim=-2)
                obj.key_cache=[raw];obj.value_cache=[raw.clone()]
                obj._ppl_logical_seen_tokens=6+9*rnd
                obj.maybe_compress((torch.ones(1,1,2,raw.shape[-2]),),None)
                torch.testing.assert_close(obj.contributions[0],obj.counts[0],rtol=0,atol=0)
                torch.testing.assert_close(model.model.layers[0].self_attn._mymethod_counts_kv,obj.counts[0],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
