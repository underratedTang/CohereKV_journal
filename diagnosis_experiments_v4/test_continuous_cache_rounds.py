#!/usr/bin/env python3
"""No Transformers/model weights: execute actual compression method over two rounds."""
import ast
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import torch
from p_geometry import Cluster,certified_routes,physical_l2_routes,merge_clusters,merge_cluster_tensors,parse_p
from k_only_mis_drop import ee_mis,fuse_weighted_slots
from method_registry import (effective_method, family_for_method,
    graph_is_importance_aware, priority_for_family, is_wo_ea_certificate)

class FakeBase:
    def __init__(self,*,model,method,family,compress_ratio,sim_thr,method_sim_thrs,
                 keepkv_config,mymethod_config,centroidkv_config,min_kv_len,
                 compress_every,stats,task_eval_dir,target_kv_len):
        self._ppl_model=model;self._stats=stats
        self._forward_calls=0;self._ppl_logical_seen_tokens=0
        self.key_cache=[];self.value_cache=[]

def replace(cache,legacy):
    cache.key_cache=[x[0] for x in legacy]
    cache.value_cache=[x[1] for x in legacy]

def split(keys,importance,positions,B,sink,recent):
    H,L,D=keys.shape
    k=torch.arange(B).expand(H,-1)
    e=torch.arange(B,L).expand(H,-1)
    return k,e

def imp(score,ptr,beta):
    return torch.ones((score.shape[0],score.shape[-1]),device=score.device)

def scale(keys):
    return torch.ones((keys.shape[0],keys.shape[1],1),device=keys.device)*10

def load_class():
    tree=ast.parse(Path(__file__).with_name('evaluate_ppl_continuous.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ContinuousPCache')
    env=dict(KVCompressingDynamicCache=FakeBase,TASK_ROOT=Path('.'),
             np=np,torch=torch,math=math,Cluster=Cluster,parse_p=parse_p,
             reference_scale_from_original_keys=scale,
             recent_window_importance=imp,select_split=split,certified_routes=certified_routes,physical_l2_routes=physical_l2_routes,
             effective_method=effective_method,is_wo_ea_certificate=is_wo_ea_certificate,
             ee_mis=ee_mis,fuse_weighted_slots=fuse_weighted_slots,
             family_for_method=family_for_method,graph_is_importance_aware=graph_is_importance_aware,
             priority_for_family=priority_for_family,merge_clusters=merge_clusters,
             merge_cluster_tensors=merge_cluster_tensors,
             _replace_dynamic_cache_inplace=replace)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'<cache>','exec'),env)
    return env['ContinuousPCache']

def test_two_rounds():
    c=load_class()
    model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace())]))
    for method in ('k_only_mis_drop','k_only_importance_mis_drop',
                   'k_only_degree_anchor_mis_drop','k_only_anchor_first_mis_drop',
                   'merge_all','coherekv_wo_mis',
                   'k_only_mis_drop_wo_ea_certificate',
                   'k_only_importance_mis_drop_wo_ea_certificate',
                   'k_only_importance_iwmean_drop'):
        obj=c(model=model,method=method,budget=.5,p=2,lambda_ea=10,lambda_ee=10,
              reference=np.array([[10.]]),sink_gate=np.array([[True]]),
              unscored_len=7,scored_len=2,reference_mode='online_first',
              sink_size=0,recent_protect_size=0,graph_backend='streamed')
        data=torch.arange(6,dtype=torch.float32).reshape(1,1,6,1)*.01
        obj.key_cache=[data];obj.value_cache=[data.clone()]
        obj._ppl_logical_seen_tokens=6
        obj.maybe_compress((torch.ones(1,1,2,6),),None)
        assert obj.compress_rounds==1
        assert obj.geom_counts[0].sum()==6,(method,obj.geom_counts)
        assert obj.counts[0].sum()==6
        # Continuous positions (6 ...14): second boundary 6+9=15.
        tail=torch.arange(6,15,dtype=torch.float32).reshape(1,1,9,1)*.01
        obj.key_cache=[torch.cat([obj.key_cache[0],tail],dim=-2)]
        obj.value_cache=[torch.cat([obj.value_cache[0],tail],dim=-2)]
        obj._ppl_logical_seen_tokens=15
        obj.maybe_compress((torch.ones(1,1,2,12),),None)
        assert obj.compress_rounds==2
        assert obj.geom_counts[0].sum()==15,(method,obj.geom_counts)
        assert obj.counts[0].sum()==15
        assert len(obj._stats['head_rounds'])==2
    print('[OK] two real compression rounds, geometry history, methods/Counts retained')

if __name__=='__main__':test_two_rounds()
