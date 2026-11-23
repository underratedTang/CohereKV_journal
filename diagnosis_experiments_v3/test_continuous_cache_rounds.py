#!/usr/bin/env python3
"""No Transformers/model weights: execute actual compression method over two rounds."""
import ast
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import torch
from history_oracle_p4 import extend_histories,merged_histories,audit_head,summarize_pairs
from p_geometry import Cluster,certified_routes,physical_l2_routes,merge_clusters,merge_cluster_tensors,merge_cluster_moments,route_diagnostics_moment,route_diagnostics_countaware,parse_p,validate_geometry_mode,route_diagnostics_p2,is_count_aware,physical_count_aware_ratio
from fusion_fidelity import audit_fusion_head
from necessity_controls import (HeadHistories,exact_history_selection,importance_count_matched,is_history_method)
from physical_ee_controls import (is_physical_method,physical_method_spec,
    physical_routes_for_method,nearest_count_matched_accept,physical_selected_ee_conflicts)
from k_only_mis_drop import ee_mis,fuse_weighted_slots
from group_d2_selection import (GROUP_METHODS,GROUP_D2,WO_HISTORY,WO_EE,MATCHED,
                                choose_group_d2,merge_group_d2_history)
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
             validate_geometry_mode=validate_geometry_mode,route_diagnostics_p2=route_diagnostics_p2,
             np=np,torch=torch,math=math,Cluster=Cluster,parse_p=parse_p,
             reference_scale_from_original_keys=scale,
             recent_window_importance=imp,select_split=split,certified_routes=certified_routes,physical_l2_routes=physical_l2_routes,
             effective_method=effective_method,is_wo_ea_certificate=is_wo_ea_certificate,
             ee_mis=ee_mis,fuse_weighted_slots=fuse_weighted_slots,
             family_for_method=family_for_method,graph_is_importance_aware=graph_is_importance_aware,
             priority_for_family=priority_for_family,merge_clusters=merge_clusters,
             merge_cluster_tensors=merge_cluster_tensors,merge_cluster_moments=merge_cluster_moments,
             route_diagnostics_moment=route_diagnostics_moment,route_diagnostics_countaware=route_diagnostics_countaware,
             is_count_aware=is_count_aware,physical_count_aware_ratio=physical_count_aware_ratio,
             audit_fusion_head=audit_fusion_head,
             is_physical_method=is_physical_method,physical_method_spec=physical_method_spec,
             physical_routes_for_method=physical_routes_for_method,
             nearest_count_matched_accept=nearest_count_matched_accept,
             HeadHistories=HeadHistories,exact_history_selection=exact_history_selection,
             importance_count_matched=importance_count_matched,is_history_method=is_history_method,
             physical_selected_ee_conflicts=physical_selected_ee_conflicts,
             extend_histories=extend_histories,merged_histories=merged_histories,
             audit_head=audit_head,summarize_pairs=summarize_pairs,
             _replace_dynamic_cache_inplace=replace,
             GROUP_METHODS=GROUP_METHODS,GROUP_D2=GROUP_D2,
             WO_HISTORY=WO_HISTORY,WO_EE=WO_EE,MATCHED=MATCHED,
             choose_group_d2=choose_group_d2,
             merge_group_d2_history=merge_group_d2_history)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'<cache>','exec'),env)
    return env['ContinuousPCache']

def test_two_rounds():
    c=load_class()
    model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace())]))
    for method in ('k_only_mis_drop','k_only_importance_mis_drop',
                   'k_only_degree_anchor_mis_drop','k_only_anchor_first_mis_drop',
                   'eviction_only','merge_all','coherekv_inf_certificate','coherekv_wo_mis',
                   'k_only_mis_drop_wo_ea_certificate',
                   'k_only_importance_mis_drop_wo_ea_certificate',
                   'k_only_importance_iwmean_drop',
                   'necessity_history_exact','necessity_history_matched',
                   'necessity_ee_importance_matched',
                   'group_d2','group_d2_wo_history',
                   'group_d2_wo_ee','group_d2_importance_matched'):
        obj=c(model=model,method=method,budget=.5,p=2,lambda_ea=10,lambda_ee=10,
              reference=np.array([[10.]]),sink_gate=np.array([[True]]),
              unscored_len=7,scored_len=2,reference_mode='online_first',
              sink_size=0,recent_protect_size=0,graph_backend='streamed')
        data=torch.arange(6,dtype=torch.float32).reshape(1,1,6,1)*.01
        obj.key_cache=[data];obj.value_cache=[data.clone()]
        obj._ppl_logical_seen_tokens=6
        obj.maybe_compress((torch.ones(1,1,2,6),),None)
        assert obj.compress_rounds==1
        assert obj.geom_counts[0].sum()==(3 if method=='eviction_only' else 6),(method,obj.geom_counts)
        assert obj.counts[0].sum()==(3 if method=='eviction_only' else 6)
        # Continuous positions (6 ...14): second boundary 6+9=15.
        tail=torch.arange(6,15,dtype=torch.float32).reshape(1,1,9,1)*.01
        obj.key_cache=[torch.cat([obj.key_cache[0],tail],dim=-2)]
        obj.value_cache=[torch.cat([obj.value_cache[0],tail],dim=-2)]
        obj._ppl_logical_seen_tokens=15
        obj.maybe_compress((torch.ones(1,1,2,12),),None)
        assert obj.compress_rounds==2
        assert obj.geom_counts[0].sum()==(3 if method=='eviction_only' else 15),(method,obj.geom_counts)
        assert obj.counts[0].sum()==(3 if method=='eviction_only' else 15)
        assert len(obj._stats['head_rounds'])==2
        if method=='coherekv_inf_certificate':
            assert math.isinf(obj.p) and obj.geometry_mode=='minkowski'
            assert obj._stats['accepted_slots']>0
            assert all(row['max_diameter_over_delta']<=1.+1e-6 for row in obj._stats['head_rounds'])
        if method=='eviction_only':
            assert obj._stats['accepted_slots']==0 and obj._stats['routed_slots']==0
            assert all((v==1).all() for v in obj.counts)
        if method.startswith('group_d2'):
            assert obj.group_m2 is not None
            assert bool(torch.all(obj.group_m2[0]>=0))
            assert int(obj._stats.get('group_d2_anchors',0))>0
            if method=='group_d2':
                assert obj._stats.get('group_d2_real_violations',0)==0
                assert obj._stats.get('group_d2_same_state_matched_anchors',0)>0
            if method=='group_d2_importance_matched':
                assert obj._stats.get('group_d2_matched_quota_shortfall',0)==0
        if is_history_method(method):
            assert (0,0) in obj.exact_histories
            assert obj.exact_histories[(0,0)].counts().sum()==15
    print('[OK] two real compression rounds, geometry history, methods/Counts retained')



def test_group_d2_rejection_changes_physical_fusion():
    c=load_class()
    model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace())]))
    k=torch.tensor([0.,10.,20.,.1,.2,2.],dtype=torch.float32).view(1,1,6,1)
    obj=c(model=model,method='group_d2',budget=.5,p=2,lambda_ea=10.,lambda_ee=10.,
        lambda_history=.1,reference=np.array([[10.]]),sink_gate=np.array([[True]]),
        unscored_len=7,scored_len=2,reference_mode='frozen_train',
        geometry_mode='exact_p2',sink_size=0,recent_protect_size=0,
        graph_backend='streamed')
    obj.key_cache=[k];obj.value_cache=[k.clone()]
    obj._ppl_logical_seen_tokens=6
    obj.maybe_compress((torch.ones(1,1,2,6),),None)
    assert int(obj._stats['routed_slots'])==3
    assert int(obj._stats['accepted_slots'])==2
    assert int(obj._stats['rejected_slots'])==1
    assert obj.counts[0][0,0,:].tolist()==[3.,1.,1.]
    assert int(obj.geom_counts[0].sum().item())==5
    assert np.allclose(obj.geom_means[0][0,0,0].item(),.1,atol=1e-9)
    assert obj.group_m2[0][0,0,0].item()>0
    assert float(obj._stats['group_d2_real_violations'])==0
    print('[OK] real Group-D2 end-to-end CPU merge rejects 1 evicted, fuses CountMean and tracks M2')

if __name__=='__main__':
    test_two_rounds()
    test_group_d2_rejection_changes_physical_fusion()
