import math
import unittest
from types import SimpleNamespace

import torch

from models.common.mymethod_common import (
    accumulate_d2o_importance_state,
    append_score_rows,
    build_merge_plan,
    effective_radius,
    merge_layer,
    normalize_selector,
    recent_window_importance,
    reference_scale_from_original_keys,
    select_keep_indices,
    select_keep_indices_d2o,
)


class _Attn: pass
class _Layer:
    def __init__(self): self.self_attn=_Attn()
class _Model:
    def __init__(self,n=1): self.model=SimpleNamespace(layers=[_Layer() for _ in range(n)])


def _basic_states(keys):
    x,L,_=keys.shape
    c=torch.ones((x,L),dtype=torch.float32)
    z=torch.zeros_like(c)
    p=torch.arange(L,dtype=torch.int32).view(1,L).expand(x,-1)
    return c,z,p


class TestGraphMISMymethod(unittest.TestCase):

    def test_selector_namespace_is_decoupled(self):
        self.assertEqual(normalize_selector('attention'),'attention')
        self.assertEqual(normalize_selector('keepkv'),'keepkv')
        self.assertEqual(normalize_selector('d2o'),'d2o')
        with self.assertRaises(ValueError):
            normalize_selector('grouplogitkv')

    def test_auto_backend_falls_back_to_torch_on_cpu(self):
        keys=torch.tensor([[[0.0],[0.6],[-0.6]]])
        c,z,p=_basic_states(keys); imp=torch.tensor([[10.,0.,0.]])
        a=build_merge_plan(keys,c,z,z,imp,p,1,torch.ones((1,1)),1.0,selector='attention',group_mode='full',route_backend='auto',graph_backend='auto')
        b=build_merge_plan(keys,c,z,z,imp,p,1,torch.ones((1,1)),1.0,selector='attention',group_mode='full',route_backend='torch',graph_backend='torch')
        self.assertTrue(torch.equal(a.best_idx,b.best_idx))
        self.assertTrue(torch.equal(a.accepted,b.accepted))
        self.assertTrue(torch.allclose(a.group_gamma_max,b.group_gamma_max))

    def test_reference_scale_matches_explicit_pair_rms(self):
        torch.manual_seed(0)
        k=torch.randn(2,3,7,5)
        got=reference_scale_from_original_keys(k,eps=0.0)
        for b in range(2):
            for h in range(3):
                explicit=torch.pdist(k[b,h].float()).square().mean().sqrt()
                self.assertTrue(torch.allclose(got[b,h,0],explicit,rtol=1e-5,atol=1e-6))

    def test_reference_scale_is_scale_equivariant(self):
        k=torch.randn(1,2,9,4)
        s=reference_scale_from_original_keys(k)
        s2=reference_scale_from_original_keys(k*3.25)
        self.assertTrue(torch.allclose(s2,s*3.25,rtol=1e-5,atol=2e-6))

    def test_effective_radius(self):
        c=torch.tensor([[1.,2.,4.]])
        r=torch.tensor([[3.,3.,3.]])
        d=torch.tensor([[9.,2.,4.]])
        got=effective_radius(c,r,d)
        self.assertTrue(torch.allclose(got,torch.tensor([[0.,1.,3.]])))

    def _conflict_example(self, mode):
        # Retain token 0. Candidates 1 and 2 are each 0.6 from anchor but 1.2 apart.
        keys=torch.tensor([[[0.0],[0.6],[-0.6]]])
        c,z,p=_basic_states(keys)
        imp=torch.tensor([[10.,0.,0.]])
        return build_merge_plan(
            keys,c,z,z,imp,p,keep_len=1,reference_scale=torch.ones((1,1)),
            lambda_mymethod=1.0,selector='attention',group_mode=mode,
            distance_chunk_size=0,
        )

    def test_graph_mis_rejects_conflicting_candidate(self):
        plan=self._conflict_example('full')
        self.assertEqual(int(plan.route_compatible.sum()),2)
        self.assertEqual(int(plan.accepted.sum()),1)
        self.assertEqual(int(plan.conflict_edges_per_anchor.sum()),1)
        self.assertEqual(int(plan.candidate_pool_sizes.max()),2)
        self.assertEqual(int(plan.selected_per_anchor.max()),1)

    def test_anchor_only_exposes_complete_group_violation_but_keeps_valid_D(self):
        plan=self._conflict_example('anchor_only')
        self.assertEqual(int(plan.accepted.sum()),2)
        # Valid propagated certificate must see candidate-candidate distance 1.2.
        self.assertGreater(float(plan.group_gamma_max[0,0]),1.19)

    def test_nonconflicting_candidates_are_all_selected(self):
        keys=torch.tensor([[[0.0],[0.3],[-0.3],[0.2]]])
        c,z,p=_basic_states(keys); imp=torch.tensor([[10.,0.,0.,0.]])
        plan=build_merge_plan(keys,c,z,z,imp,p,1,torch.ones((1,1)),1.0,
                              selector='attention',group_mode='full')
        self.assertEqual(int(plan.route_compatible.sum()),3)
        self.assertEqual(int(plan.accepted.sum()),3)
        self.assertEqual(int(plan.conflict_edges_per_anchor.sum()),0)

    def test_mis_is_independent_and_maximal(self):
        keys=torch.tensor([[[0.0],[0.55],[-0.55],[0.45],[-0.45],[0.1]]])
        c,z,p=_basic_states(keys); imp=torch.tensor([[20.,0.,0.,0.,0.,0.]])
        plan=build_merge_plan(keys,c,z,z,imp,p,1,torch.ones((1,1)),0.9,
                              selector='attention',group_mode='full')
        ev=plan.evict_idx[0]
        routed=torch.nonzero(plan.route_compatible[0]).flatten()
        selected=torch.nonzero(plan.accepted[0]).flatten()
        # Independent: every selected pair <= Delta (R=D=0 here).
        for a in range(selected.numel()):
            for b in range(a+1,selected.numel()):
                ia=int(ev[selected[a]]); ib=int(ev[selected[b]])
                self.assertLessEqual(float((keys[0,ia]-keys[0,ib]).norm()),0.90001)
        # Maximal: every routed rejected candidate conflicts with a selected candidate.
        for r in routed.tolist():
            if bool(plan.accepted[0,r]): continue
            ir=int(ev[r]); ok=False
            for s in selected.tolist():
                isel=int(ev[s])
                if float((keys[0,ir]-keys[0,isel]).norm())>0.9:
                    ok=True; break
            self.assertTrue(ok)

    def test_full_group_exact_diameter_bound(self):
        torch.manual_seed(2)
        keys=torch.randn(2,32,4)*0.2
        c,z,p=_basic_states(keys); imp=torch.randn(2,32)
        scale=torch.ones((2,1))
        plan=build_merge_plan(keys,c,z,z,imp,p,16,scale,0.75,
                              selector='attention',group_mode='full')
        for row in range(2):
            for j in range(16):
                cand=torch.nonzero(plan.accepted[row] & (plan.best_idx[row]==j)).flatten()
                if cand.numel()==0: continue
                inds=[int(plan.keep_idx[row,j])]+[int(plan.evict_idx[row,c]) for c in cand]
                diam=float(torch.cdist(keys[row,inds],keys[row,inds]).max())
                self.assertLessEqual(diam,0.7501)

    def test_merge_layer_count_weighted_centroid_and_certificate(self):
        keys=torch.tensor([[[[0.0],[0.3],[-0.3]]]])
        vals=torch.tensor([[[[0.0],[3.0],[-3.0]]]])
        counts=torch.ones((1,1,3),dtype=torch.int32)
        z=torch.zeros((1,1,3)); hist=torch.zeros((1,1,0,3)); pos=torch.arange(3,dtype=torch.int32).view(1,1,3)
        out=merge_layer(keys,vals,counts,z,z,hist,0,pos,ratio=2/3,reference_scale=torch.ones((1,1,1)),
                        lambda_mymethod=1.0,target_length=1,recent_window=0,selector='attention',
                        current_importance=torch.tensor([[[10.,0.,0.]]]),group_mode='full')
        knew,vnew,cnew,rnew,dnew,_,_=out
        self.assertEqual(int(cnew.item()),3)
        self.assertAlmostEqual(float(knew.item()),0.0,places=6)
        self.assertAlmostEqual(float(vnew.item()),0.0,places=6)
        self.assertGreaterEqual(float(rnew.item())+1e-6,0.3)
        self.assertGreaterEqual(float(dnew.item())+1e-6,0.6)
        self.assertLessEqual(float(dnew.item()),1.0001)

    def test_eviction_only_accepts_nothing(self):
        keys=torch.randn(1,10,3); c,z,p=_basic_states(keys); imp=torch.randn(1,10)
        plan=build_merge_plan(keys,c,z,z,imp,p,6,torch.ones((1,1)),0.5,
                              selector='attention',group_mode='eviction_only')
        self.assertEqual(int(plan.accepted.sum()),0)
        self.assertEqual(int(plan.route_compatible.sum()),0)

    def test_normalized_plan_scale_invariance(self):
        torch.manual_seed(3)
        keys=torch.randn(1,20,5); c,z,p=_basic_states(keys); imp=torch.randn(1,20)
        s=reference_scale_from_original_keys(keys.view(1,1,20,5)).reshape(1,1)
        a=build_merge_plan(keys,c,z,z,imp,p,12,s,0.6,selector='attention',group_mode='full')
        b=build_merge_plan(keys*7,c,z,z,imp,p,12,s*7,0.6,selector='attention',group_mode='full')
        self.assertTrue(torch.equal(a.keep_idx,b.keep_idx)); self.assertTrue(torch.equal(a.accepted,b.accepted)); self.assertTrue(torch.equal(a.best_idx,b.best_idx))

    def test_recent_selector_sink_recent_heavy(self):
        imp=torch.tensor([[0.,100.,3.,2.,1.,50.]])
        pos=torch.arange(6,dtype=torch.int32).view(1,6)
        keep=select_keep_indices(imp,pos,keep_len=4,sink_size=1,recent_window=2)
        self.assertEqual(set(keep[0].tolist()),{0,1,4,5})

    def test_d2o_selector_sink_is_inside_hh_budget_and_recent_is_protected(self):
        # keep_len=4, recent=2 => HH budget=2. Sink position 0 must consume one
        # HH slot even though its cumulative score is the lowest.
        imp=torch.tensor([[0.,100.,90.,80.,1.,2.]])
        pos=torch.arange(6,dtype=torch.int32).view(1,6)
        keep=select_keep_indices_d2o(imp,pos,keep_len=4,sink_size=1,recent_size=2)
        self.assertEqual(set(keep[0].tolist()),{0,1,4,5})

    def test_d2o_merge_keeps_anchor_importance_only(self):
        keys=torch.tensor([[[[0.0],[0.2],[0.3],[3.0]]]])
        vals=keys.clone()
        counts=torch.ones((1,1,4),dtype=torch.int32)
        z=torch.zeros((1,1,4)); hist=torch.zeros((1,1,0,4))
        pos=torch.arange(4,dtype=torch.int32).view(1,1,4)
        cumulative=torch.tensor([[[100.,50.,40.,1.]]])
        out=merge_layer(
            keys,vals,counts,z,z,hist,0,pos,ratio=0.5,
            reference_scale=torch.ones((1,1,1)),lambda_mymethod=1.0,
            target_length=2,sink_size=0,recent_window=0,selector='d2o',
            current_importance=cumulative,group_mode='full',
            return_selector_state=True,
        )
        *_, state=out
        self.assertIsNotNone(state)
        # Retained entries are the two highest cumulative scores. Source scores
        # are discarded rather than added to the destination history.
        self.assertTrue(torch.equal(state, torch.tensor([[[100.,50.]]])))

    def test_d2o_runtime_accumulator_appends_new_suffix_and_persists(self):
        a=torch.tensor([[[1.,2.]]])
        got=accumulate_d2o_importance_state(None,a)
        self.assertTrue(torch.equal(got,a))
        b=torch.tensor([[[3.,4.,5.]]])
        got=accumulate_d2o_importance_state(got,b)
        self.assertTrue(torch.equal(got,torch.tensor([[[4.,6.,5.]]])))


    def test_keepkv_log_count_ablation_changes_ranking_without_changing_history_definition(self):
        keys=torch.tensor([[[[0.0],[1.0],[2.0]]]])
        vals=keys.clone()
        counts=torch.tensor([[[1,3,1]]],dtype=torch.int32)
        z=torch.zeros((1,1,3))
        hist=torch.tensor([[[[5.0,2.0,1.0]]]])
        pos=torch.arange(3,dtype=torch.int32).view(1,1,3)
        common=dict(
            keys=keys, values=vals, counts=counts, radii=z, diameters=z,
            score_history=hist, history_ptr=0, positions=pos, ratio=0.0,
            reference_scale=torch.ones((1,1,1)), lambda_mymethod=1.0,
            target_length=1, sink_size=0, recent_window=0, score_window=1,
            selector='keepkv', group_mode='eviction_only',
        )
        out_no=merge_layer(**common, keepkv_use_log_count=False)
        out_yes=merge_layer(**common, keepkv_use_log_count=True)
        self.assertAlmostEqual(float(out_no[0].item()),0.0,places=6)
        self.assertAlmostEqual(float(out_yes[0].item()),1.0,places=6)

    def test_keepkv_history_uses_count_weighted_membership_and_sink_is_legal_anchor(self):
        keys=torch.tensor([[[[0.0],[0.2]]]])
        vals=keys.clone()
        counts=torch.tensor([[[2,1]]],dtype=torch.int32)
        z=torch.zeros((1,1,2))
        hist=torch.tensor([[[[4.0,10.0]]]])
        pos=torch.arange(2,dtype=torch.int32).view(1,1,2)
        out=merge_layer(
            keys,vals,counts,z,z,hist,0,pos,ratio=0.5,
            reference_scale=torch.ones((1,1,1)),lambda_mymethod=1.0,
            target_length=1,sink_size=1,recent_window=0,score_window=1,
            selector='keepkv',keepkv_use_log_count=False,group_mode='full',
        )
        _,_,cnew,_,_,hnew,_=out
        self.assertEqual(int(cnew.item()),3)
        self.assertAlmostEqual(float(hnew.item()),6.0,places=6)

    def test_d2o_without_protection_uses_attention_partition(self):
        imp=torch.tensor([[1.0,4.0,4.0,2.0,0.0]])
        pos=torch.arange(5,dtype=torch.int32).view(1,5)
        from models.common.mymethod_common import select_keep_indices_attention
        a=select_keep_indices_attention(imp,3)
        b=select_keep_indices_d2o(imp,pos,3,sink_size=0,recent_size=0)
        self.assertTrue(torch.equal(a,b))

    def test_d2o_reset_every_compression_clears_persistent_importance(self):
        from models.llama.merge_mymethod_llama import Repair
        model=_Model(1)
        k=torch.tensor([[[[0.0],[0.2],[0.4],[2.0]]]],dtype=torch.float32)
        v=k.clone()
        cumulative=(torch.tensor([[[10.,8.,2.,1.]]],dtype=torch.float32),)
        Repair(
            model,((k,v),),cumulative,None,ratio=0.5,lambda_mymethod=1.0,
            target_length=2,selector='d2o',d2o_sink_size=0,d2o_recent_size=0,
            d2o_use_log_count=True,d2o_score_reset_interval=1,group_mode='full',
        )
        attn=model.model.layers[0].self_attn
        self.assertFalse(hasattr(attn,'_mymethod_d2o_cumulative_importance'))
        self.assertEqual(attn._mymethod_d2o_compression_count,1)

    def test_attention_and_d2o_reset1_zero_protection_have_same_core_partition(self):
        from models.llama.merge_mymethod_llama import Repair
        k=torch.tensor([[[[0.0],[0.2],[0.4],[2.0]]]],dtype=torch.float32)
        v=k*3
        scores=(torch.tensor([[[10.,8.,2.,1.]]],dtype=torch.float32),)
        ma,md=_Model(1),_Model(1)
        a=Repair(ma,((k,v),),scores,None,ratio=0.5,lambda_mymethod=1.0,
                 target_length=2,selector='attention',group_mode='full')
        d=Repair(md,((k,v),),scores,None,ratio=0.5,lambda_mymethod=1.0,
                 target_length=2,selector='d2o',d2o_sink_size=0,d2o_recent_size=0,
                 d2o_use_log_count=True,d2o_score_reset_interval=1,group_mode='full')
        self.assertTrue(torch.equal(a[0][0],d[0][0]))
        self.assertTrue(torch.equal(a[0][1],d[0][1]))
        self.assertTrue(torch.equal(ma._mymethod_counts[0],md._mymethod_counts[0]))

    def test_score_ring(self):
        hist=torch.zeros((1,1,3,2)); rows=torch.tensor([[[[1.,2.],[3.,4.],[5.,6.],[7.,8.]]]])
        hist,ptr,scale=append_score_rows(hist,0,None,rows,0.0)
        self.assertEqual(ptr,0); self.assertTrue(torch.equal(hist[0,0],rows[0,0,-3:]))
        imp=recent_window_importance(hist.reshape(1,3,2),ptr,1.0)
        self.assertTrue(torch.equal(imp,rows[0,0,-3:].sum(dim=0).view(1,2)))

    def test_reference_scale_frozen_across_repair_rounds(self):
        from models.llama.merge_mymethod_llama import Repair
        model=_Model(1)
        k=torch.tensor([[[[0.0],[0.2],[0.4]]]]); v=k.clone(); scores=(torch.tensor([[[3.,2.,1.]]]),)
        out=Repair(model,((k,v),),scores,None,ratio=1/3,lambda_mymethod=1.0,target_length=2,
                   selector='attention',group_mode='full')
        first=model._mymethod_reference_scales[0].clone()
        k2=torch.cat([out[0][0],torch.tensor([[[[9.0]]]])],dim=2); v2=k2.clone()
        Repair(model,((k2,v2),),(torch.tensor([[[3.,2.,1.]]]),),None,ratio=1/3,lambda_mymethod=1.0,target_length=2,
               selector='attention',group_mode='full')
        self.assertTrue(torch.equal(first,model._mymethod_reference_scales[0]))

    def test_randomized_cross_round_certificate_invariant(self):
        torch.manual_seed(7)
        original=torch.randn(10,3)*0.25
        keys=original.clone().view(1,1,10,3)
        values=keys.clone()
        counts=torch.ones((1,1,10),dtype=torch.int32)
        radii=torch.zeros((1,1,10)); diam=torch.zeros((1,1,10))
        hist=torch.zeros((1,1,0,10)); pos=torch.arange(10,dtype=torch.int32).view(1,1,10)
        ref=reference_scale_from_original_keys(keys)
        members=[[original[i].clone()] for i in range(10)]
        next_orig=10
        for round_no in range(3):
            L=keys.shape[2]; keep_len=max(2,L-3)
            imp=torch.linspace(2.0,0.0,L).view(1,1,L)
            plan=build_merge_plan(
                keys.reshape(1,L,3),counts.reshape(1,L),radii.reshape(1,L),diam.reshape(1,L),
                imp.reshape(1,L),pos.reshape(1,L),keep_len,ref.reshape(1,1),1.0,
                selector='attention',group_mode='full'
            )
            new_members=[]
            for j in range(keep_len):
                group=list(members[int(plan.keep_idx[0,j])])
                for e in range(plan.evict_idx.shape[1]):
                    if bool(plan.accepted[0,e]) and int(plan.best_idx[0,e])==j:
                        group.extend(members[int(plan.evict_idx[0,e])])
                new_members.append(group)
            keys,values,counts,radii,diam,hist,pos=merge_layer(
                keys,values,counts,radii,diam,hist,0,pos,ratio=0.0,reference_scale=ref,
                lambda_mymethod=1.0,target_length=keep_len,selector='attention',
                current_importance=imp,group_mode='full'
            )
            for j,group in enumerate(new_members):
                pts=torch.stack(group)
                centroid=pts.mean(dim=0)
                exact_r=float((pts-centroid).norm(dim=-1).max())
                exact_d=float(torch.cdist(pts,pts).max())
                self.assertTrue(torch.allclose(keys[0,0,j].float(),centroid,atol=2e-6,rtol=1e-5))
                self.assertGreaterEqual(float(radii[0,0,j])+2e-5,exact_r)
                self.assertGreaterEqual(float(diam[0,0,j])+2e-5,exact_d)
            members=new_members
            # Append two new singleton tokens; S_ref must remain frozen.
            newpts=torch.randn(2,3)*0.25
            keys=torch.cat([keys,newpts.view(1,1,2,3)],dim=2); values=keys.clone()
            counts=torch.cat([counts,torch.ones((1,1,2),dtype=torch.int32)],dim=2)
            radii=torch.cat([radii,torch.zeros((1,1,2))],dim=2); diam=torch.cat([diam,torch.zeros((1,1,2))],dim=2)
            hist=torch.zeros((1,1,0,keys.shape[2]))
            pos=torch.cat([pos,torch.arange(next_orig,next_orig+2,dtype=torch.int32).view(1,1,2)],dim=2)
            members.extend([[newpts[0].clone()],[newpts[1].clone()]])
            next_orig+=2

    def test_llama_qwen_wrapper_parity(self):
        from models.llama.merge_mymethod_llama import Repair as LR
        from models.qwen.merge_mymethod_qwen import Repair as QR
        k=torch.tensor([[[[0.0],[0.3],[-0.3],[1.5]]]],dtype=torch.float32); v=k*2
        scores=(torch.tensor([[[10.,2.,1.,0.]]]),)
        ml,mq=_Model(1),_Model(1)
        a=LR(ml,((k,v),),scores,None,0.5,1.0,target_length=2,selector='attention',group_mode='full')
        b=QR(mq,((k,v),),scores,None,0.5,1.0,target_length=2,selector='attention',group_mode='full')
        self.assertTrue(torch.equal(a[0][0],b[0][0])); self.assertTrue(torch.equal(a[0][1],b[0][1]))
        self.assertTrue(torch.equal(ml._mymethod_counts[0],mq._mymethod_counts[0]))
        self.assertTrue(torch.equal(ml._mymethod_diameters[0],mq._mymethod_diameters[0]))


    def test_keepkv_score_window_is_independent_from_recent_protection_in_wrapper(self):
        from models.llama.merge_mymethod_llama import Repair
        model=_Model(1)
        k=torch.tensor([[[[0.0],[0.2],[0.4],[1.0]]]],dtype=torch.float32)
        v=k.clone()
        rows=(torch.tensor([[[[4.,3.,2.,1.],[1.,2.,3.,4.]]]],dtype=torch.float32),)
        Repair(
            model,((k,v),),rows,None,ratio=0.5,lambda_mymethod=1.0,target_length=2,
            selector='keepkv',keepkv_sink_size=0,keepkv_recent_protect_size=0,
            keepkv_score_window=2,keepkv_recency_beta=1.0,group_mode='eviction_only',
        )
        self.assertEqual(model._mymethod_score_history[0].shape[-2],2)

    def test_attention_compression_resets_period_importance_state(self):
        from models.llama.merge_mymethod_llama import Repair
        model=_Model(1)
        attn=model.model.layers[0].self_attn
        attn._mymethod_attention_period_importance=torch.tensor([[[7.,6.,5.,4.]]])
        k=torch.tensor([[[[0.0],[0.2],[0.4],[1.0]]]],dtype=torch.float32)
        v=k.clone()
        scores=(attn._mymethod_attention_period_importance.clone(),)
        Repair(model,((k,v),),scores,None,ratio=0.5,lambda_mymethod=1.0,target_length=2,
               selector='attention',group_mode='eviction_only')
        self.assertFalse(hasattr(attn,'_mymethod_attention_period_importance'))

    def test_d2o_llama_qwen_wrapper_parity_and_state_gather(self):
        from models.llama.merge_mymethod_llama import Repair as LR
        from models.qwen.merge_mymethod_qwen import Repair as QR
        k=torch.tensor([[[[0.0],[0.2],[0.4],[2.0]]]],dtype=torch.float32); v=k*3
        cumulative=(torch.tensor([[[10.,8.,2.,1.]]],dtype=torch.float32),)
        ml,mq=_Model(1),_Model(1)
        kwargs=dict(
            ratio=0.5, lambda_mymethod=1.0, target_length=2, selector='d2o',
            d2o_sink_size=0, d2o_recent_size=0,
            d2o_use_log_count=True, d2o_score_reset_interval=0, group_mode='full',
        )
        a=LR(ml,((k,v),),cumulative,None,**kwargs)
        b=QR(mq,((k,v),),cumulative,None,**kwargs)
        self.assertTrue(torch.equal(a[0][0],b[0][0])); self.assertTrue(torch.equal(a[0][1],b[0][1]))
        self.assertTrue(torch.equal(ml._mymethod_counts[0],mq._mymethod_counts[0]))
        self.assertTrue(torch.equal(ml._mymethod_diameters[0],mq._mymethod_diameters[0]))
        li=ml.model.layers[0].self_attn._mymethod_d2o_cumulative_importance
        qi=mq.model.layers[0].self_attn._mymethod_d2o_cumulative_importance
        self.assertTrue(torch.equal(li,qi))
        self.assertTrue(torch.equal(li,torch.tensor([[[10.,8.]]])))


if __name__=='__main__': unittest.main()
