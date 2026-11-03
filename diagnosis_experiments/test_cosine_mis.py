#!/usr/bin/env python3
"""CPU tests for cosine R-E routing + original graph MIS + three fusion arms.

Run from project root: python diagnosis_experiments/test_cosine_mis.py
No pretrained model, dataset download or GPU required.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from cosine_mis_plan import select_split,cosine_top1,make_plan
from cosine_mis_fusion import fuse_plan,METHODS
from real_query_capture import replay_prefix_attention,rotate_queries,output_errors


def test_cosine_routing_independent_of_l2():
    # 20x Euclidean displacement is valid because cosine=1. No R-E L2 gate.
    keys=torch.tensor([[[1.,0.],[21.,0.],[-1.,0.],[0.,1.]]],dtype=torch.float32)
    keep=torch.tensor([[0,3]]);evict=torch.tensor([[1,2]])
    best,cs,valid,dists=cosine_top1(keys,keep,evict,0.85,chunk_size=1)
    assert valid.tolist()==[[True,False]]
    assert best[0,0].item()==0 and abs(cs[0,0].item()-1)<1e-6
    assert abs(dists[0,0].item()-20.)<1e-5
    b2,c2,v2,d2=cosine_top1(keys,keep,evict,0.85,chunk_size=256)
    assert torch.equal(best,b2) and torch.equal(cs,c2) and torch.equal(valid,v2) and torch.equal(dists,d2)
    assert not v2[0,1]


def test_native_mis_independent_ee():
    # A=0, B/C both cosine-compatible but far apart along y.
    # A second retained anchor 4 is for the retained/evicted partition.
    keys=torch.tensor([[[10.,0.],[10.,1.],[10.,-1.],[20.,0.],[0.,10.]]],dtype=torch.float32)
    imp=torch.tensor([[100.,0.,0.,0.,90.]])
    pos=torch.arange(5)[None]
    keep,evict=select_split(keys,imp,pos,keep_len=2,sink_size=1,recent_window=1)
    assert keep.tolist()==[[0,4]] and evict.tolist()==[[1,2,3]]
    best,cs,routed,dists=cosine_top1(keys,keep,evict,.85)
    assert routed.tolist()==[[True,True,True]]
    assert best.tolist()==[[0,0,0]]  # same anchor regardless of EE cutoff
    assert float(dists[0,2])>=10-1e-5
    lo=make_plan(keys,keep,evict,best,cs,routed,dists,torch.tensor([[1.]]),.5)
    hi=make_plan(keys,keep,evict,best,cs,routed,dists,torch.tensor([[1.]]),2.5)
    assert torch.equal(lo.route_compatible,hi.route_compatible)
    assert torch.equal(lo.best_idx,hi.best_idx)
    assert (lo.accepted & ~lo.route_compatible).sum()==0
    assert int(lo.accepted.sum())<int(hi.accepted.sum())
    assert int(lo.conflict_edges_per_anchor.sum())>int(hi.conflict_edges_per_anchor.sum())
    v=np.arange(15,dtype=np.float32).reshape(5,3)
    results=[]
    for p in [lo,hi]:
        t=fuse_plan(keys[0].numpy(),v,keep[0].numpy(),evict[0].numpy(),
                    best[0].numpy(),routed[0].numpy(),p.accepted[0].numpy(),
                    p.candidate_pool_sizes[0].numpy(),
                    p.selected_per_anchor[0].numpy(),
                    p.conflict_edges_per_anchor[0].numpy())
        assert t.n_mis_accepted+t.n_mis_rejected==t.n_routed==3
        assert np.array_equal(t.counts,[4.,1.])
        assert np.allclose(t.values[0],v[:4].mean(axis=0))
        assert np.array_equal(t.keys['all_to_anchor'],keys[0,keep[0]].numpy())
        results.append(t)
    assert np.array_equal(results[0].keys['merge_all'],results[1].keys['merge_all'])
    assert np.array_equal(results[0].keys['all_to_anchor'],results[1].keys['all_to_anchor'])
    assert not np.array_equal(results[0].keys['mis_to_anchor'],results[1].keys['mis_to_anchor'])


def test_mis_maximal_and_native_stable():
    torch.manual_seed(2026)
    for n in (8,16,64):
        k=torch.randn((1,n,8),dtype=torch.float32)
        keep=torch.tensor([[0,1]])
        evict=torch.arange(2,n)[None]
        best,cs,routed,dists=cosine_top1(k,keep,evict,-1.)
        ee=1.2
        for ee in (1.2,2.,3.5):
            p=make_plan(k,keep,evict,best,cs,routed,dists,torch.tensor([[1.]]),ee)
            q=make_plan(k,keep,evict,best,cs,routed,dists,torch.tensor([[1.]]),ee)
            assert torch.equal(p.accepted,q.accepted)
            for anchor in range(2):
                ids=torch.where(routed[0] & (best[0]==anchor))[0]
                chosen=ids[p.accepted[0,ids]]
                if len(chosen)>1:
                    dis=torch.cdist(k[0,evict[0,chosen]],k[0,evict[0,chosen]])
                    assert float(dis.max())<=ee+1e-4, (n,ee,float(dis.max()))
                # A maximal independent set: each not chosen conflicts with at least one chosen
                for j in ids[~p.accepted[0,ids]].tolist():
                    if len(chosen):
                        d=torch.cdist(k[0,evict[0,j]][None],k[0,evict[0,chosen]])
                        assert bool(torch.any(d>ee-1e-5))


def test_fusion_formula_and_replay():
    k=np.array([[0.,0.],[1.,0.],[0.,1.],[-1.,0.],[4.,2.],[0.,-2.]],np.float32)
    v=np.arange(18,dtype=np.float32).reshape(6,3)
    keep=np.array([0,4]);evict=np.array([1,2,3,5])
    best=np.array([0,0,0,1]);routed=np.array([True]*4)
    accepted=np.array([True,True,False,False])
    rr=fuse_plan(k,v,keep,evict,best,routed,accepted,[3,1],[2,0],[2,0])
    assert rr.n_routed==4 and rr.n_mis_accepted==2 and rr.n_mis_rejected==2
    assert np.array_equal(rr.counts,[4.,2.])
    assert np.allclose(rr.keys['merge_all'][0],k[[0,1,2,3]].mean(axis=0))
    assert np.allclose(rr.keys['mis_to_anchor'][0],(2*k[0]+k[1]+k[2])/4)
    assert np.array_equal(rr.keys['all_to_anchor'],k[keep])
    assert np.allclose(rr.values[0],v[[0,1,2,3]].mean(axis=0))
    assert np.allclose(rr.values[1],v[[4,5]].mean(axis=0))
    q=torch.tensor([[[1.,0.],[0.,1.]]]).repeat(4,1,1)
    key=np.array([[1.,0.],[0.,1.]],np.float32)
    value=np.array([[1.,1.],[2.,3.]],np.float32)
    fk=torch.tensor([[.1,.1],[.3,.4]])
    fv=torch.tensor([[1.,2.],[3.,4.]])
    a=replay_prefix_attention(q,key,value,[1,1],fk,fv,torch.device('cpu'),1e-8)
    b=replay_prefix_attention(q,key,value,[2,1],fk,fv,torch.device('cpu'),1e-8)
    assert a.shape==(4,2,2) and not torch.allclose(a,b)
    met,per=output_errors(a,a)
    assert met['output_rel_error']==0 and per.shape==(2,)
    angle=torch.tensor([[[.2,.2,.2,.2]]])
    rot=rotate_queries(torch.ones(1,1,4),angle.cos(),angle.sin(),1,4)
    assert rot.shape==(1,1,4)


if __name__=='__main__':
    test_cosine_routing_independent_of_l2()
    test_native_mis_independent_ee()
    test_mis_maximal_and_native_stable()
    test_fusion_formula_and_replay()
    print('[PASS] cosine-only R-E with distant high-cosine pairs; original graph MIS independent EE threshold; maximal independent sets; exact controls & KV Count; real-Q replay')


def test_frozen_median_calibration_reference():
    import json
    from calibrate_median_reference import aggregate_median_scales,validate_payload
    scales=[np.array([[1.,2.],[3.,4.]]),
            np.array([[1.5,6.],[4.,4.]]),
            np.array([[20.,8.],[5.,30.]]),
            np.array([[2.,10.],[7.,50.]])]
    med,mean,std,stack=aggregate_median_scales(scales)
    exact=np.median(np.stack(scales),axis=0)
    assert med.shape==(2,2) and np.allclose(med.numpy(),exact)
    assert not np.allclose(med.numpy(),np.sqrt(np.mean(np.stack(scales)**2,axis=0)))
    payload=dict(schema='diagnosis_median_reference_v1',
        aggregation='median_across_sequences',frozen=True,model='example/model',
        dataset='wikitext2',calibration_split='train',
        calibration_num_sequences=4,calibration_seqlen=64,calibration_seed=42,
        num_layers=2,num_kv_heads=2,
        layer_head_scales=med.numpy().tolist(),
        sequence_layer_head_scales=stack.numpy().tolist())
    got=validate_payload(payload,model='example/model',split='train',num_sequences=4,seqlen=64,seed=42)
    assert np.array_equal(got,exact)
    assert got[0,0]!=got[0,1] and got[0,0]!=got[1,0]
    # Operational delta is independent of evaluated sequence and Budget.
    assert np.array_equal(got*.5,validate_payload(payload,model='example/model')*.5)
    for k,v in [('aggregation','S_lh_calib=sqrt(mean(S_m^2))'),('model','wrong/model'),
                ('calibration_split','test'),('layer_head_scales',[[1.,1.],[1.,1.]])]:
        changed=dict(payload);changed[k]=v
        try: validate_payload(changed,model='example/model',split='train')
        except ValueError:pass
        else:raise AssertionError('Missing fail-closed Median calibration validation for '+k)
    odd,_,_,_=aggregate_median_scales([np.ones((2,2)),np.ones((2,2))*4,np.ones((2,2))*2])
    assert np.array_equal(odd.numpy(),np.ones((2,2))*2)


if __name__=='__main__':
    test_frozen_median_calibration_reference()
    print('[PASS] frozen per-layer/head MEDIAN across independent calibration sequences; RMS exclusion; wrong-model/fake-reference refusal')
