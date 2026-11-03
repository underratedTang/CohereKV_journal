#!/usr/bin/env python3
"""No pretrained model, CUDA or download needed; tests dual routing and E-E independence."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from types import SimpleNamespace
import numpy as np
import torch
from cosine_mis_plan import cosine_top1,l2_top1,route_top1,make_plan
from cosine_mis_fusion import fuse_plan
from routing_sweep_utils import make_routing_specs,route_spec_cutoff,summarize


def test_routing_independence():
    # One very high-cosine yet physically distant Evicted Key.
    keys=torch.tensor([[[1.,0.],[21.,0.],[1.2,0.1],[0.,1.],[0.1,1.1]]])
    keep=torch.tensor([[0,3]])
    evict=torch.tensor([[1,2,4]])
    cidx,cs,cmask,cdist=cosine_top1(keys,keep,evict,.9,chunk_size=1)
    lidx,ls,lmask,ldist=l2_top1(keys,keep,evict,.5,chunk_size=1)
    assert bool(cmask[0,0]) and not bool(lmask[0,0])
    assert cidx[0,0].item()==lidx[0,0].item()==0
    assert lmask.tolist()==[[False,True,True]]
    assert float(cdist[0,0])==20.
    assert lidx.tolist()==[[0,0,1]]
    for mode,threshold in [('cosine',.9),('l2',.5)]:
        actual=route_top1(keys,keep,evict,mode,threshold,1)
        expected=cosine_top1(keys,keep,evict,threshold,100) if mode=='cosine' else l2_top1(keys,keep,evict,threshold,100)
        for a,b in zip(actual,expected):assert torch.allclose(a,b)
    # L2 uses physical distance, never applies cosine threshold.
    orthogonal=torch.tensor([[[0.,0.],[0.,1.],[.05,0.]]])
    kidx,cc,mask,dist=l2_top1(orthogonal,torch.tensor([[0,1]]),torch.tensor([[2]]),.1)
    assert mask.tolist()==[[True]] and kidx[0,0].item()==0
    assert cc[0,0].item()==0


def test_sweeps_reuse_and_mis_independence():
    args=SimpleNamespace(routing_modes='cosine,l2',cosine_threshold=None,
        cosine_thresholds='0.8,0.95',routing_l2_lambdas='0.3,0.8',
        routing_l2_distances='0.4')
    specs=make_routing_specs(args)
    assert len(specs)==5 and specs[0]==('cosine','cosine',.8)
    assert route_spec_cutoff(('l2','lambda',.8),10.)==8.
    assert route_spec_cutoff(('l2','distance',.4),10.)==.4
    for bad in ('cosine,cosine','bad'):
        try:make_routing_specs(SimpleNamespace(**{**vars(args),'routing_modes':bad}))
        except ValueError:pass
        else:raise AssertionError('Invalid routing mode accepted')

    torch.manual_seed(42)
    keys=torch.randn(1,22,4)
    keep=torch.tensor([[0,1,2,3]])
    evict=torch.arange(4,22)[None]
    reference=torch.tensor([[3.0]])
    for mode in ('cosine','l2'):
        configs=[(x,route_spec_cutoff(x,3.)) for x in specs if x[0]==mode]
        broad=min(c for _,c in configs) if mode=='cosine' else max(c for _,c in configs)
        base=route_top1(keys,keep,evict,mode,broad,3)
        for spec,cutoff in configs:
            best,cos,routed,dist=route_top1(keys,keep,evict,mode,cutoff,5)
            reused=base[2] & (base[1]>=cutoff if mode=='cosine' else base[3]<=cutoff)
            assert torch.equal(reused,routed)
            assert torch.equal(base[0][routed],best[routed])
            assert torch.allclose(base[1][routed],cos[routed],atol=1e-5)
            assert torch.allclose(base[3][routed],dist[routed],atol=1e-5)
            # MIS thresholds do not affect routing membership or anchor ids.
            lo=make_plan(keys,keep,evict,best,cos,routed,dist,reference,.3,'cpu_batched')
            hi=make_plan(keys,keep,evict,best,cos,routed,dist,reference,1.5,'cpu_batched')
            assert torch.equal(lo.route_compatible,hi.route_compatible)
            assert torch.equal(lo.best_idx,hi.best_idx)
            assert int(lo.candidate_pool_sizes.sum())==int(routed.sum())
            assert int(hi.candidate_pool_sizes.sum())==int(routed.sum())
            assert int(lo.accepted.sum())<=int(hi.accepted.sum())
            # Check returned masks vs unmodified torch implementation.
            native=make_plan(keys,keep,evict,best,cos,routed,dist,reference,.3,'torch')
            assert torch.equal(native.accepted,lo.accepted)
            assert torch.equal(native.conflict_edges_per_anchor,lo.conflict_edges_per_anchor)
            values=np.arange(22*3,dtype=np.float32).reshape(22,3)
            tri=fuse_plan(keys[0].numpy(),values,keep[0].numpy(),evict[0].numpy(),
                best[0].numpy(),routed[0].numpy(),lo.accepted[0].numpy(),
                lo.candidate_pool_sizes[0].numpy(),lo.selected_per_anchor[0].numpy(),
                lo.conflict_edges_per_anchor[0].numpy())
            assert np.all(tri.counts>=1)
            assert int(tri.counts.sum())==len(keep[0])+int(routed.sum())
            assert tri.n_mis_accepted+tri.n_mis_rejected==tri.n_routed
    # Summary MUST keep routing settings separate, not mix samples.
    sample=[]
    for mode,typ,param in specs:
        for method in ('merge_all','mis_to_anchor','all_to_anchor'):
            row=dict(routing_mode=mode,routing_parameter_type=typ,routing_parameter=param,
                routing_l2_cutoff=(3*param if typ=='lambda' else (param if typ=='distance' else 0)),
                budget=.2,ee_lambda=.5,method=method,n_mis_rejected=1)
            for field in ('output_rel_error','output_rel_p90','output_abs_error','output_rmse',
                          'delta_rel_from_merge_all','relative_improvement_pct','probe_win_fraction',
                          'n_routed','n_mis_accepted','n_affected_groups',
                          'n_native_conflict_edges','reconstruction_sse_original_key',
                          'mean_anchor_drift','mean_routed_cosine'):
                row[field]=.5
            sample.append(row)
    summary=summarize(sample)
    assert len(summary)==3*len(specs)
    assert set((r['routing_mode'],r['routing_parameter_type'],r['routing_parameter']) for r in summary)==set(specs)

if __name__=='__main__':
    test_routing_independence()
    test_sweeps_reuse_and_mis_independence()
    print('[PASS] independent Cosine/L2 Top1 sweeps; lambda/absolute L2; native MIS exact parity; triad V/Count; routing-aware summaries')
