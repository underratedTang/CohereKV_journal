#!/usr/bin/env python3
"""Correctness gate for the independent mymethod CUDA backend.

The CUDA backend is compared against the PyTorch reference on boundary-free
synthetic cases and, optionally, on captured real KV snapshots.  Real-snapshot
mismatches are reported separately because FP32 GEMM and direct-distance CUDA
kernels may round threshold-adjacent pairs differently; both outputs are still
checked for certificate safety.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from models.common import mymethod_common as mm


def plan_payload(plan):
    return {
        'routed': int(plan.route_compatible.sum().item()),
        'accepted': int(plan.accepted.sum().item()),
        'max_pool': int(plan.candidate_pool_sizes.max().item()) if plan.candidate_pool_sizes.numel() else 0,
        'conflict_edges': int(plan.conflict_edges_per_anchor.sum().item()),
        'max_rounds': int(plan.mis_rounds_per_anchor.max().item()) if plan.mis_rounds_per_anchor.numel() else 0,
        'max_group_gamma_over_delta': float((plan.group_gamma_max / plan.delta.clamp_min(1e-12)).max().item()),
    }


def compare_plans(keys4, imp4, keep_ratio, lam, tag):
    b,h,L,d=keys4.shape; x=b*h; keep=max(1,min(L-1,int(round(L*keep_ratio))))
    keys=keys4.reshape(x,L,d)
    imp=imp4.reshape(x,L).float()
    c=torch.ones((x,L),device=keys.device)
    z=torch.zeros((x,L),device=keys.device)
    p=torch.arange(L,device=keys.device,dtype=torch.int32).view(1,L).expand(x,-1)
    scale=mm.reference_scale_from_original_keys(keys4).reshape(x,1)
    kwargs=dict(keys=keys,counts=c,radii=z,diameters=z,importance=imp,positions=p,
                keep_len=keep,reference_scale=scale,lambda_mymethod=lam,
                selector='attention',group_mode='full',distance_chunk_size=256)
    pt=mm.build_merge_plan(**kwargs,route_backend='torch',graph_backend='torch')
    cu=mm.build_merge_plan(**kwargs,route_backend='cuda',graph_backend='cuda')
    fields=['keep_idx','evict_idx','route_compatible','best_idx','accepted','candidate_pool_sizes','selected_per_anchor','conflict_edges_per_anchor']
    mismatch={}
    for name in fields:
        a=getattr(pt,name); b_=getattr(cu,name)
        mismatch[name]=int((a!=b_).sum().item())
    mismatch['anchor_gamma_max_abs']=float((pt.anchor_gamma-cu.anchor_gamma).abs().max().item()) if pt.anchor_gamma.numel() else 0.0
    mismatch['group_gamma_max_abs']=float((pt.group_gamma_max-cu.group_gamma_max).abs().max().item()) if pt.group_gamma_max.numel() else 0.0
    # CUDA full groups must satisfy the method certificate irrespective of exact
    # membership parity at threshold-adjacent FP32 pairs.
    nonempty=cu.selected_per_anchor>0
    safe=True
    if bool(nonempty.any().item()):
        safe=bool((cu.group_gamma_max[nonempty] <= cu.delta.expand_as(cu.group_gamma_max)[nonempty] + 5e-5).all().item())
    return {'tag':tag,'torch':plan_payload(pt),'cuda':plan_payload(cu),'mismatch':mismatch,'cuda_certificate_safe':safe}


def synthetic(device):
    torch.manual_seed(1234)
    X,L,D=4,96,32
    # Clustered data gives non-trivial compatible pools while remaining well
    # away from exact threshold ties.
    centers=torch.randn(X,8,D,device=device)
    assign=torch.arange(L,device=device)%8
    keys=centers[:,assign,:] + 0.08*torch.randn(X,L,D,device=device)
    keys4=keys.view(1,X,L,D).to(torch.float16)
    imp=torch.rand(1,X,L,device=device)
    return compare_plans(keys4,imp,0.6,0.35,'synthetic_clustered')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--snapshot',default=None)
    ap.add_argument('--lambda_mymethod',type=float,default=0.75)
    ap.add_argument('--compress_ratio',type=float,default=0.4)
    ap.add_argument('--output_json',required=True)
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    try:
        import mymethod_cuda
    except Exception as e:
        raise RuntimeError('Install /home/tlxr/SimCalKV/mymethod_cuda_src first') from e
    if not mymethod_cuda.is_available():
        raise RuntimeError(f'mymethod_cuda unavailable: {mymethod_cuda.extension_error()!r}')
    out={'synthetic':synthetic(torch.device('cuda')),'real':[]}
    # Synthetic boundary-free parity is a hard correctness gate.
    bad=sum(v for k,v in out['synthetic']['mismatch'].items() if isinstance(v,int))
    if bad or not out['synthetic']['cuda_certificate_safe']:
        raise AssertionError(f'synthetic CUDA/reference mismatch: {out["synthetic"]}')
    if args.snapshot:
        obj=torch.load(args.snapshot,map_location='cpu',weights_only=False)
        for layer in obj['layers']:
            keys=layer['keys'].to('cuda')
            imp=layer['importance'].to('cuda')
            r=compare_plans(keys,imp,1.0-args.compress_ratio,args.lambda_mymethod,f'layer_{int(layer["layer_idx"])}')
            out['real'].append(r)
            if not r['cuda_certificate_safe']:
                raise AssertionError(f'CUDA certificate failure: {r}')
    Path(args.output_json).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
