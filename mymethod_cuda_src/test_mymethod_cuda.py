#!/usr/bin/env python3
from __future__ import annotations
import math
import sys
import torch

from mymethod_cuda import count_calibrated_qk, compatible_route, graph_mis_select, is_available, extension_error


def require_cuda():
    if not torch.cuda.is_available():
        print('SKIP: CUDA is not available')
        sys.exit(0)
    if not is_available():
        raise RuntimeError(f'mymethod_cuda extension is unavailable: {extension_error()!r}')


def test_qk(dtype):
    B,H,K,D=2,4,257,128
    q=torch.randn(B,H,1,D,device='cuda',dtype=dtype)
    k=torch.randn(B,H,K,D,device='cuda',dtype=dtype)
    logc=torch.randn(B,H,1,193,device='cuda',dtype=dtype)*0.1
    out=count_calibrated_qk(q,k,logc)
    ref=torch.matmul(q.float(),k.float().transpose(-1,-2))/math.sqrt(D)
    ref[...,:logc.shape[-1]] += logc.float()
    err=(out.float()-ref).abs().max().item()
    tol=3e-2 if dtype==torch.bfloat16 else 2e-2
    assert err < tol,(dtype,err)
    print(f'QK parity {dtype}: max_abs={err:.6g}')


def test_route(dtype):
    torch.manual_seed(2)
    X,N,M,D=3,17,23,32
    ek=torch.randn(X,N,D,device='cuda',dtype=dtype)
    rk=torch.randn(X,M,D,device='cuda',dtype=dtype)
    er=torch.rand(X,N,device='cuda')*0.05
    rr=torch.rand(X,M,device='cuda')*0.05
    ed=torch.rand(X,N,device='cuda')*0.05
    rd=torch.rand(X,M,device='cuda')*0.05
    delta=torch.full((X,1),20.0,device='cuda')
    allowed=torch.ones(X,M,device='cuda',dtype=torch.bool)
    idx,dist,gamma,ok=compatible_route(ek,rk,er,rr,ed,rd,delta,allowed)
    dist2=(ek.float()[:,:,None,:]-rk.float()[:,None,:,:]).square().sum(-1)
    tol=8*torch.finfo(torch.float32).eps*torch.maximum(delta.abs(),torch.ones_like(delta))
    margin=delta[:,None,:]-er[:,:,None]-rr[:,None,:]-tol[:,None,:]
    feas=(ed[:,:,None]<=delta[:,None,:])&(rd[:,None,:]<=delta[:,None,:])&(margin>=0)&(dist2<=margin.clamp_min(0).square())
    masked=dist2.masked_fill(~feas,float('inf'))
    refd2,refidx=masked.min(-1); refok=torch.isfinite(refd2)
    assert torch.equal(ok,refok)
    assert torch.equal(idx[ok],refidx[ok])
    assert torch.allclose(dist[ok],refd2[ok].sqrt(),rtol=1e-4,atol=2e-3)
    print(f'route parity {dtype}: compatible={int(ok.sum())}/{ok.numel()}')


def test_graph_small(dtype):
    # One pool with a path-like conflict structure.  The exact result must be an
    # independent maximal set and carry a valid group certificate.
    keys=torch.tensor([[[-0.4,0.0],[0.0,0.0],[0.4,0.0]]],device='cuda',dtype=dtype)
    r=torch.zeros((1,3),device='cuda'); d=torch.zeros_like(r)
    phys=torch.tensor([[10,11,12]],device='cuda',dtype=torch.long)
    best=torch.zeros((1,3),device='cuda',dtype=torch.long)
    compat=torch.ones((1,3),device='cuda',dtype=torch.bool)
    anchor_gamma=torch.tensor([[0.4,0.0,0.4]],device='cuda')
    delta=torch.tensor([[0.6]],device='cuda')
    accepted,gg,pools,selected,edges,rounds=graph_mis_select(keys,r,d,phys,best,compat,anchor_gamma,1,delta,'full')
    assert int(pools.item())==3
    assert int(selected.item())==2
    assert int(edges.item())==1
    assert int(rounds.item())>=1
    ids=accepted[0].nonzero().flatten().tolist()
    for a in range(len(ids)):
        for b in range(a+1,len(ids)):
            assert torch.linalg.vector_norm(keys[0,ids[a]].float()-keys[0,ids[b]].float()).item() <= 0.6+1e-5
    assert float(gg.item()) <= 0.6+1e-5
    print(f'graph small {dtype}: selected={ids}, gamma={float(gg.item()):.6g}')


def test_graph_large_exact(dtype):
    # Exercises the no-cap exact path (>256). Equal keys have no conflicts, so
    # every routed candidate must be selected in one MIS round.
    N=300
    keys=torch.zeros((1,N,1),device='cuda',dtype=dtype)
    z=torch.zeros((1,N),device='cuda')
    phys=torch.arange(N,device='cuda',dtype=torch.long).view(1,N)
    best=torch.zeros((1,N),device='cuda',dtype=torch.long)
    compat=torch.ones((1,N),device='cuda',dtype=torch.bool)
    gamma=torch.zeros((1,N),device='cuda')
    delta=torch.ones((1,1),device='cuda')
    accepted,gg,pools,selected,edges,rounds=graph_mis_select(keys,z,z,phys,best,compat,gamma,1,delta,'full')
    assert int(accepted.sum().item())==N
    assert int(pools.item())==N and int(selected.item())==N
    assert int(edges.item())==0 and int(rounds.item())==1
    print('graph large exact: passed N=300')


def main():
    require_cuda()
    for dtype in (torch.float16, torch.bfloat16):
        test_qk(dtype); test_route(dtype); test_graph_small(dtype)
    test_graph_large_exact(torch.float16)
    print('ALL mymethod_cuda smoke tests passed')

if __name__=='__main__': main()
