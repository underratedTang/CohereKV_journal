#!/usr/bin/env python3
"""Decompose routing and Graph-MIS backend performance on real KV snapshots."""
from __future__ import annotations
import argparse,json,statistics,time
from pathlib import Path
import torch
from models.common import mymethod_common as mm


def bench(fn,device,warmup,repeat):
    for _ in range(warmup): fn()
    if device.type=='cuda':
        torch.cuda.synchronize(device); base=torch.cuda.memory_allocated(device); torch.cuda.reset_peak_memory_stats(device)
    else: base=0
    vals=[]
    for _ in range(repeat):
        if device.type=='cuda': torch.cuda.synchronize(device)
        t=time.perf_counter(); fn()
        if device.type=='cuda': torch.cuda.synchronize(device)
        vals.append((time.perf_counter()-t)*1e3)
    peak=(torch.cuda.max_memory_allocated(device)-base)/2**20 if device.type=='cuda' else None
    return {'median_ms':statistics.median(vals),'mean_ms':statistics.mean(vals),'min_ms':min(vals),'peak_extra_mib':peak}


def one(layer,lam,ratio,device,warmup,repeat):
    k4=layer['keys'].to(device); imp4=layer['importance'].to(device); b,h,L,d=k4.shape; x=b*h
    keep=max(1,min(L-1,int(round(L*(1-ratio)))))
    k=k4.reshape(x,L,d); imp=imp4.reshape(x,L).float(); c=torch.ones((x,L),device=device); z=torch.zeros_like(c)
    pos=torch.arange(L,device=device,dtype=torch.int32).view(1,L).expand(x,-1); scale=mm.reference_scale_from_original_keys(k4).reshape(x,1)
    keep_idx=mm.select_keep_indices_attention(imp,keep); ev=mm._complement_indices(keep_idx,L)
    rk=mm.gather_seq(k,keep_idx); ek=mm.gather_seq(k,ev); rr=torch.zeros((x,keep),device=device); er=torch.zeros((x,L-keep),device=device)
    rd=torch.zeros_like(rr); ed=torch.zeros_like(er); delta=scale*lam; allowed=torch.ones_like(rr,dtype=torch.bool)
    def route(backend): return mm._nearest_compatible_routes(ek,rk,er,rr,ed,rd,delta,allowed,256,True,route_backend=backend)
    route_ref=route('torch'); best,dist,gamma,compat=route_ref
    def graph(backend): return mm._parallel_mis_membership(ek,er,ed,ev,best,compat,gamma,keep,delta,'full',graph_backend=backend)
    def plan(rb,gb): return mm.build_merge_plan(k,c,z,z,imp,pos,keep,scale,lam,selector='attention',group_mode='full',distance_chunk_size=256,route_backend=rb,graph_backend=gb)
    result={'metadata':{'layer_idx':int(layer['layer_idx']),'length':L,'head_dim':d,'rows':x,'keep_len':keep,'lambda':lam,'ratio':ratio}}
    for bname in ('torch','cuda'):
        result[f'route_{bname}']=bench(lambda bname=bname:route(bname),device,warmup,repeat)
        result[f'graph_{bname}']=bench(lambda bname=bname:graph(bname),device,warmup,repeat)
    for rb,gb in [('torch','torch'),('cuda','torch'),('torch','cuda'),('cuda','cuda')]:
        result[f'plan_route_{rb}_graph_{gb}']=bench(lambda rb=rb,gb=gb:plan(rb,gb),device,warmup,repeat)
    p=plan('cuda','cuda')
    result['cuda_diagnostics']={'routed':int(p.route_compatible.sum().item()),'accepted':int(p.accepted.sum().item()),
        'route_ratio':float(p.route_compatible.float().mean().item()),'accept_ratio':float(p.accepted.float().mean().item()),
        'max_pool':int(p.candidate_pool_sizes.max().item()),'conflict_edges':int(p.conflict_edges_per_anchor.sum().item()),
        'max_rounds':int(p.mis_rounds_per_anchor.max().item())}
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--snapshot',required=True); ap.add_argument('--lambdas',default='0.5,0.75'); ap.add_argument('--compress_ratio',type=float,default=0.4)
    ap.add_argument('--device',default='cuda'); ap.add_argument('--warmup',type=int,default=3); ap.add_argument('--repeat',type=int,default=10); ap.add_argument('--output_json',required=True); args=ap.parse_args()
    dev=torch.device(args.device); obj=torch.load(args.snapshot,map_location='cpu',weights_only=False); ls=[float(x) for x in args.lambdas.split(',') if x.strip()]
    out=[]
    for layer in obj['layers']:
        for lam in ls: out.append(one(layer,lam,args.compress_ratio,dev,args.warmup,args.repeat))
    Path(args.output_json).parent.mkdir(parents=True,exist_ok=True); Path(args.output_json).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
