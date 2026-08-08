#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Microbenchmark graph-MIS grouping stages on a captured real KV snapshot.

Boundary excludes model prefill/decoding and selector score collection. It reports:
  * retained/evicted split + anchor-compatible routing;
  * local conflict-graph + parallel MIS;
  * full build_merge_plan;
  * internal GroupLogitKV predecessor and CentroidKV native-core references.
"""
from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
import torch
from models.common import mymethod_common as mm
from models.common.centroidkv_common import CentroidKVConfig, compress_layer as centroid_compress


def bench(fn, device, warmup, repeat):
    for _ in range(warmup): fn()
    if device.type=='cuda': torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    vals=[]
    for _ in range(repeat):
        if device.type=='cuda': torch.cuda.synchronize(device)
        t=time.perf_counter(); fn()
        if device.type=='cuda': torch.cuda.synchronize(device)
        vals.append((time.perf_counter()-t)*1000)
    peak=(torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None)
    return {'median_ms':statistics.median(vals),'mean_ms':statistics.mean(vals),'min_ms':min(vals),'peak_allocated_mib':peak}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--snapshot',required=True); ap.add_argument('--layer_idx',type=int,default=None)
    ap.add_argument('--compress_ratio',type=float,default=0.4); ap.add_argument('--lambda_mymethod',type=float,default=0.5)
    ap.add_argument('--device',default='cuda'); ap.add_argument('--mymethod_graph_backend',choices=['auto','cuda','torch'],default='auto'); ap.add_argument('--mymethod_route_backend',choices=['auto','cuda','torch'],default='auto'); ap.add_argument('--warmup',type=int,default=3); ap.add_argument('--repeat',type=int,default=10)
    ap.add_argument('--output_json',required=True); args=ap.parse_args()
    dev=torch.device(args.device if not args.device.startswith('cuda') or torch.cuda.is_available() else 'cpu')
    obj=torch.load(args.snapshot,map_location='cpu',weights_only=False)
    layer=obj['layers'][0] if args.layer_idx is None else next(x for x in obj['layers'] if int(x['layer_idx'])==args.layer_idx)
    k4=layer['keys'].to(dev); imp4=layer['importance'].to(dev); b,h,L,d=k4.shape; x=b*h
    keep_len=max(1,int(round(L*(1-args.compress_ratio))))
    k=k4.reshape(x,L,d); imp=imp4.reshape(x,L).float(); c=torch.ones((x,L),device=dev); z=torch.zeros_like(c)
    pos=torch.arange(L,device=dev,dtype=torch.int32).view(1,L).expand(x,-1); scale=mm.reference_scale_from_original_keys(k4).reshape(x,1)
    keep=mm.select_keep_indices_attention(imp,keep_len); evict=mm._complement_indices(keep,L)
    rk=mm.gather_seq(k,keep); ek=mm.gather_seq(k,evict); rr=torch.zeros((x,keep_len),device=dev); er=torch.zeros((x,L-keep_len),device=dev)
    rd=torch.zeros_like(rr); ed=torch.zeros_like(er); delta=scale*args.lambda_mymethod; allowed=torch.ones_like(rr,dtype=torch.bool)
    cache={}
    def route_fn():
        out=mm._nearest_compatible_routes(ek,rk,er,rr,ed,rd,delta,allowed,256,True,route_backend=args.mymethod_route_backend); cache['route']=out; return out
    route=route_fn(); best,dist,gamma,compat=route
    def mis_fn():
        return mm._parallel_mis_membership(ek,er,ed,evict,best,compat,gamma,keep_len,delta,'full')
    def full_fn():
        return mm.build_merge_plan(k,c,z,z,imp,pos,keep_len,scale,args.lambda_mymethod,selector='attention',group_mode='full',distance_chunk_size=256,graph_backend=args.mymethod_graph_backend,route_backend=args.mymethod_route_backend)
    def anchor_fn():
        return mm.build_merge_plan(k,c,z,z,imp,pos,keep_len,scale,args.lambda_mymethod,selector='attention',group_mode='anchor_only',distance_chunk_size=256,graph_backend=args.mymethod_graph_backend,route_backend=args.mymethod_route_backend)
    def eviction_fn():
        return mm.build_merge_plan(k,c,z,z,imp,pos,keep_len,scale,args.lambda_mymethod,selector='attention',group_mode='eviction_only',distance_chunk_size=256,graph_backend=args.mymethod_graph_backend,route_backend=args.mymethod_route_backend)
    results={'metadata':{'layer_idx':int(layer['layer_idx']),'length':L,'heads':h,'head_dim':d,'keep_len':keep_len,'lambda_mymethod':args.lambda_mymethod,'device':str(dev),'mymethod_graph_backend':args.mymethod_graph_backend,'mymethod_route_backend':args.mymethod_route_backend}}
    plan0=full_fn()
    total=max(int(plan0.route_compatible.numel()),1)
    routed=int(plan0.route_compatible.sum().item())
    accepted=int(plan0.accepted.sum().item())
    results['grouping_diagnostics']={
        'route_compatible_ratio':routed/total,
        'accepted_entry_ratio':accepted/total,
        'mis_accept_given_routed_ratio':accepted/max(routed,1),
        'max_candidate_pool_size':int(plan0.candidate_pool_sizes.max().item()),
        'max_selected_candidates_per_group':int(plan0.selected_per_anchor.max().item()),
        'total_conflict_edges':int(plan0.conflict_edges_per_anchor.sum().item()),
        'max_actual_mis_rounds':int(plan0.mis_rounds_per_anchor.max().item()),
    }
    results['anchor_route']=bench(route_fn,dev,args.warmup,args.repeat)
    results['conflict_graph_plus_parallel_mis']=bench(mis_fn,dev,args.warmup,args.repeat)
    results['eviction_only_plan']=bench(eviction_fn,dev,args.warmup,args.repeat)
    results['anchor_only_plan']=bench(anchor_fn,dev,args.warmup,args.repeat)
    results['full_mymethod_plan']=bench(full_fn,dev,args.warmup,args.repeat)
    # Lightweight references with different native boundaries. GroupLogitKV is an
    # internal unpublished predecessor, not an external published baseline.
    try:
        from models.llama.merge_grouplogitkv_llama import _batched_group_merge_layer
        v=k4.clone(); cc=torch.ones((b,h,L),device=dev)
        results['GroupLogitKV_native_merge_core']=bench(lambda:_batched_group_merge_layer(k4,v,cc,imp4,args.compress_ratio,0.8,target_length=keep_len),dev,args.warmup,args.repeat)
    except Exception as e: results['GroupLogitKV_native_merge_core']={'error':repr(e)}
    try:
        cfg=CentroidKVConfig(); v=k4.clone(); cc=torch.ones((b,h,L),device=dev,dtype=torch.int32)
        results['CentroidKV_native_merge_core']=bench(lambda:centroid_compress(k4,v,cc,keep_len,cfg),dev,args.warmup,args.repeat)
    except Exception as e: results['CentroidKV_native_merge_core']={'error':repr(e)}
    Path(args.output_json).parent.mkdir(parents=True,exist_ok=True); Path(args.output_json).write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
