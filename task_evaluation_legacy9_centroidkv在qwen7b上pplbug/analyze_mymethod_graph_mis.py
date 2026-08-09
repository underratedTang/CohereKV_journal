#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight real-KV diagnostic for graph-MIS mymethod.

It uses a captured snapshot and asks the questions needed before expensive
PPL/LongBench runs:
  1. Does pair-compatible routing find nontrivial merge candidates?
  2. Does graph MIS still accept a nontrivial fraction (not eviction-only)?
  3. How many anchor-only groups are internally conflicting?
  4. Does full graph-MIS keep exact first-round group diameter <= lambda*S_ref?
  5. How large are candidate pools / conflict graphs / MIS round counts?
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import torch

from models.common.mymethod_common import build_merge_plan, reference_scale_from_original_keys


def parse_floats(spec: str):
    return [float(x) for x in spec.split(',') if x.strip()]


def exact_group_stats(keys, plan, lam):
    x, _, _ = keys.shape
    full_ratios=[]; anchor_ratios=[]
    full_vio=anchor_vio=full_groups=anchor_groups=multi_full=0
    # Both plans share keep/evict/routing; caller supplies plan mode separately.
    for row in range(x):
        scale=float(plan.scale[row,0])
        delta=float(plan.delta[row,0])
        for j in range(plan.keep_idx.shape[1]):
            cand=torch.nonzero(plan.accepted[row] & (plan.best_idx[row]==j),as_tuple=False).flatten()
            if cand.numel()==0: continue
            inds=[int(plan.keep_idx[row,j])]+[int(plan.evict_idx[row,c]) for c in cand]
            kk=keys[row,inds].float()
            diam=float(torch.cdist(kk,kk).max())
            ratio=diam/max(scale,1e-12)
            full_ratios.append(ratio); full_groups+=1
            if len(inds)>2: multi_full+=1
            if diam>delta+1e-4*max(1.0,delta): full_vio+=1
    return dict(
        exact_groups=full_groups,
        exact_multi_candidate_groups=multi_full,
        exact_violation_groups=full_vio,
        exact_violation_ratio=full_vio/max(full_groups,1),
        normalized_diameter_mean=(sum(full_ratios)/len(full_ratios) if full_ratios else 0.0),
        normalized_diameter_max=(max(full_ratios) if full_ratios else 0.0),
    )



def _maximum_independent_set_size(conflict: torch.Tensor) -> int:
    """Exact MIS size for a tiny CPU Boolean conflict matrix (diagnostic only)."""
    n = int(conflict.shape[0])
    if n == 0:
        return 0
    rows = []
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and bool(conflict[i, j]):
                mask |= 1 << j
        rows.append(mask)
    best = 0
    def dfs(candidates: int, chosen: int):
        nonlocal best
        if chosen + candidates.bit_count() <= best:
            return
        if candidates == 0:
            best = max(best, chosen)
            return
        # Branch on a high-degree remaining vertex to prune quickly.
        verts = [i for i in range(n) if (candidates >> i) & 1]
        v = max(verts, key=lambda i: (rows[i] & candidates).bit_count())
        bit = 1 << v
        dfs(candidates & ~bit & ~rows[v], chosen + 1)
        dfs(candidates & ~bit, chosen)
    dfs((1 << n) - 1, 0)
    return best


def mis_oracle_stats(keys, plan, max_pool: int, max_groups: int):
    """Compare degree-priority maximal MIS with exact maximum IS on small first-round pools."""
    if max_pool <= 0 or max_groups <= 0:
        return {}
    considered = optimal = 0
    selected_total = optimum_total = 0
    max_gap = 0
    x = keys.shape[0]
    for row in range(x):
        delta = float(plan.delta[row, 0])
        for j in range(plan.keep_idx.shape[1]):
            pool = torch.nonzero(
                plan.route_compatible[row] & (plan.best_idx[row] == j),
                as_tuple=False,
            ).flatten()
            m = int(pool.numel())
            if m <= 1 or m > max_pool:
                continue
            inds = plan.evict_idx[row].index_select(0, pool)
            kk = keys[row].index_select(0, inds).float().cpu()
            # Snapshot analysis starts from singleton entries (R=D=0), hence
            # conflict iff candidate-candidate Euclidean distance exceeds Delta.
            conflict = torch.cdist(kk, kk) > (delta + 1e-6 * max(1.0, delta))
            conflict.fill_diagonal_(False)
            opt = _maximum_independent_set_size(conflict)
            sel = int((plan.accepted[row, pool]).sum())
            considered += 1
            selected_total += sel
            optimum_total += opt
            max_gap = max(max_gap, opt - sel)
            if sel == opt:
                optimal += 1
            if considered >= max_groups:
                break
        if considered >= max_groups:
            break
    return {
        'mis_oracle_groups_considered': considered,
        'mis_oracle_optimal_group_fraction': optimal / max(considered, 1),
        'mis_oracle_selected_over_optimum': selected_total / max(optimum_total, 1),
        'mis_oracle_max_cardinality_gap': max_gap,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--snapshot',required=True)
    ap.add_argument('--compress_ratio',type=float,default=0.4)
    ap.add_argument('--lambdas',default='0.3,0.4,0.5,0.75,1.0')
    ap.add_argument('--distance_chunk_size',type=int,default=256)
    ap.add_argument('--output_json',required=True)
    ap.add_argument('--output_csv',default=None)
    ap.add_argument('--mis_oracle_max_pool',type=int,default=14,
                    help='Exact maximum-IS diagnostic only for routed pools up to this size; 0 disables')
    ap.add_argument('--mis_oracle_max_groups',type=int,default=200,
                    help='Maximum number of small pools per layer/lambda for exact MIS diagnostic')
    args=ap.parse_args()
    obj=torch.load(args.snapshot,map_location='cpu',weights_only=False)
    records=[]
    for layer in obj['layers']:
        keys4=layer['keys'].float(); imp4=layer['importance'].float()
        b,h,L,d=keys4.shape; x=b*h; keep_len=max(1,int(round(L*(1-args.compress_ratio))))
        keys=keys4.reshape(x,L,d); imp=imp4.reshape(x,L)
        counts=torch.ones((x,L)); zeros=torch.zeros((x,L)); pos=torch.arange(L).view(1,L).expand(x,-1)
        scale=reference_scale_from_original_keys(keys4).reshape(x,1)
        for lam in parse_floats(args.lambdas):
            plans={}
            for mode in ('anchor_only','full'):
                plans[mode]=build_merge_plan(
                    keys,counts,zeros,zeros,imp,pos,keep_len,
                    reference_scale=scale,lambda_mymethod=lam,
                    selector='attention',group_mode=mode,
                    distance_chunk_size=args.distance_chunk_size,
                )
            full=plans['full']; anchor=plans['anchor_only']
            n=full.accepted.numel()
            routed=int(full.route_compatible.sum())
            accepted=int(full.accepted.sum())
            anchor_accepted=int(anchor.accepted.sum())
            rec={
                'layer_idx':int(layer['layer_idx']),'lambda':lam,'length':L,'keep_len':keep_len,
                'reference_scale_mean':float(scale.mean()),
                'route_compatible_ratio':routed/max(n,1),
                'full_accepted_ratio':accepted/max(n,1),
                'full_accept_given_routed':accepted/max(routed,1),
                'anchor_only_accepted_ratio':anchor_accepted/max(n,1),
                'mis_rejected_from_routed_ratio':(routed-accepted)/max(routed,1),
                'max_candidate_pool_size':int(full.candidate_pool_sizes.max()),
                'max_selected_per_group':int(full.selected_per_anchor.max()),
                'conflict_edges':int(full.conflict_edges_per_anchor.sum()),
                'max_mis_rounds':int(full.mis_rounds_per_anchor.max()),
            }
            fs=exact_group_stats(keys,full,lam)
            ans=exact_group_stats(keys,anchor,lam)
            rec.update({f'full_{k}':v for k,v in fs.items()})
            rec.update({f'anchor_{k}':v for k,v in ans.items()})
            rec.update(mis_oracle_stats(keys, full, args.mis_oracle_max_pool, args.mis_oracle_max_groups))
            records.append(rec)
            print(json.dumps(rec,ensure_ascii=False))
    out=Path(args.output_json); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'snapshot':str(args.snapshot),'records':records},ensure_ascii=False,indent=2))
    if args.output_csv and records:
        cp=Path(args.output_csv); cp.parent.mkdir(parents=True,exist_ok=True)
        with cp.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)

if __name__=='__main__': main()
