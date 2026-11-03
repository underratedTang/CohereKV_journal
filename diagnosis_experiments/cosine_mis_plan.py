#!/usr/bin/env python3
"""Single-round Cosine OR Euclidean R-E routing + native E-E L2 graph MIS.

Routing threshold sweeps do not change native MIS conflict definition:
E-E L2 > ee_lambda * frozen Median reference scale. All initial C=1,R=D=0.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
TASK_ROOT = Path(__file__).resolve().parents[1] / 'task_evaluation'
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))
from models.common.mymethod_common import (
    select_keep_indices, _complement_indices, _parallel_mis_membership,
)

@dataclass
class CosinePlan:
    keep_idx: torch.Tensor
    evict_idx: torch.Tensor
    best_idx: torch.Tensor
    route_compatible: torch.Tensor
    assigned_cosine: torch.Tensor
    assigned_distance: torch.Tensor
    accepted: torch.Tensor
    candidate_pool_sizes: torch.Tensor
    selected_per_anchor: torch.Tensor
    conflict_edges_per_anchor: torch.Tensor
    mis_rounds_per_anchor: torch.Tensor
    ee_delta: torch.Tensor


def select_split(keys, importance, positions, keep_len, sink_size, recent_window):
    if keys.ndim != 3 or importance.shape != keys.shape[:2] or positions.shape != importance.shape:
        raise ValueError('Expected keys [X,L,D], importance and positions [X,L]')
    kept = select_keep_indices(importance, positions, keep_len, sink_size=sink_size,
                               recent_window=recent_window)
    evict = _complement_indices(kept, keys.shape[1])
    return kept, evict


def cosine_top1(keys, keep_idx, evict_idx, threshold, chunk_size=256):
    """Cosine-only Top1, no R-E L2 certificate or extra cosine+L2 gate.

    Reused unchanged for every E-E MIS threshold. Zero norm = unroutable.
    Tie-breaking: first retained index in ascending token position.
    """
    if not -1.0 <= float(threshold) <= 1.0:
        raise ValueError('cosine threshold must be in [-1,1]')
    x, _, dim = keys.shape
    n, m = evict_idx.shape[1], keep_idx.shape[1]
    best = torch.zeros((x,n),dtype=torch.long,device=keys.device)
    cosbest = torch.zeros((x,n),dtype=torch.float32,device=keys.device)
    compat = torch.zeros((x,n),dtype=torch.bool,device=keys.device)
    l2dist = torch.zeros((x,n),dtype=torch.float32,device=keys.device)
    if n == 0: return best,cosbest,compat,l2dist
    rows = torch.arange(x,device=keys.device)[:,None]
    kk = keys[rows,keep_idx].float()
    norms = kk.norm(dim=-1)
    normalized = F.normalize(kk,dim=-1,eps=1e-12)
    for start in range(0,n,max(1,int(chunk_size))):
        end=min(n,start+max(1,int(chunk_size)))
        ee=keys[rows,evict_idx[:,start:end]].float()
        en=ee.norm(dim=-1)
        similarity=torch.matmul(F.normalize(ee,dim=-1,eps=1e-12),normalized.transpose(-1,-2))
        similarity=similarity.masked_fill(norms[:,None,:]<=1e-12,-2.)
        score,anchor=similarity.max(dim=-1)
        good=(score>=float(threshold)) & (en>1e-12)
        best[:,start:end]=anchor
        cosbest[:,start:end]=torch.where(good,score,torch.zeros_like(score))
        compat[:,start:end]=good
        chosen=kk.gather(1,anchor.unsqueeze(-1).expand(-1,-1,dim))
        dist=(ee-chosen).norm(dim=-1)
        l2dist[:,start:end]=torch.where(good,dist,torch.zeros_like(dist))
    return best,cosbest,compat,l2dist



@torch.no_grad()
def l2_top1(keys, keep_idx, evict_idx, threshold, chunk_size=256):
    """Physical L2 nearest retained anchor. Tie -> smallest retained position.

    Acceptance iff nearest L2 distance <= threshold; no cosine gate. Returns
    same tuple as cosine_top1: best, diagnostic cosine, compatible, L2 distance.
    Computed per evicted chunk to avoid allocating [all_evicted, all_keep].
    """
    if not (float(threshold) >= 0 and float(threshold) < float('inf')):
        raise ValueError('L2 routing cutoff must be finite and nonnegative')
    x, _, dim = keys.shape
    n = evict_idx.shape[1]
    best = torch.zeros((x,n),dtype=torch.long,device=keys.device)
    cosbest = torch.zeros((x,n),dtype=torch.float32,device=keys.device)
    compat = torch.zeros((x,n),dtype=torch.bool,device=keys.device)
    l2dist = torch.zeros((x,n),dtype=torch.float32,device=keys.device)
    if not n: return best,cosbest,compat,l2dist
    rows = torch.arange(x,device=keys.device)[:,None]
    kk=keys[rows,keep_idx].float()
    for start in range(0,n,max(1,int(chunk_size))):
        end=min(n,start+max(1,int(chunk_size)))
        ee=keys[rows,evict_idx[:,start:end]].float()
        # torch.cdist is local to each chunk; it never materializes the full
        # Evicted x Retained matrix or any E-E conflict matrix.
        distances=torch.cdist(ee,kk,p=2)
        score,anchor=distances.min(dim=-1)
        good=score<=float(threshold)
        chosen=kk.gather(1,anchor.unsqueeze(-1).expand(-1,-1,dim))
        cos=F.cosine_similarity(ee,chosen,dim=-1,eps=1e-12)
        best[:,start:end]=anchor
        compat[:,start:end]=good
        cosbest[:,start:end]=torch.where(good,cos,torch.zeros_like(cos))
        l2dist[:,start:end]=torch.where(good,score,torch.zeros_like(score))
    return best,cosbest,compat,l2dist


@torch.no_grad()
def route_top1(keys, keep_idx, evict_idx, mode, cutoff, chunk_size=256):
    if mode == 'cosine':
        return cosine_top1(keys,keep_idx,evict_idx,cutoff,chunk_size)
    if mode == 'l2':
        return l2_top1(keys,keep_idx,evict_idx,cutoff,chunk_size)
    raise ValueError('Unknown routing mode: '+str(mode))


@torch.no_grad()
def _cpu_batched_native_mis(keys, keep, evict, best, routed, assigned_dist, delta,
                            pair_budget=262144):
    """Run the UNMODIFIED native torch graph-MIS on small CPU batches.

    The reference implementation pads *all* local anchor pools to the largest
    pool before allocating [num_groups, max_pool, max_pool] float matrices.
    One unusually large pool can make that tensor exceed GPU memory.  Here
    we keep the exact native distance/degree/hash/priority/MIS logic, but
    partition independent anchor pools by size and cap the padded pair count.
    This is deliberately diagnostic-only; no common/core code is modified.
    """
    device = keys.device
    key = keys.detach().to(device='cpu', dtype=torch.float32)
    ev = evict.detach().to('cpu', dtype=torch.long)
    anchor = best.detach().to('cpu', dtype=torch.long)
    route = routed.detach().to('cpu', dtype=torch.bool)
    gamma = assigned_dist.detach().to('cpu', dtype=torch.float32)
    local_delta = delta.detach().to('cpu', dtype=torch.float32)
    x, n = ev.shape
    keep_len = keep.shape[1]
    dim = key.shape[-1]
    chosen = torch.zeros((x, n), dtype=torch.bool)
    pool_stats = torch.zeros((x, keep_len), dtype=torch.int32)
    accept_stats = torch.zeros_like(pool_stats)
    conflict_stats = torch.zeros_like(pool_stats)
    round_stats = torch.zeros_like(pool_stats)
    records = []
    for row in range(x):
        candidates = torch.nonzero(route[row], as_tuple=False).flatten()
        if not candidates.numel():
            continue
        owners = anchor[row, candidates]
        if bool(((owners < 0) | (owners >= keep_len)).any()):
            raise ValueError('Routed token points outside retained-anchor indices')
        for j in torch.unique(owners, sorted=True).tolist():
            ids = candidates[owners == j]
            records.append((row, j, ids))
    # Sorting by size avoids huge padding of small pools in the same batch.
    records.sort(key=lambda item: (item[2].numel(), item[0], item[1]))
    position = 0
    while position < len(records):
        width = int(records[position][2].numel())
        end = position
        while end < len(records):
            next_width = int(records[end][2].numel())
            n_groups = end - position + 1
            if end > position and (n_groups * next_width * next_width > pair_budget
                                   or n_groups > 128):
                break
            width = next_width
            end += 1
        batch = records[position:end]
        g = len(batch)
        batched_keys = torch.zeros((g, width, dim), dtype=torch.float32)
        batched_physical = torch.zeros((g, width), dtype=torch.long)
        batched_gamma = torch.zeros((g, width), dtype=torch.float32)
        batched_mask = torch.zeros((g, width), dtype=torch.bool)
        batched_delta = torch.empty((g, 1), dtype=torch.float32)
        for slot, (row, j, ids) in enumerate(batch):
            k = ids.numel()
            batched_keys[slot, :k] = key[row, ev[row, ids]]
            batched_physical[slot, :k] = ev[row, ids]
            batched_gamma[slot, :k] = gamma[row, ids]
            batched_mask[slot, :k] = True
            batched_delta[slot, 0] = local_delta[row, 0]
        zeros = torch.zeros((g, width), dtype=torch.float32)
        # Each row is one independent anchor group: keep_len=1,
        # best_idx=0. Physical indices and true anchor distances are unchanged.
        mask, _, pools, selected, edges, rounds = _parallel_mis_membership(
            evicted_keys=batched_keys,
            evicted_rhat=zeros,
            evicted_diameter=zeros,
            evicted_physical_idx=batched_physical,
            best_idx=torch.zeros((g, width), dtype=torch.long),
            route_compatible=batched_mask,
            anchor_gamma=batched_gamma,
            keep_len=1,
            delta=batched_delta,
            group_mode='full',
            graph_backend='torch',
        )
        for slot, (row, j, ids) in enumerate(batch):
            k = ids.numel()
            chosen[row, ids] = mask[slot, :k]
            pool_stats[row, j] = pools[slot, 0]
            accept_stats[row, j] = selected[slot, 0]
            conflict_stats[row, j] = edges[slot, 0]
            round_stats[row, j] = rounds[slot, 0]
        position = end
    return (chosen.to(device), pool_stats.to(device), accept_stats.to(device),
            conflict_stats.to(device), round_stats.to(device))


def native_ee_mis(keys,keep,evict,best,routed,assigned_dist,reference_scale,ee_lambda,
                  graph_backend='torch'):
    """Exactly reuse original graph-MIS membership (degree, gamma, hash priority).

    For C=1,R=D=0: Rhat=D=0. `anchor_gamma` is the physical R-E L2
    distance, used *only* for native priority tie-breaks and diagnostics;
    routing eligibility is set by the independent routing mode. Native E-E conflicts use
    Euclidean distance > `ee_lambda * S_ref`.
    """
    x,n=evict.shape;keep_len=keep.shape[1]
    delta=reference_scale.float().reshape(x,1)*float(ee_lambda)
    if graph_backend == 'cpu_batched':
        selected,pool,num,edges,rounds=_cpu_batched_native_mis(
            keys, keep, evict, best, routed, assigned_dist, delta)
    else:
        row=torch.arange(x,device=keys.device)[:,None]
        evicted=keys[row,evict].float()
        zeros=torch.zeros((x,n),device=keys.device,dtype=torch.float32)
        selected,_,pool,num,edges,rounds=_parallel_mis_membership(
            evicted_keys=evicted,
            evicted_rhat=zeros,evicted_diameter=zeros,
            evicted_physical_idx=evict,
            best_idx=best,route_compatible=routed,
            anchor_gamma=assigned_dist,
            keep_len=keep_len,delta=delta,group_mode='full',
            graph_backend=graph_backend,
        )
    if torch.any(selected & ~routed):raise AssertionError('MIS selected a nonrouted token')
    if int(pool.sum().item()) != int(routed.sum().item()):
        raise AssertionError('MIS candidate pool differs from fixed routing')
    return selected,pool,num,edges,rounds,delta


def make_plan(keys,keep,evict,best,cosines,routed,l2dist,ref_scale,ee_lambda,graph_backend='torch'):
    selected,pool,nsel,edges,rounds,delta=native_ee_mis(
        keys,keep,evict,best,routed,l2dist,ref_scale,ee_lambda,graph_backend)
    return CosinePlan(keep,evict,best,routed,cosines,l2dist,selected,pool,nsel,edges,rounds,delta)
