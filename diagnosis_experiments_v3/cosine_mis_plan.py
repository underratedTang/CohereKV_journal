#!/usr/bin/env python3
"""Original KeepKV retained split and Cosine/L2 R-E routing only.

The degree/importance-difference-priority K-only E-E graph lives in k_only_mis_drop.py. Routing unchanged.
"""
from __future__ import annotations
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
TASK_ROOT = Path(__file__).resolve().parents[1] / 'task_evaluation'
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))
from models.common.mymethod_common import (
    select_keep_indices, _complement_indices,
)

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
