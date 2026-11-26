"""Hard MIS with independent EA/EE history weights.

Both distance endpoints use unrounded count-mean centers. R,D are propagated
without discount, so they remain upper bounds even when gates are relaxed.
"""
import math
import numpy as np
import torch

from method_registry import PRIORITY_MODES

def node_weights(mode, counts, importance):
    """One weight per physical cache slot; inputs are the same-head arrays."""
    if mode not in PRIORITY_MODES: raise ValueError(f'Unknown priority mode: {mode}')
    if counts.shape != importance.shape or counts.ndim != 1:
        raise ValueError('Expected matching 1D counts and importance')
    if mode == 'degree': return torch.ones_like(counts,dtype=torch.float64)
    c=counts.double(); i=importance.double()
    w={'count':c,'importance':i,'count_importance':c*i}[mode]
    if not torch.isfinite(w).all() or (w<0).any(): raise ValueError('MIS weights must be finite and nonnegative')
    return w

def net_gain(graph, weights):
    """Static local opportunity cost, not an exact MWIS marginal gain."""
    w=np.asarray(weights,dtype=np.float64)
    if w.shape != (graph.shape[0],) or not np.isfinite(w).all() or (w<0).any():
        raise ValueError('Invalid conflict-graph weights')
    return w - graph @ w



def _stable_hash(i):
    return (int(i)*2654435761+1013904223) & 0xffffffff


def relation_distance(keys, centers, radii, diameters, left, right, alpha):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('History weight must be in [0,1]')
    dist=torch.cdist(centers[left].double(),centers[right].double())
    if alpha == 0:
        return dist
    return torch.maximum(dist+alpha*(radii[left,None]+radii[right][None,:]),
                         alpha*torch.maximum(diameters[left,None],diameters[right][None,:]))


@torch.no_grad()
def route(keys,centers,radii,diameters,keep,evict,delta,alpha,*,chunk_size=256):
    """Closest eligible anchor; tie by smallest physical slot, as conference Eq.11."""
    order=torch.argsort(keep); anchors=keep[order]
    best=torch.zeros(len(evict),device=keys.device,dtype=torch.long)
    routed=torch.zeros_like(best,dtype=torch.bool)
    gamma=torch.zeros_like(best,dtype=torch.float64)
    metric_keys=centers
    for st in range(0,len(evict),chunk_size):
        ids=evict[st:st+chunk_size]; en=st+len(ids)
        dist=torch.cdist(metric_keys[ids].double(),metric_keys[anchors].double())
        metric=relation_distance(keys,centers,radii,diameters,ids,anchors,alpha)
        val,idx=dist.masked_fill(metric>delta+1e-5,float('inf')).min(-1)
        best[st:en]=order[idx];routed[st:en]=torch.isfinite(val)
        gamma[st:en]=metric.gather(1,idx[:,None]).squeeze(1)
    return best,routed,gamma


@torch.no_grad()
def hard_mis(keys,centers,radii,diameters,evict,best,routed,gamma,delta,alpha,*,weights=None,chunk_size=256):
    """Static weighted net gain; unit weights exactly recover degree priority."""
    if weights is not None and weights.shape != (len(keys),):
        raise ValueError("weights must index physical KV slots")
    accepted=torch.zeros_like(routed); edges=rounds=0
    for owner in best[routed].unique(sorted=True).tolist():
        loc=torch.nonzero(routed & (best==owner)).flatten(); ids=evict[loc]; n=len(ids)
        graph=np.empty((n,n),dtype=bool)
        for st in range(0,n,chunk_size):
            en=min(n,st+chunk_size)
            graph[st:en]=(relation_distance(keys,centers,radii,diameters,ids[st:en],ids,alpha)>delta+1e-5).cpu().numpy()
        graph |= graph.T; np.fill_diagonal(graph,False)
        edges+=int(np.count_nonzero(np.triu(graph,1)))
        gam=gamma[loc].tolist(); phys=ids.tolist()
        if weights is None:
            priority=1-graph.sum(1)
        else:
            priority=net_gain(graph,weights[ids].detach().double().cpu().numpy())
        order=sorted(range(n),key=lambda i:(-priority[i],gam[i],_stable_hash(phys[i]),phys[i]))
        rank=np.empty(n,dtype=int);rank[order]=np.arange(n)
        active=np.ones(n,dtype=bool);chosen=np.zeros(n,dtype=bool)
        while active.any():
            win=active & ~np.any(graph & active[None,:] & (rank[None,:]<rank[:,None]),axis=1)
            if not win.any():raise AssertionError('MIS did not progress')
            chosen |= win;active &= ~(win | graph[:,win].any(axis=1));rounds+=1
        if np.any(graph[chosen][:,chosen]) or np.any(~chosen & ~graph[:,chosen].any(axis=1)):
            raise AssertionError('Not a maximal independent set')
        accepted[loc]=torch.as_tensor(chosen,device=keys.device)
    return accepted,edges,rounds


@torch.no_grad()
def count_mean(keys,values,counts,keep,evict,best,accepted):
    """Binary merge mask; actual Count is also the attention multiplicity."""
    counts=counts.double(); total=counts[keep].clone()
    ks=keys[keep].double()*total[:,None];vs=values[keep].double()*total[:,None]
    src=evict[accepted];dst=best[accepted]
    total.index_add_(0,dst,counts[src]);ks.index_add_(0,dst,keys[src].double()*counts[src,None]);vs.index_add_(0,dst,values[src].double()*counts[src,None])
    return (ks/total[:,None]).to(keys.dtype),(vs/total[:,None]).to(values.dtype),total.float()


@torch.no_grad()
def propagate(counts,centers,radii,diameters,*,chunk_size=256):
    """Undiscounted Eq.17/18 bounds; never clip to admission thresholds."""
    if len(counts)==1:return centers[0],radii[0],diameters[0]
    w=counts.double();mu=centers.double();r=radii.double();d=diameters.double()
    total=w.sum();mean=(w[:,None]*mu).sum(0)/total
    diameter=d.max();col=torch.arange(len(w),device=w.device)[None,:]
    for st in range(0,len(w),chunk_size):
        en=min(len(w),st+chunk_size)
        cross=torch.cdist(mu[st:en],mu)+r[st:en,None]+r[None,:]
        mask=torch.arange(st,en,device=w.device)[:,None]<col
        if mask.any():diameter=torch.maximum(diameter,cross[mask].max())
    radius=torch.minimum(((total-1)/total)*diameter,(r+torch.linalg.vector_norm(mu-mean,dim=-1)).max())
    return mean,radius,diameter


@torch.no_grad()
def merge_all_baseline(keys,values,counts,keep,evict,delta,*,chunk_size=256):
    """Original package's threshold-gated physical L2 routing and FP32 fusion."""
    best=torch.zeros(len(evict),device=keys.device,dtype=torch.long);routed=torch.zeros_like(best,dtype=torch.bool)
    for st in range(0,len(evict),chunk_size):
        en=min(len(evict),st+chunk_size)
        d,owner=torch.cdist(keys[evict[st:en]].float(),keys[keep].float()).min(-1)
        best[st:en]=owner;routed[st:en]=d<=delta
    mass=counts.float();denom=mass[keep].clone();ks=keys[keep].float()*denom[:,None];vs=values[keep].float()*denom[:,None]
    src=evict[routed];dst=best[routed]
    denom.index_add_(0,dst,mass[src]);ks.index_add_(0,dst,keys[src].float()*mass[src,None]);vs.index_add_(0,dst,values[src].float()*mass[src,None])
    return (ks/denom[:,None]).to(keys.dtype),(vs/denom[:,None]).to(values.dtype),denom,best,routed
