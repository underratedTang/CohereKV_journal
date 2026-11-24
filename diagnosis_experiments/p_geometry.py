#!/usr/bin/env python3
"""Certified historical p-moment geometry for continuous CohereKV diagnostics.

A physical state represents a set of original *Key* vectors.  Its attention
key can be importance-weighted, so the certificate carries the independent
arithmetic center of the ORIGINAL represented Key vectors.  Count includes
only represented K/V states, never rejected AllCount pseudo-multiplicity.

For finite p>=1, radius and diameter are SAFE upper bounds obtained by
Minkowski and exact intra/inter-group pair-count decomposition.  They need
not equal true p-moments.  For p=inf the same recurrence is the published
CohereKV max-radius/max-diameter certificate (including radius tightening).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch


def parse_p(value):
    v=float(value)
    if not (v>=1 and (math.isfinite(v) or v==math.inf)):
        raise ValueError('p must be a finite real >=1 or inf')
    return v


def _lp_mean(values, weights, p):
    """Weighted p mean, with weights allowing zeros. Torch float64 numerics."""
    values=torch.as_tensor(values,dtype=torch.float64)
    weights=torch.as_tensor(weights,device=values.device,dtype=torch.float64)
    if values.shape!=weights.shape: raise ValueError('weights mismatch')
    active=weights>0
    if not bool(active.any()):return values.new_zeros(())
    if math.isinf(p):return values[active].max()
    # Stable log-sum-exp prevents overflow for large user-configured p.
    positive=active & (values>0)
    if not bool(positive.any()):return values.new_zeros(())
    logtotal=torch.logsumexp(torch.log(weights[positive])+p*torch.log(values[positive]),dim=0)
    return torch.exp((logtotal-torch.log(weights[active].sum()))/p)


@dataclass
class Cluster:
    count: float
    mean: torch.Tensor
    radius: float
    diameter: float


def singleton(key):
    return Cluster(1.,key.detach().double(),0.,0.)


def cross_upper(a: Cluster,b: Cluster):
    return float(torch.linalg.vector_norm(a.mean-b.mean).item()) + a.radius + b.radius


def merge_clusters(clusters, p):
    """A valid bound on full historical R_p,D_p. Input list includes anchor."""
    p=parse_p(p)
    if not clusters:raise ValueError('Cannot merge empty cluster set')
    if len(clusters)==1:return clusters[0]
    counts=torch.tensor([u.count for u in clusters],dtype=torch.float64)
    if bool((counts<1).any()):raise ValueError('Missing represented history')
    C=float(counts.sum().item())
    mu=sum((u.mean*float(u.count) for u in clusters))/C
    internal_weights=[]; internal_distances=[]
    for u in clusters:
        internal_weights.append(u.count*(u.count-1)/2.)
        internal_distances.append(u.diameter)
    for i,u in enumerate(clusters):
        for v in clusters[i+1:]:
            internal_weights.append(u.count*v.count)
            internal_distances.append(cross_upper(u,v))
    diameter=float(_lp_mean(internal_distances,internal_weights,p).item())
    radial_terms=[u.radius+float(torch.linalg.vector_norm(u.mean-mu).item()) for u in clusters]
    radius=float(_lp_mean(radial_terms,[u.count for u in clusters],p).item())
    radius=min(radius,(C-1.)/C*diameter)
    return Cluster(C,mu,radius,diameter)


@torch.no_grad()
def certified_routes(centers,radii,diameters,keep,evict,delta,chunk_size=256,*,history_limit=None):
    """Nearest compatible anchor under the SAME safe p-cross-cluster bound.

    Given valid old-cluster certificates and delta, all candidates passing
    R_e+||mu_e-mu_a||+R_a<=delta are compatible. D_e,D_a<=delta is
    checked too, to make the invariant explicit. Returned gamma is the
    certified cross-group distance used by all MIS priority variants.
    """
    keep=keep.long().reshape(-1);evict=evict.long().reshape(-1)
    device=centers.device
    n=int(evict.numel())
    best=torch.zeros(n,dtype=torch.long,device=device)
    routed=torch.zeros(n,dtype=torch.bool,device=device)
    gamma=torch.zeros(n,dtype=torch.float32,device=device)
    if n==0:return best,routed,gamma
    if keep.numel()==0:raise ValueError('Need retained anchor')
    kk=centers[keep].float()
    # Validity of an old group's certificate is part of the new invariant.
    if history_limit is None:history_limit=delta
    k_ok=diameters[keep]<=history_limit+1.e-5
    for s in range(0,n,int(chunk_size)):
        e=min(s+int(chunk_size),n)
        idx=evict[s:e]
        distances=torch.cdist(centers[idx].float(),kk,p=2)
        bound=distances+radii[idx,None]+radii[keep][None,:]
        compatible=(bound<=delta+1.e-5)&(diameters[idx,None]<=history_limit+1.e-5)&k_ok[None,:]
        # Conference routing chooses the NEAREST current key-center among
        # certificate-compatible anchors, not the smallest R_i+d+R_j.
        measured=distances.masked_fill(~compatible,float('inf'))
        vals,owners=measured.min(-1)
        good=torch.isfinite(vals)
        best[s:e]=owners
        routed[s:e]=good
        bound_chosen=bound.gather(1,owners[:,None]).squeeze(1)
        gamma[s:e]=torch.where(good,bound_chosen,torch.zeros_like(bound_chosen))
    return best,routed,gamma


@torch.no_grad()
def merge_cluster_tensors(counts, means, radii, diameters, p, block_size=128):
    """Same valid p certificates as merge_clusters; chunked quadratic work.

    Keeps [n,n] distance matrices out of GPU memory. Handles a 14K-source
    MergeAll group with <= block_size*n pair storage. Also avoids per-member
    GPU->CPU synchronizations in the normal repeated-compression path.
    """
    p=parse_p(p)
    if counts.ndim!=1 or means.ndim!=2 or means.shape[0]!=counts.numel():
        raise ValueError('Expected one-dimensional count and 2D mean inputs')
    if counts.numel()<1:raise ValueError('No members')
    c=counts.double();mu=means.double();r=radii.double();d=diameters.double()
    C=c.sum();mu_new=(mu*c[:,None]).sum(0)/C
    if len(c)==1:return C,mu_new,r[0],d[0]
    n=len(c);dev=mu.device
    radius_up=_lp_mean(r+torch.linalg.vector_norm(mu-mu_new,dim=-1),c,p)
    if math.isinf(p):
        diameter_up=d.max()
    else:
        within_weights=c*(c-1)/2
        valids=(within_weights>0)&(d>0)
        logmoment=(torch.logsumexp(torch.log(within_weights[valids])+p*torch.log(d[valids]),dim=0)
                   if bool(valids.any()) else torch.tensor(-float('inf'),device=dev,dtype=torch.float64))
    col=torch.arange(n,device=dev)[None,:]
    for start in range(0,n,int(block_size)):
        end=min(start+int(block_size),n)
        pair=torch.cdist(mu[start:end],mu,p=2)+r[start:end,None]+r[None,:]
        pair_weights=c[start:end,None]*c[None,:]
        valid=(torch.arange(start,end,device=dev)[:,None]<col)&(pair>0)
        if not bool(valid.any()):continue
        if math.isinf(p):
            diameter_up=torch.maximum(diameter_up,pair[valid].max())
        else:
            new_log=torch.logsumexp(torch.log(pair_weights[valid])+p*torch.log(pair[valid]),dim=0)
            logmoment=torch.logaddexp(logmoment,new_log)
    if not math.isinf(p):
        diameter_up=torch.exp((logmoment-torch.log(C*(C-1)/2))/p)
        diameter_up=torch.where(torch.isfinite(diameter_up),diameter_up,torch.zeros_like(diameter_up))
    radius_up=torch.minimum(radius_up,((C-1)/C)*diameter_up)
    return C,mu_new,radius_up,diameter_up


@torch.no_grad()
def physical_l2_routes(keys, keep, evict, delta, chunk_size=256):
    """Baseline MergeAll: nearest physical K under Euclidean threshold.

    No historical cluster centers, radii or diameter certificate participates.
    The p parameter is irrelevant for the L2 distance of two physical keys.
    Returns the same three tensors as certified_routes; the third contains
    the actual physical E-A L2 distance for routed candidates.
    """
    keep=keep.long().reshape(-1); evict=evict.long().reshape(-1)
    n=int(evict.numel());device=keys.device
    best=torch.zeros(n,dtype=torch.long,device=device)
    routed=torch.zeros(n,dtype=torch.bool,device=device)
    distances_out=torch.zeros(n,dtype=torch.float32,device=device)
    if not n:return best,routed,distances_out
    if not keep.numel():raise ValueError('MergeAll requires retained anchors')
    if not math.isfinite(float(delta)) or delta<=0:raise ValueError('Positive E-A cutoff required')
    retained=keys[keep].float()
    for start in range(0,n,int(chunk_size)):
        stop=min(n,start+int(chunk_size))
        d=torch.cdist(keys[evict[start:stop]].float(),retained,p=2)
        values,owners=d.min(dim=-1)
        good=values<=float(delta)
        best[start:stop]=owners
        routed[start:stop]=good
        distances_out[start:stop]=torch.where(good,values,torch.zeros_like(values))
    return best,routed,distances_out
