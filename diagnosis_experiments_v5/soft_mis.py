#!/usr/bin/env python3
"""History-aware parallel Soft MIS for the journal CohereKV experiment.

One physical K/V pair per group.  Each compression reweights the CURRENT
history-group representatives using Count * Soft MIS g; actual Count supplies the next compression mass. Attention may separately
retain the last merge denominator.  R_p and D_p describe the recursively
renormalized within-group geometry, not an exact globally weighted list
of original token vectors.  This is an
experimental heuristic, NOT a certified worst-case compactness algorithm for
finite p.  Its hard p->inf limit matches fixed-priority greedy MIS on the
conference E-E graph when input histories satisfy the conference invariant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from p_geometry import parse_p, _lp_mean


def _compatibility(gamma, delta, p):
    """Smooth geometric membership in [0,1]; conference threshold as p=inf."""
    if math.isinf(p):
        # Finite tolerance matches the conference certificate comparison.
        return (gamma <= delta+1e-5).to(gamma.dtype)
    return (float(delta) / gamma.clamp_min(torch.finfo(gamma.dtype).tiny)).clamp(max=1).pow(p)


@torch.no_grad()
def soft_anchor_routes(keys, centers, radii, diameters, keep, evict,
                       delta_ea, history_limit, *, chunk_size=256):
    """Hard *physical* E-A routing, followed by separate historical geometry.

    A routed evict always belongs to its nearest retained **physical Key**;
    routing succeeds iff that physical L2 distance <= delta_ea. Historical
    R/D do NOT affect the owner or this routing gate.  The ``hard`` return is
    a diagnostic flag for whether the SELECTED owner also satisfies the old
    historical E-A certificate; it is only used by the fixed hard-priority
    pool and the soft-weight formula, never to reroute an evict.

    Returns (owner indices relative to keep, routed mask, historical hard
    eligibility, selected-owner historical cross bound, physical L2).
    """
    keep=keep.reshape(-1).long(); evict=evict.reshape(-1).long()
    dev=keys.device; n=len(evict)
    if not len(keep):
        raise ValueError('At least one retained Anchor is required')
    if not (math.isfinite(float(delta_ea)) and float(delta_ea)>0):
        raise ValueError('delta_ea must be positive finite')
    if not (math.isfinite(float(history_limit)) and float(history_limit)>0):
        raise ValueError('history_limit must be positive finite')
    best=torch.zeros(n,device=dev,dtype=torch.long)
    routed=torch.zeros(n,device=dev,dtype=torch.bool)
    hard=torch.zeros(n,device=dev,dtype=torch.bool)
    gamma=torch.zeros(n,device=dev,dtype=torch.float64)
    physical=torch.zeros(n,device=dev,dtype=torch.float32)
    if not n: return best,routed,hard,gamma,physical
    kk=keys[keep].float()
    for start in range(0,n,max(1,int(chunk_size))):
        end=min(n,start+max(1,int(chunk_size)))
        idx=evict[start:end]
        dist=torch.cdist(keys[idx].float(),kk,p=2)
        physical_vals,owner=dist.min(dim=-1)
        # Historical distance is evaluated ONLY after the physical owner is
        # fixed.  The stored center differs from a rounded attention Key.
        selected_keep=keep[owner]
        historical_dist=torch.linalg.vector_norm(
            centers[idx].double()-centers[selected_keep].double(),dim=-1)
        bound=historical_dist+radii[idx].double()+radii[selected_keep].double()
        compatible=(bound<=float(delta_ea)+1e-5)&(
            diameters[idx].double()<=float(history_limit)+1e-5)&(
            diameters[selected_keep].double()<=float(history_limit)+1e-5)
        best[start:end]=owner
        physical[start:end]=physical_vals
        routed[start:end]=physical_vals<=float(delta_ea)+1e-5
        hard[start:end]=compatible & routed[start:end]
        gamma[start:end]=bound
    return best,routed,hard,gamma,physical


def _stable_hash(index):
    return (int(index)*2654435761+1013904223)&0xffffffff


@dataclass
class SoftPlan:
    weights: torch.Tensor
    hard_eligible: torch.Tensor
    hard_edges: int
    soft_edges: int
    parallel_rounds: int
    positive_slots: int
    fractional_slots: int
    ea_soft_slots: int
    ea_soft_weight: float
    ee_soft_slots: int
    ee_soft_weight: float
    total_weight: float


@torch.no_grad()
def soft_mis_weights(centers, radii, diameters, keep, evict, best,
                     routed, hard, gamma_ea, delta_ea, delta_ee, p,
                     *, history_limit=None, pair_chunk=256, priority='degree',
                     counts=None, importance=None):
    """Parallel evaluation of a fixed priority DAG, once per owner group.

    Priority of hard-EA-eligible nodes: degree-first (unit node mass),
    or descending continuous importance-mass delta q_i - sum_j c_ij*q_j on the same graph,
    then degree, historical E-A distance, stable hash, physical index.  Soft-only candidates
    follow that pool so their weights vanish without perturbing the hard
    selection when p=inf, *for the fixed physical routing*.  This does NOT
    imply equivalence with the conference's history-compatible owner routing.

    Graph edges: original conference E-E cross certificate
      ||mean_i-mean_j|| + R_i + R_j > delta_ee.
    Soft suppression along an edge: c_ij=1-min(1,(delta_ee/cross)^p).
    All nodes in one dependency layer are processed simultaneously on-device.
    """
    p=parse_p(p); dev=centers.device
    if priority not in ('degree','importance_mass_delta'):
        raise ValueError('Unknown Soft MIS priority')
    node_mass=None
    if priority=='importance_mass_delta':
        if counts is None or importance is None:
            raise ValueError('Importance priority requires actual Count and KeepKV importance')
        counts=torch.as_tensor(counts,device=dev,dtype=torch.float64).reshape(-1)
        importance=torch.as_tensor(importance,device=dev,dtype=torch.float64).reshape(-1)
        if (len(counts)!=len(centers) or len(importance)!=len(centers)
                or not bool(torch.isfinite(counts).all()) or not bool(torch.isfinite(importance).all())
                or bool((counts<=0).any()) or bool((importance<0).any())):
            raise ValueError('Invalid Count/KeepKV importance for Soft MIS')
        node_mass=counts*importance
        if not bool(torch.isfinite(node_mass).all()):raise ValueError('Nonfinite importance mass')
    evict=evict.reshape(-1).long();best=best.reshape(-1).long()
    routed=routed.reshape(-1).bool();hard=hard.reshape(-1).bool()
    gamma_ea=gamma_ea.reshape(-1).double()
    if history_limit is None:history_limit=max(float(delta_ea),float(delta_ee))
    if not (math.isfinite(float(history_limit)) and float(history_limit)>0):
        raise ValueError('history_limit must be positive finite')
    weights=torch.zeros(len(evict),device=dev,dtype=torch.float64)
    if not len(evict):return SoftPlan(weights,hard,0,0,0,0,0,0,0.,0,0.,0.)
    hard_edges=soft_edges=parallel_rounds=0
    ee_soft_slots=0;ee_soft_weight=0.
    owners=best[routed].unique(sorted=True).tolist()
    for owner in owners:
        loc=torch.nonzero(routed&(best==owner)).flatten()
        if not len(loc):continue
        phys=evict[loc]; n=len(loc)
        cent=centers[phys].float();radius=radii[phys].float()
        eligible=hard[loc]
        # E-E graph is computed in bounded GPU blocks; Boolean adjacency on
        # CPU prevents an enormous dense GPU matrix when one anchor owns many
        # evictions.  Full block calculations remain vectorized.
        edge_cpu=np.empty((n,n),dtype=np.bool_)
        eligible_cpu=eligible.cpu().numpy()
        for st in range(0,n,int(pair_chunk)):
            en=min(n,st+int(pair_chunk))
            cross=torch.cdist(cent[st:en],cent,p=2)+radius[st:en,None]+radius[None,:]
            edges=(cross>float(delta_ee)+1e-5).cpu().numpy()
            edge_cpu[st:en]=edges
        np.fill_diagonal(edge_cpu,False)
        # Symmetrize threshold-roundoff differences across GPU distance blocks.
        edge_cpu |= edge_cpu.T
        original=edge_cpu & eligible_cpu[:,None] & eligible_cpu[None,:]
        hard_edges+=int(np.count_nonzero(np.triu(original,1)))
        soft_edges+=int(np.count_nonzero(np.triu(edge_cpu,1)))
        degree=original.sum(1)
        gamma_cpu=gamma_ea[loc].tolist()
        physical_ids=phys.tolist()
        # Keep the v4 hard-EA priority pool and reference adjacency fixed.
        # Continuous importance costs use the SAME c_ij as weight evaluation.
        # For p=inf, unit q restores the degree-first ordering exactly.
        if node_mass is None:
            net=1.-degree.astype(np.float64)
        else:
            q=node_mass[phys]
            if math.isinf(p):
                qc=q.cpu().numpy()
                net=np.array([qc[i]-qc[original[i]].sum(dtype=np.float64)
                              for i in range(n)],dtype=np.float64)
            else:
                net=q.clone()
                for st in range(0,n,int(pair_chunk)):
                    en=min(n,st+int(pair_chunk))
                    cross=torch.cdist(cent[st:en],cent,p=2)+radius[st:en,None]+radius[None,:]
                    c=1.-_compatibility(cross,delta_ee,p)
                    mask=torch.as_tensor(original[st:en],device=dev)
                    net[st:en]-=(c.double()*mask*q[None,:]).sum(dim=1)
                net=net.cpu().numpy()
        order=sorted(range(n),key=lambda i:(not eligible_cpu[i],-float(net[i]),
                     int(degree[i]),gamma_cpu[i],_stable_hash(physical_ids[i]),physical_ids[i]))
        # Independent hard-MIS selection mask for diagnostics only.
        # Does not affect any journal weight or introduce a second method run.
        hard_selected=np.zeros(n,dtype=np.bool_)
        for i in order:
            if not eligible_cpu[i]:break
            if not np.any(hard_selected & original[i]):hard_selected[i]=True
        levels=np.zeros(n,dtype=np.int32)
        # The deterministic orientation is acyclic. Compute longest incoming
        # dependency path; equal-level nodes have no conflict with each other.
        for rank,i in enumerate(order):
            earlier=order[:rank]
            if earlier:
                predecessors=np.asarray(earlier,dtype=np.int64)
                active=predecessors[edge_cpu[i,predecessors]]
                if active.size:levels[i]=int(levels[active].max())+1
        # E-A certificates have TWO gates in conference CohereKV:
        # cross(E,A)<=Delta_EA AND D_i,D_A<=history_limit.
        # One continuous normalized risk is their max; the p=inf limit
        # reproduces both gates, even when EA and EE cutoffs differ.
        ea_risk=torch.maximum(
            gamma_ea[loc]/float(delta_ea),
            torch.maximum(diameters[phys].double(),diameters[keep[int(owner)]].double())
                /float(history_limit))
        a=_compatibility(ea_risk,1.,p)
        # In the hard limit hard-ineligible candidates must have exactly zero
        # contribution, even if they have a fallback physical owner.
        if math.isinf(p):a=torch.where(eligible,a,torch.zeros_like(a))
        local_w=torch.zeros(n,device=dev,dtype=torch.float64)
        for layer in range(int(levels.max())+1):
            ready=np.nonzero(levels==layer)[0]
            if not len(ready):continue
            # Only predecessor layers have nonzero local_w; the current layer
            # contains no conflicting nodes. The product is exact and parallel.
            ready_t=torch.as_tensor(ready,device=dev,dtype=torch.long)
            cross=(torch.cdist(cent[ready_t],cent,p=2)+
                   radius[ready_t,None]+radius[None,:])
            conflict_strength=1.-_compatibility(cross,delta_ee,p)
            factors=(1.-conflict_strength*local_w[None,:]).clamp(0.,1.)
            value=a[ready_t]*torch.prod(factors,dim=1)
            local_w[ready_t]=value
        parallel_rounds+=int(levels.max())+1
        weights[loc]=local_w
        ee_recovery=(eligible & ~torch.as_tensor(hard_selected,device=dev) & (local_w>0))
        ee_soft_slots+=int(ee_recovery.sum().item())
        ee_soft_weight+=float(local_w[ee_recovery].sum().item())
    if not bool(torch.isfinite(weights).all()) or bool((weights<0).any()) or bool((weights>1+1e-10).any()):
        raise RuntimeError('Nonfinite or out-of-range Soft MIS weights')
    pos=(weights>0)&routed
    fraction=(weights>0)&(weights<1-1e-8)&routed
    soft_ea=pos&~hard
    return SoftPlan(weights,hard,hard_edges,soft_edges,parallel_rounds,
                    int(pos.sum().item()),int(fraction.sum().item()),
                    int(soft_ea.sum().item()),float(weights[soft_ea].sum().item()),
                    ee_soft_slots,ee_soft_weight,float(weights.sum().item()))


@torch.no_grad()
def fuse_soft_slots(keys, values, counts, keep, evict, best, weights):
    """Count-based one-pair K/V fusion; no persisted effective mass W.

    For a routed Evict i, the CURRENT compression weight is g_i * C_i.
    Anchor uses its full C_A. The denominator C_tilde is temporary:
        C_tilde = C_A + sum_i g_i*C_i.
    Actual Count for future fusion is instead:
        C_new = C_A + sum_{i:g_i>0} C_i.
    Thus each next-round group starts with its REAL Count, NOT the old
    C_tilde. As a result, prior soft weighting is retained in K/V and
    history geometry, but past discounts are not persisted as a fusion mass.
    """
    keep=keep.long().reshape(-1);evict=evict.long().reshape(-1)
    best=best.long().reshape(-1)
    g=weights.reshape(-1).double()
    count=counts.double().reshape(-1)
    if not bool(torch.isfinite(count).all()) or bool((count<=0).any()):
        raise ValueError('Nonpositive or nonfinite actual Count')
    if len(evict)!=len(best) or len(evict)!=len(g):
        raise ValueError('Invalid Evict/owner/weight shapes')
    if not bool(torch.isfinite(g).all()) or bool((g<0).any()) or bool((g>1+1e-10).any()):
        raise ValueError('Invalid Soft MIS weights')
    gain=g*count[evict]
    denom=count[keep].clone()
    ksum=keys[keep].double()*denom[:,None]
    vsum=values[keep].double()*denom[:,None]
    live=gain>0
    actual_next=count[keep].clone()
    if bool(live.any()):
        owner=best[live]
        denom.index_add_(0,owner,gain[live])
        ksum.index_add_(0,owner,keys[evict[live]].double()*gain[live,None])
        vsum.index_add_(0,owner,values[evict[live]].double()*gain[live,None])
        actual_next.index_add_(0,owner,count[evict[live]])
    return ((ksum/denom[:,None]).to(keys.dtype),
            (vsum/denom[:,None]).to(values.dtype),
            actual_next.float(), denom)


@torch.no_grad()
def merge_weighted_geometry(masses,means,radii,diameters,p,*,block_size=256):
    """Recompute a merged group's current geometric moments from Count*g.

    ``masses`` here is a TEMPORARY vector of C_A and g_i*C_i, NOT a
    persistent effective historical mass W.  Input groups' R_p, D_p
    summarize their current internally normalized history; their internal
    shapes are uniformly rescaled for this one merge, and the resulting
    shape is renormalized to REAL participating Count for the next round.

    For p=2, the RMS radius is exact for this recursively renormalized
    distribution; D_2=sqrt(2)*R_2 uses two independent weighted draws.
    For finite p!=2 the recurrence is a safe moment upper bound; at p=inf
    it becomes a maximum-radius/diameter bound. The caller only uses the
    returned total as the current-round denominator, never stores it.
    """
    p=parse_p(p)
    w=masses.double().reshape(-1)
    if not len(w) or bool((w<=0).any()):raise ValueError('Positive member masses required')
    mu=means.double();r=radii.double();d=diameters.double()
    total=w.sum(); mean=(w[:,None]*mu).sum(0)/total
    if p==2:
        r2=((w*(r.square()+(mu-mean).square().sum(-1))).sum()/total).clamp_min(0)
        radius=r2.sqrt();return total,mean,radius,(2.*r2).sqrt()
    radius=_lp_mean(r+torch.linalg.vector_norm(mu-mean,dim=-1),w,p)
    if math.isinf(p):
        diameter=d.max()
    else:
        positive=(d>0)
        terms=(torch.log(w[positive].square())+p*torch.log(d[positive]))
        total_log=(torch.logsumexp(terms,dim=0) if len(terms)
                   else torch.tensor(-float('inf'),device=w.device,dtype=torch.float64))
    col=torch.arange(len(w),device=w.device)[None,:]
    for st in range(0,len(w),int(block_size)):
        en=min(len(w),st+int(block_size))
        cross=(torch.cdist(mu[st:en],mu,p=2)+r[st:en,None]+r[None,:])
        live=(torch.arange(st,en,device=w.device)[:,None]<col)&(cross>0)
        if not bool(live.any()):continue
        if math.isinf(p):
            diameter=torch.maximum(diameter,cross[live].max())
        else:
            weights=2.*w[st:en,None]*w[None,:]
            t=torch.log(weights.expand_as(cross)[live])+p*torch.log(cross[live])
            total_log=torch.logaddexp(total_log,torch.logsumexp(t,dim=0))
    if not math.isinf(p):
        diameter=torch.exp((total_log-2*torch.log(total))/p)
        diameter=torch.where(torch.isfinite(diameter),diameter,torch.zeros_like(diameter))
    return total,mean,radius,diameter
