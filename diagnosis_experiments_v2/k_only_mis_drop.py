#!/usr/bin/env python3
"""Evicted--evicted K-only MIS: degree, anchor-product, anchor-first or D-first priority.

All priorities use the SAME frozen-TRAIN Key distance graph and the SAME
KeepKV importance computed before the retained/evicted split. Only the greedy
node priority differs. A high score is not a proof of safe merging or dropping.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch

from method_registry import FUSION_METHODS, fusion_is_importance_weighted, fusion_weight_mode
METHODS = FUSION_METHODS


@dataclass
class MISPlan:
    accepted: torch.Tensor                # evicted-relative mask
    k_delta: float
    pool_sizes: torch.Tensor              # [retained anchors]
    selected_sizes: torch.Tensor
    conflict_edges: torch.Tensor
    rounds: torch.Tensor
    routed_importance_sum: float
    accepted_importance_sum: float
    rejected_importance_sum: float
    priority: str
    ea_rejected: int = 0
    original_ee_conflict_edges: int = 0
    weighted_ee_conflict_edges: int = 0
    ea_routed_importance_rejected: float = 0.0
    ea_rejected_by_group: torch.Tensor | None = None
    original_ee_edges_by_group: torch.Tensor | None = None
    original_eligible_ee_edges: int = 0
    anchor_risk_mean: float = 0.0
    anchor_risk_selected_mean: float = 0.0
    anchor_risk_rejected_mean: float = 0.0


def _hash32(index):
    x = index.long() & 0xFFFFFFFF
    return (x * 2654435761 + 1013904223) & 0xFFFFFFFF


@torch.no_grad()
def _neighbor_importance_delta(importance, conflict, valid):
    """D_i = weight_i - sum_{j in N(i)} weight_j on the FULL E--E graph.

    Inputs must be nonnegative KeepKV importance scores. Use float64 for
    summation/subtraction; masked padded nodes contribute nothing. Values can
    be negative. Equal D uses the native degree/anchor-distance/hash ties.
    """
    imp = torch.where(valid, importance.double(), 0.)
    neighbor_mass = torch.bmm(conflict.to(torch.float64), imp.unsqueeze(-1)).squeeze(-1)
    return torch.where(valid, imp - neighbor_mass, 0.)


@torch.no_grad()
def _select_batched(k, physical, gamma, importance, valid, dk, priority,
                    relative=None, importance_aware=False, return_raw=False,
                    original_valid=None, *, anchor_weight=None,
                    k_reference=None):
    """Build the same K-only E-E conflict graph for all greedy priorities."""
    g, n, _ = k.shape
    if n == 0:
        return (torch.zeros((g, 0), dtype=torch.bool, device=k.device),
                torch.zeros(g, dtype=torch.int32, device=k.device),
                torch.zeros(g, dtype=torch.int32, device=k.device))
    if priority not in ('degree', 'degree_anchor_product', 'anchor_first',
                        'importance_delta', 'importance_mass_delta'):
        raise ValueError('Unknown MIS priority')
    dk = torch.as_tensor(dk, dtype=torch.float32, device=k.device).reshape(g)
    if bool((dk <= 0).any()):
        raise ValueError('K-only threshold must be positive')
    pair = valid[:, :, None] & valid[:, None, :]
    kk = k.float()
    centered = kk - kk.mean(dim=1, keepdim=True)
    norm = centered.square().sum(-1)
    k2 = (norm[:, :, None] + norm[:, None, :] -
          2 * torch.bmm(centered, centered.transpose(1, 2))).clamp_min_(0)
    tol = 8 * torch.finfo(torch.float32).eps * torch.maximum(dk.abs(), torch.ones_like(dk))
    limit2=(dk-tol).clamp_min(0).square()[:,None,None]
    raw_conflict_eligible = pair & (k2 > limit2)
    original_pair=(original_valid[:, :, None]&original_valid[:, None, :]
                   if original_valid is not None else pair)
    raw_conflict = original_pair & (k2 > limit2)
    if importance_aware:
        if relative is None:raise ValueError('IA graph requires relative effective weight')
        r=relative.double().clamp_min(0)
        ri,rj=r[:,:,None],r[:,None,:]
        # Twice the optimal importance-weighted two-point reconstruction SSE.
        factor=torch.where(ri+rj>0,2.*ri*rj/(ri+rj).clamp_min(1e-30),0.)
        conflict=pair & ((factor*k2.double())>limit2.double())
    else:
        conflict=raw_conflict
    eye = torch.eye(n, device=k.device, dtype=torch.bool)[None]
    raw_conflict &= ~eye
    raw_conflict_eligible &= ~eye
    conflict &= ~eye
    conflict = conflict | conflict.transpose(1, 2)
    degree = conflict.sum(-1).int()
    hashes = _hash32(physical)
    di, dj = degree[:, :, None], degree[:, None, :]
    gi, gj = gamma[:, :, None], gamma[:, None, :]
    hi, hj = hashes[:, :, None], hashes[:, None, :]
    degree_better = ((dj < di) |
                     ((dj == di) & (gj < gi)) |
                     ((dj == di) & (gj == gi) & (hj < hi)))
    if priority == 'degree':
        better = degree_better & pair  # original: degree, E-A L2, hash
    elif priority == 'anchor_first':
        # Lexicographic E-A distance squared, then degree, then stable hash.
        r2 = gamma.double().square()
        ri, rj = r2[:, :, None], r2[:, None, :]
        better = ((rj < ri) | ((rj == ri) & (dj < di)) |
                  ((rj == ri) & (dj == di) & (hj < hi))) & pair
    elif priority == 'degree_anchor_product':
        # No tunable coefficient: (degree + 1) * ||K_i - K_anchor||_2^2.
        # Deterministic ties: degree, E-A L2, physical-token hash.
        product = (degree.double() + 1.) * gamma.double().square()
        pi, pj = product[:, :, None], product[:, None, :]
        better = ((pj < pi) | ((pj == pi) & degree_better)) & pair
    else:
        # Original Count*Importance D-first and its exact tie-breaks.
        score = _neighbor_importance_delta(importance, conflict, valid)
        si, sj = score[:, :, None], score[:, None, :]
        better = ((sj > si) | ((sj == si) & degree_better)) & pair
    active = valid.clone()
    selected = torch.zeros_like(valid)
    rounds = torch.zeros(g, dtype=torch.int32, device=k.device)
    for r in range(1, n + 1):
        # All-pair ordering is preserved exactly from the original selector.
        blocked = (conflict & active[:, None, :] & better).any(-1)
        winners = active & ~blocked
        selected |= winners
        rounds = torch.where(winners.any(-1), torch.full_like(rounds, r), rounds)
        removed = winners | (conflict & winners[:, None, :]).any(-1)
        active &= ~removed
        if not bool(active.any()):
            break
    edges = torch.triu(conflict, diagonal=1).sum((1, 2)).int()
    raw_edges = torch.triu(raw_conflict, diagonal=1).sum((1, 2)).int()
    valid_raw_edges = torch.triu(raw_conflict_eligible,diagonal=1).sum((1,2)).int()
    return (selected,edges,rounds,raw_edges,valid_raw_edges) if return_raw else (selected,edges,rounds)


@torch.no_grad()
def ee_mis(keys, values, keep, evict, best, routed, anchor_dist,
           k_reference, v_reference, ee_lambda, backend='gpu_batched',
           pair_budget=262144, *, counts=None, importance=None, priority='degree',
           importance_aware=False, importance_mode='count_importance',
           ):
    """One KV head [physical_slots,d]; graph/priority/importance mode explicit.

    `importance` is the already-computed KeepKV score for EACH physical slot.
    No future query, new importance model or extra threshold. IA graph
    weights may depend on the fusion family; D-first always uses Count*I.
    """
    if priority not in ('degree', 'degree_anchor_product', 'anchor_first',
                        'importance_delta', 'importance_mass_delta'):
        raise ValueError('Invalid MIS priority')
    if keys.ndim != 2 or keys.shape != values.shape:
        raise ValueError('K/V require identical [physical_slots,d] shapes')
    if not all(math.isfinite(float(x)) and float(x) > 0
               for x in (k_reference, ee_lambda)):
        raise ValueError('Positive frozen TRAIN Key scale and lambda required')
    if backend not in ('gpu_batched', 'cpu_batched', 'torch', 'cuda', 'auto'):
        raise ValueError('Unknown graph backend ' + str(backend))
    device = keys.device
    dk = float(k_reference) * float(ee_lambda)
    keep = keep.long().reshape(-1)
    evict = evict.long().reshape(-1)
    best = best.long().reshape(-1)
    routed = routed.bool().reshape(-1)
    anchor_dist = anchor_dist.float().reshape(-1)
    n = evict.numel()
    m = keep.numel()
    if best.numel() != n or routed.numel() != n or anchor_dist.numel() != n:
        raise ValueError('Bad evicted relative arrays')
    if bool(((best[routed] < 0) | (best[routed] >= m)).any()):
        raise ValueError('Routed evicted points outside anchors')
    if importance is None:
        if priority in ('importance_delta', 'importance_mass_delta'):
            raise ValueError('Importance-D MIS requires current KeepKV importance')
        importance = torch.zeros(len(keys), device=device, dtype=torch.float32)
    else:
        importance = torch.as_tensor(importance, device=device, dtype=torch.float32).reshape(-1)
        if importance.numel() != len(keys) or not bool(torch.isfinite(importance).all()):
            raise ValueError('KeepKV importance must be finite and match physical slots')
        if bool((importance < 0).any()):
            raise ValueError('KeepKV importance must be nonnegative')
    # Count-decoupled recent attention importance I. Old IA uses W=C*I;
    # new IA-I uses W=I in the E-A/E-E graph. Count stays separate in PPL.
    if importance_mode not in ('count_importance', 'importance_only'):
        raise ValueError('Unknown IA importance mode '+str(importance_mode))
    c=(torch.as_tensor(counts, device=device).float().reshape(-1)
       if counts is not None else torch.ones(len(keys),device=device))
    if c.numel()!=len(keys) or not bool(torch.isfinite(c).all()) or bool((c<=0).any()):
        raise ValueError('Counts must be finite and positive')
    w=(c.double()*importance.double() if importance_mode=='count_importance'
       else importance.double())
    eligible=routed.clone()
    relative=torch.zeros(n,device=device,dtype=torch.float64)
    ea_rejected=0
    ea_rejected_imp=0.
    if importance_aware and bool(routed.any()):
        rr=torch.nonzero(routed).flatten()
        anchor_weight=w[keep[best[rr]]]
        source_weight=w[evict[rr]]
        # Anchor-zero fallback: use a tiny positive denominator relative to
        # current round's largest W; never silently clip ratios above 1.
        floor=max(float(w.max().item())*1e-8,1e-30)
        rho=source_weight/anchor_weight.clamp_min(floor)
        relative[rr]=rho
        factor=(2*rho/(1+rho)).clamp_min(0)
        # Use the existing, physically measured E-A distance (post-RoPE Key L2).
        gate=(factor.sqrt()*anchor_dist[rr].double() <= dk)
        eligible[rr]=gate
        bad=rr[~gate]
        ea_rejected=int(bad.numel())
        ea_rejected_imp=float(importance[evict[bad]].double().sum().item())
    chosen = torch.zeros(n, dtype=torch.bool, device=device)
    pools = torch.zeros(m, dtype=torch.int32, device=device)
    accepts = torch.zeros_like(pools)
    edges = torch.zeros_like(pools)
    rounds = torch.zeros_like(pools)
    raw_edges_total=0
    eligible_edges_total=0
    raw_edges_by_group=torch.zeros_like(pools)
    src_keys = keys.detach().cpu() if backend == 'cpu_batched' else keys
    b_cpu = best.detach().cpu()
    r_cpu = routed.detach().cpu()
    records = []
    for owner in torch.unique(b_cpu[r_cpu], sorted=True).tolist():
        ids = torch.nonzero(r_cpu & (b_cpu == owner)).flatten()
        if ids.numel():
            records.append((int(owner), ids))
    records.sort(key=lambda item: (int(item[1].numel()), item[0]))
    cursor = 0
    while cursor < len(records):
        end = cursor + 1
        width = int(records[cursor][1].numel())
        while end < len(records):
            nw = int(records[end][1].numel())
            if (end - cursor + 1) * nw ** 2 > pair_budget or end - cursor >= 128:
                break
            width = nw
            end += 1
        batch = records[cursor:end]
        g = len(batch)
        work_device = src_keys.device
        indexes = torch.zeros((g, width), dtype=torch.long)
        mask = torch.zeros((g, width), dtype=torch.bool)
        owner_ids = []
        for gi, (owner, ids) in enumerate(batch):
            indexes[gi, :len(ids)] = ids
            mask[gi, :len(ids)] = True
            owner_ids.append(owner)
        indexes = indexes.to(work_device)
        mask = mask.to(work_device)
        ev_src = evict.to(work_device)
        phys = ev_src[indexes]
        gamma = anchor_dist.to(work_device)[indexes]
        local_k = src_keys[phys].float()
        # Count*I priority for every importance D-first method, including
        # K-only methods; geometric conflicts themselves remain unchanged.
        local_importance = ((c * importance).to(work_device)[phys].float()
                            if priority=='importance_mass_delta'
                            else importance.to(work_device)[phys])
        # All original routed entries retain their group membership for
        # AllCount. Only E-A passing entries enter the IA E-E conflict graph.
        local_valid=mask & eligible.to(work_device)[indexes]
        local_relative=relative.to(work_device)[indexes]
        selected, edge_count, nrounds, raw_edge_count, eligible_edge_count = _select_batched(
            local_k, phys, gamma, local_importance, local_valid,
            torch.full((g,), dk, device=work_device), priority,
            relative=local_relative, importance_aware=importance_aware,
            return_raw=True,original_valid=mask)
        raw_edges_total+=int(raw_edge_count.sum().item())
        eligible_edges_total+=int(eligible_edge_count.sum().item())
        selected = selected.to(device)
        ids_device = indexes.to(device)
        valid_device = mask.to(device)
        chosen[ids_device[valid_device]] = selected[valid_device]
        index_owner = torch.tensor(owner_ids, dtype=torch.long, device=device)
        pools[index_owner] = valid_device.sum(1).int()
        accepts[index_owner] = selected.sum(1).int()
        edges[index_owner] = edge_count.to(device)
        raw_edges_by_group[index_owner]=raw_edge_count.to(device)
        rounds[index_owner] = nrounds.to(device)
        cursor = end
    if int(pools.sum().item()) != int(routed.sum().item()):
        raise AssertionError('MIS candidate pools differ from routing')
    if bool((chosen & ~routed).any()):
        raise AssertionError('MIS accepted unrouted')
    eimp = importance[evict].double()
    accepted_imp = float(eimp[chosen].sum().item())
    rejected_imp = float(eimp[routed & ~chosen].sum().item())
    routed_imp = float(eimp[routed].sum().item())
    ea_bad=torch.zeros_like(pools)
    bad=torch.nonzero(routed&~eligible).flatten()
    if bad.numel():ea_bad.index_add_(0,best[bad],torch.ones_like(best[bad],dtype=torch.int32))
    return MISPlan(chosen, dk, pools, accepts, edges, rounds,
                   routed_imp, accepted_imp, rejected_imp, priority,
                   ea_rejected, raw_edges_total, int(edges.sum().item()),
                   ea_rejected_imp,ea_bad,raw_edges_by_group,eligible_edges_total)


@torch.no_grad()
def k_only_ee_mis(*args, **kwargs):
    return ee_mis(*args, priority='degree', **kwargs)


@torch.no_grad()
def k_only_importance_ee_mis(*args, **kwargs):
    return ee_mis(*args, priority='importance_mass_delta', **kwargs)


@torch.no_grad()
def fuse_weighted_slots(keys, values, counts, keep, evict, best, routed, accepted,
                        method, importance=None):
    """Count, I-only or Count*Importance K/V fusion; Counts remain separate.

    I is the current physical-slot score, captured WITHOUT log(Count).
    Fusion may use I-only or C*I; neither alters propagated Count.
    Drop excludes rejected K/V/Counts; AllCount counts rejected routed slots
    but never fuses their K/V. Unrouted slots are dropped in all modes.
    """
    if method not in METHODS:raise ValueError('Bad fusion arm '+str(method))
    if keys.ndim!=2 or keys.shape!=values.shape or counts.numel()!=len(keys):
        raise ValueError('Bad input shapes')
    keep=keep.long().reshape(-1);evict=evict.long().reshape(-1)
    best=best.long().reshape(-1);routed=routed.bool().reshape(-1)
    accepted=accepted.bool().reshape(-1)
    if len(accepted)!=len(evict) or bool((accepted&~routed).any()):
        raise ValueError('Accepted mask outside routed')
    c=counts.float().reshape(-1)
    if bool((~torch.isfinite(c)).any()) or bool((c<=0).any()):
        raise ValueError('Bad Count')
    if importance is None and fusion_is_importance_weighted(method):
        raise ValueError('IW fusion requires current-round attention importance')
    if importance is not None:
        importance=torch.as_tensor(importance,device=keys.device,dtype=torch.float32).reshape(-1)
        if importance.numel()!=len(keys) or bool((~torch.isfinite(importance)).any()) or bool((importance<0).any()):
            raise ValueError('Importance must be nonnegative finite per physical slot')
    iw=fusion_is_importance_weighted(method)
    mode=fusion_weight_mode(method)
    mass=(c if mode=='count' else importance if mode=='importance_only'
          else c*importance)
    members=(routed if method in ('merge_all','importance_weighted_merge_all')
             else (routed&accepted))
    sources=torch.nonzero(members).flatten()
    denom=mass[keep].clone()
    ksum=keys[keep].float()*denom[:,None]
    vsum=values[keep].float()*denom[:,None]
    c_out=c[keep].clone()
    if sources.numel():
        owners=best[sources]
        p=mass[evict[sources]]
        denom.index_add_(0,owners,p)
        ksum.index_add_(0,owners,keys[evict[sources]].float()*p[:,None])
        vsum.index_add_(0,owners,values[evict[sources]].float()*p[:,None])
        c_out.index_add_(0,owners,c[evict[sources]])
    # All-zero observed importance (e.g. newest causally unseen token):
    # deterministic count-weighted fallback for only those empty-weight groups.
    if iw and bool((denom<=0).any()):
        zero=torch.nonzero(denom<=0).flatten()
        fallback=c[keep[zero]].clone()
        kfallback=keys[keep[zero]].float()*fallback[:,None]
        vfallback=values[keep[zero]].float()*fallback[:,None]
        for j,owner in enumerate(zero.tolist()):
            ids=sources[best[sources]==owner]
            if ids.numel():
                wc=c[evict[ids]]
                fallback[j]+=wc.sum()
                kfallback[j]+=(keys[evict[ids]].float()*wc[:,None]).sum(0)
                vfallback[j]+=(values[evict[ids]].float()*wc[:,None]).sum(0)
        denom[zero]=fallback;ksum[zero]=kfallback;vsum[zero]=vfallback
    if method.endswith('_allcount'):
        reject=torch.nonzero(routed&~accepted).flatten()
        if reject.numel():c_out.index_add_(0,best[reject],c[evict[reject]])
    return ((ksum/denom[:,None]).to(keys.dtype),
            (vsum/denom[:,None]).to(values.dtype),c_out)
