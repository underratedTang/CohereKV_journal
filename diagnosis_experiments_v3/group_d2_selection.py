#!/usr/bin/env python3
"""Exact historical second-moment subset selection after PHYSICAL E-A routing.

No original Key ledger, R+d+R certificate, E-E pair matrix, or MIS.
All four diagnostic arms share the same E-A candidates and CountMean fusion.

The full selector evaluates the exact D2 of the complete represented original
Key multiset. Pair look-ahead is bounded (not exhaustive); rejected candidates
are revisited once because D2 is not monotone with additional members.
"""
from __future__ import annotations
import math
import numpy as np
import torch

GROUP_D2='group_d2'
WO_HISTORY='group_d2_wo_history'
WO_EE='group_d2_wo_ee'
MATCHED='group_d2_importance_matched'
GROUP_METHODS=(GROUP_D2,WO_HISTORY,WO_EE,MATCHED)


def _append(state, node):
    """Exact merge of (represented Count, historical mean, centered M2)."""
    n,mu,m2=state
    c,nu,q=node
    total=n+c
    delta=nu-mu
    return (total, mu+(c/total)*delta,
            max(0.0, m2+q+(n*c/total)*float(np.dot(delta,delta))))


def _d2(state):
    n,_,m2=state
    return math.sqrt(max(0.0,2*m2/(n-1))) if n>1 else 0.0


def _valid(state,threshold):
    # Absolute tolerance scaled to squared distances, to avoid false
    # rejection for microscopic FP64 arithmetic noise at the threshold.
    n,_,m2=state
    return n<=1 or 2*m2 <= (n-1)*threshold*threshold*(1+1e-10)


def _priority(indices,evict,counts,importance):
    return sorted(indices,key=lambda i:(-float(counts[evict[i]]*importance[evict[i]]),
                                         int(evict[i])))


def _choose_group(pool,evict,counts,centers,m2,importance,anchor,delta,mode,pair_window):
    ordered=_priority(pool,evict,counts,importance)
    # History ablation removes historical within-state variance only FROM
    # the selection rule; the true tracked M2 is never discarded.
    idx=[anchor]+[int(evict[i]) for i in ordered]
    nodes={j:(float(counts[j]),centers[j],
              (0.0 if mode==WO_HISTORY else float(m2[j]))) for j in idx}
    actual_nodes={j:(float(counts[j]),centers[j],float(m2[j])) for j in idx}
    state=nodes[anchor]
    accepted=[]
    skipped=[]
    used=set()
    recovery_attempts=0
    recovery_success=0
    if mode==WO_EE:
        # Intentionally omit ALL interactions between evicted states. Each
        # candidate is independently compared with the original anchor.
        # This arm does not claim that the final D2 is bounded.
        accepted=[i for i in ordered if _valid(_append(state,nodes[int(evict[i])]),delta)]
    else:
        for index,i in enumerate(ordered):
            if i in used:continue
            candidate=_append(state,nodes[int(evict[i])])
            if _valid(candidate,delta):
                state=candidate;accepted.append(i);used.add(i)
                continue
            partner=None
            # At most pair_window other candidates, constant work per failure.
            for j in ordered[index+1:index+1+pair_window]:
                if j in used:continue
                recovery_attempts+=1
                together=_append(candidate,nodes[int(evict[j])])
                if _valid(together,delta):
                    partner=j;state=together;break
            if partner is not None:
                accepted.extend((i,partner));used.add(i);used.add(partner)
                recovery_success+=1
            else:
                skipped.append(i)
        # One bounded reconsideration of early failures, in original priority
        # order, after all later candidate additions. Never irrevocably infer
        # non-feasibility from a singleton failure for finite p.
        for i in skipped:
            if i in used:continue
            candidate=_append(state,nodes[int(evict[i])])
            if _valid(candidate,delta):
                state=candidate;accepted.append(i);used.add(i)
    # Report TRUE history D2 even in history-blind modes.
    real=actual_nodes[anchor]
    for i in accepted:real=_append(real,actual_nodes[int(evict[i])])
    return accepted,dict(pair_attempts=recovery_attempts,
                         pair_success=recovery_success,
                         real_d2=_d2(real),
                         real_d2_violates=int(not _valid(real,delta)))


@torch.no_grad()
def choose_group_d2(keep,evict,best,routed,counts,centers,m2,importance,
                    delta,mode=GROUP_D2,pair_window=8):
    """Returns (bool accepted mask on evict device, selection diagnostics).

    In MATCHED mode the full selector is run only to obtain per-anchor
    cardinalities; the actual identities are sorted independently by the
    same Count*Importance priority. NO full-selector accepted vertex ID is
    used by the matched-control selection.
    """
    if mode not in GROUP_METHODS:raise ValueError(mode)
    if not math.isfinite(delta) or delta<=0 or pair_window<0:raise ValueError('Bad D2 threshold/window')
    def array(t,dtype):return np.asarray(t.detach().cpu().numpy(),dtype=dtype)
    keep_np=array(keep,np.int64).reshape(-1)
    evict_np=array(evict,np.int64).reshape(-1)
    best_np=array(best,np.int64).reshape(-1)
    route_np=array(routed,np.bool_).reshape(-1)
    count_np=array(counts,np.float64).reshape(-1)
    center_np=array(centers,np.float64)
    m2_np=array(m2,np.float64).reshape(-1)
    imp_np=array(importance,np.float64).reshape(-1)
    if not (len(evict_np)==len(best_np)==len(route_np)) or len(keep_np)==0:
        raise ValueError('Invalid group D2 routing arrays')
    if not (len(count_np)==len(center_np)==len(m2_np)==len(imp_np)) or not np.isfinite(center_np).all():
        raise ValueError('Invalid historical moments')
    if (count_np<=0).any() or (m2_np<0).any() or not np.isfinite(m2_np).all():
        raise ValueError('Counts/M2 invalid')
    accepted=np.zeros(len(evict_np),dtype=np.bool_)
    counters=dict(group_d2_anchors=0,group_d2_accepted=0,group_d2_pair_attempts=0,
                  group_d2_pair_success=0,group_d2_real_violations=0,
                  group_d2_max_real_ratio=0.,group_d2_shadow_quota=0,
                  group_d2_matched_quota_shortfall=0,
                  group_d2_same_state_matched_swaps=0,
                  group_d2_same_state_matched_violating_anchors=0,
                  group_d2_same_state_matched_anchors=0)
    pools={}
    for index in np.flatnonzero(route_np):
        pools.setdefault(int(best_np[index]),[]).append(int(index))
    for owner,pool in sorted(pools.items()):
        anchor=int(keep_np[int(owner)])
        chosen,diag=_choose_group(pool,evict_np,count_np,center_np,m2_np,
                         imp_np,anchor,float(delta),
                         GROUP_D2 if mode==MATCHED else mode,pair_window)
        if mode==GROUP_D2:
            # Counterfactual control on the IDENTICAL pre-merge physical state.
            # The separate matched PPL arm has a diverging multi-round cache
            # trajectory and must NOT be called strictly count matched to this.
            quota=len(chosen)
            shadow=_priority(pool,evict_np,count_np,imp_np)[:quota]
            counters['group_d2_same_state_matched_anchors']+=1
            counters['group_d2_same_state_matched_swaps']+=len(set(chosen)^set(shadow))//2
            truth=(float(count_np[anchor]),center_np[anchor],float(m2_np[anchor]))
            for i in shadow:
                j=int(evict_np[i])
                truth=_append(truth,(float(count_np[j]),center_np[j],float(m2_np[j])))
            counters['group_d2_same_state_matched_violating_anchors']+=int(not _valid(truth,delta))
        if mode==MATCHED:
            # Crucial: use only the full selection CARDINALITY, never IDs.
            quota=len(chosen)
            chosen=_priority(pool,evict_np,count_np,imp_np)[:quota]
            # True geometry must reflect the new, independently chosen set.
            real=(float(count_np[anchor]),center_np[anchor],float(m2_np[anchor]))
            for i in chosen:
                j=int(evict_np[i]);real=_append(real,(float(count_np[j]),center_np[j],float(m2_np[j])))
            diag['real_d2']=_d2(real)
            diag['real_d2_violates']=int(not _valid(real,delta))
            counters['group_d2_shadow_quota']+=quota
            counters['group_d2_matched_quota_shortfall']+=abs(len(chosen)-quota)
        accepted[chosen]=True
        counters['group_d2_anchors']+=1
        counters['group_d2_accepted']+=len(chosen)
        counters['group_d2_pair_attempts']+=diag['pair_attempts']
        counters['group_d2_pair_success']+=diag['pair_success']
        counters['group_d2_real_violations']+=diag['real_d2_violates']
        counters['group_d2_max_real_ratio']=max(counters['group_d2_max_real_ratio'],diag['real_d2']/delta)
    return torch.as_tensor(accepted,dtype=torch.bool,device=evict.device),counters


@torch.no_grad()
def merge_group_d2_history(keep,evict,best,accepted,counts,centers,m2):
    """Update exact original-history Count, float64 mean and M2 for real merges.

    Uses the accepted mask only; rejected/unrouted historical members are dropped.
    Returned arrays correspond exactly to physical kept anchor positions.
    """
    def array(x,dtype):return np.asarray(x.detach().cpu().numpy(),dtype=dtype)
    anchors=array(keep,np.int64).reshape(-1)
    evicted=array(evict,np.int64).reshape(-1)
    owner=array(best,np.int64).reshape(-1)
    used=array(accepted,np.bool_).reshape(-1)
    count=array(counts,np.float64).reshape(-1)
    center=array(centers,np.float64)
    variance=array(m2,np.float64).reshape(-1)
    # A single streaming pass over accepted evictions, not B scans of all
    # m candidates: O((B+m)*d), also when hundreds route to one anchor.
    states=[(float(count[a]),center[a].copy(),float(variance[a])) for a in anchors]
    for z in np.flatnonzero(used):
        j=int(owner[z]);idx=int(evicted[z])
        states[j]=_append(states[j],(float(count[idx]),center[idx],float(variance[idx])))
    result_n=[s[0] for s in states]
    result_mu=[s[1] for s in states]
    result_m2=[s[2] for s in states]
    device=keep.device
    return (torch.as_tensor(np.asarray(result_n),device=device,dtype=torch.float64),
            torch.as_tensor(np.asarray(result_mu),device=device,dtype=torch.float64),
            torch.as_tensor(np.asarray(result_m2),device=device,dtype=torch.float64))
