#!/usr/bin/env python3
"""Mechanism-necessity controls. No R_u+d+R_v historical approximation.

H1: physical nearest E-A -> exact historical E-A filter -> physical+exact
historical E-E conflict graph -> importance-first MIS.
H2: identical-state H1 CARDINALITY counterfactual, then an independent
physical-E-E-safe nearest-first search. NEVER use H1-selected membership. H2's own history is tracked
ONLY to compute its same-state H1 shadow cardinality, never to restrict the
actual H2 accepted set.

Only original unmerged Key vectors are retained for exact history tests. This
is deliberately a costly oracle diagnosis, not a deployable algorithm.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

HISTORY_EXACT = 'necessity_history_exact'
HISTORY_MATCHED = 'necessity_history_matched'
E_IMPORTANCE_MATCHED = 'necessity_ee_importance_matched'
NECESSITY_METHODS = (HISTORY_EXACT,HISTORY_MATCHED,E_IMPORTANCE_MATCHED)


def is_history_method(name):
    return name in (HISTORY_EXACT,HISTORY_MATCHED)


@dataclass
class HeadHistories:
    """CPU history; shape [represented_count,head_dim] per physical slot."""
    slots: list[np.ndarray]

    @classmethod
    def extend(cls, prior, new_original_keys, n_old):
        raw=np.asarray(new_original_keys,dtype=np.float32)
        if raw.ndim!=2:
            raise ValueError('Original Keys must be [tokens, dim]')
        slots=list(prior.slots) if prior is not None else []
        if len(slots)!=int(n_old):
            raise AssertionError('Exact-history ledger differs from physical cache')
        slots.extend(np.array(raw[i:i+1],copy=True) for i in range(len(raw)))
        return cls(slots)

    def merge(self,keep,evict,best,accepted):
        keep=np.asarray(keep,dtype=np.int64)
        evict=np.asarray(evict,dtype=np.int64)
        best=np.asarray(best,dtype=np.int64)
        accepted=np.asarray(accepted,dtype=bool)
        result=[]
        for j,ki in enumerate(keep):
            incoming=evict[accepted & (best==j)]
            result.append(np.concatenate([self.slots[int(ki)]]+
                                         [self.slots[int(u)] for u in incoming],axis=0))
        return HeadHistories(result)

    def counts(self):
        return np.array([len(x) for x in self.slots],dtype=np.int64)


def cross_maximum(a,b,block=48):
    """Exhaustive max of all historical Euclidean pair distances (float64).

    Bounded temporaries; NO radius shortcut, sampling, or approximation of
    original historical pairs. Original bf16/fp16 Key values are represented
    exactly once converted to float32 CPU for storage.
    """
    a=np.asarray(a);b=np.asarray(b)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or not len(a) or not len(b):
        raise ValueError('Nonempty same-dimensional historical groups required')
    if len(a)==1 and len(b)==1:
        return float(np.linalg.norm(a[0].astype(np.float64)-b[0].astype(np.float64)))
    maximum=0.0
    for i in range(0,len(a),block):
        ai=a[i:i+block].astype(np.float64,copy=False)
        aa=np.sum(ai*ai,axis=1)[:,None]
        for j in range(0,len(b),max(128,block)):
            bj=b[j:j+max(128,block)].astype(np.float64,copy=False)
            dist2=np.maximum(aa+np.sum(bj*bj,axis=1)[None,:]-2.*(ai@bj.T),0.)
            maximum=max(maximum,float(np.sqrt(np.max(dist2))))
    return maximum


def diameter_exact(a):
    return 0. if len(a)<2 else cross_maximum(a,a)


def _distances(keys):
    arr=np.asarray(keys,dtype=np.float64)
    square=np.maximum(np.sum(arr*arr,axis=1)[:,None]+
                      np.sum(arr*arr,axis=1)[None,:]-2.*(arr@arr.T),0.)
    return np.sqrt(square)


def _selection(conflict,w,gamma,phys):
    degree=conflict.sum(axis=1)
    merit=w-conflict.astype(np.float64)@w
    hashes=(phys.astype(np.uint64)*np.uint64(2654435761)+np.uint64(1013904223))&np.uint64(0xffffffff)
    order=sorted(range(len(phys)),key=lambda i:(-float(merit[i]),int(degree[i]),
        float(gamma[i]),int(hashes[i]),int(phys[i])))
    selected=np.zeros(len(phys),dtype=bool)
    blocked=np.zeros(len(phys),dtype=bool)
    for i in order:
        if blocked[i]:continue
        selected[i]=True
        blocked|=conflict[i]
        blocked[i]=True
    return selected


def _exact_physical_independent_quota(conflict, distances, ids, quota, node_limit=5000000):
    """Exact cardinality feasibility on the PHYSICAL conflict graph only.

    Return a nearest-first independent set of exactly ``quota`` vertices, or
    None if provably impossible. This is an oracle diagnostic, not a runtime
    compression algorithm. An explicit search-limit error is preferable to
    silently using H1's members or lowering the counterfactual quota.

    Backtracking is invoked only when deterministic greedy passes fail. It
    works for n>30 too; the previous 30-node cutoff made a feasible n=46,
    quota=4 instance fail. Python-int bitsets avoid expensive NumPy subgraphs.
    """
    graph=np.asarray(conflict,dtype=bool)
    n=graph.shape[0]
    if graph.shape!=(n,n) or not np.array_equal(graph,graph.T):
        raise ValueError('Exact quota solver needs a symmetric square graph')
    if quota==0:return np.zeros(n,dtype=bool)
    if quota<0 or quota>n:return None
    order=np.lexsort((np.asarray(ids,dtype=np.int64),
                      np.asarray(distances,dtype=np.float64)))
    ordered_graph=graph[np.ix_(order,order)].copy()
    np.fill_diagonal(ordered_graph,False)
    neighbors=[]
    for i in range(n):
        row=0
        for j in np.flatnonzero(ordered_graph[i]):
            row|=1<<int(j)
        neighbors.append(row)
    all_bits=(1<<n)-1
    seen_fail=set()
    nodes=0

    def first_bit(mask):
        return (mask & -mask).bit_length()-1

    def alpha_upper_can_reach(mask,need):
        # Clique partition of the *conflict* graph gives a safe UPPER bound
        # on the independence number. Stop early when the bound >= need.
        remaining=mask
        ncliques=0
        while remaining:
            ncliques+=1
            if ncliques>=need:return True
            v=first_bit(remaining)
            clique=1<<v
            can_join=(remaining & neighbors[v])
            while can_join:
                w=first_bit(can_join)
                clique|=1<<w
                can_join &= neighbors[w]
            remaining &= ~clique
        return False

    def dfs(available,need):
        nonlocal nodes
        if need==0:return ()
        if available.bit_count()<need:return None
        if need==1:return (first_bit(available),)
        key=(available,need)
        if key in seen_fail:return None
        nodes+=1
        if nodes>node_limit:
            raise RuntimeError('H2 exact PHYSICAL independent-set search exceeded '
                               f'{node_limit} nodes for n={n}, quota={quota}; '
                               'no historical membership was used')
        if not alpha_upper_can_reach(available,need):
            seen_fail.add(key)
            return None
        v=first_bit(available)
        # Lexicographically prefer nearer physical candidates. Backtrack when
        # this nearest candidate prevents completing the requested quota.
        rest=available & ~(1<<v)
        with_v=dfs(rest & ~neighbors[v],need-1)
        if with_v is not None:return (v,)+with_v
        without_v=dfs(rest,need)
        if without_v is not None:return without_v
        seen_fail.add(key)
        return None

    chosen=dfs(all_bits,int(quota))
    if chosen is None:return None
    selected=np.zeros(n,dtype=bool)
    selected[order[np.asarray(chosen,dtype=np.int64)]]=True
    if int(selected.sum())!=quota or graph[np.ix_(selected,selected)].any():
        raise AssertionError('Exact H2 physical quota search returned invalid set')
    return selected


def _physical_only_matched_quota(physical_conflict, gamma, phys, priority, quota):
    """Construct H2 using ONLY physical graph, distance, importance, and quota.

    In particular, H1's selected vertices must never be used as an initial
    independent set or tie-breaker. The quota is computed on H2's OWN state.
    Repeated deterministic physical-only orderings help find feasible sizes
    in graphs where a nearest-first maximal set is smaller than the target.
    A small-graph bounded exact fallback avoids false shortages in tests.
    If none reaches quota, fail visibly rather than quietly bias the control.
    """
    conflict=np.asarray(physical_conflict,dtype=bool)
    distances=np.asarray(gamma,dtype=np.float64)
    ids=np.asarray(phys,dtype=np.int64)
    merit=np.asarray(priority,dtype=np.float64)
    n=len(ids);quota=int(quota)
    if conflict.shape!=(n,n) or len(distances)!=n or len(merit)!=n:
        raise ValueError('Invalid physical-only H2 graph or priority dimensions')
    if quota<0 or quota>n:raise ValueError('Invalid independent-set quota')
    result=np.zeros(n,dtype=bool)
    if quota==0:return result,0
    degree=conflict.sum(axis=1)
    scores=[
        np.lexsort((ids,distances)),
        np.lexsort((ids,distances,-merit)),
        np.lexsort((ids,distances,degree)),
        np.lexsort((ids,distances,-merit/(degree+1))),
        np.lexsort((ids,distances,-1./(degree+1))),
    ]
    def greedy(order):
        selected=np.zeros(n,dtype=bool)
        forbidden=np.zeros(n,dtype=bool)
        for idx in order:
            idx=int(idx)
            if forbidden[idx]:continue
            selected[idx]=True
            if int(selected.sum())==quota:break
            forbidden|=conflict[idx]
            forbidden[idx]=True
        return selected
    best=None;best_count=-1;best_cost=float('inf');best_trial=0
    for trial,order in enumerate(scores):
        candidate=greedy(order)
        count=int(candidate.sum());cost=float(distances[candidate].sum())
        if count>=quota:
            if best is None or cost<best_cost:
                best=candidate;best_cost=cost;best_trial=trial
        elif best is None and count>best_count:
            best_count=count
    if best is not None:
        return best,best_trial
    # H1's chosen vertices witness existence of a physically compatible set
    # of this quota. The physical-only H2 search MUST NOT use their identities.
    # The exact fallback works for every n, including the observed n=46.
    exact=_exact_physical_independent_quota(conflict,distances,ids,quota)
    if exact is not None:
        return exact,len(scores)
    raise RuntimeError(
        f'H2 physical-only selector could not attain independent quota={quota} '
        f'from n={n} candidates (best greedy={best_count}); '
        'NO H1-selected vertices were used. Report and investigate; '
        'do not silently alter the matched-control quota.')


@torch.no_grad()
def exact_history_selection(keys,keep,evict,best,routed,gamma,counts,importance,
                            ledger,delta_ee,delta_history,matched=False):
    """Return actual accepted mask and quantitative same-state counterfactual.

    All history checks are exhaustive; no R∞ or bound is consulted. Physical
    E-A nearest assignment is already fixed by the caller.
    """
    if delta_history<=0 or delta_ee<=0:
        raise ValueError('Positive physical E-E and historical thresholds required')
    device=keys.device
    kk=keep.detach().cpu().long().numpy();ev=evict.detach().cpu().long().numpy()
    owners=best.detach().cpu().long().numpy();active=routed.detach().cpu().bool().numpy()
    gam=gamma.detach().cpu().double().numpy()
    w=(counts.detach().cpu().double().numpy()*importance.detach().cpu().double().numpy())
    physical=keys.detach().cpu().float().numpy()
    if len(ledger.slots)!=len(physical):
        raise AssertionError('Ledger is not aligned with current cache')
    tol=2e-9*max(1.,float(delta_history))
    tol_e=2e-9*max(1.,float(delta_ee))
    accepted=np.zeros(len(ev),dtype=bool)
    diameter_cache={}
    def recorded_diameter(idx):
        idx=int(idx)
        if idx not in diameter_cache:
            diameter_cache[idx]=diameter_exact(ledger.slots[idx])
        return diameter_cache[idx]
    total=dict(history_ea_candidates=0,history_ea_passed=0,
       history_ea_rejected=0,history_ee_extra_edges=0,
       history_h1_selected_shadow=0,history_selected=0,
       history_selected_merged=0,history_max_diameter_over_limit=0,
       history_max_result_diameter_ratio=0.,
       history_matched_swaps=0,history_matched_fallback_groups=0,
       history_matched_groups=0,history_matched_exact_search_groups=0,history_matched_quota_shortfall=0,
       history_physical_ee_conflicts=0)
    for owner in np.unique(owners[active]):
        indices=np.flatnonzero(active&(owners==owner))
        if not len(indices):continue
        aidx=int(kk[owner]);hist_a=ledger.slots[aidx]
        # A historic anchor already over the threshold cannot be repaired by
        # a monotone maximum-diameter requirement.
        anchor_ok=recorded_diameter(aidx)<=delta_history+tol
        allow=[]
        for i in indices:
            total['history_ea_candidates']+=1
            hist_u=ledger.slots[int(ev[i])]
            if (anchor_ok and recorded_diameter(ev[i])<=delta_history+tol and
                cross_maximum(hist_a,hist_u)<=delta_history+tol):
                allow.append(int(i))
        total['history_ea_passed']+=len(allow)
        if allow:
            sub=np.array(allow,dtype=np.int64)
            phys=ev[sub]
            pd=_distances(physical[phys])
            phys_conflict=(pd>delta_ee+tol_e)
            np.fill_diagonal(phys_conflict,False)
            conflict=phys_conflict.copy()
            for i in range(len(phys)):
                for j in range(i+1,len(phys)):
                    if conflict[i,j]:continue
                    if len(ledger.slots[int(phys[i])])==1 and len(ledger.slots[int(phys[j])])==1:
                        hdist=float(pd[i,j])  # physical singleton Key equals its original Key
                    else:
                        hdist=cross_maximum(ledger.slots[int(phys[i])],ledger.slots[int(phys[j])])
                    if hdist>delta_history+tol:
                        conflict[i,j]=True;conflict[j,i]=True
                        total['history_ee_extra_edges']+=1
            chosen=_selection(conflict,w[phys],gam[sub],phys)
            h1_selected=np.zeros(len(ev),dtype=bool)
            h1_selected[sub[chosen]]=True
        else:
            h1_selected=np.zeros(len(ev),dtype=bool)
        total['history_h1_selected_shadow']+=int(h1_selected.sum())
        if matched:
            # Candidate pool for H2 is the complete physical-EA pool. Its
            # H1 shadow is computed on H2's own cache state, not copied from
            # an independent trajectory (which would be an invalid quota).
            phys_all=ev[indices]
            graph=_distances(physical[phys_all])>delta_ee+tol_e
            np.fill_diagonal(graph,False)
            # H1 shadow provides a cardinality ONLY: H1's selected vertex
            # identities are intentionally excluded from H2 selection.
            target_quota=int(h1_selected.sum())
            select,trial=_physical_only_matched_quota(
                graph,gam[indices],phys_all,w[phys_all],target_quota)
            chosen_global=indices[select]
            total['history_matched_groups']+=1
            total['history_matched_fallback_groups']+=int(trial>0)
            total['history_matched_exact_search_groups']+=int(trial>=5)
            # Legacy column retains its name but is now explicitly the size
            # of the physical-only selection's symmetric difference from H1,
            # recorded AFTER selection; it does NOT affect selection.
            total['history_matched_swaps']+=int(np.sum(select&~h1_selected[indices]))
        else:
            chosen_global=np.flatnonzero(h1_selected)
        accepted[chosen_global]=True
        if not matched and len(chosen_global):
            # Audit the ACTUAL simultaneously merged historical group.
            hist_out=np.concatenate([hist_a]+[ledger.slots[int(ev[i])] for i in chosen_global],axis=0)
            ratio=diameter_exact(hist_out)/delta_history
            total['history_max_result_diameter_ratio']=max(total['history_max_result_diameter_ratio'],ratio)
            if ratio>1.+2e-9*max(1.,1./delta_history):
                total['history_max_diameter_over_limit']+=1
                raise RuntimeError('Exact-history MIS simultaneously merged an over-limit group')
    total['history_ea_rejected']=total['history_ea_candidates']-total['history_ea_passed']
    total['history_selected']=int(accepted.sum())
    total['history_selected_merged']=int(np.sum(ledger.counts()[ev[accepted]]>1))
    return torch.as_tensor(accepted,device=device,dtype=torch.bool),total


def importance_count_matched(best,routed,mis_accept,evict,counts,importance):
    """E–E ablation control: per-anchor MIS quota, no E-E graph filtering."""
    device=best.device
    b=best.detach().cpu().long().numpy()
    r=routed.detach().cpu().bool().numpy()
    m=mis_accept.detach().cpu().bool().numpy()
    e=evict.detach().cpu().long().numpy()
    score=(counts.detach().cpu().double().numpy()*importance.detach().cpu().double().numpy())
    out=np.zeros(len(b),dtype=bool)
    for owner in np.unique(b[r]):
        cand=np.flatnonzero(r&(b==owner));quota=int(np.sum(m&(b==owner)))
        order=sorted(cand,key=lambda i:(-float(score[e[i]]),int(e[i])))
        out[order[:quota]]=True
    if int(out.sum())!=int(m.sum()):raise AssertionError('E-E matched quota mismatch')
    return torch.as_tensor(out,device=device,dtype=torch.bool)
