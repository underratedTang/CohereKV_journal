#!/usr/bin/env python3
"""Oracle-only fixed-state WHY diagnostics for MIS versus IW MergeAll.

Uses REAL *future* queries solely to diagnose attention-mass and value error.
Never used for retained selection, routing, MIS priority, fusion or online PPL.
This is a one-round replay with physical Count=1. For multi-round Count
observations see PPL summary columns in evaluate_ppl_konly_mis.py.

The local reference below is the unfused group, not a separately retrained
model. Group errors are diagnostic, not a causal attribution of final PPL.
"""
from __future__ import annotations
import math
import numpy as np


def _logsumexp(z):
    peak = z.max(-1, keepdims=True)
    return (peak[...,0] + np.log(np.exp(z-peak).sum(-1)))


def _one_center(ids, k, v, imp):
    weights=imp[ids].astype(np.float64)
    if float(weights.sum())<=0:
        weights=np.ones(len(ids),dtype=np.float64)
    return np.average(k[ids],axis=0,weights=weights),np.average(v[ids],axis=0,weights=weights)


def _prediction(q,k,v,c):
    logits=q@k/np.sqrt(k.size) + math.log(int(c))
    return logits,v


def _errors(q, k, v, c, truth_logmass, truth_value):
    predicted_logmass,predicted_value=_prediction(q,k,v,c)
    error=predicted_logmass-truth_logmass
    return dict(
        mass_log_error_mean=float(error.mean()),
        mass_log_error_mae=float(np.abs(error).mean()),
        mass_log_error_rmse=float(np.sqrt(np.mean(error**2))),
        mass_overestimate_fraction=float(np.mean(error>0)),
        value_error_l2_mean=float(np.linalg.norm(truth_value-predicted_value,axis=-1).mean()),
    )


def group_future_query_diagnostics(keys,values,importance,keep,evict,best,routed,
                                   accepted,queries,fusion,mis_family,
                                   max_groups=8,max_addbacks_per_group=3,drop_method=None):
    """Return uniformly shaped group + rejected-addback rows for focused groups.

    Select largest affected groups deterministically without consulting future
    queries. q: [nQshared,nFuture,headDim]. IW and MIS centers come from the
    EXACT same replay fusion tensors. 'count_only' reuses full IW K/V with the
    reduced MIS Count; diagnostic counterfactual, not a deployable method.
    """
    k=np.asarray(keys,dtype=np.float64)
    v=np.asarray(values,dtype=np.float64)
    imp=np.asarray(importance,dtype=np.float64)
    keep=np.asarray(keep,dtype=np.int64)
    evict=np.asarray(evict,dtype=np.int64)
    best=np.asarray(best,dtype=np.int64)
    routed=np.asarray(routed,dtype=bool)
    accepted=np.asarray(accepted,dtype=bool)
    q=queries.detach().cpu().float().numpy().astype(np.float64).reshape(-1,k.shape[1])
    drop_method=drop_method or mis_family+'_drop'
    if drop_method not in fusion.keys:drop_method=mis_family+'_drop_allcount'
    if drop_method not in fusion.keys:return [],[]
    if 'importance_weighted_merge_all' not in fusion.keys:return [],[]
    affected=[]
    for a in range(len(keep)):
        pos=np.flatnonzero(routed&(best==a))
        rej=pos[~accepted[pos]]
        if len(pos) and len(rej):affected.append((a,pos,rej))
    affected.sort(key=lambda t:(-len(t[2]),-len(t[1]),int(keep[t[0]])))
    out=[];addback=[]
    for owner,pos,rej in affected[:int(max_groups)]:
        ids=np.r_[keep[owner],evict[pos]]
        a_ids=np.r_[keep[owner],evict[pos[accepted[pos]]]]
        full_logits=q@k[ids].T/np.sqrt(k.shape[1])
        logmass=_logsumexp(full_logits)
        probs=np.exp(full_logits-logmass[:,None])
        ideal_value=probs@v[ids]
        ki=np.asarray(fusion.keys['importance_weighted_merge_all'][owner],dtype=np.float64)
        vi=np.asarray(fusion.values['importance_weighted_merge_all'][owner],dtype=np.float64)
        ka=np.asarray(fusion.keys[drop_method][owner],dtype=np.float64)
        va=np.asarray(fusion.values[drop_method][owner],dtype=np.float64)
        ci=len(ids);ca=len(a_ids)
        iw=_errors(q,ki,vi,ci,logmass,ideal_value)
        mis=_errors(q,ka,va,ca,logmass,ideal_value)
        allc=_errors(q,ka,va,ci,logmass,ideal_value)
        countonly=_errors(q,ki,vi,ca,logmass,ideal_value)
        a_key=k[keep[owner]];a_value=v[keep[owner]]
        anchor_logits=q@a_key/np.sqrt(k.shape[1])
        anchor_top=anchor_logits>=np.quantile(anchor_logits,0.75)
        logit_shift=(q@(ka-a_key))/np.sqrt(k.shape[1])
        # In a group whose raw members all have C=1, these diagnostics are
        # exact local softmax log-mass/value error for the current state.
        row=dict(mis_family=mis_family,anchor_idx=int(keep[owner]),
            n_routed_group=len(pos),n_accepted_group=len(a_ids)-1,n_rejected_group=len(rej),
            count_full=ci,count_drop=ca,rejected_count_fraction=len(rej)/ci,
            rejected_importance_fraction=float(imp[evict[rej]].sum()/max(imp[evict[pos]].sum(),1e-12)),
            accepted_importance_fraction=float(imp[evict[pos[accepted[pos]]]].sum()/max(imp[evict[pos]].sum(),1e-12)),
            anchor_key_drift_mis=float(np.linalg.norm(ka-a_key)),
            anchor_key_drift_iw=float(np.linalg.norm(ki-a_key)),
            anchor_value_drift_mis=float(np.linalg.norm(va-a_value)),
            anchor_value_drift_iw=float(np.linalg.norm(vi-a_value)),
            iw_to_mis_key_shift=float(np.linalg.norm(ka-ki)),
            iw_to_mis_value_shift=float(np.linalg.norm(va-vi)),
            accepted_key_variance=float(np.mean(np.linalg.norm(k[a_ids]-ka,axis=-1)**2)),
            full_key_variance=float(np.mean(np.linalg.norm(k[ids]-ki,axis=-1)**2)),
            anchor_top25_logit_shift_mean=float(logit_shift[anchor_top].mean()),
            anchor_top25_logit_shift_negative_fraction=float(np.mean(logit_shift[anchor_top]<0)),
            log_count_drop_vs_iw=math.log(ca/ci),
            delta_mass_mae_mis_vs_iw=mis['mass_log_error_mae']-iw['mass_log_error_mae'],
            delta_value_error_mis_vs_iw=mis['value_error_l2_mean']-iw['value_error_l2_mean'],
            diagnostic_query_type='ORACLE future post-RoPE GQA q; never used to choose MIS',
            n_future_queries=len(q))
        for prefix,data in (('iw',iw),('mis_drop',mis),('mis_allcount',allc),('iw_center_drop_count',countonly)):
            for name,value in data.items():row[prefix+'_'+name]=value
        out.append(row)
        # Add-back negative/positive effect: current MIS subset + a rejected
        # evicted candidate. This recomputes both K/V centroid and Count.
        rej_sorted=sorted(rej,key=lambda j:(-float(imp[evict[j]]),int(evict[j])))
        base_mis_err=mis['mass_log_error_mae']
        base_val_err=mis['value_error_l2_mean']
        for rel in rej_sorted[:int(max_addbacks_per_group)]:
            new_ids=np.r_[a_ids,evict[rel]]
            k_add,v_add=_one_center(new_ids,k,v,imp)
            new=_errors(q,k_add,v_add,len(new_ids),logmass,ideal_value)
            wai=float(imp[evict[rel]])
            wa=float(imp[a_ids].sum())
            t=wai/(wai+wa) if (wai+wa)>0 else 1./len(new_ids)
            addback.append(dict(mis_family=mis_family,anchor_idx=int(keep[owner]),
                rejected_evicted_idx=int(evict[rel]),n_routed_group=len(pos),
                rejected_importance=wai,accepted_group_importance=wa,
                addback_weight_fraction=t,
                evicted_anchor_key_distance=float(np.linalg.norm(k[evict[rel]]-a_key)),
                evicted_anchor_value_distance=float(np.linalg.norm(v[evict[rel]]-a_value)),
                evicted_current_center_key_distance=float(np.linalg.norm(k[evict[rel]]-ka)),
                evicted_current_center_value_distance=float(np.linalg.norm(v[evict[rel]]-va)),
                addback_count_logit_increment=math.log((ca+1)/ca),
                delta_mass_mae_addback_minus_mis=new['mass_log_error_mae']-base_mis_err,
                delta_value_error_addback_minus_mis=new['value_error_l2_mean']-base_val_err,
                addback_mass_log_error_mean=new['mass_log_error_mean'],
                diagnostic_query_type='ORACLE future queries; no online use',
                n_future_queries=len(q)))
    return out,addback
