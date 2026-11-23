#!/usr/bin/env python3
"""CPU-only regression tests for exact history, bounded joint recovery, ablations."""
import itertools,math,random
import numpy as np
import torch
from group_d2_selection import (
    _append,_d2,_valid,choose_group_d2,merge_group_d2_history,
    GROUP_D2,WO_HISTORY,WO_EE,MATCHED)


def tensors(keys,keep,evict,importance=None,counts=None,m2=None,best=None):
    k=torch.tensor(keys,dtype=torch.float64)
    keep=torch.tensor(keep,dtype=torch.long);evict=torch.tensor(evict,dtype=torch.long)
    n=len(keys)
    return (keep,evict,torch.tensor(best if best is not None else [0]*len(evict),dtype=torch.long),
            torch.ones(len(evict),dtype=torch.bool),
            torch.tensor(counts if counts is not None else [1.]*n,dtype=torch.float64),
            k,torch.tensor(m2 if m2 is not None else [0.]*n,dtype=torch.float64),
            torch.tensor(importance if importance is not None else [1.]*n,dtype=torch.float64))


def choose(data,mode=GROUP_D2,delta=1.,pair_window=8):
    return choose_group_d2(*data,delta=delta,mode=mode,pair_window=pair_window)


def raw_d2(keys):
    k=np.asarray(keys,dtype=np.float64)
    return math.sqrt(sum(float(np.dot(a-b,a-b)) for i,a in enumerate(k) for b in k[i+1:])/(len(k)*(len(k)-1)/2)) if len(k)>1 else 0


def test_joint_recovery_and_statistics():
    data=tensors([[0],[1.1],[1.1],[0.2],[-0.2],[-1.2]], [0],[1,2,3,4,5],
                 importance=[0,9,8,7,6,5])
    mask,s=choose(data)
    assert mask.tolist()==[True,True,True,True,False],(mask,s)
    assert s['group_d2_pair_success']>=1
    assert s['group_d2_real_violations']==0
    c,mu,m2=merge_group_d2_history(data[0],data[1],data[2],mask,data[4],data[5],data[6])
    selected=[0]+[1+i for i,u in enumerate(mask.tolist()) if u]
    assert c.tolist()==[len(selected)]
    assert math.isclose(_d2((int(c[0]),mu[0].numpy(),m2[0].item())),raw_d2(np.array([[0],[1.1],[1.1],[0.2],[-0.2],[-1.2]])[selected]),abs_tol=1e-12)


def test_e_e_ablation_can_violate_when_each_anchor_pair_passes():
    data=tensors([[0],[-0.9],[0.9]],[0],[1,2])
    full,fs=choose(data,delta=1.)
    no_ee,es=choose(data,mode=WO_EE,delta=1.)
    assert sum(full.tolist())==1
    assert no_ee.tolist()==[True,True]
    assert es['group_d2_real_violations']==1


def test_history_ablation_is_genuinely_different():
    # A and E have the SAME physical mean, but A already has large historic M2.
    # Under true D2 A+E exceeds its threshold; 0-M2 shadow accepts.
    data=tensors([[0.],[0.]], [0],[1], counts=[2,1],m2=[2.,0.])
    full,_=choose(data,delta=1.)
    blind,stats=choose(data,mode=WO_HISTORY,delta=1.)
    assert full.tolist()==[False]
    assert blind.tolist()==[True]
    assert stats['group_d2_real_violations']==1


def test_matched_uses_quota_not_membership():
    # Full selects (e1,e2) via pair-lookahead, importance matches e1,e3
    # when the first 2 important choices differ from the optimal feasible pair.
    data=tensors([[0],[-.8],[.8],[-.7]],[0],[1,2,3],importance=[0,9,8,7])
    full,fs=choose(data,delta=.8)
    matched,ms=choose(data,mode=MATCHED,delta=.8)
    assert sum(full.tolist())==sum(matched.tolist())
    assert ms['group_d2_matched_quota_shortfall']==0
    # Explicit identity-independence property: matched is always pure top-K,
    # not affected by full selection membership.
    quota=sum(full.tolist())
    assert matched.tolist()==[i<quota for i in range(3)]


def test_multi_round_exact_ledger_vs_raw_keys():
    rng=random.Random(42)
    for _ in range(50):
        raw=[np.array([[rng.uniform(-2,2),rng.uniform(-2,2)] for _ in range(rng.randint(1,9))])
             for _ in range(rng.randint(2,8))]
        moments=[]
        for group in raw:
            moments.append((len(group),group.mean(0),sum(float(np.dot(x-group.mean(0),x-group.mean(0))) for x in group)))
        current=moments[0]
        history=raw[0]
        for group,stat in zip(raw[1:],moments[1:]):
            current=_append(current,stat)
            history=np.concatenate((history,group),axis=0)
            assert math.isclose(_d2(current),raw_d2(history),rel_tol=1e-10,abs_tol=1e-10)


def test_pair_and_recheck_nonmonotonic():
    data=tensors([[0],[1.1],[1.1]],[0],[1,2],importance=[0,9,8])
    full,metrics=choose(data,delta=1.)
    assert full.tolist()==[True,True],metrics
    no_pair,stats=choose(data,delta=1.,pair_window=0)
    assert no_pair.tolist()==[False,False],stats


def main():
    for fn in (test_joint_recovery_and_statistics,test_e_e_ablation_can_violate_when_each_anchor_pair_passes,
               test_history_ablation_is_genuinely_different,test_matched_uses_quota_not_membership,
               test_multi_round_exact_ledger_vs_raw_keys,test_pair_and_recheck_nonmonotonic):
        fn();print('[PASS]',fn.__name__)
    print('[PASS] all Group-D2 CPU regression tests')

if __name__=='__main__':main()
