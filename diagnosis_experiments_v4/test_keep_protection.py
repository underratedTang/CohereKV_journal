#!/usr/bin/env python3
"""CPU-only: mandatory sink and recent-slot selection survive compression rounds.

Uses the actual ContinuousPCache class extracted from the evaluator, with
model/KV routing fake dependencies, so no Transformers or model weights needed.
"""
import inspect
from types import SimpleNamespace
import numpy as np
import torch
from test_continuous_cache_rounds import load_class


def main():
    cache_class=load_class()
    obs=[]
    betas=[]

    def importance(score,ptr,beta):
        betas.append(beta)
        return torch.arange(score.shape[-1],dtype=torch.float32,device=score.device)[None].expand(score.shape[0],-1)

    def protected_split(keys, imp, positions, B, sink_size, recent_window):
        assert int(sink_size)==2 and int(recent_window)==1
        H,L,_=keys.shape
        rows=[];evicts=[]
        for h in range(H):
            pos=positions[h]
            chosen=set((pos<sink_size).nonzero().flatten().tolist())
            assert len(chosen)==2
            recent=sorted((i for i in range(L) if i not in chosen),key=lambda i:int(pos[i]),reverse=True)
            chosen.update(recent[:min(recent_window,B-len(chosen))])
            remaining=sorted((i for i in range(L) if i not in chosen),key=lambda i:float(imp[h,i]),reverse=True)
            chosen.update(remaining[:B-len(chosen)])
            keep=torch.tensor(sorted(chosen),dtype=torch.long,device=keys.device)
            evict=torch.tensor([i for i in range(L) if i not in chosen],dtype=torch.long,device=keys.device)
            assert keep.numel()==B and evict.numel()==L-B
            rows.append(keep);evicts.append(evict)
        obs.append((positions.cpu().clone(),torch.stack(rows).cpu().clone()))
        return torch.stack(rows),torch.stack(evicts)

    # The test intentionally verifies that ContinuousPCache forwards protection
    # parameters and uses ORIGINAL positions from its previous compressed round.
    globs=inspect.unwrap(cache_class.maybe_compress).__globals__
    globs['recent_window_importance']=importance
    globs['select_split']=protected_split
    model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace())]))
    cache=cache_class(model=model,method='merge_all',budget=.5,p=2,
                      lambda_ea=10,lambda_ee=10,reference=np.array([[10.]]),
                      sink_gate=np.array([[True]]),unscored_len=7,scored_len=2,
                      sink_size=2,recent_protect_size=1,recency_beta=1.25,
                      reference_mode='online_first',graph_backend='streamed')
    data=torch.arange(6,dtype=torch.float32).view(1,1,6,1)*.01
    cache.key_cache=[data];cache.value_cache=[data.clone()]
    cache._ppl_logical_seen_tokens=6
    cache.maybe_compress((torch.ones((1,1,2,6)),),None)
    assert cache.slot_positions[0][0,0,0].item()==0
    assert cache.slot_positions[0][0,0,1].item()==1
    assert cache.slot_positions[0][0,0,-1].item()==5
    tail=torch.arange(6,15,dtype=torch.float32).view(1,1,9,1)*.01
    cache.key_cache=[torch.cat([cache.key_cache[0],tail],dim=-2)]
    cache.value_cache=[torch.cat([cache.value_cache[0],tail],dim=-2)]
    cache._ppl_logical_seen_tokens=15
    cache.maybe_compress((torch.ones((1,1,2,12)),),None)
    final_pos=cache.slot_positions[0][0,0].tolist()
    assert 0 in final_pos and 1 in final_pos and 14 in final_pos, final_pos
    assert cache.compress_rounds==2 and len(obs)==2 and betas==[1.25,1.25]

    # Restore original budget semantics: mandatory sink cannot be dropped.
    try:
        cache_class(model=model,method='merge_all',budget=.25,p=2,
                    lambda_ea=1,lambda_ee=1,reference=np.array([[10.]]),
                    sink_gate=np.array([[True]]),unscored_len=7,scored_len=2,
                    sink_size=4,recent_protect_size=1)
    except ValueError as exc:
        assert 'mandatory original sink' in str(exc)
    else:
        raise AssertionError('Expected protected sink budget rejection')

    # Recent-window protection is capped when the physical budget is too small.
    clipped=cache_class(model=model,method='merge_all',budget=.5,p=2,
                        lambda_ea=1,lambda_ee=1,reference=np.array([[10.]]),
                        sink_gate=np.array([[True]]),unscored_len=7,scored_len=2,
                        sink_size=2,recent_protect_size=128)
    assert clipped.effective_recent_protect_size==1
    print('[OK] configurable sink/recent protection, beta, and logical positions across two rounds')


if __name__=='__main__':main()
