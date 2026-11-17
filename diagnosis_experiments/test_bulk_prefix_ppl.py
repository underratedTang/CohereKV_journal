#!/usr/bin/env python3
"""No-transformers CPU test for the extracted bulk-PPL protocol.

Check causal scored-token indexing with a deliberately incorrect prefill logit.
The only correct prediction is produced after compression, so this catches
accidentally scoring the prompt or the first unscored continuation warm-up.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import torch
from torch import nn


def _get_functions():
    src=Path(__file__).with_name('evaluate_ppl_konly_mis.py').read_text()
    tree=ast.parse(src)
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)
           and n.name in ('fixed_cache_capacity','evaluate_bulk_perplexity')]
    assert len(funcs)==2
    env=dict(math=math,torch=torch,nn=nn,Optional=Optional,Callable=Callable,
             DynamicCache=object,tqdm=lambda xs,desc=None:xs)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<bulk funcs>','exec'),env)
    return env['fixed_cache_capacity'],env['evaluate_bulk_perplexity']


class FakeCache:
    def __init__(self,context_len):
        self.context_len=context_len
        self.processed=0
        self.bulk_compression_done=False
        self.bulk_count=0


class FakeModel:
    def __init__(self):
        self.samples=[]

    def __call__(self,tokens,*,use_cache,past_key_values):
        assert use_cache and tokens.ndim==2
        cache=past_key_values
        # Before compression only token class 2 is probable. After bulk,
        # token class 1 is probable. Our *scored* labels are always class 1.
        logits=torch.zeros((1,tokens.shape[1],3),dtype=torch.float32)
        logits[...,1 if cache.bulk_compression_done else 2]=12.0
        cache.processed+=tokens.shape[1]
        if cache.processed==cache.context_len:
            assert not cache.bulk_compression_done
            cache.bulk_compression_done=True
            cache.bulk_count+=1
        elif cache.processed>cache.context_len:
            assert cache.bulk_compression_done and cache.bulk_count==1
        self.samples.append((int(tokens.shape[1]),cache.processed,cache.bulk_count))
        return SimpleNamespace(logits=logits,past_key_values=cache)


def test_scoring():
    capacity,eval_ppl=_get_functions()
    assert capacity(.1,4096)==410
    assert capacity(.2,4096)==819
    try:capacity(1.,4096)
    except ValueError:pass
    else:raise AssertionError('ratio>=1 must fail')
    P,N,M=8,4,3
    # Each window has distinct prefix, then warm-up=2, N targets=1.
    windows=[torch.tensor([[q]*P+[2]+[1]*N]) for q in range(M)]
    data=torch.cat(windows,dim=1)
    objects=[]
    def factory():
        obj=FakeCache(P)
        objects.append(obj)
        return obj
    model=FakeModel()
    ppl=eval_ppl(model,data,P,N,torch.device('cpu'),prefill_chunk_size=3,cache_factory=factory)
    assert ppl<1.001, f'Expected near-perfect continuation PPL, got {ppl}'
    assert len(objects)==M
    assert all(c.bulk_count==1 and c.processed==P+N for c in objects)
    # Full P prefix is processed before the very first scored query.
    assert all(model.samples[i][1]==P for i in (2,7,12)),model.samples
    assert sum(t for t,_,_ in model.samples)==M*(P+N)
    try:eval_ppl(model,data[:,:-1],P,N,torch.device('cpu'),prefill_chunk_size=3,cache_factory=factory)
    except ValueError:pass
    else:raise AssertionError('Incomplete windows must fail')
    print('[OK] one bulk compression, unscored prefix/warm-up, exact continuation targets, ratio budgets')


if __name__=='__main__':
    test_scoring()
