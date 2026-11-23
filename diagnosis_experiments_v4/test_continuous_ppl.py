#!/usr/bin/env python3
"""CPU-only test of strict scored positions and never-reset continuous cache."""
import ast
from pathlib import Path
from types import SimpleNamespace
import math
import torch
from torch import nn

def extract():
    tree=ast.parse(Path(__file__).with_name('evaluate_ppl_continuous.py').read_text())
    funcs=[x for x in tree.body if isinstance(x,ast.FunctionDef)
           and x.name in ('_steps','evaluate_continuous')]
    env=dict(torch=torch,nn=nn,math=math,DynamicCache=object)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<eval>','exec'),env)
    return env['evaluate_continuous']

class Cache:
    def __init__(self,U,S):
        self.U=U;self.S=S;self.seen=0;self.compress_rounds=0
        self.key_cache=[torch.zeros((1,1,0,1))]
        self._ppl_logical_seen_tokens=0
        self._stats={'compress_rounds':0}

class Model:
    def __call__(self,tokens,*,use_cache,past_key_values):
        c=past_key_values
        length=tokens.shape[1];start=c.seen;end=start+length
        # Class 1 should be probable only for POST-compression queries.
        logits=torch.zeros((1,length,3))
        logits[...,1 if c.compress_rounds>0 else 2]=15.
        c.seen=end;c._ppl_logical_seen_tokens=end
        if end==c.compress_rounds*(c.U+c.S)+c.U-1:
            c.compress_rounds+=1;c._stats['compress_rounds']+=1
        c.key_cache=[torch.zeros((1,1,end,1))]
        return SimpleNamespace(logits=logits,past_key_values=c)

def test_continuous():
    evaluation=extract()
    U,S,P=13,5,3
    ids=torch.tensor([[2]*(U-1)+[2]+[1]*S]*P).reshape(1,-1)
    cache=Cache(U,S)
    model=Model()
    ppl,rows=evaluation(model,ids,P,U,S,torch.device('cpu'),
                       cache_factory=lambda:cache,prefill_chunk_size=4,score_window=3)
    assert len(rows)==P and all(r['scored_tokens']==S for r in rows)
    assert cache.compress_rounds==P
    assert cache.seen==P*(U+S)
    assert ppl<1.001,(ppl,rows)
    assert rows[-1]['compression_rounds']==P
    print('[OK] continuous teacher-forced NLL; no reset; post-compression scoring')
if __name__=='__main__':test_continuous()
