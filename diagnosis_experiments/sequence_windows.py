#!/usr/bin/env python3
"""Preflight fixed non-overlapping token windows; never pad, wrap or duplicate."""
import argparse
import json
from pathlib import Path


def choose_indices(requested,total_tokens,window_length):
    if window_length<1 or not requested or min(requested)<0 or len(set(requested))!=len(requested):
        raise ValueError('Invalid requested non-overlapping windows')
    complete=total_tokens//window_length
    selected=[i for i in requested if i<complete]
    if not selected:raise ValueError(f'No requested complete windows; available={complete}')
    return selected,complete


def main():
    from transformers import AutoTokenizer
    from core_model_utils import load_token_stream
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--dataset',default='wikitext2')
    p.add_argument('--split',choices=['validation','test'],required=True)
    p.add_argument('--sequence_indices',required=True);p.add_argument('--window_length',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--use_fast_tokenizer',action='store_true')
    p.add_argument('--trust_remote_code',action='store_true');a=p.parse_args()
    tok=AutoTokenizer.from_pretrained(a.model,use_fast=a.use_fast_tokenizer,trust_remote_code=a.trust_remote_code)
    ids=load_token_stream(tok,a.dataset,split=a.split)
    requested=[int(x) for x in a.sequence_indices.split(',')]
    selected,complete=choose_indices(requested,ids.shape[-1],a.window_length)
    payload=dict(model=a.model,dataset=a.dataset,split=a.split,total_tokens=int(ids.shape[-1]),
                 window_length=a.window_length,complete_windows=complete,requested_indices=requested,
                 selected_indices=selected,excluded_indices=[i for i in requested if i not in selected])
    Path(a.output).write_text(json.dumps(payload,indent=2),encoding='utf8')
    print('[WINDOWS]',json.dumps(payload),flush=True)

if __name__=='__main__':main()
