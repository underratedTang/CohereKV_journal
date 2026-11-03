#!/usr/bin/env python3
"""Offline frozen per-layer/KV-head MEDIAN RMS-pairwise-Key-L2 reference.

Uses the UNMODIFIED reference calibration utility in motivation_experiments
for model loading helpers, independent train windows, and per-window scales;
only the aggregation here differs: median over sequences, NOT the original RMS.
No test/evaluation Query or Key enters the calibration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOTIVATION = PROJECT_ROOT / 'motivation_experiments'
if str(MOTIVATION) not in sys.path:
    sys.path.insert(0, str(MOTIVATION))
from calibrate_layer_head_scales import (
    parse_dtype, first_execution_device, cache_keys, load_calibration_token_stream,
    select_disjoint_windows, build_calibration_cache, layer_head_scales_from_key_layers,
)


def aggregate_median_scales(sequence_scales):
    """Return [layers, kv_heads] median and descriptive matrices."""
    if not sequence_scales:
        raise ValueError('No calibration sequences')
    mats = [x.detach().double().cpu() if isinstance(x,torch.Tensor)
            else torch.as_tensor(x,dtype=torch.float64) for x in sequence_scales]
    shape = tuple(mats[0].shape)
    if len(shape)!=2 or any(tuple(x.shape)!=shape for x in mats):
        raise ValueError('Each scale matrix must have identical [layers, kv_heads] shape')
    stack=torch.stack(mats,dim=0)
    if not torch.isfinite(stack).all() or (stack <= 0).any():
        raise ValueError('Invalid per-sequence scale')
    median=torch.median(stack,dim=0).values if len(stack)%2 else torch.quantile(stack,0.5,dim=0)
    return median, stack.mean(dim=0), stack.std(dim=0,unbiased=False), stack


def validate_payload(payload,*,model=None,split=None,num_sequences=None,seqlen=None,seed=None):
    """Fail closed: never misinterpret existing RMS reference as Median."""
    if payload.get('schema') != 'diagnosis_median_reference_v1':
        raise ValueError('Unexpected calibration schema')
    if payload.get('aggregation') != 'median_across_sequences':
        raise ValueError('Calibration is not MEDIAN across independent calibration sequences')
    if not payload.get('frozen',False):
        raise ValueError('Reference must be frozen')
    if payload.get('calibration_split') not in ('train','validation'):
        raise ValueError('Calibration split is not an independent train/validation split')
    for k,expected in [('model',model),('calibration_split',split),
                       ('calibration_num_sequences',num_sequences),
                       ('calibration_seqlen',seqlen),('calibration_seed',seed)]:
        if expected is not None and payload.get(k)!=expected:
            raise ValueError(f'Calibration {k}: {payload.get(k)!r} != {expected!r}')
    x=np.asarray(payload.get('layer_head_scales'),dtype=np.float64)
    if x.ndim!=2 or np.any(~np.isfinite(x)) or np.any(x<=0):
        raise ValueError('Invalid per-layer KV-head calibration scale matrix')
    if x.shape!=(int(payload['num_layers']),int(payload['num_kv_heads'])):
        raise ValueError('Calibration scale matrix dimension mismatch')
    seq=np.asarray(payload.get('sequence_layer_head_scales'),dtype=np.float64)
    if seq.shape!=(int(payload['calibration_num_sequences']),*x.shape):
        raise ValueError('Missing per-sequence scale matrix for reproducibility')
    if not np.allclose(x,np.median(seq,axis=0),rtol=1e-12,atol=1e-12):
        raise ValueError('Reference matrix is inconsistent with calibration sequence median')
    return x


def parse_args():
    p=argparse.ArgumentParser(description='Independent train-split MEDIAN reference for E-E MIS')
    p.add_argument('--model',required=True)
    p.add_argument('--dataset',default='wikitext2')
    p.add_argument('--calibration_split',default='train',choices=['train','validation'])
    p.add_argument('--calibration_num_sequences',type=int,default=16)
    p.add_argument('--calibration_seqlen',type=int,default=4096)
    p.add_argument('--calibration_seed',type=int,default=42)
    p.add_argument('--prefill_chunk_size',type=int,default=128)
    p.add_argument('--torch_dtype',default='bfloat16')
    p.add_argument('--attn_implementation',default='eager')
    p.add_argument('--device',default='cuda')
    p.add_argument('--device_map',default='auto')
    p.add_argument('--trust_remote_code',action='store_true')
    p.add_argument('--use_fast_tokenizer',action='store_true')
    p.add_argument('--output',required=True)
    return p.parse_args()


def main():
    a=parse_args()
    if a.calibration_num_sequences<2 or a.calibration_seqlen<2 or a.prefill_chunk_size<1:
        raise ValueError('Require at least 2 independent sequences, >=2 tokens, positive chunk')
    started=time.time()
    random.seed(a.calibration_seed);np.random.seed(a.calibration_seed);torch.manual_seed(a.calibration_seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.calibration_seed)
    import transformers
    from transformers import AutoConfig,AutoModel,AutoTokenizer
    cfg=AutoConfig.from_pretrained(a.model,trust_remote_code=a.trust_remote_code)
    tokenizer=AutoTokenizer.from_pretrained(a.model,use_fast=a.use_fast_tokenizer,
        trust_remote_code=a.trust_remote_code,padding_side='left')
    kwargs=dict(config=cfg,torch_dtype=parse_dtype(a.torch_dtype),
        trust_remote_code=a.trust_remote_code,attn_implementation=a.attn_implementation)
    if a.device_map.lower()!='none':kwargs['device_map']=a.device_map
    model=AutoModel.from_pretrained(a.model,**kwargs).eval()
    if a.device_map.lower()=='none':model.to(torch.device(a.device))
    device=first_execution_device(model,a.device)
    ids=load_calibration_token_stream(tokenizer,a.dataset,a.calibration_split)
    windows,starts=select_disjoint_windows(ids,num_sequences=a.calibration_num_sequences,
        seqlen=a.calibration_seqlen,seed=a.calibration_seed)
    sequence_mats=[]
    print(f'[CALIBRATION] model={a.model} source={a.dataset}/{a.calibration_split} '
          f'M={len(windows)} L={a.calibration_seqlen} aggregation=MEDIAN',flush=True)
    for m,(window,start) in enumerate(zip(windows,starts)):
        past=build_calibration_cache(model,window,device,a.prefill_chunk_size)
        mat,length,_,_=layer_head_scales_from_key_layers(cache_keys(past))
        if length!=a.calibration_seqlen:raise RuntimeError('Bad calibration cache length')
        sequence_mats.append(mat)
        del past,mat
        if torch.cuda.is_available():torch.cuda.empty_cache()
        print(f'[CALIB] {m+1}/{len(windows)}, offset={start}',flush=True)
    frozen,mean,std,stack=aggregate_median_scales(sequence_mats)
    payload={
        'schema':'diagnosis_median_reference_v1',
        'layer_head_scales':frozen.tolist(),
        'sequence_layer_head_scales':stack.tolist(),
        'sequence_scale_mean':mean.tolist(),
        'sequence_scale_std':std.tolist(),
        'aggregation':'median_across_sequences',
        'definition_per_sequence':'sqrt(2*sum_i||K_i-mean(K)||^2/(N-1)+1e-24)',
        'frozen':True,'model':a.model,'dataset':a.dataset,
        'calibration_split':a.calibration_split,
        'evaluation_split':'test',
        'calibration_num_sequences':a.calibration_num_sequences,
        'calibration_seqlen':a.calibration_seqlen,
        'calibration_seed':a.calibration_seed,
        'calibration_sequence_starts':starts,
        'num_layers':int(frozen.shape[0]),'num_kv_heads':int(frozen.shape[1]),
        'prefill_chunk_size':a.prefill_chunk_size,
        'torch_dtype':a.torch_dtype,'attn_implementation':a.attn_implementation,
        'elapsed_seconds':time.time()-started,
        'torch':torch.__version__,'transformers':transformers.__version__,
        'python':platform.python_version(),
    }
    validate_payload(payload,model=a.model,split=a.calibration_split,
        num_sequences=a.calibration_num_sequences,seqlen=a.calibration_seqlen,seed=a.calibration_seed)
    path=Path(a.output).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with (path.parent/'median_reference_by_head.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['layer_idx','kv_head_idx','median_reference','sequence_mean','sequence_std'])
        w.writeheader()
        for layer in range(frozen.shape[0]):
            for head in range(frozen.shape[1]):
                w.writerow(dict(layer_idx=layer,kv_head_idx=head,
                    median_reference=float(frozen[layer,head]),
                    sequence_mean=float(mean[layer,head]),sequence_std=float(std[layer,head])))
    print(f'[CALIBRATION COMPLETE] {path} shape={tuple(frozen.shape)}',flush=True)

if __name__=='__main__':main()
