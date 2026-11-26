#!/usr/bin/env python3
"""K-only frozen per-layer/KV-head median RMS-pairwise L2 calibration.
Compatible with the K fields of diagnosis_median_reference_v1 from the old package.
"""
import argparse
import csv
import json
import random
import math
import time
from pathlib import Path
import numpy as np
import torch


def aggregate_median_scales(sequence_scales):
    if not sequence_scales:raise ValueError('Empty calibration')
    stack=torch.stack([torch.as_tensor(x,dtype=torch.float64).cpu() for x in sequence_scales])
    if stack.ndim!=3 or not torch.isfinite(stack).all() or (stack<=0).any():raise ValueError('Bad calibration matrix')
    return torch.quantile(stack,.5,dim=0),stack.mean(0),stack.std(0,unbiased=False),stack


def key_scales(keys):
    if keys.ndim!=4 or keys.shape[0]!=1 or keys.shape[-2]<2:raise ValueError('Expected [1,H,N,D], N>=2')
    k=keys[0].double();n=k.shape[-2]
    return (2*(k-k.mean(-2,keepdim=True)).square().sum((-1,-2))/(n-1)+1e-24).sqrt().cpu()


def validate_payload(payload,*,model=None,split='train',dataset=None):
    if payload.get('schema')!='diagnosis_median_reference_v1' or payload.get('aggregation')!='median_across_sequences' or not payload.get('frozen'):
        raise ValueError('Not a frozen median reference')
    for k,v in [('model',model),('calibration_split',split),('dataset',dataset)]:
        if v is not None and payload.get(k)!=v:raise ValueError(f'Calibration {k} mismatch')
    x=np.asarray(payload['layer_head_scales'],dtype=np.float64)
    seq=np.asarray(payload['sequence_layer_head_scales'],dtype=np.float64)
    if x.shape!=(int(payload['num_layers']),int(payload['num_kv_heads'])) or seq.shape!=(int(payload['calibration_num_sequences']),*x.shape):raise ValueError('Bad calibration shapes')
    if not np.isfinite(seq).all() or not (seq>0).all() or not np.isfinite(x).all() or not (x>0).all() or not np.allclose(x,np.median(seq,axis=0),rtol=1e-12,atol=1e-12):raise ValueError('Invalid median reference')
    return x


def choose_starts(total,n,length,seed):
    if n<2 or length<2 or total//length<n:raise ValueError('Insufficient complete disjoint calibration windows')
    return sorted(i*length for i in random.Random(seed).sample(range(total//length),n))


def main():
    from transformers import AutoConfig,AutoModel,AutoTokenizer
    from core_model_utils import parse_dtype,first_execution_device,load_token_stream,cache_key_value_layers,forward_cache_chunk
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--dataset',default='wikitext2')
    p.add_argument('--calibration_split',choices=['train'],default='train')
    p.add_argument('--calibration_num_sequences',type=int,default=4);p.add_argument('--calibration_seqlen',type=int,default=4096)
    p.add_argument('--calibration_seed',type=int,default=42);p.add_argument('--prefill_chunk_size',type=int,default=128)
    p.add_argument('--torch_dtype',default='bfloat16');p.add_argument('--attn_implementation',default='eager')
    p.add_argument('--device',default='cuda');p.add_argument('--device_map',default='auto')
    p.add_argument('--trust_remote_code',action='store_true');p.add_argument('--use_fast_tokenizer',action='store_true')
    p.add_argument('--output',required=True);a=p.parse_args();start_time=time.time()
    if a.prefill_chunk_size<1:raise ValueError('Positive chunk required')
    cfg=AutoConfig.from_pretrained(a.model,trust_remote_code=a.trust_remote_code)
    if a.calibration_seqlen>getattr(cfg,'max_position_embeddings',a.calibration_seqlen):raise ValueError('Calibration length exceeds model context')
    tokenizer=AutoTokenizer.from_pretrained(a.model,use_fast=a.use_fast_tokenizer,trust_remote_code=a.trust_remote_code)
    ids=load_token_stream(tokenizer,a.dataset,a.calibration_split)
    starts=choose_starts(ids.shape[-1],a.calibration_num_sequences,a.calibration_seqlen,a.calibration_seed)
    kwargs=dict(config=cfg,torch_dtype=parse_dtype(a.torch_dtype),trust_remote_code=a.trust_remote_code,attn_implementation=a.attn_implementation)
    if a.device_map.lower()!='none':kwargs['device_map']=a.device_map
    model=AutoModel.from_pretrained(a.model,**kwargs).eval()
    if a.device_map.lower()=='none':model.to(a.device)
    device=first_execution_device(model,a.device);mats=[]
    for start in starts:
        past=None;window=ids[:,start:start+a.calibration_seqlen]
        for st in range(0,a.calibration_seqlen,a.prefill_chunk_size):
            out=forward_cache_chunk(model,window[:,st:st+a.prefill_chunk_size],past,device)
            past=out.past_key_values;del out
        keys,_=cache_key_value_layers(past)
        if len(keys)!=cfg.num_hidden_layers or any(k.shape[1]!=cfg.num_key_value_heads or k.shape[-2]!=a.calibration_seqlen for k in keys):raise RuntimeError('Unexpected calibration cache shape')
        mats.append(torch.stack([key_scales(k) for k in keys]));del keys,past,_
        print(f'[CALIBRATION] {len(mats)}/{len(starts)} start={start}',flush=True)
    med,mean,std,stack=aggregate_median_scales(mats)
    payload=dict(schema='diagnosis_median_reference_v1',aggregation='median_across_sequences',frozen=True,
        definition_per_sequence='sqrt(2*sum_i||K_i-mean(K)||^2/(N-1)+1e-24)',model=a.model,dataset=a.dataset,
        calibration_split='train',calibration_num_sequences=len(starts),calibration_seqlen=a.calibration_seqlen,
        calibration_seed=a.calibration_seed,calibration_sequence_starts=starts,num_layers=med.shape[0],num_kv_heads=med.shape[1],
        layer_head_scales=med.tolist(),sequence_layer_head_scales=stack.tolist(),sequence_scale_mean=mean.tolist(),sequence_scale_std=std.tolist(),
        torch_dtype=a.torch_dtype,attn_implementation=a.attn_implementation,use_fast_tokenizer=a.use_fast_tokenizer,
        prefill_chunk_size=a.prefill_chunk_size,elapsed_seconds=time.time()-start_time)
    validate_payload(payload,model=a.model,dataset=a.dataset)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2))
    with (path.parent/'median_reference_by_head.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['layer_idx','kv_head_idx','median_reference','sequence_mean','sequence_std'])
        for l in range(med.shape[0]):
            for h in range(med.shape[1]):w.writerow([l,h,float(med[l,h]),float(mean[l,h]),float(std[l,h])])
    print(f'[CALIBRATION COMPLETE] {path}',flush=True)

if __name__=='__main__':main()
