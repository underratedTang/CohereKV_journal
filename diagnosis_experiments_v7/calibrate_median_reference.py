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


def validate_value_payload(payload, expected_shape=None):
    """Separate, frozen TRAIN-derived V scale; never silently use K scale for V."""
    x=np.asarray(payload.get('value_layer_head_scales'), dtype=np.float64)
    seq=np.asarray(payload.get('value_sequence_layer_head_scales'), dtype=np.float64)
    if x.ndim!=2 or np.any(~np.isfinite(x)) or np.any(x<=0):
        raise ValueError('Missing positive frozen value reference')
    if expected_shape is not None and tuple(x.shape)!=tuple(expected_shape):
        raise ValueError('V reference shape does not match K reference')
    if seq.shape!=(int(payload['calibration_num_sequences']),*x.shape):
        raise ValueError('Missing V reference per-sequence calibration')
    if not np.allclose(x,np.median(seq,axis=0),rtol=1e-12,atol=1e-12):
        raise ValueError('V reference not equal to train median')
    return x


def validate_query_bounds(payload,expected_shape=None):
    """Frozen TRAIN-only sampled post-RoPE GQA Query-norm envelope.

    This is NOT a mathematical global Q bound. It is retained as a descriptive Query diagnostic, not as a compression gate.
    """
    if payload.get('query_norm_schema')!='train_post_rope_gqa_max_v1':
        raise ValueError('Missing train Query norm diagnostic; recalibrate')
    q=np.asarray(payload.get('query_norm_bound_by_kv_head'),dtype=np.float64)
    observed=np.asarray(payload.get('query_norm_observed_train_max_by_kv_head'),dtype=np.float64)
    by_seq=np.asarray(payload.get('query_norm_max_by_sequence'),dtype=np.float64)
    if q.ndim!=2 or any(v.shape!=q.shape for v in (observed,)) or (expected_shape is not None and q.shape!=tuple(expected_shape)):
        raise ValueError('Bad TRAIN Q norm bound dimensions')
    if by_seq.shape!=(int(payload['calibration_num_sequences']),*q.shape):
        raise ValueError('Missing train per-window Query norm maxima')
    if not (np.isfinite(q).all() and np.isfinite(observed).all() and np.isfinite(by_seq).all()):
        raise ValueError('Nonfinite Q norm calibration')
    if not (np.all(q>0) and np.all(observed>0) and np.all(by_seq>0)):
        raise ValueError('Nonpositive Q norm calibration')
    if not np.allclose(observed,by_seq.max(0),rtol=1e-12,atol=1e-12):
        raise ValueError('Observed Q maxima not equal to train maxima')
    safety=float(payload.get('query_norm_safety_factor',0))
    if not (np.isfinite(safety) and safety>=1.0) or not np.allclose(q,observed*safety,rtol=1e-12,atol=1e-12):
        raise ValueError('Q bound inconsistent with train maxima / safety factor')
    return q


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
    p.add_argument('--sink_length',type=int,default=1)
    p.add_argument('--sink_probe_len',type=int,default=64)
    p.add_argument('--sink_diagnostic_seqlen',type=int,default=4096)
    p.add_argument('--sink_presence_threshold',type=float,default=0.05)
    p.add_argument('--query_norm_safety_factor',type=float,default=1.05,
                   help='Train-observed post-RoPE Query norm maximum multiplier (diagnostic only)')
    p.add_argument('--output',required=True)
    return p.parse_args()


def main():
    a=parse_args()
    if not (math.isfinite(a.query_norm_safety_factor) and a.query_norm_safety_factor>=1.):
        raise ValueError('Q norm safety factor must be finite >=1')
    if a.calibration_num_sequences<2 or a.calibration_seqlen<2 or a.prefill_chunk_size<1:
        raise ValueError('Require at least 2 independent sequences, >=2 tokens, positive chunk')
    if a.sink_length!=1:
        raise ValueError('StreamingLLM Figure-12 protocol measures ONLY the first token; sink_length must equal 1')
    if not (0<a.sink_presence_threshold<1) or a.sink_diagnostic_seqlen>a.calibration_seqlen or a.sink_diagnostic_seqlen<=a.sink_probe_len or a.sink_probe_len<1:
        raise ValueError('Invalid Sink calibration size or threshold')
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
    sequence_v_mats=[]
    sequence_sink_mats=[]
    sequence_sink_qhead_mats=[]
    sequence_qnorm_max_mats=[]
    print(f'[CALIBRATION] model={a.model} source={a.dataset}/{a.calibration_split} '
          f'M={len(windows)} L={a.calibration_seqlen} aggregation=MEDIAN',flush=True)
    for m,(window,start) in enumerate(zip(windows,starts)):
        past=build_calibration_cache(model,window,device,a.prefill_chunk_size)
        mat,length,_,_=layer_head_scales_from_key_layers(cache_keys(past))
        from core_model_utils import cache_key_value_layers
        _,raw_values=cache_key_value_layers(past)
        v_mat,v_length,_,_=layer_head_scales_from_key_layers(raw_values)
        if v_length!=length:raise RuntimeError('K/V calibration lengths disagree')
        sequence_v_mats.append(v_mat)
        if length!=a.calibration_seqlen:raise RuntimeError('Bad calibration cache length')
        sequence_mats.append(mat)
        del past  # release the first full-window cache before train Sink probe
        # A separate, strictly train-only teacher-forced suffix supplies the
        # post-RoPE Q needed to measure real Attention Sink. It NEVER enters
        # future test compression; no token positions are protected/frozen.
        # Keep original full-window Median K scale unchanged.
        from core_model_utils import cache_key_value_layers
        from real_query_capture import capture_real_future_queries, require_supported_model_type
        require_supported_model_type(cfg.model_type)
        # Replicate StreamingLLM Appendix-F: the LAST query in a 4096-token
        # window attends to the FIRST token. The independent Median reference
        # continues to use the original full calibration window unchanged.
        diagnostic_len=a.sink_diagnostic_seqlen
        cut=diagnostic_len-a.sink_probe_len
        prefix_past=build_calibration_cache(model,window[:,:cut],device,a.prefill_chunk_size)
        selected_layers=list(range(int(cfg.num_hidden_layers)))
        q_by_layer,joined_past=capture_real_future_queries(model,window[:,cut:diagnostic_len],prefix_past,device,cfg,selected_layers)
        k_layers,_=cache_key_value_layers(joined_past)
        n_kv=int(cfg.num_key_value_heads)
        n_q=int(cfg.num_attention_heads)
        if n_q%n_kv:raise RuntimeError('Noninteger GQA ratio')
        group=n_q//n_kv
        # Reuse the already captured and RoPE-validated train-only Queries.
        # Aggregate ALL Q heads sharing the same KV head (GQA), and every
        # probe position. No test Query is accessed by compression.
        qnorm_max=np.zeros((len(k_layers),n_kv),dtype=np.float64)
        for li in range(len(k_layers)):
            q_norms=torch.linalg.vector_norm(q_by_layer[li].float(),dim=-1).numpy()
            for hi in range(n_kv):
                qnorm_max[li,hi]=float(q_norms[hi*group:(hi+1)*group].max())
        if not np.isfinite(qnorm_max).all() or not (qnorm_max>0).all():
            raise RuntimeError('Invalid TRAIN post-RoPE Query norm sample')
        sequence_qnorm_max_mats.append(qnorm_max)
        sink_mat=np.zeros((len(k_layers),n_kv),dtype=np.float64)
        sink_qhead=np.zeros((len(k_layers),n_q),dtype=np.float64)
        from sink_presence_utils import streamingllm_first_token_last_query
        for li,k_layer in enumerate(k_layers):
            for hi in range(n_kv):
                q=q_by_layer[li][hi*group:(hi+1)*group]
                all_k=k_layer[0,hi].detach().cpu()
                weights=streamingllm_first_token_last_query(q,all_k)
                sink_qhead[li,hi*group:(hi+1)*group]=weights
                sink_mat[li,hi]=float(weights.mean())
        if not np.isfinite(sink_mat).all() or np.any((sink_mat<0)|(sink_mat>1)):
            raise RuntimeError('Nonfinite or out of range train Sink mass')
        sequence_sink_mats.append(sink_mat)
        sequence_sink_qhead_mats.append(sink_qhead)
        del prefix_past,joined_past,q_by_layer,k_layers,sink_mat,sink_qhead,mat
        if torch.cuda.is_available():torch.cuda.empty_cache()
        print(f'[CALIB] {m+1}/{len(windows)}, offset={start}',flush=True)
    frozen,mean,std,stack=aggregate_median_scales(sequence_mats)
    frozen_v,mean_v,std_v,stack_v=aggregate_median_scales(sequence_v_mats)
    qnorm_stack=np.stack(sequence_qnorm_max_mats,axis=0)
    qnorm_observed_max=qnorm_stack.max(axis=0)
    qnorm_bound=qnorm_observed_max*a.query_norm_safety_factor
    if qnorm_bound.shape!=tuple(frozen.shape):
        raise RuntimeError('Q norm and median scale matrix shapes disagree')
    sink_stack=np.stack(sequence_sink_mats,axis=0)
    sink_mean=sink_stack.mean(axis=0)
    sink_qhead_stack=np.stack(sequence_sink_qhead_mats,axis=0)
    sink_qhead_mean=sink_qhead_stack.mean(axis=0)
    sink_qhead_std=sink_qhead_stack.std(axis=0)
    sink_layer_mean=sink_qhead_mean.mean(axis=1)
    sink_layer_std_across_heads=sink_qhead_mean.std(axis=1)
    sink_std=sink_stack.std(axis=0)
    sink_presence=(sink_mean>=a.sink_presence_threshold)
    payload={
        'schema':'diagnosis_median_reference_v1',
        'query_norm_schema':'train_post_rope_gqa_max_v1',
        'query_norm_definition':'Maximum post-RoPE Q L2 over all train calibration windows, last sink_probe_len positions, and all query heads sharing each KV head; multiplied by a safety factor. Conditional empirical cap, NOT universal bound.',
        'query_norm_sampled_positions_per_sequence':a.sink_probe_len,
        'query_norm_safety_factor':a.query_norm_safety_factor,
        'query_norm_max_by_sequence':qnorm_stack.tolist(),
        'query_norm_observed_train_max_by_kv_head':qnorm_observed_max.tolist(),
        'query_norm_bound_by_kv_head':qnorm_bound.tolist(),
        'sink_calibration_schema':'streamingllm_last_query_first_token_v2',
        'sink_measure_definition':'first token softmax probability for last query at diagnostic_seqlen, per Q head, train mean across windows, then mean across Q heads sharing each KV head',
        'sink_diagnostic_seqlen':a.sink_diagnostic_seqlen,
        'sink_first_token_mass_layer_mean':sink_layer_mean.tolist(),
        'sink_first_token_mass_layer_std_across_heads':sink_layer_std_across_heads.tolist(),
        'sink_first_token_mass_qhead_mean':sink_qhead_mean.tolist(),
        'sink_first_token_mass_qhead_std':sink_qhead_std.tolist(),
        'sink_first_token_mass_qhead_by_sequence':sink_qhead_stack.tolist(),
        'sink_length':a.sink_length,
        'sink_probe_len':a.sink_probe_len,
        'sink_presence_threshold':a.sink_presence_threshold,
        'sink_mass_mean':sink_mean.tolist(),
        'sink_mass_std':sink_std.tolist(),
        'sink_mass_by_sequence':sink_stack.tolist(),
        'sink_presence_by_kv_head':sink_presence.astype(int).tolist(),
        'layer_head_scales':frozen.tolist(),
        'value_layer_head_scales':frozen_v.tolist(),
        'value_sequence_layer_head_scales':stack_v.tolist(),
        'value_sequence_scale_mean':mean_v.tolist(),
        'value_sequence_scale_std':std_v.tolist(),
        'sequence_layer_head_scales':stack.tolist(),
        'sequence_scale_mean':mean.tolist(),
        'sequence_scale_std':std.tolist(),
        'aggregation':'median_across_sequences',
        'definition_per_sequence':'sqrt(2*sum_i||K_i-mean(K)||^2/(N-1)+1e-24)',
        'frozen':True,'model':a.model,'dataset':a.dataset,
        'calibration_split':a.calibration_split,
        'evaluation_split':'validation_or_test',
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
    validate_value_payload(payload, frozen.shape)
    validate_query_bounds(payload, frozen.shape)
    validate_payload(payload,model=a.model,split=a.calibration_split,
        num_sequences=a.calibration_num_sequences,seqlen=a.calibration_seqlen,seed=a.calibration_seed)
    path=Path(a.output).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with (path.parent/'median_reference_by_head.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['layer_idx','kv_head_idx','median_reference','value_median_reference','sequence_mean','sequence_std','value_sequence_mean','value_sequence_std','query_norm_bound','query_norm_train_max'])
        w.writeheader()
        for layer in range(frozen.shape[0]):
            for head in range(frozen.shape[1]):
                w.writerow(dict(layer_idx=layer,kv_head_idx=head,
                    median_reference=float(frozen[layer,head]),
                    value_median_reference=float(frozen_v[layer,head]),
                    value_sequence_mean=float(mean_v[layer,head]),value_sequence_std=float(std_v[layer,head]),
                    sequence_mean=float(mean[layer,head]),sequence_std=float(std[layer,head]),
                    query_norm_bound=float(qnorm_bound[layer,head]),
                    query_norm_train_max=float(qnorm_observed_max[layer,head])))
    with (path.parent/'sink_presence_by_kv_head.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['layer_idx','kv_head_idx','diagnostic_seqlen','sink_measure','sink_mass_mean','sink_mass_std','sink_presence','method'])
        w.writeheader()
        for li in range(sink_mean.shape[0]):
            for hi in range(sink_mean.shape[1]):
                flag=bool(sink_presence[li,hi])
                w.writerow(dict(layer_idx=li,kv_head_idx=hi,diagnostic_seqlen=a.sink_diagnostic_seqlen,sink_measure='last_query_first_token',
                    sink_mass_mean=float(sink_mean[li,hi]),sink_mass_std=float(sink_std[li,hi]),
                    sink_presence=int(flag),method='diagnostic_only'))
    with (path.parent/'sink_first_token_by_query_head.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['layer_idx','query_head_idx','kv_head_idx','first_token_attention_mean','first_token_attention_std'])
        w.writeheader()
        for li in range(sink_qhead_mean.shape[0]):
            for qi in range(sink_qhead_mean.shape[1]):
                w.writerow(dict(layer_idx=li,query_head_idx=qi,kv_head_idx=qi//group,
                    first_token_attention_mean=float(sink_qhead_mean[li,qi]),
                    first_token_attention_std=float(sink_qhead_std[li,qi])))
    with (path.parent/'sink_first_token_by_layer.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['layer_idx','first_token_attention_mean','std_across_query_heads'])
        w.writeheader()
        for li,(avg,sd) in enumerate(zip(sink_layer_mean,sink_layer_std_across_heads)):
            w.writerow(dict(layer_idx=li,first_token_attention_mean=float(avg),
                std_across_query_heads=float(sd)))
    print(f'[CALIBRATION COMPLETE] {path} shape={tuple(frozen.shape)} '
          f'sink_heads={int(sink_presence.sum())}/{sink_presence.size} Q_train_max_diagnostic={float(qnorm_bound.max()):.5g}',flush=True)

if __name__=='__main__':main()
