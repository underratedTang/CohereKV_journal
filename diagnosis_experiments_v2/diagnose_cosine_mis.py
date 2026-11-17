#!/usr/bin/env python3
"""Single-round REAL future-Query diagnosis: degree and E-A geometric priority ablations.

KeepKV retained selection, routing, and K-only conflict edges are shared
across priority variants; the three Degree/E-A arms use Count Mean K/V. For identical inputs, Drop and AllCount share
fused K/V and differ only in whether rejected historical Counts survive.
"""
from __future__ import annotations
import argparse,csv,gc,json,platform,time,hashlib
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoConfig,AutoModel,AutoTokenizer
from core_model_utils import (build_full_cache_with_keepkv_scores,cache_key_value_layers,
    first_execution_device,load_token_stream,parse_dtype,set_seed)
from real_query_capture import capture_real_future_queries,replay_prefix_attention,output_errors,require_supported_model_type
from experiment_utils import parse_float_csv,resolve_selection
from cosine_mis_fusion import fuse_plan
from k_only_mis_drop import ee_mis
from routing_load_stats import routing_coverage,concentration
from mis_sweep_settings import sweep_settings,sweep_label
from method_registry import (selected_families,selected_methods,FAMILY_METHODS,
    FAMILY_ALL_METHODS,effective_method,is_sink_gated,family_for_method,
    graph_is_importance_aware,priority_for_family,original_method)
from models.common.mymethod_common import recent_window_importance
from cosine_mis_plan import select_split, route_top1
from calibrate_median_reference import validate_payload, validate_value_payload, validate_query_bounds
from routing_sweep_utils import make_routing_specs,route_spec_cutoff,route_display,summarize


def args_parser():
    p=argparse.ArgumentParser(description='Cosine/L2 routing: K-only degree versus Importance-D MIS; real Q Replay')
    p.add_argument('--model',required=True)
    p.add_argument('--dataset',default='wikitext2')
    p.add_argument('--prefix_len',type=int,default=4096)
    p.add_argument('--future_probe_len',type=int,default=64)
    p.add_argument('--max_sequences',type=int,default=2)
    p.add_argument('--torch_dtype',default='bfloat16')
    p.add_argument('--attn_implementation',default='eager')
    p.add_argument('--device',default='cuda')
    p.add_argument('--device_map',default='auto')
    p.add_argument('--trust_remote_code',action='store_true')
    p.add_argument('--use_fast_tokenizer',action='store_true')
    p.add_argument('--prefill_chunk_size',type=int,default=128)
    p.add_argument('--keep_ratios',default='0.2,0.4,0.6')
    p.add_argument('--ee_lambdas',default='0.3,0.4,0.5,0.6,0.7',
                   help='Shared K-only lambda sweep, identical for BOTH priorities')
    p.add_argument('--mis_families',default='k_only,k_only_importance',
        help='Families of MIS methods; Full Cache and Merge All always included')
    p.add_argument('--mis_methods',default=None,
        help='Explicit comma-separated MIS arms, including optional _sink ablations')
    p.add_argument('--median_reference_file',required=True)
    p.add_argument('--sink_length',type=int,default=8)
    p.add_argument('--sink_presence_threshold',type=float,default=0.05)
    p.add_argument('--calibration_num_sequences',type=int,default=16)
    p.add_argument('--calibration_seqlen',type=int,default=4096)
    p.add_argument('--calibration_seed',type=int,default=42)
    p.add_argument('--routing_modes',default='cosine,l2',
                   help='comma-separated subset of cosine,l2')
    p.add_argument('--cosine_thresholds',default='0.8,0.85',
                   help='cosine R-E routing threshold sweep, dimensionless')
    p.add_argument('--routing_l2_lambdas',default='0.4,0.6,0.8',
                   help='L2 R-E cutoff multiplier of frozen per-layer/head Median S_ref')
    p.add_argument('--routing_l2_distances',default='',
                   help='optional absolute L2 R-E cutoff sweep in physical Key units')
    p.add_argument('--cosine_threshold',type=float,default=None,
                   help='deprecated single cosine cutoff alias; use --cosine_thresholds')
    p.add_argument('--capture_routing_mode',default='cosine',choices=('cosine','l2'))
    p.add_argument('--capture_routing_parameter',type=float,default=None,
                   help='select a single routing threshold for PCA; default captures every routed sweep setting')
    p.add_argument('--capture_routing_parameter_type',default='lambda',choices=('cosine','lambda','distance'))
    p.add_argument('--keepkv_sink_size',type=int,default=4)
    p.add_argument('--keepkv_recent_protect_size',type=int,default=128)
    p.add_argument('--keepkv_score_window',type=int,default=128)
    p.add_argument('--keepkv_recency_beta',type=float,default=1.)
    p.add_argument('--distance_chunk_size',type=int,default=256)
    p.add_argument('--graph_backend',default='gpu_batched',choices=('gpu_batched','cpu_batched','torch','cuda','auto'))
    p.add_argument('--layer_selection',default='0,16')
    p.add_argument('--kv_head_selection',default='0,4')
    p.add_argument('--save_per_probe',type=int,default=1)
    p.add_argument('--output_norm_eps',type=float,default=1e-8)
    p.add_argument('--capture_layer',type=int,default=16)
    p.add_argument('--capture_head',type=int,default=4)
    p.add_argument('--capture_budget',type=float,default=.2)
    p.add_argument('--capture_ee_lambda',type=float,default=.5)
    p.add_argument('--output_dir',required=True)
    return p.parse_args()


def write_csv(path,rows,append=False):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    had=append and path.is_file()
    with path.open('a' if append else 'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        if not had:w.writeheader()
        w.writerows(rows)


def attention_head_header(seq,layer,head,gate_enabled,methods):
    print(f'[ATTENTION] sequence={seq} layer={layer} KV_head={head} '
          f'| Relative-Attention-L2 (lower better) | sink_diagnostic={int(gate_enabled)}',flush=True)
    print('  ' + '-'*117,flush=True)
    print(f'  {"Budget":>6}  {"Routing":<23} {"K λ":>6} {"MIS":<7} '
          f'{"Keep":>5} {"Routed":>6} {"Accept":>6} {"Reject":>6}',flush=True)


def attention_head_row(budget,routing,lam,nkeep,tr,by_method,family,active_methods,sink_presence):
    control=('importance_weighted_merge_all' if 'importance_weighted_merge_all' in by_method else 'merge_all')
    base=float(by_method[control]['output_rel_error'])
    threshold=f'lambda={lam:.3g}; priority={tr_priority(family)}'
    print(f'  {budget:>6.0%}  {routing:<23.23} {lam:>6.2f} {family:<7} '
          f'{nkeep:>5d} {tr.n_routed:>6d} {tr.n_mis_accepted:>6d} {tr.n_mis_rejected:>6d}'
          f'  [{threshold}]',flush=True)
    print(f'    {"Method":<27} {"Relative-Attention-L2":>23} {"Δ L2 vs "+control:>27} {"Δ %":>12}',flush=True)
    for method in dict.fromkeys((control,'merge_all',*active_methods)):
        current=float(by_method[method]['output_rel_error'])
        delta=current-base
        pct=100*delta/max(base,1e-12)
        via=effective_method(method,sink_presence)
        suffix=(f' [uses {via}]' if method.endswith('_sink') else '')
        print(f'    {method:<48} {current:>18.6f} {delta:>+16.6f} {pct:>+10.3f}%{suffix}',flush=True)
    print('  ' + '-'*117,flush=True)


def tr_priority(family):
    return {'k_only':'degree ascending, E-A L2 ascending, hash ascending',
            'k_only_degree_anchor':'(degree+1)*E-A L2^2 ascending; ties degree, E-A, hash',
            'k_only_anchor_first':'E-A L2^2 ascending, degree ascending, hash ascending'}.get(
            family,'W_i - sum E-E-neighbor W_j (W=C*I; D-first)')


def save_example(root,seq,layer,head,budget,lam,triad,keys,values,keep,evict,best,routed,
                 selected,route_mode,route_type,route_value,route_cutoff,mis_family):
    affected=[r for r in triad.group_rows if r['n_mis_rejected']>0]
    family_methods=tuple(m for m in ('merge_all',*FAMILY_METHODS[mis_family]) if m in triad.keys)
    affected.sort(key=lambda r:(-r['n_mis_rejected'],-r['n_routed'],r['anchor_idx']))
    for rank,row in enumerate(affected[:4]):
        a=int(row['anchor_idx']);anchor_pos=int(np.flatnonzero(keep==a)[0])
        mask=routed & (best==anchor_pos)
        members=np.r_[a,evict[mask]]
        flags=np.r_[True,selected[mask]]
        np.savez_compressed(root/'examples'/f'{route_mode}_{route_type}{route_value:g}_{mis_family}_seq{seq}_L{layer}_H{head}_B{budget:g}_lam{lam:g}_A{a}_rank{rank}.npz',
            routing_mode=np.array(route_mode),routing_parameter_type=np.array(route_type),
            routing_parameter=np.array(route_value),routing_l2_cutoff=np.array(route_cutoff),
            original_positions=members.astype(np.int32),
            key_original=keys[members].astype(np.float32),
            value_original=values[members].astype(np.float32),
            mis_selected=flags.astype(np.int8),
            fused_keys=np.stack([triad.keys[m][anchor_pos] for m in family_methods]),
            fused_values=np.stack([triad.values[m][anchor_pos] for m in family_methods]),
            fused_counts=np.array([triad.counts[m][anchor_pos] for m in family_methods],dtype=np.float32),
            fused_names=np.asarray(family_methods))


def main():
    a=args_parser()
    families=selected_families(a.mis_families)
    methods=selected_methods(a.mis_families,a.mis_methods)
    print('[REPLAY METHODS]',methods,flush=True)
    families=tuple(f for f in families if any(family_for_method(m)==f for m in methods))
    if a.prefix_len<64 or a.future_probe_len<1 or a.max_sequences<1:
        raise ValueError('Invalid prefix/future/sequence lengths')
    budgets=sorted(set(parse_float_csv(a.keep_ratios)))
    sweeps=sweep_settings(a.ee_lambdas)
    if not all(0<x<1 for x in budgets):
        raise ValueError('Budget must be in (0,1)')
    routes=make_routing_specs(a)
    if not routes: raise ValueError('At least one routing sweep setting is required')
    root=Path(a.output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    (root/'examples').mkdir(exist_ok=True)
    # The frozen reference is estimated using independent train windows.
    # Fail closed if the provided reference is RMS aggregated, belongs to
    # another model, or has different calibration protocol.
    reference_path=Path(a.median_reference_file).resolve()
    reference_bytes=reference_path.read_bytes()
    calibration_info=json.loads(reference_bytes.decode('utf-8'))
    frozen_scales=validate_payload(calibration_info,model=a.model,
        split='train',num_sequences=a.calibration_num_sequences,
        seqlen=a.calibration_seqlen,seed=a.calibration_seed)
    v_scales=validate_value_payload(calibration_info, frozen_scales.shape)
    q_bounds=validate_query_bounds(calibration_info, frozen_scales.shape)
    if calibration_info.get('dataset')!=a.dataset:
        raise ValueError('Calibration dataset differs from test dataset')
    if calibration_info.get('torch_dtype')!=a.torch_dtype:
        raise ValueError('Calibration model dtype differs from evaluation dtype')
    if calibration_info.get('sink_calibration_schema')!='streamingllm_last_query_first_token_v2':
        raise ValueError('Missing StreamingLLM-style last-query/first-token calibration; recalibrate from train')
    if int(calibration_info.get('sink_length',-1))!=a.sink_length:
        raise ValueError('Sink length mismatch between calibration and diagnosis')
    if abs(float(calibration_info.get('sink_presence_threshold',-1))-a.sink_presence_threshold)>1e-12:
        raise ValueError('Sink presence threshold must be frozen during train calibration')
    sink_calib=np.asarray(calibration_info.get('sink_mass_mean'),dtype=np.float64)
    sink_seq=np.asarray(calibration_info.get('sink_mass_by_sequence'),dtype=np.float64)
    if sink_calib.shape!=frozen_scales.shape or sink_seq.shape!=(a.calibration_num_sequences,*frozen_scales.shape):
        raise ValueError('Sink calibration [sequence,layer,kv_head] shape mismatch')
    if not np.isfinite(sink_calib).all() or (sink_calib<0).any() or (sink_calib>1).any():
        raise ValueError('Invalid Sink calibration')
    if not np.allclose(sink_calib,sink_seq.mean(axis=0),atol=1e-10,rtol=1e-10):
        raise ValueError('Sink calibration values inconsistent with train sequences')
    sink_gate=(sink_calib>=a.sink_presence_threshold)
    calibration_sha256=hashlib.sha256(reference_bytes).hexdigest()
    print(f'[FROZEN MEDIAN] {reference_path} sha256={calibration_sha256} '
          f'head matrix={frozen_scales.shape}',flush=True)
    set_seed(20261021)
    print(f'[MIS BACKEND] {a.graph_backend} (gpu_batched: bounded native torch GPU graph, no huge padded matrices)',flush=True)
    print('[ROUTING SWEEP]',routes,flush=True)
    started=time.time()
    write_csv(root/'mis_sweep_settings.csv',sweeps)
    print('[MIS SWEEP] all priorities share identical K-only lambda:',flush=True)
    for s in sweeps:print('  '+sweep_label(s),flush=True)
    status=dict(experiment='k_only_sink8_train_frozen_kvhead_gate_v1',status='running',arguments=vars(a),
        old_results_used=False,query_independent=True,
        calibration_aggregation='median_across_sequences',
        median_reference_file=str(reference_path),calibration_sha256=calibration_sha256,
        calibration_metadata={k:v for k,v in calibration_info.items() if k not in ('sequence_layer_head_scales','layer_head_scales','sequence_scale_mean','sequence_scale_std','value_sequence_layer_head_scales','value_layer_head_scales','value_sequence_scale_mean','value_sequence_scale_std','sink_mass_by_sequence','sink_mass_mean','sink_mass_std','sink_presence_by_kv_head','query_norm_max_by_sequence','query_norm_observed_train_max_by_kv_head','query_norm_bound_by_kv_head')},
        sink_measure='streamingllm_last_query_first_token_v2',sink_length=a.sink_length,sink_presence_threshold=a.sink_presence_threshold,
        n_sink_heads=int(sink_gate.sum()),n_kv_heads=int(sink_gate.size),
        native_method='Same E-E geometric graph; degree-first, degree-anchor product, and anchor-first differ only in deterministic greedy priority; Count mean + Drop shared',
        state='SINGLE ROUND: every prefix token starts C=1, R=0, D=0; no multi-round carryover',
        routing_semantics='Cosine: top-1 maximum cosine >= threshold; L2: top-1 nearest euclidean <= cutoff (either absolute or lambda * frozen per-layer/head MEDIAN S_ref). Only one routing mode is applied in each paired config.',
        lambda_semantics='All MIS families: E-E L2 > lambda*S_ref. Routing unchanged. No E-A gate.',
        mis_sweep_settings=sweeps,
        routing_specs=[dict(mode=m,parameter_type=k,parameter=v) for m,k,v in routes],
        arms=list(methods),mis_families=list(families),
        sink_gating='*_sink methods: train-Sink physical KV head -> matching native MIS; non-Sink -> exact Merge All; fixed for evaluation',
        sink_calibration_note='StreamingLLM-style train-only LAST Query Full Attention to FIRST token at fixed diagnostic sequence length, measured per Q head and averaged within KV head; fixed threshold is our heuristic (NOT from original paper); no test Q enters compression; sink gates only *_sink method variants; native methods unaffected',
        treatment='K-only E-E MIS; rejected K/V excluded in Drop and AllCount; Drop Count=Anchor+Accepted, AllCount Count=whole routed group',
        v_reference_definition='Frozen TRAIN median pairwise RMS physical Value L2; independent of Key reference',
        query_norm_diagnostic='Frozen TRAIN-only post-RoPE max Query norm per layer/GQA KV head x safety factor; recorded as a diagnostic, never used by either MIS priority',
        importance_priority='Importance D-first W=C*I; three degree/anchor priorities ignore importance. All share K-only E-E geometry; replay C=1.',
        query_source='Actual next-text, post-RoPE GQA Query captured from uncompressed original Llama/Qwen2 (Qwen2.5), with cache-K cross-validation; never enters plan/fusion',
        replay='Teacher-forced fixed-Q replay vs original uncompressed prefix and causally visible continuation K/V; not compressed-cache autoregressive rollout/PPL',
        output_metric='relative L2 output error, averaged over all GQA query heads and future token offsets')
    (root/'run_config.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    cfg=AutoConfig.from_pretrained(a.model,trust_remote_code=a.trust_remote_code)
    expected=(int(cfg.num_hidden_layers),int(cfg.num_key_value_heads))
    if frozen_scales.shape!=expected:
        raise ValueError(f'Median reference shape={frozen_scales.shape} but model expects {expected}')
    require_supported_model_type(cfg.model_type)
    kwargs=dict(config=cfg,torch_dtype=parse_dtype(a.torch_dtype),low_cpu_mem_usage=True,
        trust_remote_code=a.trust_remote_code,attn_implementation=a.attn_implementation)
    if a.device_map.lower()!='none':kwargs['device_map']=a.device_map
    print('[MODEL]',a.model,flush=True)
    model=AutoModel.from_pretrained(a.model,**kwargs).eval()
    if a.device_map.lower()=='none':model.to(torch.device(a.device))
    device=first_execution_device(model,a.device)
    compute_device=torch.device('cuda' if a.device=='cuda' and torch.cuda.is_available() else 'cpu')
    tokenizer=AutoTokenizer.from_pretrained(a.model,use_fast=a.use_fast_tokenizer,
        trust_remote_code=a.trust_remote_code,padding_side='left')
    token_stream=load_token_stream(tokenizer,a.dataset)
    stride=a.prefix_len+a.future_probe_len
    nseq=min(a.max_sequences,int(token_stream.shape[1])//stride)
    if nseq<1:raise ValueError('Not enough contiguous tokens for prefix+continuation')
    all_rows=[];group_total=0;rej_total=0;probe_total=0
    # Train-only sink map: fixed physical-KV-head gate for *_sink arms.
    write_csv(root/'sink_presence_by_kv_head.csv',[
        dict(layer_idx=li,kv_head_idx=hi,train_sink_mass=float(sink_calib[li,hi]),
             sink_presence=int(sink_gate[li,hi]),method='train_frozen_sink_gate',method_nearest='train_frozen_sink_gate',
             threshold=a.sink_presence_threshold,sink_measure='last_query_first_token',diagnostic_seqlen=int(calibration_info['sink_diagnostic_seqlen']))
        for li in range(sink_gate.shape[0]) for hi in range(sink_gate.shape[1])])
    selected_layers=None;selected_heads=None
    hratio=int(cfg.num_attention_heads)//int(cfg.num_key_value_heads)
    if hratio*int(cfg.num_key_value_heads)!=int(cfg.num_attention_heads):
        raise RuntimeError('Noninteger GQA ratio')
    for seq in range(nseq):
        start=seq*stride
        print(f'[SEQ {seq}] prefix {start}:{start+a.prefix_len}, future={a.future_probe_len}',flush=True)
        prefix=token_stream[:,start:start+a.prefix_len]
        future=token_stream[:,start+a.prefix_len:start+stride]
        # Historical q_proj hooks ONLY span the original prefix; future-token
        # Queries are captured later, after hooks are removed, for evaluation.
        past,scores,_=build_full_cache_with_keepkv_scores(model,prefix,device,
            a.prefill_chunk_size,a.keepkv_score_window,cfg.model_type)
        original_key_layers,_=cache_key_value_layers(past)
        if selected_layers is None:selected_layers=resolve_selection(a.layer_selection,len(original_key_layers))
        if selected_heads is None:selected_heads=resolve_selection(a.kv_head_selection,int(cfg.num_key_value_heads))
        true_queries,full_past=capture_real_future_queries(model,future,past,device,cfg,selected_layers)
        full_keys,full_values=cache_key_value_layers(full_past)
        for layer in selected_layers:
            khs=full_keys[layer][0];vhs=full_values[layer][0]
            if khs.shape[-2]!=stride or vhs.shape[-2]!=stride:
                raise RuntimeError('Future original KV cache length mismatch')
            for head in selected_heads:
                attention_head_header(seq,layer,head,bool(sink_gate[layer,head]),methods)
                orig_k=khs[head,:a.prefix_len].detach().cpu().float().numpy().copy()
                orig_v=vhs[head,:a.prefix_len].detach().cpu().float().numpy().copy()
                fk=khs[head,a.prefix_len:].detach().cpu().float()
                fv=vhs[head,a.prefix_len:].detach().cpu().float()
                q=true_queries[layer][head*hratio:(head+1)*hratio]
                if q.shape[0]!=hratio or q.shape[1]!=a.future_probe_len:
                    raise RuntimeError('Query/GQA shape mismatch')
                q_norms=torch.linalg.vector_norm(q.float(),dim=-1)
                q_max=float(q_norms.max().item())
                q_exceed=float((q_norms>float(q_bounds[layer,head])).float().mean().item())
                if q_exceed>0:
                    print(f'[QUERY NORM EXCEED] seq={seq} L={layer} H={head} '
                          f'bound={q_bounds[layer,head]:.5g} test_max={q_max:.5g} exceed={q_exceed:.3%}',flush=True)
                exact=replay_prefix_attention(q,orig_k,orig_v,
                    np.ones(a.prefix_len,dtype=np.float32),fk,fv,compute_device,a.output_norm_eps)
                score=scores[layer][0,head].to(khs.device,dtype=torch.float32)
                importance=recent_window_importance(score[None],ptr=0,beta=a.keepkv_recency_beta)[0]
                positions=torch.arange(a.prefix_len,device=khs.device,dtype=torch.long).unsqueeze(0)
                plan_keys=khs[head,:a.prefix_len].unsqueeze(0).float()
                # Freeze a head-specific calibration reference across ALL test
                # sequences, budgets and E-E lambda values. No test Key distance
                # statistics can alter the operational K-only E-E threshold.
                reference=torch.tensor([[float(frozen_scales[layer,head])]],
                    dtype=torch.float32,device=plan_keys.device)
                for budget in budgets:
                    nkeep=max(1,min(a.prefix_len-1,int(round(budget*a.prefix_len))))
                    # KeepKV split is shared by all route settings. Each routing
                    # setting then stays fixed across its independent E-E sweep.
                    keep_t,evict_t=select_split(plan_keys,importance.unsqueeze(0),positions,
                        nkeep,a.keepkv_sink_size,a.keepkv_recent_protect_size)
                    # One Top-1 nearest-anchor operation per routing mode.
                    # A threshold sweep only filters that fixed assignment;
                    # the best anchor and physical distances do not depend
                    # on the acceptance threshold. Saves repeated GPU work.
                    mode_cache={}
                    for mode in sorted({spec[0] for spec in routes}):
                        eligible=[route_spec_cutoff(spec,float(reference.item()))
                                  for spec in routes if spec[0]==mode]
                        broadest=min(eligible) if mode=='cosine' else max(eligible)
                        mode_cache[mode]=route_top1(
                            plan_keys,keep_t,evict_t,mode,broadest,a.distance_chunk_size)
                    for spec in routes:
                        mode,kind,param=spec
                        cutoff=route_spec_cutoff(spec,float(reference.item()))
                        base_best,base_cos,base_mask,base_dist=mode_cache[mode]
                        routed_t=base_mask & ((base_cos>=cutoff) if mode=='cosine'
                                              else (base_dist<=cutoff))
                        best_t=base_best  # unassigned entries are masked out
                        cos_t=torch.where(routed_t,base_cos,torch.zeros_like(base_cos))
                        dist_t=torch.where(routed_t,base_dist,torch.zeros_like(base_dist))
                        keep=keep_t[0].detach().cpu().numpy()
                        evict=evict_t[0].detach().cpu().numpy()
                        best=best_t[0].detach().cpu().numpy()
                        routed=routed_t[0].detach().cpu().numpy()
                        cosine_scores=cos_t[0].detach().cpu().numpy()
                        # Explicit exact invariance checks across E-E lambda settings.
                        fixed_controls={}
                        for sweep in sweeps:
                            lam=sweep['ee_lambda']
                            # One current original K/V state and one fixed routing plan;
                            # each MIS family independently builds its own E-E graph.
                            all_current=[];all_current_probes=[]
                            baseline=None
                            iw_baseline=None
                            degree_mask=None
                            degree_edge_counts=None
                            for family in families:
                                selected_family=tuple(m for m in FAMILY_ALL_METHODS[family] if m in methods)
                                native_required=tuple(m for m in FAMILY_METHODS[family]
                                                      if any(original_method(x)==m for x in selected_family))
                                calc_methods=('merge_all',*(control for control in
                                    ('importance_weighted_merge_all',) if control in methods),*native_required)
                                dp=ee_mis(plan_keys[0],torch.from_numpy(orig_v).to(plan_keys.device),
                                    keep_t[0],evict_t[0],best_t[0],routed_t[0],dist_t[0],
                                    float(frozen_scales[layer,head]),1.0,lam,a.graph_backend,
                                    counts=torch.ones(orig_k.shape[0],device=plan_keys.device),
                                    importance=importance,priority=priority_for_family(family))
                                selected=dp.accepted.cpu().numpy()
                                gate_active=True
                                routed_coverage=routing_coverage(int(routed.sum()),len(evict))
                                risk_load=concentration(best,routed,len(keep))
                                edge_counts=dp.conflict_edges.cpu().numpy()
                                if family=='k_only':
                                    degree_mask=selected.copy()
                                    degree_edge_counts=edge_counts.copy()
                                if degree_mask is not None and family in ('k_only_importance', 'k_only_degree_anchor', 'k_only_anchor_first'):
                                    if not np.array_equal(edge_counts,degree_edge_counts):
                                        raise AssertionError('MIS priority changed conflict edge counts')
                                    n_changed_selections=int(np.count_nonzero(selected!=degree_mask))
                                else:
                                    n_changed_selections=0
                                tr=fuse_plan(orig_k,orig_v,keep,evict,best,routed,selected,
                                    dp.pool_sizes.cpu().numpy(),
                                    dp.selected_sizes.cpu().numpy(),
                                    dp.conflict_edges.cpu().numpy(),mis_family=family,
                                    importance=importance.detach().cpu().numpy(),
                                    methods_to_compute=calc_methods)
                                common=dict(sequence_idx=seq,layer_idx=layer,kv_head_idx=head,
                                    budget=budget,routing_mode=mode,routing_parameter_type=kind,
                                    routing_parameter=param,routing_l2_cutoff=(cutoff if mode=='l2' else 0.0),
                                    cosine_threshold=(param if mode=='cosine' else ''),
                                    ee_lambda=lam,sweep_id=sweep['sweep_id'],
                                    n_keep=nkeep,mis_family=family,
                                    routed_coverage=routed_coverage,
                                    **risk_load,
                                    calibrated_sink_mass=float(sink_calib[layer,head]),
                                    sink_presence=int(sink_gate[layer,head]),
                                    reference_scale=float(reference.item()),
                                    value_reference_scale=float(v_scales[layer,head]),
                                    calibration_aggregation='median_across_sequences',
                                    calibration_num_sequences=a.calibration_num_sequences,
                                    calibration_seqlen=a.calibration_seqlen,
                                    ee_delta=float(dp.k_delta),v_ee_delta='',
                                    mis_priority=tr_priority(family),
                                    n_routed=tr.n_routed,n_mis_accepted=tr.n_mis_accepted,
                                    n_mis_rejected=tr.n_mis_rejected,
                                    n_k_only_accepted=(tr.n_mis_accepted if family in ('k_only', 'k_only_degree_anchor', 'k_only_anchor_first') else 0),
                                    n_k_only_rejected=(tr.n_mis_rejected if family in ('k_only', 'k_only_degree_anchor', 'k_only_anchor_first') else 0),
                                    n_importance_mis_accepted=(tr.n_mis_accepted if family not in ('k_only','k_only_degree_anchor','k_only_anchor_first') else 0),
                                    n_importance_mis_rejected=(tr.n_mis_rejected if family not in ('k_only','k_only_degree_anchor','k_only_anchor_first') else 0),
                                    routed_importance_sum=dp.routed_importance_sum,
                                    accepted_importance_sum=float(importance.detach().cpu().numpy()[evict[selected]].sum()),
                                    rejected_importance_sum=float(importance.detach().cpu().numpy()[evict[routed & ~selected]].sum()),
                                    accepted_importance_fraction=(float(importance.detach().cpu().numpy()[evict[selected]].sum()) /
                                        max(dp.routed_importance_sum,1e-12)),
                                    accepted_mask_diff_vs_degree=n_changed_selections,
                                    reject_count_mass=float(tr.n_mis_rejected),
                                    reject_count_fraction=tr.n_mis_rejected/max(1,nkeep+tr.n_routed),
                                    routed_fraction=tr.n_routed/max(1,len(evict)),
                                    mean_routed_cosine=float(np.mean(cosine_scores[routed])) if np.any(routed) else 0.0,
                                    mis_rejected_fraction=tr.n_mis_rejected/max(1,tr.n_routed),
                                    n_affected_groups=sum(int(g['n_mis_rejected']>0) for g in tr.group_rows),
                                    n_native_conflict_edges=int(dp.conflict_edges.sum().item()),
                                    n_original_ee_conflict_edges=dp.original_ee_conflict_edges,
                                    n_original_eligible_ee_conflict_edges=dp.original_eligible_ee_edges,
                                    n_importance_ee_conflict_edges=dp.weighted_ee_conflict_edges,
                                    n_ea_compatibility_rejected=dp.ea_rejected,
                                    ea_rejected_importance_sum=dp.ea_routed_importance_rejected,
                                    conflict_pruned_fraction=(1.-dp.weighted_ee_conflict_edges/max(1,dp.original_ee_conflict_edges)
                                         if graph_is_importance_aware(family) else 0.),
                                    test_query_norm_max=q_max,
                                    test_query_norm_exceed_fraction=q_exceed)
                                if (seq==0 and layer==a.capture_layer and head==a.capture_head and
                                    abs(budget-a.capture_budget)<1e-7 and
                                    abs(lam-a.capture_ee_lambda)<1e-7 and
                                    (a.capture_routing_parameter is None or
                                     (mode==a.capture_routing_mode and
                                      abs(param-a.capture_routing_parameter)<1e-7 and
                                      (mode=='cosine' or kind==a.capture_routing_parameter_type)))):
                                    save_example(root,seq,layer,head,budget,lam,tr,orig_k,orig_v,
                                        keep,evict,best,routed,selected,mode,kind,param,cutoff,family)
                                group_common={
                                    (f'config_{key}' if key in ('n_routed','n_mis_accepted','n_mis_rejected') else key):value
                                    for key,value in common.items()
                                }
                                group_rows=[]
                                for g in tr.group_rows:
                                    anchor_owner=int(np.flatnonzero(keep==g['anchor_idx'])[0])
                                    group_rows.append({**group_common,**g,
                                        'n_ea_compatibility_rejected_group':int(dp.ea_rejected_by_group[anchor_owner].item()),
                                        'n_original_conflict_edges_group':int(dp.original_ee_edges_by_group[anchor_owner].item()),
                                        'n_ia_conflict_edges_group':int(dp.conflict_edges[anchor_owner].item()),
                                        'fusion_scope':'MIS_active'})
                                if family=='k_only_importance_iwmean':
                                    from mis_attention_diagnostics import group_future_query_diagnostics
                                    detail,addbacks=group_future_query_diagnostics(
                                        orig_k,orig_v,importance.detach().cpu().numpy(),
                                        keep,evict,best,routed,selected,q,
                                        tr,family,max_groups=8,max_addbacks_per_group=3)
                                    write_csv(root/'mis_group_future_query_diagnostics.csv',
                                        [{**group_common,**r} for r in detail],append=True)
                                    write_csv(root/'mis_rejected_addback_diagnostics.csv',
                                        [{**group_common,**r} for r in addbacks],append=True)
                                rejected_rows=[{**common,**r} for r in tr.rejected_rows]
                                write_csv(root/'cosine_mis_by_group.csv',group_rows,append=True)
                                write_csv(root/'cosine_mis_rejected.csv',rejected_rows,append=True)
                                group_total+=len(group_rows);rej_total+=len(rejected_rows)
                                if any(np.any(c<=0) for c in tr.counts.values()) or tr.n_mis_accepted+tr.n_mis_rejected!=tr.n_routed:
                                    raise AssertionError('Broken MIS/Count invariants')
                                by_method={};by_probe={};by_square={}
                                if baseline is None:
                                    full_stats,full_probes=output_errors(exact,exact,a.output_norm_eps)
                                    by_method['full_cache']=full_stats
                                    by_probe['full_cache']=full_probes
                                    by_square['full_cache']=0.
                                    approx=replay_prefix_attention(q,tr.keys['merge_all'],tr.values['merge_all'],
                                        tr.counts['merge_all'],fk,fv,compute_device,a.output_norm_eps)
                                    stats,probes=output_errors(approx,exact,a.output_norm_eps)
                                    by_method['merge_all']=stats;by_probe['merge_all']=probes
                                    rel2=(approx-exact).square().sum(-1)/exact.square().sum(-1).clamp_min(a.output_norm_eps**2)
                                    by_square['merge_all']=float(rel2.mean().item())
                                    baseline=(by_method['merge_all'],by_probe['merge_all'],by_square['merge_all'],tr.keys['merge_all'].copy(),tr.sse['merge_all'],tr.v_sse['merge_all'])
                                else:
                                    # MergeAll must remain independent of MIS family and E-E threshold.
                                    if not np.array_equal(baseline[3],tr.keys['merge_all']):
                                        raise AssertionError('Merge All changed across MIS families')
                                    by_method['merge_all']=baseline[0]
                                    by_probe['merge_all']=baseline[1]
                                    by_square['merge_all']=baseline[2]
                                if 'importance_weighted_merge_all' in methods:
                                    ctrl='importance_weighted_merge_all'
                                    if iw_baseline is None:
                                        approx_iw=replay_prefix_attention(q,tr.keys[ctrl],tr.values[ctrl],
                                            tr.counts[ctrl],fk,fv,compute_device,a.output_norm_eps)
                                        iw_stats,iw_probes=output_errors(approx_iw,exact,a.output_norm_eps)
                                        rel2_iw=(approx_iw-exact).square().sum(-1)/exact.square().sum(-1).clamp_min(a.output_norm_eps**2)
                                        iw_baseline=(iw_stats,iw_probes,float(rel2_iw.mean().item()),tr.keys[ctrl].copy())
                                    else:
                                        if not np.array_equal(iw_baseline[3],tr.keys[ctrl]):
                                            raise AssertionError('Importance Merge All changed across families')
                                    by_method[ctrl]=iw_baseline[0]
                                    by_probe[ctrl]=iw_baseline[1]
                                    by_square[ctrl]=iw_baseline[2]
                                for method in native_required:
                                    approx=replay_prefix_attention(q,tr.keys[method],tr.values[method],
                                        tr.counts[method],fk,fv,compute_device,a.output_norm_eps)
                                    stats,probes=output_errors(approx,exact,a.output_norm_eps)
                                    by_method[method]=stats;by_probe[method]=probes
                                    rel2=(approx-exact).square().sum(-1)/exact.square().sum(-1).clamp_min(a.output_norm_eps**2)
                                    by_square[method]=float(rel2.mean().item())
                                # Train-frozen sink mask only: no evaluation Query affects
                                # the branch. The proxy is an *exact* original MIS or
                                # an *exact* Merge All fusion for this one KV head.
                                for sink_method in selected_family:
                                    if not is_sink_gated(sink_method):continue
                                    proxy=effective_method(sink_method,bool(sink_gate[layer,head]))
                                    by_method[sink_method]=by_method[proxy]
                                    by_probe[sink_method]=by_probe[proxy]
                                    by_square[sink_method]=by_square[proxy]
                                for ctrl in ('merge_all',*(c for c in
                                    ('importance_weighted_merge_all',) if c in methods)):
                                    if ctrl not in fixed_controls:
                                        fixed_controls[ctrl]=(tr.keys[ctrl].copy(),by_method[ctrl]['output_rel_error'])
                                    else:
                                        saved_key,saved_error=fixed_controls[ctrl]
                                        if not np.array_equal(saved_key,tr.keys[ctrl]) or saved_error!=by_method[ctrl]['output_rel_error']:
                                            raise AssertionError(f'{ctrl} changed when ONLY E-E lambda changed')
                                base=by_method['merge_all']['output_rel_error']
                                base_probe=by_probe['merge_all']
                                iw_control=by_method.get('importance_weighted_merge_all')
                                iw_base=(iw_control['output_rel_error'] if iw_control is not None else None)
                                iw_probe=by_probe.get('importance_weighted_merge_all')
                                output_methods=(('full_cache','merge_all',
                                     *(c for c in ('importance_weighted_merge_all',)
                                       if c in methods),*selected_family)
                                     if family==families[0] else selected_family)
                                for method in output_methods:
                                    met=by_method[method];per=by_probe[method]
                                    diff=float(met['output_rel_error']-base)
                                    diff_iw=(float(met['output_rel_error']-iw_base)
                                             if iw_base is not None else float('nan'))
                                    proxy=effective_method(method,bool(sink_gate[layer,head]))
                                    if method=='full_cache':
                                        drift=float('nan');sse=float('nan');v_sse=float('nan')
                                    else:
                                        drift=float(np.mean(np.linalg.norm(tr.keys[proxy].astype(np.float64)-orig_k[keep].astype(np.float64),axis=-1)))
                                        sse=float(tr.sse[proxy]);v_sse=float(tr.v_sse[proxy])
                                    row_common=(dict(common,mis_family='baseline',n_mis_accepted=0,n_mis_rejected=0,
                                        n_k_only_accepted=0,n_k_only_rejected=0,
                                        n_importance_mis_accepted=0,n_importance_mis_rejected=0,
                                        mis_priority='baseline',routed_importance_sum=0.,
                                        accepted_importance_sum=0.,rejected_importance_sum=0.,
                                        accepted_importance_fraction=0.,
                                        accepted_mask_diff_vs_degree=0,
                                        reject_count_mass=0.,reject_count_fraction=0.,mis_rejected_fraction=0.,
                                        n_affected_groups=0,n_native_conflict_edges=0)
                                        if proxy in ('full_cache','merge_all','importance_weighted_merge_all') else common)
                                    all_current.append(dict(**row_common,method=method,
                                        sink_gate_applied=int(is_sink_gated(method)),
                                        effective_fusion_method=proxy,
                                        delta_rel_from_merge_all=diff,
                                        relative_improvement_pct=100.*(-diff)/max(base,1e-12),
                                        primary_baseline=('importance_weighted_merge_all' if iw_base is not None else 'merge_all'),
                                        delta_rel_from_iw_merge_all=diff_iw,
                                        relative_improvement_iw_pct=(100.*(-diff_iw)/max(iw_base,1e-12) if iw_base is not None else float('nan')),
                                        probe_win_fraction_vs_iw=(float(np.mean(per<iw_probe-1e-10)) if iw_probe is not None else float('nan')),
                                        probe_win_fraction=float(np.mean(per<base_probe-1e-10)),
                                        reconstruction_sse_original_key=sse,
                                        reconstruction_sse_original_value=v_sse,
                                        mean_anchor_drift=drift,
                                        output_rel_squared_error=by_square[method],
                                        source_fusion_method=proxy,**met))
                                    if a.save_per_probe:
                                        for p,(err,br) in enumerate(zip(per,base_probe)):
                                            all_current_probes.append(dict(**row_common,method=method,
                                                sink_gate_applied=int(is_sink_gated(method)),
                                                effective_fusion_method=proxy,
                                                future_token_offset=p,mean_rel_error=float(err),
                                                delta_rel_from_merge_all=float(err-br),
                                                delta_rel_from_iw_merge_all=(float(err-iw_probe[p]) if iw_probe is not None else float('nan'))))
                                attention_head_row(budget,route_display(spec,cutoff),lam,
                                                   nkeep,tr,by_method,family,selected_family,
                                                   bool(sink_gate[layer,head]))
                                del dp,tr
                            all_rows.extend(all_current)
                            write_csv(root/'cosine_mis_by_config.csv',all_current,append=True)
                            if all_current_probes:
                                write_csv(root/'cosine_mis_by_probe.csv',all_current_probes,append=True)
                                probe_total+=len(all_current_probes)
                del orig_k,orig_v,exact
                gc.collect()
        del past,full_past,scores,full_keys,full_values,true_queries
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    write_csv(root/'cosine_mis_overall.csv',summarize(all_rows))
    # Extra paired summaries keep all four arms explicit (no layer selection
    # hidden in an overall mean). Keep the independent routing/EE sweeps.
    from collections import defaultdict
    from statistics import mean as arithmetic_mean
    for grouping,name in [
        (('layer_idx','method'),'cosine_mis_by_layer.csv'),
        (('layer_idx','kv_head_idx','method'),'cosine_mis_by_layer_head.csv'),
        (('layer_idx','kv_head_idx','budget','method'),'cosine_mis_by_layer_head_budget.csv')]:
        groups=defaultdict(list)
        for row in all_rows:groups[tuple(row[k] for k in grouping)].append(row)
        summary=[]
        for key,bucket in sorted(groups.items()):
            out=dict(zip(grouping,key))
            out['n_configs']=len(bucket)
            for field in ('output_rel_error','output_rel_squared_error','output_abs_error','output_rmse',
                          'delta_rel_from_merge_all','delta_rel_from_iw_merge_all'):
                out[field]=arithmetic_mean(float(r[field]) for r in bucket)
            summary.append(out)
        write_csv(root/name,summary)
    status.update(status='completed',n_sequences=nseq,n_configuration_rows=len(all_rows),
        n_group_rows=group_total,n_rejected_rows=rej_total,n_probe_rows=probe_total,
        runtime_seconds=time.time()-started,
        selected_layers=selected_layers,selected_heads=selected_heads,
        reference_scale_source='median_across_train_sequences_frozen_per_layer_head',
        torch_version=torch.__version__,transformers_version=transformers.__version__,
        platform=platform.platform())
    (root/'run_config.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(f'[COMPLETE] K-only/Importance-MIS configs={len(all_rows)}, groups={group_total}, rejected={rej_total}, real probes={probe_total} -> {root}',flush=True)

if __name__=='__main__':main()
