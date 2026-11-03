#!/usr/bin/env python3
"""CohereKV KeepKV + independently swept Cosine/L2 R-E routing + L2 E-E MIS.

R=0, D=0, C=1: one compression round (no inherited radius / count state).
Fixed routing assignment within each (sequence, layer, KV head, budget, routing setting).
MIS E-E threshold scanned independently and only changes which rejected Keys are
replaced by the anchor. Three paired arms:
 (1) routed Merge-All, (2) MIS Rejected-to-Anchor Key, (3) All-to-Anchor Key.
All routed V, Count/logC, retain indices, and causal future K/V are identical.
The original model supplies real post-RoPE future Queries ONLY for posthoc fixed-Q
Attention replay. This does not measure compressed autoregressive PPL.
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
from cosine_mis_fusion import METHODS,fuse_plan
from models.common.mymethod_common import recent_window_importance
from cosine_mis_plan import select_split, route_top1, make_plan
from calibrate_median_reference import validate_payload
from routing_sweep_utils import make_routing_specs,route_spec_cutoff,route_display,summarize


def args_parser():
    p=argparse.ArgumentParser(description='Cosine/L2 routing sweeps; independent native E-E L2 MIS; three K-only arms, real Q')
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
    p.add_argument('--ee_lambdas',default='0.4,0.5,0.6,0.7')
    p.add_argument('--median_reference_file',required=True)
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
    p.add_argument('--graph_backend',default='cpu_batched',choices=('cpu_batched','torch','cuda','auto'))
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


def save_example(root,seq,layer,head,budget,lam,triad,keys,keep,evict,best,routed,selected,route_mode,route_type,route_value,route_cutoff):
    # Choose actual native-MIS-rejected group(s), not a generic PCA example.
    affected=[r for r in triad.group_rows if r['n_mis_rejected']>0]
    affected.sort(key=lambda r:(-r['n_mis_rejected'],-r['n_routed'],r['anchor_idx']))
    for rank,row in enumerate(affected[:4]):
        a=int(row['anchor_idx']);anchor_pos=int(np.flatnonzero(keep==a)[0])
        mask=routed & (best==anchor_pos)
        members=np.r_[a,evict[mask]]
        flags=np.r_[True,selected[mask]]
        indices=anchor_pos
        np.savez_compressed(root/'examples'/f'{route_mode}_{route_type}{route_value:g}_seq{seq}_L{layer}_H{head}_B{budget:g}_lam{lam:g}_A{a}_rank{rank}.npz',
            routing_mode=np.array(route_mode),routing_parameter_type=np.array(route_type),
            routing_parameter=np.array(route_value),routing_l2_cutoff=np.array(route_cutoff),
            original_positions=members.astype(np.int32),
            key_original=keys[members].astype(np.float32),
            mis_selected=flags.astype(np.int8),
            fused_keys=np.stack([triad.keys[m][indices] for m in METHODS]),
            fused_names=np.asarray(METHODS))


def main():
    a=args_parser()
    if a.prefix_len<64 or a.future_probe_len<1 or a.max_sequences<1:
        raise ValueError('Invalid prefix/future/sequence lengths')
    budgets=sorted(set(parse_float_csv(a.keep_ratios)))
    lamvals=sorted(set(parse_float_csv(a.ee_lambdas)))
    if not all(0<x<1 for x in budgets) or not all(x>0 for x in lamvals):
        raise ValueError('Budget must be in (0,1) and E-E MIS lambda > 0')
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
    if calibration_info.get('dataset')!=a.dataset:
        raise ValueError('Calibration dataset differs from test dataset')
    if calibration_info.get('torch_dtype')!=a.torch_dtype:
        raise ValueError('Calibration model dtype differs from evaluation dtype')
    calibration_sha256=hashlib.sha256(reference_bytes).hexdigest()
    print(f'[FROZEN MEDIAN] {reference_path} sha256={calibration_sha256} '
          f'head matrix={frozen_scales.shape}',flush=True)
    set_seed(20261021)
    print(f'[MIS BACKEND] {a.graph_backend} (cpu_batched: bucketed native CPU graph MIS; no large CUDA conflict matrices)',flush=True)
    print('[ROUTING SWEEP]',routes,flush=True)
    started=time.time()
    status=dict(experiment='dual_routing_mis_median_realq_v4',status='running',arguments=vars(a),
        old_results_used=False,query_independent=True,
        calibration_aggregation='median_across_sequences',
        median_reference_file=str(reference_path),calibration_sha256=calibration_sha256,
        calibration_metadata={k:v for k,v in calibration_info.items() if k not in ('sequence_layer_head_scales','layer_head_scales','sequence_scale_mean','sequence_scale_std')},
        native_method='UNMODIFIED CohereKV KeepKV selector and deterministic degree-priority E-E L2 MIS; memory-bounded independent CPU batches; independent Cosine/L2 Top1 routing in diagnosis_experiments',
        state='SINGLE ROUND: every prefix token starts C=1, R=0, D=0; no multi-round carryover',
        routing_semantics='Cosine: top-1 maximum cosine >= threshold; L2: top-1 nearest euclidean <= cutoff (either absolute or lambda * frozen per-layer/head MEDIAN S_ref). Only one routing mode is applied in each paired config.',
        lambda_semantics='E-E conflicts ALWAYS use independent physical L2 > ee_lambda * frozen train Median S_ref; same R-E routing plan reused across all E-E lambdas.',
        routing_specs=[dict(mode=m,parameter_type=k,parameter=v) for m,k,v in routes],
        arms=list(METHODS),
        treatment='MIS rejection changes only merged Key: rejected K is replaced by retained A Key; original rejected Value and original count remain',
        negative_control='all_to_anchor replaces ALL routed Evicted Key with retained A Key, but retains original V and count',
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
    selected_layers=None;selected_heads=None
    hratio=int(cfg.num_attention_heads)//int(cfg.num_key_value_heads)
    if hratio*int(cfg.num_key_value_heads)!=int(cfg.num_attention_heads):
        raise RuntimeError('Noninteger GQA ratio')
    for seq in range(nseq):
        start=seq*stride
        print(f'[SEQ {seq}] prefix {start}:{start+a.prefix_len}, future={a.future_probe_len}',flush=True)
        prefix=token_stream[:,start:start+a.prefix_len]
        future=token_stream[:,start+a.prefix_len:start+stride]
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
                print(f'[HEAD] seq={seq} layer={layer} KVhead={head}',flush=True)
                orig_k=khs[head,:a.prefix_len].detach().cpu().float().numpy().copy()
                orig_v=vhs[head,:a.prefix_len].detach().cpu().float().numpy().copy()
                fk=khs[head,a.prefix_len:].detach().cpu().float()
                fv=vhs[head,a.prefix_len:].detach().cpu().float()
                q=true_queries[layer][head*hratio:(head+1)*hratio]
                if q.shape[0]!=hratio or q.shape[1]!=a.future_probe_len:
                    raise RuntimeError('Query/GQA shape mismatch')
                exact=replay_prefix_attention(q,orig_k,orig_v,
                    np.ones(a.prefix_len,dtype=np.float32),fk,fv,compute_device,a.output_norm_eps)
                score=scores[layer][0,head].to(khs.device,dtype=torch.float32)
                importance=recent_window_importance(score[None],ptr=0,beta=a.keepkv_recency_beta)[0]
                positions=torch.arange(a.prefix_len,device=khs.device,dtype=torch.long).unsqueeze(0)
                plan_keys=khs[head,:a.prefix_len].unsqueeze(0).float()
                # Freeze a head-specific calibration reference across ALL test
                # sequences, budgets and E-E lambda values. No test Key distance
                # statistics can alter the operational E-E MIS threshold.
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
                        for lam in lamvals:
                            plan=make_plan(plan_keys,keep_t,evict_t,best_t,cos_t,routed_t,
                                dist_t,reference,lam,a.graph_backend)
                            selected=plan.accepted[0].detach().cpu().numpy()
                            tr=fuse_plan(orig_k,orig_v,keep,evict,best,routed,selected,
                                plan.candidate_pool_sizes[0].detach().cpu().numpy(),
                                plan.selected_per_anchor[0].detach().cpu().numpy(),
                                plan.conflict_edges_per_anchor[0].detach().cpu().numpy())
                            common=dict(sequence_idx=seq,layer_idx=layer,kv_head_idx=head,
                                budget=budget,routing_mode=mode,routing_parameter_type=kind,
                                routing_parameter=param,routing_l2_cutoff=(cutoff if mode=='l2' else 0.0),
                                cosine_threshold=(param if mode=='cosine' else ''),
                                ee_lambda=lam,n_keep=nkeep,
                                reference_scale=float(reference.item()),
                                calibration_aggregation='median_across_sequences',
                                calibration_num_sequences=a.calibration_num_sequences,
                                calibration_seqlen=a.calibration_seqlen,
                                ee_delta=float(plan.ee_delta[0,0].item()),
                                n_routed=tr.n_routed,n_mis_accepted=tr.n_mis_accepted,
                                n_mis_rejected=tr.n_mis_rejected,
                                routed_fraction=tr.n_routed/max(1,len(evict)),
                                mean_routed_cosine=float(np.mean(cosine_scores[routed])) if np.any(routed) else 0.0,
                                mis_rejected_fraction=tr.n_mis_rejected/max(1,tr.n_routed),
                                n_affected_groups=sum(int(g['n_mis_rejected']>0) for g in tr.group_rows),
                                n_native_conflict_edges=int(plan.conflict_edges_per_anchor.sum().item()))
                            # Optionally save *real* high-dimensional group arrays for publication figures.
                            if (seq==0 and layer==a.capture_layer and head==a.capture_head and
                                abs(budget-a.capture_budget)<1e-7 and
                                abs(lam-a.capture_ee_lambda)<1e-7 and
                                (a.capture_routing_parameter is None or
                                 (mode==a.capture_routing_mode and
                                  abs(param-a.capture_routing_parameter)<1e-7 and
                                  (mode=='cosine' or kind==a.capture_routing_parameter_type)))):
                                save_example(root,seq,layer,head,budget,lam,tr,orig_k,keep,evict,best,routed,selected,mode,kind,param,cutoff)
                            # Config totals and per-anchor group counts share field names.
                            # Preserve both with explicit configuration prefixes; do not
                            # pass duplicate **kwargs to dict(), which raises TypeError.
                            group_common = {
                                (f'config_{key}' if key in ('n_routed', 'n_mis_accepted', 'n_mis_rejected') else key): value
                                for key, value in common.items()
                            }
                            group_rows=[{**group_common, **g} for g in tr.group_rows]
                            rejected_rows=[{**common,**r} for r in tr.rejected_rows]
                            write_csv(root/'cosine_mis_by_group.csv',group_rows,append=True)
                            write_csv(root/'cosine_mis_rejected.csv',rejected_rows,append=True)
                            group_total+=len(group_rows);rej_total+=len(rejected_rows)
                            # Assert all 3 arms have EXACT same original Value and Count.
                            if np.any(tr.counts<=0) or tr.n_mis_accepted+tr.n_mis_rejected!=tr.n_routed:
                                raise AssertionError('Broken routed/MIS/Count invariants')
                            by_method={};by_probe={}
                            for method in METHODS:
                                approx=replay_prefix_attention(q,tr.keys[method],tr.values,
                                    tr.counts,fk,fv,compute_device,a.output_norm_eps)
                                stats,probes=output_errors(approx,exact,a.output_norm_eps)
                                by_method[method]=stats;by_probe[method]=probes
                            # Both controls ignore MIS: must be identical across E-E lambda
                            # within one fixed R-E routing setting.
                            for ctrl in ('merge_all','all_to_anchor'):
                                if ctrl not in fixed_controls:
                                    fixed_controls[ctrl]=(tr.keys[ctrl].copy(), by_method[ctrl]['output_rel_error'])
                                else:
                                    saved_key,saved_error=fixed_controls[ctrl]
                                    if not np.array_equal(saved_key,tr.keys[ctrl]) or saved_error != by_method[ctrl]['output_rel_error']:
                                        raise AssertionError(f'{ctrl} changed when ONLY E-E threshold changed')
                            base=by_method['merge_all']['output_rel_error']
                            base_probe=by_probe['merge_all']
                            current_rows=[];current_probes=[]
                            for method in METHODS:
                                met=by_method[method]; per=by_probe[method]
                                diff=float(met['output_rel_error']-base)
                                drift=float(np.mean(np.linalg.norm(tr.keys[method].astype(np.float64)-orig_k[keep].astype(np.float64),axis=-1)))
                                current_rows.append(dict(**common,method=method,
                                    delta_rel_from_merge_all=diff,
                                    relative_improvement_pct=100.*(-diff)/max(base,1e-12),
                                    probe_win_fraction=float(np.mean(per<base_probe-1e-10)),
                                    reconstruction_sse_original_key=tr.sse[method],
                                    mean_anchor_drift=drift,**met))
                                if a.save_per_probe:
                                    for p,(err,br) in enumerate(zip(per,base_probe)):
                                        current_probes.append(dict(**common,method=method,
                                            future_token_offset=p,mean_rel_error=float(err),
                                            delta_rel_from_merge_all=float(err-br)))
                            all_rows.extend(current_rows)
                            write_csv(root/'cosine_mis_by_config.csv',current_rows,append=True)
                            if current_probes:
                                write_csv(root/'cosine_mis_by_probe.csv',current_probes,append=True)
                                probe_total+=len(current_probes)
                            print(f'  [B={budget:g}, {route_display(spec,cutoff)}, EE={lam:g}] keep={nkeep} route={tr.n_routed} MIS_accept={tr.n_mis_accepted} reject={tr.n_mis_rejected} '
                                  f'relError: {by_method["merge_all"]["output_rel_error"]:.6f} -> {by_method["mis_to_anchor"]["output_rel_error"]:.6f} '
                                  f'/ {by_method["all_to_anchor"]["output_rel_error"]:.6f}',flush=True)
                            del plan,tr
                del orig_k,orig_v,exact
                gc.collect()
        del past,full_past,scores,full_keys,full_values,true_queries
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    write_csv(root/'cosine_mis_overall.csv',summarize(all_rows))
    status.update(status='completed',n_sequences=nseq,n_configuration_rows=len(all_rows),
        n_group_rows=group_total,n_rejected_rows=rej_total,n_probe_rows=probe_total,
        runtime_seconds=time.time()-started,
        selected_layers=selected_layers,selected_heads=selected_heads,
        reference_scale_source='median_across_train_sequences_frozen_per_layer_head',
        torch_version=torch.__version__,transformers_version=transformers.__version__,
        platform=platform.platform())
    (root/'run_config.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(f'[COMPLETE] triad configs={len(all_rows)}, groups={group_total}, rejected={rej_total}, real probes={probe_total} -> {root}',flush=True)

if __name__=='__main__':main()
