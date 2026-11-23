#!/usr/bin/env python3
"""Strict, token-weighted, multi-window Group-D2/conference comparison.

No assumptions of identical candidate IDs or matching quotas across divergent
cache trajectories. Cycle, head/round, calibration and configuration completeness
are validated BEFORE any final complete manifest is produced.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from result_csv import save_csv

ARMS = ('full_cache', 'eviction_only', 'merge_all', 'coherekv_inf_certificate',
        'group_d2', 'group_d2_importance_matched')
CONTRASTS = {'merge_all':'merge_all', 'conference':'coherekv_inf_certificate',
             'eviction':'eviction_only', 'matched':'group_d2_importance_matched'}
PLAN_COLS = ('setting_id','lambda_history','lambda_ea','lambda_ee','budget','pair_window')
FIXED_FIELDS = ('model','dataset','cycles','unscored_len','scored_len','reference_mode',
                'sink_size','recent_protect_size','score_window','recency_beta',
                'prefill_chunk_size','distance_chunk_size','graph_backend',
                'torch_dtype','attn_implementation','device_map','fusion_sample_topk')


def read_csv(path):
    with Path(path).open(newline='',encoding='utf8') as fp:
        return list(csv.DictReader(fp))


def num(value):
    v=float(value)
    if not math.isfinite(v):raise ValueError('Nonfinite numeric value: '+str(value))
    return v


def integer(value):
    v=num(value)
    if not v.is_integer():raise ValueError('Expected integer: '+str(value))
    return int(v)


def validate_plan(root):
    rows=read_csv(Path(root)/'sweep_plan.csv')
    if not rows or set(rows[0])!=set(PLAN_COLS):raise ValueError('Invalid sweep_plan.csv')
    ids=set();settings=set()
    for row in rows:
        sid=row['setting_id']
        if (not sid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'
                          for c in sid) or sid in ids):raise ValueError('Duplicate/invalid setting ID '+sid)
        ids.add(sid)
        vals=tuple(num(row[k]) for k in ('lambda_history','lambda_ea','lambda_ee','budget'))
        k=integer(row['pair_window'])
        if min(vals)<=0 or vals[-1]>=1 or not 0<=k<=128:raise ValueError('Invalid setting')
        if (*vals,k) in settings:raise ValueError('Duplicate numeric setting')
        settings.add((*vals,k))
    return rows


def validate_window(folder,setting,seq_idx):
    if not (folder/'continuous_run_complete.json').is_file():
        raise RuntimeError(f'Missing completion marker: {folder}')
    cfg=json.loads((folder/'continuous_config.json').read_text(encoding='utf8'))
    done=json.loads((folder/'continuous_run_complete.json').read_text(encoding='utf8'))
    if integer(cfg['sequence_index'])!=seq_idx:raise RuntimeError('Wrong sequence index')
    p=int(cfg['cycles']);s=int(cfg['scored_len']);n=p*s
    start,end=integer(cfg['token_start']),integer(cfg['token_end'])
    if end-start!=p*(integer(cfg['unscored_len'])+s) or start<0:
        raise RuntimeError('Invalid token span')
    for ck,sk in (('lambda_history','lambda_history'),('group_d2_pair_window','pair_window')):
        if not math.isclose(num(cfg[ck]),num(setting[sk]),rel_tol=1e-10):
            raise RuntimeError('Setting mismatch: '+ck)
    pair=cfg['resolved_lambda_pairs']
    if len(pair)!=1 or any(not math.isclose(num(pair[0][ck]),num(setting[sk]),rel_tol=1e-10)
                       for ck,sk in (('lambda_ea','lambda_ea'),('lambda_ee','lambda_ee'))):
        raise RuntimeError('EA/EE mismatch')
    if not math.isclose(num(cfg['budgets']),num(setting['budget']),rel_tol=1e-10):
        raise RuntimeError('Budget mismatch')
    if cfg['p_values']!='2' or cfg['geometry_modes']!='exact_p2':
        raise RuntimeError('Group-D2 must use exact p=2; conference arm overrides p=inf internally')
    if cfg.get('methods')!=list(ARMS):
        raise RuntimeError(f'Unexpected configured methods: {cfg.get("methods")}')
    if done.get('config')!=cfg:
        raise RuntimeError('Completion marker does not match current config')
    data=read_csv(folder/'ppl_continuous.csv')
    cycles=read_csv(folder/'ppl_continuous_cycles.csv')
    heads=read_csv(folder/'ppl_continuous_head_rounds.csv')
    if (len(data)!=len(ARMS) or {x['method'] for x in data}!=set(ARMS)
            or len(cycles)!=len(ARMS)*p or not heads):
        raise RuntimeError('Missing method/cycle/head records')
    if integer(done.get('n_results',-1))!=len(data) or integer(done.get('n_cycles',-1))!=len(cycles):
        raise RuntimeError('Bad completed record counts')
    rows={x['method']:x for x in data}
    indexed=defaultdict(list)
    for x in cycles:indexed[x['method']].append(x)
    for method in ARMS:
        row=rows[method];chunk=indexed[method]
        if len(chunk)!=p or sorted(integer(x['cycle']) for x in chunk)!=list(range(1,p+1)):
            raise RuntimeError(f'Missing/duplicate cycles for {method}')
        if integer(row['scored_tokens'])!=n or any(integer(x['scored_tokens'])!=s for x in chunk):
            raise RuntimeError('Bad token score count')
        avg=sum(num(x['cycle_nll_sum']) for x in chunk)/n
        if not math.isclose(avg,num(row['nll_per_token']),rel_tol=1e-7,abs_tol=5e-7):
            raise RuntimeError(f'Cycle NLL does not match method NLL: {method}')
        if not math.isclose(math.exp(avg),num(row['ppl']),rel_tol=1e-6):
            raise RuntimeError('PPL is inconsistent with NLL')
        for key in ('lambda_ea','lambda_ee','budget'):
            if not math.isclose(num(row[key]),num(setting[key]),rel_tol=1e-10):
                raise RuntimeError('Wrong method configuration '+key)
        if method=='coherekv_inf_certificate':
            if not math.isinf(float(row['p'])) or row['geometry_mode']!='minkowski':
                raise RuntimeError('Conference comparator must use p=inf Minkowski certificate')
        elif float(row['p'])!=2. or row['geometry_mode']!='exact_p2':
            raise RuntimeError('Non-conference methods must report exact p=2 mode')
    if integer(rows['group_d2']['group_d2_real_violations'])!=0:
        raise RuntimeError('Group-D2 history compactness failure')
    if integer(rows['group_d2_importance_matched']['group_d2_matched_quota_shortfall'])!=0:
        raise RuntimeError('Own-trajectory shadow matched quota shortfall')
    # Check each compressed arm's complete layer x head x round table. A
    # missing head or round must never silently look like zero acceptance.
    head_keysets=[]
    for method in ARMS[1:]:
        subset=[x for x in heads if x['method']==method]
        keyset={(integer(x['round']),integer(x['layer_idx']),integer(x['kv_head_idx'])) for x in subset}
        if len(subset)!=len(keyset) or not keyset or {t[0] for t in keyset}!=set(range(1,p+1)):
            raise RuntimeError('Missing/duplicate compressed head rounds: '+method)
        if sum(integer(x['accepted_slots']) for x in subset)!=integer(rows[method]['accepted_slots']):
            raise RuntimeError('Accepted sum mismatch in '+method)
        if sum(integer(x['routed_slots']) for x in subset)!=integer(rows[method]['routed_slots']):
            raise RuntimeError('Routed sum mismatch in '+method)
        if sum(integer(x['accepted_previously_merged_slots']) for x in subset)!=integer(rows[method]['accepted_previously_merged_slots']):
            raise RuntimeError('Repeated history sum mismatch in '+method)
        if method=='eviction_only' and any(integer(x['accepted_slots']) or integer(x['routed_slots']) for x in subset):
            raise RuntimeError('Eviction-only arm performed a merge')
        if method=='coherekv_inf_certificate' and any(num(x['max_diameter_over_delta'])>1.+2e-6 for x in subset):
            raise RuntimeError('Conference infinity certificate exceeded diameter threshold')
        head_keysets.append(keyset)
    if any(keys!=head_keysets[0] for keys in head_keysets[1:]):
        raise RuntimeError('Method trajectories have different layer/head/round coverage')
    if [x for x in heads if x['method']=='full_cache']:
        raise RuntimeError('Unexpected full-cache compression head records')
    return cfg,rows,cycles,heads,(start,end)


def aggregate(root,indices):
    root=Path(root)
    plan=validate_plan(root)
    if not indices or min(indices)<0 or len(set(indices))!=len(indices):
        raise ValueError('Distinct nonnegative windows required')
    global_calibration=None;global_fixed=None
    expected_streams=None
    all_settings=[];all_pooled=[];all_windows=[];all_rounds=[]
    for setting in plan:
        sid=setting['setting_id'];setting_dir=root/'settings'/sid
        window_records=[];raw_results=[];spans=[];streams=[]
        cycle_agg=defaultdict(list);head_agg=defaultdict(list)
        for seq_idx in indices:
            folder=setting_dir/'sequences'/f'seq_{seq_idx}'/'ppl_continuous'
            cfg,rows,cycles,heads,span=validate_window(folder,setting,seq_idx)
            if any(span[0]<b and a<span[1] for a,b in spans):raise RuntimeError('Overlapping windows')
            spans.append(span)
            frozen=cfg['reference_sha256']
            fixed=tuple(str(cfg[field]) for field in FIXED_FIELDS)
            if global_calibration is None:global_calibration=frozen
            if global_fixed is None:global_fixed=fixed
            if frozen!=global_calibration or fixed!=global_fixed:
                raise RuntimeError('Calibration or fixed inference settings differ across runs')
            stream=cfg['scored_stream_sha256'];streams.append(stream)
            win=dict(setting_id=sid,sequence_index=seq_idx,token_start=span[0],
                     token_end=span[1],scored_tokens=int(cfg['cycles'])*int(cfg['scored_len']),
                     stream_sha256=stream)
            for method in ARMS:
                win[method+'_nll']=num(rows[method]['nll_per_token'])
                win[method+'_ppl']=num(rows[method]['ppl'])
                win[method+'_accepted']=integer(rows[method]['accepted_slots'])
                win[method+'_remerged']=integer(rows[method]['accepted_previously_merged_slots'])
            for name,method in CONTRASTS.items():
                win['group_d2_minus_'+name+'_nll']=win['group_d2_nll']-win[method+'_nll']
            window_records.append(win);all_windows.append(win)
            raw_results.extend(rows.values())
            for c in cycles:cycle_agg[(c['method'],integer(c['cycle']))].append(c)
            for h in heads:head_agg[(h['method'],integer(h['round']))].append(h)
        if len(set(streams))!=len(streams):raise RuntimeError('Duplicate streams across windows')
        if expected_streams is None:expected_streams=tuple(streams)
        if expected_streams!=tuple(streams):raise RuntimeError('Different streams between settings')
        pooled=[]
        for method in ARMS:
            subset=[r for r in raw_results if r['method']==method]
            count=sum(integer(r['scored_tokens']) for r in subset)
            nll=sum(integer(r['scored_tokens'])*num(r['nll_per_token']) for r in subset)/count
            pooled.append(dict(method=method,windows=len(subset),scored_tokens=count,
                pooled_nll_per_token=nll,pooled_ppl=math.exp(nll),
                accepted_slots_sum=sum(integer(r['accepted_slots']) for r in subset),
                accepted_previously_merged_slots_sum=sum(integer(r['accepted_previously_merged_slots']) for r in subset),
                group_d2_real_violations_sum=sum(integer(r.get('group_d2_real_violations',0) or 0) for r in subset),
                group_d2_same_state_matched_swaps_sum=sum(integer(r.get('group_d2_same_state_matched_swaps',0) or 0) for r in subset)))
        by_method={x['method']:x for x in pooled};group=by_method['group_d2']
        entry=dict(**setting,windows=len(indices),scored_tokens=group['scored_tokens'],
                   group_d2_nll=group['pooled_nll_per_token'],group_d2_ppl=group['pooled_ppl'],
                   group_d2_acceptance_ratio_to_merge_all=(group['accepted_slots_sum']/max(1,by_method['merge_all']['accepted_slots_sum'])),
                   group_d2_history_violations=group['group_d2_real_violations_sum'])
        for method in ARMS:
            entry[method+'_pooled_nll']=by_method[method]['pooled_nll_per_token']
            entry[method+'_pooled_ppl']=by_method[method]['pooled_ppl']
            entry[method+'_accepted_sum']=by_method[method]['accepted_slots_sum']
            entry[method+'_remerged_sum']=by_method[method]['accepted_previously_merged_slots_sum']
            all_pooled.append(dict(**setting,**by_method[method]))
        for name,method in CONTRASTS.items():
            entry['group_d2_minus_'+name+'_nll']=entry['group_d2_nll']-by_method[method]['pooled_nll_per_token']
            entry['group_d2_wins_'+name+'_windows']=sum(w['group_d2_minus_'+name+'_nll']<0 for w in window_records)
        all_settings.append(entry)
        # Preserve per-round metrics, including the conference radius saturation.
        setting_rounds=[]
        for (method,round_idx),runs in sorted(cycle_agg.items()):
            hr=head_agg.get((method,round_idx),[])
            record=dict(setting_id=sid,method=method,round=round_idx,windows=len(runs),
                cycle_nll_per_token=sum(num(x['cycle_nll_sum']) for x in runs)/sum(integer(x['scored_tokens']) for x in runs),
                routed_slots=sum(integer(x['routed_slots']) for x in hr),
                accepted_slots=sum(integer(x['accepted_slots']) for x in hr),
                remerged_slots=sum(integer(x['accepted_previously_merged_slots']) for x in hr),
                max_radius_over_delta=(max((num(x['max_radius_over_delta']) for x in hr if x['max_radius_over_delta']!=''),default='')),
                max_diameter_over_delta=(max((num(x['max_diameter_over_delta']) for x in hr if x['max_diameter_over_delta']!=''),default='')),
                max_history_d2_over_delta=(max((num(x['real_history_d2_over_delta_max']) for x in hr if x['real_history_d2_over_delta_max']!=''),default='')))
            record['acceptance_rate']=(record['accepted_slots']/record['routed_slots'] if record['routed_slots'] else '')
            setting_rounds.append(record);all_rounds.append(record)
        save_csv(setting_dir/'group_d2_per_sequence.csv',window_records)
        save_csv(setting_dir/'group_d2_all_methods.csv',raw_results)
        save_csv(setting_dir/'group_d2_pooled_methods.csv',pooled)
        save_csv(setting_dir/'group_d2_round_metrics.csv',setting_rounds)
        save_csv(setting_dir/'group_d2_paired_contrasts.csv',[
            dict(contrast='group_d2_minus_'+name,
                 mean_delta_nll=sum(w['group_d2_minus_'+name+'_nll'] for w in window_records)/len(window_records),
                 n_windows_group_better=sum(w['group_d2_minus_'+name+'_nll']<0 for w in window_records))
            for name in CONTRASTS])
        (setting_dir/'group_d2_manifest.json').write_text(json.dumps(dict(
            indices=indices,methods=ARMS,complete=True,
            note='Trajectories diverge after compression; matched arm quotas are computed on its own trajectory. '
                 'Same-state matched swaps are counterfactual diagnostics only.'),indent=2),encoding='utf8')
    ranked=[dict(rank_by_group_d2_ppl=i+1,**r) for i,r in enumerate(sorted(all_settings,key=lambda x:x['group_d2_nll']))]
    save_csv(root/'group_d2_sweep_settings.csv',all_settings)
    save_csv(root/'group_d2_sweep_pooled_methods.csv',all_pooled)
    save_csv(root/'group_d2_sweep_per_window.csv',all_windows)
    save_csv(root/'group_d2_sweep_rounds.csv',all_rounds)
    save_csv(root/'group_d2_sweep_ranked.csv',ranked)
    (root/'group_d2_sweep_manifest.json').write_text(json.dumps(dict(
        experiment='group_d2_crossround_validation_v20',complete=True,settings=len(plan),
        windows=indices,methods=ARMS,shared_frozen_reference_sha256=global_calibration,
        common_streams=expected_streams,
        conference_comparator='p=inf Minkowski R/D certificate + original degree-priority MIS; '
                              'same frozen TRAIN scale and literal lambda_EA/EE as other arms',
        caveat='Matched arm has independent trajectory; same-state counterfactual diagnostics are not matched PPL.',
        warning='Held-out windows provide validation; generalization requires additional models and corpora.'
    ),indent=2),encoding='utf8')
    return all_settings,ranked


def plot(root,settings,font,dpi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_continuous_ppl import setup_font,write
    setup_font(font)
    root=Path(root)
    # Group by physical route, budget and lookahead; do not connect points
    # measured with different lambda_EA as if they were one curve.
    groups=defaultdict(list)
    for x in settings:
        groups[(x['lambda_ea'],x['lambda_ee'],x['budget'],x['pair_window'])].append(x)
    fig,ax=plt.subplots(figsize=(9,5))
    for key,items in sorted(groups.items()):
        items=sorted(items,key=lambda x:num(x['lambda_history']))
        for m in ('merge_all','coherekv_inf_certificate','group_d2','group_d2_importance_matched'):
            ax.plot([num(x['lambda_history']) for x in items],
                    [num(x[m+'_pooled_ppl']) for x in items],marker='o',
                    label=f'{m} EA={key[0]} EE={key[1]}')
    ax.set(xlabel='Historical threshold multiplier $\\lambda_H$',ylabel='Pooled PPL',
           title='Cross-round compactness comparison: held-out windows')
    ax.grid(alpha=.2);ax.legend(fontsize=7)
    write(fig,root/'sweep_figures','group_d2_validation_ppl',dpi)
    fig,ax=plt.subplots(figsize=(9,5))
    for key,items in sorted(groups.items()):
        items=sorted(items,key=lambda x:num(x['lambda_history']))
        for m in ('conference','merge_all','matched','eviction'):
            ax.plot([num(x['lambda_history']) for x in items],
                    [num(x['group_d2_minus_'+m+'_nll']) for x in items],marker='o',
                    label=f'D2 - {m}, EA={key[0]}')
    ax.axhline(0,color='gray',linewidth=1)
    ax.set(xlabel='Historical threshold multiplier $\\lambda_H$',
           ylabel='Paired pooled NLL/token (negative favors Group-D2)',
           title='Matched-data comparisons, not matched trajectories')
    ax.grid(alpha=.2);ax.legend(fontsize=7)
    write(fig,root/'sweep_figures','group_d2_validation_contrasts',dpi)
    rounds=read_csv(root/'group_d2_sweep_rounds.csv')
    fig,ax=plt.subplots(figsize=(9,5))
    for setting_id in sorted({x['setting_id'] for x in rounds}):
        for m in ('group_d2','coherekv_inf_certificate'):
            items=sorted((x for x in rounds if x['setting_id']==setting_id and x['method']==m),
                         key=lambda x:integer(x['round']))
            if items:
                ax.plot([integer(x['round']) for x in items],
                        [num(x['acceptance_rate'])*100 if x['acceptance_rate']!='' else float('nan') for x in items],
                        marker='.',label=f'{setting_id} / {m}')
    ax.set(xlabel='Compression round',ylabel='Routed candidate acceptance (%)',
           title='Cross-round acceptance: same selection policy, divergent states')
    ax.grid(alpha=.2);ax.legend(fontsize=7)
    write(fig,root/'sweep_figures','group_d2_validation_acceptance',dpi)
    fig,ax=plt.subplots(figsize=(9,5))
    for setting_id in sorted({x['setting_id'] for x in rounds}):
        items=sorted((x for x in rounds if x['setting_id']==setting_id and x['method']=='coherekv_inf_certificate'),
                     key=lambda x:integer(x['round']))
        if items:
            ax.plot([integer(x['round']) for x in items],
                    [num(x['max_radius_over_delta']) for x in items],marker='.',label=setting_id)
    ax.axhline(1,color='gray',linestyle='--',linewidth=1)
    ax.set(xlabel='Compression round',ylabel='Maximum certificate radius / threshold',
           title='Conference-style R certificate saturation')
    ax.grid(alpha=.2);ax.legend(fontsize=7)
    write(fig,root/'sweep_figures','group_d2_validation_radius_saturation',dpi)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input_dir',required=True)
    ap.add_argument('--sequence_indices',required=True)
    ap.add_argument('--validate_only',type=int,choices=(0,1),default=0)
    ap.add_argument('--plots',type=int,choices=(0,1),default=0)
    ap.add_argument('--font_family',default='Times New Roman')
    ap.add_argument('--png_dpi',type=int,default=600)
    a=ap.parse_args()
    indices=[int(x) for x in a.sequence_indices.split(',') if x.strip()]
    if a.validate_only:
        plan=validate_plan(a.input_dir)
        print('[PLAN]',len(plan),'settings',len(indices),'windows',flush=True)
        return
    settings,ranked=aggregate(a.input_dir,indices)
    print('[COMPLETE]',len(settings),'settings;',len(indices),'windows; best D2 setting',
          ranked[0]['setting_id'],round(ranked[0]['group_d2_ppl'],6),flush=True)
    if a.plots:plot(a.input_dir,settings,a.font_family,a.png_dpi)

if __name__=='__main__':main()
