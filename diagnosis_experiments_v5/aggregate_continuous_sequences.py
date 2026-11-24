#!/usr/bin/env python3
"""Validate distinct continuous-PPL windows and pool NLL over scored tokens.

Each window is a separate evaluation run/cache/RoPE origin. Never average PPLs.
Method identities, fixed parameters, calibration and per-cycle token accounting
must match across all windows before an aggregate completion marker is written.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

KEYS=('method','p','budget','lambda_ea','lambda_ee')
SEQ_KEYS={'sequence_index','token_start','token_end','scored_stream_sha256','output_dir'}
SUM_FIELDS=('routed_slots','accepted_slots','rejected_slots','dropped_slots',
            'mis_conflict_edges','geometry_over_delta',
            'soft_positive_slots','soft_fractional_slots','soft_ea_slots',
            'soft_parallel_rounds','soft_ee_slots')
FLOAT_SUM_FIELDS=('soft_weight_sum','soft_ea_weight_sum','soft_ee_weight_sum',
                  'routed_importance_mass','contributed_importance_mass')


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def save_csv(path,records):
    if not records:raise ValueError(f'Empty CSV output: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]),extrasaction='raise')
        writer.writeheader();writer.writerows(records)
    tmp.replace(path)


def fnum(value):
    result=float(value)
    if not math.isfinite(result):raise ValueError(f'Nonfinite number {value!r}')
    return result


def integer(value):
    value=fnum(value)
    if not value.is_integer():raise ValueError(f'Not an integer: {value}')
    return int(value)


def row_key(row):
    return tuple(str(row[key]) for key in KEYS)


def check_close(actual,expected,what):
    if not math.isclose(actual,expected,rel_tol=1e-7,abs_tol=5e-7):
        raise RuntimeError(f'{what}: {actual} != {expected}')


def validate_window(root,index):
    src=root/'sequences'/f'seq_{index}'/'ppl_continuous'
    files=['continuous_config.json','continuous_run_complete.json',
           'ppl_continuous.csv','ppl_continuous_cycles.csv']
    for name in files:
        if not (src/name).is_file():raise RuntimeError(f'Missing {src/name}')
    cfg=json.loads((src/files[0]).read_text(encoding='utf-8'))
    marker=json.loads((src/files[1]).read_text(encoding='utf-8'))
    if marker.get('config')!=cfg:raise RuntimeError(f'Completion marker mismatch: seq {index}')
    if integer(cfg['sequence_index'])!=index:
        raise RuntimeError(f'Wrong sequence_index in seq {index}')
    cycles=integer(cfg['cycles']);U=integer(cfg['unscored_len']);S=integer(cfg['scored_len'])
    start=integer(cfg['token_start']);end=integer(cfg['token_end']);length=cycles*(U+S)
    if (start,end)!=(index*length,(index+1)*length):
        raise RuntimeError(f'Unexpected token window in seq {index}')
    stream=str(cfg.get('scored_stream_sha256',''))
    if len(stream)!=64 or any(c not in '0123456789abcdef' for c in stream):
        raise RuntimeError(f'Bad input stream hash in seq {index}')
    rows=read_csv(src/'ppl_continuous.csv')
    cycle_rows=read_csv(src/'ppl_continuous_cycles.csv')
    if not rows or not cycle_rows:
        raise RuntimeError(f'Empty results in seq {index}')
    if len(rows)!=integer(marker['n_results']) or len(cycle_rows)!=integer(marker['n_cycles']):
        raise RuntimeError(f'Incomplete row counts in seq {index}')
    expected_methods=set(cfg['methods'])
    if not expected_methods:
        raise RuntimeError(f'No methods declared in seq {index}')
    expected_keys={(method,float(p),float(b),float(ea),float(ee))
                   for p,methods in cfg['methods_by_p'].items() for method in methods
                   for b in str(cfg['budgets']).split(',')
                   for ea in str(cfg['lambda_ea']).split(',')
                   for ee in str(cfg['lambda_ee']).split(',')}
    actual_keys={(r['method'],float(r['p']),float(r['budget']),float(r['lambda_ea']),float(r['lambda_ee'])) for r in rows}
    if len(rows)!=len(expected_keys) or actual_keys!=expected_keys:
        raise RuntimeError(f'Incomplete configured method/settings coverage in seq {index}')
    by_key={}
    for row in rows:
        key=row_key(row)
        if key in by_key:raise RuntimeError(f'Duplicate method/config {key} in seq {index}')
        if integer(row['scored_tokens'])!=cycles*S:
            raise RuntimeError(f'Wrong scored token count in seq {index}, {key}')
        if integer(row['sequence_index'])!=index or integer(row['token_start'])!=start or integer(row['token_end'])!=end:
            raise RuntimeError(f'Incorrect sequence metadata for {key}, seq {index}')
        if row['stream_sha256']!=stream:
            raise RuntimeError(f'Incorrect stream SHA for {key}, seq {index}')
        check_close(math.exp(fnum(row['nll_per_token'])),fnum(row['ppl']),f'PPL seq {index} {key}')
        check_close(fnum(row['nll_sum'])/integer(row['scored_tokens']),fnum(row['nll_per_token']),f'NLL sum seq {index} {key}')
        by_key[key]=row
    by_cycle=defaultdict(dict)
    for r in cycle_rows:
        key=row_key(r)
        cycle=integer(r['cycle'])
        if key not in by_key or cycle in by_cycle[key]:
            raise RuntimeError(f'Invalid or duplicate cycle for {key}, seq {index}')
        if integer(r['scored_tokens'])!=S:
            raise RuntimeError(f'Wrong cycle scored count in seq {index}')
        check_close(fnum(r['cycle_nll_sum'])/S,fnum(r['cycle_nll_per_token']),
                    f'cycle nll seq {index} {key} cycle {cycle}')
        by_cycle[key][cycle]=r
    if set(by_cycle)!=set(by_key):
        raise RuntimeError(f'Missing cycle groups in seq {index}')
    for key,record in by_key.items():
        cr=by_cycle[key]
        if set(cr)!=set(range(1,cycles+1)):
            raise RuntimeError(f'Missing cycles in seq {index}, {key}')
        nll=sum(fnum(r['cycle_nll_sum']) for r in cr.values())/(cycles*S)
        check_close(nll,fnum(record['nll_per_token']),f'NLL seq {index} {key}')
    return cfg,by_key,by_cycle,(start,end),stream


def aggregate(root,indices):
    root=Path(root)
    out=root/'aggregate'
    out.mkdir(parents=True,exist_ok=True)
    marker=out/'aggregate_complete.json'
    marker.unlink(missing_ok=True)
    if not indices or min(indices)<0 or len(set(indices))!=len(indices):
        raise ValueError('Sequence indices must be unique and nonnegative')
    window_manifest=root/'sequence_windows.json'
    if window_manifest.exists():
        planned=json.loads(window_manifest.read_text(encoding='utf8'))
        if planned['selected_indices']!=indices:raise RuntimeError('Aggregate window list differs from preflight')
    records=[];spans=[];hashes=[];master_cfg=None;master_keys=None
    for idx in indices:
        cfg,rows,cycle_rows,span,sha=validate_window(root,idx)
        fixed={k:v for k,v in cfg.items() if k not in SEQ_KEYS}
        if master_cfg is None:
            master_cfg=fixed;master_keys=set(rows)
        if fixed!=master_cfg:
            raise RuntimeError(f'Fixed evaluation settings/calibration differ in seq {idx}')
        if set(rows)!=master_keys:
            raise RuntimeError(f'Method/config coverage differs in seq {idx}')
        if any(span[0]<other[1] and other[0]<span[1] for other in spans):
            raise RuntimeError('Overlapping token windows')
        if sha in hashes:raise RuntimeError('Duplicate window token hashes')
        records.append((idx,rows,cycle_rows));spans.append(span);hashes.append(sha)
    pooled=[];pooled_cycles=[];per_sequence=[];deltas=[]
    for idx,rows,_ in records:
        per_sequence.extend(dict(row) for row in rows.values())
        # Paired comparisons on the SAME token window, not independent means.
        groups=defaultdict(dict)
        for key,row in rows.items():
            groups[key[1:]][key[0]]=row
        for setting,arms in groups.items():
            if not {'full_cache','merge_all','eviction_only','coherekv_conference'}<=set(arms):
                raise RuntimeError(f'Essential baseline missing in seq {idx} setting {setting}')
            for method,row in arms.items():
                deltas.append(dict(sequence_index=idx,method=method,
                    p=setting[0],budget=setting[1],lambda_ea=setting[2],lambda_ee=setting[3],
                    nll_per_token=fnum(row['nll_per_token']),
                    delta_nll_vs_eviction=fnum(row['nll_per_token'])-fnum(arms['eviction_only']['nll_per_token']),
                    delta_nll_vs_merge_all=fnum(row['nll_per_token'])-fnum(arms['merge_all']['nll_per_token']),
                    delta_nll_vs_degree_mis=(fnum(row['nll_per_token'])-fnum(arms['coherekv_soft_mis']['nll_per_token']) if 'coherekv_soft_mis' in arms else ''),
                    delta_nll_vs_full=fnum(row['nll_per_token'])-fnum(arms['full_cache']['nll_per_token']),
                    delta_nll_vs_conference=fnum(row['nll_per_token'])-fnum(arms['coherekv_conference']['nll_per_token'])))
    for key in sorted(master_keys):
        individual=[rows[key] for _,rows,_ in records]
        scored=sum(integer(r['scored_tokens']) for r in individual)
        total_nll=math.fsum(fnum(r['nll_sum']) for r in individual)
        avg=total_nll/scored
        out_row=dict(individual[0])
        # All method-specific metadata remain valid, but these fields apply to
        # an individual input window and must not masquerade as a pooled window.
        for field in ('sequence_index','token_start','token_end','stream_sha256'):
            out_row.pop(field,None)
        out_row['scored_tokens']=scored
        out_row['nll_per_token']=avg
        out_row['ppl']=math.exp(avg)
        out_row['overall_ppl']=out_row['ppl']
        out_row['nll_sum']=total_nll
        seq_ppls=[fnum(r['ppl']) for r in individual]
        out_row['mean_ppl']=statistics.mean(seq_ppls)
        out_row['std_ppl']=statistics.stdev(seq_ppls) if len(seq_ppls)>1 else ''
        full_key=('full_cache',*key[1:])
        paired=[fnum(rows[key]['nll_per_token'])-fnum(rows[full_key]['nll_per_token']) for _,rows,_ in records]
        out_row['delta_nll_vs_full']=math.fsum(integer(rows[key]['scored_tokens'])*d for (_,rows,_),d in zip(records,paired))/scored
        out_row['std_delta_nll_vs_full']=statistics.stdev(paired) if len(paired)>1 else ''
        out_row['compress_rounds']=sum(integer(r['compress_rounds']) for r in individual)
        for name in SUM_FIELDS:
            out_row[name]=sum(integer(r[name]) for r in individual)
        for name in FLOAT_SUM_FIELDS:
            out_row[name]=sum(fnum(r[name]) for r in individual)
        out_row['importance_retention_rate']=(out_row['contributed_importance_mass']/out_row['routed_importance_mass'] if out_row['routed_importance_mass']>0 else '')
        for name in ('max_before_len','max_after_len'):
            out_row[name]=max(integer(r[name]) for r in individual)
        out_row['seconds']=sum(fnum(r['seconds']) for r in individual)
        out_row['reused_baseline']=all(r['reused_baseline']=='True' for r in individual)
        out_row['sequence_indices']=','.join(map(str,indices))
        out_row['windows']=len(indices)
        pooled.append(out_row)
        running_nll=0.;running_scored=0
        num_cycles=integer(master_cfg['cycles'])
        for cycle in range(1,num_cycles+1):
            cr=[cycle_rows[key][cycle] for _,_,cycle_rows in records]
            nll=sum(fnum(r['cycle_nll_sum']) for r in cr)
            count=sum(integer(r['scored_tokens']) for r in cr)
            running_nll+=nll;running_scored+=count
            record=dict(cr[0])
            for field in ('sequence_index','token_start','token_end','stream_sha256'):
                record.pop(field,None)
            record.update(cycle_nll_sum=nll,scored_tokens=count,
                          cycle_nll_per_token=nll/count,cycle_ppl=math.exp(nll/count),
                          cumulative_nll_per_token=running_nll/running_scored,
                          cumulative_ppl=math.exp(running_nll/running_scored),
                          compression_rounds=sum(integer(r['compression_rounds']) for r in cr),
                          routed_slots=sum(integer(r['routed_slots']) for r in cr),
                          accepted_slots=sum(integer(r['accepted_slots']) for r in cr),
                          rejected_slots=sum(integer(r['rejected_slots']) for r in cr),
                          geometry_over_delta=sum(integer(r['geometry_over_delta']) for r in cr),
                          soft_positive_slots=sum(integer(r['soft_positive_slots']) for r in cr),
                          soft_fractional_slots=sum(integer(r['soft_fractional_slots']) for r in cr),
                          soft_ea_slots=sum(integer(r['soft_ea_slots']) for r in cr),
                          soft_ee_slots=sum(integer(r['soft_ee_slots']) for r in cr),
                          soft_parallel_rounds=sum(integer(r['soft_parallel_rounds']) for r in cr),
                          soft_weight_sum=sum(fnum(r['soft_weight_sum']) for r in cr),
                          soft_ea_weight_sum=sum(fnum(r['soft_ea_weight_sum']) for r in cr),
                          soft_ee_weight_sum=sum(fnum(r['soft_ee_weight_sum']) for r in cr),
                          physical_kv_len=sum(integer(r['physical_kv_len']) for r in cr)/len(cr),
                          sequence_indices=','.join(map(str,indices)),windows=len(indices))
            for name in ('routed_importance_mass','contributed_importance_mass'):
                record[name]=sum(fnum(r[name]) for r in cr)
            record['importance_retention_rate']=(record['contributed_importance_mass']/record['routed_importance_mass'] if record['routed_importance_mass']>0 else '')
            pooled_cycles.append(record)
        check_close(running_nll/running_scored,avg,f'Pooled cycle-total NLL {key}')
    save_csv(out/'ppl_continuous.csv',pooled)
    # Deduplicate display-grid baseline replicas; rank using pooled token NLL only.
    unique={}
    for row in pooled:
        k=tuple(str(row[x]) for x in ('method','actual_budget','actual_p','actual_lambda_ea','actual_lambda_ee','actual_reference_mode'))
        if k in unique:
            check_close(fnum(row['nll_per_token']),fnum(unique[k]['nll_per_token']),f'Baseline reuse {k}')
            if unique[k]['reused_baseline'] and not row['reused_baseline']:unique[k]=dict(row)
        else:unique[k]=dict(row)
    ranked=sorted(unique.values(),key=lambda r:(r['method'],fnum(r['nll_per_token'])))
    for row in ranked:
        for name in ('budget','p','lambda_ea','lambda_ee'):row.pop(name,None)
    save_csv(out/'ppl_sweep_unique.csv',ranked)
    best=[]
    for method in sorted({r['method'] for r in ranked}):
        groups=defaultdict(list)
        for r in ranked:
            if r['method']==method:groups[(r['actual_budget'],r['actual_p'])].append(r)
        for group in groups.values():best.append(min(group,key=lambda r:fnum(r['nll_per_token'])))
    save_csv(out/'ppl_sweep_best_by_method_p.csv',best)
    save_csv(out/'ppl_continuous_cycles.csv',pooled_cycles)
    save_csv(out/'ppl_continuous_per_sequence.csv',per_sequence)
    save_csv(out/'ppl_continuous_paired_deltas.csv',deltas)
    manifest=dict(implementation='journal_soft_mis_sweep_multiseq_v6',sequence_indices=indices,
                  token_spans=[dict(sequence_index=i,start=s[0],end=s[1],stream_sha256=h)
                               for i,s,h in zip(indices,spans,hashes)],
                  common_configuration=master_cfg,
                  scored_tokens_per_method=sum(integer(r['scored_tokens']) for r in
                                               [records[i][1][next(iter(master_keys))] for i in range(len(records))]),
                  methods=sorted(set(key[0] for key in master_keys)),
                  method_configuration_rows=len(master_keys),
                  summary='Independent caches per window; pooled NLL is scored-token weighted; PPL=exp(pooled NLL).')
    marker.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'[AGGREGATE] complete windows={indices}, methods={manifest["methods"]}, rows={len(pooled)}, '
          f'scored_tokens/method={manifest["scored_tokens_per_method"]}',flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input_dir',required=True)
    p.add_argument('--sequence_indices',required=True,
                   help='Comma separated nonnegative indices as used by run_soft_mis.sh')
    a=p.parse_args()
    indices=[int(x.strip()) for x in a.sequence_indices.split(',') if x.strip()]
    aggregate(Path(a.input_dir),indices)


if __name__=='__main__':main()
