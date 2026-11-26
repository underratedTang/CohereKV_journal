#!/usr/bin/env python3
"""Validate unique configurations/windows and pool exact NLL sums, not PPL."""
import argparse,csv,json,math,statistics
from pathlib import Path
from ppl_log import emit_final_summary

KEYS=('config_id','method','budget','reference_mode','alpha_ea','alpha_ee','lambda_ea','lambda_ee','priority_mode')
SEQ_KEYS={'sequence_index','token_start','token_end','scored_stream_sha256','output_dir'}
SUM_FIELDS=('routed_slots','accepted_slots','rejected_slots','dropped_slots','mis_conflict_edges','mis_parallel_rounds','geometry_over_delta')
FLOAT_SUM_FIELDS=('evicted_mass','merged_mass','dropped_mass','routed_importance_mass','contributed_importance_mass','routed_raw_importance','contributed_raw_importance')

def read_csv(p):
    with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def save_csv(p,rows):
    if not rows:return
    with p.with_suffix('.tmp').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    p.with_suffix('.tmp').replace(p)

def close(a,b,msg):
    if not math.isclose(float(a),float(b),rel_tol=1e-7,abs_tol=5e-7):raise RuntimeError(msg+f': {a} vs {b}')

def identity(r):return tuple(str(r[k]) for k in KEYS)

def strip_window(r):return {k:v for k,v in r.items() if k not in ('sequence_index','token_start','token_end','stream_sha256')}

def rates(r):
    def ratio(a,b):return a/b if b else ''
    total=r['accepted_slots']+r['dropped_slots']
    r.update(merge_rate=ratio(r['accepted_slots'],total),assignment_rate=ratio(r['routed_slots'],total),
             mis_rejection_rate=ratio(r['rejected_slots'],r['routed_slots']),
             represented_mass_merge_rate=ratio(r['merged_mass'],r['evicted_mass']),
             importance_retention_rate=ratio(r['contributed_importance_mass'],r['routed_importance_mass']),
             raw_importance_retention_rate=ratio(r['contributed_raw_importance'],r['routed_raw_importance']))
    return r

def validate_window(root,index):
    src=root/'sequences'/f'seq_{index}'/'ppl_continuous'
    cfg=json.loads((src/'continuous_config.json').read_text());marker=json.loads((src/'continuous_run_complete.json').read_text())
    if cfg!=marker['config'] or cfg['sequence_index']!=index:raise RuntimeError('Completion/config mismatch')
    n=cfg['cycles'];S=cfg['scored_len'];length=n*(cfg['unscored_len']+S);start=index*length;end=start+length
    sha=cfg['scored_stream_sha256']
    if (cfg['token_start'],cfg['token_end'])!=(start,end) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):raise RuntimeError('Wrong token span/hash')
    expected={identity(c) for c in cfg['configurations']}
    raw=read_csv(src/'ppl_continuous.csv');cr=read_csv(src/'ppl_continuous_cycles.csv')
    if len(raw)!=marker['n_results'] or len(cr)!=marker['n_cycles'] or len(raw)!=len(expected) or {identity(r) for r in raw}!=expected or len(cr)!=len(expected)*n:raise RuntimeError('Incomplete configuration coverage')
    def meta(r):
        if int(r['sequence_index'])!=index or int(r['token_start'])!=start or int(r['token_end'])!=end or r['stream_sha256']!=sha:raise RuntimeError('Window metadata mismatch')
    rows={r['config_id']:r for r in raw};cycles={k:{} for k in rows}
    if len(rows)!=len(expected):raise RuntimeError('Duplicate config ID')
    for r in cr:
        meta(r);k=r['config_id'];c=int(r['cycle'])
        if identity(r)!=identity(rows[k]) or c in cycles[k] or not 1<=c<=n or int(r['scored_tokens'])!=S:raise RuntimeError('Cycle config/coverage mismatch')
        close(float(r['cycle_nll_sum'])/S,r['cycle_nll_per_token'],'cycle NLL');close(math.exp(float(r['cycle_nll_per_token'])),r['cycle_ppl'],'cycle PPL')
        if int(r['logical_tokens'])!=c*(cfg['unscored_len']+S):raise RuntimeError('Cycle logical position mismatch')
        expected_rounds=0 if r['method']=='full_cache' else c
        if int(r['compression_rounds'])!=expected_rounds:raise RuntimeError('Wrong compression rounds')
        expected_length=c*(cfg['unscored_len']+S) if r['method']=='full_cache' else max(1,round((cfg['unscored_len']-1)*float(r['budget'])))+S+1
        if int(r['physical_kv_len'])!=expected_length:raise RuntimeError('Wrong physical cache length')
        cycles[k][c]=r
    for k,r in rows.items():
        meta(r)
        if int(r['scored_tokens'])!=n*S or set(cycles[k])!=set(range(1,n+1)):raise RuntimeError('Missing scored targets/cycles')
        total=math.fsum(float(c['cycle_nll_sum']) for c in cycles[k].values())
        close(total,r['nll_sum'],'sequence NLL sum');close(total/(n*S),r['nll_per_token'],'sequence NLL');close(math.exp(total/(n*S)),r['ppl'],'sequence PPL')
        cumulative=0.
        for c in range(1,n+1):
            row=cycles[k][c];cumulative+=float(row['cycle_nll_sum']);close(cumulative/(c*S),row['cumulative_nll_per_token'],'cumulative NLL')
    return cfg,rows,cycles,(start,end),sha

def aggregate(root,indices):
    root=Path(root);out=root/'aggregate';out.mkdir(parents=True,exist_ok=True)
    marker=out/'aggregate_complete.json';marker.unlink(missing_ok=True)
    if not indices or min(indices)<0 or len(set(indices))!=len(indices):raise ValueError('Invalid window indices')
    manifest=root/'sequence_windows.json'
    if manifest.exists() and json.loads(manifest.read_text())['selected_indices']!=indices:raise RuntimeError('Window manifest mismatch')
    records=[];fixed=None;spans=[];hashes=[]
    for idx in indices:
        cfg,rows,cycles,span,sha=validate_window(root,idx)
        common={k:v for k,v in cfg.items() if k not in SEQ_KEYS}
        if fixed is None:fixed=common
        if common!=fixed or sha in hashes or any(span[0]<b and a<span[1] for a,b in spans):raise RuntimeError('Mixed configurations or duplicate windows')
        records.append((rows,cycles));spans.append(span);hashes.append(sha)
    configs=fixed['configurations'];pooled=[];pooled_cycles=[];per_seq=[];paired=[]
    for rows,_ in records:
        per_seq.extend(rows.values())
        for r in rows.values():
            def delta(method):
                matching=[b for b in rows.values() if b['method']==method and (method=='full_cache' or b['budget']==r['budget']) and (method!='merge_all' or b['reference_mode']==r['reference_mode'] and b['lambda_ea']==r['lambda_ea'])]
                return float(r['nll_per_token'])-float(matching[0]['nll_per_token']) if matching else ''
            degree_match=[b for b in rows.values() if r['method']=='coherekv_journal' and b['method']=='coherekv_journal' and b['priority_mode']=='degree' and all(b[k]==r[k] for k in ('budget','reference_mode','alpha_ea','alpha_ee','lambda_ea','lambda_ee'))]
            delta_degree=float(r['nll_per_token'])-float(degree_match[0]['nll_per_token']) if degree_match else ''
            paired.append(dict(**{k:r[k] for k in KEYS},sequence_index=r['sequence_index'],nll_per_token=r['nll_per_token'],
                               delta_nll_vs_full=delta('full_cache'),delta_nll_vs_eviction=delta('eviction_only'),delta_nll_vs_merge_all=delta('merge_all'),delta_nll_vs_degree=delta_degree))
    for c in configs:
        k=c['config_id'];rr=[r[k] for r,_ in records];count=sum(int(r['scored_tokens']) for r in rr);nll=math.fsum(float(r['nll_sum']) for r in rr)
        row=strip_window(rr[0]);row.update(scored_tokens=count,nll_sum=nll,nll_per_token=nll/count,ppl=math.exp(nll/count),overall_ppl=math.exp(nll/count),
            mean_ppl=statistics.mean(float(r['ppl']) for r in rr),std_ppl=statistics.stdev(float(r['ppl']) for r in rr) if len(rr)>1 else '',
            seconds=sum(float(r['seconds']) for r in rr),compress_rounds=sum(int(r['compress_rounds']) for r in rr),windows=len(rr),sequence_indices=','.join(map(str,indices)))
        for f in SUM_FIELDS:row[f]=sum(int(r[f]) for r in rr)
        for f in FLOAT_SUM_FIELDS:row[f]=math.fsum(float(r[f]) for r in rr)
        for f in ('max_before_len','max_after_len'):row[f]=max(int(r[f]) for r in rr)
        for f in ('delta_nll_vs_full','delta_nll_vs_eviction','delta_nll_vs_merge_all','delta_nll_vs_degree'):
            vals=[r[f] for r in paired if r['config_id']==k];row[f]=statistics.mean(vals) if vals and all(v!='' for v in vals) else ''
        pooled.append(rates(row));running=0.;scored=0
        for i in range(1,fixed['cycles']+1):
            cr=[cycles[k][i] for _,cycles in records];cs=sum(int(r['scored_tokens']) for r in cr);cn=math.fsum(float(r['cycle_nll_sum']) for r in cr);running+=cn;scored+=cs
            x=strip_window(cr[0]);x.update(scored_tokens=cs,cycle_nll_sum=cn,cycle_nll_per_token=cn/cs,cycle_ppl=math.exp(cn/cs),cumulative_nll_per_token=running/scored,cumulative_ppl=math.exp(running/scored),compression_rounds=sum(int(r['compression_rounds']) for r in cr),windows=len(cr))
            for f in SUM_FIELDS:x[f]=sum(int(r[f]) for r in cr)
            for f in FLOAT_SUM_FIELDS:x[f]=math.fsum(float(r[f]) for r in cr)
            pooled_cycles.append(rates(x))
        close(running,nll,'pooled NLL')
    ranked=sorted(pooled,key=lambda r:r['overall_ppl']);save_csv(out/'ppl_sweep_unique.csv',ranked);save_csv(out/'ppl_continuous.csv',pooled)
    save_csv(out/'ppl_continuous_cycles.csv',pooled_cycles);save_csv(out/'ppl_continuous_per_sequence.csv',per_seq);save_csv(out/'ppl_continuous_paired_deltas.csv',paired)
    best={}
    for r in ranked:best.setdefault((r['method'],r['budget'],r['reference_mode']),r)
    save_csv(out/'ppl_sweep_best.csv',list(best.values()))
    result=dict(implementation='journal_weighted_mis_v4',sequence_indices=indices,common_configuration=fixed,configurations=len(configs),
                token_spans=[dict(sequence_index=i,start=s[0],end=s[1],stream_sha256=h) for i,s,h in zip(indices,spans,hashes)],scored_tokens_per_config=pooled[0]['scored_tokens'])
    marker.write_text(json.dumps(result,indent=2));emit_final_summary(ranked,indices,result['scored_tokens_per_config']);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--input_dir',required=True);p.add_argument('--sequence_indices',required=True);a=p.parse_args();aggregate(a.input_dir,[int(i) for i in a.sequence_indices.split(',')])
if __name__=='__main__':main()
