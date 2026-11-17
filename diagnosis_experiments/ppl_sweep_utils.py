"""Dependency-free PPL sweep definitions, deduplication, summaries, and CSV helpers.

Each route threshold applies ONLY to its routing mode; E-E lambdas sweep MIS
independently. Controls that do not run MIS are evaluated once and copied to
all E-E settings with an explicit `evaluation_reused` flag.
"""
from __future__ import annotations
import csv
import math
import os
from pathlib import Path

from method_registry import METHODS,selected_methods
METHOD_COLUMNS=METHODS


def parse_floats_csv(value, name, *, lower=None, upper=None):
    if value is None:raise ValueError(f'Empty {name} sweep')
    values=[]
    for part in str(value).split(','):
        part=part.strip()
        if not part:continue
        try: v=float(part)
        except ValueError as e:raise ValueError(f'Invalid {name} entry: {part!r}') from e
        if (not math.isfinite(v) or
            (lower is not None and v<=lower) or
            (upper is not None and v>upper)):
            raise ValueError(f'Invalid {name} entry: {v}; required {lower} < value <= {upper}')
        if v not in values:values.append(v)
    if not values:raise ValueError(f'Empty {name} sweep')
    return sorted(values)


def routing_specs(modes, cosine_thresholds, l2_lambdas):
    if isinstance(modes,str):modes=[v.strip() for v in modes.split(',') if v.strip()]
    if not modes or len(set(modes))!=len(modes) or any(v not in ('cosine','l2') for v in modes):
        raise ValueError('Expected unique routing modes from cosine,l2')
    out=[]
    for mode in modes:
        settings=cosine_thresholds if mode=='cosine' else l2_lambdas
        for threshold in settings:out.append((mode,float(threshold)))
    return out


def numeric_key(x):
    return f'{float(x):.12g}'


def setting_key(mode,param,budget,ee,method,metric='l2'):
    return (str(mode),numeric_key(param),numeric_key(budget),numeric_key(ee),str(method),str(metric))


def computation_key(mode,param,budget,ee,method,metric='l2'):
    """Only MIS-dependent methods need re-evaluation per E-E cutoff."""
    if method=='full_cache':return ('full_cache',)
    if method in ('merge_all','importance_weighted_merge_all'):
        return (method,mode,numeric_key(param),numeric_key(budget))
    return (method,mode,numeric_key(param),numeric_key(budget),str(metric),numeric_key(ee))


def config_comparison_rows(rows,method_order=None):
    methods=tuple(method_order or dict.fromkeys(r['method'] for r in rows))
    if 'merge_all' not in methods:return []
    groups={}
    for r in rows:
        key=(r['routing_mode'],numeric_key(r['routing_parameter']),
             numeric_key(r['budget']),r.get('ee_metric','l2'),numeric_key(r['ee_lambda']))
        groups.setdefault(key,{})[r['method']]=r
    out=[]
    for (mode,param,budget,metric,ee),group in sorted(groups.items()):
        active_methods=tuple(m for m in methods if m in group)
        if 'merge_all' not in group or 'full_cache' not in group:continue
        base=float(group['merge_all']['ppl'])
        iwbase=float(group['importance_weighted_merge_all']['ppl']) if 'importance_weighted_merge_all' in group else None
        reference=next(iter(group.values()))
        entry=dict(routing_mode=mode,routing_parameter=float(param),budget=float(budget),
                   ee_lambda=float(ee),ee_metric=metric,sweep_id=int(reference.get('sweep_id',0)),
                   fixed_kv_budget_tokens=group['merge_all']['fixed_kv_budget_tokens'])
        entry['primary_baseline']='importance_weighted_merge_all' if iwbase is not None else 'merge_all'
        # Rectangular CSV schema even though L2 and cosine methods have
        # intentionally different threshold grids and different active arms.
        for m in methods:
            entry[m]=float(group[m]['ppl']) if m in group else ''
        for m in methods:
            if m not in ('full_cache','merge_all'):
                entry['delta_'+m]=(entry[m]-base if m in group else '')
                entry['relative_delta_pct_'+m]=(100*(entry[m]-base)/base if m in group else '')
            if 'importance_weighted_merge_all' in methods and m!='full_cache':
                entry['delta_vs_iw_'+m]=(entry[m]-iwbase if iwbase is not None and m in group else '')
                entry['relative_delta_pct_vs_iw_'+m]=(100*(entry[m]-iwbase)/max(iwbase,1e-12)
                    if iwbase is not None and m in group else '')
        out.append(entry)
    return out


def finalize_ppl_group_deltas(group):
    """Fill paired PPL deltas after *all* methods in a configuration finish.

    In particular, `merge_all` is evaluated before the optional IW control.
    Comparing it to IW while the loop is still on `merge_all` raised an
    incorrect ordering AssertionError. This works for normal and resumed rows.
    """
    merge=group.get('merge_all')
    iw=group.get('importance_weighted_merge_all')
    if merge is None:
        raise ValueError('Completed PPL group is missing Merge All')
    base=float(merge['ppl'])
    iw_ppl=float(iw['ppl']) if iw is not None else None
    for method,row in group.items():
        if method=='full_cache':
            row['delta_ppl_from_merge_all']=''
            row['relative_delta_ppl_pct']=''
            row['delta_ppl_from_iw_merge_all']=''
            row['relative_delta_ppl_from_iw_pct']=''
            continue
        p=float(row['ppl'])
        d=p-base
        row['delta_ppl_from_merge_all']=d
        row['relative_delta_ppl_pct']=100.0*d/max(base,1e-12)
        if iw_ppl is None:
            row['delta_ppl_from_iw_merge_all']=''
            row['relative_delta_ppl_from_iw_pct']=''
        else:
            delta=p-iw_ppl
            row['delta_ppl_from_iw_merge_all']=delta
            row['relative_delta_ppl_from_iw_pct']=100.0*delta/max(iw_ppl,1e-12)
    return group

def atomic_write_csv(path, rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def formatted_ppl_summary(group,mode,param,budget,ee,method_order=None,ee_metric='l2'):
    methods=tuple(method_order or group)
    vals={m:float(group[m]['ppl']) for m in methods}
    route=f'cos>={param:g}' if mode=='cosine' else f'L2lambda={param:g}'
    example=next(iter(group.values()))
    sweep=(f'E-E-lambda={ee:g} (frozen TRAIN S_ref)' if ee_metric=='l2'
           else f'E-E-cosine-distance={ee:g} (1 - cosine, no S_ref)')
    control=('importance_weighted_merge_all' if 'importance_weighted_merge_all' in vals
             else 'merge_all')
    lines=[f'[PPL RESULT] B={budget:.0%} routing={route} {sweep} | PPL (lower better; Delta vs {control})',
           f'  Full={vals["full_cache"]:.5f}  MergeAll={vals["merge_all"]:.5f}'+
           (f'  {control}={vals[control]:.5f}' if control!='merge_all' else '')]
    for m in methods:
        if m in ('full_cache','merge_all',control):continue
        diff=vals[m]-vals[control]
        lines.append(f'  {m}={vals[m]:.5f} | Delta={diff:+.5f} ({100*diff/max(vals[control],1e-12):+.3f}%)')
    return '\n'.join(lines)
