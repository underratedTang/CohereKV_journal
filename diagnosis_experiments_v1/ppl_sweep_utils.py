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


def setting_key(mode,param,budget,ee,method):
    return (str(mode),numeric_key(param),numeric_key(budget),numeric_key(ee),str(method))


def computation_key(mode,param,budget,ee,method):
    """Only MIS-dependent methods need re-evaluation per E-E cutoff."""
    if method=='full_cache':return ('full_cache',)
    if method in ('merge_all','importance_weighted_merge_all'):
        return (method,mode,numeric_key(param),numeric_key(budget))
    return (method,mode,numeric_key(param),numeric_key(budget),numeric_key(ee))


def config_comparison_rows(rows,method_order=None):
    methods=tuple(method_order or dict.fromkeys(r['method'] for r in rows))
    if 'merge_all' not in methods:return []
    groups={}
    for r in rows:
        key=(r['routing_mode'],numeric_key(r['routing_parameter']),
             numeric_key(r['budget']),numeric_key(r['ee_lambda']))
        groups.setdefault(key,{})[r['method']]=r
    out=[]
    for (mode,param,budget,ee),group in sorted(groups.items()):
        if not all(m in group for m in methods):continue
        base=float(group['merge_all']['ppl'])
        reference=next(iter(group.values()))
        entry=dict(routing_mode=mode,routing_parameter=float(param),budget=float(budget),
                   ee_lambda=float(ee),sweep_id=int(reference.get('sweep_id',0)),
                   fixed_kv_budget_tokens=group['merge_all']['fixed_kv_budget_tokens'])
        for m in methods:entry[m]=float(group[m]['ppl'])
        for m in methods:
            if m not in ('full_cache','merge_all'):
                entry['delta_'+m]=entry[m]-base
                entry['relative_delta_pct_'+m]=100*(entry[m]-base)/base
        out.append(entry)
    return out

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


def formatted_ppl_summary(group,mode,param,budget,ee,method_order=None):
    methods=tuple(method_order or group)
    vals={m:float(group[m]['ppl']) for m in methods}
    route=f'cos>={param:g}' if mode=='cosine' else f'L2lambda={param:g}'
    example=next(iter(group.values()))
    sweep=f'K-only-lambda={ee:g} (degree vs importance priority)'
    lines=[f'[PPL RESULT] B={budget:.0%} routing={route} {sweep} | PPL (lower better; Δ vs Merge All)',
           f'  Full={vals["full_cache"]:.5f}  MergeAll={vals["merge_all"]:.5f}']
    for m in methods:
        if m in ('full_cache','merge_all'):continue
        diff=vals[m]-vals['merge_all']
        lines.append(f'  {m}={vals[m]:.5f} | Delta={diff:+.5f} ({100*diff/vals["merge_all"]:+.3f}%)')
    return '\n'.join(lines)
