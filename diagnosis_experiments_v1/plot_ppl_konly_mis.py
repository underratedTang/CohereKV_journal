#!/usr/bin/env python3
"""K-only original-degree vs importance-difference MIS PPL sweep figures: separate route threshold and E-E dimensions.

No implicit averaging over E-E lambda. Writes vector PDF and 600dpi PNG via
low-memory PDF rasterizer, using the existing diagnosis plotting conventions.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_cosine_mis import apply_font,save,COL,LABEL,MIS_METHODS


def close(a,b):return abs(float(a)-float(b))<1e-8

SWEEP_BY_LAMBDA={}

def sweep_id(lam):
    return float(lam)

def paired_label(lam):
    return f'MIS lambda={lam:g} (TRAIN S_ref)'

def set_sweep_axis(ax,ees):
    ax.set_xticks([float(e) for e in ees],[f'{float(e):g}' for e in ees])
    ax.set_xlabel('MIS lambda (TRAIN S_ref multiplier)')

def route_tag(mode,param):
    return f'{mode}_{param:.8g}'.replace('.','p').replace('-','m')


def route_title(mode,param):
    return f'Cosine routing >= {param:g}' if mode=='cosine' else f'L2 routing lambda = {param:g}'


def read_full_results(path):
    with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('No PPL rows')
    keys=set()
    for r in rows:
        key=(r['routing_mode'],float(r['routing_parameter']),float(r['ee_lambda']),
             float(r['budget']),r['method'])
        if key in keys:raise ValueError(f'Duplicate PPL sweep configuration: {key}')
        keys.add(key)
    return rows


def lookup(rows,method,mode,param,budget,ee):
    found=[float(r['ppl']) for r in rows
           if r['method']==method and r['routing_mode']==mode and
           close(r['routing_parameter'],param) and close(r['budget'],budget) and
           close(r['ee_lambda'],ee)]
    if len(found)!=1:return None
    return found[0]


def plot_budget_comparison(rows,out,dpi,mode,param,focus):
    budgets=sorted({float(r['budget']) for r in rows
                    if r['routing_mode']==mode and close(r['routing_parameter'],param)})
    fig,ax=plt.subplots(figsize=(9.3,5.1))
    count=0
    for method in LABEL:
        xy=[(100*b,lookup(rows,method,mode,param,b,focus)) for b in budgets]
        xy=[(x,y) for x,y in xy if y is not None]
        if not xy:continue
        ax.plot([x for x,_ in xy],[y for _,y in xy],marker='o',linewidth=1.6,
                color=COL[method],label=LABEL[method]);count+=1
    if not count:plt.close(fig);return
    ax.set_xticks([100*b for b in budgets],[f'{b:.0%}' for b in budgets])
    ax.set_xlabel('Fixed physical KV budget')
    ax.set_ylabel('Test perplexity (lower is better)')
    ax.set_title(f'{route_title(mode,param)} | {paired_label(focus)}')
    ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.tight_layout();save(fig,out,f'ppl_budget_{route_tag(mode,param)}_ee_{focus:g}',dpi)


def plot_ee_sweep(rows,out,dpi,mode,param,budget):
    ees=sorted({float(r['ee_lambda']) for r in rows
                if r['routing_mode']==mode and close(r['routing_parameter'],param)
                and close(r['budget'],budget)})
    if len(ees)<2:return
    fig,ax=plt.subplots(figsize=(9.3,5.1));count=0
    methods=['merge_all',*MIS_METHODS]
    for method in methods:
        xy=[(sweep_id(ee),lookup(rows,method,mode,param,budget,ee)) for ee in ees]
        xy=[(x,y) for x,y in xy if y is not None]
        if not xy:continue
        ax.plot([x for x,_ in xy],[y for _,y in xy],marker='o',linewidth=1.7,
                linestyle='--' if method=='merge_all' else '-',
                color=COL[method],label=LABEL[method]);count+=1
    if not count:plt.close(fig);return
    set_sweep_axis(ax,ees)
    ax.set_ylabel('Test perplexity (lower is better)')
    ax.set_title(f'{route_title(mode,param)} | KV budget {budget:.0%} | Shared K-only threshold sweep')
    ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.tight_layout();save(fig,out,f'ppl_ee_sweep_{route_tag(mode,param)}_budget_{budget:g}',dpi)


def plot_route_sweep(rows,out,dpi,mode,budget,focus):
    params=sorted({float(r['routing_parameter']) for r in rows if r['routing_mode']==mode})
    if len(params)<2:return
    fig,ax=plt.subplots(figsize=(9.3,5.1));count=0
    for method in LABEL:
        xy=[(p,lookup(rows,method,mode,p,budget,focus)) for p in params]
        xy=[(x,y) for x,y in xy if y is not None]
        if not xy:continue
        ax.plot([x for x,_ in xy],[y for _,y in xy],marker='o',linewidth=1.7,
                color=COL[method],label=LABEL[method]);count+=1
    if not count:plt.close(fig);return
    ax.set_xlabel('Cosine routing threshold' if mode=='cosine' else 'L2 routing lambda')
    ax.set_ylabel('Test perplexity (lower is better)')
    ax.set_title(f'{mode.upper()} routing sweep | budget={budget:.0%}, {paired_label(focus)}')
    ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.tight_layout();save(fig,out,f'ppl_routing_sweep_{mode}_budget_{budget:g}_ee_{focus:g}',dpi)


def plot_budget_delta(rows,out,dpi,mode,param,focus):
    budgets=sorted({float(r['budget']) for r in rows
                    if r['routing_mode']==mode and close(r['routing_parameter'],param)})
    fig,ax=plt.subplots(figsize=(9.3,5.1));count=0
    for method in MIS_METHODS:
        xy=[]
        for b in budgets:
            value=lookup(rows,method,mode,param,b,focus)
            base=lookup(rows,'merge_all',mode,param,b,focus)
            if value is not None and base is not None:xy.append((100*b,value-base))
        if not xy:continue
        ax.plot([x for x,_ in xy],[y for _,y in xy],marker='o',linewidth=1.7,
                color=COL[method],label=LABEL[method]);count+=1
    if not count:plt.close(fig);return
    ax.axhline(0,color=COL['merge_all'],ls=':',lw=1.2,label='Merge All baseline')
    ax.set_xticks([100*b for b in budgets],[f'{b:.0%}' for b in budgets])
    ax.set_xlabel('Fixed physical KV budget')
    ax.set_ylabel('PPL minus Merge All (negative better)')
    ax.set_title(f'{route_title(mode,param)} | Paired PPL delta, {paired_label(focus)}')
    ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.tight_layout();save(fig,out,f'ppl_delta_budget_{route_tag(mode,param)}_ee_{focus:g}',dpi)


def plot_ee_delta(rows,out,dpi,mode,param,budget):
    ees=sorted({float(r['ee_lambda']) for r in rows
                if r['routing_mode']==mode and close(r['routing_parameter'],param)
                and close(r['budget'],budget)})
    if len(ees)<2:return
    fig,ax=plt.subplots(figsize=(9.3,5.1));count=0
    for method in MIS_METHODS:
        xy=[]
        for ee in ees:
            value=lookup(rows,method,mode,param,budget,ee)
            base=lookup(rows,'merge_all',mode,param,budget,ee)
            if value is not None and base is not None:xy.append((sweep_id(ee),value-base))
        if not xy:continue
        ax.plot([x for x,_ in xy],[y for _,y in xy],marker='o',linewidth=1.7,
                color=COL[method],label=LABEL[method]);count+=1
    if not count:plt.close(fig);return
    ax.axhline(0,color=COL['merge_all'],ls=':',lw=1.2,label='Merge All baseline')
    set_sweep_axis(ax,ees)
    ax.set_ylabel('PPL minus Merge All (negative better)')
    ax.set_title(f'{route_title(mode,param)} | KV budget {budget:.0%} | Paired PPL delta')
    ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.tight_layout();save(fig,out,f'ppl_delta_ee_{route_tag(mode,param)}_budget_{budget:g}',dpi)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input_dir',required=True)
    p.add_argument('--font_family',default='Times New Roman')
    p.add_argument('--strict_times_new_roman',type=int,default=1)
    p.add_argument('--png_dpi',type=int,default=600)
    p.add_argument('--focus_ee_lambda',type=float,default=0.5)
    a=p.parse_args()
    root=Path(a.input_dir).resolve();src=root/'ppl_konly_importance_mis.csv'
    if not src.is_file():raise FileNotFoundError(src)
    rows=read_full_results(src)
    global SWEEP_BY_LAMBDA
    SWEEP_BY_LAMBDA={round(float(r['ee_lambda']),10):r for r in rows}
    apply_font(a.font_family,bool(a.strict_times_new_roman))
    ees=sorted({float(r['ee_lambda']) for r in rows})
    focus=min(ees,key=lambda e:abs(e-a.focus_ee_lambda))
    if not close(focus,a.focus_ee_lambda):
        print(f'[PPL PLOT] requested E-E focus {a.focus_ee_lambda:g} not present; using {focus:g}',flush=True)
    out=root/'figures'
    routes=sorted({(r['routing_mode'],float(r['routing_parameter'])) for r in rows})
    for mode,param in routes:
        plot_budget_comparison(rows,out,a.png_dpi,mode,param,focus)
        plot_budget_delta(rows,out,a.png_dpi,mode,param,focus)
        budgets=sorted({float(r['budget']) for r in rows if r['routing_mode']==mode and
                        close(r['routing_parameter'],param)})
        for budget in budgets:
            plot_ee_sweep(rows,out,a.png_dpi,mode,param,budget)
            plot_ee_delta(rows,out,a.png_dpi,mode,param,budget)
    for mode in sorted({m for m,_ in routes}):
        budgets=sorted({float(r['budget']) for r in rows if r['routing_mode']==mode})
        for budget in budgets:plot_route_sweep(rows,out,a.png_dpi,mode,budget,focus)
    print(f'[PPL FIGURES COMPLETE] {out}',flush=True)

if __name__=='__main__':main()
