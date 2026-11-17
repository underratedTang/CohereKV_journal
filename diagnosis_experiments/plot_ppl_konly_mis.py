#!/usr/bin/env python3
"""K-only degree vs E-A geometric priority PPL sweep figures: separate routing and E-E dimensions.

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
ACTIVE_EE_METRIC='l2'

def sweep_id(lam):
    return float(lam)

def paired_label(lam):
    return (f'MIS lambda={lam:g} (TRAIN S_ref)' if ACTIVE_EE_METRIC=='l2'
            else f'Cosine E-E distance delta={lam:g}')

def set_sweep_axis(ax,ees):
    ax.set_xticks([float(e) for e in ees],[f'{float(e):g}' for e in ees])
    ax.set_xlabel('MIS lambda (TRAIN S_ref multiplier)' if ACTIVE_EE_METRIC=='l2'
                  else 'Cosine E-E distance delta = 1 - cosine')

def route_tag(mode,param):
    return f'{mode}_{param:.8g}'.replace('.','p').replace('-','m')


def route_title(mode,param):
    return f'Cosine routing >= {param:g}' if mode=='cosine' else f'L2 routing lambda = {param:g}'


def read_full_results(path):
    with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('No PPL rows')
    keys=set()
    for r in rows:
        key=(r['routing_mode'],float(r['routing_parameter']),r.get('ee_metric','l2'),
             float(r['ee_lambda']),float(r['budget']),r['method'])
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
    methods=['importance_weighted_merge_all','merge_all',*MIS_METHODS]
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
    ax.set_title(f'{route_title(mode,param)} | KV budget {budget:.0%} | E-E {ACTIVE_EE_METRIC} threshold sweep')
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


def plot_primary_priority_ppl(rows,out,dpi,mode,param,focus):
    """PPL of fixed Count Mean and Drop: priority ablation against Count-MergeAll."""
    main=('merge_all','k_only_mis_drop','k_only_degree_anchor_mis_drop',
          'k_only_anchor_first_mis_drop','k_only_importance_mis_drop')
    if ACTIVE_EE_METRIC=='cosine':
        main=('merge_all',*(x+'_cosee' for x in main[1:]))
    budgets=sorted({float(r['budget']) for r in rows
                    if r['routing_mode']==mode and close(r['routing_parameter'],param)})
    if not budgets or lookup(rows,main[0],mode,param,budgets[0],focus) is None:return
    fig,axs=plt.subplots(1,2,figsize=(13,4.8))
    total=0
    for method in main:
        xy=[(b,lookup(rows,method,mode,param,b,focus)) for b in budgets]
        xy=[(b,v) for b,v in xy if v is not None]
        if not xy:continue
        axs[0].plot([100*b for b,_ in xy],[v for _,v in xy],marker='o',lw=1.8,
                    color=COL[method],label=LABEL[method]);total+=1
        delta=[]
        for b,v in xy:
            baseline=lookup(rows,main[0],mode,param,b,focus)
            if baseline is not None:delta.append((b,v-baseline))
        if delta:
            axs[1].plot([100*b for b,_ in delta],[v for _,v in delta],marker='o',lw=1.8,
                        color=COL[method],label=LABEL[method])
    if total<2:plt.close(fig);return
    axs[0].set_title('Multi-round PPL (lower better)');axs[0].set_ylabel('Test perplexity')
    axs[1].axhline(0,ls=':',color=COL[main[0]])
    axs[1].set_title('MIS priority effect');axs[1].set_ylabel('PPL minus Count-MergeAll')
    for ax in axs:
        ax.set_xlabel('Fixed physical KV budget (%)');ax.grid(alpha=.17);ax.legend(fontsize=7)
    fig.suptitle(f'{route_title(mode,param)} | E-E lambda={focus:g} | Degree / E-A priority ablation')
    fig.tight_layout();save(fig,out,f'PRIMARY_ppl_degree_anchor_priority_{route_tag(mode,param)}_ee_{focus:g}',dpi)



def plot_cosine_vs_l2(rows,out,dpi):
    """For each physical method compare best *observed* PPL per E-E metric.

    This is an in-sample sweep descriptive summary, not a validated optimum.
    Baselines are reused and never attributed to a particular metric.
    """
    from method_registry import canonical_method
    import csv
    merged={}
    baseline={}
    for r in rows:
        config=(r['routing_mode'],float(r['routing_parameter']),float(r['budget']))
        m=r['method']
        if m in ('full_cache','merge_all','importance_weighted_merge_all'):
            baseline[(config,m)]=float(r['ppl'])
            continue
        metric=r.get('ee_metric','l2')
        base=canonical_method(m)+('_sink' if m.endswith('_sink') else '')
        key=(config,base,metric)
        val=float(r['ppl']);ee=float(r['ee_lambda'])
        if key not in merged or val<merged[key][0]:merged[key]=(val,ee,m)
    summary=[]
    pairs=sorted({(config,base) for (config,base,metric) in merged})
    for config,base in pairs:
        l2=merged.get((config,base,'l2'))
        cos=merged.get((config,base,'cosine'))
        if l2 is None or cos is None:continue
        row=dict(routing_mode=config[0],routing_parameter=config[1],budget=config[2],
                 base_method=base,merge_all_ppl=baseline.get((config,'merge_all'),''),
                 l2_best_ppl=l2[0],l2_best_lambda=l2[1],
                 cosine_best_ppl=cos[0],cosine_best_delta=cos[1],
                 cosine_minus_l2_ppl=cos[0]-l2[0])
        summary.append(row)
    if summary:
        csv_file=out.parent/'ppl_ee_metric_best_observed.csv'
        with csv_file.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        for config in sorted({(x['routing_mode'],x['routing_parameter'],x['budget']) for x in summary}):
            selected=[x for x in summary if (x['routing_mode'],x['routing_parameter'],x['budget'])==config]
            fig,ax=plt.subplots(figsize=(max(7,1.4*len(selected)),4.5))
            xs=list(range(len(selected)))
            ax.bar([x-.19 for x in xs],[r['l2_best_ppl'] for r in selected],width=.36,label='Euclidean (best scanned lambda)')
            ax.bar([x+.19 for x in xs],[r['cosine_best_ppl'] for r in selected],width=.36,label='Cosine (best scanned delta)')
            ax.set_xticks(xs,[LABEL.get(r['base_method'],r['base_method']) for r in selected],rotation=22,ha='right')
            ax.set_ylabel('Observed minimum PPL (lower better)')
            ax.set_title(f'Routing {config[0]}={config[1]:g}; budget={config[2]:.0%} (in-sample scan)')
            ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
            fig.tight_layout()
            save(fig,out,f'ppl_best_metric_{route_tag(config[0],config[1])}_b_{config[2]:g}',dpi)


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
    metrics=sorted({r.get('ee_metric','l2') for r in rows})
    global ACTIVE_EE_METRIC
    for metric in metrics:
        ACTIVE_EE_METRIC=metric
        part=[r for r in rows if r.get('ee_metric','l2')==metric]
        metric_out=out/metric
        metric_ees=sorted({float(r['ee_lambda']) for r in part})
        target=(focus if metric=='l2' else metric_ees[0])
        routes=sorted({(r['routing_mode'],float(r['routing_parameter'])) for r in part})
        for mode,param in routes:
            plot_primary_priority_ppl(part,metric_out,a.png_dpi,mode,param,target)
            plot_budget_comparison(part,metric_out,a.png_dpi,mode,param,target)
            plot_budget_delta(part,metric_out,a.png_dpi,mode,param,target)
            budgets=sorted({float(r['budget']) for r in part if r['routing_mode']==mode and
                            close(r['routing_parameter'],param)})
            for budget in budgets:
                plot_ee_sweep(part,metric_out,a.png_dpi,mode,param,budget)
                plot_ee_delta(part,metric_out,a.png_dpi,mode,param,budget)
        for mode in sorted({m for m,_ in routes}):
            budgets=sorted({float(r['budget']) for r in part if r['routing_mode']==mode})
            for budget in budgets:plot_route_sweep(part,metric_out,a.png_dpi,mode,budget,target)
    plot_cosine_vs_l2(rows,out,a.png_dpi)
    print(f'[PPL FIGURES COMPLETE] {out}',flush=True)

if __name__=='__main__':main()
