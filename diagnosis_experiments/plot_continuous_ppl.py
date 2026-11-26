#!/usr/bin/env python3
"""Plot pooled PPL for the dual-alpha grid without mixing fixed settings."""
import argparse,csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    p=argparse.ArgumentParser();p.add_argument('--input_dir',required=True);p.add_argument('--font_family',default='DejaVu Sans');p.add_argument('--png_dpi',type=int,default=300);a=p.parse_args()
    font_manager.findfont(a.font_family,fallback_to_default=False);plt.rcParams['font.family']=a.font_family
    root=Path(a.input_dir);out=root/'figures';out.mkdir(exist_ok=True)
    with (root/'ppl_sweep_unique.csv').open() as f:rows=list(csv.DictReader(f))
    journal=[r for r in rows if r['method']=='coherekv_journal']
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=a.png_dpi);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    for prefix,xkey,ykey,fixed in [
        ('alpha','alpha_ea','alpha_ee',('lambda_ea','lambda_ee')),
        ('threshold','lambda_ea','lambda_ee',('alpha_ea','alpha_ee')),
    ]:
        groups=defaultdict(list)
        for r in journal:groups[tuple(r[k] for k in ('budget','reference_mode','priority_mode',*fixed))].append(r)
        for i,(k,rr) in enumerate(groups.items()):
            xs=sorted({float(r[xkey]) for r in rr});ys=sorted({float(r[ykey]) for r in rr})
            lookup={(float(r[xkey]),float(r[ykey])):float(r['overall_ppl']) for r in rr}
            title=' / '.join(f'{name}={val}' for name,val in zip(('budget','ref','priority',*fixed),k))
            if len(xs)>1 and len(ys)>1:
                z=np.array([[lookup.get((x,y),np.nan) for x in xs] for y in ys]);fig,ax=plt.subplots(figsize=(7,5));im=ax.imshow(z,origin='lower',aspect='auto');fig.colorbar(im,ax=ax,label='Pooled PPL')
                ax.set_xticks(range(len(xs)),xs);ax.set_yticks(range(len(ys)),ys);ax.set(xlabel=xkey,ylabel=ykey,title=title)
                for yi in range(len(ys)):
                    for xi in range(len(xs)):
                        if np.isfinite(z[yi,xi]):ax.text(xi,yi,f'{z[yi,xi]:.3f}',ha='center',va='center',fontsize=8)
                save(fig,f'{prefix}_heatmap_{i:03d}')
            for axis,other,values,other_values in [(xkey,ykey,xs,ys),(ykey,xkey,ys,xs)]:
                if len(values)<2:continue
                fig,ax=plt.subplots(figsize=(7,4))
                for v in other_values:
                    points=sorted((float(r[axis]),float(r['overall_ppl'])) for r in rr if float(r[other])==v)
                    if points:ax.plot(*zip(*points),marker='o',label=f'{other}={v:g}')
                ax.set(xlabel=axis,ylabel='Pooled PPL',title=title);ax.legend();ax.grid(alpha=.25)
                save(fig,f'{axis}_curve_{i:03d}')

    # Compare priority modes only within identical geometric settings.
    groups=defaultdict(list)
    fixed=('budget','reference_mode','alpha_ea','alpha_ee','lambda_ea','lambda_ee')
    for r in journal:groups[tuple(r[k] for k in fixed)].append(r)
    for i,(k,rr) in enumerate(groups.items()):
        if len(rr)<2:continue
        modes=('degree','count','importance','count_importance')
        rr=sorted(rr,key=lambda r:modes.index(r['priority_mode']))
        title=' / '.join(f'{name}={val}' for name,val in zip(fixed,k))
        for metric in ('overall_ppl','importance_retention_rate','raw_importance_retention_rate'):
            valid=[r for r in rr if r.get(metric) not in ('',None)]
            if not valid:continue
            fig,ax=plt.subplots(figsize=(8,4));ax.bar([r['priority_mode'] for r in valid],[float(r[metric]) for r in valid])
            ax.set(ylabel=metric,title=title);ax.tick_params(axis='x',labelrotation=15)
            if metric.endswith('rate'):ax.set_ylim(0,1.05)
            save(fig,f'priority_{metric}_{i:03d}')

if __name__=='__main__':main()
