#!/usr/bin/env python3
"""Focused mechanism figures from oracle future-q group and add-back CSVs.

Figures use saved REAL future-query diagnostics only; they do not influence
selection, routing, thresholds or actual model predictions.
"""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_cosine_mis import apply_font,save


def rows(path):
    if not path.exists():return []
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def scatter(data,out,name,xkey,ykey,xlabel,ylabel,title,dpi):
    if not data:return
    fig,ax=plt.subplots(figsize=(7,5))
    labels={'k_only_importance_iwmean':'Count×I D-first'}
    colors={'k_only_importance_iwmean':'#AA3C83'}
    for fam in sorted({r['mis_family'] for r in data}):
        part=[r for r in data if r['mis_family']==fam]
        ax.scatter([float(r[xkey]) for r in part],[float(r[ykey]) for r in part],
                   s=15,alpha=.38,label=f'{labels.get(fam,fam)} (n={len(part)})',
                   color=colors.get(fam,'#555555'),rasterized=True)
    ax.axhline(0,lw=1,linestyle='--',color='#555555')
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
    ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.tight_layout();save(fig,out,name,dpi)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input_dir',required=True)
    p.add_argument('--font_family',default='Times New Roman')
    p.add_argument('--strict_times_new_roman',type=int,choices=(0,1),default=1)
    p.add_argument('--png_dpi',type=int,default=600)
    a=p.parse_args();root=Path(a.input_dir)
    apply_font(a.font_family,bool(a.strict_times_new_roman))
    groups=rows(root/'mis_group_future_query_diagnostics.csv')
    addbacks=rows(root/'mis_rejected_addback_diagnostics.csv')
    out=root/'figures'
    scatter(groups,out,'MECHANISM_count_drop_vs_future_mass_error',
            'rejected_count_fraction','delta_mass_mae_mis_vs_iw',
            'Rejected Count fraction','MIS-Drop − IW-MergeAll group log-mass MAE',
            'ORACLE future queries | Negative = MIS better',a.png_dpi)
    if groups:
        for r in groups:
            r['anchor_key_drift_delta']=str(float(r['anchor_key_drift_mis'])-float(r['anchor_key_drift_iw']))
    scatter(groups,out,'MECHANISM_anchor_drift_vs_mass_error',
            'anchor_key_drift_delta','delta_mass_mae_mis_vs_iw',
            'MIS − IW anchor Key drift', 'MIS-Drop − IW-MergeAll group log-mass MAE',
            'ORACLE future queries | Geometry vs attention',a.png_dpi)
    scatter(addbacks,out,'MECHANISM_rejected_addback_key_distance',
            'evicted_anchor_key_distance','delta_mass_mae_addback_minus_mis',
            'Rejected token − Anchor Key distance', 'Add-back − MIS group log-mass MAE',
            'ORACLE add-back | Negative = candidate helps',a.png_dpi)
    print(f'[MECHANISM FIGURES] groups={len(groups)} addbacks={len(addbacks)} folder={out}',flush=True)
if __name__=='__main__':main()
