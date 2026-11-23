#!/usr/bin/env python3
"""Group-D2 continuous PPL figures (Times New Roman; paired PDF and PNG)."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def setup_font(name):
    if name!='Times New Roman':raise ValueError('Times New Roman is required')
    try:font_manager.findfont(name,fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError('Times New Roman unavailable. Numerical results were already packaged; '
                           'install Times New Roman on the server to generate PNG/PDF.') from exc
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
        'pdf.fonttype':42,'ps.fonttype':42,'font.size':10,
        'axes.titlesize':11,'axes.labelsize':10,'savefig.facecolor':'white'})


def write(fig,folder,stem,dpi):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder/(stem+'.pdf'),bbox_inches='tight')
    fig.savefig(folder/(stem+'.png'),bbox_inches='tight',dpi=dpi)
    plt.close(fig)


def read_rows(path):
    with open(path,newline='',encoding='utf8') as f:return list(csv.DictReader(f))


def plot(root,dpi):
    root=Path(root);out=root/'figures'
    rows=read_rows(root/'ppl_continuous.csv')
    cycles=read_rows(root/'ppl_continuous_cycles.csv')
    heads=read_rows(root/'ppl_continuous_head_rounds.csv')
    if not rows or not cycles or not heads:raise RuntimeError('Missing completed PPL/cycle/head diagnostics')
    methods=list(dict.fromkeys(r['method'] for r in rows))
    fig,ax=plt.subplots(figsize=(9.6,5.2))
    for m in methods:
        vals=sorted([r for r in cycles if r['method']==m],key=lambda r:int(r['cycle']))
        if vals:ax.plot([int(x['cycle']) for x in vals],
                        [float(x['cycle_nll_per_token']) for x in vals],
                        linewidth=1.4,marker='o',markersize=2.8,label=m)
    ax.set_xlabel('Compression / scoring cycle');ax.set_ylabel('NLL per scored token')
    ax.set_title(f'Group-D2: {max(int(x["cycle"]) for x in cycles)} continuous compression rounds')
    ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
    write(fig,out,'group_d2_continuous_cycle_nll',dpi)

    group=[r for r in rows if r['method']!='full_cache']
    fig,ax=plt.subplots(figsize=(9.6,5))
    ax.barh([r['method'] for r in group],[int(float(r['accepted_slots'])) for r in group])
    ax.set_xlabel('Accepted routed physical states across all heads / rounds')
    ax.set_title('Fusion volume (trajectories differ after first compression)')
    ax.grid(axis='x',alpha=.2)
    write(fig,out,'group_d2_accepted_states',dpi)

    group=[r for r in rows if r['method'].startswith('group_d2')]
    if group:
        fig,ax=plt.subplots(figsize=(9.6,5))
        ax.barh([r['method'] for r in group],[int(float(r.get('group_d2_real_violations',0) or 0)) for r in group])
        ax.set_xlabel('Actual history D2 violations among routed anchor groups')
        ax.set_title('Full Group-D2 must have zero violations')
        ax.grid(axis='x',alpha=.2)
        write(fig,out,'group_d2_history_violations',dpi)

    selected=[r for r in heads if r['method']=='group_d2']
    if selected:
        grouped={}
        for r in selected:
            if r.get('real_history_d2_over_delta_max','')=='':continue
            grouped.setdefault(int(r['round']),[]).append(float(r['real_history_d2_over_delta_max']))
        if grouped:
            x=sorted(grouped)
            fig,ax=plt.subplots(figsize=(9.6,5))
            ax.plot(x,[max(grouped[r]) for r in x],marker='o',markersize=3)
            ax.axhline(1,color='gray',linestyle='--',linewidth=1)
            ax.set_xlabel('Compression round')
            ax.set_ylabel('Max exact historical D2 / threshold')
            ax.set_title('Full Group-D2 tightness across all layers / KV heads')
            ax.grid(alpha=.2)
            write(fig,out,'group_d2_max_history_ratio',dpi)
    print('[PLOT] Times New Roman PNG+PDF:',out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input_dir',required=True)
    ap.add_argument('--font_family',default='Times New Roman')
    ap.add_argument('--png_dpi',type=int,default=600)
    a=ap.parse_args();setup_font(a.font_family);plot(a.input_dir,a.png_dpi)

if __name__=='__main__':main()
