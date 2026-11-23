#!/usr/bin/env python3
"""Continuous PPL publication plots; exact Times New Roman, both PDF and PNG."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def read_rows(path):
    with path.open(newline='',encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def setup_font(name):
    if name!='Times New Roman':
        raise ValueError('Plots are required to use Times New Roman')
    try:
        font_manager.findfont(name,fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError('Times New Roman is not installed; install it locally and rerun plotting. '
                           'Numeric results are already packaged.') from error
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
                         'pdf.fonttype':42,'ps.fonttype':42,
                         'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                         'savefig.facecolor':'white'})


def write(fig,folder,stem,dpi):
    folder.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder/(stem+'.pdf'),bbox_inches='tight')
    fig.savefig(folder/(stem+'.png'),bbox_inches='tight',dpi=dpi)
    plt.close(fig)


def same_config(row,first):
    return all(str(row[key])==str(first[key]) for key in
               ('budget','lambda_ea','lambda_ee','p'))


def plot_all(rows,cycles,heads,out,dpi):
    keys=sorted({(r['budget'],r['lambda_ea'],r['lambda_ee'],r['p']) for r in rows},
                key=lambda x:tuple(float(y) for y in x))
    for config in keys:
        group=[r for r in rows if (r['budget'],r['lambda_ea'],r['lambda_ee'],r['p'])==config]
        cyc=[r for r in cycles if (r['budget'],r['lambda_ea'],r['lambda_ee'],r['p'])==config]
        if not group or not cyc:continue
        label='B{}_EA{}_EE{}_p{}'.format(*config).replace('.','p')
        # Per-cycle NLL: avoids misleading PPL means.
        fig,ax=plt.subplots(figsize=(8.5,5))
        for row in group:
            method=row['method']
            series=sorted((c for c in cyc if c['method']==method),
                          key=lambda c:int(c['cycle']))
            if series:ax.plot([int(r['cycle']) for r in series],
                              [float(r['cycle_nll_per_token']) for r in series],
                              marker='o',linewidth=1.5,label=method)
        ax.set_xlabel('Continuous compression/scoring cycle')
        ax.set_ylabel('NLL / scored token (lower better)')
        ax.set_title(f'Continuous PPL | B={config[0]}, EA={config[1]}, EE={config[2]}, p={config[3]}')
        ax.grid(alpha=.2);ax.legend(fontsize=7,loc='best')
        write(fig,out,'continuous_cycle_nll_'+label,dpi)
        # Comparison with zero-merge KeepKV eviction at identical slot budget.
        eviction=next((float(r['nll_per_token']) for r in group
                       if r['method']=='eviction_only'),None)
        if eviction is not None:
            pairs=[(r['method'],float(r['nll_per_token'])-eviction)
                   for r in group if r['method'] not in ('full_cache','eviction_only')]
            if pairs:
                fig,ax=plt.subplots(figsize=(max(7,0.40*len(pairs)+4),4.6))
                ax.bar(range(len(pairs)),[v for _,v in pairs])
                ax.axhline(0,color='gray',linewidth=1)
                ax.set_xticks(range(len(pairs)),[m for m,_ in pairs],rotation=45,ha='right',fontsize=8)
                ax.set_ylabel('NLL/token - EvictionOnly (negative better)')
                ax.set_title(f'Merging gain vs eviction only | B={config[0]}, EA={config[1]}, EE={config[2]}, p={config[3]}')
                ax.grid(axis='y',alpha=.2)
                write(fig,out,'continuous_delta_nll_vs_eviction_'+label,dpi)
        # Paired total NLL differences from MergeAll.
        merge=next((float(row['nll_per_token']) for row in group if row['method']=='merge_all'),None)
        if merge is not None:
            pairs=[(r['method'],float(r['nll_per_token'])-merge) for r in group
                   if r['method'] not in ('merge_all','full_cache','eviction_only')]
            if pairs:
                fig,ax=plt.subplots(figsize=(max(7,0.40*len(pairs)+4),4.6))
                ax.bar(range(len(pairs)),[v for _,v in pairs])
                ax.axhline(0,color='gray',linewidth=1)
                ax.set_xticks(range(len(pairs)),[m for m,_ in pairs],rotation=45,ha='right',fontsize=8)
                ax.set_ylabel('NLL/token - MergeAll (negative better)')
                ax.set_title(f'Paired continuous NLL | B={config[0]}, EA={config[1]}, EE={config[2]}, p={config[3]}')
                ax.grid(axis='y',alpha=.2)
                write(fig,out,'continuous_delta_nll_'+label,dpi)
    # p comparison for same budget and lambda pair, aggregated over scored tokens.
    settings=sorted({(r['budget'],r['lambda_ea'],r['lambda_ee']) for r in rows},
                    key=lambda s:tuple(map(float,s)))
    for budget,ea,ee in settings:
        candidates=[r for r in rows if (r['budget'],r['lambda_ea'],r['lambda_ee'])==(budget,ea,ee)
                    and r['method'] not in ('merge_all','full_cache','eviction_only')]
        if not candidates:continue
        fig,ax=plt.subplots(figsize=(8.5,5))
        values=sorted({float(r['p']) for r in candidates})
        for method in sorted({r['method'] for r in candidates}):
            series=sorted((r for r in candidates if r['method']==method),key=lambda r:float(r['p']))
            positions=[values.index(float(r['p'])) for r in series]
            ys=[float(r['nll_per_token']) for r in series]
            ax.plot(positions,ys,marker='o',label=method)
        ax.set_xticks(range(len(values)),['inf' if math.isinf(x) else f'{x:g}' for x in values])
        ax.set_xlabel('Historical p (inf = conference limit)')
        ax.set_ylabel('Total scored NLL/token (lower better)')
        ax.set_title(f'Historical p sweep | budget={budget}, EA={ea}, EE={ee}')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
        suffix=f'B{budget}_EA{ea}_EE{ee}'.replace('.','p')
        write(fig,out,'p_sweep_nll_'+suffix,dpi)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_dir',required=True)
    parser.add_argument('--font_family',default='Times New Roman')
    parser.add_argument('--png_dpi',type=int,default=600)
    args=parser.parse_args()
    root=Path(args.input_dir)
    setup_font(args.font_family)
    rows=read_rows(root/'ppl_continuous.csv')
    cycles=read_rows(root/'ppl_continuous_cycles.csv')
    heads=read_rows(root/'ppl_continuous_head_rounds.csv') if (root/'ppl_continuous_head_rounds.csv').exists() else []
    plot_all(rows,cycles,heads,root/'figures',args.png_dpi)
    print('[PLOT]',root/'figures',flush=True)

if __name__=='__main__':main()
