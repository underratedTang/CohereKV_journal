#!/usr/bin/env python3
"""Sweep publication plots: pooled PPL, per-window PPL, paired NLL, baselines."""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

LABELS={'full_cache':'Full Cache','eviction_only':'Eviction-only','merge_all':'MergeAll',
        'coherekv_conference':'Conference CohereKV','coherekv_soft_mis':'Degree MIS',
        'coherekv_soft_mis_importance':'Importance MIS',
        'coherekv_soft_mis_log_count':'Degree MIS / log Count',
        'coherekv_soft_mis_importance_log_count':'Importance MIS / log Count'}
CANDIDATES=tuple(m for m in LABELS if m.startswith('coherekv_soft_mis'))

def read_rows(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def setup_font(name):
    if name!='Times New Roman':raise ValueError('Times New Roman required')
    try:font_manager.findfont(name,fallback_to_default=False)
    except ValueError as e:raise RuntimeError('Install Times New Roman locally; numeric results already packaged') from e
    plt.rcParams.update({'font.family':'serif','font.serif':[name], 'pdf.fonttype':42,'ps.fonttype':42,
                         'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'savefig.facecolor':'white',
                         'mathtext.fontset':'custom','mathtext.rm':name,'mathtext.it':name+':italic','mathtext.bf':name+':bold'})

def write(fig,folder,stem,dpi):
    folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/(stem+'.pdf'),bbox_inches='tight')
    fig.savefig(folder/(stem+'.png'),bbox_inches='tight',dpi=dpi)
    plt.close(fig)

def fmt(x):return 'inf' if math.isinf(float(x)) else f'{float(x):g}'
def grid_matrix(rows,ea,ee,field):
    lookup={(float(r['lambda_ea']),float(r['lambda_ee'])):float(r[field]) for r in rows}
    return np.array([[lookup.get((a,e),np.nan) for a in ea] for e in ee])

def heatmaps(rows,out,stem,dpi,field='ppl'):
    for budget in sorted({float(r['budget']) for r in rows}):
        data=[r for r in rows if float(r['budget'])==budget and r['method'] in CANDIDATES and r.get(field,'')!='']
        if not data:continue
        ea=sorted({float(r['lambda_ea']) for r in data});ee=sorted({float(r['lambda_ee']) for r in data})
        all_data=data
        for gamma in sorted({float(r['gamma']) for r in all_data}):
            data=[r for r in all_data if float(r['gamma'])==gamma]
            ps=sorted({float(r['p']) for r in data});n=len(ps)
            vals=[float(r[field]) for r in all_data];lo=min(vals);hi=max(vals)
            if hi==lo:hi=lo+1e-10
            active=[m for m in CANDIDATES if any(r['method']==m for r in data)]
            fig,axes=plt.subplots(len(active),n,squeeze=False,figsize=(3.6*n+0.6,3.1*len(active)),layout='constrained')
            for i,m in enumerate(active):
                for j,p in enumerate(ps):
                    ax=axes[i,j];selected=[r for r in data if r['method']==m and float(r['p'])==p]
                    if not selected:
                        ax.set_axis_off();continue
                    matrix=grid_matrix(selected,ea,ee,field)
                    im=ax.imshow(matrix,vmin=lo,vmax=hi,cmap='viridis_r',aspect='auto',origin='lower')
                    for y,x in np.ndindex(matrix.shape):
                        val=matrix[y,x]
                        if np.isfinite(val):ax.text(x,y,f'{val:.3f}',ha='center',va='center',fontsize=8,color='white' if (val-lo)/(hi-lo)>.55 else 'black')
                    if np.isfinite(matrix).any():
                        y,x=np.unravel_index(np.nanargmin(matrix),matrix.shape)
                        ax.add_patch(plt.Rectangle((x-.49,y-.49),.98,.98,fill=False,edgecolor='red',linewidth=1.5))
                    ax.set_xticks(range(len(ea)),[fmt(x) for x in ea]);ax.set_yticks(range(len(ee)),[fmt(x) for x in ee])
                    ax.set_xlabel(r'$\lambda_{EA}$');ax.set_ylabel(r'$\lambda_{EE}$')
                    ax.set_title(f'{LABELS[m]} | p={fmt(p)}, gamma={fmt(gamma)}')
            fig.colorbar(im,ax=axes.ravel().tolist(),label='PPL' if field=='ppl' else 'NLL difference vs Full Cache',shrink=.85)
            write(fig,out,f'{stem}_B{fmt(budget)}_G{fmt(gamma)}'.replace('.','p'),dpi)

def baseline_plot(unique,out,dpi):
    budgets=sorted({float(r['actual_budget']) for r in unique if r['actual_budget']!=''}) or [None]
    for b in budgets:
        fig,ax=plt.subplots(figsize=(6.5,4),layout='constrained')
        for m in ('merge_all','coherekv_conference'):
            rows=sorted((r for r in unique if r['method']==m and float(r['actual_budget'])==b),key=lambda r:float(r['actual_lambda_ea']))
            if rows:ax.plot([float(r['actual_lambda_ea']) for r in rows],[float(r['ppl']) for r in rows],marker='o',label=LABELS[m])
        for m in ('full_cache','eviction_only'):
            rows=[r for r in unique if r['method']==m and (r['actual_budget']=='' or float(r['actual_budget'])==b)]
            if rows:ax.axhline(float(rows[0]['ppl']),linestyle='--',label=LABELS[m])
        if not ax.lines:
            plt.close(fig);continue
        ax.set_xlabel('Lambda (EA for MergeAll; shared for Conference)');ax.set_ylabel('Overall PPL');ax.legend();ax.grid(alpha=.2)
        write(fig,out,('baseline_sweep_full_only' if b is None else f'baseline_sweep_B{fmt(b)}'.replace('.','p')),dpi)

def best_plot(best,paired,out,dpi):
    for budget in sorted({float(r['actual_budget']) for r in best if r['actual_budget']!=''}) or [None]:
        rows=[r for r in best if r['actual_budget']=='' or float(r['actual_budget'])==budget]
        labels=[];vals=[];samples=[]
        for r in rows:
            m=r['method'];label=LABELS[m]
            if m in CANDIDATES:label+=' p='+fmt(r['actual_p'])+' g='+fmt(r['actual_gamma'])
            labels.append(label);vals.append(float(r['ppl']))
            match=[x for x in paired if x['method']==m and (budget is None or float(x['budget'])==budget)]
            if m in CANDIDATES:match=[x for x in match if float(x['p'])==float(r['actual_p']) and float(x['gamma'])==float(r['actual_gamma'])]
            if r['actual_lambda_ea']!='':match=[x for x in match if float(x['lambda_ea'])==float(r['actual_lambda_ea'])]
            if m in CANDIDATES:match=[x for x in match if float(x['lambda_ee'])==float(r['actual_lambda_ee'])]
            seen={x['sequence_index']:float(x['delta_nll_vs_full']) for x in match if x.get('delta_nll_vs_full','')!=''}
            samples.append(list(seen.values()))
        fig,ax=plt.subplots(figsize=(max(8,0.48*len(labels)),5.5),layout='constrained');ax.bar(range(len(vals)),vals)
        ax.set_xticks(range(len(vals)),labels,rotation=30,ha='right');ax.set_ylabel('Overall PPL');ax.set_title('Best grid setting per method / p / gamma');ax.grid(axis='y',alpha=.2)
        write(fig,out,('best_ppl_full_only' if budget is None else f'best_ppl_B{fmt(budget)}'.replace('.','p')),dpi)
        if all(samples):
            fig,ax=plt.subplots(figsize=(max(8,0.48*len(labels)),5.5),layout='constrained');ax.boxplot(samples,showmeans=True)
            ax.set_xticks(range(1,len(labels)+1),labels,rotation=30,ha='right');ax.axhline(0,linestyle='--',color='black',linewidth=.8)
            ax.set_ylabel('Paired NLL difference vs Full Cache');ax.set_title('Sequence variation at selected settings');ax.grid(axis='y',alpha=.2)
            write(fig,out,('best_paired_nll_full_only' if budget is None else f'best_paired_nll_B{fmt(budget)}'.replace('.','p')),dpi)


def pgamma_plot(best,out,dpi):
    """Best validation Lambda pair for each actual p/gamma, not test selection."""
    budgets=sorted({float(r['actual_budget']) for r in best if r['method'] in CANDIDATES})
    for budget in budgets:
        data=[r for r in best if r['method'] in CANDIDATES and float(r['actual_budget'])==budget]
        ps=sorted({float(r['actual_p']) for r in data})
        gs=sorted({float(r['actual_gamma']) for r in data})
        active=[m for m in CANDIDATES if any(r['method']==m for r in data)]
        lo=min(float(r['ppl']) for r in data);hi=max(float(r['ppl']) for r in data)
        if hi==lo:hi=lo+1e-10
        fig,axes=plt.subplots(1,len(active),squeeze=False,figsize=(3.8*len(active)+.6,3.5),layout='constrained')
        for i,m in enumerate(active):
            ax=axes[0,i]
            lookup={(float(r['actual_p']),float(r['actual_gamma'])):float(r['ppl']) for r in data if r['method']==m}
            matrix=np.array([[lookup.get((p,g),np.nan) for g in gs] for p in ps])
            im=ax.imshow(matrix,vmin=lo,vmax=hi,cmap='viridis_r',aspect='auto',origin='lower')
            for y,x in np.ndindex(matrix.shape):
                val=matrix[y,x]
                if np.isfinite(val):ax.text(x,y,f'{val:.3f}',ha='center',va='center',fontsize=8,color='white' if (val-lo)/(hi-lo)>.55 else 'black')
            ax.set_xticks(range(len(gs)),[fmt(x) for x in gs])
            ax.set_yticks(range(len(ps)),[fmt(x) for x in ps])
            ax.set_xlabel('Decay gamma');ax.set_ylabel('Geometry p');ax.set_title(LABELS[m])
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Best Overall PPL over Lambda grid',shrink=.85)
        write(fig,out,f'best_p_gamma_B{fmt(budget)}'.replace('.','p'),dpi)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input_dir',required=True);p.add_argument('--font_family',default='Times New Roman');p.add_argument('--png_dpi',type=int,default=600);a=p.parse_args()
    root=Path(a.input_dir);out=root/'figures';setup_font(a.font_family)
    rows=read_rows(root/'ppl_continuous.csv');seq=read_rows(root/'ppl_continuous_per_sequence.csv')
    heatmaps(rows,out,'overall_ppl_heatmap',a.png_dpi)
    heatmaps(rows,out,'overall_delta_nll_heatmap',a.png_dpi,field='delta_nll_vs_full')
    for idx in sorted({int(r['sequence_index']) for r in seq}):
        heatmaps([r for r in seq if int(r['sequence_index'])==idx],out/'per_sequence',f'seq_{idx}_ppl_heatmap',a.png_dpi)
    baseline_plot(read_rows(root/'ppl_sweep_unique.csv'),out,a.png_dpi)
    best=read_rows(root/'ppl_sweep_best_by_method_p_gamma.csv')
    best_plot(best,read_rows(root/'ppl_continuous_paired_deltas.csv'),out,a.png_dpi)
    pgamma_plot(best,out,a.png_dpi)
    print('[PLOT]',out,flush=True)

if __name__=='__main__':main()
