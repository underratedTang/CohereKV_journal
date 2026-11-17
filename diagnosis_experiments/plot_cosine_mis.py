#!/usr/bin/env python3
"""Low-RSS publication figures comparing degree and E-A MIS priorities and retained controls.

Each chart is produced in a separate process, saved to vector PDF, then
rasterized to 600-DPI PNG with Poppler/Ghostscript (never huge Matplotlib Agg).
Per-probe CSV is streamed, not loaded into Python dictionaries.
"""
from __future__ import annotations
import argparse,csv,gc,shutil,subprocess,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

COL={'full_cache':'#737373','merge_all':'#2563EB',
     'k_only_mis_drop':'#0891B2','k_only_mis_drop_allcount':'#7C3AED',
     'k_only_degree_anchor_mis_drop':'#E66A17',
     'k_only_anchor_first_mis_drop':'#C35076',
     'k_only_importance_mis_drop':'#0D9488',
     'k_only_importance_mis_drop_allcount':'#C2410C',
     'k_only_mis_drop_sink':'#19769C',
     'k_only_mis_drop_allcount_sink':'#6434A8',
     'k_only_importance_mis_drop_sink':'#15785A',
     'k_only_importance_mis_drop_allcount_sink':'#AC401B'}
LABEL={'full_cache':'Full Cache','merge_all':'Merge All',
       'k_only_mis_drop':'K-only MIS-Drop (Degree-first)',
       'k_only_degree_anchor_mis_drop':'Degree × E-A² MIS-Drop',
       'k_only_anchor_first_mis_drop':'E-A-first MIS-Drop',
       'k_only_mis_drop_allcount':'K-only MIS-Drop AllCount',
       'k_only_importance_mis_drop':'Importance-D MIS-Drop',
       'k_only_importance_mis_drop_allcount':'Importance-D MIS-AllCount',
       'k_only_mis_drop_sink':'K-only Drop / Sink-only',
       'k_only_mis_drop_allcount_sink':'K-only AllCount / Sink-only',
       'k_only_importance_mis_drop_sink':'Importance-D Drop / Sink-only',
       'k_only_importance_mis_drop_allcount_sink':'Importance-D AllCount / Sink-only'}
# Retained Count/Importance fusion controls; keep the old palette stable.
COL.update({
    'importance_weighted_merge_all':'#2C4D92',
    'k_only_importance_imean_drop':'#9C3A18',
    'k_only_importance_imean_drop_allcount':'#D36F35',
    'k_only_importance_imean_drop_sink':'#A65B36',
    'k_only_importance_imean_drop_allcount_sink':'#CC8750',
    'k_only_importance_iwmean_drop':'#AA3C83',
})
LABEL.update({
    'importance_weighted_merge_all':'C×I MergeAll',
    'k_only_importance_imean_drop':'K-only D-first + I-only Drop',
    'k_only_importance_imean_drop_allcount':'K-only D-first + I-only AllCount',
    'k_only_importance_imean_drop_sink':'K-only D-first + I-only Drop (sink)',
    'k_only_importance_imean_drop_allcount_sink':'K-only D-first + I-only AllCount (sink)',
    'k_only_importance_iwmean_drop':'E-E D-first + IW Drop',
})
# Each method's cosine E-E ablation differs only in the graph metric;
# use corresponding palette, dashed line style via separate plot sections.
from method_registry import MIS_METHODS as _MIS_REGISTRY
for _m in _MIS_REGISTRY:
    if not (_m.endswith('_cosee') or _m.endswith('_cosee_sink')):continue
    _base=_m[:-6] if _m.endswith('_cosee') else _m[:-11]+'_sink'
    LABEL[_m]=LABEL.get(_base,_base.replace('_',' ').title())+' [Cosine E-E]'
    COL[_m]=COL.get(_base,'#4B5563')
MIS_METHODS=tuple(m for m in LABEL if m not in ('full_cache','merge_all',
                                     'importance_weighted_merge_all'))
FUSED_MARKERS={'merge_all':'D','k_only_mis_drop':'o',
               'k_only_mis_drop_allcount':'v','k_only_importance_mis_drop':'P',
               'k_only_importance_mis_drop_allcount':'^',
               'k_only_mis_drop_sink':'s',
               'k_only_mis_drop_allcount_sink':'d',
               'k_only_importance_mis_drop_sink':'X',
               'k_only_importance_mis_drop_allcount_sink':'*'}

def active_methods(rows,include_baselines=True):
    found={r['method'] for r in rows}
    return [m for m in LABEL if m in found and (include_baselines or m in MIS_METHODS)]

def apply_font(family,strict):
    if strict and family!='Times New Roman':raise ValueError('Strict font must be Times New Roman')
    try:font_manager.findfont(family,fallback_to_default=False)
    except ValueError as ex:
        if strict:raise RuntimeError('Times New Roman font not found. Numeric ZIP was already created. Install font and re-run plotting.') from ex
        family='Liberation Serif'
    plt.rcParams.update({'font.family':'serif','font.serif':[family],
        'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white',
        'axes.spines.top':False,'axes.spines.right':False})


def save(fig,folder,name,dpi):
    folder.mkdir(parents=True,exist_ok=True)
    pdf=folder/(name+'.pdf');png=folder/(name+'.png')
    try:fig.savefig(pdf,bbox_inches='tight')
    finally:
        plt.close(fig);gc.collect()
    if shutil.which('pdftoppm'):
        subprocess.run(['pdftoppm','-f','1','-l','1','-singlefile',
                        '-r',str(dpi),'-png',str(pdf),str(folder/name)],check=True)
    elif shutil.which('gs'):
        subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE',
                        '-sDEVICE=png16m',f'-r{dpi}',f'-sOutputFile={png}',str(pdf)],check=True)
    else:
        raise RuntimeError('Low-memory 600-DPI PNG needs pdftoppm or gs; vector PDF already saved: '+str(pdf))
    print('[FIG]',name,flush=True)


def read(path):
    if not path.is_file():return []
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def num(row,k):return float(row[k])
def eq(x,y):return abs(float(x)-float(y))<1e-7
def mean(rows,k):return float(np.mean([num(r,k) for r in rows])) if rows else np.nan

def route_key(row):return (row['routing_mode'],row['routing_parameter_type'],float(row['routing_parameter']))
def route_specs(rows):return sorted(set(route_key(row) for row in rows))
def subset(rows,spec):return [row for row in rows if route_key(row)==spec]
def route_title(spec):
    mode,kind,value=spec
    if mode=='cosine':return f'Cosine R-E Top1: similarity >= {value:g}'
    if kind=='lambda':return f'L2 R-E Top1: distance <= {value:g} x S_calib (Median)'
    return f'L2 R-E Top1: distance <= {value:g} (absolute Key L2)'
def route_slug(spec):
    m,k,x=spec
    value=(f'{x:.8g}').replace('-','m').replace('.','p')
    return f'{m}_{k}_{value}'
SWEEP_BY_LAMBDA={}
ACTIVE_EE_METRIC='l2'

def sweep_id(lam):
    return float(lam)

def paired_label(lam):
    return (f'MIS lambda = {lam:g} (TRAIN S_ref multiplier)' if ACTIVE_EE_METRIC=='l2'
            else f'E-E cosine distance delta = {lam:g}')

def set_sweep_axis(ax,lambdas):
    ax.set_xticks([float(x) for x in lambdas],[f'{float(x):g}' for x in lambdas])
    ax.set_xlabel('MIS lambda (TRAIN S_ref multiplier)' if ACTIVE_EE_METRIC=='l2'
                  else 'Cosine E-E distance delta = 1 - cosine')

def figure_title(spec,lam):return f'{route_title(spec)} | {paired_label(lam)}'


def _aligned_method_panel(data,out,dpi,lam,spec,*,delta):
    """Common-scale horizontal per-budget panels with method names aligned.

    The old vertical grouped bars squeezed method names into 3 very wide
    panels, obscuring small Relative-Attention-L2 differences.
    """
    data=subset(data,spec)
    methods=active_methods(data,include_baselines=not delta)
    budgets=sorted(set(num(r,'budget') for r in data))
    if not methods or not budgets:return
    metric='delta_rel_from_merge_all' if delta else 'output_rel_error'
    values={b:[mean([r for r in data if eq(r['ee_lambda'],lam) and
             eq(r['budget'],b) and r['method']==m],metric) for m in methods]
            for b in budgets}
    finite=[v for arr in values.values() for v in arr if np.isfinite(v)]
    if not finite:return
    lo=min(finite+[0.]);hi=max(finite+[0.]);spread=max(hi-lo,1e-4)
    if delta:
        radius=max(abs(lo),abs(hi),1e-5)
        xmin,xmax=-radius*1.42,radius*1.42
    else:
        xmin,xmax=0,max(hi*1.24,1e-4)
    fig,axs=plt.subplots(1,len(budgets),figsize=(5.0*len(budgets),max(4.0,0.63*len(methods)+1.9)),
                         sharex=True,sharey=True,squeeze=False)
    y=np.arange(len(methods))
    for ax,b in zip(axs.flat,budgets):
        arr=values[b]
        for idx,(m,v) in enumerate(zip(methods,arr)):
            if not np.isfinite(v):continue
            ax.hlines(idx,0,v,lw=1.7,color=COL[m],alpha=.7)
            ax.scatter([v],[idx],s=52,c=COL[m],zorder=3)
            ax.annotate(f'{v:+.5f}' if delta else f'{v:.5f}',(v,idx),
                        xytext=(5,0),textcoords='offset points',fontsize=8,
                        ha='left',va='center',color=COL[m])
        ax.axvline(0,ls=':',lw=1.2,color=COL['merge_all'])
        ax.set_xlim(xmin,xmax)
        ax.set_yticks(y,[LABEL[m] for m in methods],fontsize=8.7)
        ax.set_ylim(len(methods)-.5,-.5)
        ax.set_title(f'KV Budget {b:.0%}')
        ax.set_xlabel('Δ Relative-Attention-L2' if delta else 'Relative-Attention-L2')
        ax.grid(axis='x',alpha=.16)
    if delta:
        fig.suptitle(figure_title(spec,lam)+' | Difference from Merge All (negative is better)',y=1.03)
        name='method_delta_vs_merge_all_'+route_slug(spec)
    else:
        fig.suptitle(figure_title(spec,lam)+' | Real-Query Relative-Attention-L2 (lower is better)',y=1.03)
        name='method_error_'+route_slug(spec)
    fig.tight_layout();save(fig,out,name,dpi)


def plot_method_comparison(data,out,dpi,lam,spec):
    _aligned_method_panel(data,out,dpi,lam,spec,delta=False)


def plot_method_delta(data,out,dpi,lam,spec):
    _aligned_method_panel(data,out,dpi,lam,spec,delta=True)


def plot_lambda_sensitivity(data,out,dpi,spec):
    data=subset(data,spec)
    methods=active_methods(data,include_baselines=False)
    budgets=sorted(set(num(r,'budget') for r in data))
    lambdas=sorted(set(num(r,'ee_lambda') for r in data))
    fig,axs=plt.subplots(1,len(budgets),figsize=(5.2*len(budgets),4.2),squeeze=False)
    for ax,b in zip(axs.flat,budgets):
        ax.axhline(0,color=COL['merge_all'],ls=':',lw=1.4,label='Merge All baseline')
        for m in methods:
            yy=[mean([r for r in data if eq(r['ee_lambda'],lam) and eq(r['budget'],b) and r['method']==m],
                'delta_rel_from_merge_all') for lam in lambdas]
            ax.plot([sweep_id(x) for x in lambdas],yy,marker='o',lw=1.7,color=COL[m],label=LABEL[m])
        set_sweep_axis(ax,lambdas)
        ax.set_title(f'KV Budget {b:.0%}')
        ax.set_ylabel('Relative-Attention-L2 minus Merge All (negative better)')
        ax.grid(alpha=.17);ax.legend(fontsize=7)
    fig.suptitle(f'{route_title(spec)} | Constant R-E assignment / group counts',y=1.025)
    fig.tight_layout();save(fig,out,'attention_delta_vs_ee_'+route_slug(spec),dpi)


def plot_mis_coverage(data,out,dpi,spec):
    data=subset(data,spec)
    methods=[m for m in active_methods(data,include_baselines=False)
             if m.endswith('_mis_drop') or m.endswith('_mis_drop_sink')]
    lambdas=sorted(set(num(r,'ee_lambda') for r in data))
    budgets=sorted(set(num(r,'budget') for r in data))
    fig,axs=plt.subplots(1,2,figsize=(11,4.15))
    for b in budgets:
        merged=[r for r in data if eq(r['budget'],b) and r['method']=='merge_all']
        coverage=[mean([r for r in merged if eq(r['ee_lambda'],lam)],'routed_fraction') for lam in lambdas]
        axs[0].plot([sweep_id(x) for x in lambdas],coverage,marker='o',label=f'Budget {b:.0%}')
        for method in methods:
            rr=[r for r in data if eq(r['budget'],b) and r['method']==method]
            rejected=[mean([r for r in rr if eq(r['ee_lambda'],lam)],'mis_rejected_fraction') for lam in lambdas]
            axs[1].plot([sweep_id(x) for x in lambdas],rejected,marker='o',color=COL[method],
                        ls='-' if b==budgets[0] else ':',label=f'{LABEL[method]}, {b:.0%}')
    axs[0].set_ylabel('Routed / Evicted (fraction)')
    axs[1].set_ylabel('MIS Rejected / Routed (fraction)')
    for ax in axs:
        set_sweep_axis(ax,lambdas);ax.grid(alpha=.17);ax.legend(fontsize=7)
    fig.suptitle(f'{route_title(spec)} | Shared K-only E-E conflicts: degree vs importance priority')
    fig.tight_layout();save(fig,out,'routing_coverage_vs_ee_'+route_slug(spec),dpi)


def plot_conditional_by_layer(data,out,dpi,lam,spec):
    data=subset(data,spec)
    methods=active_methods(data,include_baselines=False)
    relevant=[r for r in data if eq(r['ee_lambda'],lam) and num(r,'n_mis_rejected')>0 and r['method'] in methods]
    if not relevant:return
    layers=sorted(set(int(r['layer_idx']) for r in relevant))
    budgets=sorted(set(num(r,'budget') for r in relevant))
    ncols=2 if len(methods)>1 else 1
    nrows=int(np.ceil(len(methods)/ncols))
    fig,axs=plt.subplots(nrows,ncols,figsize=(6.3*ncols,3.4*nrows),squeeze=False)
    xs=np.arange(len(layers));width=.75/max(1,len(budgets))
    for idx,meth in enumerate(methods):
        ax=axs.flat[idx]
        for i,b in enumerate(budgets):
            y=[mean([r for r in relevant if eq(r['budget'],b) and int(r['layer_idx'])==l
                     and r['method']==meth],'delta_rel_from_merge_all') for l in layers]
            ax.bar(xs+i*width-.375+width/2,y,width,label=f'Budget {b:.0%}')
        ax.axhline(0,color='black',lw=.7)
        ax.set_xticks(xs,[str(l) for l in layers],fontsize=7);ax.set_title(LABEL[meth]);ax.set_xlabel('Layer')
        ax.set_ylabel('Relative-Attention-L2 minus Merge All')
        ax.grid(axis='y',alpha=.13);ax.legend(fontsize=7)
    for idx in range(len(methods),len(axs.flat)):axs.flat[idx].axis('off')
    fig.suptitle(f'{figure_title(spec,lam)} | conditioned on each method MIS rejection')
    fig.tight_layout();save(fig,out,'conditional_by_layer_'+route_slug(spec),dpi)


def plot_routing_sweep(data,out,dpi,lam,mode,kind):
    base=[r for r in data if r['routing_mode']==mode and r['routing_parameter_type']==kind and eq(r['ee_lambda'],lam)]
    methods=active_methods(base,include_baselines=False)
    xx=sorted(set(num(r,'routing_parameter') for r in base))
    budgets=sorted(set(num(r,'budget') for r in base))
    fig,axs=plt.subplots(2,2,figsize=(11.8,8.1))
    for b in budgets:
        rr=[r for r in base if eq(r['budget'],b)]
        xdata=[r for r in rr if r['method']=='merge_all']
        label=f'Budget {b:.0%}'
        axs[0,0].plot(xx,[mean([r for r in xdata if eq(r['routing_parameter'],x)],'routed_fraction') for x in xx],marker='o',label=label)
        for meth in methods:
            if not meth.endswith('_mis_drop'):continue
            mdata=[r for r in rr if r['method']==meth]
            axs[0,1].plot(xx,[mean([r for r in mdata if eq(r['routing_parameter'],x)],'mis_rejected_fraction') for x in xx],marker='o',color=COL[meth],label=f'{LABEL[meth]} / {label}')
        for method in ('merge_all',*methods):
            rows=[r for r in rr if r['method']==method]
            axs[1,0].plot(xx,[mean([r for r in rows if eq(r['routing_parameter'],x)],'output_rel_error') for x in xx],
                          ls='--' if method=='merge_all' else '-',marker='o',ms=3,color=COL[method],label=f'{LABEL[method]} / {label}')
        for method in methods:
            rows=[r for r in rr if r['method']==method]
            axs[1,1].plot(xx,[mean([r for r in rows if eq(r['routing_parameter'],x)],'delta_rel_from_merge_all') for x in xx],
                          marker='o',ms=3,color=COL[method],label=f'{LABEL[method]} / {label}')
    axs[0,0].set_title('R-E Routing coverage');axs[0,0].set_ylabel('Routed / Evicted')
    axs[0,1].set_title('E-E MIS rejection');axs[0,1].set_ylabel('Rejected / Routed')
    axs[1,0].set_title('Real-Q Attention Output Error');axs[1,0].set_ylabel('Mean relative L2 error (lower better)')
    axs[1,1].set_title('Paired delta for each selected method')
    axs[1,1].set_ylabel('Relative-Attention-L2 minus Merge All (negative better)')
    axs[1,1].axhline(0,color='black',linewidth=.8,ls=':')
    xlabel=('R-E Cosine similarity threshold' if mode=='cosine' else
        ('R-E L2 cutoff / frozen Median S_calib' if kind=='lambda' else 'R-E physical L2 cutoff'))
    for ax in axs.flat:
        ax.set_xlabel(xlabel);ax.grid(alpha=.17)
        handles,labels=ax.get_legend_handles_labels()
        if handles:ax.legend(fontsize=6.6,ncol=2)
    fig.suptitle(f'{mode.upper()} Routing sweep | {paired_label(lam)}',y=1.01)
    fig.tight_layout();save(fig,out,f'routing_sweep_{mode}_{kind}',dpi)


def plot_routing_ee_heatmap(data,out,dpi,mode,kind):
    methods=active_methods([r for r in data if r['routing_mode']==mode and r['routing_parameter_type']==kind],include_baselines=False)
    for method in methods:
        relevant=[r for r in data if r['method']==method and r['routing_mode']==mode and r['routing_parameter_type']==kind]
        if not relevant:continue
        xs=sorted(set(num(r,'ee_lambda') for r in relevant))
        ys=sorted(set(num(r,'routing_parameter') for r in relevant))
        grid=np.array([[mean([r for r in relevant if eq(r['ee_lambda'],ee) and eq(r['routing_parameter'],rr)],
                            'delta_rel_from_merge_all') for ee in xs] for rr in ys])
        fig,ax=plt.subplots(figsize=(max(6.,len(xs)*1.),max(4.6,len(ys)*.65)))
        vmax=max(1e-6,float(np.nanmax(np.abs(grid))))
        im=ax.imshow(grid,cmap='coolwarm',vmin=-vmax,vmax=vmax,aspect='auto',origin='lower')
        fig.colorbar(im,ax=ax,label='Relative-Attention-L2 minus Merge All (negative better)')
        ax.set_xticks(np.arange(len(xs)),[f'{a:g}' for a in xs]);ax.set_yticks(np.arange(len(ys)),[f'{a:g}' for a in ys])
        ax.set_xlabel('Shared K-only lambda' if ACTIVE_EE_METRIC=='l2' else 'E-E cosine distance delta')
        ax.set_ylabel('Cosine threshold' if mode=='cosine' else ('R-E L2 / Median S_calib' if kind=='lambda' else 'R-E L2 cutoff'))
        ax.set_title(f'{LABEL[method]} | {mode.upper()} routing / E-E MIS')
        fig.tight_layout();save(fig,out,f'routing_ee_heatmap_{mode}_{kind}_{method}',dpi)


def stream_probe_means(path,lam,ee_metric='l2'):
    if not path.is_file():return {}
    sums=defaultdict(float);counts=defaultdict(int)
    with path.open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if not eq(row['ee_lambda'],lam) or row['method'] not in MIS_METHODS or row.get('ee_metric','l2')!=ee_metric:continue
            key=(route_key(row),row['method'],int(row['future_token_offset']))
            sums[key]+=float(row['delta_rel_from_merge_all']);counts[key]+=1
    by_route=defaultdict(dict)
    for (spec,method,offset),value in sums.items():
        by_route[spec][(method,offset)]=value/counts[(spec,method,offset)]
    return dict(by_route)


def plot_per_probe(averages,out,dpi,lam,spec):
    if not averages:return
    offsets=sorted({p for method,p in averages})
    methods=[m for m in MIS_METHODS if any((m,p) in averages for p in offsets)]
    fig,ax=plt.subplots(figsize=(10.8,4.4))
    for method in methods:
        yy=[averages.get((method,p),np.nan) for p in offsets]
        ax.plot(offsets,yy,label=LABEL[method],color=COL[method],lw=1.6)
    ax.axhline(0,color=COL['merge_all'],ls=':',lw=1.3)
    ax.set_xlabel('Real continuation token offset')
    ax.set_ylabel('Relative-Attention-L2 minus Merge All (negative better)')
    ax.set_title(figure_title(spec,lam)+' | paired real Query')
    ax.grid(alpha=.17);ax.legend()
    fig.tight_layout();save(fig,out,'real_query_per_token_'+route_slug(spec),dpi)


def local_pca(x,means):
    centered=x-x[0]
    if np.linalg.norm(centered)<1e-8:
        basis=np.eye(x.shape[1],2)
        return centered@basis,(means-x[0])@basis,0.
    _,s,vh=np.linalg.svd(centered,full_matrices=False)
    basis=vh[:2].T
    if basis.shape[1]<2:basis=np.pad(basis,((0,0),(0,2-basis.shape[1])))
    return centered@basis,(means-x[0])@basis,float((s[:2]**2).sum()/max(1e-12,(s*s).sum()))


def plot_examples(root,out,dpi,example_paths):
    for path in example_paths:
        with np.load(path,allow_pickle=False) as o:
            x=o['key_original'].astype(np.float64)
            ids=o['original_positions'];chosen=o['mis_selected'].astype(bool)
            means=o['fused_keys'].astype(np.float64);names=o['fused_names']
            spec=(str(o['routing_mode']),str(o['routing_parameter_type']),float(o['routing_parameter']))
            ee=float(path.stem.split('_lam')[-1].split('_')[0])
            cutoff=float(o['routing_l2_cutoff'])
        xy,center,ratio=local_pca(x,means)
        fig,ax=plt.subplots(figsize=(7.,5.9))
        fam=('k_only_importance' if any(str(m).startswith('k_only_importance') for m in names) else 'k_only')
        accept_color=COL['k_only_importance_mis_drop' if fam=='k_only_importance' else 'k_only_mis_drop']
        reject_color=COL['k_only_importance_mis_drop_allcount' if fam=='k_only_importance' else 'k_only_mis_drop_allcount']
        ax.scatter(xy[chosen,0],xy[chosen,1],c=accept_color,s=52,label=f'{fam} MIS Accepted (plus anchor)')
        if (~chosen).any():ax.scatter(xy[~chosen,0],xy[~chosen,1],c=reject_color,s=85,marker='x',label=f'{fam} MIS Rejected E')
        ax.scatter(xy[0,0],xy[0,1],c=COL['merge_all'],marker='*',s=225,zorder=5,label='Retained anchor')
        for pt,name in zip(center,names):
            ax.scatter([pt[0]],[pt[1]],c=COL[str(name)],s=91,marker=FUSED_MARKERS.get(str(name),'o'),
                       edgecolor='white',linewidths=.5,label=f'Fused Key: {LABEL[str(name)]}',zorder=6)
        if len(x)<=15:
            for pt,idx in zip(xy,ids):ax.annotate(str(idx),pt,xytext=(3,3),textcoords='offset points',fontsize=7)
        routing_text=route_title(spec)
        if spec[0]=='l2':routing_text+=f' | actual L2 cutoff={cutoff:.3g}'
        ax.set_title(f'{routing_text}\nE-E distance threshold = {ee:g} | A={ids[0]}\n'
                     f'Accepted={int(chosen.sum())-1}, rejected={int((~chosen).sum())}; PCA var={ratio:.1%}',fontsize=9)
        ax.set_xlabel('Group PCA axis 1 (Key units)');ax.set_ylabel('Group PCA axis 2 (Key units)')
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.15);ax.legend(fontsize=7)
        fig.text(.5,.01,'PCA display only; MIS uses full-dimensional vectors; within-family Drop variants share fused K/V.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.035,1,1));save(fig,out,'group_geometry_'+path.stem,dpi)


def plot_calibration_heatmap(root,out,dpi):
    import json
    p=root/'calibration'/'median_reference.json'
    if not p.exists():raise FileNotFoundError('Missing frozen Median calibration: '+str(p))
    obj=json.loads(p.read_text(encoding='utf-8'))
    if obj.get('aggregation')!='median_across_sequences':raise ValueError('Not independently Median-calibrated')
    mat=np.asarray(obj['layer_head_scales'],dtype=np.float64)
    fig,ax=plt.subplots(figsize=(max(7.,mat.shape[1]*.62),max(5.,mat.shape[0]*.28)))
    im=ax.imshow(mat,origin='lower',aspect='auto',cmap='viridis')
    fig.colorbar(im,ax=ax,label='Median RMS pairwise Key L2 distance')
    ax.set_xlabel('KV head');ax.set_ylabel('Transformer layer')
    ax.set_xticks(np.arange(mat.shape[1]));ax.set_title('Frozen independent train-set reference | Median across calibration sequences')
    fig.tight_layout();save(fig,out,'frozen_median_reference_layer_head',dpi)
    vmat=np.asarray(obj['value_layer_head_scales'],dtype=np.float64)
    if vmat.shape!=mat.shape:raise ValueError('Frozen K/V calibration shape mismatch')
    fig,ax=plt.subplots(figsize=(max(7.,vmat.shape[1]*.62),max(5.,vmat.shape[0]*.28)))
    im=ax.imshow(vmat,origin='lower',aspect='auto',cmap='viridis')
    fig.colorbar(im,ax=ax,label='Median RMS pairwise Value L2 distance')
    ax.set_xlabel('KV head');ax.set_ylabel('Transformer layer')
    ax.set_xticks(np.arange(vmat.shape[1]))
    ax.set_title('Frozen independent Value reference | Median across train sequences')
    fig.tight_layout();save(fig,out,'frozen_value_median_reference_layer_head',dpi)
    sink=np.asarray(obj['sink_mass_mean'],dtype=np.float64)
    pres=sink>=float(obj['sink_presence_threshold'])
    fig,ax=plt.subplots(figsize=(max(7.,sink.shape[1]*.68),max(5.,sink.shape[0]*.29)))
    im=ax.imshow(sink,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='viridis')
    fig.colorbar(im,ax=ax,label='Last-query Full Attention to first token')
    ys,xs=np.where(~pres)
    ax.scatter(xs,ys,marker='x',s=38,color='red',linewidths=1.0,label='Non-Sink (diagnostic only)')
    ax.set_xlabel('KV Head');ax.set_ylabel('Layer');ax.set_xticks(np.arange(sink.shape[1]))
    ax.legend(fontsize=8)
    ax.set_title('Last-query first-token Sink (StreamingLLM-style, train only)')
    fig.tight_layout();save(fig,out,'train_sink_presence_layer_head',dpi)
    # StreamingLLM Figure-12-inspired per-Q-head diagnostic before GQA
    # aggregation. The binary action remains per KV head, not per Q head.
    qcsv=root/'calibration'/'sink_first_token_by_query_head.csv'
    qrows=read(qcsv)
    if qrows:
        layers=max(int(x['layer_idx']) for x in qrows)+1
        qheads=max(int(x['query_head_idx']) for x in qrows)+1
        qmat=np.full((layers,qheads),np.nan,dtype=np.float64)
        for row in qrows:
            qmat[int(row['layer_idx']),int(row['query_head_idx'])]=float(row['first_token_attention_mean'])
        if not np.isfinite(qmat).all():raise ValueError('Missing first-token Q head calibration rows')
        fig,ax=plt.subplots(figsize=(max(10,qheads*.3),max(5,layers*.25)))
        im=ax.imshow(qmat,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='viridis')
        fig.colorbar(im,ax=ax,label='Last-query attention to token 0')
        ax.set_xlabel('Query Head');ax.set_ylabel('Layer')
        ax.set_title('StreamingLLM-style first-token attention per Query Head (train)')
        fig.tight_layout();save(fig,out,'train_first_token_sink_query_head',dpi)
        # StreamingLLM Figure-12-style layer mean with across-Q-head std.
        qmean=qmat.mean(axis=1);qstd=qmat.std(axis=1)
        fig,ax=plt.subplots(figsize=(9,4.7))
        ax.errorbar(np.arange(layers),qmean,yerr=qstd,marker='o',markersize=3,
            lw=1.2,elinewidth=.9,capsize=2,color='#2563EB')
        ax.set_ylim(0,1.05)
        ax.set_xlabel('Layer');ax.set_ylabel('Attention weight on first token')
        ax.set_title('StreamingLLM-style 4096th-query Sink | Q-head variation')
        ax.grid(alpha=.15)
        fig.tight_layout();save(fig,out,'train_first_token_sink_layer_with_head_std',dpi)



def plot_five_method_by_layer(data,out,dpi,lam,spec):
    rr=[r for r in subset(data,spec) if eq(r['ee_lambda'],lam)]
    if not rr:return
    layers=sorted({int(r['layer_idx']) for r in rr})
    fig,ax=plt.subplots(figsize=(max(10,len(layers)*.42),5.0))
    for method in active_methods(rr):
        vals=[mean([r for r in rr if int(r['layer_idx'])==layer and r['method']==method],
                   'output_rel_error') for layer in layers]
        ax.plot(layers,vals,marker='o',ms=3,lw=1.7,color=COL[method],label=LABEL[method])
    ax.set_xticks(layers)
    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel('Mean real-Q relative Attention Output L2 error')
    ax.set_title('Paired arms by Layer | '+figure_title(spec,lam))
    ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8)
    fig.tight_layout();save(fig,out,'method_by_layer_'+route_slug(spec),dpi)


def plot_reject_vs_delta(data,out,dpi,lam,spec):
    relevant=[r for r in subset(data,spec) if eq(r['ee_lambda'],lam) and r['method'] in MIS_METHODS]
    if not relevant:return
    fig,ax=plt.subplots(figsize=(8.3,5.3))
    for method in active_methods(relevant,include_baselines=False):
        rr=[r for r in relevant if r['method']==method]
        xs=[num(r,'mis_rejected_fraction') for r in rr]
        ys=[num(r,'delta_rel_from_merge_all') for r in rr]
        ax.scatter(xs,ys,c=COL[method],alpha=.48,s=16,label=LABEL[method])
    ax.axhline(0,lw=.8,color='black')
    ax.set_xlabel('MIS Rejected / routed (method-specific)')
    ax.set_ylabel('Relative-Attention-L2 minus Merge All (negative better)')
    ax.set_title('K-only MIS priority comparison: rejection vs Attention error')
    ax.grid(alpha=.16);ax.legend(fontsize=7)
    fig.tight_layout();save(fig,out,'reject_fraction_vs_error_'+route_slug(spec),dpi)


def choose_examples(root,metric=None):
    # Small, representative routing-specific PCA schematics. Sweep coverage
    # itself is shown by the routing x E-E plots, not hundreds of example PNGs.
    by_mode=defaultdict(list)
    for path in sorted((root/'examples').glob('*.npz')):
        if metric is not None and ('_ee'+metric+'_') not in path.stem:continue
        if path.name.startswith('cosine_'):by_mode['cosine'].append(path)
        elif path.name.startswith('l2_'):by_mode['l2'].append(path)
    return [paths[0] for _,paths in sorted(by_mode.items()) if paths]



def plot_primary_priority_comparison(data,out,dpi,lam,spec):
    """Core paired ablation: Count Mean/Drop with three different MIS priorities."""
    data=subset(data,spec)
    data=[r for r in data if eq(r['ee_lambda'],lam)]
    main=('merge_all','k_only_mis_drop','k_only_degree_anchor_mis_drop',
          'k_only_anchor_first_mis_drop','k_only_importance_mis_drop')
    if ACTIVE_EE_METRIC=='cosine':main=('merge_all',*(m+'_cosee' for m in main[1:]))
    found=set(r['method'] for r in data)
    methods=[m for m in main if m in found]
    if methods[0:1]!=['merge_all'] or len(methods)<2:return
    budgets=sorted(set(num(r,'budget') for r in data))
    if not budgets:return
    fig,axs=plt.subplots(1,2,figsize=(13.0,5.0))
    xx=np.arange(len(budgets))
    for method in methods:
        rows_by_budget=[[r for r in data if r['method']==method and eq(r['budget'],b)] for b in budgets]
        errors=[mean(rows,'output_rel_error') for rows in rows_by_budget]
        deltas=[mean(rows,'delta_rel_from_merge_all') for rows in rows_by_budget]
        axs[0].plot(xx,errors,marker='o',lw=1.9,color=COL[method],label=LABEL[method])
        axs[1].plot(xx,deltas,marker='o',lw=1.9,color=COL[method],label=LABEL[method])
    for ax in axs:
        ax.set_xticks(xx,[f'{b:.0%}' for b in budgets]);ax.set_xlabel('Retained KV budget')
        ax.grid(alpha=.18)
        ax.legend(fontsize=7)
    axs[0].set_title('Future-Query Relative Attention L2')
    axs[0].set_ylabel('Error (lower better)')
    axs[1].axhline(0,color=COL['merge_all'],ls=':',lw=1.2)
    axs[1].set_ylabel('Delta vs Count-MergeAll (negative better)')
    axs[1].set_title('MIS priority ablation (same Count Mean fusion)')
    fig.suptitle(figure_title(spec,lam)+' | Degree / E-A priority ablation')
    fig.tight_layout();save(fig,out,'PRIMARY_degree_anchor_priority_'+route_slug(spec),dpi)



def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input_dir',required=True)
    p.add_argument('--ee_metric',choices=('both','l2','cosine'),default='both')
    p.add_argument('--font_family',default='Times New Roman')
    p.add_argument('--strict_times_new_roman',type=int,default=1)
    p.add_argument('--png_dpi',type=int,default=600)
    p.add_argument('--primary_ee_lambda',type=float,default=.5)
    p.add_argument('--focus_cosine_threshold',type=float,default=.85)
    p.add_argument('--focus_l2_lambda',type=float,default=.6)
    p.add_argument('--_stage',default=None,help=argparse.SUPPRESS)
    p.add_argument('--_route',default=None,help=argparse.SUPPRESS)
    p.add_argument('--_example',default=None,help=argparse.SUPPRESS)
    a=p.parse_args()
    root=Path(a.input_dir).resolve();out=root/'figures'
    global ACTIVE_EE_METRIC
    global SWEEP_BY_LAMBDA
    SWEEP_BY_LAMBDA={round(float(s['ee_lambda']),10):s for s in read(root/'mis_sweep_settings.csv')}
    configs=read(root/'cosine_mis_by_config.csv')
    if not configs:raise FileNotFoundError(root/'cosine_mis_by_config.csv')
    available={r.get('ee_metric','l2') for r in configs}
    if a.ee_metric=='both' and len(available)>1:
        for metric in sorted(available):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),
                '--input_dir',str(root),'--ee_metric',metric,
                '--font_family',a.font_family,'--strict_times_new_roman',str(a.strict_times_new_roman),
                '--png_dpi',str(a.png_dpi),'--primary_ee_lambda',str(a.primary_ee_lambda),
                '--focus_cosine_threshold',str(a.focus_cosine_threshold),
                '--focus_l2_lambda',str(a.focus_l2_lambda)],check=True)
        return
    metric=next(iter(available)) if a.ee_metric=='both' else a.ee_metric
    configs=[r for r in configs if r.get('ee_metric','l2')==metric]
    if not configs:raise ValueError('No replay configurations for E-E metric '+metric)
    ACTIVE_EE_METRIC=metric
    out=out/metric
    if not {'routing_mode','routing_parameter_type','routing_parameter'}.issubset(configs[0]):
        raise ValueError('This plotter requires dual-routing sweep CSV with routing columns')
    specs=route_specs(configs)
    lambdas=sorted(set(num(r,'ee_lambda') for r in configs))
    lam=min(lambdas,key=lambda x:abs(x-a.primary_ee_lambda))
    if a._stage is None:
        # Keep *all* numeric routing/EE sweep results, but render per-threshold
        # detailed plots only for one focus setting per routing mode. 2D
        # routing/EE heatmaps and routing sweeps visualize *all* settings.
        focus=[]
        for mode in sorted({x[0] for x in specs}):
            candidates=[x for x in specs if x[0]==mode]
            if mode=='cosine':
                selected=min(candidates,key=lambda x:abs(x[2]-a.focus_cosine_threshold))
            else:
                normed=[x for x in candidates if x[1]=='lambda']
                selected=min(normed,key=lambda x:abs(x[2]-a.focus_l2_lambda)) if normed else candidates[0]
            focus.append(selected)
        base=[sys.executable,str(Path(__file__).resolve()),'--input_dir',str(root),
              '--font_family',a.font_family,'--strict_times_new_roman',str(a.strict_times_new_roman),
              '--png_dpi',str(a.png_dpi),'--primary_ee_lambda',str(lam),
              '--focus_cosine_threshold',str(a.focus_cosine_threshold),
              '--focus_l2_lambda',str(a.focus_l2_lambda),'--ee_metric',metric]
        subprocess.run([*base,'--_stage','calibration'],check=True)
        for spec in focus:
            route=route_slug(spec)
            for stage in ('method','method_delta','lambda','coverage','conditional','probe','by_layer','reject_scatter','priority'):
                subprocess.run([*base,'--_stage',stage,'--_route',route],check=True)
        for mode,kind in sorted(set((x[0],x[1]) for x in specs)):
            route=f'{mode}_{kind}'
            for stage in ('routing','heatmap'):
                subprocess.run([*base,'--_stage',stage,'--_route',route],check=True)
        for path in choose_examples(root,metric):
            subprocess.run([*base,'--_stage','example','--_example',str(path)],check=True)
        print('[PLOTS DONE]',out,flush=True)
        return
    apply_font(a.font_family,bool(a.strict_times_new_roman))
    if a._stage=='calibration':plot_calibration_heatmap(root,out,a.png_dpi);return
    if a._stage=='example':
        if not a._example:raise ValueError('Missing example for example stage')
        path=Path(a._example).resolve()
        if path.parent!=(root/'examples').resolve() or path.suffix!='.npz':raise ValueError('Invalid example path')
        plot_examples(root,out,a.png_dpi,[path]);return
    if a._stage in ('routing','heatmap'):
        mode,kind=a._route.split('_',1)
        if a._stage=='routing':plot_routing_sweep(configs,out,a.png_dpi,lam,mode,kind)
        else:plot_routing_ee_heatmap(configs,out,a.png_dpi,mode,kind)
        return
    if not a._route:raise ValueError('Missing routing selection for substage')
    chosen=[sp for sp in specs if route_slug(sp)==a._route]
    if len(chosen)!=1:raise ValueError('Unknown routing setting '+a._route)
    spec=chosen[0]
    if a._stage=='priority':plot_primary_priority_comparison(configs,out,a.png_dpi,lam,spec)
    elif a._stage=='method':plot_method_comparison(configs,out,a.png_dpi,lam,spec)
    elif a._stage=='method_delta':plot_method_delta(configs,out,a.png_dpi,lam,spec)
    elif a._stage=='by_layer':plot_five_method_by_layer(configs,out,a.png_dpi,lam,spec)
    elif a._stage=='reject_scatter':plot_reject_vs_delta(configs,out,a.png_dpi,lam,spec)
    elif a._stage=='lambda':plot_lambda_sensitivity(configs,out,a.png_dpi,spec)
    elif a._stage=='coverage':plot_mis_coverage(configs,out,a.png_dpi,spec)
    elif a._stage=='conditional':plot_conditional_by_layer(configs,out,a.png_dpi,lam,spec)
    elif a._stage=='probe':
        all_averages=stream_probe_means(root/'cosine_mis_by_probe.csv',lam,metric)
        plot_per_probe(all_averages.get(spec,{}),out,a.png_dpi,lam,spec)
    else:raise ValueError('Unknown plotting stage '+str(a._stage))

if __name__=='__main__':main()
