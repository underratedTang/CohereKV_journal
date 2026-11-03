#!/usr/bin/env python3
"""Low-RSS publication figures for independently swept cosine/L2 Routing and E-E MIS.

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

COL={'merge_all':'#2563EB','mis_to_anchor':'#0D9488','all_to_anchor':'#C07843'}
LABEL={'merge_all':'Routed Merge All','mis_to_anchor':'MIS Rejected-to-Anchor','all_to_anchor':'All-to-Anchor'}


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
def figure_title(spec,lam):return f'{route_title(spec)} | independent E-E L2 cutoff = {lam:g} x S_calib'


def plot_method_comparison(data,out,dpi,lam,spec):
    data=subset(data,spec)
    budgets=sorted(set(num(r,'budget') for r in data))
    fig,axs=plt.subplots(1,len(budgets),figsize=(5.0*len(budgets),4.3),squeeze=False)
    for ax,b in zip(axs.flat,budgets):
        rr=[r for r in data if eq(r['ee_lambda'],lam) and eq(r['budget'],b)]
        ys=[mean([r for r in rr if r['method']==m],'output_rel_error') for m in LABEL]
        xx=np.arange(len(LABEL))
        for x,m,y in zip(xx,LABEL,ys):ax.bar(x,y,width=.66,color=COL[m],alpha=.9)
        ax.set_xticks(xx,[LABEL[m].replace(' ','\n',1) for m in LABEL],fontsize=8)
        ax.set_title(f'KV Budget {b:.0%}')
        ax.set_ylabel('Mean relative Attention Output Error (lower better)')
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15)
    fig.suptitle(figure_title(spec,lam),y=1.02)
    fig.tight_layout();save(fig,out,'three_method_error_'+route_slug(spec),dpi)


def plot_lambda_sensitivity(data,out,dpi,spec):
    data=subset(data,spec)
    budgets=sorted(set(num(r,'budget') for r in data))
    lambdas=sorted(set(num(r,'ee_lambda') for r in data))
    fig,axs=plt.subplots(1,len(budgets),figsize=(5*len(budgets),4.2),squeeze=False)
    for ax,b in zip(axs.flat,budgets):
        ax.axhline(0,color=COL['merge_all'],ls=':',lw=1.4,label='Merge All baseline')
        for m in ('mis_to_anchor','all_to_anchor'):
            yy=[mean([r for r in data if eq(r['ee_lambda'],lam) and eq(r['budget'],b) and r['method']==m],
                'relative_improvement_pct') for lam in lambdas]
            ax.plot(lambdas,yy,marker='o',lw=1.7,color=COL[m],label=LABEL[m])
        ax.set_xlabel('E-E MIS lambda (independent physical L2)')
        ax.set_title(f'KV Budget {b:.0%}')
        ax.set_ylabel('Relative improvement vs Merge All (%)')
        ax.grid(alpha=.17);ax.legend(fontsize=8)
    fig.suptitle(f'{route_title(spec)} | Constant R-E assignment / Value / Count',y=1.025)
    fig.tight_layout();save(fig,out,'improvement_vs_ee_'+route_slug(spec),dpi)


def plot_mis_coverage(data,out,dpi,spec):
    base=[r for r in subset(data,spec) if r['method']=='merge_all']
    lambdas=sorted(set(num(r,'ee_lambda') for r in base))
    budgets=sorted(set(num(r,'budget') for r in base))
    fig,axs=plt.subplots(1,2,figsize=(11,4.15))
    for b in budgets:
        rr=[r for r in base if eq(r['budget'],b)]
        coverage=[mean([r for r in rr if eq(r['ee_lambda'],lam)],'routed_fraction') for lam in lambdas]
        rejected=[mean([r for r in rr if eq(r['ee_lambda'],lam)],'mis_rejected_fraction') for lam in lambdas]
        axs[0].plot(lambdas,coverage,marker='o',label=f'Budget {b:.0%}')
        axs[1].plot(lambdas,rejected,marker='o',label=f'Budget {b:.0%}')
    axs[0].set_ylabel('Routed / Evicted (fraction)')
    axs[1].set_ylabel('MIS Rejected / Routed (fraction)')
    for ax in axs:
        ax.set_xlabel('E-E MIS lambda (independent L2)');ax.grid(alpha=.17);ax.legend(fontsize=9)
    fig.suptitle(f'{route_title(spec)} | frozen median-calibrated E-E conflicts')
    fig.tight_layout();save(fig,out,'routing_coverage_vs_ee_'+route_slug(spec),dpi)


def plot_conditional_by_layer(data,out,dpi,lam,spec):
    relevant=[r for r in subset(data,spec) if eq(r['ee_lambda'],lam) and num(r,'n_mis_rejected')>0]
    if not relevant:return
    layers=sorted(set(int(r['layer_idx']) for r in relevant))
    budgets=sorted(set(num(r,'budget') for r in relevant))
    fig,axs=plt.subplots(1,2,figsize=(11.2,4.3))
    xs=np.arange(len(layers));width=.75/max(1,len(budgets))
    for i,b in enumerate(budgets):
        for meth,ax in [('mis_to_anchor',axs[0]),('all_to_anchor',axs[1])]:
            y=[mean([r for r in relevant if eq(r['budget'],b) and int(r['layer_idx'])==l
                     and r['method']==meth],'relative_improvement_pct') for l in layers]
            ax.bar(xs+i*width-.375+width/2,y,width,label=f'Budget {b:.0%}')
    for meth,ax in [('mis_to_anchor',axs[0]),('all_to_anchor',axs[1])]:
        ax.axhline(0,color='black',lw=.7)
        ax.set_xticks(xs,[str(l) for l in layers]);ax.set_title(LABEL[meth]);ax.set_xlabel('Layer')
        ax.set_ylabel('Improvement where MIS rejects (%)')
        ax.grid(axis='y',alpha=.13);ax.legend(fontsize=8)
    fig.suptitle(f'{figure_title(spec,lam)} | conditioned on MIS rejection')
    fig.tight_layout();save(fig,out,'conditional_by_layer_'+route_slug(spec),dpi)


def plot_routing_sweep(data,out,dpi,lam,mode,kind):
    base=[r for r in data if r['routing_mode']==mode and r['routing_parameter_type']==kind and eq(r['ee_lambda'],lam)]
    xx=sorted(set(num(r,'routing_parameter') for r in base))
    budgets=sorted(set(num(r,'budget') for r in base))
    fig,axs=plt.subplots(2,2,figsize=(11.8,8.1))
    for b in budgets:
        rr=[r for r in base if eq(r['budget'],b)]
        xdata=[r for r in rr if r['method']=='merge_all']
        mdata=[r for r in rr if r['method']=='mis_to_anchor']
        label=f'Budget {b:.0%}'
        axs[0,0].plot(xx,[mean([r for r in xdata if eq(r['routing_parameter'],x)],'routed_fraction') for x in xx],marker='o',label=label)
        axs[0,1].plot(xx,[mean([r for r in xdata if eq(r['routing_parameter'],x)],'mis_rejected_fraction') for x in xx],marker='o',label=label)
        axs[1,0].plot(xx,[mean([r for r in mdata if eq(r['routing_parameter'],x)],'output_rel_error') for x in xx],marker='o',label=f'MIS / {label}')
        axs[1,0].plot(xx,[mean([r for r in xdata if eq(r['routing_parameter'],x)],'output_rel_error') for x in xx],marker='x',ls='--',label=f'Merge All / {label}')
        axs[1,1].plot(xx,[mean([r for r in mdata if eq(r['routing_parameter'],x)],'delta_rel_from_merge_all') for x in xx],marker='o',label=label)
    axs[0,0].set_title('R-E Routing coverage');axs[0,0].set_ylabel('Routed / Evicted')
    axs[0,1].set_title('E-E MIS rejection');axs[0,1].set_ylabel('Rejected / Routed')
    axs[1,0].set_title('Real-Q Attention Output Error');axs[1,0].set_ylabel('Mean relative error (lower better)')
    axs[1,1].set_title('Paired effect of MIS-to-Anchor');axs[1,1].set_ylabel('MIS minus Merge All (negative better)')
    axs[1,1].axhline(0,color='black',linewidth=.8,ls=':')
    xlabel=('R-E Cosine similarity threshold' if mode=='cosine' else
        ('R-E L2 cutoff / frozen Median S_calib' if kind=='lambda' else 'R-E physical L2 cutoff'))
    for ax in axs.flat:
        ax.set_xlabel(xlabel);ax.grid(alpha=.17);ax.legend(fontsize=7)
    fig.suptitle(f'{mode.upper()} Routing threshold sweep | E-E MIS uses independent L2 lambda={lam:g}',y=1.01)
    fig.tight_layout();save(fig,out,f'routing_sweep_{mode}_{kind}',dpi)


def plot_routing_ee_heatmap(data,out,dpi,mode,kind):
    relevant=[r for r in data if r['method']=='mis_to_anchor' and r['routing_mode']==mode and r['routing_parameter_type']==kind]
    xs=sorted(set(num(r,'ee_lambda') for r in relevant))
    ys=sorted(set(num(r,'routing_parameter') for r in relevant))
    grid=np.array([[mean([r for r in relevant if eq(r['ee_lambda'],ee) and eq(r['routing_parameter'],rr)],
                        'delta_rel_from_merge_all') for ee in xs] for rr in ys])
    fig,ax=plt.subplots(figsize=(max(6.,len(xs)*1.),max(4.6,len(ys)*.65)))
    vmax=max(1e-6,float(np.nanmax(np.abs(grid))))
    im=ax.imshow(grid,cmap='coolwarm',vmin=-vmax,vmax=vmax,aspect='auto',origin='lower')
    fig.colorbar(im,ax=ax,label='MIS-to-Anchor minus Merge All (negative better)')
    ax.set_xticks(np.arange(len(xs)),[f'{a:g}' for a in xs]);ax.set_yticks(np.arange(len(ys)),[f'{a:g}' for a in ys])
    ax.set_xlabel('E-E MIS L2 cutoff multiplier (independent)')
    ax.set_ylabel('Cosine threshold' if mode=='cosine' else ('R-E L2 / Median S_calib' if kind=='lambda' else 'R-E L2 cutoff'))
    ax.set_title(f'{mode.upper()} Routing / E-E MIS two-dimensional sweep | mean across budgets, layers, heads')
    fig.tight_layout();save(fig,out,f'routing_ee_heatmap_{mode}_{kind}',dpi)


def stream_probe_means(path,lam):
    if not path.is_file():return {}
    sums=defaultdict(float);counts=defaultdict(int)
    with path.open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if not eq(row['ee_lambda'],lam) or row['method'] not in ('mis_to_anchor','all_to_anchor'):continue
            key=(route_key(row),row['method'],int(row['future_token_offset']))
            sums[key]+=float(row['delta_rel_from_merge_all']);counts[key]+=1
    by_route=defaultdict(dict)
    for (spec,method,offset),value in sums.items():
        by_route[spec][(method,offset)]=value/counts[(spec,method,offset)]
    return dict(by_route)


def plot_per_probe(averages,out,dpi,lam,spec):
    if not averages:return
    offsets=sorted({p for method,p in averages})
    fig,ax=plt.subplots(figsize=(10.8,4.4))
    for method in ('mis_to_anchor','all_to_anchor'):
        yy=[averages.get((method,p),np.nan) for p in offsets]
        ax.plot(offsets,yy,label=LABEL[method],color=COL[method],lw=1.6)
    ax.axhline(0,color=COL['merge_all'],ls=':',lw=1.3)
    ax.set_xlabel('Real continuation token offset')
    ax.set_ylabel('Delta relative output error vs Merge All (negative better)')
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
        ax.scatter(xy[chosen,0],xy[chosen,1],c=COL['mis_to_anchor'],s=52,label='MIS accepted K (plus anchor)')
        if (~chosen).any():ax.scatter(xy[~chosen,0],xy[~chosen,1],c=COL['all_to_anchor'],s=85,marker='x',label='MIS rejected E')
        ax.scatter(xy[0,0],xy[0,1],c=COL['merge_all'],marker='*',s=225,zorder=5,label='Retained anchor')
        for j,(pt,name) in enumerate(zip(center,names)):
            ax.scatter([pt[0]],[pt[1]],c=COL[str(name)],s=91,marker=['D','P','s'][j],
                       edgecolor='white',linewidths=.5,label=f'Fused Key: {LABEL[str(name)]}',zorder=6)
        if len(x)<=15:
            for pt,idx in zip(xy,ids):ax.annotate(str(idx),pt,xytext=(3,3),textcoords='offset points',fontsize=7)
        routing_text=route_title(spec)
        if spec[0]=='l2':routing_text+=f' | actual L2 cutoff={cutoff:.3g}'
        ax.set_title(f'{routing_text}\nE-E conflict: physical L2 > {ee:g} x S_calib | A={ids[0]}\n'
                     f'Accepted={int(chosen.sum())-1}, rejected={int((~chosen).sum())}; PCA var={ratio:.1%}',fontsize=9)
        ax.set_xlabel('Group PCA axis 1 (Key units)');ax.set_ylabel('Group PCA axis 2 (Key units)')
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.15);ax.legend(fontsize=7)
        fig.text(.5,.01,'Schematic projection only; both Routing and MIS decisions use full-dimensional Keys.',ha='center',fontsize=8)
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


def choose_examples(root):
    # Small, representative routing-specific PCA schematics. Sweep coverage
    # itself is shown by the routing x E-E plots, not hundreds of example PNGs.
    by_mode=defaultdict(list)
    for path in sorted((root/'examples').glob('*.npz')):
        if path.name.startswith('cosine_'):by_mode['cosine'].append(path)
        elif path.name.startswith('l2_'):by_mode['l2'].append(path)
    return [paths[0] for _,paths in sorted(by_mode.items()) if paths]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input_dir',required=True)
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
    configs=read(root/'cosine_mis_by_config.csv')
    if not configs:raise FileNotFoundError(root/'cosine_mis_by_config.csv')
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
              '--focus_l2_lambda',str(a.focus_l2_lambda)]
        subprocess.run([*base,'--_stage','calibration'],check=True)
        for spec in focus:
            route=route_slug(spec)
            for stage in ('method','lambda','coverage','conditional','probe'):
                subprocess.run([*base,'--_stage',stage,'--_route',route],check=True)
        for mode,kind in sorted(set((x[0],x[1]) for x in specs)):
            route=f'{mode}_{kind}'
            for stage in ('routing','heatmap'):
                subprocess.run([*base,'--_stage',stage,'--_route',route],check=True)
        for path in choose_examples(root):
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
    if a._stage=='method':plot_method_comparison(configs,out,a.png_dpi,lam,spec)
    elif a._stage=='lambda':plot_lambda_sensitivity(configs,out,a.png_dpi,spec)
    elif a._stage=='coverage':plot_mis_coverage(configs,out,a.png_dpi,spec)
    elif a._stage=='conditional':plot_conditional_by_layer(configs,out,a.png_dpi,lam,spec)
    elif a._stage=='probe':
        all_averages=stream_probe_means(root/'cosine_mis_by_probe.csv',lam)
        plot_per_probe(all_averages.get(spec,{}),out,a.png_dpi,lam,spec)
    else:raise ValueError('Unknown plotting stage '+str(a._stage))

if __name__=='__main__':main()
