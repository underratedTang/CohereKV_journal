#!/usr/bin/env python3
from __future__ import annotations
import argparse,glob,json,statistics
from pathlib import Path


def med(xs): return statistics.median(xs) if xs else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--result_dir',required=True); ap.add_argument('--output_json',required=True); ap.add_argument('--output_md',required=True); args=ap.parse_args()
    root=Path(args.result_dir); bench=[]; validations=[]; ppl=[]; parity_check=None
    for f in sorted(root.glob('engineering_benchmark_*.json')):
        try: bench.extend(json.loads(f.read_text()))
        except Exception: pass
    for f in sorted(root.glob('backend_validation_*.json')):
        try: validations.append(json.loads(f.read_text()))
        except Exception: pass
    for f in sorted(root.glob('ppl_parity_*.json')):
        try: ppl.append((f.name,json.loads(f.read_text())))
        except Exception: pass
    pc=root/'ppl_backend_parity_check.json'
    if pc.exists():
        try: parity_check=json.loads(pc.read_text())
        except Exception: parity_check=None
    route_speed=[]; graph_speed=[]; plan_speed=[]; mem_reduction=[]
    rows=[]
    for r in bench:
        rt=r['route_torch']['median_ms']; rc=r['route_cuda']['median_ms']; gt=r['graph_torch']['median_ms']; gc=r['graph_cuda']['median_ms']
        pt=r['plan_route_torch_graph_torch']['median_ms']; pc=r['plan_route_cuda_graph_cuda']['median_ms']; ph=r['plan_route_torch_graph_cuda']['median_ms']
        route_speed.append(rt/rc if rc else 0); graph_speed.append(gt/gc if gc else 0); plan_speed.append(pt/ph if ph else 0)
        mt=r['plan_route_torch_graph_torch'].get('peak_extra_mib'); mc=r['plan_route_torch_graph_cuda'].get('peak_extra_mib')
        if mt is not None and mc is not None and mc>0: mem_reduction.append(mt/mc)
        rows.append({'length':r['metadata']['length'],'layer':r['metadata']['layer_idx'],'lambda':r['metadata']['lambda'],
                     'route_speedup':rt/rc if rc else None,'graph_speedup':gt/gc if gc else None,'plan_speedup':pt/ph if ph else None,
                     'torch_plan_ms':pt,'cuda_plan_ms':pc,'hybrid_plan_ms':ph,'cuda_max_pool':r['cuda_diagnostics']['max_pool']})
    parity=[]
    for name,x in ppl:
        parity.append({'file':name,'selector':x.get('mymethod_selector'),'ppl':x.get('ppl'),'compress_time_s':x.get('total_compress_time_s'),
                       'route_backend':x.get('mymethod_route_backend'),'graph_backend':x.get('mymethod_graph_backend')})
    def choose(v):
        m=med(v)
        if m is None: return 'unknown'
        return 'cuda' if m>=1.05 else 'torch'
    out={'aggregate':{'median_route_speedup':med(route_speed),'median_graph_speedup':med(graph_speed),'median_full_plan_speedup':med(plan_speed),
                      'median_peak_memory_reduction':med(mem_reduction),'recommended_route_backend':choose(route_speed),'recommended_graph_backend':choose(graph_speed)},
         'benchmarks':rows,'validations':validations,'ppl_parity':parity,'ppl_backend_parity_check':parity_check}
    Path(args.output_json).write_text(json.dumps(out,indent=2))
    a=out['aggregate']; lines=['# mymethod engineering summary','',
      f"- median route speedup (torch/cuda): {a['median_route_speedup']}",f"- median graph speedup (torch/cuda): {a['median_graph_speedup']}",
      f"- median validated hybrid-plan speedup (torch route + cuda graph): {a['median_full_plan_speedup']}",f"- median peak-memory reduction: {a['median_peak_memory_reduction']}",
      f"- recommended route backend from this run: **{a['recommended_route_backend']}**",f"- recommended graph backend from this run: **{a['recommended_graph_backend']}**",'',
      'The recommendation is mechanical: CUDA must be at least 5% faster in the median. Correctness/certificate validation remains a hard gate.','',
      f"- PPL backend parity hard gate: {None if parity_check is None else parity_check.get('pass')}",'',
      '| L | layer | lambda | route x | graph x | hybrid plan x | torch plan ms | hybrid plan ms | cuda/cuda ms | max pool |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['length']} | {r['layer']} | {r['lambda']} | {r['route_speedup']:.3f} | {r['graph_speedup']:.3f} | {r['plan_speedup']:.3f} | {r['torch_plan_ms']:.3f} | {r['hybrid_plan_ms']:.3f} | {r['cuda_plan_ms']:.3f} | {r['cuda_max_pool']} |")
    if parity:
        lines += ['','## PPL backend parity','','| file | selector | route | graph | PPL | compression s |','|---|---|---|---|---:|---:|']
        for r in parity: lines.append(f"| {r['file']} | {r['selector']} | {r['route_backend']} | {r['graph_backend']} | {r['ppl']} | {r['compress_time_s']} |")
    Path(args.output_md).write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__': main()
