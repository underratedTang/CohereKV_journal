#!/usr/bin/env python3
"""Hard gate for Torch-vs-CUDA PPL parity at a fixed selector/configuration."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def load(path: str):
    obj=json.loads(Path(path).read_text())
    # evaluate_ppl_kvmerge writes a single result object for one dataset.
    if isinstance(obj,list):
        if len(obj)!=1: raise ValueError(f'{path}: expected one result, got {len(obj)}')
        obj=obj[0]
    return obj


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--attention_torch',required=True)
    ap.add_argument('--attention_cuda',required=True)
    ap.add_argument('--keepkv_torch',required=True)
    ap.add_argument('--keepkv_cuda',required=True)
    ap.add_argument('--ppl_abs_tol',type=float,default=5e-3)
    ap.add_argument('--output_json',required=True)
    args=ap.parse_args()
    pairs=[('attention',args.attention_torch,args.attention_cuda),('keepkv',args.keepkv_torch,args.keepkv_cuda)]
    rows=[]; ok=True
    for selector,pt,cu in pairs:
        a,b=load(pt),load(cu)
        pa=float(a['ppl']); pb=float(b['ppl']); diff=abs(pa-pb)
        same_selector=(a.get('mymethod_selector')==selector and b.get('mymethod_selector')==selector)
        row={'selector':selector,'torch_ppl':pa,'cuda_ppl':pb,'abs_diff':diff,'tol':args.ppl_abs_tol,
             'same_selector_metadata':same_selector,'pass':bool(same_selector and diff<=args.ppl_abs_tol)}
        rows.append(row); ok &= row['pass']
    out={'pass':ok,'rows':rows}
    Path(args.output_json).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
    if not ok: raise SystemExit(2)

if __name__=='__main__': main()
