"""Unique experiment configurations: unused parameters are empty, never fake grid rows."""
from itertools import product
import math
from method_registry import select_methods, PRIORITY_MODES


def numbers(csv,lo,hi,*,strict_lo=False,strict_hi=False):
    values=list(dict.fromkeys(float(x.strip()) for x in str(csv).split(',') if x.strip()))
    if not values or any(not math.isfinite(x) or x<lo or x>hi or (strict_lo and x==lo) or (strict_hi and x==hi) for x in values):
        raise ValueError(f'Invalid numeric grid {csv}')
    return values


def choices(csv,allowed):
    vals=list(dict.fromkeys(x.strip() for x in csv.split(',') if x.strip()))
    if not vals or any(v not in allowed for v in vals):raise ValueError(f'Invalid choices: {csv}; allowed {allowed}')
    return vals


def build_configs(a):
    methods=select_methods(a.methods)
    budgets=numbers(a.budgets,0,1,strict_lo=True,strict_hi=True)
    alphas_ea=numbers(a.alpha_ea,0,1)
    alphas_ee=numbers(a.alpha_ee,0,1)
    eas=numbers(a.lambda_ea,0,float('inf'),strict_lo=True)
    ees=numbers(a.lambda_ee,0,float('inf'),strict_lo=True)
    refs=choices(a.reference_modes,('frozen_train','online_first'))
    priorities=choices(a.priority_modes,PRIORITY_MODES)
    settings=list(product(alphas_ea,alphas_ee,eas,ees,priorities))
    configs=[]
    def add(method,budget='',reference_mode='',alpha_ea='',alpha_ee='',ea='',ee='',priority=''):
        configs.append(dict(config_id=f'c{len(configs):05d}',method=method,budget=budget,
            reference_mode=reference_mode,alpha_ea=alpha_ea,alpha_ee=alpha_ee,lambda_ea=ea,lambda_ee=ee,priority_mode=priority))
    if 'full_cache' in methods:add('full_cache')
    for b in budgets:
        if 'eviction_only' in methods:add('eviction_only',b)
        for ref in refs:
            if 'merge_all' in methods:
                # Threshold-gated baseline from original code, no historical geometry or MIS.
                for ea in sorted(set(eas)):add('merge_all',b,ref,ea=ea)
            if 'coherekv_journal' in methods:
                for x,y,ea,ee,priority in settings:add('coherekv_journal',b,ref,x,y,ea,ee,priority)
    return configs


def main():
    import argparse,json
    from collections import Counter
    from method_registry import METHODS
    p=argparse.ArgumentParser(description='Print Cartesian sweep plan without model/GPU dependencies')
    p.add_argument('--methods',default=','.join(METHODS));p.add_argument('--budgets',default='0.1')
    p.add_argument('--alpha_ea',default='0');p.add_argument('--alpha_ee',default='1');p.add_argument('--lambda_ea',default='0.5');p.add_argument('--lambda_ee',default='0.5')
    p.add_argument('--reference_modes',default='frozen_train')
    p.add_argument('--priority_modes',default=','.join(PRIORITY_MODES))
    a=p.parse_args();configs=build_configs(a);print(json.dumps(dict(total=len(configs),counts=Counter(c['method'] for c in configs),configurations=configs),indent=2))

if __name__=='__main__':main()
