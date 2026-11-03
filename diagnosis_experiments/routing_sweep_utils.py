#!/usr/bin/env python3
"""Dependency-light routing sweep setup and statistics helpers."""
from collections import defaultdict
import numpy as np
from experiment_utils import parse_float_csv

def summarize(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[(r['routing_mode'],r['routing_parameter_type'],r['routing_parameter'],r['budget'],r['ee_lambda'],r['method'])].append(r)
    out=[]
    for key,group in sorted(grouped.items()):
        rec=dict(zip(('routing_mode','routing_parameter_type','routing_parameter','budget','ee_lambda','method'),key))
        rec['n_paired_configs']=len(group)
        for field in ('routing_l2_cutoff','output_rel_error','output_rel_p90','output_abs_error','output_rmse',
                      'delta_rel_from_merge_all','relative_improvement_pct','probe_win_fraction',
                      'n_routed','n_mis_accepted','n_mis_rejected','n_affected_groups',
                      'n_native_conflict_edges','reconstruction_sse_original_key',
                      'mean_anchor_drift','mean_routed_cosine'):
            rec[field]=float(np.mean([r[field] for r in group]))
        rec['n_changed_configs']=sum(r['n_mis_rejected']>0 for r in group)
        out.append(rec)
    return out


def make_routing_specs(a):
    """Independent routing grid. L2 lambdas use *train-only* Median S_ref.

    Spec=(mode, threshold_type, param). This is independent of ee_lambdas.
    """
    raw=[x.strip().lower() for x in a.routing_modes.split(',') if x.strip()]
    if len(raw)!=len(set(raw)) or any(x not in ('cosine','l2') for x in raw):
        raise ValueError('routing_modes must be unique members of cosine,l2')
    specs=[]
    if 'cosine' in raw:
        vals=([a.cosine_threshold] if a.cosine_threshold is not None
              else parse_float_csv(a.cosine_thresholds))
        if not vals or any(not -1 <= t <= 1 or not np.isfinite(t) for t in vals):
            raise ValueError('cosine routing thresholds must be finite in [-1,1]')
        specs.extend(('cosine','cosine',t) for t in sorted(set(vals)))
    if 'l2' in raw:
        lambdas=parse_float_csv(a.routing_l2_lambdas) if a.routing_l2_lambdas.strip() else []
        distances=parse_float_csv(a.routing_l2_distances) if a.routing_l2_distances.strip() else []
        if not (lambdas or distances) or any(t<=0 or not np.isfinite(t) for t in lambdas+distances):
            raise ValueError('L2 routing lambdas/distances must be finite and > 0')
        specs.extend(('l2','lambda',t) for t in sorted(set(lambdas)))
        specs.extend(('l2','distance',t) for t in sorted(set(distances)))
    return specs

def route_spec_cutoff(spec, frozen_scale):
    mode, kind, value = spec
    return value*frozen_scale if kind=='lambda' else value

def route_display(spec, cutoff):
    mode,kind,value=spec
    if mode=='cosine': return f'cos={value:g}'
    return f'L2 {kind}={value:g} (cutoff={cutoff:.4g})'

