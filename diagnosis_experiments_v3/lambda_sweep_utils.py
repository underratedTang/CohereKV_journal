#!/usr/bin/env python3
"""Strict paired E-A / E-E sweep, and lossless per-configuration summaries.

Pairs are explicit: 3 E-A thresholds x 3 EE/EA ratios is NINE cases,
not the cartesian product of a 3-element E-A and 9-element E-E array.
All comparisons are between runs with the SAME lambda pair and fixed budget.
"""
from __future__ import annotations
import math

NO_MIS = 'merge_all'
PHYSICAL_MIS = 'physical_gated_importance_mis_100'
COUNT_MATCHED = 'physical_gated_nearest_matched_100'
COMPARATORS = (NO_MIS, PHYSICAL_MIS, COUNT_MATCHED)


def parse_lambda_pairs(value):
    """Return an ordered unique tuple of (ea,ee), both finite and positive."""
    if not isinstance(value,str) or not value.strip():
        raise ValueError('lambda_pairs must be a nonempty comma-separated EA:EE list')
    parsed=[]
    for item in value.split(','):
        parts=item.strip().split(':')
        if len(parts)!=2 or not all(x.strip() for x in parts):
            raise ValueError('Invalid lambda pair (expected EA:EE): '+repr(item))
        try:
            ea,ee=map(float,parts)
        except ValueError as exc:
            raise ValueError('Non-numeric lambda pair: '+repr(item)) from exc
        if not (math.isfinite(ea) and math.isfinite(ee) and ea>0 and ee>0):
            raise ValueError('All EA/EE lambda values must be finite and positive')
        if (ea,ee) in parsed:
            raise ValueError('Duplicate lambda pair: '+repr(item))
        parsed.append((ea,ee))
    return tuple(parsed)


def _ratio(numerator,denominator):
    return numerator/denominator if denominator else float('nan')


def _num(row,key):
    return float(row.get(key,0) or 0)


def build_sweep_summary(rows):
    """One row per completed triple, never aggregate NLL from different pairs."""
    keyed={}
    for r in rows:
        key=(str(r['budget']),str(r['p']),str(r['geometry_mode']),
             float(r['lambda_ea']),float(r['lambda_ee']))
        by_method=keyed.setdefault(key,{})
        if r['method'] in by_method:
            raise ValueError('Duplicate method at same lambda pair: '+repr((key,r['method'])))
        by_method[r['method']]=r
    summary=[]
    for (budget,p,mode,ea,ee),methods in keyed.items():
        if any(name not in methods for name in COMPARATORS):
            continue
        no,mis,matched=[methods[k] for k in COMPARATORS]
        if int(_num(mis,'scored_tokens'))!=int(_num(matched,'scored_tokens')) or \
           int(_num(no,'scored_tokens'))!=int(_num(mis,'scored_tokens')):
            raise ValueError('Paired methods scored different token counts')
        # MIS vs matched accept quotas are identical ONLY conditional on each
        # run's CURRENT cache state, not guaranteed across divergent trajectories.
        row=dict(budget=budget,p=p,geometry_mode=mode,lambda_ea=ea,lambda_ee=ee,
                 ee_over_ea=ee/ea,scored_tokens=int(_num(mis,'scored_tokens')),
                 no_mis_nll=_num(no,'nll_per_token'),mis_nll=_num(mis,'nll_per_token'),
                 matched_nll=_num(matched,'nll_per_token'),
                 no_mis_ppl=_num(no,'ppl'),mis_ppl=_num(mis,'ppl'),matched_ppl=_num(matched,'ppl'),
                 mis_minus_no_mis_nll=_num(mis,'nll_per_token')-_num(no,'nll_per_token'),
                 mis_minus_matched_nll=_num(mis,'nll_per_token')-_num(matched,'nll_per_token'),
                 matched_minus_no_mis_nll=_num(matched,'nll_per_token')-_num(no,'nll_per_token'),
                 no_mis_routed=int(_num(no,'routed_slots')),
                 mis_routed=int(_num(mis,'routed_slots')),
                 matched_routed=int(_num(matched,'routed_slots')),
                 no_mis_accepted=int(_num(no,'accepted_slots')),
                 mis_accepted=int(_num(mis,'accepted_slots')),
                 matched_accepted=int(_num(matched,'accepted_slots')),
                 no_mis_ee_pairs=int(_num(no,'physical_ee_accepted_pairs')),
                 mis_ee_pairs=int(_num(mis,'physical_ee_accepted_pairs')),
                 matched_ee_pairs=int(_num(matched,'physical_ee_accepted_pairs')),
                 no_mis_ee_conflicts=int(_num(no,'physical_ee_accepted_conflicts')),
                 mis_ee_conflicts=int(_num(mis,'physical_ee_accepted_conflicts')),
                 matched_ee_conflicts=int(_num(matched,'physical_ee_accepted_conflicts')),
                 no_mis_ee_conflict_rate=_ratio(_num(no,'physical_ee_accepted_conflicts'),_num(no,'physical_ee_accepted_pairs')),
                 mis_ee_conflict_rate=_ratio(_num(mis,'physical_ee_accepted_conflicts'),_num(mis,'physical_ee_accepted_pairs')),
                 matched_ee_conflict_rate=_ratio(_num(matched,'physical_ee_accepted_conflicts'),_num(matched,'physical_ee_accepted_pairs')),
                 no_mis_repeated_accepted=int(_num(no,'accepted_previously_merged_slots')),
                 mis_repeated_accepted=int(_num(mis,'accepted_previously_merged_slots')),
                 matched_repeated_accepted=int(_num(matched,'accepted_previously_merged_slots')),
                 no_mis_seconds=_num(no,'seconds'),mis_seconds=_num(mis,'seconds'),
                 matched_seconds=_num(matched,'seconds'))
        summary.append(row)
    return sorted(summary,key=lambda r:(float(r['budget']),float(r['p']),r['geometry_mode'],
                                        r['lambda_ea'],r['ee_over_ea']))


def build_sweep_cycles(rows):
    """Paired cycle-level deltas (not average PPL), partial triples omitted."""
    keyed={}
    for r in rows:
        key=(str(r['budget']),str(r['p']),str(r['geometry_mode']),
             float(r['lambda_ea']),float(r['lambda_ee']),int(r['cycle']))
        by_method=keyed.setdefault(key,{})
        if r['method'] in by_method:raise ValueError('Duplicate scored cycle for method')
        by_method[r['method']]=r
    results=[]
    for (budget,p,mode,ea,ee,cycle),by_method in keyed.items():
        if not all(k in by_method for k in COMPARATORS):continue
        no,mis,matched=[by_method[k] for k in COMPARATORS]
        ns=[int(float(r['scored_tokens'])) for r in (no,mis,matched)]
        if len(set(ns))!=1:raise ValueError('Misaligned scored tokens in cycle')
        a,b,c=[float(r['cycle_nll_per_token']) for r in (no,mis,matched)]
        results.append(dict(budget=budget,p=p,geometry_mode=mode,lambda_ea=ea,lambda_ee=ee,
                            ee_over_ea=ee/ea,cycle=cycle,scored_tokens=ns[0],
                            no_mis_nll=a,mis_nll=b,matched_nll=c,
                            mis_minus_no_mis_nll=b-a,mis_minus_matched_nll=b-c))
    return sorted(results,key=lambda r:(float(r['budget']),float(r['p']),r['geometry_mode'],
                                         r['lambda_ea'],r['ee_over_ea'],r['cycle']))
