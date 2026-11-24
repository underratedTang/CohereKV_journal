#!/usr/bin/env python3
"""Selectable sweep arms, with hard-limit attention ablation deduplication."""
import math

METHODS=('full_cache','eviction_only','merge_all','coherekv_conference',
         'coherekv_soft_mis','coherekv_soft_mis_importance',
         'coherekv_soft_mis_log_count','coherekv_soft_mis_importance_log_count')

def select_methods(csv):
    selected=tuple(m.strip() for m in csv.split(',') if m.strip())
    if not selected or len(set(selected))!=len(selected) or any(m not in METHODS for m in selected):
        raise ValueError('Select at least one unique known method from '+','.join(METHODS))
    return tuple(m for m in METHODS if m in selected)

def methods_for_p(p,selected=METHODS):
    # Skip a hard log-Count arm only if its equivalent Contribution arm runs.
    return tuple(m for m in selected if not (math.isinf(float(p)) and
        m.endswith('_log_count') and m.removesuffix('_log_count') in selected))

def family_for_method(method):
    if method not in METHODS:raise ValueError('Unknown method '+str(method))
    if method=='coherekv_conference':return 'conference'
    return 'soft_mis' if method.startswith('coherekv_soft_mis') else 'baseline'

def priority_for_method(method):
    if family_for_method(method)=='baseline':return 'none'
    return 'importance_mass_delta' if 'importance' in method else 'degree'

def attention_for_method(method):
    if method=='full_cache':return 'none'
    return 'log_contribution' if family_for_method(method)=='soft_mis' and not method.endswith('_log_count') else 'log_count'
