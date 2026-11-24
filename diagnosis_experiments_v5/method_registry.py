#!/usr/bin/env python3
"""Main sweep methods. Baselines are reused across irrelevant grid axes."""
METHODS=('full_cache','eviction_only','merge_all','coherekv_conference',
         'coherekv_soft_mis','coherekv_soft_mis_importance')

def methods_for_p(p):
    return METHODS

def family_for_method(method):
    if method not in METHODS:raise ValueError('Unknown method '+str(method))
    if method=='coherekv_conference':return 'conference'
    return 'soft_mis' if method.startswith('coherekv_soft_mis') else 'baseline'

def priority_for_method(method):
    if family_for_method(method)=='baseline':return 'none'
    return 'importance_mass_delta' if 'importance' in method else 'degree'

def attention_for_method(method):
    if method=='full_cache':return 'none'
    return 'log_contribution' if family_for_method(method)=='soft_mis' else 'log_count'
