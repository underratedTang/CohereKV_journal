#!/usr/bin/env python3
"""Experiment method registry; all importance D-first arms use Count*I priority.

Retained/Evicted selection stays count-independent. Optional pure-I MIS
fusion arms remain available as secondary ablations, without an I-only MergeAll.
"""
BASE_METHODS = ('full_cache', 'merge_all')
OPTIONAL_BASE_METHODS = ('importance_weighted_merge_all',)
FAMILIES = ('k_only', 'k_only_degree_anchor', 'k_only_anchor_first',
            'k_only_importance', 'k_only_importance_iwmean',
            'k_only_importance_imean')
FAMILY_METHODS = {
    'k_only': ('k_only_mis_drop', 'k_only_mis_drop_allcount'),
    'k_only_degree_anchor': ('k_only_degree_anchor_mis_drop',),
    'k_only_anchor_first': ('k_only_anchor_first_mis_drop',),
    'k_only_importance': ('k_only_importance_mis_drop',
                          'k_only_importance_mis_drop_allcount'),
    'k_only_importance_iwmean': ('k_only_importance_iwmean_drop',),
    'k_only_importance_imean': ('k_only_importance_imean_drop',
                                'k_only_importance_imean_drop_allcount'),
}
# Only the legacy Count/Importance and pure-I arms keep their sink variants.
SINK_FAMILIES = ('k_only', 'k_only_importance', 'k_only_importance_imean')
FAMILY_SINK_METHODS = {f: tuple(m+'_sink' for m in FAMILY_METHODS[f]) if f in SINK_FAMILIES else () for f in FAMILIES}
FAMILY_ALL_METHODS = {f: FAMILY_METHODS[f]+FAMILY_SINK_METHODS[f] for f in FAMILIES}
MIS_METHODS = tuple(m for f in FAMILIES for m in FAMILY_ALL_METHODS[f])
METHODS = BASE_METHODS + OPTIONAL_BASE_METHODS + MIS_METHODS
FUSION_METHODS = BASE_METHODS[1:] + OPTIONAL_BASE_METHODS + tuple(
    m for f in FAMILIES for m in FAMILY_METHODS[f])


def graph_is_importance_aware(family):
    if family not in FAMILIES: raise ValueError(family)
    return False


def priority_for_family(family):
    if family not in FAMILIES: raise ValueError(family)
    return ('degree' if family == 'k_only' else
            'degree_anchor_product' if family == 'k_only_degree_anchor' else
            'anchor_first' if family == 'k_only_anchor_first' else
            'importance_mass_delta')


def fusion_weight_mode(method):
    """Fusion weighting is independent of the Count*I D-first priority."""
    if method.startswith('k_only_importance_imean_'):
        return 'importance_only'
    if (method == 'importance_weighted_merge_all' or
        method.startswith('k_only_importance_iwmean_')):
        return 'count_importance'
    return 'count'


def fusion_is_importance_weighted(method):
    return fusion_weight_mode(method) != 'count'


def selected_families(csv):
    values = [v.strip() for v in str(csv).split(',') if v.strip()]
    if not values or len(set(values))!=len(values) or any(v not in FAMILIES for v in values):
        raise ValueError('Invalid --mis_families: '+str(csv)+'; allowed '+','.join(FAMILIES))
    return tuple(f for f in FAMILIES if f in values)


def selected_methods(families_csv, methods_csv=None):
    families=selected_families(families_csv)
    permitted=tuple(m for f in families for m in FAMILY_ALL_METHODS[f])
    if methods_csv is None:
        enabled=permitted
    else:
        enabled=tuple(m.strip() for m in str(methods_csv).split(',') if m.strip())
        if len(set(enabled))!=len(enabled) or any(m not in (*permitted,*OPTIONAL_BASE_METHODS) for m in enabled):
            raise ValueError('Invalid or duplicate --mis_methods: '+str(methods_csv))
        if not any(m in permitted for m in enabled):
            raise ValueError('Enable at least one MIS method')
    return (BASE_METHODS + tuple(m for m in OPTIONAL_BASE_METHODS if m in enabled)
            + tuple(m for m in permitted if m in enabled))


def family_for_method(method):
    if method in (*BASE_METHODS,*OPTIONAL_BASE_METHODS):return 'baseline'
    for family,names in FAMILY_ALL_METHODS.items():
        if method in names:return family
    raise ValueError('Unknown method '+str(method))


def is_sink_gated(method):
    if method not in METHODS:raise ValueError('Unknown method '+str(method))
    return method.endswith('_sink')


def original_method(method):
    if method not in METHODS:raise ValueError('Unknown method '+str(method))
    return method[:-5] if method.endswith('_sink') else method


def effective_method(method, sink_presence):
    return original_method(method) if (not is_sink_gated(method) or bool(sink_presence)) else 'merge_all'
