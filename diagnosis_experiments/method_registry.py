#!/usr/bin/env python3
"""Registry for continuous CohereKV experiments and paired E-A ablations.

For EVERY original MIS arm (including sink-gated, AllCount, and alternative
fusion weights), the paired ``*_wo_ea_certificate`` arm changes only the E-A
routing gate. E-E MIS, priority, K/V/Count fusion and head gating are identical.
"""
EA_ABLATION_SUFFIX = '_wo_ea_certificate'
BASE_METHODS = ('full_cache', 'merge_all')
OPTIONAL_BASE_METHODS = ('importance_weighted_merge_all', 'coherekv_wo_mis')
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
SINK_FAMILIES = ('k_only', 'k_only_importance', 'k_only_importance_imean')
FAMILY_SINK_METHODS = {
    f: tuple(m + '_sink' for m in FAMILY_METHODS[f]) if f in SINK_FAMILIES else ()
    for f in FAMILIES
}
# Original methods, before applying the orthogonal E-A certificate ablation.
FAMILY_NATIVE_METHODS = {
    f: FAMILY_METHODS[f] + FAMILY_SINK_METHODS[f] for f in FAMILIES
}
FAMILY_WO_EA_METHODS = {
    f: tuple(m + EA_ABLATION_SUFFIX for m in FAMILY_NATIVE_METHODS[f])
    for f in FAMILIES
}
FAMILY_ALL_METHODS = {
    f: FAMILY_NATIVE_METHODS[f] + FAMILY_WO_EA_METHODS[f] for f in FAMILIES
}
MIS_METHODS = tuple(m for f in FAMILIES for m in FAMILY_ALL_METHODS[f])
METHODS = BASE_METHODS + OPTIONAL_BASE_METHODS + MIS_METHODS
# Only non-sink method names reach the fusion kernel (sink gating is resolved
# before compression). Each ablation must preserve the base fusion semantics.
FUSION_METHODS = BASE_METHODS[1:] + OPTIONAL_BASE_METHODS + tuple(
    m for f in FAMILIES for m in (
        FAMILY_METHODS[f] + tuple(x + EA_ABLATION_SUFFIX for x in FAMILY_METHODS[f])))


def without_ea_certificate(method):
    """Return the corresponding original name; does not alter the method family."""
    return (method[:-len(EA_ABLATION_SUFFIX)]
            if str(method).endswith(EA_ABLATION_SUFFIX) else method)


def is_wo_ea_certificate(method):
    if method not in METHODS:
        raise ValueError('Unknown method ' + str(method))
    return method.endswith(EA_ABLATION_SUFFIX)


def graph_is_importance_aware(family):
    if family not in FAMILIES:
        raise ValueError(family)
    return False


def priority_for_family(family):
    if family not in FAMILIES:
        raise ValueError(family)
    return ('degree' if family == 'k_only' else
            'degree_anchor_product' if family == 'k_only_degree_anchor' else
            'anchor_first' if family == 'k_only_anchor_first' else
            'importance_mass_delta')


def fusion_weight_mode(method):
    """Paired E-A ablation keeps precisely the original K/V fusion weights."""
    method = without_ea_certificate(method)
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
    if (not values or len(set(values)) != len(values) or
            any(v not in FAMILIES for v in values)):
        raise ValueError('Invalid --mis_families: ' + str(csv) +
                         '; allowed ' + ','.join(FAMILIES))
    return tuple(f for f in FAMILIES if f in values)


def selected_methods(families_csv, methods_csv=None):
    families = selected_families(families_csv)
    permitted = tuple(m for f in families for m in FAMILY_ALL_METHODS[f])
    if methods_csv is None:
        enabled = permitted
    else:
        enabled = tuple(m.strip() for m in str(methods_csv).split(',') if m.strip())
        if (len(set(enabled)) != len(enabled) or
                any(m not in (*permitted, *OPTIONAL_BASE_METHODS) for m in enabled)):
            raise ValueError('Invalid or duplicate --mis_methods: ' + str(methods_csv))
        if not enabled:
            raise ValueError('Enable at least one optional baseline or MIS method')
    return (BASE_METHODS +
            tuple(m for m in OPTIONAL_BASE_METHODS if m in enabled) +
            tuple(m for m in permitted if m in enabled))


def family_for_method(method):
    if method in (*BASE_METHODS, *OPTIONAL_BASE_METHODS):
        return 'baseline'
    for family, names in FAMILY_ALL_METHODS.items():
        if method in names:
            return family
    raise ValueError('Unknown method ' + str(method))


def is_sink_gated(method):
    if method not in METHODS:
        raise ValueError('Unknown method ' + str(method))
    return without_ea_certificate(method).endswith('_sink')


def original_method(method):
    """Remove sink gating only, preserving the E-A ablation if selected."""
    if method not in METHODS:
        raise ValueError('Unknown method ' + str(method))
    base = without_ea_certificate(method)
    if base.endswith('_sink'):
        base = base[:-5]
    return base + (EA_ABLATION_SUFFIX if is_wo_ea_certificate(method) else '')


def effective_method(method, sink_presence):
    return (original_method(method)
            if not is_sink_gated(method) or bool(sink_presence)
            else 'merge_all')
