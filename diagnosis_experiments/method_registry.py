#!/usr/bin/env python3
"""MIS experiment registry. _cosee methods differ ONLY in E-E conflict metric."""
BASE_METHODS=('full_cache','merge_all')
OPTIONAL_BASE_METHODS=('importance_weighted_merge_all',)
FAMILIES=('k_only','k_only_degree_anchor','k_only_anchor_first',
          'k_only_importance','k_only_importance_iwmean','k_only_importance_imean')
_NATIVE={
 'k_only':('k_only_mis_drop','k_only_mis_drop_allcount'),
 'k_only_degree_anchor':('k_only_degree_anchor_mis_drop',),
 'k_only_anchor_first':('k_only_anchor_first_mis_drop',),
 'k_only_importance':('k_only_importance_mis_drop','k_only_importance_mis_drop_allcount'),
 'k_only_importance_iwmean':('k_only_importance_iwmean_drop',),
 'k_only_importance_imean':('k_only_importance_imean_drop','k_only_importance_imean_drop_allcount'),
}
SINK_FAMILIES=('k_only','k_only_importance','k_only_importance_imean')
FAMILY_METHODS={f:tuple(x for m in _NATIVE[f] for x in (m,m+'_cosee')) for f in FAMILIES}
FAMILY_SINK_METHODS={f:tuple(m+'_sink' for m in FAMILY_METHODS[f]) if f in SINK_FAMILIES else () for f in FAMILIES}
FAMILY_ALL_METHODS={f:FAMILY_METHODS[f]+FAMILY_SINK_METHODS[f] for f in FAMILIES}
MIS_METHODS=tuple(m for f in FAMILIES for m in FAMILY_ALL_METHODS[f])
METHODS=BASE_METHODS+OPTIONAL_BASE_METHODS+MIS_METHODS
FUSION_METHODS=BASE_METHODS[1:]+OPTIONAL_BASE_METHODS+tuple(m for f in FAMILIES for m in FAMILY_METHODS[f])

def is_cosine_ee_method(method):
 return method.endswith('_cosee') or method.endswith('_cosee_sink')

def original_method(method):
 if method not in METHODS:raise ValueError('Unknown method '+str(method))
 return method[:-5] if method.endswith('_sink') else method

def canonical_method(method):
 name=method[:-5] if method.endswith('_sink') else method
 return name[:-6] if name.endswith('_cosee') else name

def ee_metric_for_method(method):
 return 'cosine' if is_cosine_ee_method(method) else 'l2'

def graph_is_importance_aware(family):
 if family not in FAMILIES:raise ValueError(family)
 return False

def priority_for_family(family):
 if family not in FAMILIES:raise ValueError(family)
 return ('degree' if family=='k_only' else
         'degree_anchor_product' if family=='k_only_degree_anchor' else
         'anchor_first' if family=='k_only_anchor_first' else 'importance_mass_delta')

def fusion_weight_mode(method):
 m=canonical_method(method)
 if m.startswith('k_only_importance_imean_'):return 'importance_only'
 if m=='importance_weighted_merge_all' or m.startswith('k_only_importance_iwmean_'):
  return 'count_importance'
 return 'count'

def fusion_is_importance_weighted(method):return fusion_weight_mode(method)!='count'

def selected_families(csv):
 values=[v.strip() for v in str(csv).split(',') if v.strip()]
 if not values or len(set(values))!=len(values) or any(v not in FAMILIES for v in values):
  raise ValueError('Invalid --mis_families: '+str(csv))
 return tuple(f for f in FAMILIES if f in values)

def selected_methods(families_csv,methods_csv=None):
 families=selected_families(families_csv)
 permitted=tuple(m for f in families for m in FAMILY_ALL_METHODS[f])
 if methods_csv is None:enabled=permitted
 else:
  enabled=tuple(m.strip() for m in str(methods_csv).split(',') if m.strip())
  if len(set(enabled))!=len(enabled) or any(m not in (*permitted,*OPTIONAL_BASE_METHODS) for m in enabled):
   raise ValueError('Invalid/duplicate --mis_methods: '+str(methods_csv))
  if not any(m in permitted for m in enabled):raise ValueError('Enable at least one MIS method')
 return BASE_METHODS+tuple(m for m in OPTIONAL_BASE_METHODS if m in enabled)+tuple(m for m in permitted if m in enabled)

def family_for_method(method):
 if method in (*BASE_METHODS,*OPTIONAL_BASE_METHODS):return 'baseline'
 for f,names in FAMILY_ALL_METHODS.items():
  if method in names:return f
 raise ValueError('Unknown method '+str(method))

def is_sink_gated(method):
 if method not in METHODS:raise ValueError('Unknown method '+str(method))
 return method.endswith('_sink')

def effective_method(method,sink_presence):
 return original_method(method) if (not is_sink_gated(method) or bool(sink_presence)) else 'merge_all'
