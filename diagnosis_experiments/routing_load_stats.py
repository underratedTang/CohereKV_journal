#!/usr/bin/env python3
"""Routing collision/load diagnostics, independent of any method gate."""
from __future__ import annotations

def routing_coverage(routed_slots: int, evicted_slots: int) -> float:
    if evicted_slots <= 0: return 0.0
    if routed_slots < 0 or routed_slots > evicted_slots:
        raise ValueError('Invalid routed/evicted count')
    return float(routed_slots)/float(evicted_slots)


def concentration(best, routed, n_keep):
    """Unnormalized collision diagnostics, with no future information."""
    import numpy as np
    b = np.asarray(best, dtype=np.int64)
    r = np.asarray(routed, dtype=bool)
    if b.shape != r.shape or n_keep < 1:
        raise ValueError('Bad routing arrays')
    cnt = np.bincount(b[r], minlength=n_keep).astype(np.float64)
    total = float(cnt.sum())
    if total <= 0: return {'routing_group_hhi':0.0,'max_group_load_fraction':0.0,
                           'mean_routed_group_size':0.0,'n_routed_groups':0}
    frac=cnt/total
    return dict(routing_group_hhi=float(np.dot(frac,frac)),
        max_group_load_fraction=float(frac.max()),
        mean_routed_group_size=float(total/max(1,int(np.count_nonzero(cnt)))),
        n_routed_groups=int(np.count_nonzero(cnt)))
