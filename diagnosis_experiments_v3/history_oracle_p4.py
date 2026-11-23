#!/usr/bin/env python3
"""Sampled, read-only true-history p=4 E-A oracle for continuous CohereKV.

This module NEVER alters routing, E-E conflicts, MIS, attention K/V, or Count.
Original pre-merge Keys/positions are retained on CPU ONLY for explicitly
selected (layer, KV-head) pairs. Unsupported/empty-window cases are marked,
not interpreted as zero distance. Sampling is deterministic and stratified by
whether evicted states have already been merged (represented Count > 1).

A "window" means the subset of REPRESENTED original Keys whose logical token
position is >= current logical time - window. It is not a redefinition of
physical attention Count. Exponential weights are positive and normalized
inside the current original-key set, so time passing without new members does
not by itself reduce the normalized geometry.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class History:
    keys: np.ndarray       # [C, head_dim] original pre-merge Keys, float64 CPU
    positions: np.ndarray  # [C] original logical positions, int64 CPU

    def __post_init__(self):
        if self.keys.ndim != 2 or self.positions.shape != (self.keys.shape[0],):
            raise ValueError('History Keys and positions shape mismatch')
        if self.keys.shape[0] == 0:
            raise ValueError('An existing physical KV must have at least one original Key')


def singleton(key, position):
    return History(np.asarray(key, dtype=np.float64).reshape(1, -1).copy(),
                   np.asarray([position], dtype=np.int64))


def combine(histories):
    if not histories:
        raise ValueError('Missing source history')
    if len(histories) == 1:
        return histories[0]
    return History(np.concatenate([h.keys for h in histories], axis=0),
                   np.concatenate([h.positions for h in histories], axis=0))


def extend_histories(previous, new_keys, new_positions, expected_old):
    """Build pre-compression state list, keeping original raw (not mean) Keys."""
    if len(previous) != expected_old:
        raise AssertionError(f'Oracle previous slots {len(previous)} != cache previous slots {expected_old}')
    if len(new_keys) != len(new_positions):
        raise ValueError('New oracle Keys and positions mismatch')
    return list(previous) + [singleton(k, p) for k, p in zip(new_keys, new_positions)]


def merged_histories(source, keep, evict, best, accepted):
    """Exactly track actual physical fusion choices, not counterfactual choices."""
    keep = np.asarray(keep, dtype=np.int64)
    evict = np.asarray(evict, dtype=np.int64)
    best = np.asarray(best, dtype=np.int64)
    accepted = np.asarray(accepted, dtype=bool)
    owners = [[] for _ in keep]
    for physical, owner, yes in zip(evict, best, accepted):
        if yes:
            owners[int(owner)].append(source[int(physical)])
    return [combine([source[int(k)], *members]) for k, members in zip(keep, owners)]


def spaced_sample(ids, count):
    ids = np.asarray(ids, dtype=np.int64)
    if count <= 0 or len(ids) == 0:
        return ids[:0]
    if len(ids) <= count:
        return ids
    positions = (np.arange(count, dtype=np.float64) + .5) * len(ids) / count
    return ids[np.floor(positions).astype(np.int64)]


def stratified_evictions(evict, historical_counts, limit):
    evict = np.asarray(evict, dtype=np.int64)
    if len(evict) <= limit:
        return evict
    old = evict[np.asarray(historical_counts)[evict] > 1]
    new = evict[np.asarray(historical_counts)[evict] <= 1]
    n_old = min(len(old), (limit + 1) // 2)
    n_new = min(len(new), limit - n_old)
    n_old = min(len(old), limit - n_new)
    return np.sort(np.concatenate((spaced_sample(old, n_old), spaced_sample(new, n_new))))


def squared_distances(x, y):
    """Squared Euclidean distances, avoiding a [C1,C2,D] broadcast tensor."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return np.zeros((len(x), len(y)), dtype=np.float64)
    # Memory O(C1*C2), not O(C1*C2*D).
    squared = np.sum(x*x, axis=1)[:, None] + np.sum(y*y, axis=1)[None, :] - 2.0*(x@y.T)
    return np.maximum(squared, 0.0)


def _pair_moments(h1, h2, *, weights_1=None, weights_2=None, mask_1=None, mask_2=None):
    """Exact weighted cross T4 and union D4, with original-key intra-pairs.

    Uses only active subsets. For all-positive exponential weights, this
    agrees with the full represented history when tau -> infinity.
    Empty subsets => no cross comparison; union D4 only meaningful when both
    subsets exist, so both are returned NaN in that case.
    """
    x, y = h1.keys, h2.keys
    if mask_1 is not None:
        x = x[np.asarray(mask_1, dtype=bool)]
    if mask_2 is not None:
        y = y[np.asarray(mask_2, dtype=bool)]
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return math.nan, math.nan, n, m
    w1 = np.ones(n, dtype=np.float64) if weights_1 is None else np.asarray(weights_1, dtype=np.float64)
    w2 = np.ones(m, dtype=np.float64) if weights_2 is None else np.asarray(weights_2, dtype=np.float64)
    if w1.shape != (n,) or w2.shape != (m,) or np.any(w1 <= 0) or np.any(w2 <= 0):
        raise ValueError('Positive weights for active Keys required')
    cross_sq = squared_distances(x, y)
    cross_sum = float(np.einsum('i,ij,j->', w1, cross_sq**2, w2, optimize=True))
    cross_den = float(w1.sum() * w2.sum())
    cross_p4 = cross_sum / cross_den

    def within(z, w):
        if len(z) < 2:
            return 0.0, 0.0
        sq = squared_distances(z, z)
        upper = np.triu_indices(len(z), 1)
        pair_w = (w[:, None] * w[None, :])[upper]
        return float(np.dot((sq**2)[upper], pair_w)), float(pair_w.sum())

    left_sum, left_den = within(x, w1)
    right_sum, right_den = within(y, w2)
    union_den = left_den + right_den + cross_den
    union_p4 = (left_sum + right_sum + cross_sum) / union_den
    return max(cross_p4, 0.)**.25, max(union_p4, 0.)**.25, n, m


def exact_pair(h1, h2, *, now=None, window=None, tau=None):
    if window is not None and tau is not None:
        raise ValueError('Window and exponential decay are mutually exclusive')
    if window is not None:
        if now is None or window <= 0:
            raise ValueError('Window needs current position and positive width')
        return _pair_moments(h1, h2, mask_1=h1.positions >= now - window,
                             mask_2=h2.positions >= now - window)
    if tau is not None:
        if now is None or tau <= 0:
            raise ValueError('Decay needs current position and positive tau')
        # Subtract the common maximum logarithmic weight for numerical stability.
        all_pos = np.concatenate((h1.positions, h2.positions))
        center = int(all_pos.max())
        a = np.exp((h1.positions.astype(np.float64)-center) / tau)
        b = np.exp((h2.positions.astype(np.float64)-center) / tau)
        # For very old originals, exp can underflow; removing zero-effective
        # mass is the natural machine-precision limit, and is counted.
        ma = a > np.finfo(np.float64).tiny
        mb = b > np.finfo(np.float64).tiny
        return _pair_moments(h1, h2, weights_1=a[ma], weights_2=b[mb],
                             mask_1=ma, mask_2=mb)
    return _pair_moments(h1, h2)


def _pass(value, threshold):
    return bool(math.isfinite(value) and value <= threshold + 2e-9*max(1., threshold))


def examine_pair(h1, h2, *, now, delta_ea, delta_cert,
                 bound_minkowski, bound_interp, bound_fourth,
                 selected_bound, windows=(1024,2048,4096), taus=(1024.,4096.)):
    t4, d4, c1, c2 = exact_pair(h1,h2)
    out = dict(count_evicted=c1,count_anchor=c2,
               full_t4=t4,full_union_d4=d4,
               minkowski_bound=bound_minkowski,interp_bound=bound_interp,
               fourth_bound=bound_fourth,selected_bound=selected_bound,
               pass_selected_certificate=_pass(selected_bound,delta_ea),
               pass_full_cross=_pass(t4,delta_ea),
               pass_full_union=_pass(d4,delta_cert),
               # Exact cross misses that a pair may fit within union budget.
               full_union_only=(_pass(d4,delta_cert) and not _pass(t4,delta_ea)),
               certificate_false_reject=(_pass(t4,delta_ea) and not _pass(selected_bound,delta_ea)))
    for w in windows:
        t,d,n,m = exact_pair(h1,h2,now=now,window=int(w))
        out.update({f'window_{w}_evicted_count':n,f'window_{w}_anchor_count':m,
                    f'window_{w}_t4':t,f'window_{w}_union_d4':d,
                    f'window_{w}_pass_cross':_pass(t,delta_ea),
                    f'window_{w}_pass_union':_pass(d,delta_cert)})
    for tau in taus:
        num = int(tau)
        t,d,n,m = exact_pair(h1,h2,now=now,tau=float(tau))
        out.update({f'decay_{num}_evicted_count':n,f'decay_{num}_anchor_count':m,
                    f'decay_{num}_t4':t,f'decay_{num}_union_d4':d,
                    f'decay_{num}_pass_cross':_pass(t,delta_ea),
                    f'decay_{num}_pass_union':_pass(d,delta_cert)})
    return out


def summarize_pairs(pairs):
    """Simple count summary; never infer population-wide routing rates."""
    if not pairs:
        return {}
    ans={'sampled_candidate_pairs':len(pairs)}
    fields=['pass_selected_certificate','pass_selected_gate','pass_full_cross',
            'pass_full_cross_with_intra','pass_full_union',
            'full_union_only','certificate_false_reject']
    fields += [f'window_{w}_pass_cross' for w in (1024,2048,4096)]
    fields += [f'window_{w}_pass_union' for w in (1024,2048,4096)]
    fields += [f'decay_{w}_pass_cross' for w in (1024,4096)]
    fields += [f'decay_{w}_pass_union' for w in (1024,4096)]
    for field in fields:
        ans[field+'_count'] = sum(bool(row.get(field,False)) for row in pairs)
    for w in (1024,2048,4096):
        ans[f'window_{w}_both_nonempty'] = sum(int(row[f'window_{w}_evicted_count']>0
                            and row[f'window_{w}_anchor_count']>0) for row in pairs)
    return ans


def exact_cluster_moments(h):
    """True unweighted R4 and D4 of a represented original-Key group."""
    x=h.keys
    centered=x-x.mean(axis=0)
    r4=float(np.mean(np.sum(centered*centered,axis=1)**2))**.25
    if len(x)<2:
        return r4,0.0
    sq=squared_distances(x,x)
    tri=np.triu_indices(len(x),1)
    d4=float(np.mean(sq[tri]**2))**.25
    return r4,d4


def audit_head(histories, keep, evict, accepted_owner, routed, *,
               centers, radius_p, radius_2, radius_inf, diameter_p,
               delta_ea, delta_cert, now, round_index, layer, head,
               evictions_per_head=16, anchors_per_eviction=4, geometry_mode="moment_interp",
               physical_keys=None):
    """Read-only oracle sampled on one head's actual current cache states.

    Anchor sampling: nearest ORIGINAL-history geometric centers, plus the
    actually chosen routed anchor (if any). It is NOT an exhaustive search of
    all anchors, so these rows estimate *sampled pair* compatibility, not the
    full routing recovery rate. Sampling oversamples merged evicted states.
    """
    keep=np.asarray(keep,dtype=np.int64)
    evict=np.asarray(evict,dtype=np.int64)
    owner=np.asarray(accepted_owner,dtype=np.int64)
    routed=np.asarray(routed,dtype=bool)
    centers=np.asarray(centers,dtype=np.float64)
    radius_p=np.asarray(radius_p,dtype=np.float64)
    radius_2=np.asarray(radius_2,dtype=np.float64)
    radius_inf=np.asarray(radius_inf,dtype=np.float64)
    diameter_p=np.asarray(diameter_p,dtype=np.float64)
    if physical_keys is not None:
        physical_keys=np.asarray(physical_keys,dtype=np.float64)
        if physical_keys.shape != centers.shape:
            raise ValueError("Oracle physical Keys must match geometric centers shape")
    if geometry_mode not in ("moment_interp","fourth_moment","count_aware","count_aware_phys_075","count_aware_phys_100"):
        raise ValueError("Oracle only supports certified p4 certificate modes")
    if len(histories)!=len(centers):
        raise AssertionError('Oracle history length differs from geometric cache')
    if not (len(evict)==len(owner)==len(routed)):
        raise ValueError('Eviction owner arrays differ')
    lengths=np.fromiter((len(h.keys) for h in histories),count=len(histories),dtype=np.int64)
    sampled=stratified_evictions(evict,lengths,evictions_per_head)
    selected_position={int(ev):i for i,ev in enumerate(evict)}
    local_moments={}
    def true_stats(idx):
        idx=int(idx)
        if idx not in local_moments:
            local_moments[idx]=exact_cluster_moments(histories[idx])
        return local_moments[idx]
    out=[]
    for ev in sampled:
        e=int(ev)
        diff=centers[keep]-centers[e]
        dist=np.linalg.norm(diff,axis=1)
        near=np.argsort(dist,kind='stable')[:anchors_per_eviction].tolist()
        o=selected_position[e]
        if routed[o] and int(owner[o]) not in near:
            near.append(int(owner[o]))
        re, de=true_stats(e)
        for rank,ai in enumerate(near):
            a=int(keep[ai]);ra,da=true_stats(a)
            d=float(dist[ai]);mink=d+radius_p[e]+radius_p[a]
            t2=(radius_2[e]**2+d*d+radius_2[a]**2)**.5
            uinf=radius_inf[e]+d+radius_inf[a]
            interp=min(mink, (t2*uinf)**.5)
            upper4=(d**4+6*d*d*(radius_2[e]**2+radius_2[a]**2)
                    +radius_p[e]**4+radius_p[a]**4+6*radius_2[e]**2*radius_2[a]**2
                    +4*d*(radius_2[e]*radius_p[e]**2+radius_2[a]*radius_p[a]**2))**.25
            fourth=min(mink,upper4)
            # The caller's geometry mode specifies the selected certificate.
            row=dict(round=round_index,layer_idx=layer,kv_head_idx=head,
                     logical_position=now,evicted_physical_idx=e,anchor_physical_idx=a,
                     sampled_evicted_merged=int(lengths[e]>1),
                     candidate_rank=rank,actual_routed_anchor=int(routed[o] and owner[o]==ai),
                     center_l2=d,delta_ea=delta_ea,delta_cert=delta_cert,
                     true_evicted_r4=re,true_anchor_r4=ra,
                     true_evicted_d4=de,true_anchor_d4=da,
                     certificate_evicted_r4=float(radius_p[e]),
                     certificate_anchor_r4=float(radius_p[a]),
                     certificate_evicted_d4=float(diameter_p[e]),
                     certificate_anchor_d4=float(diameter_p[a]))
            row.update(examine_pair(histories[e],histories[a],now=now,
                        delta_ea=delta_ea,delta_cert=delta_cert,
                        bound_minkowski=mink,bound_interp=interp,
                        bound_fourth=fourth,selected_bound={"minkowski":mink,"moment_interp":interp,"fourth_moment":fourth,"count_aware":interp,"count_aware_phys_075":interp,"count_aware_phys_100":interp}[geometry_mode]))
            if geometry_mode.startswith('count_aware'):
                # Pairwise *certificate* gate, with represented-original Count;
                # the true two-group oracle is a distinct, weaker comparator.
                slack=((int(lengths[e])-1)/(2.*int(lengths[a]))
                       *max(0.,1.-(float(diameter_p[e])/delta_cert)**4))
                ca_pass=bool(float(diameter_p[e])<=delta_cert
                             and float(interp)/delta_cert <= (1.+slack)**.25)
                row['pass_selected_certificate']=ca_pass
                row['certificate_false_reject']=bool(row['pass_full_cross'] and not ca_pass)
                row['pass_countaware_pair']=ca_pass
                row['countaware_extra_p4_budget_ratio']=float(slack)
            else:
                row['pass_countaware_pair']=False
                row['countaware_extra_p4_budget_ratio']=''
            # Unweighted exact individual diameters also checked: a pair is a
            # valid exact cross gate only if both source groups are valid.
            row['pass_physical_gate']=True
            if geometry_mode in ('count_aware_phys_075','count_aware_phys_100'):
                if physical_keys is None:
                    raise ValueError('Physical Count-aware oracle requires physical Keys')
                physical_l2=float(np.linalg.norm(physical_keys[e]-physical_keys[a]))
                ratio=.75 if geometry_mode.endswith('_075') else 1.0
                row['physical_key_l2']=physical_l2
                row['physical_key_l2_over_delta']=physical_l2/delta_ea
                row['pass_physical_gate']=bool(physical_l2<=ratio*delta_ea+2e-9*max(1.,delta_cert))
            else:
                row['physical_key_l2']=''
                row['physical_key_l2_over_delta']=''
            row['pass_selected_gate']=bool(row['pass_selected_certificate']
                            and row['pass_physical_gate']
                            and _pass(diameter_p[e],delta_cert) and _pass(diameter_p[a],delta_cert))
            row['pass_full_cross_with_intra']=bool(row['pass_full_cross']
                                     and _pass(de,delta_cert) and _pass(da,delta_cert))
            row['pass_interp_with_intra']=bool(_pass(interp,delta_ea)
                            and _pass(diameter_p[e],delta_cert) and _pass(diameter_p[a],delta_cert))
            row['pass_fourth_with_intra']=bool(_pass(fourth,delta_ea)
                            and _pass(diameter_p[e],delta_cert) and _pass(diameter_p[a],delta_cert))
            out.append(row)
    return out
