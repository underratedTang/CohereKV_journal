#!/usr/bin/env python3
"""Certified historical p-moment geometry for continuous CohereKV diagnostics.

A physical state represents a set of original *Key* vectors.  Its attention
key can be importance-weighted, so the certificate carries the independent
arithmetic center of the ORIGINAL represented Key vectors.  Count includes
only represented K/V states, never rejected AllCount pseudo-multiplicity.

For finite p>=1, radius and diameter are SAFE upper bounds obtained by
Minkowski and exact intra/inter-group pair-count decomposition.  They need
not equal true p-moments.  For p=inf the same recurrence is the published
CohereKV max-radius/max-diameter certificate (including radius tightening).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch

GEOMETRY_MODES = ("minkowski", "exact_p2", "moment_interp", "fourth_moment",
                  "count_aware", "count_aware_phys_075", "count_aware_phys_100")
PHYSICAL_COUNT_AWARE_RATIOS = {"count_aware_phys_075": 0.75,
                               "count_aware_phys_100": 1.0}


def is_count_aware(mode):
    return mode == "count_aware" or mode in PHYSICAL_COUNT_AWARE_RATIOS


def physical_count_aware_ratio(mode):
    return PHYSICAL_COUNT_AWARE_RATIOS.get(mode)


def validate_geometry_mode(mode, p):
    if mode not in GEOMETRY_MODES:
        raise ValueError(f"Unknown geometry mode: {mode}")
    p=parse_p(p)
    if mode == "exact_p2" and p != 2.:
        raise ValueError("exact_p2 requires p=2")
    if (mode == "moment_interp" or is_count_aware(mode)) and p < 2.:
        raise ValueError("moment_interp currently supports 2<=p<=inf")
    if mode == "fourth_moment" and p != 4.:
        raise ValueError("fourth_moment requires p=4")
    return mode

def _l2_cdist(a, b):
    """Double precision distance, same pointwise kernel for route and E-E."""
    return torch.cdist(a.double(), b.double(), p=2,
                       compute_mode='donot_use_mm_for_euclid_dist')


def parse_p(value):
    v=float(value)
    if not (v>=1 and (math.isfinite(v) or v==math.inf)):
        raise ValueError('p must be a finite real >=1 or inf')
    return v


def _lp_mean(values, weights, p):
    """Weighted p mean, with weights allowing zeros. Torch float64 numerics."""
    values=torch.as_tensor(values,dtype=torch.float64)
    weights=torch.as_tensor(weights,device=values.device,dtype=torch.float64)
    if values.shape!=weights.shape: raise ValueError('weights mismatch')
    active=weights>0
    if not bool(active.any()):return values.new_zeros(())
    if math.isinf(p):return values[active].max()
    # Stable log-sum-exp prevents overflow for large user-configured p.
    positive=active & (values>0)
    if not bool(positive.any()):return values.new_zeros(())
    logtotal=torch.logsumexp(torch.log(weights[positive])+p*torch.log(values[positive]),dim=0)
    return torch.exp((logtotal-torch.log(weights[active].sum()))/p)



def fourth_cross_upper(dist, radius_4_u, radius_4_v, radius_2_u, radius_2_v):
    """Certified bound for E[||X-Y||_2^4]**(1/4), for independent
    uniform original historical Keys X in u and Y in v.

    X and Y are centered at their arithmetic original-key means. R2 is exact
    and R4 is a safe upper certificate. The bound follows by expanding
    ||(mu_u-mu_v)+(X-mu_u)-(Y-mu_v)||^4, then bounding the centered
    directional second/third and cross-covariance terms by Cauchy--Schwarz.
    No R_inf, pair history, covariance matrices or approximation of p is used.
    """
    d=torch.as_tensor(dist,dtype=torch.float64)
    a=torch.as_tensor(radius_4_u,device=d.device,dtype=torch.float64)
    b=torch.as_tensor(radius_4_v,device=d.device,dtype=torch.float64)
    ra=torch.as_tensor(radius_2_u,device=d.device,dtype=torch.float64)
    rb=torch.as_tensor(radius_2_v,device=d.device,dtype=torch.float64)
    if bool(((d<0)|(a<0)|(b<0)|(ra<0)|(rb<0)).any()):
        raise ValueError('Nonnegative distances and moment bounds required')
    moment=(d.pow(4) + 6*d.square()*(ra.square()+rb.square())
            +a.pow(4)+b.pow(4)+6*ra.square()*rb.square()
            +4*d*(ra*a.square()+rb*b.square()))
    return moment.clamp_min(0).pow(0.25)


def cross_moment_bound(dist, radius_p_u, radius_p_v, p, geometry_mode="minkowski", *,
                       radius_2_u=None, radius_2_v=None,
                       radius_inf_u=None, radius_inf_v=None):
    """Safe upper bound on cross-cluster L2-distance p-mean, elementwise.

    All inputs are broadcastable float64 tensors (or Python scalars). For the
    interpolated bound, T2 is EXACT from arithmetic original-key means and
    second central moments; Uinf is an upper bound on every cross-pair distance.
    z^p <= Uinf^(p-2) z^2 gives T_p <= T2^(2/p) Uinf^(1-2/p).
    The minimum with Minkowski is still a valid upper bound. At p=inf this is
    the conference R_inf + ||mu_u-mu_v|| + R_inf certificate. No stored originals.
    """
    p=parse_p(p)
    validate_geometry_mode(geometry_mode,p)
    dist=torch.as_tensor(dist,dtype=torch.float64)
    ru=torch.as_tensor(radius_p_u,device=dist.device,dtype=torch.float64)
    rv=torch.as_tensor(radius_p_v,device=dist.device,dtype=torch.float64)
    old=ru+dist+rv
    if geometry_mode=='minkowski':return old
    # Count-aware changes ONLY E-A admissibility; cross p-upper bound is identical
    # to moment_interp in E-E conflicts and in history-certificate propagation.
    if is_count_aware(geometry_mode):geometry_mode='moment_interp'
    if geometry_mode=='exact_p2':
        return (dist.square()+ru.square()+rv.square()).clamp_min(0).sqrt()
    if geometry_mode=='fourth_moment':
        if radius_2_u is None or radius_2_v is None:
            raise ValueError('fourth_moment needs exact historical R2')
        return torch.minimum(old,fourth_cross_upper(
            dist,ru,rv,radius_2_u,radius_2_v))
    if radius_inf_u is None or radius_inf_v is None:
        if math.isinf(p):
            radius_inf_u,radius_inf_v=ru,rv
        else:
            raise ValueError('moment_interp needs valid historical R_inf bounds')
    if radius_2_u is None or radius_2_v is None:
        if p==2.:
            radius_2_u,radius_2_v=ru,rv
        elif math.isinf(p):
            radius_2_u,radius_2_v=ru,rv  # unused at the conference limit
        else:
            raise ValueError('moment_interp needs exact historical R2')
    uinf=(torch.as_tensor(radius_inf_u,device=dist.device,dtype=torch.float64)
          +dist+torch.as_tensor(radius_inf_v,device=dist.device,dtype=torch.float64))
    if math.isinf(p):return uinf
    r2u=torch.as_tensor(radius_2_u,device=dist.device,dtype=torch.float64)
    r2v=torch.as_tensor(radius_2_v,device=dist.device,dtype=torch.float64)
    t2=(dist.square()+r2u.square()+r2v.square()).clamp_min(0).sqrt()
    if p==2.:return t2
    # log-domain interpolation prevents overflow with large finite p.
    tiny=torch.finfo(torch.float64).tiny
    interp=torch.exp((2./p)*torch.log(t2.clamp_min(tiny))+
                     (1.-2./p)*torch.log(uinf.clamp_min(tiny)))
    interp=torch.where((t2==0)|(uinf==0),torch.zeros_like(interp),interp)
    return torch.minimum(old,interp)




def count_aware_pair_allowed(cross, evicted_counts, anchor_counts,
                             evicted_diameters, delta, p):
    """Certified E-A pair gate using only EVicted group's unused pair budget.

    S_u = choose(C_u,2)*(delta**p-Dhat_u**p) >= 0.
    Allow C_a*C_u*(B_au**p-delta**p)_+ <= S_u.
    Thus every accepted u pays for its own E-A excess; ordinary geometric
    E-E MIS guarantees all accepted E-E cross terms <= delta**p. Independent
    of the number of simultaneously accepted nodes. No group-level search.

    Express in units of delta to avoid overflow from dimensionful p-powers.
    Valid for finite p; p=inf must use conference baseline unchanged.
    Inputs broadcast; arrays double. The caller separately requires old
    intra-group Dhat <= delta. No epsilon is added to the safety budget.
    """
    p=parse_p(p)
    if math.isinf(p):
        raise ValueError('Count-aware moment slack is defined only for finite p')
    if p<2.:raise ValueError('Count-aware E-A requires p>=2')
    if not (math.isfinite(float(delta)) and float(delta)>0):
        raise ValueError('Positive finite history delta required')
    b=torch.as_tensor(cross,dtype=torch.float64)
    cu=torch.as_tensor(evicted_counts,device=b.device,dtype=torch.float64)
    ca=torch.as_tensor(anchor_counts,device=b.device,dtype=torch.float64)
    du=torch.as_tensor(evicted_diameters,device=b.device,dtype=torch.float64)
    if bool(((cu<1)|(ca<1)|(du<0)|(b<0)).any()):
        raise ValueError('Invalid Count/diameter/distance')
    # <=delta baseline always qualifies; for higher B use the evicted group
    # statistical slack only. We do not borrow anchor slack or re-use u slack.
    old_valid=du<=float(delta)
    residual=(1.-(du/float(delta)).clamp_min(0).pow(p)).clamp_min(0)
    excess=(b/float(delta)).pow(p)-1.
    allowance=((cu-1.)/(2.*ca))*residual
    return (excess<=allowance)&old_valid


@torch.no_grad()
def route_diagnostics_countaware(centers,radii,radius_2,radius_inf,diameters,counts,
                                 keep,evict,delta,p,chunk_size=256):
    """Only Count-aware metrics, derived from the shared same-state gate audit.

    The complete audit integrates these metrics into route_diagnostics_moment
    to avoid an additional large E-A distance matrix in every compression.
    """
    all_stats=route_diagnostics_moment(centers,centers,radii,radius_2,radius_inf,
               diameters,keep,evict,delta,delta,p,chunk_size,counts=counts)
    return {k:v for k,v in all_stats.items() if k.startswith('ea_countaware_')
            or k=='ea_moment_merged_evicted_possible'}

@torch.no_grad()
def merge_cluster_moments(counts, means, radius_p, diameter_p,
                          radius_2, radius_inf, diameter_inf, p, block_size=128,
                          *, geometry_mode='moment_interp'):
    """Multi-round moment-interpolated p-certificate update.

    Returns C, mu, Rhat_p, Dhat_p, EXACT R2, Rhat_inf, Dhat_inf.
    `fourth_moment` additionally tightens R4 with a centered fourth-order
    shift inequality, and uses the fourth cross bound for D4 pair terms.
    Uses original-represented token counts (not synthetic AllCount values).
    In p=2 all p-statistics are EXACT; in p=inf the R/D update is the original
    conference certificate. Only O(block_size*n) temporary pair storage.
    """
    p=parse_p(p)
    validate_geometry_mode(geometry_mode,p)
    if geometry_mode not in ('moment_interp','fourth_moment','count_aware','count_aware_phys_075','count_aware_phys_100'):
        raise ValueError('merge_cluster_moments accepts moment_interp or fourth_moment')
    if p<2:raise ValueError('Moment interpolation requires p >= 2')
    if counts.ndim!=1 or means.ndim!=2 or counts.numel()!=means.shape[0] or not counts.numel():
        raise ValueError('Invalid cluster moment shapes')
    c=counts.double();mu=means.double()
    rp=radius_p.double();dp=diameter_p.double()
    r2=radius_2.double();ri=radius_inf.double();di=diameter_inf.double()
    if any(x.shape!=c.shape for x in (rp,dp,r2,ri,di)):
        raise ValueError('Cluster certificate shapes mismatch')
    if bool((c<1).any()) or bool((rp<0).any()) or bool((ri<0).any()):
        raise ValueError('Invalid historical cluster certificates')
    C=c.sum();mean=(mu*c[:,None]).sum(0)/C
    offsets=torch.linalg.vector_norm(mu-mean,dim=-1)
    var2=(c*(r2.square()+offsets.square())).sum()/C
    r2new=var2.clamp_min(0).sqrt()
    zero=c.new_zeros(())
    if len(c)==1:
        return C,mean,rp[0],dp[0],r2new,ri[0],di[0]
    # Exact p=2 needs no pair matrix: maintain a valid, possibly looser
    # auxiliary infinity certificate in O(n*d) rather than O(n^2).
    if p==2.:
        shifted=ri+torch.linalg.vector_norm(mu-mean,dim=-1)
        ri_new=shifted.max()
        di_new=torch.maximum(di.max(),2.*ri_new)
        ri_new=torch.minimum(ri_new,(C-1.)/C*di_new)
        dp_new=(2.*C/(C-1.)*var2).clamp_min(0).sqrt()
        return C,mean,r2new,dp_new,r2new,ri_new,di_new
    # p>2 maintains a conference-style max radius/diameter certificate.
    di_new=di.max()
    rp_shift=(rp+offsets)
    ri_shift=(ri+offsets).max()
    if p==2.:
        dpnew=(2.*C/(C-1.)*var2).clamp_min(0).sqrt()
        rpnew=r2new
    elif math.isinf(p):
        dpnew=di_new
        rpnew=zero
    else:
        rpnew=_lp_mean(rp_shift,c,p)
        if geometry_mode=='fourth_moment':
            # Exact central R2 plus certified old R4: fourth-order shift
            # inequality for each child, then count-weighted p-moment.
            shift4=(rp.pow(4)+offsets.pow(4)
                    +6*offsets.square()*r2.square()
                    +4*offsets*r2*rp.square())
            rpnew=torch.minimum(rpnew,((c*shift4).sum()/C).clamp_min(0).pow(.25))
        within=c*(c-1.)/2.
        nz=(within>0)&(dp>0)
        logmoment=(torch.logsumexp(torch.log(within[nz])+p*torch.log(dp[nz]),dim=0)
                   if bool(nz.any()) else c.new_tensor(-float('inf')))
    n=len(c);col=torch.arange(n,device=mu.device)[None,:]
    for start in range(0,n,block_size):
        end=min(start+int(block_size),n)
        dist=_l2_cdist(mu[start:end],mu)
        uinf=ri[start:end,None]+dist+ri[None,:]
        mask=(torch.arange(start,end,device=mu.device)[:,None]<col)
        if bool(mask.any()):
            di_new=torch.maximum(di_new,uinf[mask].max())
        if p==2. or math.isinf(p):continue
        cross=cross_moment_bound(dist,rp[start:end,None],rp[None,:],p,geometry_mode,
                                 radius_2_u=r2[start:end,None],radius_2_v=r2[None,:],
                                 radius_inf_u=ri[start:end,None],radius_inf_v=ri[None,:])
        valid=mask&(cross>0)
        if bool(valid.any()):
            weights=c[start:end,None]*c[None,:]
            term=torch.logsumexp(torch.log(weights[valid])+p*torch.log(cross[valid]),dim=0)
            logmoment=torch.logaddexp(logmoment,term)
    ri_new=torch.minimum(ri_shift,(C-1.)/C*di_new)
    if math.isinf(p):
        return C,mean,ri_new,di_new,r2new,ri_new,di_new
    if p!=2.:
        dpnew=torch.exp((logmoment-torch.log(C*(C-1.)/2.))/p)
        dpnew=torch.where(torch.isfinite(dpnew),dpnew,zero)
        rpnew=torch.minimum(rpnew,(C-1.)/C*dpnew)
    return C,mean,rpnew,dpnew,r2new,ri_new,di_new


@torch.no_grad()
def route_diagnostics_moment(keys,centers,radius_p,radius_2,radius_inf,diameters,
                             keep,evict,delta,history_limit,p,chunk_size=256,
                             *,counts=None):
    """Compare old p-Minkowski vs new moment bound on identical states/anchors."""
    keep=keep.long().reshape(-1);evict=evict.long().reshape(-1)
    tol=2e-9*max(1.,float(history_limit))
    k_valid=diameters[keep].double()<=history_limit+tol
    result=dict(ea_minkowski_possible=0,ea_moment_possible=0,
                ea_moment_rescues_minkowski=0,ea_physical_possible=0,
                ea_fourth_possible=0,ea_fourth_rescues_minkowski=0,
                ea_fourth_rescues_moment=0,ea_moment_only_not_fourth=0,
                ea_old_evicted_over_limit=int((diameters[evict].double()>history_limit+tol).sum().item()),
                ea_old_anchor_over_limit=int((~k_valid).sum().item()))
    if counts is not None:
        if len(counts)!=len(centers):raise ValueError('Geometry Count length mismatch')
        result.update(ea_countaware_possible=0,ea_countaware_rescues_moment=0,
                      ea_countaware_only_moment=0,ea_countaware_merged_evicted_possible=0,
                      ea_moment_merged_evicted_possible=0,
                      ea_countaware_rescues_previously_merged=0,
                      ea_countaware_over_delta_possible=0,
                      ea_countaware_eligible_evicted=0,
                      ea_countaware_eligible_merged_evicted=0)
        for suffix in ('075','100'):
            result.update({f'ea_countaware_phys_{suffix}_possible':0,
                           f'ea_countaware_phys_{suffix}_rescue_moment':0,
                           f'ea_countaware_phys_{suffix}_blocks_countaware':0,
                           f'ea_countaware_phys_{suffix}_merged_possible':0})
    for start in range(0,int(evict.numel()),int(chunk_size)):
        idx=evict[start:start+int(chunk_size)]
        d=_l2_cdist(centers[idx],centers[keep])
        mink=d+radius_p[idx,None].double()+radius_p[keep][None,:].double()
        moment=cross_moment_bound(d,radius_p[idx,None],radius_p[keep][None,:],p,
                                  'moment_interp',radius_2_u=radius_2[idx,None],
                                  radius_2_v=radius_2[keep][None,:],
                                  radius_inf_u=radius_inf[idx,None],
                                  radius_inf_v=radius_inf[keep][None,:])
        valid=(diameters[idx].double()[:,None]<=history_limit+tol)&k_valid[None,:]
        m=((mink<=delta+tol)&valid).any(-1)
        i=((moment<=delta+tol)&valid).any(-1)
        physical_d=_l2_cdist(keys[idx],keys[keep])
        physical=(physical_d<=delta+tol).any(-1)
        result['ea_minkowski_possible']+=int(m.sum().item())
        result['ea_moment_possible']+=int(i.sum().item())
        result['ea_moment_rescues_minkowski']+=int((i&~m).sum().item())
        if counts is not None:
            if math.isinf(p):
                new=(moment<=delta+tol)&valid
            else:
                new=(count_aware_pair_allowed(moment,counts[idx,None],counts[keep][None,:],
                      diameters[idx,None],delta,p)&valid)
            new_any=new.any(-1)
            merged=counts[idx]>1
            result['ea_countaware_possible']+=int(new_any.sum())
            result['ea_countaware_rescues_moment']+=int((new_any&~i).sum())
            result['ea_countaware_only_moment']+=int((i&~new_any).sum())
            result['ea_countaware_merged_evicted_possible']+=int((new_any&merged).sum())
            result['ea_moment_merged_evicted_possible']+=int((i&merged).sum())
            result['ea_countaware_rescues_previously_merged']+=int((new_any&~i&merged).sum())
            result['ea_countaware_over_delta_possible']+=int((new&(moment>delta+tol)).any(-1).sum())
            result['ea_countaware_eligible_evicted']+=int(idx.numel())
            result['ea_countaware_eligible_merged_evicted']+=int(merged.sum())
            for suffix,ratio in (('075',.75),('100',1.0)):
                protected=(new & (physical_d <= ratio*float(delta)+tol)).any(-1)
                if math.isinf(p):protected=new_any
                result[f'ea_countaware_phys_{suffix}_possible']+=int(protected.sum())
                result[f'ea_countaware_phys_{suffix}_rescue_moment']+=int((protected&~i).sum())
                result[f'ea_countaware_phys_{suffix}_blocks_countaware']+=int((new_any&~protected).sum())
                result[f'ea_countaware_phys_{suffix}_merged_possible']+=int((protected&merged).sum())
        if p==4.:
            fourth=cross_moment_bound(
                d,radius_p[idx,None],radius_p[keep][None,:],p,'fourth_moment',
                radius_2_u=radius_2[idx,None],radius_2_v=radius_2[keep][None,:])
            f=((fourth<=delta+tol)&valid).any(-1)
            result['ea_fourth_possible']+=int(f.sum().item())
            result['ea_fourth_rescues_minkowski']+=int((f&~m).sum().item())
            result['ea_fourth_rescues_moment']+=int((f&~i).sum().item())
            result['ea_moment_only_not_fourth']+=int((i&~f).sum().item())
        result['ea_physical_possible']+=int(physical.sum().item())
    return result

@dataclass
class Cluster:
    count: float
    mean: torch.Tensor
    radius: float
    diameter: float


def singleton(key):
    return Cluster(1.,key.detach().double(),0.,0.)


def cross_upper(a: Cluster,b: Cluster):
    return float(torch.linalg.vector_norm(a.mean-b.mean).item()) + a.radius + b.radius


def merge_clusters(clusters, p):
    """A valid bound on full historical R_p,D_p. Input list includes anchor."""
    p=parse_p(p)
    if not clusters:raise ValueError('Cannot merge empty cluster set')
    if len(clusters)==1:return clusters[0]
    counts=torch.tensor([u.count for u in clusters],dtype=torch.float64)
    if bool((counts<1).any()):raise ValueError('Missing represented history')
    C=float(counts.sum().item())
    mu=sum((u.mean*float(u.count) for u in clusters))/C
    internal_weights=[]; internal_distances=[]
    for u in clusters:
        internal_weights.append(u.count*(u.count-1)/2.)
        internal_distances.append(u.diameter)
    for i,u in enumerate(clusters):
        for v in clusters[i+1:]:
            internal_weights.append(u.count*v.count)
            internal_distances.append(cross_upper(u,v))
    diameter=float(_lp_mean(internal_distances,internal_weights,p).item())
    radial_terms=[u.radius+float(torch.linalg.vector_norm(u.mean-mu).item()) for u in clusters]
    radius=float(_lp_mean(radial_terms,[u.count for u in clusters],p).item())
    radius=min(radius,(C-1.)/C*diameter)
    return Cluster(C,mu,radius,diameter)


@torch.no_grad()
def certified_routes(centers,radii,diameters,keep,evict,delta,chunk_size=256,*,
                     history_limit=None,geometry_mode='minkowski',p=2.,
                     radius_2=None,radius_inf=None,counts=None,physical_keys=None):
    """Nearest history-compatible anchor. Both modes use float64 comparisons.

    minkowski: cross p moment <= R_u + ||mu_u-mu_a|| + R_a.
    exact_p2: exact cross RMS = sqrt(R2_u^2 + center_distance^2 + R2_a^2).
    Old groups whose diameter exceeds history_limit cannot join new groups.
    """
    validate_geometry_mode(geometry_mode,p)
    keep=keep.long().reshape(-1);evict=evict.long().reshape(-1)
    device=centers.device
    n=int(evict.numel())
    best=torch.zeros(n,dtype=torch.long,device=device)
    routed=torch.zeros(n,dtype=torch.bool,device=device)
    gamma=torch.zeros(n,dtype=torch.float64,device=device)
    if n==0:return best,routed,gamma
    if keep.numel()==0:raise ValueError('Need retained anchor')
    if history_limit is None:history_limit=delta
    tol=2e-9*max(1.,float(history_limit))
    k_ok=diameters[keep].double()<=history_limit+tol
    if is_count_aware(geometry_mode):
        if counts is None or len(counts)!=len(centers):
            raise ValueError('Count-aware routing requires represented-original geometric Counts')
        if not math.isclose(float(delta),float(history_limit),rel_tol=1e-12,abs_tol=0.):
            raise ValueError('Count-aware requires the SAME delta for E-A, E-E and p-compactness')
    physical_ratio=physical_count_aware_ratio(geometry_mode)
    # p=inf is exactly the conference baseline; finite-p physical protection
    # only removes some already-certified E-A edges, so it cannot weaken Dp.
    if physical_ratio is not None and not math.isinf(parse_p(p)):
        if physical_keys is None or physical_keys.shape != centers.shape:
            raise ValueError('Physical-gated Count-aware needs physical Keys [slots,dim]')
        physical_keep=physical_keys[keep].double()
    else:
        physical_keep=None
    kk=centers[keep].double()
    for s in range(0,n,int(chunk_size)):
        e=min(s+int(chunk_size),n)
        idx=evict[s:e]
        distances=_l2_cdist(centers[idx],kk)
        bound=cross_moment_bound(distances,radii[idx].double()[:,None],
                                 radii[keep].double()[None,:],p,geometry_mode,
                                 radius_2_u=(radius_2[idx].double()[:,None] if radius_2 is not None else None),
                                 radius_2_v=(radius_2[keep].double()[None,:] if radius_2 is not None else None),
                                 radius_inf_u=(radius_inf[idx].double()[:,None] if radius_inf is not None else None),
                                 radius_inf_v=(radius_inf[keep].double()[None,:] if radius_inf is not None else None))
        if is_count_aware(geometry_mode) and not math.isinf(parse_p(p)):
            gate=count_aware_pair_allowed(bound,counts[idx].double()[:,None],
                 counts[keep].double()[None,:],diameters[idx].double()[:,None],delta,p)
        else:
            gate=(bound<=delta+tol)
        if physical_keep is not None:
            # Same two physical KV states as CountMean will actually fuse.
            physical_dist=_l2_cdist(physical_keys[idx],physical_keep)
            gate=gate & (physical_dist <= physical_ratio*float(delta)+tol)
        compatible=gate&(diameters[idx].double()[:,None]<=history_limit+tol)&k_ok[None,:]
        # Preserve conference's nearest-current-center tie-breaking among valid anchors.
        measured=distances.masked_fill(~compatible,float('inf'))
        vals,owners=measured.min(-1)
        good=torch.isfinite(vals)
        best[s:e]=owners
        routed[s:e]=good
        gamma[s:e]=torch.where(good,bound.gather(1,owners[:,None]).squeeze(1),
                             torch.zeros_like(vals))
    return best,routed,gamma


@torch.no_grad()
def route_diagnostics_p2(keys,centers,radii,diameters,keep,evict,
                         delta,history_limit,chunk_size=256):
    """Same states, three route gates; does not change routing/selection.

    Counts are per evicted PHYSICAL slot (not original represented Count).
    A pass requires at least one compatible retained anchor. Both history
    gates check the same old-diameter validity, unlike the physical baseline.
    """
    keep=keep.long().reshape(-1);evict=evict.long().reshape(-1)
    stats=dict(ea_physical_possible=0,ea_minkowski_possible=0,
               ea_exact_p2_possible=0,ea_exact_rescues_minkowski=0,
               ea_physical_but_not_exact=0,ea_old_evicted_over_limit=0,
               ea_old_anchor_over_limit=0)
    tol=2e-9*max(1.,float(history_limit))
    valid_keep=diameters[keep].double()<=history_limit+tol
    stats['ea_old_anchor_over_limit']=int((~valid_keep).sum().item())
    stats['ea_old_evicted_over_limit']=int((diameters[evict].double()>history_limit+tol).sum().item())
    for s in range(0,len(evict),int(chunk_size)):
        idx=evict[s:s+int(chunk_size)]
        physical=_l2_cdist(keys[idx],keys[keep])
        d=_l2_cdist(centers[idx],centers[keep])
        rb=radii[idx].double()[:,None];ra=radii[keep].double()[None,:]
        permitted=(diameters[idx].double()[:,None]<=history_limit+tol)&valid_keep[None,:]
        phys=(physical<=delta+tol).any(dim=-1)
        mink=((d+rb+ra<=delta+tol)&permitted).any(dim=-1)
        exact=(((d.square()+rb.square()+ra.square()).sqrt()<=delta+tol)&permitted).any(dim=-1)
        stats['ea_physical_possible']+=int(phys.sum().item())
        stats['ea_minkowski_possible']+=int(mink.sum().item())
        stats['ea_exact_p2_possible']+=int(exact.sum().item())
        stats['ea_exact_rescues_minkowski']+=int((exact&~mink).sum().item())
        stats['ea_physical_but_not_exact']+=int((phys&~exact).sum().item())
    return stats


@torch.no_grad()
def merge_cluster_tensors(counts, means, radii, diameters, p, block_size=128, *, geometry_mode="minkowski"):
    """Same valid p certificates as merge_clusters; chunked quadratic work.

    Keeps [n,n] distance matrices out of GPU memory. Handles a 14K-source
    MergeAll group with <= block_size*n pair storage. Also avoids per-member
    GPU->CPU synchronizations in the normal repeated-compression path.
    """
    p=parse_p(p)
    validate_geometry_mode(geometry_mode,p)
    if counts.ndim!=1 or means.ndim!=2 or means.shape[0]!=counts.numel():
        raise ValueError('Expected one-dimensional count and 2D mean inputs')
    if counts.numel()<1:raise ValueError('No members')
    c=counts.double();mu=means.double();r=radii.double();d=diameters.double()
    C=c.sum();mu_new=(mu*c[:,None]).sum(0)/C
    if len(c)==1:return C,mu_new,r[0],d[0]
    if geometry_mode=='exact_p2':
        # Exact parallel-axis / variance decomposition over represented originals.
        # No R_u+distance conservatism, no O(m^2) pair matrix, no stored originals.
        sq=torch.linalg.vector_norm(mu-mu_new,dim=-1).square()
        var=((r.square()+sq)*c).sum()/C
        rad=var.clamp_min(0).sqrt()
        dia=(2*C/(C-1)*var).clamp_min(0).sqrt()
        return C,mu_new,rad,dia
    n=len(c);dev=mu.device
    radius_up=_lp_mean(r+torch.linalg.vector_norm(mu-mu_new,dim=-1),c,p)
    if math.isinf(p):
        diameter_up=d.max()
    else:
        within_weights=c*(c-1)/2
        valids=(within_weights>0)&(d>0)
        logmoment=(torch.logsumexp(torch.log(within_weights[valids])+p*torch.log(d[valids]),dim=0)
                   if bool(valids.any()) else torch.tensor(-float('inf'),device=dev,dtype=torch.float64))
    col=torch.arange(n,device=dev)[None,:]
    for start in range(0,n,int(block_size)):
        end=min(start+int(block_size),n)
        pair=torch.cdist(mu[start:end],mu,p=2)+r[start:end,None]+r[None,:]
        pair_weights=c[start:end,None]*c[None,:]
        valid=(torch.arange(start,end,device=dev)[:,None]<col)&(pair>0)
        if not bool(valid.any()):continue
        if math.isinf(p):
            diameter_up=torch.maximum(diameter_up,pair[valid].max())
        else:
            new_log=torch.logsumexp(torch.log(pair_weights[valid])+p*torch.log(pair[valid]),dim=0)
            logmoment=torch.logaddexp(logmoment,new_log)
    if not math.isinf(p):
        diameter_up=torch.exp((logmoment-torch.log(C*(C-1)/2))/p)
        diameter_up=torch.where(torch.isfinite(diameter_up),diameter_up,torch.zeros_like(diameter_up))
    radius_up=torch.minimum(radius_up,((C-1)/C)*diameter_up)
    return C,mu_new,radius_up,diameter_up


@torch.no_grad()
def physical_l2_routes(keys, keep, evict, delta, chunk_size=256):
    """Baseline MergeAll: nearest physical K under Euclidean threshold.

    No historical cluster centers, radii or diameter certificate participates.
    The p parameter is irrelevant for the L2 distance of two physical keys.
    Returns the same three tensors as certified_routes; the third contains
    the actual physical E-A L2 distance for routed candidates.
    """
    keep=keep.long().reshape(-1); evict=evict.long().reshape(-1)
    n=int(evict.numel());device=keys.device
    best=torch.zeros(n,dtype=torch.long,device=device)
    routed=torch.zeros(n,dtype=torch.bool,device=device)
    distances_out=torch.zeros(n,dtype=torch.float32,device=device)
    if not n:return best,routed,distances_out
    if not keep.numel():raise ValueError('MergeAll requires retained anchors')
    if not math.isfinite(float(delta)) or delta<=0:raise ValueError('Positive E-A cutoff required')
    retained=keys[keep].float()
    for start in range(0,n,int(chunk_size)):
        stop=min(n,start+int(chunk_size))
        d=torch.cdist(keys[evict[start:stop]].float(),retained,p=2)
        values,owners=d.min(dim=-1)
        good=values<=float(delta)
        best[start:stop]=owners
        routed[start:stop]=good
        distances_out[start:stop]=torch.where(good,values,torch.zeros_like(values))
    return best,routed,distances_out
