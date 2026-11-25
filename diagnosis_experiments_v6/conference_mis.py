"""Conference CohereKV equations (10)-(15): certified assignment and fixed-priority MIS.
Shared CountMean and infinity-order certificate propagation are in soft_mis.py.
"""
import numpy as np
import torch
from soft_mis import SoftPlan, _stable_hash

@torch.no_grad()
def conference_routes(keys, centers, radii, diameters, keep, evict, delta_ea, history_limit, *, chunk_size=256):
    # Select the closest compatible representative, tie by smallest slot index.
    dev=keys.device; n=len(evict)
    best=torch.zeros(n,device=dev,dtype=torch.long)
    routed=torch.zeros(n,device=dev,dtype=torch.bool)
    gamma=torch.zeros(n,device=dev,dtype=torch.float64)
    physical=torch.zeros_like(gamma)
    order=torch.argsort(keep); anchors=keep[order]
    mu=centers.double(); r=radii.double(); d=diameters.double()
    for st in range(0,n,chunk_size):
        en=min(n,st+chunk_size); ids=evict[st:en]
        dist=torch.cdist(mu[ids],mu[anchors])
        cert=torch.maximum(dist+r[ids,None]+r[anchors][None,:],torch.maximum(d[ids,None],d[anchors][None,:]))
        valid=cert<=delta_ea+1e-5
        vals,idx=dist.masked_fill(~valid,float('inf')).min(-1)
        best[st:en]=order[idx]; routed[st:en]=torch.isfinite(vals)
        gamma[st:en]=cert.gather(1,idx[:,None]).squeeze(1)
        physical[st:en]=dist.gather(1,idx[:,None]).squeeze(1)
    return best,routed,routed.clone(),gamma,physical

@torch.no_grad()
def conference_weights(centers,radii,diameters,keep,evict,best,routed,hard,gamma_ea,delta_ea,delta_ee,p,*,pair_chunk=256,**kwargs):
    weights=torch.zeros(len(evict),device=centers.device,dtype=torch.float64)
    edges=rounds=0
    for owner in best[routed].unique(sorted=True).tolist():
        loc=torch.nonzero(routed & (best==owner)).flatten(); ids=evict[loc]; n=len(ids)
        mu=centers[ids].double(); r=radii[ids].double(); d=diameters[ids].double()
        graph=np.empty((n,n),dtype=bool)
        for st in range(0,n,pair_chunk):
            en=min(n,st+pair_chunk)
            cross=torch.cdist(mu[st:en],mu)+r[st:en,None]+r[None,:]
            cert=torch.maximum(cross,torch.maximum(d[st:en,None],d[None,:]))
            graph[st:en]=(cert>delta_ee+1e-5).cpu().numpy()
        graph |= graph.T; np.fill_diagonal(graph,False)
        edges+=int(np.count_nonzero(np.triu(graph,1)))
        deg=graph.sum(1); gam=gamma_ea[loc].tolist(); phys=ids.tolist()
        order=sorted(range(n),key=lambda i:(int(deg[i]),gam[i],_stable_hash(phys[i]),phys[i]))
        rank=np.empty(n,dtype=int); rank[order]=np.arange(n)
        active=np.ones(n,dtype=bool); chosen=np.zeros(n,dtype=bool)
        while active.any():
            win=active & ~np.any(graph & active[None,:] & (rank[None,:]<rank[:,None]),axis=1)
            if not win.any():raise AssertionError('MIS did not progress')
            chosen |= win; active &= ~(win | graph[:,win].any(axis=1)); rounds+=1
        if np.any(graph[chosen][:,chosen]) or np.any(~chosen & ~graph[:,chosen].any(axis=1)):
            raise AssertionError('Not a maximal independent set')
        weights[loc]=torch.as_tensor(chosen,device=weights.device,dtype=weights.dtype)
    count=int((weights>0).sum().item())
    return SoftPlan(weights,hard,edges,edges,rounds,count,0,0,0.,0,0.,float(count))
