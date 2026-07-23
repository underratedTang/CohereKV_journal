"""Official-mechanism KeepKV cache update for llama models.

This module is designed to work with ``keepkv_runtime_patch_llama.py``.  The
runtime patch captures query-head-reduced ``exp(raw_attention_logits)`` rows
before softmax and applies ``log(cluster_count)`` during decoding.  This module
maintains the remaining official KeepKV state:

* fixed cache budget after the first compression;
* attention sink + heavy hitter + recent retention;
* recent-query score history with distance weighting;
* recent-token exposure correction;
* Electoral Votes / cluster counts;
* count- and score-weighted ZIP K/V merging;
* synchronous merging of recent score-history columns.

The public ``Repair_keepkv`` / ``KeepKV`` signatures remain compatible with the
existing task_evaluation harness. ``ratio`` is the evicted fraction used only to
initialize a fixed budget.  ``kv_budget`` may be supplied to request an absolute
physical-slot budget.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple
import torch
import torch.nn.functional as F

_EPS = 1e-8


def _as_legacy_tuple(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, tuple):
        return past_key_values
    if isinstance(past_key_values, list):
        return tuple(past_key_values)
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def _num_kv_groups(attn_mod, h_kv: int) -> int:
    groups = getattr(attn_mod, "num_key_value_groups", None)
    if groups is not None:
        return int(groups)
    h_q = getattr(attn_mod, "num_heads", getattr(attn_mod, "num_attention_heads", None))
    if h_q is None or int(h_q) % int(h_kv) != 0:
        raise ValueError(f"Cannot infer GQA groups: Hq={h_q}, Hkv={h_kv}")
    return int(h_q) // int(h_kv)


def _make_log_bias_q(counts: torch.Tensor, groups: int) -> torch.Tensor:
    x = torch.log(counts.float().clamp_min(1.0))
    if groups != 1:
        x = torch.repeat_interleave(x, repeats=groups, dim=1)
    return x.unsqueeze(-2).contiguous()


def reset_keepkv_state(model) -> None:
    for name in (
        "_keepkv_counts", "_keepkv_recent_scores", "_keepkv_cache_budgets",
        "_keepkv_heavy_budgets", "_keepkv_recent_budgets", "_keepkv_log_bias_q",
    ):
        setattr(model, name, None)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            attn = layer.self_attn
            attn._keepkv_enabled = False
            attn._keepkv_log_bias_q = None
            attn._keepkv_recent_score_rows = None


def _state_item(model, name: str, layer_idx: int):
    xs = getattr(model, name, None)
    return None if xs is None or layer_idx >= len(xs) else xs[layer_idx]


def _align_counts(model, layer_idx: int, K: torch.Tensor) -> torch.Tensor:
    B, H, L, _ = K.shape
    old = _state_item(model, "_keepkv_counts", layer_idx)
    if old is None or old.shape[:2] != (B, H):
        return torch.ones((B, H, L), device=K.device, dtype=torch.float32)
    old = old.to(K.device, torch.float32)
    if old.shape[-1] < L:
        old = torch.cat([old, torch.ones((B, H, L-old.shape[-1]), device=K.device)], -1)
    return old[..., :L].contiguous()


def _runtime_rows(attn_mod, B: int, H: int, L: int, fallback_attention=None) -> torch.Tensor:
    """Return [B,H,R,L] recent exp(raw-logit) rows.

    The runtime patch is authoritative.  A softmax-attention fallback is kept
    only to make failures explicit and preserve compatibility with older runs.
    """
    rows = getattr(attn_mod, "_keepkv_recent_score_rows", None)
    if rows is not None:
        rows = rows.to(dtype=torch.float32)
        if rows.shape[0] == B and rows.shape[1] == H:
            if rows.shape[-1] < L:
                rows = F.pad(rows, (0, L-rows.shape[-1]), value=0.0)
            return rows[..., :L].contiguous()
    if fallback_attention is None:
        raise RuntimeError("KeepKV runtime score rows are missing; install keepkv_runtime_patch before evaluation")
    a = fallback_attention
    if a.dim() == 3:
        return a.float().unsqueeze(-2)[..., :L]
    if a.dim() != 4:
        raise ValueError(f"Unexpected attention shape {tuple(a.shape)}")
    hq = a.shape[1]
    if hq != H:
        if hq % H != 0:
            raise ValueError(f"Hq={hq} not divisible by Hkv={H}")
        a = a.view(B, H, hq // H, a.shape[2], a.shape[3]).mean(2)
    return a.float()[..., :L]


def _append_history(old: Optional[torch.Tensor], new: torch.Tensor, recent_budget: int, L: int) -> torch.Tensor:
    if old is not None:
        old = old.to(new.device, torch.float32)
        if old.shape[-1] < L:
            old = F.pad(old, (0, L-old.shape[-1]), value=0.0)
        old = old[..., :L]
        new = torch.cat([old, new], dim=-2)
    return new[..., -max(1, recent_budget):, :].contiguous()


def _choose_budgets(model, layer_idx: int, L: int, ratio: float, kv_budget: Optional[int], heavy_ratio: Optional[float], recent_ratio: Optional[float]):
    old_total = _state_item(model, "_keepkv_cache_budgets", layer_idx)
    old_heavy = _state_item(model, "_keepkv_heavy_budgets", layer_idx)
    old_recent = _state_item(model, "_keepkv_recent_budgets", layer_idx)
    if old_total is not None:
        return min(int(old_total), L), int(old_heavy), int(old_recent)
    total = int(kv_budget) if kv_budget is not None else max(1, L - int(L * float(ratio)))
    total = min(total, L)
    if heavy_ratio is not None or recent_ratio is not None:
        hr = 0.1 if heavy_ratio is None else float(heavy_ratio)
        rr = 0.1 if recent_ratio is None else float(recent_ratio)
        denom = max(hr + rr, _EPS)
        recent = int(round(total * rr / denom))
    else:
        recent = total // 2
    recent = min(max(recent, 1 if total > 1 else 0), total)
    heavy = total - recent
    return total, heavy, recent


def _select_keep(scores: torch.Tensor, total: int, heavy_budget: int, recent_budget: int, sink_len: int) -> torch.Tensor:
    X, L = scores.shape
    total = min(total, L)
    recent = min(recent_budget, total, L)
    heavy = total - recent
    sink = min(sink_len, heavy, L)
    recent_start = L - recent
    sink_idx = torch.arange(sink, device=scores.device)[None].expand(X, -1)
    recent_idx = torch.arange(recent_start, L, device=scores.device)[None].expand(X, -1)
    middle_start, middle_end = sink, max(sink, recent_start)
    k = min(max(heavy - sink, 0), max(middle_end-middle_start, 0))
    if k:
        hh = torch.topk(scores[:, middle_start:middle_end], k=k, dim=-1).indices + middle_start
    else:
        hh = torch.empty((X,0), device=scores.device, dtype=torch.long)
    idx = torch.sort(torch.cat([sink_idx, hh, recent_idx], -1), -1).values
    if idx.shape[-1] != total:
        mask = torch.zeros((X,L), dtype=torch.bool, device=scores.device); mask.scatter_(1,idx,True)
        fill = torch.topk(scores.masked_fill(mask, -torch.inf), total-idx.shape[-1], dim=-1).indices
        idx = torch.sort(torch.cat([idx,fill],-1),-1).values
    return idx


def _gather3(x, idx): return torch.gather(x, 1, idx[...,None].expand(-1,-1,x.shape[-1]))
def _gather2(x, idx): return torch.gather(x, 1, idx)


def _compress_layer(K, V, C, hist, total, heavy_budget, recent_budget, sink_len, sim_thr, distance_weight):
    B,H,L,D = K.shape
    if L <= total:
        return K.contiguous(), V.contiguous(), C.contiguous(), hist[..., :L].contiguous()
    X=B*H; Kx=K.reshape(X,L,D); Vx=V.reshape(X,L,D); Cx=C.reshape(X,L)
    hx=hist.reshape(X,hist.shape[-2],L)
    R=hx.shape[-2]
    dist=torch.linspace(1.0,float(distance_weight),R,device=K.device,dtype=torch.float32)
    scores=(hx*dist[None,:,None]).sum(-2).clamp_min(_EPS)
    keep=_select_keep(scores,total,heavy_budget,recent_budget,sink_len)
    mask=torch.zeros((X,L),dtype=torch.bool,device=K.device); mask.scatter_(1,keep,True)
    evict=(~mask).nonzero(as_tuple=False)[:,1].view(X,L-total)
    Kk=_gather3(Kx,keep).float(); Vk=_gather3(Vx,keep).float(); Ck=_gather2(Cx,keep).float(); Sk=_gather2(scores,keep)
    Ke=_gather3(Kx,evict).float(); Ve=_gather3(Vx,evict).float(); Ce=_gather2(Cx,evict).float(); Se=_gather2(scores,evict)
    # Official recent-token exposure correction.
    if recent_budget > 0:
        n=min(recent_budget,total)
        corr = n / torch.arange(n,0,-1,device=K.device,dtype=torch.float32)
        Sk[:,-n:] *= corr[None]
    sim=torch.bmm(F.normalize(Ke,dim=-1,eps=_EPS),F.normalize(Kk,dim=-1,eps=_EPS).transpose(1,2))
    best,target=sim.max(-1); accept=best >= float(sim_thr)
    Wk=(Ck*Sk).clamp_min(_EPS); We=(Ce*Se).clamp_min(_EPS)*accept.float()
    addW=torch.zeros_like(Wk); addW.scatter_add_(1,target,We)
    addC=torch.zeros_like(Ck); addC.scatter_add_(1,target,Ce*accept.float())
    addK=torch.zeros_like(Kk); addK.scatter_add_(1,target[...,None].expand(-1,-1,D),Ke*We[...,None])
    addV=torch.zeros_like(Vk); addV.scatter_add_(1,target[...,None].expand(-1,-1,D),Ve*We[...,None])
    denom=(Wk+addW).clamp_min(_EPS)
    Knew=(Kk*Wk[...,None]+addK)/denom[...,None]
    Vnew=(Vk*Wk[...,None]+addV)/denom[...,None]
    Cnew=Ck+addC
    # Merge score-history columns with cluster-count weighting.
    hk=torch.gather(hx,2,keep[:,None,:].expand(-1,R,-1))
    he=torch.gather(hx,2,evict[:,None,:].expand(-1,R,-1))
    numer=hk*Ck[:,None,:]
    addH=torch.zeros_like(numer)
    addH.scatter_add_(2,target[:,None,:].expand(-1,R,-1),he*(Ce*accept.float())[:,None,:])
    hnew=(numer+addH)/Cnew.clamp_min(1.0)[:,None,:]
    return (Knew.to(K.dtype).reshape(B,H,total,D).contiguous(),
            Vnew.to(V.dtype).reshape(B,H,total,D).contiguous(),
            Cnew.reshape(B,H,total).contiguous(),
            hnew.reshape(B,H,R,total).contiguous())


def KeepKV(model, key_value_cache, attention=None, hidden_states=None, ratio: float=0.8,
           sim_thr: float=0.8, sink_len: int=4, recent_fraction: float=0.5,
           kv_budget: Optional[int]=None, heavy_ratio: Optional[float]=None,
           recent_ratio: Optional[float]=None, distance_weight: float=1.2):
    del hidden_states, recent_fraction
    cache=_as_legacy_tuple(key_value_cache)
    if cache is None: raise ValueError("key_value_cache is None")
    new_past=[]; counts=[]; histories=[]; totals=[]; heavies=[]; recents=[]; biases=[]
    for i,(K,V) in enumerate(cache):
        B,H,L,D=K.shape; attn_mod=model.model.layers[i].self_attn
        total,heavy,recent=_choose_budgets(model,i,L,ratio,kv_budget,heavy_ratio,recent_ratio)
        C=_align_counts(model,i,K)
        fallback=None if attention is None or i>=len(attention) else attention[i]
        rows=_runtime_rows(attn_mod,B,H,L,fallback)
        old_hist=_state_item(model,"_keepkv_recent_scores",i)
        hist=_append_history(old_hist,rows,recent,L)
        Kn,Vn,Cn,Hn=_compress_layer(K,V,C,hist,total,heavy,recent,sink_len,sim_thr,distance_weight)
        bias=_make_log_bias_q(Cn,_num_kv_groups(attn_mod,H)).to(Kn.device,Kn.dtype)
        attn_mod._keepkv_enabled=True; attn_mod._keepkv_log_bias_q=bias; attn_mod._keepkv_recent_score_rows=None
        new_past.append((Kn,Vn)); counts.append(Cn); histories.append(Hn)
        totals.append(total); heavies.append(heavy); recents.append(recent); biases.append(bias)
    model._keepkv_counts=counts; model._keepkv_recent_scores=histories
    model._keepkv_cache_budgets=totals; model._keepkv_heavy_budgets=heavies; model._keepkv_recent_budgets=recents
    model._keepkv_log_bias_q=biases
    return tuple(new_past)


def Repair_keepkv(*args, **kwargs): return KeepKV(*args, **kwargs)
def Repair(*args, **kwargs): return KeepKV(*args, **kwargs)
