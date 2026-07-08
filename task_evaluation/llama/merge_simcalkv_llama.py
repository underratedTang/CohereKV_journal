import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

EPS = 1e-8


def _aggregate_attention_scores_to_kv_heads(layer_attention: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Return cumulative token scores with shape (B, H_kv, L).

    Works for MHA, GQA and MQA. HF attentions are over query heads with shape
    (B, H_q, Q, K). KV cache is over KV heads with shape (B, H_kv, K, D).
    """
    if layer_attention.dim() == 3:
        # Pre-reduced token scores [B,H_kv,K] captured by the runtime patch.
        if layer_attention.shape[1] != num_kv_heads:
            raise ValueError(f"Expected pre-reduced scores with H_kv={num_kv_heads}, got {tuple(layer_attention.shape)}")
        return layer_attention
    if layer_attention.dim() != 4:
        raise ValueError(f"attention must be (B,Hq,Q,K) or score (B,H_kv,K), got {tuple(layer_attention.shape)}")
    B, H_q, Q, L = layer_attention.shape
    if H_q == num_kv_heads:
        attn_kv = layer_attention
    else:
        if H_q % num_kv_heads != 0:
            raise ValueError(f"H_q={H_q} is not divisible by H_kv={num_kv_heads}")
        groups = H_q // num_kv_heads
        attn_kv = layer_attention.view(B, num_kv_heads, groups, Q, L).mean(dim=2)
    return attn_kv.sum(dim=2)


def _get_all_qt_for_kv_heads(layer_idx: int, model, hidden_states: List[torch.Tensor], num_kv_heads: int) -> torch.Tensor:
    attn = model.model.layers[layer_idx].self_attn
    Q = attn.q_proj(hidden_states[layer_idx])
    B, S, _ = Q.shape
    num_q_heads = getattr(model.config, "num_attention_heads", getattr(attn, "num_heads", None))
    if num_q_heads is None:
        raise ValueError("Cannot infer number of query heads")
    head_dim = Q.shape[-1] // num_q_heads
    Q = Q.view(B, S, num_q_heads, head_dim).transpose(1, 2)[:, :, -1, :]  # (B,Hq,D)
    if num_q_heads == num_kv_heads:
        return Q
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"H_q={num_q_heads} is not divisible by H_kv={num_kv_heads}")
    groups = num_q_heads // num_kv_heads
    return Q.view(B, num_kv_heads, groups, head_dim).mean(dim=2)


def _merge_by_token_simcal(key1: torch.Tensor, key2: torch.Tensor) -> torch.Tensor:
    return 0.5 * (key1 + key2) + math.log(2.0)


def _get_w(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # q: (1,D), k: (N,D)
    d = q.shape[-1]
    logits = (q @ k.transpose(0, 1)) / math.sqrt(d)
    logits = logits.clamp(min=-50.0, max=50.0)
    return torch.exp(logits).clamp_min(EPS).squeeze(0)


def _merge_one_head(q_t: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scores: torch.Tensor, ratio: float, sim_thr: float) -> Tuple[torch.Tensor, torch.Tensor]:
    L, D = K.shape
    num_evict = int(L * float(ratio))
    if num_evict <= 0:
        return K, V
    num_evict = min(num_evict, L - 1)
    if num_evict <= 0:
        return K, V

    _, evict_idx = torch.topk(-scores, num_evict)
    keep_mask = torch.ones(L, dtype=torch.bool, device=K.device)
    keep_mask[evict_idx] = False
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]

    Ke, Ve = K[evict_idx], V[evict_idx]
    Kc, Vc = K[keep_idx], V[keep_idx]
    if Ke.numel() == 0 or Kc.numel() == 0:
        return Kc, Vc

    sim = F.normalize(Ke.float(), dim=1) @ F.normalize(Kc.float(), dim=1).transpose(0, 1)
    max_sim, target = sim.max(dim=1)
    if sim_thr <= -1.0:
        merge_mask = torch.ones_like(max_sim, dtype=torch.bool)
    else:
        merge_mask = max_sim >= float(sim_thr)
    if not merge_mask.any():
        return Kc, Vc

    Ke_sel, Ve_sel = Ke[merge_mask], Ve[merge_mask]
    target = target[merge_mask]
    N_keep = Kc.shape[0]

    sum_Ke = torch.zeros((N_keep, D), device=K.device, dtype=K.dtype)
    sum_Ke.scatter_add_(0, target[:, None].expand(-1, D), Ke_sel)
    count = torch.zeros(N_keep, device=K.device, dtype=K.dtype)
    count.scatter_add_(0, target, torch.ones_like(target, dtype=K.dtype))
    active = count > 0

    Kc_new = Kc.clone()
    grouped_Ke = torch.zeros_like(sum_Ke)
    grouped_Ke[active] = sum_Ke[active] / count[active, None]
    Kc_new[active] = _merge_by_token_simcal(grouped_Ke[active], Kc[active])

    w_e = _get_w(q_t, Ke_sel)
    w_c = _get_w(q_t, Kc)
    sum_w_e = torch.zeros(N_keep, device=K.device, dtype=K.dtype)
    sum_w_e.scatter_add_(0, target, w_e.to(K.dtype))
    sum_w_e_Ve = torch.zeros((N_keep, D), device=K.device, dtype=K.dtype)
    sum_w_e_Ve.scatter_add_(0, target[:, None].expand(-1, D), (w_e[:, None].to(K.dtype) * Ve_sel))
    Vc_new = (sum_w_e_Ve + w_c[:, None].to(K.dtype) * Vc) / (sum_w_e + w_c.to(K.dtype)).clamp_min(1e-6)[:, None]
    return Kc_new, Vc_new


def Repair(model, key_value_cache, attention, hidden_states, ratio: float, sim_thr: float = -1.0):
    cache = tuple(key_value_cache)
    out = []
    for layer_idx, (K_layer, V_layer) in enumerate(cache):
        B, H_kv, L, D = K_layer.shape
        if B != 1:
            raise ValueError("This evaluator currently expects batch_size=1")
        scores = _aggregate_attention_scores_to_kv_heads(attention[layer_idx], H_kv).to(K_layer.device)
        q_all = _get_all_qt_for_kv_heads(layer_idx, model, hidden_states, H_kv).to(device=K_layer.device, dtype=K_layer.dtype)
        Ks, Vs = [], []
        for h in range(H_kv):
            K_new, V_new = _merge_one_head(q_all[0, h].unsqueeze(0), K_layer[0, h], V_layer[0, h], scores[0, h], ratio, sim_thr)
            Ks.append(K_new)
            Vs.append(V_new)
        out.append((torch.stack(Ks, dim=0).unsqueeze(0), torch.stack(Vs, dim=0).unsqueeze(0)))
    return tuple(out)
