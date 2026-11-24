#!/usr/bin/env python3
"""StreamingLLM Figure-12-style train-only first-token sink diagnostic.

Sink measurements gate ONLY the optional *_sink methods; native MIS is untouched.
"""
from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def streamingllm_first_token_last_query(query, full_keys) -> np.ndarray:
    """StreamingLLM Figure-12-style diagnostic, PER attention (Q) head.

    For each Q head use only the LAST query of a *causal* full sequence,
    calculate softmax over all keys 0..T-1 and return probability of key 0.
    No prefix-8 pooling or 64-query averaging is involved.
    query: [num_q_heads, query_len, head_dim]; full_keys: [full_len, head_dim].
    This is calibration-only and must never use evaluation queries for gating.
    """
    q=torch.as_tensor(query,dtype=torch.float64,device='cpu')
    k=torch.as_tensor(full_keys,dtype=torch.float64,device='cpu')
    if q.ndim!=3 or k.ndim!=2 or q.shape[-1]!=k.shape[-1] or not q.shape[1] or not k.shape[0]:
        raise ValueError('Invalid post-RoPE Q/K for StreamingLLM sink diagnostic')
    last=q[:,-1,:]
    logits=torch.matmul(last,k.T) * float(last.shape[-1]**-0.5)
    mass=logits.softmax(-1)[:,0]
    if not torch.isfinite(mass).all():
        raise ValueError('Nonfinite first-token sink probability')
    return mass.numpy().copy()
