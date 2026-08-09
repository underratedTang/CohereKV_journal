# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress


def bipartite_soft_matching(key: torch.Tensor, n_ex: int, n_chunks: int, chunk_size: int, r: int):
    """
    Perform bipartite soft matching to find pairs of tokens to merge.
    
    Parameters
    ----------
    key : torch.Tensor
        Key tensor with shape (bsz, n_head, seq_len, head_dim)
    n_ex : int
        Number of extra tokens that don't fit into chunks
    n_chunks : int
        Number of chunks
    chunk_size : int
        Size of each chunk
    r : int
        Number of pairs to merge
    
    Returns
    -------
    unm_idx : torch.Tensor
        Indices of tokens to preserve (unmerged)
    src_idx : torch.Tensor
        Indices of source tokens to merge
    dst_idx : torch.Tensor
        Indices of destination tokens to merge into
    """
    bsz, n_head, _, head_dim = key.shape
    key = key / key.norm(dim=-1, keepdim=True)
    
    if n_ex == 0:
        key_chunks = key[:, :, :, :].view(bsz, n_head, n_chunks, chunk_size, head_dim)
        ex_node_max = None
        ex_node_idx = None
    else: 
        key_chunks = key[:, :, :-n_ex, :].view(bsz, n_head, n_chunks, chunk_size, head_dim)
        key_ex = key[:, :, -n_ex:, :]
        ex_a, ex_b = key_ex[:, :, ::2, :], key_ex[:, :, 1::2, :]
        scores_ex = ex_a @ ex_b.transpose(-1, -2)  # (bsz, n_head, n_l//2, n_l//2)
        ex_node_max, ex_node_idx = scores_ex.max(dim=-1)  # (bsz, n_head, n_l//2)
    
    a, b = key_chunks[:, :, :, ::2, :], key_chunks[:, :, :, 1::2, :]  # (bsz, n_head, n_c, n//2, head_dim)
    scores = a @ b.transpose(-1, -2)  # (bsz, n_head, n_c, n//2, n//2)
    
    node_max, node_idx = scores.max(dim=-1)  # (bsz, n_head, n_c, n//2)
    sort_score = node_max.reshape(bsz, n_head, -1)
    if n_ex != 0:
        sort_score = torch.cat((sort_score, ex_node_max), dim=2)
    edge_value, edge_idx = sort_score.sort(dim=-1, descending=True)
    
    edge_idx = edge_idx.unsqueeze(-1)
    
    unm_idx = edge_idx[:, :, r:, :]  # (bsz, n_head, n//2 - r, 1)
    src_idx = edge_idx[:, :, :r, :]  # (bsz, n_head, r, 1)
    
    chunk_offset = src_idx // (chunk_size // 2)  # (bsz, n_head, r, 1)
    local_idx = node_idx.reshape(bsz, n_head, -1, 1)
    if n_ex != 0:
        local_idx = torch.cat((local_idx, ex_node_idx.unsqueeze(-1)), dim=2)
    local_offset = local_idx.gather(dim=-2, index=src_idx)
    dst_idx = chunk_offset * chunk_size // 2 + local_offset
    
    return unm_idx, src_idx, dst_idx


@dataclass
class CentroidKVPress(BasePress):
    """
    CentroidKV: KV cache compression using bipartite soft matching and centroid merging.
    
    Implements a compression method that merges similar key-value pairs by computing
    weighted centroids. Uses bipartite soft matching to identify pairs of tokens to merge,
    then merges them by computing weighted averages based on their counts.
    
    Based on the bipartite soft matching algorithm from the RULER project.
    
    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    sink_budget : int, default=16
        Number of initial tokens to always preserve (sink tokens).
    local_budget : int, default=64
        Number of most recent tokens to always preserve (local tokens).
    chunk_size : int, default=1024
        Size of chunks for bipartite matching.
    merge_ratio : float, default=0.7
        Ratio of tokens to merge within each compression step.
    gap_step : int, default=64
        Number of tokens to exceed cache budget before triggering compression.
    """
    
    compression_ratio: float = 0.0
    sink_budget: int = 16
    local_budget: int = 64
    chunk_size: int = 256
    merge_ratio: float = 0.8
    decrease_rate: float = 0.2
    
    
    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        assert self.sink_budget >= 0, "sink_budget must be non-negative"
        assert self.local_budget >= 0, "local_budget must be non-negative"
        assert self.chunk_size > 0, "chunk_size must be positive"
        assert 0 <= self.merge_ratio <= 1, "merge_ratio must be between 0 and 1"
    
    def _bsmerge(
        self,
        step: int,
        key: torch.Tensor,
        value: torch.Tensor,
        count: torch.Tensor,
        n_local: int,
        n_sink: int,
        n_total: int,
        chunk_size: int,
        max_merge: int,
        ratio: float,
    ):
        """
        Perform bipartite soft merging of key-value pairs.
        
        Parameters
        ----------
        step : int
            Current merge step (0 for first merge)
        key : torch.Tensor
            Key tensor with shape (bsz, n_head, n_total, head_dim)
        value : torch.Tensor
            Value tensor with shape (bsz, n_head, n_total, head_dim)
        count : torch.Tensor
            Count tensor with shape (bsz, n_head, n_total, 1)
        n_local : int
            Number of local (recent) tokens to preserve
        n_sink : int
            Number of sink (initial) tokens to preserve
        n_total : int
            Total number of tokens
        chunk_size : int
            Size of chunks for matching
        max_merge : int
            Maximum number of pairs to merge
        ratio : float
            Ratio of tokens to merge
        
        Returns
        -------
        key : torch.Tensor
            Merged key tensor
        value : torch.Tensor
            Merged value tensor
        count : torch.Tensor
            Updated count tensor
        n_total : int
            New total number of tokens
        """
        bsz, n_head, _, head_dim = key.shape
        
        n_ex = (n_total - n_sink - n_local) % chunk_size
        if n_ex % 2 != 0:
            n_ex -= 1
            n_local += 1
        n_chunks = (n_total - n_sink - n_local) // chunk_size
        if n_chunks == 0:
            chunk_size = n_total - n_sink - n_local
            n_chunks = 1
            n_ex = 0
        
        sink_key = key[:, :, :n_sink, :]
        sink_value = value[:, :, :n_sink, :]
        sink_count = count[:, :, :n_sink, :]
        
        local_key = key[:, :, -n_local:, :]
        local_value = value[:, :, -n_local:, :]
        local_count = count[:, :, -n_local:, :]
        
        merge_key = key[:, :, n_sink:-n_local, :]
        merge_value = value[:, :, n_sink:-n_local, :]
        merge_count = count[:, :, n_sink:-n_local, :]
        
        n_merge = min(max_merge, int(merge_key.shape[-2] * ratio / 2))
        
        unm_idx, src_idx, dst_idx = bipartite_soft_matching(merge_key, n_ex, n_chunks, chunk_size, n_merge)
        
        unm_idx = unm_idx.expand(*unm_idx.shape[:-1], head_dim)
        src_idx = src_idx.expand(*src_idx.shape[:-1], head_dim)
        dst_idx = dst_idx.expand(*dst_idx.shape[:-1], head_dim)
        
        preserved_key = merge_key[:, :, ::2, :].gather(dim=-2, index=unm_idx)
        preserved_value = merge_value[:, :, ::2, :].gather(dim=-2, index=unm_idx)
        
        if step == 0:
            preserved_count = torch.ones((bsz, n_head, unm_idx.shape[-2], 1), dtype=count.dtype, device=count.device)
        else:
            preserved_count = merge_count[:, :, ::2, :].gather(dim=-2, index=unm_idx[..., 0].unsqueeze(-1))
        
        src_key = merge_key[:, :, ::2, :].gather(dim=-2, index=src_idx)
        src_value = merge_value[:, :, ::2, :].gather(dim=-2, index=src_idx)
        
        if step == 0:
            src_count = torch.ones((bsz, n_head, src_idx.shape[-2], 1), dtype=count.dtype, device=count.device)
        else:
            src_count = merge_count[:, :, ::2, :].gather(dim=-2, index=src_idx[..., 0].unsqueeze(-1))
        
        src_key.mul_(src_count)
        src_value.mul_(src_count)
        
        if step == 0:
            dst_key = merge_key[:, :, 1::2, :]
            dst_value = merge_value[:, :, 1::2, :]
        else:
            dst_key = merge_key[:, :, 1::2, :] * merge_count[:, :, 1::2, :]
            dst_value = merge_value[:, :, 1::2, :] * merge_count[:, :, 1::2, :]
        
        dst_count = merge_count[:, :, 1::2, :].scatter_reduce(-2, dst_idx[..., 0].unsqueeze(-1), src_count, reduce="sum")
        dst_key = dst_key.scatter_reduce(-2, dst_idx, src_key, reduce="sum")
        dst_value = dst_value.scatter_reduce(-2, dst_idx, src_value, reduce="sum")
        
        dst_key.div_(dst_count)
        dst_value.div_(dst_count)
        
        count = torch.cat((sink_count, preserved_count, dst_count, local_count), dim=-2)
        key = torch.cat((sink_key, preserved_key, dst_key, local_key), dim=-2)
        value = torch.cat((sink_value, preserved_value, dst_value, local_value), dim=-2)
        
        n_total = count.shape[-2]
        
        return key, value, count, n_total
    
    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compress key-value cache using bipartite soft matching and centroid merging.
        
        Parameters
        ----------
        module : nn.Module
            The transformer attention layer.
        hidden_states : torch.Tensor
            Hidden states with shape (batch_size, seq_len, hidden_dim).
        keys : torch.Tensor
            Key tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        values : torch.Tensor
            Value tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        attentions : torch.Tensor
            Attention weights (not used in this implementation).
        kwargs : dict
            Additional keyword arguments.
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Compressed keys and values.
        """
        if self.compression_ratio == 0:
            return keys, values
        
        bsz, n_head, k_len, head_dim = keys.shape
        
        # Initialize count tensor (tracking how many tokens are merged into each position)
        count = torch.ones((bsz, n_head, k_len, 1), dtype=keys.dtype, device=keys.device)
        
        # Calculate target cache size
        cache_budget = int(k_len * (1 - self.compression_ratio))
        
        # If current size is within budget, no compression needed
        if k_len <= cache_budget:
            return keys, values
        
        # Perform iterative merging until we reach the target size
        merge_step = 0
        current_key = keys.clone()
        current_value = values.clone()
        current_count = count.clone()
        current_length = k_len
        
        num_local = self.local_budget
        num_sink = self.sink_budget
        
        # Add maximum iterations to prevent infinite loops
        max_iterations = 10
        
        while current_length > cache_budget and merge_step < max_iterations:
            if self.decrease_rate == 0:
                effective_step = merge_step
            else:
                max_linear_steps = max(0, math.floor((self.merge_ratio - 1e-6) / self.decrease_rate))
                effective_step = min(merge_step, max_linear_steps)
            merge_ratio = self.merge_ratio - effective_step * self.decrease_rate
            max_merge = current_length - cache_budget
            
            current_key, current_value, current_count, current_length = self._bsmerge(
                merge_step,
                current_key,
                current_value,
                current_count,
                num_local,
                num_sink,
                current_length,
                self.chunk_size,
                max_merge,
                merge_ratio,
            )
            merge_step += 1
        
        return current_key, current_value
