#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from random_query_probes import (  # noqa: E402
    collect_independent_random_query_states,
    sample_distinct_random_token_ids,
)


class FakeCache:
    def __init__(self, length: int):
        self.length = int(length)
    def get_seq_length(self):
        return self.length
    def crop(self, length: int):
        self.length = int(length)


class FakeOut:
    def __init__(self, cache):
        self.past_key_values = cache


class FakeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(4))
        self.num_heads = 2
        self.head_dim = 2

    def forward(self, hidden_states, position_embeddings=None, **kwargs):
        return hidden_states


class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeLayer()])

    def forward(self, input_ids, past_key_values, use_cache, output_attentions,
                output_hidden_states, return_dict):
        assert input_ids.shape == (1,1)
        assert past_key_values.length == 3, "previous probe leaked into next probe"
        tid = float(int(input_ids.item()))
        hidden = torch.tensor([[[tid, 2.0, 3.0, 4.0]]], dtype=torch.float32)
        # Identity RoPE for the unit test: post-RoPE Q == q_proj(hidden).
        cos = torch.ones((1,1,2), dtype=torch.float32)
        sin = torch.zeros((1,1,2), dtype=torch.float32)
        self.layers[0].self_attn(hidden_states=hidden, position_embeddings=(cos,sin))
        past_key_values.length = 4
        return FakeOut(past_key_values)


def main():
    ids = sample_distinct_random_token_ids(vocab_size=50, num_probes=12, excluded_ids={0,1,2}, seed=7)
    assert len(ids)==12 and len(set(ids))==12 and not ({0,1,2}&set(ids))
    assert ids == sample_distinct_random_token_ids(vocab_size=50, num_probes=12, excluded_ids={0,1,2}, seed=7)

    cache = FakeCache(3)
    batch = collect_independent_random_query_states(
        model=FakeModel(), prefix_past=cache, input_device=torch.device("cpu"),
        token_ids=[5,9], selected_layers=[0], prefix_len=3, store_dtype=torch.float32,
    )
    qs = batch.queries_by_layer[0]
    assert tuple(qs.shape) == (2,2,2)  # [Hq,P,D]
    assert torch.allclose(qs[:,0], torch.tensor([[5.,2.],[3.,4.]]))
    assert torch.allclose(qs[:,1], torch.tensor([[9.,2.],[3.,4.]]))
    assert cache.length == 3
    print("PASS: independent probes capture post-RoPE Q and restore one frozen prefix")


if __name__ == "__main__":
    main()


def test_natural_future_query_branches_restore_same_prefix():
    from random_query_probes import collect_independent_query_states
    cache = FakeCache(3)
    batch = collect_independent_query_states(
        model=FakeModel(), prefix_past=cache, input_device=torch.device("cpu"),
        token_ids=[5, 9, 12], selected_layers=[0], prefix_len=3, store_dtype=torch.float32,
    )
    assert batch.token_ids == [5, 9, 12]
    assert tuple(batch.queries_by_layer[0].shape) == (2, 3, 2)
    assert cache.length == 3
