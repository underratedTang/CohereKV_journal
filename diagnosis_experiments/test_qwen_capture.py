#!/usr/bin/env python3
"""CPU-only Qwen2.5/Llama Post-RoPE + GQA capture sanity tests.

Does not download model weights. Checks global/per-layer rotary hook variants,
projection biases, head grouping and fail-closed cache-K validation.
"""
import sys
import types
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_query_capture import (
    capture_real_future_queries, require_supported_model_type,
    replay_prefix_attention, rotate_half, rotate_queries,
)


class FakeRotary(torch.nn.Module):
    def forward(self, tokens):
        length = int(tokens.shape[1]); dim = int(tokens.shape[2])
        steps = torch.arange(length, dtype=torch.float32)[:, None]
        phase = steps * (torch.arange(dim // 2, dtype=torch.float32)[None] + 1) / 19
        phase = torch.cat((phase, phase), dim=-1)
        return phase.cos().unsqueeze(0), phase.sin().unsqueeze(0)


class FakeAttn(torch.nn.Module):
    def __init__(self, n_query, n_kv, dim, per_layer):
        super().__init__()
        self.q_proj = torch.nn.Linear(11, n_query * dim, bias=True)
        self.k_proj = torch.nn.Linear(11, n_kv * dim, bias=True)
        if per_layer:
            self.rotary_emb = FakeRotary()


class FakeLayer(torch.nn.Module):
    def __init__(self, n_query, n_kv, dim, per_layer):
        super().__init__()
        self.self_attn = FakeAttn(n_query, n_kv, dim, per_layer)


class FakeInner(torch.nn.Module):
    def __init__(self, n_query, n_kv, dim, per_layer):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeLayer(n_query, n_kv, dim, per_layer) for _ in range(2)])
        if not per_layer:
            self.rotary_emb = FakeRotary()


class FakeModel(torch.nn.Module):
    def __init__(self, n_query, n_kv, dim, per_layer):
        super().__init__()
        self.model = FakeInner(n_query, n_kv, dim, per_layer)
        self.n_query = n_query; self.n_kv = n_kv; self.dim = dim
        self.per_layer = per_layer
        self.corrupt_key = False

    def forward_fake(self, future_tokens):
        torch.manual_seed(103)
        hidden = torch.randn(1, future_tokens.shape[1], 11)
        if not self.per_layer:
            cos, sin = self.model.rotary_emb(hidden.new_zeros(1, hidden.shape[1], self.dim))
        keys = []; values = []
        for layer in self.model.layers:
            attn = layer.self_attn
            qproj = attn.q_proj(hidden)
            kproj = attn.k_proj(hidden)
            del qproj
            if self.per_layer:
                cos, sin = attn.rotary_emb(hidden.new_zeros(1, hidden.shape[1], self.dim))
            k = kproj.view(1, hidden.shape[1], self.n_kv, self.dim).transpose(1,2)
            k = k*cos[:, None] + rotate_half(k)*sin[:, None]
            if self.corrupt_key:
                k = k.roll(shifts=1, dims=-1)
            keys.append(k)
            values.append(torch.ones_like(k))
        return types.SimpleNamespace(past_key_values=types.SimpleNamespace(
            key_cache=keys, value_cache=values
        ))


def test_configuration(model_type, n_query, n_kv, per_layer):
    torch.manual_seed(3)
    fake = FakeModel(n_query, n_kv, 8, per_layer)
    cfg = types.SimpleNamespace(model_type=model_type,num_attention_heads=n_query,
        num_key_value_heads=n_kv, head_dim=8, hidden_size=11)
    stub = types.ModuleType('core_model_utils')
    stub.forward_cache_chunk = lambda model, future, past, device: model.forward_fake(future)
    stub.transformer_layers = lambda model: model.model.layers
    stub.cache_key_value_layers = lambda past: (past.key_cache, past.value_cache)
    old = sys.modules.get('core_model_utils')
    sys.modules['core_model_utils'] = stub
    try:
        future = torch.tensor([[7,8,9,10]])
        queries, cache = capture_real_future_queries(fake, future, None,
            torch.device('cpu'), cfg, [0,1])
        assert set(queries) == {0,1}
        for layer in [0,1]:
            assert queries[layer].shape == (n_query,4,8)
            assert torch.isfinite(queries[layer]).all()
            assert cache.key_cache[layer].shape == (1,n_kv,4,8)
        # Ensure a RoPE / KV head-order mismatch triggers a hard error.
        fake.corrupt_key = True
        try:
            capture_real_future_queries(fake, future, None,torch.device('cpu'),cfg,[0])
        except RuntimeError as e:
            assert 'disagrees' in str(e)
        else:
            raise AssertionError('Corrupt KV should fail post-RoPE validation')
    finally:
        if old is None:
            sys.modules.pop('core_model_utils',None)
        else:
            sys.modules['core_model_utils'] = old


def test_replay_causal():
    torch.manual_seed(5)
    q = torch.randn(4,3,8)
    pk = torch.randn(5,8); pv = torch.randn(5,8)
    fk = torch.randn(3,8); fv = torch.randn(3,8)
    approx = replay_prefix_attention(q,pk,pv,torch.ones(5),fk,fv,'cpu',1e-8)
    exact = []
    for h in range(4):
        rows = []
        for t in range(3):
            keys = torch.cat((pk,fk[:t+1]))
            vals = torch.cat((pv,fv[:t+1]))
            rows.append(((q[h,t] @ keys.T / 8**.5).softmax(-1) @ vals))
        exact.append(torch.stack(rows))
    assert torch.allclose(approx,torch.stack(exact), atol=1e-6)


def main():
    assert require_supported_model_type('qwen2') == 'qwen2'
    assert require_supported_model_type('llama') == 'llama'
    try:
        require_supported_model_type('qwen3')
    except RuntimeError:pass
    else:raise AssertionError('Unexpected model admitted')
    # Qwen2.5-3B/7B: native 16/2 and 28/4 query/KV heads.
    test_configuration('qwen2',16,2,False)
    test_configuration('qwen2',28,4,False)
    test_configuration('qwen2',16,2,True)
    test_configuration('llama',8,2,False)
    test_replay_causal()
    print('[PASS] Qwen2.5-3B (16/2), 7B (28/4), Llama GQA; global/layer RoPE, real cache-K validation, causal replay')

if __name__=='__main__':
    main()
