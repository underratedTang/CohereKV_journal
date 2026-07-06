"""
Tests and microbenchmarks for GroupLogitKV calibrated_qk.

The CUDA extension implements the single-query decoding kernel:

    out = q @ k^T / sqrt(D) + prefix_logC

Shapes:
    q:    [B, H, 1, D]
    k:    [B, H, K, D]
    logc: [B, H, 1, Kc], where 0 <= Kc <= K

If Kc < K, positions j >= Kc receive zero bias. This matches the
GroupLogitKV runtime convention: compressed prefix has log(C), generated
suffix tokens have C=1 and therefore log(C)=0.

This test file checks:
  1. numerical correctness against PyTorch reference;
  2. prefix-short and zero-length logC behavior;
  3. expected rejection of q_len > 1 and logC longer than K;
  4. microbenchmarks against several PyTorch baselines.
"""

from __future__ import annotations

import argparse
import math
from typing import Callable, Iterable, Tuple

import torch

from grouplogitkv_calibrated_qk import calibrated_qk


Shape = Tuple[int, int, int, int, int]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _reference(q: torch.Tensor, k: torch.Tensor, logc: torch.Tensor) -> torch.Tensor:
    """Reference: qk / sqrt(D) plus prefix logC; suffix bias is zero."""
    d = q.shape[-1]
    ref = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d)
    k_logc = logc.shape[-1]
    if k_logc > 0:
        ref[..., :k_logc] = ref[..., :k_logc] + logc
    return ref


def _reference_bmm(q: torch.Tensor, k: torch.Tensor, logc: torch.Tensor) -> torch.Tensor:
    """BMM reference with [B,H] flattened; useful as a fairer q_len=1 baseline."""
    bsz, heads, q_len, dim = q.shape
    _, _, key_len, _ = k.shape
    q2 = q.reshape(bsz * heads, q_len, dim)
    k2 = k.reshape(bsz * heads, key_len, dim)
    out = torch.bmm(q2, k2.transpose(1, 2)).reshape(bsz, heads, q_len, key_len)
    out = out / math.sqrt(dim)
    k_logc = logc.shape[-1]
    if k_logc > 0:
        out[..., :k_logc] = out[..., :k_logc] + logc
    return out


def _tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    if dtype is torch.float32:
        return 3e-4, 3e-4
    if dtype is torch.float16:
        return 5e-2, 5e-2
    if dtype is torch.bfloat16:
        return 8e-2, 8e-2
    return 1e-3, 1e-3


def _make_inputs(
    B: int,
    H: int,
    K: int,
    D: int,
    Kc: int,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    q = torch.randn(B, H, 1, D, device="cuda", dtype=dtype, generator=gen)
    k = torch.randn(B, H, K, D, device="cuda", dtype=dtype, generator=gen)
    logc = torch.randn(B, H, 1, Kc, device="cuda", dtype=dtype, generator=gen) * 0.1
    return q, k, logc


def _check_case(B: int, H: int, K: int, D: int, Kc: int, dtype: torch.dtype, seed: int) -> None:
    q, k, logc = _make_inputs(B, H, K, D, Kc, dtype, seed)

    y = calibrated_qk(q, k, logc)
    ref = _reference(q, k, logc)
    torch.cuda.synchronize()

    if y.shape != (B, H, 1, K):
        raise AssertionError(f"unexpected output shape: got {tuple(y.shape)}, expected {(B, H, 1, K)}")

    max_diff = (y.float() - ref.float()).abs().max().item()
    mean_diff = (y.float() - ref.float()).abs().mean().item()
    rtol, atol = _tolerances(dtype)

    ok = torch.allclose(y.float(), ref.float(), rtol=rtol, atol=atol)
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] correctness dtype={_dtype_name(dtype):>8s}, "
        f"B={B:>2d}, H={H:>3d}, K={K:>5d}, D={D:>3d}, Kc={Kc:>5d}: "
        f"shape={tuple(y.shape)}, max_diff={max_diff:.6g}, mean_diff={mean_diff:.6g}"
    )
    if not ok:
        raise AssertionError(
            f"calibrated_qk mismatch: dtype={dtype}, shape={(B, H, K, D, Kc)}, "
            f"max_diff={max_diff}, mean_diff={mean_diff}, rtol={rtol}, atol={atol}"
        )


def _expect_error(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - this is an explicit negative test
        print(f"[PASS] expected error for {name}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        return
    raise AssertionError(f"expected an error for {name}, but the call succeeded")


def _negative_tests(dtype: torch.dtype) -> None:
    print("\nNegative tests")
    q, k, logc = _make_inputs(B=1, H=4, K=64, D=64, Kc=32, dtype=dtype, seed=123)

    def q_len_gt_1() -> None:
        q2 = torch.randn(1, 4, 2, 64, device="cuda", dtype=dtype)
        calibrated_qk(q2, k, logc)

    def logc_longer_than_k() -> None:
        bad_logc = torch.randn(1, 4, 1, 65, device="cuda", dtype=dtype)
        calibrated_qk(q, k, bad_logc)

    _expect_error("q_len > 1", q_len_gt_1)
    _expect_error("logC length > key length", logc_longer_than_k)


def _cuda_event_ms(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    # Make one output live so the compiler/runtime cannot trivially elide work.
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize()

    # Touch the output once outside the measured region for sanity.
    if out is None:
        raise RuntimeError("benchmark function produced no output")
    _ = float(out.flatten()[0].detach().float().cpu())

    return start.elapsed_time(end) / iters


def _benchmark_case(
    B: int,
    H: int,
    K: int,
    D: int,
    Kc: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    seed: int,
) -> None:
    q, k, logc = _make_inputs(B, H, K, D, Kc, dtype, seed)
    scale = 1.0 / math.sqrt(D)

    def torch_matmul_raw() -> torch.Tensor:
        return torch.matmul(q, k.transpose(2, 3))

    def torch_matmul_scale() -> torch.Tensor:
        return torch.matmul(q, k.transpose(2, 3)) * scale

    def torch_matmul_scale_logc() -> torch.Tensor:
        return _reference(q, k, logc)

    def torch_bmm_scale_logc() -> torch.Tensor:
        return _reference_bmm(q, k, logc)

    def custom() -> torch.Tensor:
        return calibrated_qk(q, k, logc)

    # Check equality once before timing.
    ref = torch_matmul_scale_logc()
    y = custom()
    torch.cuda.synchronize()
    max_diff = (y.float() - ref.float()).abs().max().item()
    mean_diff = (y.float() - ref.float()).abs().mean().item()

    timings = [
        ("torch matmul raw", _cuda_event_ms(torch_matmul_raw, warmup, iters)),
        ("torch matmul + scale", _cuda_event_ms(torch_matmul_scale, warmup, iters)),
        ("torch matmul + scale + prefix logC", _cuda_event_ms(torch_matmul_scale_logc, warmup, iters)),
        ("torch bmm(flat BH) + scale + prefix logC", _cuda_event_ms(torch_bmm_scale_logc, warmup, iters)),
        ("calibrated_qk custom", _cuda_event_ms(custom, warmup, iters)),
    ]

    print(
        f"\nBenchmark dtype={_dtype_name(dtype)}, B={B}, H={H}, K={K}, D={D}, Kc={Kc}, "
        f"warmup={warmup}, iters={iters}"
    )
    print(f"custom vs torch reference: max_diff={max_diff:.6g}, mean_diff={mean_diff:.6g}")
    width = max(len(name) for name, _ in timings)
    for name, ms in timings:
        print(f"{name:<{width}} : {ms:.4f} ms")


def _available_dtypes() -> Iterable[torch.dtype]:
    yield torch.float32
    yield torch.float16
    if torch.cuda.is_bf16_supported():
        yield torch.bfloat16


def main() -> None:
    parser = argparse.ArgumentParser(description="Test GroupLogitKV calibrated_qk CUDA extension")
    parser.add_argument("--quick", action="store_true", help="run fewer correctness cases")
    parser.add_argument("--bench-only", action="store_true", help="skip correctness and negative tests")
    parser.add_argument("--no-bench", action="store_true", help="skip microbenchmarks")
    parser.add_argument("--iters", type=int, default=500, help="benchmark iterations")
    parser.add_argument("--warmup", type=int, default=50, help="benchmark warmup iterations")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for grouplogitkv_calibrated_qk tests")

    torch.manual_seed(args.seed)
    print(f"torch={torch.__version__}, cuda={torch.version.cuda}, device={torch.cuda.get_device_name(0)}")

    primary_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if not args.bench_only:
        print("\nCorrectness tests")
        shapes: list[Shape]
        if args.quick:
            shapes = [
                (1, 32, 953, 128, 953),
                (1, 32, 953, 128, 512),
                (1, 32, 953, 128, 0),
            ]
            dtypes = [primary_dtype]
        else:
            shapes = [
                (1, 32, 953, 128, 953),   # full-length logC
                (1, 32, 953, 128, 512),   # prefix-short logC
                (1, 32, 953, 128, 1),     # single biased prefix position
                (1, 32, 953, 128, 0),     # zero-bias path
                (2, 8, 257, 64, 128),     # batch sanity
                (1, 4, 1, 128, 1),        # minimal key length
                (1, 4, 4096, 128, 2048),  # longer decode cache
            ]
            dtypes = list(_available_dtypes())

        for dtype in dtypes:
            for idx, (B, H, K, D, Kc) in enumerate(shapes):
                _check_case(B, H, K, D, Kc, dtype, seed=args.seed + idx)

        _negative_tests(primary_dtype)

    if not args.no_bench:
        _benchmark_case(
            B=1,
            H=32,
            K=953,
            D=128,
            Kc=512,
            dtype=primary_dtype,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed + 1000,
        )
        if not args.quick:
            _benchmark_case(
                B=1,
                H=32,
                K=4096,
                D=128,
                Kc=2048,
                dtype=primary_dtype,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed + 1001,
            )


if __name__ == "__main__":
    main()
