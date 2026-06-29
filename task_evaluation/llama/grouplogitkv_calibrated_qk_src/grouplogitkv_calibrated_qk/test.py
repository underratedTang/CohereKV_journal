import math
import time
import torch
from grouplogitkv_calibrated_qk import calibrated_qk


def main():
    torch.manual_seed(0)
    B, H, K, D = 1, 32, 953, 128
    dtype = torch.bfloat16

    q = torch.randn(B, H, 1, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, K, D, device="cuda", dtype=dtype)
    logc = torch.randn(B, H, 1, K, device="cuda", dtype=dtype) * 0.1

    y = calibrated_qk(q, k, logc)
    torch.cuda.synchronize()

    ref = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(D)
    ref = ref + logc
    torch.cuda.synchronize()

    print("shape:", y.shape, y.dtype)
    print("max diff:", (y.float() - ref.float()).abs().max().item())
    print("mean diff:", (y.float() - ref.float()).abs().mean().item())

    iters = 200

    for _ in range(20):
        y0 = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(D)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        y0 = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(D)
    torch.cuda.synchronize()
    t1 = time.time()

    for _ in range(20):
        y1 = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(D)
        y1 = y1 + logc
    torch.cuda.synchronize()
    t2 = time.time()
    for _ in range(iters):
        y1 = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(D)
        y1 = y1 + logc
    torch.cuda.synchronize()
    t3 = time.time()

    for _ in range(20):
        y2 = calibrated_qk(q, k, logc)
    torch.cuda.synchronize()
    t4 = time.time()
    for _ in range(iters):
        y2 = calibrated_qk(q, k, logc)
    torch.cuda.synchronize()
    t5 = time.time()

    print(f"torch QK only       : {(t1 - t0) / iters * 1000:.4f} ms")
    print(f"torch QK + logC     : {(t3 - t2) / iters * 1000:.4f} ms")
    print(f"calibrated_qk custom: {(t5 - t4) / iters * 1000:.4f} ms")


if __name__ == "__main__":
    main()
