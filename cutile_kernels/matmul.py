"""
cuTile 分块矩阵乘法

C = A @ B，A: (M, K), B: (K, N), C: (M, N)

使用 ct.mma 融合乘加，FP32 累加器。
"""

import cuda.tile as ct
import torch
from math import ceil
from triton_kernels.gpu_utils import get_arch_params


@ct.kernel
def matmul_kernel(A, B, C, M: ct.Constant[int], N: ct.Constant[int], K: ct.Constant[int],
                  TM: ct.Constant[int], TN: ct.Constant[int], TK: ct.Constant[int]):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    num_tiles_k = ct.cdiv(K, TK)

    acc = ct.full((TM, TN), 0.0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    for k in range(num_tiles_k):
        a = ct.load(A, index=(pid_m, k), shape=(TM, TK), padding_mode=zero_pad)
        b = ct.load(B, index=(k, pid_n), shape=(TK, TN), padding_mode=zero_pad)
        acc = ct.mma(a, b, acc)

    ct.store(C, index=(pid_m, pid_n), tile=acc)


def cutile_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """cuTile 分块矩阵乘法。a: (M, K), b: (K, N) -> (M, N)"""
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape

    TM, TN, TK = get_arch_params()["cutile_matmul_tile"]
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = (ceil(M / TM), ceil(N / TN), 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel,
              (a, b, c, M, N, K, TM, TN, TK))
    return c

# ============================================================
# 共享-A 融合 matmul（gate+up, 2026-09-02 v2.5 优化）
# ============================================================

@ct.kernel
def matmul_pair_kernel(A, B1, B2, C1, C2,
                       M: ct.Constant[int], N: ct.Constant[int], K: ct.Constant[int],
                       TM: ct.Constant[int], TN: ct.Constant[int], TK: ct.Constant[int]):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    num_tiles_k = ct.cdiv(K, TK)

    acc1 = ct.full((TM, TN), 0.0, dtype=ct.float32)
    acc2 = ct.full((TM, TN), 0.0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    for k in range(num_tiles_k):
        a = ct.load(A, index=(pid_m, k), shape=(TM, TK), padding_mode=zero_pad)
        b1 = ct.load(B1, index=(k, pid_n), shape=(TK, TN), padding_mode=zero_pad)
        b2 = ct.load(B2, index=(k, pid_n), shape=(TK, TN), padding_mode=zero_pad)
        acc1 = ct.mma(a, b1, acc1)
        acc2 = ct.mma(a, b2, acc2)

    ct.store(C1, index=(pid_m, pid_n), tile=acc1)
    ct.store(C2, index=(pid_m, pid_n), tile=acc2)


def cutile_matmul_pair(a: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor):
    """返回 (a@b1, a@b2)。A 一次读。输出 fp32（同 cutile_matmul 约定）。"""
    M, K = a.shape
    _, N = b1.shape
    assert b2.shape == b1.shape

    TM, TN, TK = get_arch_params()["cutile_matmul_tile"]
    grid = (ceil(M / TM), ceil(N / TN), 1)
    c1 = torch.empty(M, N, device=a.device, dtype=torch.float32)
    c2 = torch.empty(M, N, device=a.device, dtype=torch.float32)
    ct.launch(torch.cuda.current_stream(), grid, matmul_pair_kernel,
              (a, b1, b2, c1, c2, M, N, K, TM, TN, TK))
    return c1, c2
