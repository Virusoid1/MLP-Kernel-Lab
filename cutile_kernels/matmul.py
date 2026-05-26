"""
cuTile 分块矩阵乘法

C = A @ B，A: (M, K), B: (K, N), C: (M, N)

使用 ct.mma 融合乘加，FP32 累加器。
"""

import cuda.tile as ct
import torch
from math import ceil


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

    TM, TN, TK = 32, 32, 32
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = (ceil(M / TM), ceil(N / TN), 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel,
              (a, b, c, M, N, K, TM, TN, TK))
    return c
