"""
cuTile 融合 MLP first layer kernel

融合 matmul + bias + GELU 激活，减少全局内存写入次数。
"""

import cuda.tile as ct
import torch
from math import ceil


@ct.kernel
def mlp_first_layer_kernel(A, B, Bias, C,
                            M: ct.Constant[int], N: ct.Constant[int],
                            K: ct.Constant[int], TM: ct.Constant[int],
                            TN: ct.Constant[int], TK: ct.Constant[int]):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    num_tiles_k = ct.cdiv(K, TK)

    acc = ct.full((TM, TN), 0.0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # matmul 循环
    for k in range(num_tiles_k):
        a = ct.load(A, index=(pid_m, k), shape=(TM, TK), padding_mode=zero_pad)
        b = ct.load(B, index=(k, pid_n), shape=(TK, TN), padding_mode=zero_pad)
        acc = ct.mma(a, b, acc)

    # bias add
    bias = ct.load(Bias, index=(pid_n,), shape=(TN,), padding_mode=zero_pad)
    acc = acc + bias

    # GELU (tanh 近似)
    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (acc + 0.044715 * acc * acc * acc)
    result = 0.5 * acc * (1.0 + ct.tanh(inner))

    ct.store(C, index=(pid_m, pid_n), tile=result)


def mlp_first_layer_cutile(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """融合 matmul + bias + GELU。x: (M, K), w: (K, N), bias: (N,) -> (M, N)"""
    M, K = x.shape
    N = weight.shape[1]
    TM, TN, TK = 32, 32, 32
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = (ceil(M / TM), ceil(N / TN), 1)
    ct.launch(torch.cuda.current_stream(), grid, mlp_first_layer_kernel,
              (x, weight, bias, out, M, N, K, TM, TN, TK))
    return out
