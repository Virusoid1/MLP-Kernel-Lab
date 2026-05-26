"""
cuTile MatMul Backward 和 Activation Backward kernel

MatMul Backward:
- dA = dC @ B^T  (dC: MxN, B: KxN, dA: MxK)
- dB = A^T @ dC  (A: MxK, dC: MxN, dB: KxN)

Activation Backward:
- ReLU backward
- GELU backward (tanh 近似)
- SiLU backward
"""

import cuda.tile as ct
import torch
from math import ceil


# ============ MatMul Backward: dA = dC @ B^T ============

@ct.kernel
def matmul_backward_a_kernel(dC, B, dA,
                              M: ct.Constant[int], N: ct.Constant[int],
                              K: ct.Constant[int], TM: ct.Constant[int],
                              TN: ct.Constant[int], TK: ct.Constant[int]):
    pid_m = ct.bid(0)
    pid_k = ct.bid(1)
    num_tiles_n = ct.cdiv(N, TN)

    acc = ct.full((TM, TK), 0.0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    for n_tile in range(num_tiles_n):
        dc = ct.load(dC, index=(pid_m, n_tile), shape=(TM, TN), padding_mode=zero_pad)
        # B^T 的 tile: B 是 (K, N)，需要 (N, TK) 的 tile 即 B 的第 n_tile 列、第 pid_k 行
        # ct.load B index=(pid_k, n_tile) 得到 (TK, TN)，需要转置为 (TN, TK)
        b = ct.load(B, index=(pid_k, n_tile), shape=(TK, TN), padding_mode=zero_pad)
        b_T = ct.transpose(b)  # (TN, TK)
        acc = ct.mma(dc, b_T, acc)  # (TM, TK)

    ct.store(dA, index=(pid_m, pid_k), tile=acc)


def matmul_backward_a(dC: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """dA = dC @ B^T。dC: (M, N), B: (K, N) -> dA: (M, K)"""
    M, N = dC.shape
    K = B.shape[0]
    TM, TN, TK = 32, 32, 32
    dA = torch.empty((M, K), device=dC.device, dtype=torch.float32)

    grid = (ceil(M / TM), ceil(K / TK), 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul_backward_a_kernel,
              (dC, B, dA, M, N, K, TM, TN, TK))
    return dA


# ============ MatMul Backward: dB = A^T @ dC ============

@ct.kernel
def matmul_backward_b_kernel(A, dC, dB,
                              M: ct.Constant[int], N: ct.Constant[int],
                              K: ct.Constant[int], TM: ct.Constant[int],
                              TN: ct.Constant[int], TK: ct.Constant[int]):
    pid_k = ct.bid(0)
    pid_n = ct.bid(1)
    num_tiles_m = ct.cdiv(M, TM)

    acc = ct.full((TK, TN), 0.0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    for m_tile in range(num_tiles_m):
        a = ct.load(A, index=(m_tile, pid_k), shape=(TM, TK), padding_mode=zero_pad)
        dc = ct.load(dC, index=(m_tile, pid_n), shape=(TM, TN), padding_mode=zero_pad)
        a_T = ct.transpose(a)  # (TK, TM)
        acc = ct.mma(a_T, dc, acc)  # (TK, TN)

    ct.store(dB, index=(pid_k, pid_n), tile=acc)


def matmul_backward_b(A: torch.Tensor, dC: torch.Tensor) -> torch.Tensor:
    """dB = A^T @ dC。A: (M, K), dC: (M, N) -> dB: (K, N)"""
    M, K = A.shape
    N = dC.shape[1]
    TM, TN, TK = 32, 32, 32
    dB = torch.empty((K, N), device=A.device, dtype=torch.float32)

    grid = (ceil(K / TK), ceil(N / TN), 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul_backward_b_kernel,
              (A, dC, dB, M, N, K, TM, TN, TK))
    return dB


# ============ Activation Backward ============

@ct.kernel
def relu_backward_kernel(GradOut, X, GradIn, N: ct.Constant[int], TILE: ct.Constant[int]):
    pid = ct.bid(0)
    g = ct.load(GradOut, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    x = ct.load(X, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(GradIn, index=(pid,), tile=ct.where(x > 0.0, g, 0.0))


def relu_backward(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """cuTile ReLU backward。"""
    orig_shape = grad_output.shape
    g_flat = grad_output.reshape(-1)
    x_flat = x.reshape(-1)
    n = g_flat.shape[0]
    TILE = 512
    out_flat = torch.empty_like(g_flat)

    grid = (ceil(n / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, relu_backward_kernel,
              (g_flat, x_flat, out_flat, n, TILE))
    return out_flat.reshape(orig_shape)


@ct.kernel
def gelu_backward_kernel(GradOut, X, GradIn, N: ct.Constant[int], TILE: ct.Constant[int]):
    pid = ct.bid(0)
    g = ct.load(GradOut, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    x = ct.load(X, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)

    sqrt_2_over_pi = 0.7978845608028654
    u = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    tanh_u = ct.tanh(u)
    sech2_u = 1.0 - tanh_u * tanh_u
    du_dx = sqrt_2_over_pi * (1.0 + 0.134145 * x * x)
    gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx

    ct.store(GradIn, index=(pid,), tile=g * gelu_grad)


def gelu_backward(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """cuTile GELU backward（tanh 近似）。"""
    orig_shape = grad_output.shape
    g_flat = grad_output.reshape(-1)
    x_flat = x.reshape(-1)
    n = g_flat.shape[0]
    TILE = 512
    out_flat = torch.empty_like(g_flat)

    grid = (ceil(n / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, gelu_backward_kernel,
              (g_flat, x_flat, out_flat, n, TILE))
    return out_flat.reshape(orig_shape)


@ct.kernel
def silu_backward_kernel(GradOut, X, GradIn, N: ct.Constant[int], TILE: ct.Constant[int]):
    pid = ct.bid(0)
    g = ct.load(GradOut, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    x = ct.load(X, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)

    sig = 1.0 / (1.0 + ct.exp(-x))
    silu_grad = sig * (1.0 + x * (1.0 - sig))

    ct.store(GradIn, index=(pid,), tile=g * silu_grad)


def silu_backward(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """cuTile SiLU backward。"""
    orig_shape = grad_output.shape
    g_flat = grad_output.reshape(-1)
    x_flat = x.reshape(-1)
    n = g_flat.shape[0]
    TILE = 512
    out_flat = torch.empty_like(g_flat)

    grid = (ceil(n / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, silu_backward_kernel,
              (g_flat, x_flat, out_flat, n, TILE))
    return out_flat.reshape(orig_shape)
