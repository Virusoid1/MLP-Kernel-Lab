"""
Triton fused MLP first layer kernel（autotune 优化版）

H = GELU(X @ W1 + bias)

在 matmul kernel 基础上融合 bias add 和 GELU activation，
减少中间结果的 global memory 写入。

针对 RTX 5070 Ti (Blackwell SM12.0) 和 RTX 3070 Laptop (Ampere SM8.6) 优化：
- autotune 自动选择最优 BLOCK 配置
- TF32 启用（allow_tf32=True）利用 tensor core 加速
"""

import torch
import triton
import triton.language as tl

from triton_kernels.precision import precision


_FUSED_MLP_CONFIGS = [
    # 大 tile：Blackwell 最优
    # SM 12.0: 228KB shared memory → 大 tile + 多 pipeline stages
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=8,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=8,
    ),
    # 中大 tile：Blackwell/Ada 最优
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_stages=3, num_warps=8,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_stages=3, num_warps=8,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=4,
    ),
    # 中 tile：Ampere 平衡
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_stages=3, num_warps=4,
    ),
    # 小 tile
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=4,
    ),
]


@triton.autotune(configs=_FUSED_MLP_CONFIGS, key=["M", "N", "K"])
@triton.jit
def mlp_first_layer_kernel(
    X_ptr, W1_ptr, bias_ptr, H_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_hm, stride_hn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    融合 matmul + bias + GELU kernel。
    每个 program 处理输出 H 的一个 (BLOCK_M, BLOCK_N) 子块。
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    b = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W1_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K

        x = tl.load(x_ptrs, mask=m_mask & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask, other=0.0)
        acc += tl.dot(x, w, allow_tf32=ALLOW_TF32)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    acc = acc + b[None, :]

    # GELU(tanh 近似)
    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (acc + 0.044715 * acc * acc * acc)
    tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    result = 0.5 * acc * (1.0 + tanh_inner)

    h_ptrs = H_ptr + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hn
    tl.store(h_ptrs, result, mask=m_mask & n_mask)


def mlp_first_layer_triton(
    X: torch.Tensor, W1: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Triton fused MLP first layer: H = GELU(X @ W1 + bias)。autotune 自动选择最优配置。"""
    assert X.is_cuda and W1.is_cuda and bias.is_cuda
    M, K = X.shape
    K2, N = W1.shape
    assert K == K2

    H = torch.empty((M, N), device=X.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    mlp_first_layer_kernel[grid](
        X, W1, bias, H,
        M, N, K,
        X.stride(0), X.stride(1),
        W1.stride(0), W1.stride(1),
        H.stride(0), H.stride(1),
        ALLOW_TF32=precision.allow_tf32,
    )
    return H
