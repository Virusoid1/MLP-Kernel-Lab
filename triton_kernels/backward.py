"""
Triton MatMul Backward 和 Activation Backward kernel（autotune 优化版）

MatMul Backward（前向 C = A @ B）：
- dA = dC @ B^T:  autotune 分块 matmul（带 tl.trans）
- dB = A^T @ dC:  autotune 分块 matmul（带 tl.trans）

Activation Backward（逐元素梯度）：
- ReLU backward:   grad_input = grad_output * (input > 0)
- GELU backward:   grad_input = grad_output * GELU'(input)
- SiLU backward:   grad_input = grad_output * SiLU'(input)

针对 RTX 5070 Ti (Blackwell SM12.0) 和 RTX 3070 Laptop (Ampere SM8.6) 优化：
- autotune 自动选择最优 BLOCK 配置
- TF32 启用（allow_tf32=True）利用 tensor core 加速
"""

import torch
import triton
import triton.language as tl

from triton_kernels.precision import precision


# ====== Backward autotune 配置 ======

_BACKWARD_A_CONFIGS = [
    # 大 tile：Blackwell 最优
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_stages=3, num_warps=8,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_stages=3, num_warps=8,
    ),
    # 中 tile：Ampere 平衡
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=4,
    ),
    # 小 tile
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=4,
    ),
]

_BACKWARD_B_CONFIGS = [
    # 大 tile：Blackwell 最优
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_K": 8},
        num_stages=3, num_warps=8,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_K": 8},
        num_stages=3, num_warps=8,
    ),
    # 中 tile：Ampere 平衡
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_K": 8},
        num_stages=4, num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_K": 8},
        num_stages=4, num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_SIZE_K": 8},
        num_stages=4, num_warps=4,
    ),
    # 小 tile
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_K": 8},
        num_stages=4, num_warps=4,
    ),
]


# ============ MatMul Backward: dA = dC @ B^T ============

@triton.autotune(configs=_BACKWARD_A_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_backward_a_kernel(
    dC_ptr, B_ptr, dA_ptr,
    M, N, K,
    stride_dcm, stride_dcn,
    stride_bk, stride_bn,
    stride_dam, stride_dak,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """计算 dA = dC @ B^T。dC: (M, N), B: (K, N), dA: (M, K)"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    dC_ptrs = dC_ptr + offs_m[:, None] * stride_dcm + offs_n[None, :] * stride_dcn
    B_ptrs = B_ptr + offs_k[None, :] * stride_bk + offs_n[:, None] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        n_mask = (n_start + offs_n)[:, None] < N
        dC = tl.load(dC_ptrs, mask=(offs_m[:, None] < M) & (n_start + offs_n)[None, :] < N, other=0.0)
        B_tile = tl.load(B_ptrs, mask=n_mask & (offs_k[None, :] < K), other=0.0)
        acc += tl.dot(dC, B_tile, allow_tf32=ALLOW_TF32)
        dC_ptrs += BLOCK_N * stride_dcn
        B_ptrs += BLOCK_N * stride_bn

    dA_ptrs = dA_ptr + offs_m[:, None] * stride_dam + offs_k[None, :] * stride_dak
    tl.store(dA_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))


def matmul_backward_a(dC: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """dA = dC @ B^T。dC: (M, N), B: (K, N) -> dA: (M, K)"""
    assert dC.dim() == 2 and B.dim() == 2
    assert dC.shape[1] == B.shape[1], f"N mismatch: dC {dC.shape}, B {B.shape}"
    M, N = dC.shape
    K = B.shape[0]
    dA = torch.empty((M, K), device=dC.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(K, meta["BLOCK_K"]),
    )
    matmul_backward_a_kernel[grid](
        dC, B, dA,
        M, N, K,
        dC.stride(0), dC.stride(1),
        B.stride(0), B.stride(1),
        dA.stride(0), dA.stride(1),
        ALLOW_TF32=precision.allow_tf32,
    )
    return dA


# ============ MatMul Backward: dB = A^T @ dC ============

@triton.autotune(configs=_BACKWARD_B_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_backward_b_kernel(
    A_ptr, dC_ptr, dB_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_dcm, stride_dcn,
    stride_dbk, stride_dbn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """计算 dB = A^T @ dC。A: (M, K), dC: (M, N), dB: (K, N)"""
    pid = tl.program_id(0)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_K * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_k = group_id * GROUP_SIZE_K
    group_size_k = min(num_pid_k - first_pid_k, GROUP_SIZE_K)
    pid_k = first_pid_k + ((pid % num_pid_in_group) % group_size_k)
    pid_n = (pid % num_pid_in_group) // group_size_k

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)

    A_ptrs = A_ptr + offs_m[None, :] * stride_am + offs_k[:, None] * stride_ak
    dC_ptrs = dC_ptr + offs_m[:, None] * stride_dcm + offs_n[None, :] * stride_dcn
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        m_mask = (m_start + offs_m)[None, :] < M
        A_tile = tl.load(A_ptrs, mask=m_mask & (offs_k[:, None] < K), other=0.0)
        dC = tl.load(dC_ptrs, mask=((m_start + offs_m)[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(A_tile, dC, allow_tf32=ALLOW_TF32)
        A_ptrs += BLOCK_M * stride_am
        dC_ptrs += BLOCK_M * stride_dcm

    dB_ptrs = dB_ptr + offs_k[:, None] * stride_dbk + offs_n[None, :] * stride_dbn
    tl.store(dB_ptrs, acc, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))


def matmul_backward_b(A: torch.Tensor, dC: torch.Tensor) -> torch.Tensor:
    """dB = A^T @ dC。A: (M, K), dC: (M, N) -> dB: (K, N)"""
    assert A.dim() == 2 and dC.dim() == 2
    assert A.shape[0] == dC.shape[0], f"M mismatch: A {A.shape}, dC {dC.shape}"
    M, K = A.shape
    N = dC.shape[1]
    dB = torch.empty((K, N), device=A.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(K, meta["BLOCK_K"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    matmul_backward_b_kernel[grid](
        A, dC, dB,
        M, N, K,
        A.stride(0), A.stride(1),
        dC.stride(0), dC.stride(1),
        dB.stride(0), dB.stride(1),
        ALLOW_TF32=precision.allow_tf32,
    )
    return dB


# ============ Activation Backward autotune 配置 ============

_ACT_BWD_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 512}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
]


# ============ ReLU Backward ============

@triton.autotune(configs=_ACT_BWD_CONFIGS, key=["n_elements"])
@triton.jit
def relu_backward_kernel(
    grad_out_ptr, input_ptr, grad_in_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """ReLU 反向：grad_input = grad_output * (input > 0)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(grad_in_ptr + offsets, tl.where(x > 0, grad_out, 0.0), mask=mask)


def relu_backward(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Triton ReLU 反向。支持任意形状。"""
    grad_input = torch.empty_like(grad_output)
    n = grad_output.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    relu_backward_kernel[grid](grad_output, x, grad_input, n)
    return grad_input


# ============ GELU Backward ============

@triton.autotune(configs=_ACT_BWD_CONFIGS, key=["n_elements"])
@triton.jit
def gelu_backward_kernel(
    grad_out_ptr, input_ptr, grad_in_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GELU'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    sqrt_2_over_pi = 0.7978845608028654
    u = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    tanh_u = 2.0 * tl.sigmoid(2.0 * u) - 1.0
    sech2_u = 1.0 - tanh_u * tanh_u
    du_dx = sqrt_2_over_pi * (1.0 + 0.134145 * x * x)
    gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx
    tl.store(grad_in_ptr + offsets, grad_out * gelu_grad, mask=mask)


def gelu_backward(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Triton GELU 反向（tanh 近似）。支持任意形状。"""
    grad_input = torch.empty_like(grad_output)
    n = grad_output.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    gelu_backward_kernel[grid](grad_output, x, grad_input, n)
    return grad_input


# ============ SiLU Backward ============

@triton.autotune(configs=_ACT_BWD_CONFIGS, key=["n_elements"])
@triton.jit
def silu_backward_kernel(
    grad_out_ptr, input_ptr, grad_in_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sig = tl.sigmoid(x)
    silu_grad = sig * (1.0 + x * (1.0 - sig))
    tl.store(grad_in_ptr + offsets, grad_out * silu_grad, mask=mask)


def silu_backward(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Triton SiLU 反向。支持任意形状。"""
    grad_input = torch.empty_like(grad_output)
    n = grad_output.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    silu_backward_kernel[grid](grad_output, x, grad_input, n)
    return grad_input
