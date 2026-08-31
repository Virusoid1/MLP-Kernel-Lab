"""
Triton 分块矩阵乘法（autotune 优化版）

C = A @ B，其中 A: (M, K), B: (K, N), C: (M, N)

针对 RTX 5070 Ti (Blackwell SM12.0) 和 RTX 3070 Laptop (Ampere SM8.6) 优化：
- 使用 @triton.autotune 自动选择最优 BLOCK_M/N/K、num_warps、num_stages
- 大 tile (128x128) 利用 Blackwell 充足的 shared memory
- 中 tile (64x64) 平衡 Ampere 的 shared memory 限制
- GROUP_SIZE_M 优化 L2 缓存命中率
- TF32 模式可选启用 tensor core 加速（SM 8.0+）
"""

import torch
import triton
import triton.language as tl

from triton_kernels.precision import precision


# 针对 5070 Ti (Blackwell) 和 3070 Laptop (Ampere) 的 autotune 配置
# Blackwell: 更多 shared memory → 大 tile + 多 pipeline stages
# Ampere: 较少 shared memory → 中等 tile + 适度 pipeline
_MATMUL_CONFIGS = [
    # --- 大 tile：Blackwell 最优 ---
    # SM 12.0: 228KB shared memory → 大 tile + 多 pipeline stages
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=8,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=8,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_stages=5, num_warps=8,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_stages=6, num_warps=16,
    ),
    # --- 中大 tile：Blackwell/Ada 最优 ---
    # 128x128 充分利用 shared memory，8 warps 提高 SM 占用率
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
    # --- 中 tile：Ampere 平衡配置 ---
    # 64x64 适配 Ampere shared memory 限制，4 warps 平衡占用率
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_stages=3, num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_stages=5, num_warps=2,
    ),
    # --- 小 tile：MNIST 等小矩阵场景 ---
    # 64x64 强制配置:小 M (B=64) 走 2x N tile,num_warps=4 平衡 register / shmem
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=4,
    ),
    # 64x128:小 M 时把 N tile 翻倍,适合 (64, 1024) shape
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_stages=3, num_warps=4,
    ),
    # 128x64:M tile 大一倍,适合 (64, 784) K=784 的场景
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_stages=3, num_warps=4,
    ),
]


@triton.autotune(configs=_MATMUL_CONFIGS, key=["M", "N", "K"])
@triton.jit
def tiled_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    分块矩阵乘法 kernel（autotune 优化版）。
    每个 program 处理输出 C 的一个 (BLOCK_M, BLOCK_N) 子块。
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

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    # FP16/BF16 输入走 tl.dot 时累加器保持 FP32（标准做法），避免同 dtype 限制
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K

        a = tl.load(a_ptrs, mask=m_mask & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask, other=0.0)
        # fp16/bf16: 三参 dot 带 fp32 累加器（输入低精度，输出高精度）
        # fp32: 保持二参 dot + FP32 累加（原路径，避免三参语义差异）
        if a.dtype == tl.float32:
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        else:
            acc = tl.dot(a, b, acc, input_precision="ieee" if not ALLOW_TF32 else "tf32")

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask & n_mask)


def tiled_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Triton 分块矩阵乘法接口（autotune）。
    a: (M, K)  b: (K, N)  ->  返回 (M, N)
    首次调用会自动 benchmark 所有配置并缓存最优结果。
    """
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape[1] == b.shape[0], f"Shape mismatch: {a.shape} @ {b.shape}"
    M, K = a.shape
    _, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # autotune 会自动选择 BLOCK_M/N/K 和 GROUP_SIZE_M
    # 这里传默认值，autotune 装饰器会覆盖
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    tiled_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ALLOW_TF32=precision.allow_tf32,
    )
    return c
