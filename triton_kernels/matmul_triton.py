"""
Day 6: Triton 矩阵乘法

C[M, N] = A[M, K] @ B[K, N]

学习要点:
  - @triton.jit 装饰器
  - tl.program_id 获取 block 索引
  - tl.load / tl.store 读写 global memory
  - tl.dot 做 block-level 矩阵乘
  - mask 处理越界
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # TODO: 实现 Triton matmul kernel
    #
    # 提示:
    #   pid_m = tl.program_id(0)
    #   pid_n = tl.program_id(1)
    #
    #   offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    #   offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    #   offs_k = tl.arange(0, BLOCK_K)
    #
    #   a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    #   b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    #
    #   acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    #   for k in range(0, tl.cdiv(K, BLOCK_K)):
    #       a = tl.load(a_ptrs, mask=offs_m[:, None] < M and offs_k[None, :] < K, other=0.0)
    #       b = tl.load(b_ptrs, mask=offs_k[:, None] < K and offs_n[None, :] < N, other=0.0)
    #       acc += tl.dot(a, b)
    #       a_ptrs += BLOCK_K * stride_ak
    #       b_ptrs += BLOCK_K * stride_bk
    #
    #   c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    #   c_mask = offs_m[:, None] < M and offs_n[None, :] < N
    #   tl.store(c_ptrs, acc, mask=c_mask)
    pass


def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Triton matmul 入口函数"""
    assert A.is_cuda and B.is_cuda
    assert A.dim() == 2 and B.dim() == 2
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    # TODO: 启动 kernel
    # matmul_kernel[grid](
    #     A, B, C,
    #     M, N, K,
    #     A.stride(0), A.stride(1),
    #     B.stride(0), B.stride(1),
    #     C.stride(0), C.stride(1),
    #     BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    # )

    return C


if __name__ == "__main__":
    # 简单测试
    M, K, N = 512, 768, 3072
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)

    C_triton = matmul_triton(A, B)
    C_torch = A @ B

    if C_triton is not None:
        max_err = (C_triton - C_torch).abs().max().item()
        print(f"Triton matmul: M={M} K={K} N={N}, max_error={max_err:.6f}")
    else:
        print("Triton matmul: kernel not implemented yet")
