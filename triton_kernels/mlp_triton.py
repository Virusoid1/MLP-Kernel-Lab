"""
Day 6+: Triton MLP kernel

H = GELU(X @ W1 + bias)

学习要点:
  - 在 matmul kernel 基础上融合 bias add 和 activation
  - tl.load 加载 bias vector
  - 在 store 前应用 activation
"""

import torch
import triton
import triton.language as tl


def gelu(x):
    """Triton GELU (tanh 近似)"""
    # TODO: 用 tl.libdevice 或手写实现
    # return 0.5 * x * (1.0 + tl.libdevice.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    return x  # placeholder


@triton.jit
def mlp_first_layer_kernel(
    X_ptr, W1_ptr, bias_ptr, H_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_hm, stride_hn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # TODO: 实现 fused matmul + bias + GELU
    # 提示: 与 matmul_kernel 类似, 在 store 前加上 bias 和 gelu
    pass


def mlp_first_layer_triton(
    X: torch.Tensor, W1: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Triton fused MLP first layer: H = GELU(X @ W1 + bias)"""
    assert X.is_cuda and W1.is_cuda and bias.is_cuda
    M, K = X.shape
    K2, N = W1.shape
    assert K == K2

    H = torch.empty((M, N), device=X.device, dtype=X.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    # TODO: 启动 kernel

    return H


if __name__ == "__main__":
    M, K, N = 512, 768, 3072
    X = torch.randn(M, K, device='cuda', dtype=torch.float32)
    W1 = torch.randn(K, N, device='cuda', dtype=torch.float32)
    bias = torch.randn(N, device='cuda', dtype=torch.float32)

    H_triton = mlp_first_layer_triton(X, W1, bias)
    H_torch = torch.nn.functional.gelu(X @ W1 + bias)

    if H_triton is not None:
        max_err = (H_triton - H_torch).abs().max().item()
        print(f"Triton MLP first layer: max_error={max_err:.6f}")
    else:
        print("Triton MLP first layer: kernel not implemented yet")
