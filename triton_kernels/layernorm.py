"""
Triton LayerNorm kernel (forward + backward)

forward:  y = gamma * (x - mean) / sqrt(var + eps) + beta
backward: d_x, d_gamma, d_beta
"""

import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_forward_kernel(
    X_ptr, Y_ptr,
    Gamma_ptr, Beta_ptr,
    Mean_ptr, Rstd_ptr,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + row * stride + cols, mask=mask, other=0.0)

    # mean
    x_sum = tl.sum(x, axis=0)
    mean = x_sum / N

    # variance
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # normalize + affine
    x_hat = x_centered * rstd
    gamma = tl.load(Gamma_ptr + cols, mask=mask, other=1.0)
    beta = tl.load(Beta_ptr + cols, mask=mask, other=0.0)
    y = gamma * x_hat + beta

    tl.store(Y_ptr + row * stride + cols, y, mask=mask)
    tl.store(Mean_ptr + row, mean)
    tl.store(Rstd_ptr + row, rstd)


@triton.jit
def layernorm_backward_kernel(
    DY_ptr, X_ptr,
    Gamma_ptr,
    Mean_ptr, Rstd_ptr,
    DX_ptr,
    DGamma_ptr, DBeta_ptr,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    dy = tl.load(DY_ptr + row * stride + cols, mask=mask, other=0.0)
    x = tl.load(X_ptr + row * stride + cols, mask=mask, other=0.0)
    gamma = tl.load(Gamma_ptr + cols, mask=mask, other=1.0)
    mean = tl.load(Mean_ptr + row)
    rstd = tl.load(Rstd_ptr + row)

    x_hat = (x - mean) * rstd

    # d_x = rstd * (dy * gamma - mean(dy*gamma) - x_hat * mean(dy*gamma*x_hat))
    dy_gamma = dy * gamma
    c1 = tl.sum(dy_gamma, axis=0) / N
    c2 = tl.sum(dy_gamma * x_hat, axis=0) / N
    dx = rstd * (dy_gamma - c1 - x_hat * c2)

    tl.store(DX_ptr + row * stride + cols, dx, mask=mask)

    # d_gamma, d_beta: atomic add 跨行累加
    tl.atomic_add(DGamma_ptr + cols, dy * x_hat, mask=mask)
    tl.atomic_add(DBeta_ptr + cols, dy, mask=mask)


def layernorm_forward(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (y, mean, rstd)。"""
    assert x.is_contiguous()
    B, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)

    y = torch.empty_like(x)
    mean = torch.empty(B, device=x.device, dtype=x.dtype)
    rstd = torch.empty(B, device=x.device, dtype=x.dtype)

    grid = (B,)
    layernorm_forward_kernel[grid](
        x, y, gamma, beta, mean, rstd,
        x.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y, mean, rstd


def layernorm_backward(
    dy: torch.Tensor, x: torch.Tensor,
    gamma: torch.Tensor,
    mean: torch.Tensor, rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (d_x, d_gamma, d_beta)。"""
    assert dy.is_contiguous()
    B, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)

    dx = torch.empty_like(x)
    d_gamma = torch.zeros_like(gamma)
    d_beta = torch.zeros_like(gamma)

    grid = (B,)
    layernorm_backward_kernel[grid](
        dy, x, gamma, mean, rstd,
        dx, d_gamma, d_beta,
        x.stride(0), N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dx, d_gamma, d_beta
