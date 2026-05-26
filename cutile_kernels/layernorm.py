"""
cuTile LayerNorm kernel (forward + backward)

forward:  y = gamma * (x - mean) / sqrt(var + eps) + beta
backward: d_x, d_gamma, d_beta
"""

import cuda.tile as ct
import torch
from math import ceil


@ct.kernel
def layernorm_forward_kernel(X, Gamma, Beta, Y, Mean, Rstd,
                              B: ct.Constant[int], N: ct.Constant[int],
                              TILE: ct.Constant[int], eps: ct.Constant[float]):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    # 计算 mean
    mean_acc = ct.zeros((1, TILE), dtype=ct.float32)
    for j in range(num_tiles):
        x_tile = ct.load(X, index=(row, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
        mean_acc = mean_acc + x_tile
    mean_val = ct.sum(mean_acc, axis=1, keepdims=True) / N  # (1, 1)

    # 计算 variance
    var_acc = ct.zeros((1, TILE), dtype=ct.float32)
    for j in range(num_tiles):
        x_tile = ct.load(X, index=(row, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
        centered = x_tile - mean_val
        var_acc = var_acc + centered * centered
    var_val = ct.sum(var_acc, axis=1, keepdims=True) / N  # (1, 1)
    rstd_val = 1.0 / ct.sqrt(var_val + eps)  # (1, 1)

    ct.store(Mean, index=(row,), tile=ct.reshape(mean_val, (1,)))
    ct.store(Rstd, index=(row,), tile=ct.reshape(rstd_val, (1,)))

    # normalize + affine
    for j in range(num_tiles):
        x_tile = ct.load(X, index=(row, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
        gamma_tile = ct.load(Gamma, index=(j,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
        beta_tile = ct.load(Beta, index=(j,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
        x_hat = (x_tile - mean_val) * rstd_val
        y = x_hat * gamma_tile + beta_tile
        ct.store(Y, index=(row, j), tile=y)


def layernorm_forward(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (y, mean, rstd)。x: (B, N)"""
    B, N = x.shape
    TILE = 256
    y = torch.empty_like(x)
    mean = torch.empty(B, device=x.device, dtype=x.dtype)
    rstd = torch.empty(B, device=x.device, dtype=x.dtype)

    grid = (B, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, layernorm_forward_kernel,
              (x, gamma, beta, y, mean, rstd, B, N, TILE, eps))
    return y, mean, rstd


@ct.kernel
def layernorm_backward_kernel(DY, X, Gamma, Mean, Rstd, DX, DGamma, DBeta,
                               B: ct.Constant[int], N: ct.Constant[int],
                               TILE: ct.Constant[int]):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    mean_val_1d = ct.load(Mean, index=(row,), shape=(1,))
    rstd_val_1d = ct.load(Rstd, index=(row,), shape=(1,))
    mean_val = ct.reshape(mean_val_1d, (1, 1))
    rstd_val = ct.reshape(rstd_val_1d, (1, 1))

    # 第一遍：计算 c1 = mean(dy*gamma), c2 = mean(dy*gamma*x_hat)
    c1_acc = ct.zeros((1, TILE), dtype=ct.float32)
    c2_acc = ct.zeros((1, TILE), dtype=ct.float32)
    for j in range(num_tiles):
        dy_tile = ct.load(DY, index=(row, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
        x_tile = ct.load(X, index=(row, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
        gamma_tile = ct.load(Gamma, index=(j,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
        x_hat = (x_tile - mean_val) * rstd_val
        dy_gamma = dy_tile * gamma_tile
        c1_acc = c1_acc + dy_gamma
        c2_acc = c2_acc + dy_gamma * x_hat

    c1 = ct.sum(c1_acc, axis=1, keepdims=True) / N  # (1, 1)
    c2 = ct.sum(c2_acc, axis=1, keepdims=True) / N  # (1, 1)

    # 第二遍：计算 dx 和累加 d_gamma/d_beta
    for j in range(num_tiles):
        dy_tile = ct.load(DY, index=(row, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
        x_tile = ct.load(X, index=(row, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
        gamma_tile = ct.load(Gamma, index=(j,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
        x_hat = (x_tile - mean_val) * rstd_val
        dy_gamma = dy_tile * gamma_tile
        dx = rstd_val * (dy_gamma - c1 - x_hat * c2)

        ct.store(DX, index=(row, j), tile=dx)
        # scatter atomic_add: indices = j * TILE + arange(TILE)
        offsets = j * TILE + ct.arange(TILE, dtype=ct.int32)
        dgamma_update = ct.reshape(dy_tile * x_hat, (TILE,))
        dbeta_update = ct.reshape(dy_tile, (TILE,))
        ct.atomic_add(DGamma, offsets, dgamma_update)
        ct.atomic_add(DBeta, offsets, dbeta_update)


def layernorm_backward(
    dy: torch.Tensor, x: torch.Tensor,
    gamma: torch.Tensor,
    mean: torch.Tensor, rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (d_x, d_gamma, d_beta)。"""
    B, N = x.shape
    TILE = 256
    dx = torch.empty_like(x)
    d_gamma = torch.zeros_like(gamma)
    d_beta = torch.zeros_like(gamma)

    grid = (B, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, layernorm_backward_kernel,
              (dy, x, gamma, mean, rstd, dx, d_gamma, d_beta, B, N, TILE))
    return dx, d_gamma, d_beta
