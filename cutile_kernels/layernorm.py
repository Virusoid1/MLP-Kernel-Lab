"""
cuTile LayerNorm kernel (forward + backward)

forward:  y = gamma * (x - mean) / sqrt(var + eps) + beta
backward: d_x, d_gamma, d_beta

v2 4.x 修复（Blackwell sm120 实测暴露）：
  cuTile 对超出 N 的列 load 使用 ZERO 填充，但 sm120 上填充路径返回垃圾数据
  （Ampere tile=256 恰好整除测试 N=256 掩盖了问题；Blackwell tile=512 暴露）。
  修复：N 不整除 TILE 时在 Python 侧将 x/gamma/beta/dy 补零到 N_pad（TILE 整数倍），
  kernel 全 tile 加载全有效列；mean/var/c1/c2 除以传入的真实 N（real_N），
  输出切回前 N 列。对任何 N 均不越界、不依赖 ZERO 填充数值。
"""

import cuda.tile as ct
import torch
from math import ceil
from triton_kernels.gpu_utils import get_arch_params


@ct.kernel
def layernorm_forward_kernel(X, Gamma, Beta, Y, Mean, Rstd,
                              B: ct.Constant[int], N: ct.Constant[int],
                              TILE: ct.Constant[int], eps: ct.Constant[float],
                              REAL_N: ct.Constant[int]):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    # 计算 mean
    mean_acc = ct.zeros((1, TILE), dtype=ct.float32)
    for j in range(num_tiles):
        x_tile = ct.load(X, index=(row, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
        mean_acc = mean_acc + x_tile
    mean_val = ct.sum(mean_acc, axis=1, keepdims=True) / REAL_N  # (1, 1)

    # 计算 variance：var = E[x^2] - mean^2（零填充列贡献 0^2，精确等价真实方差；
    # 若用 centered 求和，(0-mean)^2 会污染方差 —— Blackwell 实测暴露）
    var_acc = ct.zeros((1, TILE), dtype=ct.float32)
    for j in range(num_tiles):
        x_tile = ct.load(X, index=(row, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
        var_acc = var_acc + x_tile * x_tile
    ex2_val = ct.sum(var_acc, axis=1, keepdims=True) / REAL_N  # (1, 1)
    var_val = ex2_val - mean_val * mean_val  # (1, 1)
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


def _launch_forward(x, gamma, beta, eps, B, N, N_pad, TILE):
    y = torch.empty_like(x)
    mean = torch.empty(B, device=x.device, dtype=x.dtype)
    rstd = torch.empty(B, device=x.device, dtype=x.dtype)
    grid = (B, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, layernorm_forward_kernel,
              (x, gamma, beta, y, mean, rstd, B, N_pad, TILE, eps, N))
    return y, mean, rstd


def layernorm_forward(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (y, mean, rstd)。x: (B, N)"""
    B, N = x.shape
    TILE = get_arch_params()["cutile_layernorm_tile"]
    N_pad = ((N + TILE - 1) // TILE) * TILE
    if N_pad == N:
        return _launch_forward(x, gamma, beta, eps, B, N, N, TILE)
    # 补零到 TILE 整数倍（cuTile sm120 ZERO 填充不可靠）
    xp = torch.zeros(B, N_pad, device=x.device, dtype=x.dtype)
    xp[:, :N] = x
    gp = torch.zeros(N_pad, device=gamma.device, dtype=gamma.dtype)
    gp[:N] = gamma
    bp = torch.zeros(N_pad, device=beta.device, dtype=beta.dtype)
    bp[:N] = beta
    yp, mean, rstd = _launch_forward(xp, gp, bp, eps, B, N, N_pad, TILE)
    y = torch.empty_like(x)
    y.copy_(yp[:, :N])
    return y, mean, rstd


@ct.kernel
def layernorm_backward_kernel(DY, X, Gamma, Mean, Rstd, DX, DGamma, DBeta,
                               B: ct.Constant[int], N: ct.Constant[int],
                               TILE: ct.Constant[int], REAL_N: ct.Constant[int]):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    mean_val_1d = ct.load(Mean, index=(row,), shape=(1,))
    rstd_val_1d = ct.load(Rstd, index=(row,), shape=(1,))
    mean_val = ct.reshape(mean_val_1d, (1, 1))
    rstd_val = ct.reshape(rstd_val_1d, (1, 1))

    # 第一遍：计算 c1 = mean(dy*gamma), c2 = mean(dy*gamma*x_hat)（除以 REAL_N）
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

    c1 = ct.sum(c1_acc, axis=1, keepdims=True) / REAL_N  # (1, 1)
    c2 = ct.sum(c2_acc, axis=1, keepdims=True) / REAL_N  # (1, 1)

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


def _launch_backward(dy, x, gamma, mean, rstd, B, N, N_pad, TILE):
    dx = torch.empty_like(x)
    d_gamma = torch.zeros_like(gamma)
    d_beta = torch.zeros_like(gamma)
    grid = (B, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, layernorm_backward_kernel,
              (dy, x, gamma, mean, rstd, dx, d_gamma, d_beta, B, N_pad, TILE, N))
    return dx, d_gamma, d_beta


def layernorm_backward(
    dy: torch.Tensor, x: torch.Tensor,
    gamma: torch.Tensor,
    mean: torch.Tensor, rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (d_x, d_gamma, d_beta)。"""
    B, N = x.shape
    TILE = get_arch_params()["cutile_layernorm_tile"]
    N_pad = ((N + TILE - 1) // TILE) * TILE
    if N_pad == N:
        return _launch_backward(dy, x, gamma, mean, rstd, B, N, N, TILE)
    dyp = torch.zeros(B, N_pad, device=dy.device, dtype=dy.dtype)
    dyp[:, :N] = dy
    xp = torch.zeros(B, N_pad, device=x.device, dtype=x.dtype)
    xp[:, :N] = x
    gp = torch.zeros(N_pad, device=gamma.device, dtype=gamma.dtype)
    gp[:N] = gamma
    dxp, dg_pad, db_pad = _launch_backward(dyp, xp, gp, mean, rstd, B, N, N_pad, TILE)
    dx = torch.empty_like(x)
    dx.copy_(dxp[:, :N])
    return dx, dg_pad[:N], db_pad[:N]
