"""
cuTile element-wise 算子：BiasAdd、ReLU、GELU、SiLU、融合 BiasAdd+ReLU
"""

import cuda.tile as ct
import torch
from math import ceil


@ct.kernel
def bias_add_kernel(X, Bias, Out, N: ct.Constant[int], TILE: ct.Constant[int]):
    row = ct.bid(0)
    # X: (M, N), index=(row, 0) 加载整行
    x = ct.load(X, index=(row, 0), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(Bias, index=(0, 0), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
    result = x + b
    ct.store(Out, index=(row, 0), tile=result)


def bias_add(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """cuTile BiasAdd。x: (M, N), bias: (N,) -> (M, N)"""
    M, N = x.shape
    out = torch.empty_like(x)
    TILE = N if (N & (N - 1)) == 0 else 1 << (N - 1).bit_length()

    grid = (M, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, bias_add_kernel,
              (x, bias.reshape(1, N), out, N, TILE))
    return out


@ct.kernel
def relu_kernel(X, Out, N: ct.Constant[int], TILE: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(X, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(Out, index=(pid,), tile=ct.where(x > 0.0, x, 0.0))


def relu(x: torch.Tensor) -> torch.Tensor:
    """cuTile ReLU。"""
    orig_shape = x.shape
    x_flat = x.reshape(-1)
    n = x_flat.shape[0]
    TILE = 512
    out_flat = torch.empty_like(x_flat)

    grid = (ceil(n / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, relu_kernel,
              (x_flat, out_flat, n, TILE))
    return out_flat.reshape(orig_shape)


@ct.kernel
def gelu_kernel(X, Out, N: ct.Constant[int], TILE: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(X, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    result = 0.5 * x * (1.0 + ct.tanh(inner))
    ct.store(Out, index=(pid,), tile=result)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """cuTile GELU（tanh 近似）。"""
    orig_shape = x.shape
    x_flat = x.reshape(-1)
    n = x_flat.shape[0]
    TILE = 512
    out_flat = torch.empty_like(x_flat)

    grid = (ceil(n / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, gelu_kernel,
              (x_flat, out_flat, n, TILE))
    return out_flat.reshape(orig_shape)


@ct.kernel
def silu_kernel(X, Out, N: ct.Constant[int], TILE: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(X, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    sigmoid_x = 1.0 / (1.0 + ct.exp(-x))
    ct.store(Out, index=(pid,), tile=x * sigmoid_x)


def silu(x: torch.Tensor) -> torch.Tensor:
    """cuTile SiLU。"""
    orig_shape = x.shape
    x_flat = x.reshape(-1)
    n = x_flat.shape[0]
    TILE = 512
    out_flat = torch.empty_like(x_flat)

    grid = (ceil(n / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, silu_kernel,
              (x_flat, out_flat, n, TILE))
    return out_flat.reshape(orig_shape)


@ct.kernel
def bias_add_relu_kernel(X, Bias, Out, N: ct.Constant[int], TILE: ct.Constant[int]):
    row = ct.bid(0)
    x = ct.load(X, index=(row, 0), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(Bias, index=(0, 0), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)
    y = x + b
    ct.store(Out, index=(row, 0), tile=ct.where(y > 0.0, y, 0.0))


def bias_add_relu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """融合 BiasAdd + ReLU。x: (M, N), bias: (N,) -> ReLU(x + bias)"""
    M, N = x.shape
    out = torch.empty_like(x)
    TILE = N if (N & (N - 1)) == 0 else 1 << (N - 1).bit_length()

    grid = (M, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, bias_add_relu_kernel,
              (x, bias.reshape(1, N), out, N, TILE))
    return out
