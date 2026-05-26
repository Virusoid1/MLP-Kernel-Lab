"""
cuTile 融合 SwiGLU kernel

SwiGLU(x) = x * sigmoid(x)
"""

import cuda.tile as ct
import torch
from math import ceil


@ct.kernel
def swiglu_kernel(X, Out, N: ct.Constant[int], TILE: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(X, index=(pid,), shape=(TILE,))
    sigmoid_x = 1.0 / (1.0 + ct.exp(-x))
    ct.store(Out, index=(pid,), tile=x * sigmoid_x)


def swiglu_cutile(x: torch.Tensor) -> torch.Tensor:
    """cuTile SwiGLU = x * sigmoid(x)。"""
    orig_shape = x.shape
    x_flat = x.reshape(-1)
    n = x_flat.shape[0]
    TILE = 512
    out_flat = torch.empty_like(x_flat)

    grid = (ceil(n / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, swiglu_kernel,
              (x_flat, out_flat, n, TILE))
    return out_flat.reshape(orig_shape)
