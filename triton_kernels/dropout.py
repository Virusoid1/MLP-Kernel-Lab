"""
Triton dropout kernel

Dropout：output_i = input_i * mask_i / (1 - p)
- mask_i ~ Bernoulli(1 - p)
- inverted dropout 保持期望值不变
"""

import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    """Dropout kernel，每个 program 处理一行。"""
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x = tl.load(row_start + col_offsets, mask=mask, other=0.0)

    rand = tl.rand(seed, row_idx * n_cols + col_offsets)
    keep = rand > p

    scale = 1.0 / (1.0 - p)
    output = tl.where(keep, x * scale, 0.0)

    output_row_start = output_ptr + row_idx * input_row_stride
    tl.store(output_row_start + col_offsets, output, mask=mask)


def triton_dropout(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Triton dropout。x: 2D 张量，p: 置零概率。"""
    assert x.dim() == 2, "Only 2D input is supported"
    assert 0.0 <= p < 1.0, "p must be in [0, 1)"
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    seed = torch.randint(0, 2**31, (1,)).item()

    grid = (n_rows,)
    dropout_kernel[grid](
        x, output,
        x.stride(0), output.stride(0),
        n_cols,
        p, seed,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
