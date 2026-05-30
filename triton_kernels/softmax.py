"""
Triton Softmax kernel

逐行 softmax，减去行最大值防止 exp 溢出。
每个 program 处理一行，支持任意列数（自动 padding 到 2 的幂）。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """逐行 softmax kernel。每个 program 处理一行。"""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(
        input_ptr + row_idx * input_row_stride + col_offsets,
        mask=mask, other=-float("inf"),
    )

    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)

    tl.store(
        output_ptr + row_idx * output_row_stride + col_offsets,
        numerator / denominator,
        mask=mask,
    )


def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Triton Softmax。x: (M, N) -> (M, N)
    数值稳定：减去行最大值后 exp。
    """
    assert x.dim() == 2
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)

    softmax_kernel[(M,)](
        x, output,
        N,
        x.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
