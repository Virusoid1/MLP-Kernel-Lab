"""
Triton 2D 池化算子：MaxPool2D、AvgPool2D

输入格式 NCHW。每个 program 处理一个输出像素 (n, c, h_out, w_out)。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    input_ptr, output_ptr,
    N, C, H, W,
    H_out, W_out,
    kernel_size, stride, padding,
    BLOCK_HK: tl.constexpr, BLOCK_WK: tl.constexpr,
):
    """MaxPool2D kernel。每个 program 处理一个输出像素。"""
    idx = tl.program_id(0)
    total = N * C * H_out * W_out
    if idx >= total:
        return

    # 从 1D index 还原 (n, c, h_out, w_out)
    w_out = idx % W_out
    h_out = (idx // W_out) % H_out
    nc = idx // (H_out * W_out)
    c = nc % C
    n = nc // C

    # 计算输入窗口起点
    h_start = h_out * stride - padding
    w_start = w_out * stride - padding

    max_val = -float("inf")

    for kh in range(BLOCK_HK):
        h_in = h_start + kh
        if h_in < 0 or h_in >= H:
            continue
        for kw in range(BLOCK_WK):
            w_in = w_start + kw
            if w_in < 0 or w_in >= W:
                continue
            offset = ((n * C + c) * H + h_in) * W + w_in
            val = tl.load(input_ptr + offset)
            max_val = tl.where(val > max_val, val, max_val)

    out_offset = ((n * C + c) * H_out + h_out) * W_out + w_out
    tl.store(output_ptr + out_offset, max_val)


def maxpool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int | None = None,
    padding: int = 0,
) -> torch.Tensor:
    """
    Triton MaxPool2D。x: (N, C, H, W) -> (N, C, H_out, W_out)
    H_out = (H + 2*padding - kernel_size) / stride + 1
    """
    assert x.dim() == 4
    if stride is None:
        stride = kernel_size

    N, C, H, W = x.shape
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    output = torch.full((N, C, H_out, W_out), -float("inf"), device=x.device, dtype=x.dtype)

    total = N * C * H_out * W_out
    BLOCK_HK = triton.next_power_of_2(kernel_size)
    BLOCK_WK = triton.next_power_of_2(kernel_size)

    maxpool2d_kernel[(total,)](
        x, output,
        N, C, H, W,
        H_out, W_out,
        kernel_size, stride, padding,
        BLOCK_HK=BLOCK_HK, BLOCK_WK=BLOCK_WK,
    )
    return output


@triton.jit
def avgpool2d_kernel(
    input_ptr, output_ptr,
    N, C, H, W,
    H_out, W_out,
    kernel_size, stride, padding,
    BLOCK_HK: tl.constexpr, BLOCK_WK: tl.constexpr,
):
    """AvgPool2D kernel。每个 program 处理一个输出像素。"""
    idx = tl.program_id(0)
    total = N * C * H_out * W_out
    if idx >= total:
        return

    w_out = idx % W_out
    h_out = (idx // W_out) % H_out
    nc = idx // (H_out * W_out)
    c = nc % C
    n = nc // C

    h_start = h_out * stride - padding
    w_start = w_out * stride - padding

    acc = 0.0
    count = 0

    for kh in range(BLOCK_HK):
        h_in = h_start + kh
        if h_in < 0 or h_in >= H:
            continue
        for kw in range(BLOCK_WK):
            w_in = w_start + kw
            if w_in < 0 or w_in >= W:
                continue
            offset = ((n * C + c) * H + h_in) * W + w_in
            val = tl.load(input_ptr + offset)
            acc += val
            count += 1

    out_offset = ((n * C + c) * H_out + h_out) * W_out + w_out
    tl.store(output_ptr + out_offset, acc / count)


def avgpool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int | None = None,
    padding: int = 0,
) -> torch.Tensor:
    """
    Triton AvgPool2D。x: (N, C, H, W) -> (N, C, H_out, W_out)
    只计算有效窗口内的元素平均（不统计 padding 区域）。
    """
    assert x.dim() == 4
    if stride is None:
        stride = kernel_size

    N, C, H, W = x.shape
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    output = torch.zeros((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    total = N * C * H_out * W_out
    BLOCK_HK = triton.next_power_of_2(kernel_size)
    BLOCK_WK = triton.next_power_of_2(kernel_size)

    avgpool2d_kernel[(total,)](
        x, output,
        N, C, H, W,
        H_out, W_out,
        kernel_size, stride, padding,
        BLOCK_HK=BLOCK_HK, BLOCK_WK=BLOCK_WK,
    )
    return output
