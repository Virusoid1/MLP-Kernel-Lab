"""
Triton Conv2D (im2col + matmul)

将 im2col 展开和矩阵乘法拆为两个阶段：
1. im2col: 将输入 (N,C,H,W) 展开为 (N*H_out*W_out, C*KH*KW) 的矩阵
2. matmul: 调用已有的 tiled_matmul 完成 col @ weight.T 得到输出

这种分解方式清晰展示卷积与矩阵乘法的关系，便于对比优化。
"""

import torch
import triton
import triton.language as tl

from triton_kernels.matmul import tiled_matmul


@triton.jit
def im2col_kernel(
    input_ptr, col_ptr,
    N, C, H, W,
    H_out, W_out,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    im2col kernel。每个 program 处理 col 矩阵的一行（对应一个输出像素位置）。
    col: (N*H_out*W_out, C*KH*KW)
    """
    idx = tl.program_id(0)
    total_rows = N * H_out * W_out
    if idx >= total_rows:
        return

    # 从 1D index 还原 (n, h_out, w_out)
    w_out = idx % W_out
    h_out = (idx // W_out) % H_out
    n = idx // (H_out * W_out)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_len = C * KH * KW
    col_mask = col_offsets < col_len

    # 将 col_offsets 解码为 (c, kh, kw)
    kw_idx = col_offsets % KW
    kh_idx = (col_offsets // KW) % KH
    c_idx = col_offsets // (KH * KW)

    # 对应输入坐标
    h_in = h_out * stride_h - pad_h + kh_idx
    w_in = w_out * stride_w - pad_w + kw_idx

    # 边界检查
    valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)

    # 计算输入偏移
    input_offsets = (n * C + c_idx) * H * W + h_in * W + w_in
    vals = tl.load(input_ptr + input_offsets, mask=valid & col_mask, other=0.0)

    # 写入 col 矩阵
    row_base = idx * col_len
    tl.store(col_ptr + row_base + col_offsets, vals, mask=col_mask)


def im2col(
    x: torch.Tensor,
    kh: int, kw: int,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """
    Triton im2col。将 (N,C,H,W) 展开为 (N*H_out*W_out, C*KH*KW)。
    """
    N, C, H, W = x.shape
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1

    col_len = C * kh * kw
    col = torch.zeros(N * H_out * W_out, col_len, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(col_len)

    total_rows = N * H_out * W_out
    im2col_kernel[(total_rows,)](
        x, col,
        N, C, H, W,
        H_out, W_out,
        kh, kw,
        stride, stride,
        padding, padding,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return col


def conv2d_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """
    Triton Conv2D（im2col + matmul）。
    input:  (N, C_in, H, W)
    weight: (C_out, C_in, KH, KW)
    bias:   (C_out,) 或 None
    -> (N, C_out, H_out, W_out)
    """
    assert input.dim() == 4
    assert weight.dim() == 4

    N, C_in, H, W = input.shape
    C_out, _, KH, KW = weight.shape

    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    # im2col: (N*H_out*W_out, C_in*KH*KW)
    col = im2col(input, KH, KW, stride=stride, padding=padding)

    # weight 展平为 (C_out, C_in*KH*KW)，转置为 (C_in*KH*KW, C_out)
    w_flat = weight.reshape(C_out, -1).T.contiguous()

    # matmul: (N*H_out*W_out, C_in*KH*KW) @ (C_in*KH*KW, C_out) -> (N*H_out*W_out, C_out)
    output = tiled_matmul(col, w_flat)

    # 加 bias
    if bias is not None:
        output = output + bias.unsqueeze(0)

    # reshape 为 (N, C_out, H_out, W_out)
    output = output.reshape(N, H_out * W_out, C_out).permute(0, 2, 1).reshape(N, C_out, H_out, W_out).contiguous()

    return output
