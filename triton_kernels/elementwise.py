"""
Triton element-wise 算子：BiasAdd、ReLU、GELU、SiLU

访存密集型算子，瓶颈在数据搬运。融合 kernel 减少全局内存往返。
autotune 自动选择最优 BLOCK_SIZE 以适配不同 GPU 架构。
"""

import torch
import triton
import triton.language as tl


# elementwise kernel 的 autotune 配置
_ELEMENTWISE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
]


@triton.jit
def bias_add_kernel(
    input_ptr, bias_ptr, output_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """BiasAdd kernel。每个 program 处理一行。"""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x = tl.load(input_ptr + row_idx * input_row_stride + col_offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)

    tl.store(output_ptr + row_idx * output_row_stride + col_offsets, x + b, mask=mask)


def bias_add(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Triton BiasAdd。x: (M, N)  bias: (N,) -> (M, N)"""
    assert x.dim() == 2
    assert bias.dim() == 1 and bias.shape[0] == x.shape[1]
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)

    bias_add_kernel[(M,)](
        x, bias, output,
        N,
        x.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


@triton.autotune(configs=_ELEMENTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def relu_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """ReLU kernel。逐元素 output = max(0, input)。"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, tl.where(x > 0, x, 0.0), mask=mask)


def relu(x: torch.Tensor) -> torch.Tensor:
    """Triton ReLU。支持任意形状。"""
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    relu_kernel[grid](x, output, n)
    return output


@triton.autotune(configs=_ELEMENTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def gelu_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    GELU kernel（tanh 近似，与 PyTorch approximate='tanh' 一致）。
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    用 tl.tanh 直接计算。
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    output = 0.5 * x * (1.0 + tanh_inner)

    tl.store(output_ptr + offsets, output, mask=mask)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Triton GELU（tanh 近似）。支持任意形状。"""
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    gelu_kernel[grid](x, output, n)
    return output


@triton.autotune(configs=_ELEMENTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def silu_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SiLU kernel。SiLU(x) = x * sigmoid(x)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, x * tl.sigmoid(x), mask=mask)


def silu(x: torch.Tensor) -> torch.Tensor:
    """Triton SiLU。支持任意形状。"""
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    silu_kernel[grid](x, output, n)
    return output


@triton.jit
def bias_add_relu_kernel(
    input_ptr, bias_ptr, output_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """融合 BiasAdd + ReLU：一个 kernel 完成 bias 加法 + ReLU 激活。"""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x = tl.load(input_ptr + row_idx * input_row_stride + col_offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)

    y = x + b
    tl.store(output_ptr + row_idx * output_row_stride + col_offsets,
             tl.where(y > 0, y, 0.0), mask=mask)


def bias_add_relu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """融合 BiasAdd + ReLU。x: (M, N)  bias: (N,) -> ReLU(x + bias)"""
    assert x.dim() == 2
    assert bias.dim() == 1 and bias.shape[0] == x.shape[1]
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)

    bias_add_relu_kernel[(M,)](
        x, bias, output,
        N,
        x.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
