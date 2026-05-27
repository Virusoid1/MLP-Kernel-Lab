"""
Triton fused SwiGLU kernel

hidden = SiLU(gate) * up
用于 Llama/Qwen 系列 FFN
autotune 自动选择最优 BLOCK_SIZE 以适配不同 GPU 架构。
"""

import torch
import triton
import triton.language as tl

_SWIGLU_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
]


@triton.autotune(configs=_SWIGLU_CONFIGS, key=["n_elements"])
@triton.jit
def swiglu_kernel(
    gate_ptr, up_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SwiGLU kernel。
    output = SiLU(gate) * up = gate * sigmoid(gate) * up
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    output = silu_gate * up

    tl.store(output_ptr + offsets, output, mask=mask)


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Triton fused SwiGLU。gate/up: 同形状张量。"""
    assert gate.is_cuda and up.is_cuda
    assert gate.shape == up.shape
    output = torch.empty_like(gate)
    n_elements = gate.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    swiglu_kernel[grid](
        gate, up, output,
        n_elements,
    )
    return output
