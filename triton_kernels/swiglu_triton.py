"""
Triton fused SwiGLU kernel

hidden = SiLU(gate) * up
用于 Llama/Qwen 系列 FFN
"""

import torch
import triton
import triton.language as tl


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

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    swiglu_kernel[grid](
        gate, up, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
