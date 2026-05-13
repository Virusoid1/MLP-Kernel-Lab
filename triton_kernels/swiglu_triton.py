"""
Day 11: Triton fused SwiGLU kernel

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
    # TODO: 实现 Triton SwiGLU
    #
    # 提示:
    #   pid = tl.program_id(0)
    #   offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    #   mask = offs < n_elements
    #
    #   gate = tl.load(gate_ptr + offs, mask=mask)
    #   up = tl.load(up_ptr + offs, mask=mask)
    #
    #   sigmoid_gate = tl.sigmoid(gate)   # 或 1 / (1 + tl.exp(-gate))
    #   silu_gate = gate * sigmoid_gate
    #   output = silu_gate * up
    #
    #   tl.store(output_ptr + offs, output, mask=mask)
    pass


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Triton fused SwiGLU"""
    assert gate.is_cuda and up.is_cuda
    assert gate.shape == up.shape
    output = torch.empty_like(gate)
    n_elements = gate.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # TODO: 启动 kernel

    return output


if __name__ == "__main__":
    M, N = 512, 4096
    gate = torch.randn(M, N, device='cuda', dtype=torch.float32)
    up = torch.randn(M, N, device='cuda', dtype=torch.float32)

    out_triton = swiglu_triton(gate, up)
    out_torch = torch.nn.functional.silu(gate) * up

    if out_triton is not None:
        max_err = (out_triton - out_torch).abs().max().item()
        print(f"Triton SwiGLU: max_error={max_err:.6f}")
    else:
        print("Triton SwiGLU: kernel not implemented yet")
