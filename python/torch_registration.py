"""
PyTorch 自定义算子集成（P1，v2 计划）。

完成 torch.library 完整注册链路:
  - schema (LIB.define)
  - CPU / CUDA 实现 (LIB.impl, 挂 "mlp_kernel::swiglu")
  - FakeTensor / meta (LIB.impl "Meta")
  - autograd (torch.library.register_autograd)
  - torch.library.opcheck / gradcheck / torch.compile 验证

signature op: fused SwiGLU (gate, up) -> SiLU(gate)*up
"""

from __future__ import annotations

import torch

LIBRARY_NAME = "mlp_kernel"
OP_NAME = "mlp_kernel::swiglu"


def _silu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up


# schema 定义
LIB = torch.library.Library(LIBRARY_NAME, "DEF")  # type: ignore[attr-defined]
LIB.define("swiglu(Tensor gate, Tensor up) -> Tensor")


# 公共调用入口: 由 Python 实现的自动分发 (Local 定义会自动派发到 registered impl)
def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU: SiLU(gate) * up. 注册后走 dispatch 到 CPU/CUDA impl。"""
    return torch.ops.mlp_kernel.swiglu.default(gate, up)


# CPU impl
def _swiglu_cpu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return _silu_reference(gate, up)


LIB.impl("swiglu", _swiglu_cpu, "CPU")


# CUDA impl (优先复用 Triton kernel)
def _swiglu_cuda(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    try:
        from triton_kernels.swiglu_triton import swiglu_triton
        if gate.is_cuda and up.is_cuda and gate.dtype in (torch.float32, torch.float16):
            return swiglu_triton(gate, up)
    except Exception:
        pass
    return _silu_reference(gate, up)


LIB.impl("swiglu", _swiglu_cuda, "CUDA")


# Meta (FakeTensor) impl
def _swiglu_meta(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.empty(gate.shape, dtype=gate.dtype, device="meta")


LIB.impl("swiglu", _swiglu_meta, "Meta")


# autograd
def _setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1])


def _backward(ctx, grad_output):
    gate, up = ctx.saved_tensors
    sig = torch.sigmoid(gate)
    grad_gate = grad_output * (sig * (1 + gate * (1 - sig))) * up
    grad_up = grad_output * torch.nn.functional.silu(gate)
    return grad_gate, grad_up


torch.library.register_autograd(  # type: ignore[attr-defined]
    OP_NAME, _backward, setup_context=_setup,
)


def opcheck() -> None:
    """torch.library.opcheck: schema/meta/fake/autograd/dispatch 一致性。"""
    from torch.library import opcheck as _opcheck

    gate = torch.randn(4, 64, device="cuda", dtype=torch.float32)
    up = torch.randn(4, 64, device="cuda", dtype=torch.float32)
    _opcheck(torch.ops.mlp_kernel.swiglu.default, (gate, up))


def gradcheck_swiglu(device: str = "cuda", dtype: torch.dtype = torch.float64) -> bool:
    from torch.autograd import gradcheck

    gate = torch.randn(3, 16, device=device, dtype=dtype, requires_grad=True)
    up = torch.randn(3, 16, device=device, dtype=dtype, requires_grad=True)

    def fn(g, u):
        return swiglu(g, u).sum()

    return bool(gradcheck(fn, (gate, up), eps=1e-6, atol=1e-4, rtol=1e-3))


def compile_smoke(device: str = "cuda") -> torch.Tensor:
    gate = torch.randn(8, 32, device=device)
    up = torch.randn(8, 32, device=device)
    fn = torch.compile(lambda g, u: swiglu(g, u) + swiglu(u, g))
    return fn(gate, up)


def register_ops() -> None:
    """幂等: 验证已注册 + 触发 (import 时已完成注册, 此处仅自我检查)。"""
    op = torch.ops.mlp_kernel.swiglu
    assert op is not None, "swiglu op not registered"
    return None
