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




# ============================================================
# mlp_kernel::matmul  (推广 P1: swiglu -> matmul)
#   C = A @ B, 完整链: schema / CPU+CUDA / Meta / autograd
# ============================================================

LIB.define("matmul(Tensor A, Tensor B) -> Tensor")


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """C = A @ B. 注册后走 dispatch。"""
    return torch.ops.mlp_kernel.matmul.default(A, B)


def _matmul_cpu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A @ B


LIB.impl("matmul", _matmul_cpu, "CPU")


def _matmul_cuda(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # 1) 本机编译的 CUDA kernel（fp16 -> WMMA matmul_half；fp32 -> tiled_auto）
    try:
        import mlp_cuda  # type: ignore[import-not-found]
        if A.dtype == torch.float16 and B.dtype == torch.float16:
            return mlp_cuda.matmul_half(A, B)
        if A.dtype == torch.float32 and B.dtype == torch.float32:
            return mlp_cuda.matmul_tiled_auto(A, B)
    except Exception:
        pass
    # 2) Triton 兜底（fp32/fp16 同 dtype）
    try:
        from triton_kernels.matmul import tiled_matmul
        if A.dtype in (torch.float32, torch.float16) and A.dtype == B.dtype:
            return tiled_matmul(A, B)
    except Exception:
        pass
    # 3) 参考
    return A @ B


LIB.impl("matmul", _matmul_cuda, "CUDA")


def _matmul_meta(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.empty(A.shape[0], B.shape[1], dtype=A.dtype, device="meta")


LIB.impl("matmul", _matmul_meta, "Meta")


def _matmul_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1])


def _matmul_backward(ctx, grad_output):
    A, B = ctx.saved_tensors
    return grad_output @ B.t(), A.t() @ grad_output


torch.library.register_autograd(  # type: ignore[attr-defined]
    "mlp_kernel::matmul", _matmul_backward, setup_context=_matmul_setup,
)


def matmul_opcheck() -> None:
    from torch.library import opcheck as _opcheck
    A = torch.randn(4, 64, device="cuda", dtype=torch.float32)
    B = torch.randn(64, 32, device="cuda", dtype=torch.float32)
    _opcheck(torch.ops.mlp_kernel.matmul.default, (A, B))


def gradcheck_matmul(device: str = "cuda", dtype: torch.dtype = torch.float64) -> bool:
    from torch.autograd import gradcheck
    A = torch.randn(3, 16, device=device, dtype=dtype, requires_grad=True)
    B = torch.randn(16, 8, device=device, dtype=dtype, requires_grad=True)
    return bool(gradcheck(lambda a, b: matmul(a, b).sum(), (A, B), eps=1e-6, atol=1e-4, rtol=1e-3))


def matmul_compile_smoke(device: str = "cuda") -> torch.Tensor:
    # (8,32)@(32,16)=(8,16)；第二项 (16,32)@(32,8)=(16,8) 转置后同为 (8,16)
    A = torch.randn(8, 32, device=device)
    B = torch.randn(32, 16, device=device)
    return torch.compile(lambda a, b: matmul(a, b) + matmul(b.t(), a.t()).t())(A, B)


# ============================================================
# mlp_kernel::layernorm  (推广 P1: swiglu -> layernorm)
#   y = LayerNorm(x, weight, bias, eps), 单输出；mean/rstd 内部化
# ============================================================

LIB.define("layernorm(Tensor x, Tensor weight, Tensor bias, float eps) -> Tensor")


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              eps: float = 1e-5) -> torch.Tensor:
    return torch.ops.mlp_kernel.layernorm.default(x, weight, bias, eps)


def _layernorm_cpu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                   eps: float) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, (x.size(-1),), weight, bias, eps)


LIB.impl("layernorm", _layernorm_cpu, "CPU")


def _layernorm_cuda(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                    eps: float) -> torch.Tensor:
    # 2D fp32 且 mlp_cuda 可用 -> 本机 kernel；否则 PyTorch 等价参考
    try:
        import mlp_cuda  # type: ignore[import-not-found]
        if x.dim() == 2 and x.dtype == torch.float32                 and weight.dtype == torch.float32 and bias.dtype == torch.float32:
            return mlp_cuda.layernorm_forward(x, weight, bias, eps)[0]
    except Exception:
        pass
    return torch.nn.functional.layer_norm(x, (x.size(-1),), weight, bias, eps)


LIB.impl("layernorm", _layernorm_cuda, "CUDA")


def _layernorm_meta(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                    eps: float) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device="meta")


LIB.impl("layernorm", _layernorm_meta, "Meta")


def _layernorm_setup(ctx, inputs, output):
    # 保存 x / weight / eps（bias 梯度 = sum(dy)，backward 内直接算）
    ctx.save_for_backward(inputs[0], inputs[1])
    ctx.eps = inputs[3]


def _layernorm_backward(ctx, grad_y):
    x, weight = ctx.saved_tensors
    eps = ctx.eps
    N = x.size(-1)
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    rstd = 1.0 / torch.sqrt(var + eps)
    xc = x - mean
    g = grad_y * rstd * weight
    # dx = rstd*w*(g - mean_N(g)) - rstd^3*w*xc*mean_N(g*xc)
    dx = g - g.mean(-1, keepdim=True) - xc * (grad_y * xc * rstd * weight).mean(-1, keepdim=True) * rstd * rstd
    reduce_dims = tuple(range(grad_y.dim() - 1))
    dw = (grad_y * xc * rstd).sum(reduce_dims)
    db = grad_y.sum(reduce_dims)
    return dx, dw, db, None


torch.library.register_autograd(  # type: ignore[attr-defined]
    "mlp_kernel::layernorm", _layernorm_backward, setup_context=_layernorm_setup,
)


def layernorm_opcheck() -> None:
    from torch.library import opcheck as _opcheck
    x = torch.randn(4, 16, device="cuda", dtype=torch.float32)
    w = torch.randn(16, device="cuda", dtype=torch.float32)
    b = torch.randn(16, device="cuda", dtype=torch.float32)
    _opcheck(torch.ops.mlp_kernel.layernorm.default, (x, w, b, 1e-5))


def gradcheck_layernorm(device: str = "cuda", dtype: torch.dtype = torch.float64) -> bool:
    from torch.autograd import gradcheck
    x = torch.randn(3, 16, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(16, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(16, device=device, dtype=dtype, requires_grad=True)
    return bool(gradcheck(lambda a, ww, bb: layernorm(a, ww, bb, 1e-5).sum(),
                          (x, w, b), eps=1e-6, atol=1e-4, rtol=1e-3))


def layernorm_compile_smoke(device: str = "cuda") -> torch.Tensor:
    x = torch.randn(8, 32, device=device)
    w = torch.randn(32, device=device)
    b = torch.randn(32, device=device)
    return torch.compile(lambda a, ww, bb: layernorm(a, ww, bb, 1e-5))(x, w, b)


def register_ops() -> None:
    """幂等: 验证已注册 + 触发 (import 时已完成注册, 此处仅自我检查)。"""
    for name in ("swiglu", "matmul", "layernorm"):
        op = getattr(torch.ops.mlp_kernel, name)
        assert op is not None, f"{name} op not registered"
    return None
