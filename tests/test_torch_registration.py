"""torch.library 自定义算子集成测试（P1 v2）。

验证: schema / CPU+CUDA impl / Meta(FakeTensor) / autograd / opcheck / gradcheck / torch.compile。

运行: pytest tests/test_torch_registration.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python.torch_registration import (
    swiglu, opcheck, gradcheck_swiglu, compile_smoke, register_ops,
    matmul, layernorm,
    matmul_opcheck, gradcheck_matmul, matmul_compile_smoke,
    layernorm_opcheck, gradcheck_layernorm, layernorm_compile_smoke,
)


@pytest.fixture(scope="module", autouse=True)
def _registered():
    register_ops()
    yield


def test_opcheck():
    """torch.library.opcheck: schema/meta/FakeTensor/dispatch/autograd 一致性。"""
    opcheck()  # 抛异常即失败


def test_gradcheck():
    """数值梯度验证（fp64）。"""
    assert gradcheck_swiglu("cuda", torch.float64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_forward_equals_torch(dtype):
    gate = torch.randn(4, 64, device="cuda", dtype=dtype)
    up = torch.randn(4, 64, device="cuda", dtype=dtype)
    ref = torch.nn.functional.silu(gate.float()) * up.float()
    out = swiglu(gate, up).float()
    err = (out - ref).abs().max().item()
    assert err < 5e-2, f"dtype={dtype} max_abs_err={err:.3e}"


def test_backward_gradients_finite():
    g = torch.randn(4, 32, device="cuda", requires_grad=True)
    u = torch.randn(4, 32, device="cuda", requires_grad=True)
    y = swiglu(g, u).sum()
    y.backward()
    assert torch.isfinite(g.grad).all()
    assert torch.isfinite(u.grad).all()


def test_backward_matches_manual():
    """手工解析梯度核对：dg = go * (s + g*s*(1-s)) * up, du = go * silu(g)"""
    g = torch.randn(4, 16, device="cuda", requires_grad=True)
    u = torch.randn(4, 16, device="cuda", requires_grad=True)
    y = swiglu(g, u).sum()
    y.backward()
    sig = torch.sigmoid(g.detach())
    dg_manual = (sig * (1 + g.detach() * (1 - sig))) * u.detach()
    du_manual = torch.nn.functional.silu(g.detach())
    assert torch.allclose(g.grad, dg_manual, atol=1e-5)
    assert torch.allclose(u.grad, du_manual, atol=1e-5)


def test_torch_compile_smoke():
    y = compile_smoke("cuda")
    assert tuple(y.shape) == (8, 32)
    assert torch.isfinite(y).all()


def test_meta_shape_deduction():
    """Meta/FakeTensor: compile 前可静态推导 shape（不碰 GPU）。"""
    gate = torch.empty(5, 6, device="meta")
    up = torch.empty(5, 6, device="meta")
    out = torch.ops.mlp_kernel.swiglu.default(gate, up)
    assert tuple(out.shape) == (5, 6)


# ============================================================
# mlp_kernel::matmul
# ============================================================

def test_matmul_opcheck():
    matmul_opcheck()


def test_matmul_gradcheck():
    assert gradcheck_matmul("cuda", torch.float64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_matmul_forward_equals_torch(dtype):
    a = torch.randn(4, 64, device="cuda", dtype=dtype)
    b = torch.randn(64, 32, device="cuda", dtype=dtype)
    ref = (a.float() @ b.float()).float()
    out = matmul(a, b).float()
    err = (out - ref).abs().max().item()
    assert err < 5e-2, f"dtype={dtype} max_abs_err={err:.3e}"


def test_matmul_backward_shapes_finite():
    a = torch.randn(4, 32, device="cuda", requires_grad=True)
    b = torch.randn(32, 16, device="cuda", requires_grad=True)
    matmul(a, b).sum().backward()
    assert tuple(a.grad.shape) == tuple(a.shape)
    assert tuple(b.grad.shape) == tuple(b.shape)
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_matmul_torch_compile_smoke():
    y = matmul_compile_smoke("cuda")
    assert tuple(y.shape) == (8, 16)
    assert torch.isfinite(y).all()


def test_matmul_meta_shape_deduction():
    a = torch.empty(5, 6, device="meta")
    b = torch.empty(6, 7, device="meta")
    out = torch.ops.mlp_kernel.matmul.default(a, b)
    assert tuple(out.shape) == (5, 7)


# ============================================================
# mlp_kernel::layernorm
# ============================================================

def test_layernorm_opcheck():
    layernorm_opcheck()


def test_layernorm_gradcheck():
    assert gradcheck_layernorm("cuda", torch.float64)


def test_layernorm_forward_equals_torch():
    x = torch.randn(4, 16, device="cuda", dtype=torch.float32)
    w = torch.randn(16, device="cuda", dtype=torch.float32)
    b = torch.randn(16, device="cuda", dtype=torch.float32)
    ref = torch.nn.functional.layer_norm(x, (16,), w, b, 1e-5)
    out = layernorm(x, w, b, 1e-5)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_layernorm_backward_finite():
    x = torch.randn(4, 16, device="cuda", requires_grad=True)
    w = torch.randn(16, device="cuda", requires_grad=True)
    b = torch.randn(16, device="cuda", requires_grad=True)
    layernorm(x, w, b, 1e-5).sum().backward()
    assert x.grad is not None and w.grad is not None and b.grad is not None
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all() and torch.isfinite(b.grad).all()


def test_layernorm_compile_smoke():
    y = layernorm_compile_smoke("cuda")
    assert tuple(y.shape) == (8, 32)
    assert torch.isfinite(y).all()


def test_layernorm_meta_shape_deduction():
    x = torch.empty(5, 6, device="meta")
    w = torch.empty(6, device="meta")
    b = torch.empty(6, device="meta")
    out = torch.ops.mlp_kernel.layernorm.default(x, w, b, 1e-5)
    assert tuple(out.shape) == (5, 6)

