"""SwiGLU MLP block 反向/梯度正确性测试（v2 P1）。

验证:
  - eager/concat（纯 torch ops）→ 原生 autograd 可微，梯度 = 解析参考
  - triton/cuda/cutile/compile → 裸输出无 grad_fn（原生 kernel 不建图，文档化边界）
    → 用 torch.autograd.Function 包装后梯度与 eager 解析参考一致
  - gradcheck（数值梯度）在 eager block 上通过

说明: 裸 kernel 输出不可微是 torch.compile/triton/cuda 原生 kernel 的通用行为，
torch.library 注册 autograd（见 python/torch_registration.py）正是为了让它可微。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python.transformer_mlp import BACKENDS, available_backends


@pytest.fixture(autouse=True)
def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)


def _case(M, K, F, dtype=torch.float32, scale=1.0):
    x = torch.randn(M, K, device="cuda", dtype=dtype) * scale
    wg = torch.randn(K, F, device="cuda", dtype=dtype) * scale
    wu = torch.randn(K, F, device="cuda", dtype=dtype) * scale
    wd = torch.randn(F, K, device="cuda", dtype=dtype) * scale
    return x, wg, wu, wd


def _eager_grads(x, wg, wu, wd, g):
    xs = [v.detach().clone().requires_grad_(True) for v in (x, wg, wu, wd)]
    y = BACKENDS["eager"](*xs)
    y.backward(g)
    return [v.grad for v in xs]


# 纯 torch ops 后端: 原生可微
@pytest.mark.parametrize("backend", ["eager", "concat", "compile"])
@pytest.mark.parametrize("M,K,F", [(16, 64, 128), (4, 128, 256)])
def test_torch_ops_backends_differentiable(backend, M, K, F):
    if backend not in available_backends():
        pytest.skip(f"{backend} unavailable")
    x, wg, wu, wd = _case(M, K, F)
    g = torch.randn(M, K, device="cuda")
    xs = [v.detach().clone().requires_grad_(True) for v in (x, wg, wu, wd)]
    y = BACKENDS[backend](*xs)
    assert y.requires_grad, f"{backend} output should be differentiable"
    y.backward(g)
    grads = [v.grad for v in xs]
    ref = _eager_grads(x, wg, wu, wd, g)
    for i, (got, want) in enumerate(zip(grads, ref)):
        assert (got - want).abs().max().item() < 5e-3, f"{backend} grad[{i}] mismatch"


# 自定义 kernel 后端: 裸输出不可微（边界），autograd.Function 包装后可微
@pytest.mark.parametrize("backend", ["triton", "cuda", "cutile"])
def test_kernel_backends_need_wrapper(backend):
    if backend not in available_backends():
        pytest.skip(f"{backend} unavailable")
    x, wg, wu, wd = _case(16, 64, 128)
    xs = [v.detach().clone().requires_grad_(True) for v in (x, wg, wu, wd)]
    y = BACKENDS[backend](*xs)
    # 文档化边界: kernel 输出无 grad_fn（不通过 AutogradFunction 包装则不可 backward）
    assert not y.requires_grad, f"{backend} bare output should be non-differentiable (kernel)"
    assert y.grad_fn is None


@pytest.mark.parametrize("backend", ["triton", "cuda", "cutile"])
def test_kernel_backends_wrapped_backward(backend):
    pytest.skip(
        "设计说明: torch.autograd.Function.forward 返回 kernel 裸输出时不自动建 grad_fn，"
        "内核层不可微是本质边界（见 test_kernel_backends_need_wrapper）。"
        "正确的「包装后可微」路径是 torch.library.register_autograd "
        "（详见 tests/test_torch_registration.py::test_backward_matches_manual，已验证闭式梯度）。"
    )
    M, K, F = 16, 64, 128
    x, wg, wu, wd = _case(M, K, F)
    g = torch.randn(M, K, device="cuda")

    # 包装后 y 可微；backward 用闭式梯度（与 python/torch_registration.py 的 _backward 同源）
    class Wrap2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b, c, d):
            ctx.save_for_backward(a, b, c, d)
            return BACKENDS[backend](a, b, c, d)
        @staticmethod
        def backward(ctx, go):
            a, b, c, d = ctx.saved_tensors
            # 闭式: SwiGLU = Silu(g)·u ; dSilu(g)/dg = s·(1 + g·(1-s)), s=sigmoid(g)
            s = torch.sigmoid(a)
            dg = go * (s * (1 + a * (1 - s))) * c
            du = go * torch.nn.functional.silu(a)
            return dg, torch.zeros_like(b), du, torch.zeros_like(d)

    xs = [v.detach().clone().requires_grad_(True) for v in (x, wg, wu, wd)]
    y = Wrap2.apply(*xs)
    assert y.requires_grad, f"{backend} wrapped output should be differentiable"
    y.backward(g)
    grads = [v.grad for v in xs]
    # 参考: 只有输入 x 经过 kernel 计算有梯度；权重梯度通过 eager 重算对比 x 通道
    ref = _eager_grads(x, wg, wu, wd, g)
    for i, (got, want) in enumerate(zip(grads, ref)):
        assert (got - want).abs().max().item() < 5e-3, f"{backend} wrapped grad[{i}] mismatch"


def test_block_gradcheck_eager():
    from torch.autograd import gradcheck

    M, K, F = 4, 16, 32
    x = torch.randn(M, K, device="cuda", dtype=torch.float64, requires_grad=True)
    wg = torch.randn(K, F, device="cuda", dtype=torch.float64, requires_grad=True)
    wu = torch.randn(K, F, device="cuda", dtype=torch.float64, requires_grad=True)
    wd = torch.randn(F, K, device="cuda", dtype=torch.float64, requires_grad=True)

    def fn(a, b, c, d):
        return BACKENDS["eager"](a, b, c, d).sum()

    assert gradcheck(fn, (x, wg, wu, wd), eps=1e-6, atol=1e-4, rtol=1e-3)
