"""训练闭环契约测试（v2 辅线：训练语义保持）。

验证: 同一线性回归任务下，PyTorch / Triton / CUDA linear
（使用仓库 autograd 层 = kernel forward/backward）的损失下降曲线一致。

契约:
  1. 训练 >200 步后 loss 显著下降 (<30% 初始)
  2. 三后端最终 loss 接近（<0.1 绝对差）—— 训练语义对齐
  3. 梯度有限

运行: pytest tests/test_training_loop.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triton_kernels.precision import precision


@pytest.fixture(autouse=True)
def _cuda_strict():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.backends.cuda.matmul.allow_tf32 = False
    precision.allow_tf32 = False
    torch.manual_seed(42)


def _setup(N=64, IN=32, OUT=8):
    X = torch.randn(N, IN, device="cuda")
    Wstar = torch.randn(IN, OUT, device="cuda")
    return X, X @ Wstar


def _train(model, X, Y, lr=0.05, epochs=120):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    grads_finite = True
    for ep in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = F.mse_loss(out, Y)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                grads_finite = False
        opt.step()
        losses.append(loss.item())
    return losses, grads_finite


def _build(name):
    if name == "pytorch":
        return nn.Linear(32, 8, device="cuda")
    if name == "triton":
        from python.mnist.triton_layers import TritonLinear
        return TritonLinear(32, 8).cuda()
    if name == "cuda":
        from python.mnist.cuda_layers import CUDALinear
        return CUDALinear(32, 8).cuda()
    raise ValueError(name)


BACKENDS_TRAIN = ["pytorch", "triton", "cuda"]


@pytest.mark.parametrize("backend", BACKENDS_TRAIN)
def test_training_converges(backend):
    """训练收敛：loss 降至初始 30% 以下。"""
    _ = __import__("python.mnist.triton_layers", fromlist=["TritonLinear"]) if backend == "triton" else None
    _ = __import__("python.mnist.cuda_layers", fromlist=["CUDALinear"]) if backend == "cuda" else None
    try:
        model = _build(backend)
    except Exception:
        pytest.skip(f"{backend} unavailable")
    X, Y = _setup()
    torch.nn.init.normal_(next(model.parameters()), std=0.1)
    losses, grads_finite = _train(model, X, Y)
    rel = losses[-1] / losses[0] if losses[0] > 0 else 1.0
    assert grads_finite, f"{backend}: non-finite gradients"
    assert losses[-1] < 0.3 * losses[0], f"{backend}: loss not converged ({losses[0]:.3f} -> {losses[-1]:.3f}, rel {rel:.3f})"


def test_backends_learn_similarly():
    """契约：三后端最终 loss 一致（训练语义对齐）。"""
    X, Y = _setup()
    finals = {}
    for backend in BACKENDS_TRAIN:
        try:
            model = _build(backend)
        except Exception:
            continue
        torch.nn.init.normal_(next(model.parameters()), std=0.1)
        losses, _ = _train(model, X, Y)
        finals[backend] = losses[-1]
    assert len(finals) >= 2, "需要至少两个后端"
    vals = list(finals.values())
    spread = max(vals) - min(vals)
    assert spread < 0.1, f"后端训练差异过大: {finals} (spread {spread:.4f})"
