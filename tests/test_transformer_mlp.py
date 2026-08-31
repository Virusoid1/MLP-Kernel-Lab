"""SwiGLU MLP block 正确性测试（v2 主线）。

以 eager FP32 为 reference，逐 backend 校验:
- decode/prefill/train shape（含非 tile 整除、极小 M）
- FP32 / FP16 / BF16（reference 用同 dtype eager，容差按 dtype 收紧）
- 多后端一致性（triton / cuda / cutile / compile）

运行:
    pytest tests/test_transformer_mlp.py -v
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python.transformer_mlp import (
    swiglu_block_eager, available_backends, BACKENDS,
)


def _make_case(M, K, F, dtype, seed=42, scale=1.0):
    torch.manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=dtype) * scale
    w_gate = torch.randn(K, F, device="cuda", dtype=dtype) * scale
    w_up = torch.randn(K, F, device="cuda", dtype=dtype) * scale
    w_down = torch.randn(F, K, device="cuda", dtype=dtype) * scale
    return x, w_gate, w_up, w_down


# reference dtype 一律用 FP32（避免 dtype 内部近似污染 reference）
def _reference(M, K, F, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.float32)
    w_gate = torch.randn(K, F, device="cuda", dtype=torch.float32)
    w_up = torch.randn(K, F, device="cuda", dtype=torch.float32)
    w_down = torch.randn(F, K, device="cuda", dtype=torch.float32)
    return x, w_gate, w_up, w_down


def _max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def _norm_l2_err(a, b):
    d = (a.float() - b.float())
    return (d.norm() / (b.float().norm() + 1e-12)).item()


@pytest.fixture(autouse=True)
def _cuda_guard():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# 每个 (backend, shape) 一个用例如一颗 claim 条目
SHAPES = [
    # decode（小 M，非 tile 整除）
    (1, 768, 3072), (4, 768, 3072), (16, 768, 3072), (32, 768, 3072),
    (1, 4096, 11008), (32, 4096, 11008),
    # prefill / train
    (128, 768, 3072), (512, 768, 3072), (256, 4096, 3072),
    # 边界：极小 K、非整除
    (7, 33, 65), (3, 64, 128),
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("M,K,F", SHAPES)
@pytest.mark.parametrize("backend", ["eager", "triton", "cuda", "cutile", "compile"])
def test_swiglu_block_equals_reference(M, K, F, backend):
    """FP32: 各后端输出 vs eager FP32 reference（max_abs 装）"""
    x, wg, wu, wd = _reference(M, K, F)
    ref = swiglu_block_eager(x, wg, wu, wd)
    fn = BACKENDS.get(backend)
    if fn is None:
        pytest.skip(f"backend {backend} not available")
    try:
        out = fn(x, wg, wu, wd)
    except Exception as e:
        if backend not in available_backends():
            pytest.skip(f"{backend} unavailable: {e}")
        raise
    err = _max_abs_err(out, ref)
    # FP32: 各后端相对 eager 的误差应在 matmul 正常噪声内（tf32 关闭）
    assert err < 5e-3, f"{backend} M={M} K={K} F={F} max_abs_err={err:.3e}"


@pytest.mark.parametrize("M,K,F", [(64, 768, 3072), (512, 768, 3072)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", ["eager", "triton", "cuda", "cutile"])
def test_swiglu_block_dtype_consistent(M, K, F, dtype, backend):
    """FP16/BF16: 后端 vs 同 dtype eager（同一数据，容差按 dtype）"""
    x, wg, wu, wd = _make_case(M, K, F, dtype)
    ref = swiglu_block_eager(x, wg, wu, wd)
    fn = BACKENDS.get(backend)
    if fn is None or backend not in available_backends():
        pytest.skip(f"{backend} unavailable")
    out = fn(x, wg, wu, wd)
    tol = 2e-2 if dtype == torch.float16 else 3e-2  # bf16 位宽更粗
    err = _max_abs_err(out, ref)
    assert err < tol, f"{backend} {dtype} M={M} max_abs_err={err:.3e}"


def test_available_backends_reports_current_env():
    """available_backends() 至少包含 eager/concat/compile（本机 GPU 上应包 triton/cuda）"""
    got = available_backends()
    assert "eager" in got and "concat" in got
    assert "cuda" in got  # 本仓库主线要求 CUDA 扩展可用
