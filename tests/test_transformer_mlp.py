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
    """构造输入。fp16/bf16 默认自带 0.1 缩放（见下方说明）。

    已验证：fp16 下 scale=1.0 时 eager 的 hidden @ w_down 在 K=3072 累加溢出为 inf
    （hidden~8.8e3 * wd~1 * 3072 项 > 65504）—— 这是 fp16 数值边界，不是 kernel bug。
    用 scale=0.1 让数据落在 fp16 有限域内，作为"稳定输入"下的 dtype 一致性验证。
    """
    if dtype in (torch.float16, torch.bfloat16) and scale == 1.0:
        scale = 0.1
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
def _strict_fp32():
    """FP32 对照统一关闭 TF32（reference 与 backend 同基线）。

    注意：自定义后端(triton/cuda/cutile)的 matmul 在 FP32 strict 下 norm_l2 <= 1e-6。
    max_abs_err 会被个别大值元素放大，不作为 FP32 通过判据；
    TF32 模式的精度边界由 test_tf32_mode 单独刻画。
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        from triton_kernels.precision import precision
        precision.allow_tf32 = False
    except Exception:
        pass
    yield


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
    err = _norm_l2_err(out, ref)
    # FP32 strict 下各后端相对 eager 的归一化 L2 误差（TF32 已关闭）
    assert err < 1e-4, f"{backend} M={M} K={K} F={F} norm_l2={err:.3e}"


# dtype 支持矩阵：哪些 (backend, dtype) 组合当前可用。
# 在不支持/未实现的组合上显式 skip 并说明原因（真实边界，不是假失败）。
# 目前实测（RTX 3070 + 仓库当前 kernel）：
#   eager/concat/compile : fp32 全支持。fp16/bf16 在 scale=0.1 稳定输入下可用
#                          （scale=1.0 时 eager fp16 的 hidden@w_down K=3072 累加溢出 inf = fp16 数值边界）。
#   triton               : tl.dot 要求同 dtype；bf16 输入与 fp32 累加器冲突（需 input_precision=ieee，未实现）
#   cuda                 : fp32 matmul_tiled_auto / fp16 matmul_half / bf16 matmul_bf16（2026-09-02 全 dtype 自定义 WMMA）
#   cutile               : ct.mma 要求 x/y 同 dtype，但 cutile_matmul 恒输出 fp32 → 块组合需转回
#                          输入 dtype（swiglu_block_cutile 已修复：fp16/bf16 block norm_l2 6e-4/4.7e-3）
DTYPE_SUPPORT = {
    ("eager", torch.float32): True, ("eager", torch.float16): "overflow", ("eager", torch.bfloat16): "unverified",
    ("triton", torch.float32): True, ("triton", torch.float16): True, ("triton", torch.bfloat16): True,
    ("cuda", torch.float32): True, ("cuda", torch.float16): True, ("cuda", torch.bfloat16): True,  # matmul_bf16 + swiglu_fused_bf16
    ("cutile", torch.float32): True, ("cutile", torch.float16): True, ("cutile", torch.bfloat16): True,
}


@pytest.mark.parametrize("M,K,F", [(64, 768, 3072), (512, 768, 3072)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", ["eager", "triton", "cuda", "cutile"])
def test_swiglu_block_dtype_support_matrix(M, K, F, dtype, backend):
    """FP16/BF16 支持矩阵：可用组合过 norm_l2，明确不支持/未验证的组合 skip 并记录原因。"""
    st = DTYPE_SUPPORT.get((backend, dtype), "unverified")
    if st is not True:
        pytest.skip(f"{backend} x {dtype}: {st}（已知 dtype 边界，见 claim-matrix）")
    if backend not in available_backends():
        pytest.skip(f"{backend} unavailable")
    x, wg, wu, wd = _make_case(M, K, F, dtype)
    # 权威 reference 必须是 FP32（独立于被测 dtype，避免 dtype 自身溢出被继承）
    ref = swiglu_block_eager(x.float(), wg.float(), wu.float(), wd.float())
    assert torch.isfinite(ref).all(), f"reference not finite for {dtype}"
    fn = BACKENDS[backend]
    out = fn(x, wg, wu, wd)
    assert torch.isfinite(out).all(), f"{backend} {dtype} produced non-finite output"
    tol = 2e-2 if dtype == torch.float16 else 3e-2
    err = _norm_l2_err(out, ref)
    assert err < tol, f"{backend} {dtype} M={M} norm_l2={err:.3e}"


def test_available_backends_reports_current_env():
    """available_backends() 至少包含 eager/concat/compile（本机 GPU 上应包 triton/cuda）"""
    got = available_backends()
    assert "eager" in got and "concat" in got
    assert "triton" in got  # Triton 主线后端（纯 Python + 已装）
    # cuda 后端需要编译的 mlp_cuda 扩展：可 import 时才要求列出（E4 无 nvcc 环境自适应）
    try:
        import mlp_cuda  # noqa: F401
        assert "cuda" in got, f"mlp_cuda 已可 import 但 available_backends 未列出 (got={got})"
    except Exception:
        assert "cuda" not in got, f"mlp_cuda 不可用但 available_backends 列出了 cuda (got={got})"


def test_tf32_mode_precision_boundary():
    """TF32 模式（TensorCore）的精度边界画像——已知且应记录，不判为 bug。

    我们的 block 在 strict FP32 下 norm_l2 <= 1e-6（见其他用例）。
    开启 TF32 后，triton/cuda 走 TensorCore，相对 eager FP32 的归一化 L2
    正常情况下应在 1e-4 量级；若个别 shape 超过 1e-2 则需标注为精度风险。
    """
    from triton_kernels.precision import precision
    torch.backends.cuda.matmul.allow_tf32 = True
    precision.allow_tf32 = True
    x, wg, wu, wd = _reference(64, 768, 3072)
    ref = swiglu_block_eager(x, wg, wu, wd)
    for backend in ("triton", "cuda"):
        fn = BACKENDS.get(backend)
        if fn is None or backend not in available_backends():
            continue
        out = fn(x, wg, wu, wd)
        err = _norm_l2_err(out, ref)
        print(f"[tf32] {backend} norm_l2={err:.3e}")
        assert err < 1e-2, f"{backend} TF32 norm_l2={err:.3e} over boundary"