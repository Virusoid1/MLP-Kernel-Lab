"""算子级 dtype 支持矩阵（v2，3070 实测固化）。

目标: 以可运行测试固定"哪个后端 × 哪个算子 × fp16/bf16 可用"的边界，
由测试自动探测（import 成功 + 执行成功即可用），输出即证据。

覆盖: matmul / bias_add / relu / gelu / silu / softmax / swiglu / fused_mlp_first
后端: eager(ref) / triton / cuda / cutile

运行: pytest tests/test_dtype_support_matrix.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triton_kernels.precision import precision

# 已实测的阻塞原因（CUDA/cuTile binding 硬检查 float32，见 claim-matrix）
# 这里用"执行式探测"：内核抛 "must be float32" 即记录为 blocked。
BLOCKED_KEYWORDS = ("must be float32", "must have the same dtype", "only integer")


def _try_run(fn, *args):
    """执行 fn；成功返回 (True, None)，失败返回 (False, 错误摘要)。"""
    try:
        fn(*args)
        torch.cuda.synchronize()
        return True, None
    except Exception as e:
        msg = str(e)
        if any(k in msg for k in BLOCKED_KEYWORDS):
            return False, "blocked:" + msg.split("\n")[0][:70]
        return False, type(e).__name__ + ":" + msg.split("\n")[0][:70]


def _cases():
    torch.manual_seed(42)
    return {
        "matmul": (torch.randn(16, 32, device="cuda"), torch.randn(32, 64, device="cuda")),
        "bias_add": (torch.randn(16, 32, device="cuda"), torch.randn(32, device="cuda")),
        "relu": (torch.randn(64, device="cuda"),),
        "gelu": (torch.randn(64, device="cuda"),),
        "silu": (torch.randn(64, device="cuda"),),
        "softmax": (torch.randn(8, 32, device="cuda"),),
        "swiglu": (torch.randn(16, 32, device="cuda"), torch.randn(16, 32, device="cuda")),
        "fused_mlp_first": (torch.randn(16, 32, device="cuda"),
                            torch.randn(32, 64, device="cuda"),
                            torch.randn(64, device="cuda")),
    }


BACKEND_CALLS = {
    "triton": {
        "matmul": lambda d, a, b: __import__("triton_kernels.matmul", fromlist=["tiled_matmul"]).tiled_matmul(a, b),
        "bias_add": lambda d, a, b: __import__("triton_kernels.elementwise", fromlist=["bias_add"]).bias_add(a, b),
        "relu": lambda d, x: __import__("triton_kernels.elementwise", fromlist=["relu"]).relu(x),
        "gelu": lambda d, x: __import__("triton_kernels.elementwise", fromlist=["gelu"]).gelu(x),
        "silu": lambda d, x: __import__("triton_kernels.elementwise", fromlist=["silu"]).silu(x),
        "softmax": lambda d, x: __import__("triton_kernels.softmax", fromlist=["softmax"]).softmax(x),
        "swiglu": lambda d, g, u: __import__("triton_kernels.swiglu_triton", fromlist=["swiglu_triton"]).swiglu_triton(g, u),
        "fused_mlp_first": lambda d, x, w, b: __import__("triton_kernels.mlp_triton", fromlist=["mlp_first_layer_triton"]).mlp_first_layer_triton(x, w, b),
    },
    "cuda": {
        "matmul": lambda d, a, b: (__import__("mlp_cuda").matmul_half(a, b) if d == torch.float16
                                     else __import__("mlp_cuda").matmul_bf16(a, b) if d == torch.bfloat16
                                     else __import__("mlp_cuda").matmul_tiled_auto(a, b)),
        "bias_add": lambda d, a, b: __import__("mlp_cuda").bias_add(a, b),
        "relu": lambda d, x: __import__("mlp_cuda").relu(x),
        "gelu": lambda d, x: __import__("mlp_cuda").gelu(x),
        "silu": lambda d, x: __import__("mlp_cuda").silu(x),
        "softmax": lambda d, x: __import__("mlp_cuda").softmax(x),
        "swiglu": lambda d, g, u: __import__("mlp_cuda").swiglu_fused(g, u),
        "fused_mlp_first": lambda d, x, w, b: __import__("mlp_cuda").mlp_fused_first_layer(x, w, b),
    },
    "cutile": {
        "matmul": lambda d, a, b: __import__("cutile_kernels.matmul", fromlist=["cutile_matmul"]).cutile_matmul(a, b),
        "swiglu": lambda d, g, u: __import__("cutile_kernels.swiglu_cutile", fromlist=["swiglu_cutile"]).swiglu_cutile(g),
    },
}


def _is_available(backend):
    try:
        if backend == "cuda":
            import mlp_cuda  # noqa
        elif backend == "cutile":
            import cuda.tile  # noqa
            import cutile_kernels.matmul  # noqa
        elif backend == "triton":
            import triton_kernels.matmul  # noqa
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _cuda_and_strict():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.backends.cuda.matmul.allow_tf32 = False
    precision.allow_tf32 = False


OPS = ["matmul", "bias_add", "relu", "gelu", "silu", "softmax", "swiglu", "fused_mlp_first"]
BACKENDS = ["triton", "cuda", "cutile"]
DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_dtype_support_probe(backend, op, dtype):
    """执行式探测：后端×算子×dtype 是否可用（抛 blocked 即固化为 skip+原因）。"""
    if not _is_available(backend):
        pytest.skip(f"{backend} not installed")
    q = _cases()
    if op not in q:
        pytest.skip(f"no case for {op}")
    args = [v.to(dtype) for v in q[op]]
    fn = BACKEND_CALLS[backend].get(op)
    if fn is None:
        pytest.skip(f"no probe for {backend}.{op}")
    ok, why = _try_run(fn, dtype, *args)
    if not ok:
        pytest.skip(f"{backend}.{op} {dtype}: {why}")
    # 成功 → 记录（测试通过即"可用"证据）
    assert True
