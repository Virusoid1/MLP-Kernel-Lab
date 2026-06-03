"""
算子级横向对比：PyTorch vs Triton vs CUDA

逐算子测试正确性（L2误差、最大绝对误差）和性能（延迟中位数/P95）。

用法:
    python benchmark_ops.py                    # 全部算子，默认尺寸
    python benchmark_ops.py --warmup 50 --iters 200
    python benchmark_ops.py --sizes small      # 只测小尺寸
    python benchmark_ops.py --ops matmul,gelu  # 只测指定算子
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from triton_kernels import (
    tiled_matmul,
    bias_add as triton_bias_add,
    relu as triton_relu,
    gelu as triton_gelu,
    silu as triton_silu,
    bias_add_relu as triton_bias_add_relu,
    matmul_backward_a,
    matmul_backward_b,
    relu_backward as triton_relu_backward,
    gelu_backward as triton_gelu_backward,
    silu_backward as triton_silu_backward,
    mlp_first_layer_triton,
    swiglu_triton,
    conv2d_triton,
    maxpool2d as triton_maxpool2d,
    avgpool2d as triton_avgpool2d,
    softmax as triton_softmax,
)
from triton_kernels.precision import precision
from python.mnist.benchmark import capture_metadata, p95
from python.mnist.stats import stable_median  # noqa: F401  (exported for future use)

try:
    import mlp_cuda
    _HAS_CUDA = True
except ImportError:
    mlp_cuda = None
    _HAS_CUDA = False

try:
    import cutile_kernels  # noqa: F401
    _HAS_CUTILE = True
except ImportError:
    cutile_kernels = None  # type: ignore
    _HAS_CUTILE = False

# 全局 dtype:由 main 的 dtype sweep 设置,bench_* 函数读取
_CURRENT_DTYPE = torch.float32
_DTYPE_NAMES = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}


# ============================================================
# 数据结构
# ============================================================

@dataclass
class BenchResult:
    name: str
    shapes: str
    dtype: str = "fp32"
    pytorch_ms: float = 0.0
    triton_ms: float = 0.0
    cuda_ms: float = 0.0
    cutile_ms: float = 0.0
    pytorch_p95_ms: float = 0.0
    triton_p95_ms: float = 0.0
    cuda_p95_ms: float = 0.0
    cutile_p95_ms: float = 0.0
    triton_l2_err: float = 0.0
    triton_max_err: float = 0.0
    cuda_l2_err: float = 0.0
    cuda_max_err: float = 0.0
    cutile_l2_err: float = 0.0
    cutile_max_err: float = 0.0
    triton_speedup: float = 0.0
    cuda_speedup: float = 0.0
    cutile_speedup: float = 0.0
    # roofline 指标 (--roofline 启用时填充, 否则 0)
    flops: int = 0
    bytes_io: int = 0
    pytorch_tflops: float = 0.0
    triton_tflops: float = 0.0
    cuda_tflops: float = 0.0
    pytorch_gbps: float = 0.0
    triton_gbps: float = 0.0
    cuda_gbps: float = 0.0


# ============================================================
# 测试尺寸
# ============================================================

_SIZE_PRESETS = {
    "small": [
        {"M": 64, "K": 128, "N": 64, "elem": 4096},
        {"M": 128, "K": 256, "N": 128, "elem": 16384},
    ],
    "medium": [
        {"M": 256, "K": 512, "N": 256, "elem": 65536},
        {"M": 512, "K": 768, "N": 512, "elem": 262144},
    ],
    "large": [
        {"M": 1024, "K": 1024, "N": 1024, "elem": 1048576},
        {"M": 2048, "K": 2048, "N": 2048, "elem": 4194304},
    ],
    "conv": [
        {"C_in": 3, "C_out": 16, "H": 32, "W": 32, "K": 3, "stride": 1, "pad": 1, "N": 16},
        {"C_in": 16, "C_out": 32, "H": 16, "W": 16, "K": 3, "stride": 1, "pad": 1, "N": 16},
    ],
    "pool": [
        {"C": 16, "H": 32, "W": 32, "K": 2, "stride": 2, "pad": 0, "N": 16},
        {"C": 32, "H": 16, "W": 16, "K": 2, "stride": 2, "pad": 0, "N": 16},
    ],
}

ALL_SIZES = []
for _preset in _SIZE_PRESETS.values():
    ALL_SIZES.extend(_preset)


# ============================================================
# 通用 benchmark 工具
# ============================================================

def _bench(fn, *args, warmup=20, iters=100, **kwargs):
    """返回 (median_ms, p95_ms)。"""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return float(np.median(arr)), float(np.percentile(arr, 95))


def _errors(ref: torch.Tensor, test: torch.Tensor):
    """返回 (l2_error, max_abs_error)。"""
    diff = (ref.float() - test.float()).detach()
    l2 = torch.norm(diff).item()
    max_abs = diff.abs().max().item()
    return l2, max_abs


# ============================================================
# 算子测试
# ============================================================

def bench_matmul(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    M, K, N = size["M"], size["K"], size["N"]
    A = torch.randn(M, K, device="cuda", dtype=_CURRENT_DTYPE)
    B = torch.randn(K, N, device="cuda", dtype=_CURRENT_DTYPE)

    # T (P1-bis): reference 强制 FP32 累加, 不受 precision.allow_tf32 影响.
    # 之前 ref 在 TF32 模式下也走 TF32, 与 triton.tl.dot(allow_tf32=True)
    # 共享 Tensor Core 但累加次序有差, L2 噪声 ~1e+01 (norm). 显式 FP32 让
    # L2 偏差真实反映 "backend 是否启用 TF32 / FP16 / FP32" 而不是
    # "TF32 vs TF32 累加次序噪声".
    # T (P1-bis): reference 强制 FP32 累加, 不受 precision.allow_tf32 影响.
    # 之前 ref 在 TF32 模式下也走 TF32, 与 triton.tl.dot(allow_tf32=True)
    # 共享 Tensor Core 但累加次序有差, L2 噪声 ~1e+01 (norm). 显式 FP32 让
    # L2 偏差真实反映 "backend 是否启用 TF32 / FP16 / FP32" 而不是
    # "TF32 vs TF32 累加次序噪声".
    # AC: --ref-tf32 flag 让用户切回老基线 (ref=TF32, 与原 baseline.json 可比).
    _ref_tf32_flag = getattr(sys.modules[__name__], "_REF_TF32", False)
    _saved_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = (
        precision.allow_tf32 if _ref_tf32_flag else False
    )
    try:
        ref = torch.matmul(A, B)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = _saved_tf32

    # Triton
    tr_out = tiled_matmul(A, B)
    tr_l2, tr_max = _errors(ref, tr_out)
    tr_med, tr_p95 = _bench(tiled_matmul, A, B, warmup=warmup, iters=iters)

    # PyTorch
    pt_med, pt_p95 = _bench(torch.matmul, A, B, warmup=warmup, iters=iters)

    results = [BenchResult(
        name="matmul", shapes=f"({M},{K})@({K},{N})",
        pytorch_ms=pt_med, triton_ms=tr_med,
        pytorch_p95_ms=pt_p95, triton_p95_ms=tr_p95,
        triton_l2_err=tr_l2, triton_max_err=tr_max,
        triton_speedup=pt_med / tr_med if tr_med > 0 else 0,
    )]

    if _HAS_CUDA:
        cu_out = mlp_cuda.matmul_tiled_auto(A, B)
        cu_l2, cu_max = _errors(ref, cu_out)
        cu_med, cu_p95 = _bench(mlp_cuda.matmul_tiled_auto, A, B, warmup=warmup, iters=iters)
        results[0].cuda_ms = cu_med
        results[0].cuda_p95_ms = cu_p95
        results[0].cuda_l2_err = cu_l2
        results[0].cuda_max_err = cu_max
        results[0].cuda_speedup = pt_med / cu_med if cu_med > 0 else 0

    return results


def _bench_activation(name: str, size: dict, warmup: int, iters: int,
                      pt_fn, tr_fn, cu_fn_name: str) -> BenchResult:
    n = size["elem"]
    x = torch.randn(n, device="cuda", dtype=_CURRENT_DTYPE)

    ref = pt_fn(x)

    tr_out = tr_fn(x)
    tr_l2, tr_max = _errors(ref, tr_out)
    tr_med, tr_p95 = _bench(tr_fn, x, warmup=warmup, iters=iters)
    pt_med, pt_p95 = _bench(pt_fn, x, warmup=warmup, iters=iters)

    r = BenchResult(
        name=name, shapes=f"({n},)",
        pytorch_ms=pt_med, triton_ms=tr_med,
        pytorch_p95_ms=pt_p95, triton_p95_ms=tr_p95,
        triton_l2_err=tr_l2, triton_max_err=tr_max,
        triton_speedup=pt_med / tr_med if tr_med > 0 else 0,
    )

    if _HAS_CUDA and cu_fn_name:
        cu_fn = getattr(mlp_cuda, cu_fn_name)
        cu_out = cu_fn(x)
        cu_l2, cu_max = _errors(ref, cu_out)
        cu_med, cu_p95 = _bench(cu_fn, x, warmup=warmup, iters=iters)
        r.cuda_ms = cu_med
        r.cuda_p95_ms = cu_p95
        r.cuda_l2_err = cu_l2
        r.cuda_max_err = cu_max
        r.cuda_speedup = pt_med / cu_med if cu_med > 0 else 0

    return r


def bench_gelu(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    return [_bench_activation(
        "gelu", size, warmup, iters,
        lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
        triton_gelu, "gelu",
    )]


def bench_relu(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    return [_bench_activation(
        "relu", size, warmup, iters,
        torch.relu, triton_relu, "relu",
    )]


def bench_silu(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    return [_bench_activation(
        "silu", size, warmup, iters,
        torch.nn.functional.silu, triton_silu, "silu",
    )]


def _bench_act_backward(name: str, size: dict, warmup: int, iters: int,
                        pt_fn, tr_fn, cu_fn_name: str) -> BenchResult:
    n = size["elem"]
    x = torch.randn(n, device="cuda", dtype=_CURRENT_DTYPE, requires_grad=True)
    grad = torch.randn(n, device="cuda", dtype=_CURRENT_DTYPE)

    # PyTorch autograd 参考
    y = pt_fn(x)
    ref = torch.autograd.grad(y, x, grad)[0]

    tr_out = tr_fn(grad, x.detach())
    tr_l2, tr_max = _errors(ref, tr_out)
    tr_med, tr_p95 = _bench(tr_fn, grad, x.detach(), warmup=warmup, iters=iters)

    # PyTorch backward benchmark（前向+反向一起）
    def _pt_step():
        x2 = x.detach().requires_grad_(True)
        y2 = pt_fn(x2)
        y2.backward(grad)

    pt_med, pt_p95 = _bench(_pt_step, warmup=warmup, iters=iters)

    r = BenchResult(
        name=name, shapes=f"({n},)",
        pytorch_ms=pt_med, triton_ms=tr_med,
        pytorch_p95_ms=pt_p95, triton_p95_ms=tr_p95,
        triton_l2_err=tr_l2, triton_max_err=tr_max,
        triton_speedup=pt_med / tr_med if tr_med > 0 else 0,
    )

    if _HAS_CUDA and cu_fn_name:
        cu_fn = getattr(mlp_cuda, cu_fn_name)
        cu_out = cu_fn(grad, x.detach())
        cu_l2, cu_max = _errors(ref, cu_out)
        cu_med, cu_p95 = _bench(cu_fn, grad, x.detach(), warmup=warmup, iters=iters)
        r.cuda_ms = cu_med
        r.cuda_p95_ms = cu_p95
        r.cuda_l2_err = cu_l2
        r.cuda_max_err = cu_max
        r.cuda_speedup = pt_med / cu_med if cu_med > 0 else 0

    return r


def bench_gelu_backward(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    return [_bench_act_backward(
        "gelu_backward", size, warmup, iters,
        lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
        triton_gelu_backward, "gelu_backward",
    )]


def bench_relu_backward(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    return [_bench_act_backward(
        "relu_backward", size, warmup, iters,
        torch.relu, triton_relu_backward, "relu_backward",
    )]


def bench_silu_backward(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    return [_bench_act_backward(
        "silu_backward", size, warmup, iters,
        torch.nn.functional.silu, triton_silu_backward, "silu_backward",
    )]


def bench_bias_add(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    M, N = size["M"], size["N"]
    x = torch.randn(M, N, device="cuda", dtype=_CURRENT_DTYPE)
    bias = torch.randn(N, device="cuda", dtype=_CURRENT_DTYPE)

    ref = x + bias

    tr_out = triton_bias_add(x, bias)
    tr_l2, tr_max = _errors(ref, tr_out)
    tr_med, tr_p95 = _bench(triton_bias_add, x, bias, warmup=warmup, iters=iters)

    pt_med, pt_p95 = _bench(lambda a, b: a + b, x, bias, warmup=warmup, iters=iters)

    r = BenchResult(
        name="bias_add", shapes=f"({M},{N})+({N},)",
        pytorch_ms=pt_med, triton_ms=tr_med,
        pytorch_p95_ms=pt_p95, triton_p95_ms=tr_p95,
        triton_l2_err=tr_l2, triton_max_err=tr_max,
        triton_speedup=pt_med / tr_med if tr_med > 0 else 0,
    )

    if _HAS_CUDA:
        cu_out = mlp_cuda.bias_add(x, bias)
        cu_l2, cu_max = _errors(ref, cu_out)
        cu_med, cu_p95 = _bench(mlp_cuda.bias_add, x, bias, warmup=warmup, iters=iters)
        r.cuda_ms = cu_med
        r.cuda_p95_ms = cu_p95
        r.cuda_l2_err = cu_l2
        r.cuda_max_err = cu_max
        r.cuda_speedup = pt_med / cu_med if cu_med > 0 else 0

    return [r]


def bench_matmul_backward(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    M, K, N = size["M"], size["K"], size["N"]
    A = torch.randn(M, K, device="cuda", dtype=_CURRENT_DTYPE, requires_grad=True)
    B = torch.randn(K, N, device="cuda", dtype=_CURRENT_DTYPE, requires_grad=True)
    dC = torch.randn(M, N, device="cuda", dtype=_CURRENT_DTYPE)

    # PyTorch 参考
    C = torch.matmul(A, B)
    grads = torch.autograd.grad(C, (A, B), dC)
    ref_dA, ref_dB = grads

    results = []

    # --- dA = dC @ B^T ---
    tr_dA = matmul_backward_a(dC, B.detach())
    tr_l2, tr_max = _errors(ref_dA, tr_dA)
    tr_med, tr_p95 = _bench(matmul_backward_a, dC, B.detach(), warmup=warmup, iters=iters)

    def _pt_dA():
        a2 = A.detach().requires_grad_(True)
        c2 = torch.matmul(a2, B.detach())
        torch.autograd.grad(c2, a2, dC)

    pt_med, pt_p95 = _bench(_pt_dA, warmup=warmup, iters=iters)

    r_dA = BenchResult(
        name="matmul_backward_dA", shapes=f"dC({M},{N})@B^T({N},{K})",
        pytorch_ms=pt_med, triton_ms=tr_med,
        pytorch_p95_ms=pt_p95, triton_p95_ms=tr_p95,
        triton_l2_err=tr_l2, triton_max_err=tr_max,
        triton_speedup=pt_med / tr_med if tr_med > 0 else 0,
    )

    if _HAS_CUDA:
        cu_dA = mlp_cuda.matmul_transB(dC, B.detach())
        cu_l2, cu_max = _errors(ref_dA, cu_dA)
        cu_med, cu_p95 = _bench(mlp_cuda.matmul_transB, dC, B.detach(), warmup=warmup, iters=iters)
        r_dA.cuda_ms = cu_med
        r_dA.cuda_p95_ms = cu_p95
        r_dA.cuda_l2_err = cu_l2
        r_dA.cuda_max_err = cu_max
        r_dA.cuda_speedup = pt_med / cu_med if cu_med > 0 else 0

    results.append(r_dA)

    # --- dB = A^T @ dC ---
    tr_dB = matmul_backward_b(A.detach(), dC)
    tr_l2, tr_max = _errors(ref_dB, tr_dB)
    tr_med, tr_p95 = _bench(matmul_backward_b, A.detach(), dC, warmup=warmup, iters=iters)

    def _pt_dB():
        b2 = B.detach().requires_grad_(True)
        c2 = torch.matmul(A.detach(), b2)
        torch.autograd.grad(c2, b2, dC)

    pt_med, pt_p95 = _bench(_pt_dB, warmup=warmup, iters=iters)

    r_dB = BenchResult(
        name="matmul_backward_dB", shapes=f"A^T({K},{M})@dC({M},{N})",
        pytorch_ms=pt_med, triton_ms=tr_med,
        pytorch_p95_ms=pt_p95, triton_p95_ms=tr_p95,
        triton_l2_err=tr_l2, triton_max_err=tr_max,
        triton_speedup=pt_med / tr_med if tr_med > 0 else 0,
    )

    if _HAS_CUDA:
        cu_dB = mlp_cuda.matmul_transA(A.detach(), dC)
        cu_l2, cu_max = _errors(ref_dB, cu_dB)
        cu_med, cu_p95 = _bench(mlp_cuda.matmul_transA, A.detach(), dC, warmup=warmup, iters=iters)
        r_dB.cuda_ms = cu_med
        r_dB.cuda_p95_ms = cu_p95
        r_dB.cuda_l2_err = cu_l2
        r_dB.cuda_max_err = cu_max
        r_dB.cuda_speedup = pt_med / cu_med if cu_med > 0 else 0

    results.append(r_dB)
    return results


def bench_fused_mlp_first(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    M, K, N = size["M"], size["K"], size["N"]
    X = torch.randn(M, K, device="cuda", dtype=_CURRENT_DTYPE)
    W = torch.randn(K, N, device="cuda", dtype=_CURRENT_DTYPE)
    bias = torch.randn(N, device="cuda", dtype=_CURRENT_DTYPE)

    # PyTorch reference: matmul + bias + GELU
    ref = torch.nn.functional.gelu(torch.matmul(X, W) + bias, approximate="tanh")

    # Triton fused
    tr_out = mlp_first_layer_triton(X, W, bias)
    tr_l2, tr_max = _errors(ref, tr_out)
    tr_med, tr_p95 = _bench(mlp_first_layer_triton, X, W, bias, warmup=warmup, iters=iters)

    # PyTorch unfused
    def _pt_unfused():
        return torch.nn.functional.gelu(torch.matmul(X, W) + bias, approximate="tanh")

    pt_med, pt_p95 = _bench(_pt_unfused, warmup=warmup, iters=iters)

    r = BenchResult(
        name="fused_mlp_first", shapes=f"GELU(({M},{K})@({K},{N})+{N})",
        pytorch_ms=pt_med, triton_ms=tr_med,
        pytorch_p95_ms=pt_p95, triton_p95_ms=tr_p95,
        triton_l2_err=tr_l2, triton_max_err=tr_max,
        triton_speedup=pt_med / tr_med if tr_med > 0 else 0,
    )

    if _HAS_CUDA:
        cu_out = mlp_cuda.mlp_fused_first_layer(X, W, bias)
        cu_l2, cu_max = _errors(ref, cu_out)
        cu_med, cu_p95 = _bench(mlp_cuda.mlp_fused_first_layer, X, W, bias, warmup=warmup, iters=iters)
        r.cuda_ms = cu_med
        r.cuda_p95_ms = cu_p95
        r.cuda_l2_err = cu_l2
        r.cuda_max_err = cu_max
        r.cuda_speedup = pt_med / cu_med if cu_med > 0 else 0

    return [r]


def bench_swiglu(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    n = size["elem"]
    gate = torch.randn(n, device="cuda", dtype=_CURRENT_DTYPE)
    up = torch.randn(n, device="cuda", dtype=_CURRENT_DTYPE)

    ref = torch.nn.functional.silu(gate) * up

    tr_out = swiglu_triton(gate, up)
    tr_l2, tr_max = _errors(ref, tr_out)
    tr_med, tr_p95 = _bench(swiglu_triton, gate, up, warmup=warmup, iters=iters)

    def _pt_swiglu():
        return torch.nn.functional.silu(gate) * up

    pt_med, pt_p95 = _bench(_pt_swiglu, warmup=warmup, iters=iters)

    r = BenchResult(
        name="swiglu", shapes=f"SiLU({n},)*({n},)",
        pytorch_ms=pt_med, triton_ms=tr_med,
        pytorch_p95_ms=pt_p95, triton_p95_ms=tr_p95,
        triton_l2_err=tr_l2, triton_max_err=tr_max,
        triton_speedup=pt_med / tr_med if tr_med > 0 else 0,
    )

    if _HAS_CUDA:
        cu_out = mlp_cuda.swiglu_fused(gate, up)
        cu_l2, cu_max = _errors(ref, cu_out)
        cu_med, cu_p95 = _bench(mlp_cuda.swiglu_fused, gate, up, warmup=warmup, iters=iters)
        r.cuda_ms = cu_med
        r.cuda_p95_ms = cu_p95
        r.cuda_l2_err = cu_l2
        r.cuda_max_err = cu_max
        r.cuda_speedup = pt_med / cu_med if cu_med > 0 else 0

    return [r]


def bench_bias_add_relu(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    M, N = size["M"], size["N"]
    x = torch.randn(M, N, device="cuda", dtype=_CURRENT_DTYPE)
    bias = torch.randn(N, device="cuda", dtype=_CURRENT_DTYPE)

    ref = torch.relu(x + bias)

    tr_out = triton_bias_add_relu(x, bias)
    tr_l2, tr_max = _errors(ref, tr_out)
    tr_med, tr_p95 = _bench(triton_bias_add_relu, x, bias, warmup=warmup, iters=iters)

    pt_med, pt_p95 = _bench(lambda a, b: torch.relu(a + b), x, bias, warmup=warmup, iters=iters)

    r = BenchResult(
        name="bias_add_relu", shapes=f"ReLU(({M},{N})+{N})",
        pytorch_ms=pt_med, triton_ms=tr_med,
        pytorch_p95_ms=pt_p95, triton_p95_ms=tr_p95,
        triton_l2_err=tr_l2, triton_max_err=tr_max,
        triton_speedup=pt_med / tr_med if tr_med > 0 else 0,
    )
    # CUDA 没有 fused bias_add_relu，跳过
    return [r]


def bench_conv2d(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    N = size["N"]
    C_in, C_out = size["C_in"], size["C_out"]
    H, W = size["H"], size["W"]
    K = size["K"]
    stride, pad = size["stride"], size["pad"]

    x = torch.randn(N, C_in, H, W, device="cuda", dtype=_CURRENT_DTYPE)
    weight = torch.randn(C_out, C_in, K, K, device="cuda", dtype=_CURRENT_DTYPE)
    bias = torch.randn(C_out, device="cuda", dtype=_CURRENT_DTYPE)

    # PyTorch reference
    conv_pt = torch.nn.Conv2d(C_in, C_out, K, stride=stride, padding=pad, bias=True).to("cuda")
    conv_pt.weight.data.copy_(weight)
    conv_pt.bias.data.copy_(bias)
    ref = conv_pt(x)

    # Triton
    tr_out = conv2d_triton(x, weight, bias, stride=stride, padding=pad)
    tr_l2, tr_max = _errors(ref, tr_out)
    tr_med, tr_p95 = _bench(conv2d_triton, x, weight, bias, warmup=warmup, iters=iters,
                             stride=stride, padding=pad)

    pt_med, pt_p95 = _bench(conv_pt, x, warmup=warmup, iters=iters)

    r = BenchResult(
        name="conv2d", shapes=f"({N},{C_in},{H},{W})*{K}k->{C_out}ch",
        pytorch_ms=pt_med, triton_ms=tr_med,
        pytorch_p95_ms=pt_p95, triton_p95_ms=tr_p95,
        triton_l2_err=tr_l2, triton_max_err=tr_max,
        triton_speedup=pt_med / tr_med if tr_med > 0 else 0,
    )

    if _HAS_CUDA:
        cu_out = mlp_cuda.conv2d(x, weight, bias, stride, pad)
        cu_l2, cu_max = _errors(ref, cu_out)
        cu_med, cu_p95 = _bench(mlp_cuda.conv2d, x, weight, bias, stride, pad,
                                 warmup=warmup, iters=iters)
        r.cuda_ms = cu_med
        r.cuda_p95_ms = cu_p95
        r.cuda_l2_err = cu_l2
        r.cuda_max_err = cu_max
        r.cuda_speedup = pt_med / cu_med if cu_med > 0 else 0

    return [r]


def bench_maxpool2d(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    N = size["N"]
    C, H, W = size["C"], size["H"], size["W"]
    K, stride, pad = size["K"], size["stride"], size["pad"]

    x = torch.randn(N, C, H, W, device="cuda", dtype=_CURRENT_DTYPE)

    ref = torch.nn.functional.max_pool2d(x, K, stride=stride, padding=pad)

    tr_out = triton_maxpool2d(x, K, stride=stride, padding=pad)
    tr_l2, tr_max = _errors(ref, tr_out)
    tr_med, tr_p95 = _bench(triton_maxpool2d, x, K, warmup=warmup, iters=iters,
                             stride=stride, padding=pad)

    pt_med, pt_p95 = _bench(
        torch.nn.functional.max_pool2d, x, K,
        warmup=warmup, iters=iters, stride=stride, padding=pad)

    r = BenchResult(
        name="maxpool2d", shapes=f"({N},{C},{H},{W})*{K}k",
        pytorch_ms=pt_med, triton_ms=tr_med,
        pytorch_p95_ms=pt_p95, triton_p95_ms=tr_p95,
        triton_l2_err=tr_l2, triton_max_err=tr_max,
        triton_speedup=pt_med / tr_med if tr_med > 0 else 0,
    )

    if _HAS_CUDA:
        cu_out = mlp_cuda.maxpool2d(x, K, stride, pad)
        cu_l2, cu_max = _errors(ref, cu_out)
        cu_med, cu_p95 = _bench(mlp_cuda.maxpool2d, x, K, stride, pad,
                                 warmup=warmup, iters=iters)
        r.cuda_ms = cu_med
        r.cuda_p95_ms = cu_p95
        r.cuda_l2_err = cu_l2
        r.cuda_max_err = cu_max
        r.cuda_speedup = pt_med / cu_med if cu_med > 0 else 0

    return [r]


def bench_avgpool2d(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    N = size["N"]
    C, H, W = size["C"], size["H"], size["W"]
    K, stride, pad = size["K"], size["stride"], size["pad"]

    x = torch.randn(N, C, H, W, device="cuda", dtype=_CURRENT_DTYPE)

    ref = torch.nn.functional.avg_pool2d(x, K, stride=stride, padding=pad)

    tr_out = triton_avgpool2d(x, K, stride=stride, padding=pad)
    tr_l2, tr_max = _errors(ref, tr_out)
    tr_med, tr_p95 = _bench(triton_avgpool2d, x, K, warmup=warmup, iters=iters,
                             stride=stride, padding=pad)

    pt_med, pt_p95 = _bench(
        torch.nn.functional.avg_pool2d, x, K,
        warmup=warmup, iters=iters, stride=stride, padding=pad)

    r = BenchResult(
        name="avgpool2d", shapes=f"({N},{C},{H},{W})*{K}k",
        pytorch_ms=pt_med, triton_ms=tr_med,
        pytorch_p95_ms=pt_p95, triton_p95_ms=tr_p95,
        triton_l2_err=tr_l2, triton_max_err=tr_max,
        triton_speedup=pt_med / tr_med if tr_med > 0 else 0,
    )

    if _HAS_CUDA:
        cu_out = mlp_cuda.avgpool2d(x, K, stride, pad)
        cu_l2, cu_max = _errors(ref, cu_out)
        cu_med, cu_p95 = _bench(mlp_cuda.avgpool2d, x, K, stride, pad,
                                 warmup=warmup, iters=iters)
        r.cuda_ms = cu_med
        r.cuda_p95_ms = cu_p95
        r.cuda_l2_err = cu_l2
        r.cuda_max_err = cu_max
        r.cuda_speedup = pt_med / cu_med if cu_med > 0 else 0

    return [r]


def bench_softmax(size: dict, warmup: int, iters: int) -> list[BenchResult]:
    M, N = size["M"], size["N"]
    x = torch.randn(M, N, device="cuda", dtype=_CURRENT_DTYPE)

    ref = torch.nn.functional.softmax(x, dim=1)

    tr_out = triton_softmax(x)
    tr_l2, tr_max = _errors(ref, tr_out)
    tr_med, tr_p95 = _bench(triton_softmax, x, warmup=warmup, iters=iters)

    pt_med, pt_p95 = _bench(torch.nn.functional.softmax, x, 1, warmup=warmup, iters=iters)

    r = BenchResult(
        name="softmax", shapes=f"({M},{N})",
        pytorch_ms=pt_med, triton_ms=tr_med,
        pytorch_p95_ms=pt_p95, triton_p95_ms=tr_p95,
        triton_l2_err=tr_l2, triton_max_err=tr_max,
        triton_speedup=pt_med / tr_med if tr_med > 0 else 0,
    )

    if _HAS_CUDA:
        cu_out = mlp_cuda.softmax(x)
        cu_l2, cu_max = _errors(ref, cu_out)
        cu_med, cu_p95 = _bench(mlp_cuda.softmax, x, warmup=warmup, iters=iters)
        r.cuda_ms = cu_med
        r.cuda_p95_ms = cu_p95
        r.cuda_l2_err = cu_l2
        r.cuda_max_err = cu_max
        r.cuda_speedup = pt_med / cu_med if cu_med > 0 else 0

    return [r]


# ============================================================
# 算子注册表
# ============================================================

OP_REGISTRY = {
    "matmul": bench_matmul,
    "gelu": bench_gelu,
    "relu": bench_relu,
    "silu": bench_silu,
    "gelu_backward": bench_gelu_backward,
    "relu_backward": bench_relu_backward,
    "silu_backward": bench_silu_backward,
    "bias_add": bench_bias_add,
    "matmul_backward": bench_matmul_backward,
    "fused_mlp_first": bench_fused_mlp_first,
    "swiglu": bench_swiglu,
    "bias_add_relu": bench_bias_add_relu,
    "conv2d": bench_conv2d,
    "maxpool": bench_maxpool2d,
    "avgpool": bench_avgpool2d,
    "softmax": bench_softmax,
}


# ============================================================
# 输出
# ============================================================

def _fmt_err(val: float) -> str:
    if val < 1e-10:
        return "0.000"
    elif val < 1e-3:
        return f"{val:.2e}"
    else:
        return f"{val:.4f}"


def _fmt_speedup(val: float) -> str:
    if val >= 1.0:
        return f"{val:.2f}x"
    else:
        return f"{val:.2f}x"


def print_results(results: list[BenchResult]):
    has_cuda = any(r.cuda_ms > 0 for r in results)

    if has_cuda:
        header = (
            f"  {'Operator':<24} {'Shapes':<28} | "
            f"{'PT ms':>8} {'Tr ms':>8} {'CU ms':>8} | "
            f"{'Tr/PT':>7} {'CU/PT':>7} | "
            f"{'Tr L2':>10} {'Tr Max':>10} {'CU L2':>10} {'CU Max':>10}"
        )
    else:
        header = (
            f"  {'Operator':<24} {'Shapes':<28} | "
            f"{'PT ms':>8} {'Tr ms':>8} | "
            f"{'Tr/PT':>7} | "
            f"{'Tr L2':>10} {'Tr Max':>10}"
        )
    print(f"\n{'='*len(header)}")
    print(f"  Operator-Level Comparison: PyTorch vs Triton" + (" vs CUDA" if has_cuda else ""))
    print(f"{'='*len(header)}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        if has_cuda:
            print(
                f"  {r.name:<24} {r.shapes:<28} | "
                f"{r.pytorch_ms:8.3f} {r.triton_ms:8.3f} {r.cuda_ms:8.3f} | "
                f"{_fmt_speedup(r.triton_speedup):>7} {_fmt_speedup(r.cuda_speedup):>7} | "
                f"{_fmt_err(r.triton_l2_err):>10} {_fmt_err(r.triton_max_err):>10} "
                f"{_fmt_err(r.cuda_l2_err):>10} {_fmt_err(r.cuda_max_err):>10}"
            )
        else:
            print(
                f"  {r.name:<24} {r.shapes:<28} | "
                f"{r.pytorch_ms:8.3f} {r.triton_ms:8.3f} | "
                f"{_fmt_speedup(r.triton_speedup):>7} | "
                f"{_fmt_err(r.triton_l2_err):>10} {_fmt_err(r.triton_max_err):>10}"
            )

    print()

    # 汇总
    print("  Summary (speedup vs PyTorch, median latency):")
    tr_speedups = [r.triton_speedup for r in results if r.triton_speedup > 0]
    cu_speedups = [r.cuda_speedup for r in results if r.cuda_speedup > 0]
    if tr_speedups:
        print(f"    Triton avg: {np.mean(tr_speedups):.2f}x  "
              f"min: {np.min(tr_speedups):.2f}x  max: {np.max(tr_speedups):.2f}x")
    if cu_speedups:
        print(f"    CUDA   avg: {np.mean(cu_speedups):.2f}x  "
              f"min: {np.min(cu_speedups):.2f}x  max: {np.max(cu_speedups):.2f}x")


def export_json(results: list[BenchResult], path: str, metadata: dict | None = None):
    rows = []
    for r in results:
        d = {
            "name": r.name,
            "shapes": r.shapes,
            "dtype": r.dtype,
            "pytorch_ms": r.pytorch_ms,
            "triton_ms": r.triton_ms,
            "pytorch_p95_ms": r.pytorch_p95_ms,
            "triton_p95_ms": r.triton_p95_ms,
            "triton_l2_err": r.triton_l2_err,
            "triton_max_err": r.triton_max_err,
            "triton_speedup": r.triton_speedup,
        }
        if r.cuda_ms > 0:
            d.update({
                "cuda_ms": r.cuda_ms,
                "cuda_p95_ms": r.cuda_p95_ms,
                "cuda_l2_err": r.cuda_l2_err,
                "cuda_max_err": r.cuda_max_err,
                "cuda_speedup": r.cuda_speedup,
            })
        if r.cutile_ms > 0:
            d.update({
                "cutile_ms": r.cutile_ms,
                "cutile_p95_ms": r.cutile_p95_ms,
                "cutile_l2_err": r.cutile_l2_err,
                "cutile_max_err": r.cutile_max_err,
                "cutile_speedup": r.cutile_speedup,
            })
        if r.flops > 0:
            d.update({
                "flops": r.flops,
                "bytes_io": r.bytes_io,
                "pytorch_tflops": r.pytorch_tflops,
                "triton_tflops": r.triton_tflops,
                "cuda_tflops": r.cuda_tflops,
                "pytorch_gbps": r.pytorch_gbps,
                "triton_gbps": r.triton_gbps,
                "cuda_gbps": r.cuda_gbps,
            })
        rows.append(d)

    # 新 schema: 顶部 metadata + rows[] (向后兼容: 旧 reader 读 rows 字段即可)
    payload = {"metadata": metadata or {}, "rows": rows}

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to: {path}")


# ============================================================
# 主入口
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Operator-level benchmark")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--sizes", type=str, default="small,medium",
                   help="Comma-separated: small,medium,large")
    p.add_argument("--ops", type=str, default=None,
                   help="Comma-separated operator names (default: all)")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--precision", type=str, default="tf32", choices=["tf32", "fp32"],
                   help="Precision mode: tf32 (tensor cores) or fp32 (strict)")
    p.add_argument("--dtypes", type=str, default="fp32",
                   help="Comma-separated dtypes to sweep: fp32,fp16,bf16")
    p.add_argument("--roofline", action="store_true",
                   help="Compute achieved TFLOPS / GB/s per row (matmul-class ops only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ref-tf32", action="store_true",
                   help="Bench_matmul reference uses TF32 (matches original baseline.json). "
                        "Default: ref uses FP32 (exposes backend-vs-cuBLAS-FP32 drift).")
    return p.parse_args()


def _annotate_roofline(results: list[BenchResult]) -> None:
    """对 matmul-class ops 填 flops / bytes_io / tflops / gbps。

    现仅处理: matmul / matmul_backward_dA / matmul_backward_dB / fused_mlp_first / conv2d.
    其他 elementwise ops 不算 TFLOPS (没意义) 但算 GB/s.
    """
    import re
    for r in results:
        # 解析 shapes 字符串拿到 M,K,N
        # matmul: "(M,K)@(K,N)"
        m = re.search(r"\((\d+),(\d+)\)@\((\d+),(\d+)\)", r.shapes)
        if m:
            M, K, _, N = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            elem_bytes = 2 if r.dtype in ("fp16", "bf16") else 4
            r.flops = 2 * M * K * N
            r.bytes_io = (M * K + K * N + M * N) * elem_bytes
            for backend in ("pytorch", "triton", "cuda"):
                ms = getattr(r, f"{backend}_ms")
                if ms > 0:
                    setattr(r, f"{backend}_tflops", r.flops / (ms * 1e-3) / 1e12)
                    setattr(r, f"{backend}_gbps", r.bytes_io / (ms * 1e-3) / 1e9)
            continue
        # elementwise: "(n,)"  -> bytes only
        m2 = re.search(r"^\((\d+),\)", r.shapes)
        if m2:
            n = int(m2.group(1))
            elem_bytes = 2 if r.dtype in ("fp16", "bf16") else 4
            r.bytes_io = 2 * n * elem_bytes  # 1 read + 1 write
            for backend in ("pytorch", "triton", "cuda"):
                ms = getattr(r, f"{backend}_ms")
                if ms > 0:
                    setattr(r, f"{backend}_gbps", r.bytes_io / (ms * 1e-3) / 1e9)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    torch.manual_seed(args.seed)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {args.warmup}, Iters: {args.iters}")
    if _HAS_CUDA:
        print("CUDA kernels: available")
    else:
        print("CUDA kernels: NOT available (mlp_cuda not installed)")
    if _HAS_CUTILE:
        print("cuTile kernels: available")
    else:
        print("cuTile kernels: NOT available (cuda-tile not installed)")

    # 精度配置
    if args.precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        precision.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        precision.allow_tf32 = True
    print(f"Precision: {args.precision}")

    # dtype sweep
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    requested_dtypes = []
    for s in args.dtypes.split(","):
        s = s.strip()
        if s in dtype_map:
            requested_dtypes.append((s, dtype_map[s]))
        else:
            print(f"WARNING: unknown dtype '{s}', skipping")
    if not requested_dtypes:
        requested_dtypes = [("fp32", torch.float32)]
    print(f"Dtypes:    {[d[0] for d in requested_dtypes]}")
    print(f"Roofline:  {'on' if args.roofline else 'off'}")

    # 选择尺寸
    size_names = [s.strip() for s in args.sizes.split(",")]
    sizes = []
    for name in size_names:
        if name in _SIZE_PRESETS:
            sizes.extend(_SIZE_PRESETS[name])
        else:
            print(f"WARNING: unknown size preset '{name}', skipping")

    # 选择算子
    if args.ops:
        op_names = [s.strip() for s in args.ops.split(",")]
    else:
        op_names = list(OP_REGISTRY.keys())

    # 运行 (dtype 外层循环, 写全局 _CURRENT_DTYPE)
    global _CURRENT_DTYPE
    all_results: list[BenchResult] = []
    for dtype_name, dtype_obj in requested_dtypes:
        _CURRENT_DTYPE = dtype_obj
        print(f"\n--- dtype={dtype_name} ---")
        for op_name in op_names:
            if op_name not in OP_REGISTRY:
                print(f"WARNING: unknown op '{op_name}', skipping")
                continue
            # bf16/fp16 跳过 conv/pool (op 内部 hardcode fp32)
            if dtype_name != "fp32" and op_name in ("conv2d", "maxpool", "avgpool"):
                continue
            bench_fn = OP_REGISTRY[op_name]
            for size in sizes:
                try:
                    results = bench_fn(size, args.warmup, args.iters)
                except Exception as e:
                    print(f"  [{op_name}] {size} dtype={dtype_name} FAILED: {e}")
                    continue
                for r in results:
                    r.dtype = dtype_name
                all_results.extend(results)

    if args.roofline:
        _annotate_roofline(all_results)

    print_results(all_results)

    md = capture_metadata(args)
    md["sizes_resolved"] = sizes
    md["ops"] = op_names
    md["has_cuda"] = _HAS_CUDA
    md["has_cutile"] = _HAS_CUTILE
    md["roofline"] = args.roofline
    md["ref_tf32"] = args.ref_tf32

    # AC: --ref-tf32 切老基线 (ref 用 TF32), 与原始 baseline.json 可比.
    # 默认 False = ref 严格 FP32, 暴露 backend 真实数值偏差.
    sys.modules[__name__]._REF_TF32 = args.ref_tf32

    if args.output:
        export_json(all_results, args.output, metadata=md)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        export_json(all_results, f"results/op_bench_{ts}.json", metadata=md)


if __name__ == "__main__":
    main()
