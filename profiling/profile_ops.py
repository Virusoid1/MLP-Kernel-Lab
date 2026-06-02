"""
算子级 profiling driver (覆盖三个自研 backend + PyTorch baseline)

与 nsys / ncu 配合使用:
    nsys profile -t cuda,nvtx -o results/nsys_ops python profiling/profile_ops.py
    ncu --set full --nvtx --nvtx-include "triton/*" \
        -o results/ncu_ops python profiling/profile_ops.py --backend triton

每个算子 / backend 用 NVTX range 包裹,Nsight 中可按 backend 折叠时间线.

用法:
    python profiling/profile_ops.py                       # 全部 backend
    python profiling/profile_ops.py --backend cuda triton # 仅指定 backend
    python profiling/profile_ops.py --ops matmul gelu     # 仅指定算子
    python profiling/profile_ops.py --M 1024 --N 1024 --K 1024
    python profiling/profile_ops.py --export-trace        # 导出 Chrome trace
"""

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@contextmanager
def nvtx(label: str):
    """NVTX range wrapper(失败回退为 no-op,方便无 NVTX 环境运行)。"""
    try:
        torch.cuda.nvtx.range_push(label)
        yield
    finally:
        try:
            torch.cuda.nvtx.range_pop()
        except Exception:
            pass


def time_fn(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ============================================================
# Backend 注册
# ============================================================

def _load_backend(name):
    """按需 import 各 backend,缺失时返回 None。"""
    if name == "pytorch":
        return {"matmul": lambda a, b: a @ b,
                "gelu": torch.nn.functional.gelu,
                "layernorm": lambda x, g, b: torch.nn.functional.layer_norm(x, x.shape[-1:], g, b)}
    if name == "triton":
        try:
            from triton_kernels import tiled_matmul, gelu as triton_gelu
            from triton_kernels.layernorm import layernorm_triton
            return {"matmul": tiled_matmul,
                    "gelu": triton_gelu,
                    "layernorm": layernorm_triton}
        except ImportError as e:
            print(f"[skip] triton backend: {e}")
            return None
    if name == "cuda":
        try:
            import mlp_cuda
            return {"matmul": mlp_cuda.matmul_tiled_auto,
                    "gelu": mlp_cuda.gelu,
                    "layernorm": lambda x, g, b: mlp_cuda.layernorm_forward(x, g, b, 1e-5)[0]}
        except ImportError as e:
            print(f"[skip] cuda backend: {e} (run `make install` first)")
            return None
    if name == "cutile":
        try:
            from cutile_kernels.matmul import tiled_matmul as ct_matmul
            from cutile_kernels.elementwise import gelu as ct_gelu
            from cutile_kernels.layernorm import layernorm_cutile
            return {"matmul": ct_matmul,
                    "gelu": ct_gelu,
                    "layernorm": layernorm_cutile}
        except ImportError as e:
            print(f"[skip] cutile backend: {e} (pip install cuda-tile)")
            return None
    raise ValueError(f"Unknown backend: {name}")


ALL_BACKENDS = ["pytorch", "triton", "cuda", "cutile"]
ALL_OPS = ["matmul", "gelu", "layernorm"]


def run_one(backend_name, ops_dict, op_name, M, K, N, warmup, iters):
    """单 (backend, op) 测量,带 NVTX,返回平均延迟 ms。"""
    label = f"{backend_name}/{op_name}"
    if op_name == "matmul":
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        fn = lambda: ops_dict["matmul"](a, b)
    elif op_name == "gelu":
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        fn = lambda: ops_dict["gelu"](x)
    elif op_name == "layernorm":
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        gamma = torch.ones(N, device="cuda", dtype=torch.float32)
        beta = torch.zeros(N, device="cuda", dtype=torch.float32)
        fn = lambda: ops_dict["layernorm"](x, gamma, beta)
    else:
        raise ValueError(op_name)

    with nvtx(label):
        ms = time_fn(fn, warmup=warmup, iters=iters)
    return ms


def main():
    parser = argparse.ArgumentParser(description="Per-op profiling driver across backends")
    parser.add_argument("--backend", nargs="+", default=ALL_BACKENDS,
                        choices=ALL_BACKENDS, help="Backends to profile")
    parser.add_argument("--ops", nargs="+", default=ALL_OPS,
                        choices=ALL_OPS, help="Ops to profile")
    parser.add_argument("--M", type=int, default=512)
    parser.add_argument("--K", type=int, default=768)
    parser.add_argument("--N", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--export-trace", action="store_true",
                        help="Export Chrome trace via torch.profiler")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"M={args.M} K={args.K} N={args.N}  warmup={args.warmup} iters={args.iters}")

    backends = {b: _load_backend(b) for b in args.backend}
    backends = {b: d for b, d in backends.items() if d is not None}
    if not backends:
        print("No backend available."); sys.exit(1)

    rows = []  # (op, backend, ms)
    if args.export_trace:
        results_dir = Path("results"); results_dir.mkdir(exist_ok=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for op in args.ops:
                for backend, ops_dict in backends.items():
                    ms = run_one(backend, ops_dict, op, args.M, args.K, args.N,
                                 args.warmup, args.iters)
                    rows.append((op, backend, ms))
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = results_dir / f"trace_ops_{ts}.json.gz"
        prof.export_chrome_trace(str(out))
        print(f"\nChrome trace: {out}")
    else:
        for op in args.ops:
            for backend, ops_dict in backends.items():
                ms = run_one(backend, ops_dict, op, args.M, args.K, args.N,
                             args.warmup, args.iters)
                rows.append((op, backend, ms))

    # 汇总表格
    print(f"\n{'op':<12} {'backend':<10} {'ms':>10}")
    print("-" * 36)
    for op, backend, ms in rows:
        print(f"{op:<12} {backend:<10} {ms:>10.4f}")


if __name__ == "__main__":
    main()
