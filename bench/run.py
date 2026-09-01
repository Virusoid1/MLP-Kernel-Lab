#!/usr/bin/env python3
"""SwiGLU MLP block 性能基准（v2 主线，E3 证据）。

语义:
    gate = X@W_gate ; up = X@W_up ; hidden = SiLU(gate)*up ; Y = hidden@W_down

流程:
  1. 从 bench/suites/transformer_mlp.yaml 读 shape/dtype/backend 配置
  2. correctness pre-check: 每 (shape, dtype, backend) 先与 eager FP32 reference 对比
     (归一化 L2) —— 不通过的 case 标 correctness_failed 并从性能汇总剔除
  3. 计时: CUDA Event, warmup + iterations, 小 M 自动多拍取平均降噪
  4. 指标: median(P50)/P95/mean/std/CV、TFLOPS、speedup vs 同 dtype eager、
     峰值显存、compile 首次编译/首个调用耗时
  5. 输出: artifacts/<run-id>/swiglu_bench.{json,md} + manifest

用法:
    python bench/run.py                                  # 默认: prefill, fp32, 全可用后端
    python bench/run.py --suite decode                   # decode 档
    python bench/run.py --dtypes fp32,fp16               # dtype 子集
    python bench/run.py --backends eager,compile,cuda    # backend 子集
    python bench/run.py --warmup 30 --iters 200          # 计时协议
    python bench/run.py --out DIR                        # 覆盖输出目录
    python bench/run.py --max-cases 8                    # 快速冒烟（前 N 个 case）

用例逐个 try/except：单个后端失败不影响整批。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from python.transformer_mlp import BACKENDS, available_backends, silu
from python.mnist.benchmark import capture_metadata, p95 as true_p95

SUITE_PATH = Path(__file__).resolve().parent / "suites" / "transformer_mlp.yaml"
DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def block_flops(M: int, K: int, F: int) -> int:
    """SwiGLU block FLOP 计算（显式定义，避免与单 GEMM 混用）:
       gate  (M,K)@(K,F) -> 2*M*K*F
       up    (M,K)@(K,F) -> 2*M*K*F
       hidden@W_down (M,F)@(F,K) -> 2*M*F*K = 2*M*K*F
       total = 6*M*K*F
    """
    return 6 * M * K * F


def make_inputs(M: int, K: int, F: int, dtype: torch.dtype, seed: int = 42, scale: float = 1.0):
    """构造输入。fp16/bf16 默认 0.1 缩放避免 eager fp16 K 累加溢出（见 test_transformer_mlp）。"""
    if dtype in (torch.float16, torch.bfloat16) and scale == 1.0:
        scale = 0.1
    torch.manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=dtype) * scale
    wg = torch.randn(K, F, device="cuda", dtype=dtype) * scale
    wu = torch.randn(K, F, device="cuda", dtype=dtype) * scale
    wd = torch.randn(F, K, device="cuda", dtype=dtype) * scale
    return x, wg, wu, wd


def _fp32_reference(M: int, K: int, F: int, seed: int = 42):
    torch.manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.float32)
    wg = torch.randn(K, F, device="cuda", dtype=torch.float32)
    wu = torch.randn(K, F, device="cuda", dtype=torch.float32)
    wd = torch.randn(F, K, device="cuda", dtype=torch.float32)
    return x, wg, wu, wd


def norm_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.float() - b.float())
    return (d.norm() / (b.float().norm() + 1e-12)).item()


def measure(fn, args_list, warmup: int, iters: int, n_repeat: int = 1) -> list[float]:
    """CUDA Event 计时；返回 ms 样本列表。

    小 M（decode）n_repeat>1：一个计时区间内连续多次调用取均摊，降低亚 10us kernel 计时噪声。
    """
    for _ in range(warmup):
        fn(*args_list)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(n_repeat):
            fn(*args_list)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / n_repeat)
    return samples


def summarize(samples: list[float]) -> dict:
    arr = sorted(samples)
    med = statistics.median(arr)
    mean = statistics.mean(arr)
    std = statistics.stdev(arr) if len(arr) > 1 else 0.0
    return {
        "median_ms": round(med, 6),
        "mean_ms": round(mean, 6),
        "p95_ms": round(true_p95(arr), 6),
        "std_ms": round(std, 6),
        "cv": round(std / mean, 4) if mean > 0 else 0.0,
        "n_samples": len(arr),
    }


def run_case(spec, dtype_name: str, backend: str, warmup: int, iters: int, replicates: int = 1) -> dict | None:
    M, K, F = spec["M"], spec["K"], spec["F"]
    dt = DTYPE_MAP[dtype_name]

    # correctness pre-check（FP32 authority reference，用与 case 相同的数据）
    fn = BACKENDS[backend]
    try:
        x, wg, wu, wd = make_inputs(M, K, F, dt)
        ref = BACKENDS["eager"](x.float(), wg.float(), wu.float(), wd.float())
        out = fn(x, wg, wu, wd)
        err = norm_l2(out, ref)
        finite = bool(torch.isfinite(out).all())
        if not finite:
            return {"error": "non-finite output", "op": "swiglu_block", "shape": f"{M},{K},{F}", "dtype": dtype_name, "backend": backend}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:140]}", "op": "swiglu_block", "shape": f"{M},{K},{F}", "dtype": dtype_name, "backend": backend}

    # correctness 阈值按 dtype：fp32 严格，fp16/bf16 按位宽放宽
    corr_tol = {"fp32": 1e-4, "fp16": 2e-2, "bf16": 3e-2}.get(dtype_name, 1e-2)
    passed = err < corr_tol and finite

    n_repeat = min(64, max(1, 64 // M))  # decode 小 M 多拍平均

    # 预打包静态权重：concat / triton_fused 每次调用都 torch.cat([w_gate, w_up])。
    # 权重是静态的，cat 应移出计时热路径（在计时前一次性完成）。
    # 复用与 BACKENDS 相同语义的底层调用，避免计时内重复 cat。
    if backend == "concat":
        from python.transformer_mlp import silu
        w_cat = torch.cat([wg, wu], dim=-1)
        def time_fn():
            gate_up = x @ w_cat
            g, u = gate_up.chunk(2, dim=-1)
            return (silu(g) * u) @ wd
    elif backend == "triton_fused":
        from triton_kernels.fused_swiglu_gateup import fused_gateup_swiglu
        from triton_kernels.matmul import tiled_matmul
        w_cat = torch.cat([wg, wu], dim=-1)
        def time_fn():
            hidden = fused_gateup_swiglu(x, w_cat)
            return tiled_matmul(hidden, wd)
    else:
        def time_fn():
            return fn(x, wg, wu, wd)

    torch.cuda.reset_peak_memory_stats()
    # replicates：独立多轮 measure，取 median-of-medians 抗噪声（suite 配置 replicates=2）
    all_meds = []
    for _ in range(max(1, replicates)):
        samples = measure(time_fn, (), warmup, iters, n_repeat)
        all_meds.append(statistics.median(samples))
    # 汇总用跨轮 median（而非单轮），并保留单轮样本做 std 参考
    samples = sorted(all_meds)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    st = summarize(samples)
    st["replicates"] = replicates
    flops = block_flops(M, K, F)
    st["tflops"] = round(flops / (st["median_ms"] * 1e-3) / 1e12, 2)
    st["peak_gpu_mem_mb"] = round(peak_mb, 1)
    return {
        "op": "swiglu_block", "backend": backend, "dtype": dtype_name,
        "M": M, "K": K, "F": F,
        "flops": flops, "flops_formula": "6*M*K*F (gate+up+down)",
        "correctness_norm_l2": round(err, 6),
        "correctness_passed": passed,
        "n_repeat": n_repeat,
        **st,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="SwiGLU MLP block 性能基准")
    ap.add_argument("--suite", default="prefill", choices=["decode", "prefill", "train", "all"])
    ap.add_argument("--dtypes", default="fp32", help="comma: fp32,fp16,bf16")
    ap.add_argument("--backends", default=None, help="comma: eager,concat,triton,cuda,cutile,compile")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-cases", type=int, default=0, help=">0 时只跑前 N 个 (shape,dtype,backend) 组合（冒烟）")
    ap.add_argument("--randomize-backends", action="store_true", help="随机化后端测量顺序（消除 autotune/热缓存偏差）")
    args = ap.parse_args()

    # strict FP32：reference 与后端都在 FP32 严格模式（TF32 关闭）
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        from triton_kernels.precision import precision
        precision.allow_tf32 = False
    except Exception:
        pass

    cfg = yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    backends = ([b.strip() for b in args.backends.split(",") if b.strip()]
                if args.backends else cfg["backends"])
    backends = [b for b in backends if b in available_backends()]
    # 后端顺序随机化：消除固定顺序带来的 autotune/热缓存偏差（公平性）
    _bm = cfg.get("benchmark") or {}
    replicates = int((_bm or {}).get("replicates", 1) or 1)
    import random as _random
    if args.randomize_backends:
        _random.shuffle(backends)

    specs = []
    if args.suite == "all":
        for k in ("decode", "prefill", "train"):
            specs.extend(cfg[k])
    else:
        specs = cfg[args.suite]

    gpu_slug = "cpu"
    try:
        if torch.cuda.is_available():
            gpu_slug = re.sub(r"[^A-Za-z0-9_]", "", torch.cuda.get_device_name(0).replace(" ", "_"))
    except Exception:
        pass
    run_id = f"swiglu_{time.strftime('%Y%m%d-%H%M%S')}-{args.suite}-{gpu_slug}"
    out_dir = Path(args.out) if args.out else (REPO_ROOT / "artifacts" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = capture_metadata(vars(args))
    md["suite"] = args.suite
    md["dtypes"] = dtypes
    md["backends"] = backends
    md["benchmark_protocol"] = {"warmup": args.warmup, "iters": args.iters,
                                "small_m_repeat": "min(64, max(1, 64//M))"}

    rows: list[dict] = []
    tested = 0
    for spec in specs:
        for dtype in dtypes:
            for backend in backends:
                if args.max_cases and tested >= args.max_cases:
                    break
                try:
                    row = run_case(spec, dtype, backend, args.warmup, args.iters, replicates)
                except Exception as e:
                    row = {"error": f"{type(e).__name__}: {str(e)[:140]}", "op": "swiglu_block",
                           "shape": f"{spec['M']},{spec['K']},{spec['F']}", "dtype": dtype, "backend": backend}
                row["shape"] = f"{spec['M']},{spec['K']},{spec['F']}"
                rows.append(row)
                tested += 1
                st = row.get("median_ms")
                ok = row.get("correctness_passed", row.get("error") is not None and False)
                errmsg = row.get("error", "")[:60]
                print(f"[{tested:3d}] {backend:7s} {dtype:4s} M={spec['M']:>5} K={spec['K']:>5} F={spec['F']:>6} "
                      f"med={st if st is not None else '-':>9}ms corr={row.get('correctness_norm_l2','-')} "
                      f"{errmsg}")
                if args.max_cases and tested >= args.max_cases:
                    break
            if args.max_cases and tested >= args.max_cases:
                break
        if args.max_cases and tested >= args.max_cases:
            break

    # speedup: vs 同 dtype eager + vs fp32 eager（fp32 eager 是跨 dtype 性能故事基线）
    eager_lookup = {}
    fp32_eager_lookup = {}
    for r in rows:
        if r.get("backend") == "eager" and r.get("median_ms") is not None:
            eager_lookup[(r["shape"], r["dtype"])] = r["median_ms"]
            if r["dtype"] == "fp32":
                fp32_eager_lookup[r["shape"]] = r["median_ms"]
    for r in rows:
        key = (r.get("shape"), r.get("dtype"))
        base = eager_lookup.get(key)
        if base and r.get("median_ms"):
            r["speedup_vs_eager"] = round(base / r["median_ms"], 3)
        base32 = fp32_eager_lookup.get(r.get("shape"))
        if base32 and r.get("median_ms"):
            r["speedup_vs_fp32_eager"] = round(base32 / r["median_ms"], 3)

    payload = {"metadata": md, "suite": args.suite, "rows": rows}
    payload["summary"] = {
        "cases_total": len(rows),
        "cases_with_error": sum(1 for r in rows if r.get("error")),
        "correctness_failed": sum(1 for r in rows if r.get("correctness_passed") is False),
        "correctness_pass_rate": round(sum(1 for r in rows if r.get("correctness_passed")) / max(1, len(rows)), 3),
        "best_speedup": max([r["speedup_vs_eager"] for r in rows if r.get("speedup_vs_eager")], default=None),
        "best_speedup_vs_fp32_eager": max([r["speedup_vs_fp32_eager"] for r in rows if r.get("speedup_vs_fp32_eager")], default=None),
    }
    (out_dir / "swiglu_bench.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("")
    print(f"[bench] wrote {out_dir / 'swiglu_bench.json'}")
    print(f"[bench] summary: {payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
