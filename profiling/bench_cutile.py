"""
cuTile benchmark driver (per-op + end-to-end MLP)

设计:
  - 启动时 GPU warmup (200 次 matmul 跑通 cublas / cache / clock)
  - 每个测项跑 4 轮;每轮 = warmup_iters + measure_iters,取该轮 median ms
  - 报告 4 轮里后 3 轮的平均(丢弃第 1 轮,避开 lazy compile / autotune)
  - 算子覆盖 cutile_kernels 公开的全部 op
  - 最后跑 CUTILEMLP 端到端 (训练 step + 推理 step)

用法 (在装有 PyTorch + cuda-tile 的环境内):
    python profiling/bench_cutile.py
    python profiling/bench_cutile.py --M 1024 --K 1024 --N 1024
    python profiling/bench_cutile.py --rounds 4 --warmup 20 --iters 100 --output results/cutile_bench.json

退出码:
  0 = 全部测项跑通
  2 = cuda-tile / mlp_cuda 未安装(预检失败)
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python.mnist.benchmark import capture_metadata  # noqa: E402


# ============================================================
# 计时 & GPU warmup
# ============================================================

def gpu_warmup(secs: float = 2.0):
    """跑足够多 matmul 把 GPU 提到 P0 + 加载 cublas + 暖 cache。"""
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(2)
    print(f"[warmup] GPU={torch.cuda.get_device_name(0)} for {secs}s...")
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    t0 = time.time()
    n = 0
    while time.time() - t0 < secs:
        c = a @ b
        n += 1
    torch.cuda.synchronize()
    print(f"[warmup] done after {n} matmul iters")


def time_round(fn, warmup_iters: int, measure_iters: int) -> float:
    """单轮: warmup_iters 次预热 + measure_iters 次测量, 返回 median ms。"""
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]
    for i in range(measure_iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) for i in range(measure_iters))
    return times[len(times) // 2]


def bench(name: str, fn, rounds: int, warmup_iters: int, measure_iters: int) -> dict:
    """跑 rounds 轮,返回 {round_ms: [...], discarded: r0, mean_last_3: x, std: y}。"""
    per_round = []
    for r in range(rounds):
        ms = time_round(fn, warmup_iters, measure_iters)
        per_round.append(ms)
        print(f"  [{name}] round {r+1}/{rounds}: {ms:.4f} ms")
    last_3 = per_round[1:]  # 丢第 1 轮
    mean = statistics.mean(last_3)
    std = statistics.stdev(last_3) if len(last_3) > 1 else 0.0
    return {
        "round_ms": per_round,
        "discarded": per_round[0],
        "mean_last_3_ms": mean,
        "std_last_3_ms": std,
    }


# ============================================================
# 预检 (确保 cuTile + mlp_cuda 可 import; 缺失立即报错退出)
# ============================================================

def precheck():
    try:
        from cutile_kernels import (
            cutile_matmul,
            bias_add as ct_bias_add,
            gelu as ct_gelu,
            silu as ct_silu,
            relu as ct_relu,
            matmul_backward_a as ct_dA,
            matmul_backward_b as ct_dB,
            gelu_backward as ct_gelu_b,
            layernorm_forward as ct_ln_fwd,
            layernorm_backward as ct_ln_bwd,
            mlp_first_layer_cutile as ct_mlp1,
            swiglu_cutile as ct_swiglu,
        )
    except ImportError as e:
        print(f"\n[FATAL] cuTile not importable: {e}")
        print("  Install: pip install cuda-tile")
        sys.exit(2)

    return {
        "matmul": cutile_matmul,
        "bias_add": ct_bias_add,
        "gelu": ct_gelu,
        "silu": ct_silu,
        "relu": ct_relu,
        "matmul_backward_a": ct_dA,
        "matmul_backward_b": ct_dB,
        "gelu_backward": ct_gelu_b,
        "layernorm": ct_ln_fwd,
        "layernorm_backward": ct_ln_bwd,
        "mlp_first_layer": ct_mlp1,
        "swiglu": ct_swiglu,
    }


# ============================================================
# 算子级 benchmark
# ============================================================

def bench_ops(ops, M, K, N, rounds, warmup, iters) -> dict:
    results = {}
    device, dtype = "cuda", torch.float32

    # ---- matmul: (M,K) @ (K,N) ----
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    results["matmul"] = bench("matmul", lambda: ops["matmul"](a, b),
                              rounds, warmup, iters)

    # ---- matmul backward: dA = dC @ B^T ----
    dC = torch.randn(M, N, device=device, dtype=dtype)
    results["matmul_backward_a"] = bench("matmul_backward_a",
                                          lambda: ops["matmul_backward_a"](dC, b),
                                          rounds, warmup, iters)
    # dB = A^T @ dC
    results["matmul_backward_b"] = bench("matmul_backward_b",
                                          lambda: ops["matmul_backward_b"](a, dC),
                                          rounds, warmup, iters)

    # ---- elementwise: bias_add / gelu / silu / relu ----
    x = torch.randn(M, N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)
    results["bias_add"] = bench("bias_add", lambda: ops["bias_add"](x, bias),
                                rounds, warmup, iters)
    results["gelu"] = bench("gelu", lambda: ops["gelu"](x), rounds, warmup, iters)
    results["silu"] = bench("silu", lambda: ops["silu"](x), rounds, warmup, iters)
    results["relu"] = bench("relu", lambda: ops["relu"](x), rounds, warmup, iters)

    # ---- activation backward ----
    dy = torch.randn(M, N, device=device, dtype=dtype)
    results["gelu_backward"] = bench("gelu_backward",
                                     lambda: ops["gelu_backward"](dy, x),
                                     rounds, warmup, iters)

    # ---- LayerNorm ----
    gamma = torch.ones(N, device=device, dtype=dtype)
    beta = torch.zeros(N, device=device, dtype=dtype)
    results["layernorm"] = bench(
        "layernorm", lambda: ops["layernorm"](x, gamma, beta),
        rounds, warmup, iters)
    # layernorm backward 需要 mean/rstd
    y_ln, mean_ln, rstd_ln = ops["layernorm"](x, gamma, beta)
    results["layernorm_backward"] = bench(
        "layernorm_backward",
        lambda: ops["layernorm_backward"](dy, x, gamma, mean_ln, rstd_ln),
        rounds, warmup, iters)

    # ---- fused: mlp_first_layer = GELU(X @ W + bias) ----
    X = torch.randn(M, K, device=device, dtype=dtype)
    W = torch.randn(K, N, device=device, dtype=dtype)
    results["mlp_fused_first_layer"] = bench(
        "mlp_fused_first_layer",
        lambda: ops["mlp_first_layer"](X, W, bias),
        rounds, warmup, iters)

    # ---- fused: swiglu (单参数 x, 实现为 x * sigmoid(x)) ----
    sx = torch.randn(M, N, device=device, dtype=dtype)
    results["swiglu"] = bench("swiglu", lambda: ops["swiglu"](sx),
                              rounds, warmup, iters)

    return results


# ============================================================
# 端到端 MLP
# ============================================================

def bench_mlp(rounds: int, warmup_iters: int, measure_iters: int) -> dict:
    """CUTILEMLP 训练 1 step + 推理 1 step,各跑 rounds 轮。"""
    from python.mnist.cutile_model import CUTILEMLP
    from python.mnist.model import MLPConfig

    config = MLPConfig(
        hidden_dims=[784, 1024, 512, 256, 10],
        activation="relu",
        dropout=0.1,
        use_layernorm=True,
    )
    model = CUTILEMLP(config).cuda()

    B = 256
    x = torch.randn(B, 784, device="cuda", dtype=torch.float32)
    y = torch.randint(0, 10, (B,), device="cuda")
    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def train_step():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()

    @torch.inference_mode()
    def infer_step():
        return model(x)

    out = {}
    model.train()
    out["mlp_train_step"] = bench("mlp_train_step", train_step,
                                  rounds, warmup_iters, measure_iters)
    model.eval()
    out["mlp_infer_step"] = bench("mlp_infer_step", infer_step,
                                  rounds, warmup_iters, measure_iters)
    # 吞吐(samples/sec)
    out["mlp_infer_throughput_samples_per_sec"] = \
        B / (out["mlp_infer_step"]["mean_last_3_ms"] * 1e-3)
    return out


# ============================================================
# cuTile matmul tile sweep（E4 / Blackwell 选 tile）
# ============================================================

TILE_CANDIDATES = [(16, 16, 16), (32, 32, 32), (32, 64, 32), (64, 64, 32),
                   (32, 32, 64), (64, 32, 32), (128, 64, 32)]


def tile_sweep_matmul(M, K, N, rounds, warmup, iters) -> tuple:
    """对 cutile matmul 测多种 TM/TN/TK，返回 (best_tile, ms, table)。

    通过 monkeypatch triton_kernels.gpu_utils.get_arch_params 的
    cutile_matmul_tile 字段实现（gpu_utils 内部有模块级缓存，逐 tile 打补丁）。
    """
    import triton_kernels.gpu_utils as gu
    from cutile_kernels.matmul import cutile_matmul
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    best = None
    table = []
    orig = gu.get_arch_params
    for T in TILE_CANDIDATES:
        def fake(T_=T):
            d = dict(orig())
            d["cutile_matmul_tile"] = T_
            return d
        gu.get_arch_params = fake
        try:
            res = bench("tilesweep", lambda: cutile_matmul(a, b), rounds, warmup, iters)
            ms = res["mean_last_3_ms"]  # 丢第 1 轮的均值，与 bench_cutile 全脚本一致
        except Exception as e:
            ms = None
            print(f"  tile={T} FAIL {str(e)[:40]}")
        if ms is not None:
            table.append((T, ms))
            if best is None or ms < best[1]:
                best = (T, ms)
    gu.get_arch_params = orig
    return best, table


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="cuTile benchmark (per-op + MLP)")
    parser.add_argument("--M", type=int, default=512)
    parser.add_argument("--K", type=int, default=768)
    parser.add_argument("--N", type=int, default=3072)
    parser.add_argument("--rounds", type=int, default=4,
                        help="共跑几轮 (后 N-1 轮取平均)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="每轮内的 warmup 次数")
    parser.add_argument("--iters", type=int, default=100,
                        help="每轮内的 measure 次数")
    parser.add_argument("--gpu-warmup-secs", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="results/cutile_bench.json")
    parser.add_argument("--skip-mlp", action="store_true",
                        help="只跑算子, 跳过 MLP 端到端")
    parser.add_argument("--tile-sweep", action="store_true",
                        help="对 cutile matmul 做 tile 单变量 sweep（E4/Blackwell 选 tile）")
    args = parser.parse_args()

    ops = precheck()
    gpu_warmup(args.gpu_warmup_secs)

    tile_sweep_result = None
    if args.tile_sweep:
        print(f"\n=== cuTile matmul tile sweep (M={args.M} K={args.K} N={args.N}) ===")
        best, table = tile_sweep_matmul(args.M, args.K, args.N,
                                        args.rounds, args.warmup, args.iters)
        tile_sweep_result = {"best_tile": list(best[0]), "best_ms": best[1],
                             "table": [[list(t), ms] for t, ms in table]}
        print(f"  best tile: {best[0]} at {best[1]:.4f} ms")
        for t, ms in table:
            mark = " <--" if t == best[0] else ""
            print(f"    {t}  {ms:.4f} ms{mark}")

    print(f"\n=== Per-op benchmark (M={args.M} K={args.K} N={args.N}) ===")
    op_results = bench_ops(ops, args.M, args.K, args.N,
                           args.rounds, args.warmup, args.iters)

    mlp_results = {}
    if not args.skip_mlp:
        print(f"\n=== End-to-end CUTILEMLP ===")
        mlp_results = bench_mlp(args.rounds, max(5, args.warmup // 2),
                                max(10, args.iters // 4))

    # ---- 汇总 ----
    metadata = capture_metadata(args)
    metadata.update({
        "M": args.M, "K": args.K, "N": args.N,
        "rounds": args.rounds,
        "warmup_iters": args.warmup,
        "measure_iters": args.iters,
        "gpu_warmup_secs": args.gpu_warmup_secs,
    })
    all_results = {
        "metadata": metadata,
        # legacy key 'config' kept so old readers still see it
        "config": {
            "M": args.M, "K": args.K, "N": args.N,
            "rounds": args.rounds,
            "warmup_iters": args.warmup,
            "measure_iters": args.iters,
            "gpu_warmup_secs": args.gpu_warmup_secs,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "ops": op_results,
        "mlp": mlp_results,
        "tile_sweep": tile_sweep_result,
    }

    # ---- 打印表格 ----
    print(f"\n{'='*70}")
    print(f"  cuTile benchmark — mean of last 3 of {args.rounds} rounds")
    print(f"{'='*70}")
    print(f"{'item':<28} {'discarded(r1)':>14} {'mean(r2-r4)':>14} {'std':>10}")
    print("-" * 70)
    for k, v in {**op_results, **{k: v for k, v in mlp_results.items() if isinstance(v, dict)}}.items():
        print(f"{k:<28} {v['discarded']:>14.4f} {v['mean_last_3_ms']:>14.4f} {v['std_last_3_ms']:>10.4f}")
    if "mlp_infer_throughput_samples_per_sec" in mlp_results:
        print(f"\n  mlp_infer_throughput: {mlp_results['mlp_infer_throughput_samples_per_sec']:.0f} samples/sec")

    # ---- 落盘 ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
