"""
统一 Benchmark 框架

输出指标: latency_ms, p50, p95, TFLOPS, speedup_vs_torch, speedup_vs_naive

用法:
    python bench/benchmark.py
    python bench/benchmark.py --impl cuda_tiled --M 512 --K 4096 --N 11008
    python bench/benchmark.py --config bench/benchmark_shapes.yaml --output results/bench.csv
"""

import argparse
import csv
import time
import yaml
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

from python.mlp_reference import (
    matmul_ref, mlp_first_layer_ref, mlp_two_layer_ref,
    swiglu_ref, correctness_check,
)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def cuda_timer(fn, warmup: int = 20, repeat: int = 100):
    """CUDA 同步计时器

    返回: (latency_ms_list, median_ms, p95_ms)
    """
    # warmup
    for _ in range(warmup):
        fn()

    latencies = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))

    latencies = sorted(latencies)
    median = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    return latencies, median, p95


def compute_tflops(matmul_flops: int, latency_ms: float) -> float:
    """计算 TFLOPS = FLOPs / (time_s * 1e12)"""
    return matmul_flops / (latency_ms * 1e-3) / 1e12


def matmul_flops(M: int, K: int, N: int) -> int:
    """单次 matmul 的浮点运算量: 2 * M * K * N"""
    return 2 * M * K * N


def generate_inputs(M, K, N, dtype, device='cuda'):
    """生成测试输入 tensor"""
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    return A, B


def run_torch_baseline(A, B):
    """PyTorch baseline: A @ B"""
    return A @ B


def run_cuda_naive(A, B):
    """TODO: 调用 CUDA naive matmul"""
    # from python.torch_extension import matmul_naive
    # return matmul_naive(A.float(), B.float()).to(A.dtype)
    raise NotImplementedError("Install CUDA extension first: python setup.py install")


def run_cuda_tiled(A, B):
    """TODO: 调用 CUDA tiled matmul"""
    raise NotImplementedError("Install CUDA extension first: python setup.py install")


def run_cuda_fused(X, W1, bias):
    """TODO: 调用 CUDA fused MLP first layer"""
    raise NotImplementedError("Install CUDA extension first: python setup.py install")


def run_triton_matmul(A, B):
    """TODO: 调用 Triton matmul"""
    # from triton_kernels.matmul_triton import matmul_triton
    # return matmul_triton(A, B)
    raise NotImplementedError("Triton kernel not implemented yet")


IMPLEMENTATIONS = {
    'torch': run_torch_baseline,
    'cuda_naive': run_cuda_naive,
    'cuda_tiled': run_cuda_tiled,
    'cuda_fused': run_cuda_fused,
    'triton_matmul': run_triton_matmul,
}


def benchmark_matmul(M, K, N, dtype_str, impl_name, fn, warmup, repeat):
    """对单次 matmul 做 benchmark 并返回结果"""
    dtype = torch.float32 if dtype_str == 'fp32' else torch.float16
    A, B = generate_inputs(M, K, N, dtype)

    flops = matmul_flops(M, K, N)

    def run():
        return fn(A, B)

    latencies, median, p95 = cuda_timer(run, warmup=warmup, repeat=repeat)
    tflops = compute_tflops(flops, median)

    # correctness check (与 torch baseline 比较)
    try:
        result = run()
        ref = torch_baseline_fn(A, B)  # need to define this
    except:
        result = None
        ref = None

    return {
        'impl': impl_name,
        'dtype': dtype_str,
        'M': M, 'K': K, 'N': N,
        'FLOPs': flops,
        'latency_median_ms': median,
        'latency_p95_ms': p95,
        'TFLOPS': tflops,
        'speedup_vs_torch': 0.0,  # filled by compare step
    }


def run_benchmarks(config_path=None, impl_filter=None, output_path=None):
    """主 benchmark 入口"""
    # 默认配置
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'quick_test': {'M': [128], 'K': [1024], 'N': [4096]},
            'dtypes': ['fp32'],
            'implementations': ['torch'],
            'benchmark_params': {'warmup': 20, 'repeat': 100, 'seed': 42},
        }

    params = config.get('benchmark_params', {})
    warmup = params.get('warmup', 20)
    repeat = params.get('repeat', 100)
    set_seed(params.get('seed', 42))

    dtypes = config.get('dtypes', ['fp32'])
    implementations = impl_filter or config.get('implementations', ['torch'])

    # 收集所有 shape 组合
    shapes = []
    for shape_group, shape_config in config.items():
        if shape_group in ('dtypes', 'implementations', 'benchmark_params'):
            continue
        for m in shape_config.get('M', [512]):
            for k in shape_config.get('K', [768]):
                for n in shape_config.get('N', [3072]):
                    shapes.append((m, k, n))

    results = []

    for M, K, N in shapes:
        dtype = torch.float32  # TODO: 遍历 dtypes

        # 先跑 torch baseline
        A, B = generate_inputs(M, K, N, dtype)
        ref_fn = run_torch_baseline
        _, ref_median, _ = cuda_timer(lambda: ref_fn(A, B), warmup=warmup, repeat=repeat)

        for impl_name in implementations:
            print(f"  Benchmarking: {impl_name}  M={M} K={K} N={N}")

            try:
                fn = IMPLEMENTATIONS[impl_name]
            except KeyError:
                print(f"    Unknown implementation: {impl_name}")
                continue

            try:
                if impl_name == 'torch':
                    A2, B2 = A.clone(), B.clone()
                    latencies, median, p95 = cuda_timer(
                        lambda: fn(A2, B2), warmup=warmup, repeat=repeat
                    )
                else:
                    latencies, median, p95 = cuda_timer(
                        lambda: fn(A, B), warmup=max(1, warmup // 5), repeat=min(50, repeat // 3)
                    )
            except NotImplementedError as e:
                print(f"    SKIP: {e}")
                continue
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

            flops = matmul_flops(M, K, N)
            tflops = compute_tflops(flops, median)
            speedup = ref_median / median if median > 0 else 0

            row = {
                'impl': impl_name,
                'dtype': 'fp32',
                'M': M, 'K': K, 'N': N,
                'FLOPs': flops,
                'latency_median_ms': round(median, 4),
                'latency_p95_ms': round(p95, 4),
                'TFLOPS': round(tflops, 4),
                'speedup_vs_torch': round(speedup, 3),
            }
            results.append(row)
            print(f"    median={median:.3f}ms  TFLOPS={tflops:.2f}  speedup={speedup:.2f}x")

    # 输出
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='MLP Kernel Benchmark')
    parser.add_argument('--config', type=str, default='bench/benchmark_shapes.yaml',
                        help='YAML config path')
    parser.add_argument('--impl', type=str, nargs='+',
                        help='Filter implementations (e.g. torch cuda_tiled)')
    parser.add_argument('--output', type=str, default='results/benchmark_results.csv',
                        help='Output CSV path')
    parser.add_argument('--M', type=int, default=0, help='Override M')
    parser.add_argument('--K', type=int, default=0, help='Override K')
    parser.add_argument('--N', type=int, default=0, help='Override N')
    parser.add_argument('--dtype', type=str, default='', help='Override dtype')
    args = parser.parse_args()

    print("=" * 60)
    print("MLP Kernel Benchmark")
    print("=" * 60)

    results = run_benchmarks(
        config_path=args.config,
        impl_filter=args.impl,
        output_path=args.output,
    )

    # 打印汇总表格
    if results:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"{'impl':<16} {'dtype':<6} {'M':<6} {'K':<6} {'N':<6} {'latency_ms':<12} {'TFLOPS':<10} {'speedup':<8}")
        print("-" * 64)
        for r in results:
            print(f"{r['impl']:<16} {r['dtype']:<6} {r['M']:<6} {r['K']:<6} {r['N']:<6} "
                  f"{r['latency_median_ms']:<12.4f} {r['TFLOPS']:<10.2f} {r['speedup']:<8.2f}x")


if __name__ == '__main__':
    main()
