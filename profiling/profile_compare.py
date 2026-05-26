"""
PyTorch vs Triton MLP profiling 对比

使用 torch.profiler 记录两个模型的 kernel 级别耗时，
输出时间分布对比，支持导出 Chrome trace 格式。

用法:
    python profiling/profile_compare.py
    python profiling/profile_compare.py --epochs 2
    python profiling/profile_compare.py --export-trace
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python.mnist.model import MLP, MLPConfig
from python.mnist.triton_model import TritonMLP
from python.mnist.trainer import create_mnist_loaders


def profile_model(model: torch.nn.Module, name: str, train_loader, epochs: int = 1):
    """使用 torch.profiler profile 模型训练。"""
    device = next(model.parameters()).device
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    data_iter = iter(train_loader)

    # warmup
    model.train()
    for _ in range(3):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    # profiled steps
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(5):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()

    # 提取关键指标
    print(f"\n{'='*60}")
    print(f"  {name} - Top 15 CUDA kernels by CUDA time")
    print(f"{'='*60}")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15,
        max_name_width=60,
    ))

    return prof


def compare_profiles(pt_prof, tr_prof):
    """对比两个 profiler 的关键指标。"""
    print(f"\n{'='*72}")
    print("  Profile Comparison Summary")
    print(f"{'='*72}")

    def get_stats(prof, label):
        avg = prof.key_averages()
        cuda_total = sum(e.cuda_time_total for e in avg)
        cpu_total = sum(e.cpu_time_total for e in avg)
        n_kernels = len([e for e in avg if e.cuda_time_total > 0])
        return cuda_total, cpu_total, n_kernels

    pt_cuda, pt_cpu, pt_kernels = get_stats(pt_prof, "PyTorch")
    tr_cuda, tr_cpu, tr_kernels = get_stats(tr_prof, "Triton")

    print(f"  {'Metric':<30} {'PyTorch':>15} {'Triton':>15} {'Ratio':>10}")
    print(f"  {'-'*70}")
    print(f"  {'CUDA time total (us)':<30} {pt_cuda:>15,.0f} {tr_cuda:>15,.0f} {tr_cuda/pt_cuda:>10.2f}x")
    print(f"  {'CPU time total (us)':<30} {pt_cpu:>15,.0f} {tr_cpu:>15,.0f} {tr_cpu/pt_cpu:>10.2f}x")
    print(f"  {'CUDA kernel count':<30} {pt_kernels:>15} {tr_kernels:>15} {tr_kernels/pt_kernels:>10.2f}x")


def main():
    parser = argparse.ArgumentParser(description="PyTorch vs Triton MLP Profiling")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--export-trace", action="store_true", help="导出 Chrome trace 文件")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA required for profiling")
        sys.exit(1)

    config = MLPConfig(hidden_dims=[784, 256, 128, 10], activation="gelu")

    # 数据
    train_loader, _ = create_mnist_loaders(batch_size=args.batch_size)

    # PyTorch MLP
    print("Profiling PyTorch MLP...")
    pt_model = MLP(config).to(device)
    pt_prof = profile_model(pt_model, "PyTorch MLP", train_loader, args.epochs)

    # Triton MLP
    print("\nProfiling Triton MLP...")
    tr_model = TritonMLP(config).to(device)
    tr_prof = profile_model(tr_model, "Triton MLP", train_loader, args.epochs)

    # 对比
    compare_profiles(pt_prof, tr_prof)

    # 导出 trace
    if args.export_trace:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")

        pt_prof.export_chrome_trace(str(results_dir / f"trace_pytorch_{ts}.json.gz"))
        tr_prof.export_chrome_trace(str(results_dir / f"trace_triton_{ts}.json.gz"))
        print(f"\n  Chrome traces exported to results/")
        print(f"  View with: chrome://tracing")


if __name__ == "__main__":
    main()
