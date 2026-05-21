"""
结果输出：控制台格式化表格 + JSON 导出。
"""

import json
import time
from pathlib import Path
from typing import Optional


def print_summary(
    training_history: list[dict],
    benchmark: dict,
    config: dict,
    total_params: int,
    total_time_s: float,
):
    """打印控制台格式化的训练与基准测试结果。"""
    print()
    print("=" * 72)
    print("  Training Results")
    print("=" * 72)

    dims = config.get("hidden_dims", [])
    print(f"  Architecture: {' -> '.join(str(d) for d in dims)}")
    print(f"  Activation:   {config.get('activation', '-')}")
    print(f"  Dropout:      {config.get('dropout', 0.0)}")
    print(f"  Batch Size:   {config.get('batch_size', '-')}")
    print(f"  Parameters:   {total_params:,}")
    print()

    header = f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in training_history:
        print(
            f"  {m['epoch']:5d}  "
            f"{m['train_loss']:10.4f}  "
            f"{m['train_acc']:9.2%}  "
            f"{m['val_loss']:8.4f}  "
            f"{m['val_acc']:7.2%}"
        )

    best = max(training_history, key=lambda m: m["val_acc"])
    print()
    print(f"  Best val_acc: {best['val_acc']:.2%} (epoch {best['epoch']})")
    print(f"  Total training time: {total_time_s:.1f}s")

    # Benchmark
    if benchmark:
        print()
        print("=" * 72)
        print("  Benchmark Results")
        print("=" * 72)

        train_b = benchmark.get("training", {})
        inf_b = benchmark.get("inference", {})

        print(f"  [Training]")
        print(f"    Step time:  {train_b.get('step_time_ms_median', 0):.3f}ms (median) / {train_b.get('step_time_ms_p95', 0):.3f}ms (p95)")
        print(f"    Throughput: {train_b.get('samples_per_sec', 0):,.0f} samples/sec")
        print(f"    GPU memory: {train_b.get('peak_gpu_memory_mb', 0):.1f} MB")

        print(f"  [Inference]")
        print(f"    Latency:    {inf_b.get('latency_ms_per_batch_median', 0):.3f}ms/batch (median) / {inf_b.get('latency_ms_per_batch_p95', 0):.3f}ms/batch (p95)")
        print(f"    Throughput: {inf_b.get('samples_per_sec', 0):,.0f} samples/sec")
        print(f"    GPU memory: {inf_b.get('peak_gpu_memory_mb', 0):.1f} MB")


def export_json(
    training_history: list[dict],
    benchmark: Optional[dict],
    config: dict,
    backend: str,
    total_params: int,
    total_time_s: float,
    results_dir: str,
    filename: Optional[str] = None,
) -> str:
    """导出结果为 JSON 文件，返回文件路径。"""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"mnist_{backend}_{ts}.json"

    best = max(training_history, key=lambda m: m["val_acc"]) if training_history else {}

    output = {
        "config": config,
        "backend": backend,
        "training_history": training_history,
        "benchmark": benchmark,
        "final_metrics": {
            "test_accuracy": best.get("val_acc", 0.0),
            "test_loss": best.get("val_loss", 0.0),
            "total_params": total_params,
            "total_training_time_s": total_time_s,
        },
    }

    filepath = results_path / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {filepath}")
    return str(filepath)
