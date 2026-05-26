"""
PyTorch MLP vs Triton MLP 对比训练 & 基准测试

同一配置下分别训练两个模型，输出并排对比：
- 每 epoch 准确率
- 训练/推理延迟
- GPU 显存占用

用法:
    python run_compare.py                        # 默认配置
    python run_compare.py --epochs 5             # 5 epochs 快速对比
    python run_compare.py --hidden "784,512,10"  # 自定义架构
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from python.mnist.model import MLP, MLPConfig
from python.mnist.triton_model import TritonMLP
from python.mnist.trainer import Trainer, create_mnist_loaders
from python.mnist.benchmark import benchmark_training, benchmark_inference
from triton_kernels.precision import precision

try:
    from python.mnist.cuda_model import CUDAMLP
    _HAS_CUDA = True
except ImportError:
    CUDAMLP = None
    _HAS_CUDA = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch vs Triton MLP Compare")
    parser.add_argument("--config", type=str, default="configs/mnist_mlp.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden", type=str, default=None)
    parser.add_argument("--activation", type=str, default=None, choices=["relu", "gelu", "silu"])
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--cuda", action="store_true", help="Also compare CUDA MLP (3-way)")
    parser.add_argument("--precision", type=str, default="tf32", choices=["tf32", "fp32"],
                        help="Precision mode: tf32 (tensor cores) or fp32 (strict)")
    return parser.parse_args()


def load_config(args) -> dict:
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.hidden is not None:
        config["model"]["hidden_dims"] = [int(x) for x in args.hidden.split(",")]
    if args.activation is not None:
        config["model"]["activation"] = args.activation
    if args.dropout is not None:
        config["model"]["dropout"] = args.dropout
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.seed is not None:
        config["seed"] = args.seed
    return config


def train_model(model_cls, name: str, config: dict, train_loader, test_loader, device: torch.device):
    """训练单个模型，返回 (metrics_history, training_time_s, model)。"""
    mlp_config = MLPConfig.from_dict(config["model"])
    model = model_cls(mlp_config).to(device)

    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Architecture: {' -> '.join(str(d) for d in mlp_config.hidden_dims)}")
    print(f"  Activation:   {mlp_config.activation}")
    print(f"  Parameters:   {model.get_num_parameters():,}")
    print(f"  Device:       {device}")
    print(f"{'='*60}")

    trainer = Trainer(
        model,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        device=device,
    )

    t_start = time.perf_counter()
    history = trainer.fit(
        train_loader, test_loader,
        epochs=config["training"]["epochs"],
    )
    t_total = time.perf_counter() - t_start

    return history, t_total, model, trainer


def print_comparison(
    pytorch_history: list, pytorch_time: float, pytorch_params: int,
    triton_history: list, triton_time: float, triton_params: int,
    pytorch_bench: dict | None, triton_bench: dict | None,
    epochs: int,
    cu_history: list | None = None, cu_time: float | None = None,
    cu_params: int = 0, cu_bench: dict | None = None,
):
    has_cuda = cu_history is not None
    label = "PyTorch vs Triton" + (" vs CUDA" if has_cuda else "")
    print(f"\n{'='*90}")
    print(f"  Comparison: {label}")
    print(f"{'='*90}")

    # 每 epoch 对比
    if has_cuda:
        header = (
            f"  {'Epoch':>5}  | "
            f"{'PyTorch Acc':>11} {'PyTorch Loss':>12}  | "
            f"{'Triton Acc':>10} {'Triton Loss':>11}  | "
            f"{'CUDA Acc':>8} {'CUDA Loss':>10}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        for p, t, c in zip(pytorch_history, triton_history, cu_history):
            print(
                f"  {p['epoch']:5d}  | "
                f"{p['val_acc']:11.2%} {p['val_loss']:12.6f}  | "
                f"{t['val_acc']:10.2%} {t['val_loss']:11.6f}  | "
                f"{c['val_acc']:8.2%} {c['val_loss']:10.6f}"
            )
    else:
        header = (
            f"  {'Epoch':>5}  | "
            f"{'PyTorch Acc':>11} {'PyTorch Loss':>12}  | "
            f"{'Triton Acc':>10} {'Triton Loss':>11}  | "
            f"{'Acc Diff':>8}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        for p, t in zip(pytorch_history, triton_history):
            acc_diff = t["val_acc"] - p["val_acc"]
            print(
                f"  {p['epoch']:5d}  | "
                f"{p['val_acc']:11.2%} {p['val_loss']:12.6f}  | "
                f"{t['val_acc']:10.2%} {t['val_loss']:11.6f}  | "
                f"{acc_diff:+8.2%}"
            )

    # 最终准确率
    p_best = max(pytorch_history, key=lambda m: m["val_acc"])
    t_best = max(triton_history, key=lambda m: m["val_acc"])
    print()
    print(f"  PyTorch best val_acc: {p_best['val_acc']:.2%} (epoch {p_best['epoch']})")
    print(f"  Triton  best val_acc: {t_best['val_acc']:.2%} (epoch {t_best['epoch']})")
    if has_cuda:
        c_best = max(cu_history, key=lambda m: m["val_acc"])
        print(f"  CUDA    best val_acc: {c_best['val_acc']:.2%} (epoch {c_best['epoch']})")

    # 训练时间
    print(f"\n  Training time:")
    print(f"    PyTorch: {pytorch_time:.1f}s")
    print(f"    Triton:  {triton_time:.1f}s")
    if has_cuda:
        print(f"    CUDA:    {cu_time:.1f}s")

    # 基准测试对比
    if pytorch_bench and triton_bench:
        print(f"\n{'='*90}")
        print("  Benchmark Comparison")
        print(f"{'='*90}")

        pt = pytorch_bench["training"]
        tt = triton_bench["training"]
        print(f"  [Training Step]")
        print(f"    PyTorch: {pt['step_time_ms_median']:.3f}ms (median) / {pt['step_time_ms_p95']:.3f}ms (p95)")
        print(f"    Triton:  {tt['step_time_ms_median']:.3f}ms (median) / {tt['step_time_ms_p95']:.3f}ms (p95)")
        if cu_bench:
            ct = cu_bench["training"]
            print(f"    CUDA:    {ct['step_time_ms_median']:.3f}ms (median) / {ct['step_time_ms_p95']:.3f}ms (p95)")
        print(f"    PyTorch throughput: {pt['samples_per_sec']:,.0f} samples/sec")
        print(f"    Triton  throughput: {tt['samples_per_sec']:,.0f} samples/sec")
        if cu_bench:
            print(f"    CUDA    throughput: {ct['samples_per_sec']:,.0f} samples/sec")

        pi = pytorch_bench["inference"]
        ti = triton_bench["inference"]
        print(f"\n  [Inference]")
        print(f"    PyTorch: {pi['latency_ms_per_batch_median']:.3f}ms/batch (median)")
        print(f"    Triton:  {ti['latency_ms_per_batch_median']:.3f}ms/batch (median)")
        if cu_bench:
            ci = cu_bench["inference"]
            print(f"    CUDA:    {ci['latency_ms_per_batch_median']:.3f}ms/batch (median)")
        print(f"    PyTorch throughput: {pi['samples_per_sec']:,.0f} samples/sec")
        print(f"    Triton  throughput: {ti['samples_per_sec']:,.0f} samples/sec")
        if cu_bench:
            print(f"    CUDA    throughput: {ci['samples_per_sec']:,.0f} samples/sec")


def main():
    args = parse_args()
    config = load_config(args)
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA 不可用，Triton kernel 需要 CUDA。")
        sys.exit(1)
    print(f"Device: {device}")

    # 精度配置
    if args.precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        precision.allow_tf32 = False
        if _HAS_CUDA:
            from python.mnist.cuda_layers import CUDALinearFunction
            CUDALinearFunction.use_cublas = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        precision.allow_tf32 = True
        if _HAS_CUDA:
            from python.mnist.cuda_layers import CUDALinearFunction
            CUDALinearFunction.use_cublas = True
    print(f"Precision: {args.precision}")

    # 数据加载（两个模型共享数据）
    batch_size = config["training"]["batch_size"]
    train_loader, test_loader = create_mnist_loaders(
        batch_size=batch_size,
        data_dir=config.get("output", {}).get("data_dir", "./data"),
    )
    print(f"MNIST: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")

    # 训练 PyTorch MLP
    set_seed(config.get("seed", 42))
    pt_history, pt_time, pt_model, pt_trainer = train_model(
        MLP, "PyTorch MLP", config, train_loader, test_loader, device
    )

    # 训练 Triton MLP（重置 seed 保证数据顺序一致）
    set_seed(config.get("seed", 42))
    tr_history, tr_time, tr_model, tr_trainer = train_model(
        TritonMLP, "Triton MLP", config, train_loader, test_loader, device
    )

    # 训练 CUDA MLP (可选)
    cu_history, cu_time, cu_model, cu_trainer = None, None, None, None
    if args.cuda:
        if not _HAS_CUDA:
            print("WARNING: CUDA MLP 不可用（mlp_cuda 模块未安装），跳过 CUDA 对比。")
        else:
            set_seed(config.get("seed", 42))
            cu_history, cu_time, cu_model, cu_trainer = train_model(
                CUDAMLP, "CUDA MLP", config, train_loader, test_loader, device
            )

    # 基准测试
    pt_bench = None
    tr_bench = None
    cu_bench = None
    if not args.no_bench:
        print(f"\n{'='*60}")
        print("  Running benchmarks...")
        print(f"{'='*60}")

        pt_bench_tr = benchmark_training(
            pt_model, train_loader, pt_trainer.criterion, pt_trainer.optimizer,
            warmup_steps=config["benchmark"]["warmup_steps"],
            measure_steps=config["benchmark"]["measure_steps"],
        )
        pt_bench_inf = benchmark_inference(
            pt_model, test_loader,
            warmup_batches=config["benchmark"]["inference_warmup"],
            measure_batches=config["benchmark"]["inference_measure"],
        )
        pt_bench = {"training": pt_bench_tr, "inference": pt_bench_inf}

        tr_bench_tr = benchmark_training(
            tr_model, train_loader, tr_trainer.criterion, tr_trainer.optimizer,
            warmup_steps=config["benchmark"]["warmup_steps"],
            measure_steps=config["benchmark"]["measure_steps"],
        )
        tr_bench_inf = benchmark_inference(
            tr_model, test_loader,
            warmup_batches=config["benchmark"]["inference_warmup"],
            measure_batches=config["benchmark"]["inference_measure"],
        )
        tr_bench = {"training": tr_bench_tr, "inference": tr_bench_inf}

        if cu_model is not None:
            cu_bench_tr = benchmark_training(
                cu_model, train_loader, cu_trainer.criterion, cu_trainer.optimizer,
                warmup_steps=config["benchmark"]["warmup_steps"],
                measure_steps=config["benchmark"]["measure_steps"],
            )
            cu_bench_inf = benchmark_inference(
                cu_model, test_loader,
                warmup_batches=config["benchmark"]["inference_warmup"],
                measure_batches=config["benchmark"]["inference_measure"],
            )
            cu_bench = {"training": cu_bench_tr, "inference": cu_bench_inf}

    # 输出对比
    print_comparison(
        pt_history, pt_time, pt_model.get_num_parameters(),
        tr_history, tr_time, tr_model.get_num_parameters(),
        pt_bench, tr_bench,
        epochs=config["training"]["epochs"],
        cu_history=cu_history, cu_time=cu_time,
        cu_params=cu_model.get_num_parameters() if cu_model else 0,
        cu_bench=cu_bench,
    )

    # 导出 JSON
    results_dir = config.get("output", {}).get("results_dir", "results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    p_best = max(pt_history, key=lambda m: m["val_acc"]) if pt_history else {}
    t_best = max(tr_history, key=lambda m: m["val_acc"]) if tr_history else {}

    result = {
        "config": config,
        "pytorch": {
            "training_history": pt_history,
            "benchmark": pt_bench,
            "best_val_acc": p_best.get("val_acc", 0),
            "training_time_s": pt_time,
            "total_params": pt_model.get_num_parameters(),
        },
        "triton": {
            "training_history": tr_history,
            "benchmark": tr_bench,
            "best_val_acc": t_best.get("val_acc", 0),
            "training_time_s": tr_time,
            "total_params": tr_model.get_num_parameters(),
        },
    }

    if cu_history is not None:
        cu_best = max(cu_history, key=lambda m: m["val_acc"]) if cu_history else {}
        result["cuda"] = {
            "training_history": cu_history,
            "benchmark": cu_bench,
            "best_val_acc": cu_best.get("val_acc", 0),
            "training_time_s": cu_time,
            "total_params": cu_model.get_num_parameters() if cu_model else 0,
        }

    out_path = Path(results_dir) / f"compare_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
