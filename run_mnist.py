"""
MNIST MLP 训练 + 基准测试入口

用法:
    python run_mnist.py                          # 默认 20 epochs + benchmark
    python run_mnist.py --epochs 5 --no-bench    # 仅训练 5 epochs
    python run_mnist.py --hidden "784,512,10"    # 自定义架构
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from python.mnist.model import MLP, MLPConfig
from python.mnist.trainer import Trainer, create_mnist_loaders
from python.mnist.benchmark import benchmark_training, benchmark_inference
from python.mnist.dashboard import print_summary, export_json


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST MLP Training & Benchmark")
    parser.add_argument("--config", type=str, default="configs/mnist_mlp.yaml")
    # 以下参数可覆盖 YAML 配置
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden", type=str, default=None,
                        help="逗号分隔的 hidden dims, 如 '784,512,10'")
    parser.add_argument("--activation", type=str, default=None,
                        choices=["relu", "gelu", "silu"])
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-bench", action="store_true", help="跳过基准测试")
    parser.add_argument("--no-train", action="store_true", help="跳过训练（需已有模型）")
    return parser.parse_args()


def load_config(args) -> dict:
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # CLI 覆盖
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


def main():
    args = parse_args()
    config = load_config(args)

    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA 不可用，将使用 CPU 运行。")
    print(f"Device: {device}")

    # ---- 模型 ----
    mlp_config = MLPConfig.from_dict(config["model"])
    model = MLP(mlp_config)
    print(f"Model: {' -> '.join(str(d) for d in mlp_config.hidden_dims)}")
    print(f"Parameters: {model.get_num_parameters():,}")

    # ---- 数据 ----
    batch_size = config["training"]["batch_size"]
    train_loader, test_loader = create_mnist_loaders(
        batch_size=batch_size,
        data_dir=config.get("output", {}).get("data_dir", "./data"),
    )
    print(f"MNIST: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")

    t_start = time.perf_counter()

    # ---- 训练 ----
    metrics_history = []
    if not args.no_train:
        trainer = Trainer(
            model, lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        checkpoint_dir = None
        if config.get("output", {}).get("save_checkpoint", False):
            checkpoint_dir = "checkpoints"

        metrics_history = trainer.fit(
            train_loader, test_loader,
            epochs=config["training"]["epochs"],
            checkpoint_dir=checkpoint_dir,
        )

    t_total = time.perf_counter() - t_start

    # ---- 基准测试 ----
    bench_results = None
    if not args.no_bench:
        print()
        print(f"{'='*60}")
        print("  Running benchmarks...")
        print(f"{'='*60}")

        bench_train = benchmark_training(
            model, train_loader, trainer.criterion, trainer.optimizer,
            warmup_steps=config["benchmark"]["warmup_steps"],
            measure_steps=config["benchmark"]["measure_steps"],
        )
        bench_infer = benchmark_inference(
            model, test_loader,
            warmup_batches=config["benchmark"]["inference_warmup"],
            measure_batches=config["benchmark"]["inference_measure"],
        )
        bench_results = {"training": bench_train, "inference": bench_infer}

    # ---- 输出 ----
    print_summary(
        training_history=metrics_history,
        benchmark=bench_results,
        config={**config["model"], "batch_size": batch_size,
                "learning_rate": config["training"]["learning_rate"],
                "epochs": config["training"]["epochs"]},
        total_params=model.get_num_parameters(),
        total_time_s=t_total,
    )

    export_json(
        training_history=metrics_history,
        benchmark=bench_results,
        config={**config["model"], **config["training"], **config.get("benchmark", {})},
        backend="pytorch",
        total_params=model.get_num_parameters(),
        total_time_s=t_total,
        results_dir=config.get("output", {}).get("results_dir", "results"),
    )


if __name__ == "__main__":
    main()
