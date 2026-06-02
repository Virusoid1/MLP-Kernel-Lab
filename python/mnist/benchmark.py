"""CUDA Event 基准测试 + 复现元数据 + 公开统计函数。

本文件原本只提供 4 个 benchmark_* 函数,现在加 3 个公开 API:
    capture_metadata(args=None)  -> dict  (落入 JSON results 顶部)
    p95(values)                  -> float (替代 int(len*0.95) off-by-one)
    median(values)               -> float (转发 statistics.median)
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import time
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from python.mnist.stats import p95 as _p95, percentile, stable_median  # noqa: F401


# --- 公开统计 API (供 benchmark_ops.py / bench_cutile.py / run_compare.py import) ---

def p95(values: Iterable[float]) -> float:
    """True 95th percentile, linear interpolation. Replaces ``int(len*0.95)`` floor."""
    return _p95(values)


def median(values: Iterable[float]) -> float:
    arr = sorted(float(v) for v in values)
    if not arr:
        raise ValueError("median requires non-empty list")
    return statistics.median(arr)


# --- 复现元数据 ----------------------------------------------------------------

def _git_sha(short: bool = True) -> str:
    try:
        cmd = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=2.0,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=2.0,
        )
        if out.returncode == 0:
            return out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else "unknown"
    except Exception:
        pass
    return "unknown"


def _gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"name": "cpu", "cc": "n/a", "vram_gb": 0}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "cc": f"{props.major}.{props.minor}",
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
    }


def _cudnn_version() -> str:
    try:
        v = torch.backends.cudnn.version()
        return str(v) if v is not None else "n/a"
    except Exception:
        return "n/a"


def capture_metadata(args=None) -> dict:
    """收集 GPU / 驱动 / 框架 / 配置 / git / 时间戳, 嵌入到 JSON 结果顶部。

    ``args`` 是 argparse Namespace (有就好), 若为 None 则跳过 CLI 字段。
    """
    md = {
        "gpu": _gpu_info(),
        "driver": _driver_version(),
        "torch": torch.__version__,
        "cudnn": _cudnn_version(),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "git_sha": _git_sha(short=True),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if args is not None:
        try:
            args_dict = vars(args).copy()
            md["args"] = {k: (v if isinstance(v, (str, int, float, bool, list, type(None))) else str(v))
                          for k, v in args_dict.items()}
        except TypeError:
            pass
    return md


# --- benchmark 工具 (沿用原 API, 内部统一用 p95) -------------------------------

def _gpu_memory_mb() -> float:
    """当前 peak GPU 分配 (MiB)。"""
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def benchmark_training_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict:
    """单步训练 GPU 耗时 (forward + loss + backward + optimizer.step)。"""
    model.train()

    for _ in range(3):  # warmup
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    end.record()

    torch.cuda.synchronize()

    return {
        "step_time_ms": start.elapsed_time(end),
        "peak_gpu_memory_mb": _gpu_memory_mb(),
    }


def benchmark_inference_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> dict:
    """单 batch 推理 GPU 耗时。"""
    model.eval()

    with torch.inference_mode():
        for _ in range(5):  # warmup
            _ = model(x)

    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.inference_mode():
        _ = model(x)
    end.record()

    torch.cuda.synchronize()

    return {
        "latency_ms": start.elapsed_time(end),
        "peak_gpu_memory_mb": _gpu_memory_mb(),
    }


def benchmark_training(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 5,
    measure_steps: int = 20,
) -> dict:
    """训练性能基准:多次采样取 median/mean/真 p95。"""
    model.train()
    data_iter = iter(loader)

    for _ in range(warmup_steps):
        x, y = next(data_iter)
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    step_times: list[float] = []
    torch.cuda.reset_peak_memory_stats()

    for _ in range(measure_steps):
        x, y = next(data_iter)
        x = x.cuda()
        y = y.cuda()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        end.record()

        torch.cuda.synchronize()
        step_times.append(start.elapsed_time(end))

    batch_size = loader.batch_size
    med = statistics.median(step_times)

    return {
        "step_time_ms_median": med,
        "step_time_ms_mean": statistics.mean(step_times),
        "step_time_ms_p95": p95(step_times),
        "samples_per_sec": batch_size / (med / 1000),
        "peak_gpu_memory_mb": _gpu_memory_mb(),
    }


def benchmark_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    warmup_batches: int = 10,
    measure_batches: int = 50,
) -> dict:
    """推理性能基准。"""
    model.eval()
    data_iter = iter(loader)

    with torch.inference_mode():
        for _ in range(warmup_batches):
            try:
                x, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, _ = next(data_iter)
            x = x.cuda()
            _ = model(x)

    latencies: list[float] = []
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for _ in range(measure_batches):
            try:
                x, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, _ = next(data_iter)
            x = x.cuda()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = model(x)
            end.record()

            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

    batch_size = loader.batch_size
    med = statistics.median(latencies)

    return {
        "latency_ms_per_batch_median": med,
        "latency_ms_per_batch_mean": statistics.mean(latencies),
        "latency_ms_per_batch_p95": p95(latencies),
        "samples_per_sec": batch_size / (med / 1000),
        "peak_gpu_memory_mb": _gpu_memory_mb(),
    }
