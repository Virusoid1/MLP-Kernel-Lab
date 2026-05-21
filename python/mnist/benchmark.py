"""
CUDA Event 基准测试：训练吞吐/推理延迟/GPU 内存。
"""

import statistics
import torch
from torch.utils.data import DataLoader


def _gpu_memory_mb() -> float:
    """当前已分配的 GPU 内存 (MB)。"""
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def benchmark_training_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict:
    """测量单步训练的 GPU 耗时（forward + loss + backward + optimizer.step）。"""
    model.train()

    # warmup
    for _ in range(3):
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
    """测量单 batch 推理的 GPU 耗时。"""
    model.eval()

    # warmup
    with torch.inference_mode():
        for _ in range(5):
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
    """训练性能基准：多次采样取中位/均值/p95。"""
    model.train()

    data_iter = iter(loader)

    # warmup
    for _ in range(warmup_steps):
        x, y = next(data_iter)
        x, y = x.to(model.device if hasattr(model, "device") else "cuda"), y.to(model.device if hasattr(model, "device") else "cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    step_times = []
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
    step_times.sort()

    return {
        "step_time_ms_median": statistics.median(step_times),
        "step_time_ms_mean": statistics.mean(step_times),
        "step_time_ms_p95": step_times[int(len(step_times) * 0.95)],
        "samples_per_sec": batch_size / (statistics.median(step_times) / 1000),
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

    # warmup
    with torch.inference_mode():
        for _ in range(warmup_batches):
            x, _ = next(data_iter)
            x = x.cuda()
            _ = model(x)

    latencies = []
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for _ in range(measure_batches):
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
    latencies.sort()

    return {
        "latency_ms_per_batch_median": statistics.median(latencies),
        "latency_ms_per_batch_mean": statistics.mean(latencies),
        "latency_ms_per_batch_p95": latencies[int(len(latencies) * 0.95)],
        "samples_per_sec": batch_size / (statistics.median(latencies) / 1000),
        "peak_gpu_memory_mb": _gpu_memory_mb(),
    }
