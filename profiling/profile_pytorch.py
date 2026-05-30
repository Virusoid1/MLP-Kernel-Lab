"""
PyTorch kernel 的 Nsight profiling 辅助脚本。

用于对 PyTorch 自定义 CUDA kernel 或 Triton kernel 进行性能分析。
支持 nsys 时间线分析和 ncu kernel 级分析。

用法:
    # 普通 benchmark（无 profiling）
    python profile_pytorch.py

    # nsys 时间线分析
    nsys profile --trace=cuda,nvtx -o reports/nsys_pytorch python profile_pytorch.py --nsys

    # ncu kernel 分析（分析单个 kernel）
    ncu --set full --launch-skip 10 --launch-count 1 -o reports/ncu_pytorch python profile_pytorch.py --ncu

    # ncu 只分析特定 kernel 名称
    ncu --set full --kernel-name "regex:.*triton.*" -o reports/ncu_triton python profile_pytorch.py --ncu
"""

import argparse
import time
from contextlib import contextmanager

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False
    # 回退方案：用字符串标记
    class _NVTXStub:
        @staticmethod
        def push_range(name, color=0):
            pass
        @staticmethod
        def pop_range():
            pass
        @staticmethod
        def mark(name):
            pass
    nvtx = _NVTXStub()


# ---------- NVTX 辅助工具 ----------
@contextmanager
def nvtx_range(name: str):
    """用 with 语句自动标记 NVTX 范围"""
    nvtx.push_range(name)
    try:
        yield
    finally:
        nvtx.pop_range()


# ---------- Benchmark 工具 ----------
def benchmark_fn(fn, warmup=10, repeat=100, label=""):
    """测量函数执行时间（CUDA events）"""
    if not HAS_TORCH or not torch.cuda.is_available():
        print(f"[SKIP] {label}: PyTorch CUDA 不可用")
        return None

    # warmup
    with nvtx_range(f"{label} warmup"):
        for _ in range(warmup):
            fn()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with nvtx_range(f"{label} benchmark"):
        for _ in range(repeat):
            fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    avg_ms = ms / repeat
    print(f"[{label}] avg: {avg_ms:.4f} ms ({repeat} iters, total {ms:.1f} ms)")
    return avg_ms


# ---------- 测试用例 ----------
def bench_linear(num_features=1024, hidden_dim=4096, batch_size=512):
    """Linear 层 benchmark"""
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float32)
    w = torch.randn(hidden_dim, num_features, device="cuda", dtype=torch.float32)
    b = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)

    with nvtx_range("Linear alloc"):
        pass  # 数据已就绪

    def forward():
        return F.linear(x, w, b)

    benchmark_fn(forward, label="Linear")


def bench_matmul(M=2048, N=2048, K=2048):
    """矩阵乘法 benchmark"""
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    def forward():
        return A @ B

    benchmark_fn(forward, label=f"MatMul {M}x{K} * {K}x{N}")


def bench_elementwise(n=16 * 1024 * 1024):
    """逐元素操作 benchmark"""
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    def add():
        return a + b

    def mul():
        return a * b

    def fused():
        return a * 0.5 + b * 0.5

    benchmark_fn(add, label="Elementwise Add")
    benchmark_fn(mul, label="Elementwise Mul")
    benchmark_fn(fused, label="Elementwise Fused")


def bench_softmax(batch=4096, features=1024):
    """Softmax benchmark"""
    x = torch.randn(batch, features, device="cuda", dtype=torch.float32)

    def forward():
        return F.softmax(x, dim=-1)

    benchmark_fn(forward, label=f"Softmax {batch}x{features}")


def bench_reduction(n=16 * 1024 * 1024):
    """规约操作 benchmark"""
    x = torch.randn(n, device="cuda", dtype=torch.float32)

    def sum_all():
        return x.sum()

    def mean_all():
        return x.mean()

    def max_all():
        return x.max()

    benchmark_fn(sum_all, label="Reduce Sum")
    benchmark_fn(mean_all, label="Reduce Mean")
    benchmark_fn(max_all, label="Reduce Max")


def bench_mlp(layers=None, batch_size=512, input_dim=784, num_classes=10):
    """完整 MLP forward + backward benchmark"""
    if layers is None:
        layers = [512, 256, 128]

    with nvtx_range("MLP setup"):
        # 构建权重
        dims = [input_dim] + layers + [num_classes]
        weights = []
        biases = []
        for i in range(len(dims) - 1):
            w = torch.randn(dims[i + 1], dims[i], device="cuda", dtype=torch.float32)
            b = torch.randn(dims[i + 1], device="cuda", dtype=torch.float32)
            w.requires_grad_(True)
            b.requires_grad_(True)
            weights.append(w)
            biases.append(b)

        x = torch.randn(batch_size, input_dim, device="cuda", dtype=torch.float32)
        target = torch.randint(0, num_classes, (batch_size,), device="cuda")

    def forward_backward():
        with nvtx_range("MLP forward"):
            h = x
            for i in range(len(weights) - 1):
                h = F.linear(h, weights[i], biases[i])
                h = F.relu(h)
            logits = F.linear(h, weights[-1], biases[-1])
            loss = F.cross_entropy(logits, target)

        with nvtx_range("MLP backward"):
            loss.backward()

        # 清零梯度
        for w in weights:
            w.grad = None

    benchmark_fn(forward_backward, label=f"MLP {dims}")


# ---------- 主入口 ----------
def main():
    parser = argparse.ArgumentParser(description="PyTorch Nsight Profiling")
    parser.add_argument("--nsys", action="store_true", help="nsys 模式（添加额外标记）")
    parser.add_argument("--ncu", action="store_true", help="ncu 模式（减少迭代次数）")
    parser.add_argument("--repeat", type=int, default=None, help="覆盖迭代次数")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("需要安装 PyTorch: pip install torch")
        return

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    # ncu 模式下减少迭代，否则 profile 时间过长
    if args.ncu and args.repeat is None:
        print("[NCU 模式] 减少迭代次数以加速分析")

    with nvtx_range("PyTorch Benchmark Suite"):
        bench_elementwise()
        bench_matmul()
        bench_softmax()
        bench_reduction()
        bench_linear()
        bench_mlp()

    print("\nDone.")


if __name__ == "__main__":
    main()
