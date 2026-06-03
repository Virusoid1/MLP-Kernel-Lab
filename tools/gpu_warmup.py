"""GPU 预热脚本:跑 1 分钟计算密集 matmul 把 GPU 拉稳到 P0 + 暖 cache + 触发 boost clock.

为什么需要:
- benchmark / profiling 第一轮 GPU 时钟未升,P0 没就位,L2 偏差 5-10%;
- cold cache 让首轮 bandwidth 偏低;
- nvrtc / Triton autotune 第一次编译需要 ~5-30s,不算"稳态";
- unified memory prefetch / DMA engine 暖机。

用法:
    python tools/gpu_warmup.py                    # 默认 60s
    python tools/gpu_warmup.py --seconds 120     # 长 bench 前
    python tools/gpu_warmup.py --seconds 60 --matmul 4096  # 更大压力

输出: 控制台 + stderr 一行 warmup summary (start/end 时间 + 实际跑了几轮 matmul)。
"""
from __future__ import annotations

import argparse
import time

import torch


def warmup(seconds: float = 60.0, matmul_dim: int = 2048) -> int:
    """跑 matmul × seconds 秒 + activation + bias 混合, 返回实际完成的 iters。"""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available"); return 0

    print(f"[warmup] device={torch.cuda.get_device_name(0)}  target={seconds}s  matmul_dim={matmul_dim}")
    sys_start = time.perf_counter()
    torch.cuda.synchronize()

    a = torch.randn(matmul_dim, matmul_dim, device="cuda", dtype=torch.float32)
    b = torch.randn(matmul_dim, matmul_dim, device="cuda", dtype=torch.float32)
    bias = torch.randn(matmul_dim, device="cuda", dtype=torch.float32)
    a = a.contiguous(); b = b.contiguous()

    iters = 0
    while time.perf_counter() - sys_start < seconds:
        # matmul 主体
        c = torch.matmul(a, b)
        # 加点 mixed op 让 cache / DMA / reduction 单元都暖
        d = torch.nn.functional.gelu(c)
        e = d + bias
        f = torch.nn.functional.layer_norm(e, e.shape[-1:])
        iters += 1
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - sys_start
    print(f"[warmup] done in {elapsed:.1f}s ({iters} iterations, ~{elapsed/max(iters,1)*1000:.1f} ms/iter)")
    return iters


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="Total warmup duration in seconds (default 60)")
    ap.add_argument("--matmul", type=int, default=2048,
                    help="Square matmul dimension (default 2048; larger = more pressure)")
    args = ap.parse_args()
    warmup(seconds=args.seconds, matmul_dim=args.matmul)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
