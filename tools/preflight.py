#!/usr/bin/env python3
"""多机器构建预检（v2 其它机器代码）。

在一台新 GPU 机器上执行一次，检查:
  - torch 版本与 CUDA runtime/nvcc 是否匹配（构建 CUDA extension 的前置）
  - 当前 GPU / compute capability / 显存
  - Ampere(8.6) vs Blackwell(12.0) lane 判定
  - 给出 TORCH_CUDA_ARCH_LIST 建议（与本次 v2 setup.py 动态探测一致）

用法:
    python tools/preflight.py                 # 人类可读
    python tools/preflight.py --json          # 机器可读

状态码: 0 = 可直接 make install; 1 = 警告(可构建但有 notes); 2 = 阻断(需修)。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys

import torch


def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _nvcc_release() -> str:
    out = _run(["nvcc", "--version"])
    for line in out.splitlines():
        if "release" in line.lower():
            return line.strip()
    return "n/a"


def _driver_version() -> str:
    return _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()[:1] or ["n/a"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    info = {}
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda or "cpu"

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = {"name": props.name, "cc": f"{props.major}.{props.minor}",
                       "vram_gb": round(props.total_memory / 1e9, 1),
                       "sms": props.multi_processor_count}
        cc = (props.major, props.minor)
        if cc >= (12, 0):
            lane = "blackwell"
            arch_list = "12.0"
        elif cc >= (8, 6) and cc < (9, 0):
            lane = "ampere"
            arch_list = "8.6"
        elif cc >= (8, 0):
            lane = "ampere(8.0)"
            arch_list = "8.0"
        else:
            lane = "unsupported"
            arch_list = "?"
        info["lane"] = lane
        info["recommended_arch_list"] = arch_list
    else:
        info["gpu"] = None
        info["lane"] = "cpu-only"
        info["recommended_arch_list"] = "8.6"

    info["nvcc"] = _nvcc_release()
    info["driver"] = _driver_version()

    # 工具链一致性: torch.cuda vs nvcc 主版本
    torch_cuda_major = (info["torch_cuda"] or "0").split(".")[0]
    nvcc_ver = _run(["nvcc", "--version"])
    nvcc_major = ""
    for line in nvcc_ver.splitlines():
        if "release" in line.lower():
            v = line.split("release")[-1].strip().split(",")[0].strip().split(".")[0]
            nvcc_major = v
            break

    status = 0
    notes = []
    if info["lane"] in ("unsupported", "cpu-only"):
        notes.append("GPU 不在 Ampere(8.6)/Blackwell(12.0) 支持集，CUDA extension 不保证构建。")
        status = max(status, 2)
    if torch_cuda_major and nvcc_major and torch_cuda_major != nvcc_major:
        notes.append(
            f"⚠️ torch 编译用 CUDA {torch_cuda_major}，但 PATH 中 nvcc 是 {nvcc_major} — "
            "构建将失败。请加 PATH=/usr/local/cuda-<匹配版本>/bin 或安装匹配工具链（reproduce.py 会自动探测 /usr/local/cuda*）。")
        status = max(status, 1)
    if not info["nvcc"] or info["nvcc"] == "n/a":
        notes.append("未找到 nvcc — CUDA extension 无法构建（GPU 推理/Triton 仍可用）。")
        status = max(status, 1)

    note = "; ".join(notes) if notes else "OK — 可直接 make install / make reproduce"

    if args.json:
        print(json.dumps({**info, "status": status, "notes": notes}, ensure_ascii=False, indent=2))
        return status

    print("=" * 60)
    print("  MLP-Kernel-Lab Preflight (v2 multi-machine)")
    print("=" * 60)
    print(f"  python  : {info['python']}")
    print(f"  torch   : {info['torch']} (cuda={info['torch_cuda']})")
    if info["gpu"]:
        print(f"  gpu     : {info['gpu']['name']} (cc {info['gpu']['cc']}, {info['gpu']['vram_gb']} GB, {info['gpu']['sms']} SM)")
        print(f"  lane    : {info['lane']} -> TORCH_CUDA_ARCH_LIST={info['recommended_arch_list']}")
    print(f"  nvcc    : {info['nvcc']}")
    print(f"  driver  : {info['driver']}")
    print(f"  status  : {status} — {note}")
    print("=" * 60)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
