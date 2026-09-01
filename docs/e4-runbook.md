# E4 Runbook — 跨设备复现实操手册（3080 Ti / 3090 Ti / 5070 Ti）

> 目标：在任何一台目标机器上，从 0 产出完整跨设备对比报告（含数字与证据），全程可追溯（manifest）。
> 3070 已验证的代码/脚本原样复用。

## 0. 阶段与定义

- **E4** = 在不同 GPU 上复现 E2/E3 结果，证明结论不依赖单设备
- 目标机器：3080 Ti (sm86) / 3090 Ti (sm86, Ampere) / **5070 Ti (sm120, Blackwell)**
- 3070 Laptop (sm86) 已 E2/E3 验证；E4 重点 = 同架构第二设备 + Blackwell 跨代
- **3080 Ti 预期差异（vs 3070 Laptop，同架构 sm86，数字以实机 preflight+reproduce 为准）**：

| 属性 | 3070 Laptop | 3080 Ti (桌面) | 对结果的影响 |
|---|---|---|---|
| SM 数 | 40 | **80（2x）** | fp16 TFLOPS 应 ~2x（Triton roofline 86% 上限可能到 ~50-55 TFLOPS） |
| 显存带宽 | 448 GB/s | **912 GB/s（2x）** | decode 权重读 42-91→~90-180 GB/s；带宽 bound 结论不变 |
| VRAM | 8 GB | **12 GB** | 可跑更大 shape（M≥4096 压力测试）|
| 功耗/TGP | ~139.8W 实测（节流） | ~350W 桌面（散热充裕） | 数字更稳定；节流不构成变量 |
| decode 摊销 | 82.9x（M1→M256 实测） | 预期更大（带宽 2x + 权重复用） | E4 重点复核指标 |

## 1. 事前准备

```bash
# 1) 代码（离线可靠）：git bundle（公网 TLS 常被 GFW 阻断）
#    开发机(3070)上：git bundle create v2.bundle v2-transformer-mlp
#    拷贝 v2.bundle 到目标机器（U盘/内网 scp）

# 2) 目标机器：git clone v2.bundle v2-transformer-mlp repodir && cd repodir && git checkout v2-transformer-mlp

# 3) Python 3.12 venv + 依赖
python3.12 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130  # 或按 GPU 选 cu<ver>
pip install pytest triton cuda-tile pyyaml

# 4) CUDA 工具链（reproduce.py 自动探测 /usr/local/cuda-*，验证于 13.2）
export CUDA_HOME=$(ls -d /usr/local/cuda-* | sort -V | tail -1)
export PATH=$CUDA_HOME/bin:$PATH
```

## 2. 到机快速验证（< 5 分钟）

> ✅ 已验证（2026-09-01, 3070 上完整演练 simulate offline onboarding）：
> bundle clone → checkout → preflight(lane=ampere,exit 0) → import smoke → doc-ref guard ALL OK。
> 真机唯一差异是 GPU 型号（preflight 会自动识别）。

```bash
# 架构 lane 检测 + 工具链一致性（应输出 lane=ampere 或 blackwell）
python tools/preflight.py --json

# 一键验证：preflight -> import smoke -> 快速 tests -> bench(含热采集) -> status
make verify PYTHON=$(which python)
```

## 3. 全量复现（E4 主证据）

```bash
make reproduce PYTHON=$(which python)   # build + 213 tests + bench smoke + manifest

python bench/run.py --suite all --dtypes fp32,fp16 --backends eager,concat,triton,triton_fused,cuda,cutile,compile --warmup 20 --iters 100

python scripts/gpu_telemetry.py --interval 2 --cmd "python bench/run.py --suite prefill --dtypes fp16"
```

## 4. 对比记录模板（每台机器填一份）

```
GPU           : <型号>
SM 数 / cc    : <sms> / <cc>   (preflight --json)
CUDA/torch    : <nvcc>/<torch>  (manifest)
---- make reproduce ----
tests         : 212 / <passed> / <skipped>    (应 ≥ 3070)
manifest      : artifacts/<run-id>/manifest.json
---- 性能 ----
fp16 Triton best vs eager-fp32 : <x>   (3070 = 3.52x all-suite)
fp16 六后端 corr               : <norm_l2 分布>
cutile / cuda fp16 lat         : <ms>  (3070 = 11.8 / 29-118)
---- 热 ----
max_temp / max_power / throttled : <...>
---- 结论 ----
<正确性一致？性能趋势？跨代差异(Blackwell vs Ampere)？>
```

## 5. Blackwell (5070 Ti) 特别注意

- lane=blackwell sm120；setup.py 动态探测（TORCH_CUDA_ARCH_LIST=12.0）
- cuda-tile ≥ 1.3.0（Blackwell）；cuTile 在 Blackwell 是主战场（3070 慢是 Ampere 限制）
- Triton fp16 应更进一步；decode 带宽物理上限仍适用
- 若 cutile tile (32,64,32) 在 Blackwell 非最优：用 tile sweep 实验重选（见 claim-matrix cutile 行）

## 6. 完成后回报

- 三产物：manifest 链路(artifacts)、E4 对比表(追加 compat-matrix)、实验报告(docs/experiments/e4-<gpu>-<date>.md)
- 更新 EVIDENCE.md E4 状态列
