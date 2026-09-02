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
make reproduce PYTHON=$(which python)   # build + 235 tests + bench smoke + manifest

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
cutile / cuda fp16 lat         : <ms>  (3070 终冻结 = 6.5 / 13.4；调优前 11.8 / 29-118)
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


## 5.5 3080 Ti 实测记录（2026-09-01，host c3d5ff0d7eeb，双卡）

**环境**: driver 535.104.05 / torch 2.9.1+cu126 / triton 3.5.1 / Python 3.12.3 / 无 sudo / 无 git / 无 nvcc
**部署**: scp 拷贝仓库 + 用户 site-packages（cuda-tile tar + pytest + Python.h via CPATH）

| 项 | 3070 Laptop | **3080 Ti（实测）** |
|---|---|---|
| fp16 Triton 加速 vs eager-fp32 | 2.4-3.5x | **2.16x-3.69x**（prefill 6 shapes） |
| 512×4096×11008 fp16 triton | 4.60ms | **2.76ms（1.67x）** |
| fp16 正确性 | norm_l2 2e-4~6e-4 | **同档（transformer_mlp 42p + P1 8/8 + dtype 16p）** |
| P1 opcheck/gradcheck | 8/8 | **8/8** |
| decode M=1（fp16 triton） | 0.78ms | **0.361ms（带宽翻倍效应）** |
| decode 摊销 M1→M256 | 82.9x | **76.6x**（带宽 bound 跨设备一致） |

**环境边界发现**（E4 价值：跨设备差异）:
1. Triton 需 CPATH 指向既有 Python.h（远程 py312-headers）
2. cuda-tile 1.x 要求 driver ≥r580；**3080 Ti driver 535 不兼容 → cutile 不可用**（3070 driver 610 OK）
3. mlp_cuda 未编译（无 nvcc）→ cuda 后端不可用；相关测试自动 skip
4. available_backends() 只查 import 不查 driver 兼容 → 3080 Ti 上 cutile 报可用但运行失败（建议后续加固）

## 5.6 跨设备对比汇总（3070 / 3080 Ti / 3090 Ti，均实测）

| 指标（fp16 triton） | 3070 Laptop | 3080 Ti | 3090 Ti |
|---|---|---|---|
| SM 数（preflight） | 40 | 80 | **84** |
| fp16 加速 vs eager-fp32 | 2.4-3.5x | 2.16-3.69x | **2.46-3.40x** |
| 2048×4096×11008 | 19.2ms(fp32 57.6) | 11.3ms(29.8) | **7.4ms(22.0)** |
| decode M=1 | 0.78ms | 0.361ms | **0.359ms** |
| decode 摊销 | 82.9x | 76.6x | **86.4x** |
| P1 opcheck/gradcheck | 8/8 | 8/8 | **8/8** |

- **结论**: 同架构（sm86）三设备 fp16 正确性一致（norm_l2 2e-4~6e-4），速度随 SM/带宽提升；
  decode 摊销跨设备均 ~77-86x——证明"weight-read 带宽 bound"与 batch 摊销结论与设备无关
- 3070 上的 2.4-3.5x 加速范围在三设备上一致成立（E4 复现成功）

## 5.7 大 shape 压力 + 双卡并行（2026-09-01 实测）

**大 shape（3070 8GB 无法运行；大显存优势）**:

| shape (fp16 triton) | 3080 Ti (12GB) | 3090 Ti (24GB) |
|---|---|---|
| 4096×768×3072 | 0.966ms | 0.868ms |
| 8192×768×3072 | — | 1.691ms |
| 4096×4096×11008 | 21.766ms | **14.386ms（1.51x）** |
| 8192×4096×11008 | — | 28.202ms |

**双卡并行**（CUDA_VISIBLE_DEVICES=0/1 并发，M=512×4096×11008 fp16 triton）:

| 机 | GPU0 | GPU1 | 单卡参照 |
|---|---|---|---|
| 3080 Ti | 2.570ms | 2.130ms | 2.76ms |
| 3090 Ti | 1.918ms | 1.934ms | 1.98ms |

- **结论**: 双卡并行无性能衰减（每卡独立跑满）；3090 Ti 24GB 可支撑 M=8192 压力 shape（28.2ms）
- 大 shape 下 3090 Ti vs 3080 Ti = 1.51x（SM 84 vs 80 + 更大带宽）

## 5.8 5070 Ti（Blackwell sm120）实测记录（2026-09-02）

**跨代验证**：preflight 首次识别 lane=blackwell（cc 12.0 / 70 SM / nvcc 13.2）。

| 指标（fp16 triton） | 3070 | 3080 Ti | 3090 Ti | **5070 Ti** |
|---|---|---|---|---|
| 跨精度 fp16/fp32 | 2.4-3.5x | 2.16-3.69x | 2.46-3.40x | **2.62-3.00x** |
| 同精度 fp16/fp16（大 shape） | 1.07x | ~1.09x | ~1.03x | **0.97-1.07x** |
| 512×4096×11008 triton | 4.60ms | 2.76ms | 1.98ms | **1.52ms** |
| decode M=1 | 0.78ms | 0.361ms | 0.359ms | **0.384ms** |
| decode 摊销 | 82.9x | 76.6x | 86.4x | **125.2x** |
| P1 / 正确性 | 8/8, 213t | 8/8, 42p | 8/8, 43p | **8/8, 60p** |

- **Blackwell 结论**：跨代正确性（六后端 + P1）完全成立；性能上 triton fp16 大 shape 3x vs fp32（跨精度），但**同精度 vs cuBLAS 无显著优势**（Blackwell cuBLAS 更强）；decode 摊销到 125x。
- cutile 在 Blackwell driver 610 下**可用**（cutile_probe=True），六后端在单设备上全可用。
- **cuda 后端（mlp_cuda）后续已在本机为 sm120 单独编译**（`TORCH_CUDA_ARCH_LIST=12.0 setup.py build_ext --inplace`，nvcc 13.2）：transformer_mlp -k cuda 15p/2s、原算子 cuda_kernels 16p、P1 8/8；fp16 prefill 三后端（eager/triton/cuda）18 rows 正确性 100%，cuda 后端 fp16 在 Blackwell 仍慢于 cuBLAS（M2048 34.7ms vs eager 6.1ms），如实记录。证据 = artifacts_5070ti/fp16_prefill_cuda_5070ti.json。
- **fp16 算子解锁（2026-09-02, 双架构验证）**：mlp_cuda 增 swiglu_fused/softmax/relu/gelu/silu 的 fp16 分派（fp16 in→fp32 math→fp16 out）；3070(sm86) 与 5070 Ti(sm120) dtype-matrix cuda-fp16 均为 **5 行 PASSED**；cuda fp16 block 不再回退 F.silu（norm_l2 5e-4）。
- **Blackwell 全量复现 216/235（0 failed，2026-09-02）**：在 5070 Ti 上 `tools/reproduce.py --skip-build` 全量跑通 —— 与 3070 冻结基线（235t/216p/0f/19s）**完全一致**；期间发现并修复 cuTile layernorm Blackwell bug（tile 512 + N<TILE 零填充垃圾 + 方差污染，commit 1707aed）。manifest = artifacts/20260902-045845-1a9e1c9-NVIDIA_GeForce_RTX_5070_Ti/（fix 已提交 1707aed）。

## 6. 完成后回报

- 三产物：manifest 链路(artifacts)、E4 对比表(追加 compat-matrix)、实验报告(docs/experiments/e4-<gpu>-<date>.md)
- 更新 EVIDENCE.md E4 状态列
