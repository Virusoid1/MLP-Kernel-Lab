# REPRODUCE.md — 从零复现 v2 实验

> 目标：任何机器（3070 / 3080 Ti / 3090 Ti / 5070 Ti）上，从克隆到产出完整可追溯结果，
> 只需 4 条命令。本文件是唯一入口；细节见 EVIDENCE.md（claim→证据）、KNOWN-LIMITATIONS.md（边界）。

## 1. 环境要求

- **Linux 或 WSL2**（Windows 用户推荐 WSL2 + Ubuntu 22.04/24.04）
- **Python 3.12**（本项目验证于 3.12.3）
- **CUDA**: 自动探测（`tools/reproduce.py` 会自动匹配 `/usr/local/cuda-*`），验证于 CUDA 13.2
- **GPU**: sm_80 及以上（Ampere/Ada/Hopper/Blackwell）；开发机为 RTX 3070 Laptop (sm_86)

### 依赖版本（已锁定，经验证）

```
Python 3.12.3
torch 2.11.0+cu130        (pip install torch --index-url https://download.pytorch.org/whl/cu130)
triton 3.6.0              (随 torch 附带或 pip install triton)
cuda-tile 1.3.0           (pip install cuda-tile; import 名为 cuda.tile)
pytest 9.0.3
nvidia-smi 驱动 ≥ 610
```

## 2. 一键复现（4 条命令）

```bash
git clone https://github.com/Virusoid1/MLP-Kernel-Lab.git
cd MLP-Kernel-Lab && git checkout v2-transformer-mlp

export CUDA_HOME=$(ls -d /usr/local/cuda-* | sort -V | tail -1)
export PATH=$CUDA_HOME/bin:$PATH

# 一条命令：preflight(架构/lane) -> build -> 全量测试(213) -> bench smoke -> manifest 归档
make reproduce PYTHON=/path/to/venv/bin/python
```

- 产物：`artifacts/<run-id>/manifest.json`（commit/GPU/驱动/CUDA/torch/triton 全字段）、summary.md、correctness/benchmark 数据
- 无 CUDA 扩展也可跑 Triton 路径（C++ 扩展仅 CUDA 后端需要）

### 版本 2 快速验证（其它机器首次）

```bash
bash scripts/verify.sh /path/to/venv/bin/python
# preflight -> import smoke -> quick tests -> bench(含 GPU 热采集) -> status
```

### 单步复现

```bash
# 全量测试（213 项）
python -m pytest tests/ -q --tb=short

# SwiGLU block 性能（CUDA Event, manifest）
python bench/run.py --suite all --dtypes fp32,fp16 --backends eager,concat,triton,triton_fused,cuda,cutile,compile

# GPU 热稳定/节流采集（任意命令包裹）
python scripts/gpu_telemetry.py --cmd "python bench/run.py --suite prefill --dtypes fp16"

# 多机 preflight（架构 lane / 工具链一致性）
python tools/preflight.py
```

## 3. 预期结果（3070 Laptop, 2026-09-01, commit 66cf3c1）

| 项 | 预期 |
|---|---|
| make reproduce | **226 tests: 203 passed / 0 failed / 23 skipped**（f43dec8 冻结；+4 = cuda bf16 全自定义） |
| fp16 六后端正确性 | norm_l2 2.4e-4 ~ 6e-4（eager/triton/triton_fused/cuda/cutile/compile） |
| fp16 Triton 加速 | all-suite 266-case best **3.52x** vs eager-fp32（prefill/train M≥512 2.4-3.5x） |
| 热状态（大负载） | 73°C / 139.8W / util 100% / throttled=true（笔记本节流事实） |
| fp16 训练 | eager 与 Triton 均收敛（loss rel<0.005） |

> 不同 GPU（3080 Ti/3090 Ti/5070 Ti）数字会因带宽/SM 数变化；正确性契约不变。
> 架构适配：`setup.py` 动态探测；Blackwell (sm_120) 需要 CUDA ≥ 13 + 对应 torch。

## 4. 数字口径

- **性能**：CUDA Event 计时，warmup + median；strict FP32 对照（TF32 关闭）
- **正确性**：one-shot 归一化误差 norm_l2 / 训练 loss 收敛
- **可追溯**：每个数字可定位到 `artifacts/<run-id>/manifest.json`（含 git commit，防"旧数字"）
