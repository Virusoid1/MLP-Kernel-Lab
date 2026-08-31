# Claim–Evidence 矩阵（v2 基线）

> 更新于：v2 启动（`v2-transformer-mlp` 分支基线 = 远程 master `d41f257`，2026-06 状态）。
> 目的：把 README / CHANGELOG / 简历中的每个主张映射到具体证据路径，并按证据等级标记，
> 防止"未复跑的历史数字"与"当前可定位的结果"混写。
>
> 证据等级（沿用 `resume/interview-prep/.../10-project-experiment-roadmap.md`）：
> E0 计划 / E1 实现 / E2 正确性（当前环境测试+reference+边界）/ E3 性能（协议+raw data+profiles）/ E4 复现（第二设备/独立脚本）。

## 说明（本文件的前提）

- 本矩阵基于 **GitHub 远程 master（`d41f257`）** 的仓库内容编写，不是 WSL 本地旧树（`1c6e586`）。
- WSL 本地旧树比远程落后 11 个 commit（含 `d621338` 重构：拆分 `kernels/mlp/*.cu`、删除 `bench/`、新增 `tools/`）。
- 因此：**以这个文件为准**，而不是任何基于旧树的讨论。旧树事实仅作为 legacy 参考。

## 主张矩阵

| 主张 | 当前源码/证据路径 | 当前运行证据 | 状态 | 级别 |
|---|---|---|---|---|
| 三个自定义 GPU 后端（Triton / CUDA / cuTile）可 import | `triton_kernels/`、`kernels/mlp/*.cu` + `setup.py`、`cutile_kernels/` | 未在当前环境复跑（缺 pytest/GPU 复验） | 实现存在，未复核 | E1 |
| 四后端 FP32 精度对齐 98.52%–98.71% | CHANGELOG 2026-06-03 #1（wmma64 修复后 15-epoch） | `results/compare_20260602_235625.json`（5070 Ti 实测），README 完整 4-backend 实测小节 | 历史结果（RTX 5070 Ti），当前 GPU 未复跑 | E3(5070 Ti) / 未复现 |
| 55 项 Python + 6 项 C++ 测试通过 | `tests/test_triton_kernels.py` / `test_cuda_kernels.py` / `test_cutile_kernels.py` / `test_kernels.cu` | **实际 pytest 函数数 = 18 + 13 + 15 = 46**（README 写 55 已过时）；当前环境缺 pytest → 未复跑 | 测试数量需以真实报告为准，不沿用 55 | E1 |
| 精度控制 TF32/FP32 全局切换，多后端公平对比 | `triton_kernels/precision.py`、`run_compare.py --precision`、`benchmark_ops.py --precision/--ref-tf32` | 旧 `results/baseline.json` 为 tf32；`--ref-tf32` 可切 ref | 实现存在 | E1 |
| 统一算子级 benchmark 入口 | `benchmark_ops.py`（唯一权威入口；`bench/` 已删除） | `make bench` / `make bench-quick` / `make test` → `benchmark_ops.py` | 已统一（比 v2 计划预期的更早完成） | E2 |
| Manifest 元数据（GPU/driver/torch/git） | `python/mnist/benchmark.py::capture_metadata()` + `benchmark_ops.py::export_json()`（`{metadata, rows}` schema） | `results/baseline.json`（5070 Ti, git_sha=8991e9d） | 部分实现；**缺 triton/cutile/nvcc 版本、git_dirty、raw samples** | E3 雏形 |
| 测量-分析-优化闭环 | `Makefile bench-op/analyze/gate`、`tools/analyze_bench.py`、`tools/run_full_eval.sh`、`tools/gpu_warmup.py` | `results/baseline*.json` + CHANGELOG 编号条目 | 已存在 | E2 |
| Nsight workflow | `profiling/run_ncu.sh`（rewrite, 与已删 `bench/` 解耦）、`profile_nsys.sh`、`profile_ops.py` | `CHANGELOG` 记录 ncu 使用（wmma64 / launch_bounds 定位） | 实现存在 | E2 |
| CUDA 端到端性能（训练总时长/推理延迟最低） | CHANGELOG 2026-06-03 #1（CUDA 27.0s / 0.298ms / 858K samples/s） | `results/four_backend_fp32_v2.log`, `results/compare_20260602_235625.json` | **仅 5070 Ti**；3070 Laptop 未复跑；机制=WMMA64+launch_bounds 修复 | E3(5070 Ti) |
| cuTile 单 step 微基准 | `profiling/bench_cutile.py`（4 轮取后 3，warmup） | `results/cutile_bench.json`（5070 Ti FP32） | 仅 5070 Ti；3070 未验证 | E3(5070 Ti) |
| Transformer MLP 推理主线（SwiGLU block 闭环） | 仅有算子：`triton_kernels/swiglu_triton.py`、`cutile_kernels/swiglu_cutile.py`、`kernels/mlp/fused.cu`；**无 `X@W_gate / X@W_up → SiLU(gate)*up → @W_down` 完整 block** | 无 decode/prefill shape sweep；无 eager/compile/自定义后端对比 | **未闭环（v2 主攻方向）** | E1 |
| PyTorch 自定义算子集成（schema/FakeTensor/opcheck/gradcheck/compile） | 现有 autograd 层（`triton_layers.py` 等）为 `torch.autograd.Function`；**无 torch.library schema/FakeTensor/opcheck 证据** | 未做 | **未实现（v2 计划 P1）** | E0 |
| 构建架构可移植（非硬编码 sm_120） | `setup.py` 仍硬编码 `compute_86/sm_86` + `compute_120/sm_120`；`Makefile` test-cuda 硬编码 `sm_86` | 3070 Laptop 构建未验证干净（依赖现存根目录 `.so`） | **hardcode 待改造（v2 G1）** | E1 |
| 不依赖根目录旧 `.so` | 根目录 `mlp_cuda.cpython-312-...so` 为已构建产物（被 `*.so` 忽略） | 未验证"从干净环境仅通过 make install 生成" | **待验证（v2 G1）** | E1 |

## Legacy 参照（旧树事实，仅作背景，不视为当前状态）

- WSL 旧树（`1c6e586`）曾含 `bench/benchmark.py` 占位实现（`run_cuda_*`/`run_triton_matmul` 为 `NotImplementedError`）——
  已在远程 `d621338` 删除，`bench/` 目录不存在，故"接通 bench/benchmark.py"计划项**作废**。
- WSL 旧树曾含 `python/mlp_reference.py`（未跟踪文件）——远程无此文件；reference 将由 v2 的
  `python/transformer_mlp.py`（或 `bench/correctness.py`）重新承载。
- WSL 本地有约 40 个未跟踪 `results/*.json`（含 `op_bench_stable_*.json` 的 Triton L2 接近 1 的告警数据）——
  未进入 git；v2 阶段一不覆盖、不转存，只登记为 legacy。

## 使用规则

1. 任何 README / 简历 / 报告引用本矩阵中的主张，必须引用对应证据路径（runs/JSON/log）。
2. 新实验一律写入 `experiments/YYYYMMDD_<commit>/`（不可覆盖），并带 manifest。
3. "测试数 X 项"只能从真实 pytest 报告生成，不再引用 README 旧数字。
4. 性能数字必须标注 GPU、dtype、shape、协议；跨 GPU 不比毫秒，只比"相对同机 baseline 的 speedup"。
