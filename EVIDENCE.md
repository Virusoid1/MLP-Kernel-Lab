# EVIDENCE.md — MLP-Kernel-Lab v2

> Claim → Evidence path → Status → Limitation。每条主张必须可追溯（commit / manifest / 测试报告）。
> 写简历或面试前先读这里；数字以"证据路径"为准，不引用本文之外的估算。
>
> 基线: v2-transformer-mlp @ `d2132d2`（2026-09-01）；环境: RTX 3070 Laptop (sm_86), torch 2.11.0+cu130, triton 3.6.0, cuda-tile 1.3.0, driver 610.88。

## 正确性

| Claim | Evidence Path | Status | Level |
|---|---|---|---|
| 全量测试通过 | `make reproduce`（commit 66cf3c1, 2026-09-01）→ `artifacts/20260901-025730-66cf3c1-*/manifest.json` | **212 tests: 175 passed / 0 failed / 37 skipped**（含 fp16 SwiGLU-block 训练） | E2 |
| 55 原始算子测试 | `tests/test_{triton,cuda,cutile}_kernels.py` | passed（并入 209 全量） | E2 |
| SwiGLU block 六后端正确性 | `tests/test_transformer_mlp.py`（DTYPE_SUPPORT 矩阵） | fp16: 六后端闭环（corr 100%）| E2 |
| 算子级 dtype 矩阵 | `tests/test_dtype_support_matrix.py`（执行式探测） | 20 passed / blocked-skips 记录边界 | E2 |
| 训练语义保持 | `tests/test_training_loop.py` | 4/4：pytorch/triton/cuda 收敛一致（loss 27.6→2.3, rel<0.085）| E2 |
| P1 torch.library 集成 | `tests/test_torch_registration.py` | 8/8：opcheck/gradcheck/compile/backward 全通过 | E2 |

## 性能

| Claim | Evidence Path | Status | Level |
|---|---|---|---|
| fp16 Triton 加速 | `bench/run.py` all-suite sweep → `artifacts/swiglu_20260901-013853-all-*/swiglu_bench.json` | **266-case all-suite best 3.52x vs eager-fp32**（decode+prefill+train 全含, corr 100%）；典型 prefill/train M≥512 达 2.4-3.5x，512×4096×11008 = 4.60ms | E3 |
| fp16 六后端性能地图 | `artifacts/swiglu_20260901-010220-prefill-*/swiglu_bench.json`（新 cutile tile） | triton 4.60 < compile 6.42 < triton_fused 7.02 < cutile 16.7 < cuda 29-118 ms（如实记录） | E3 |
| decode 摊销 | 实验报告更新 4 + README | per-token ↓80x（M=1→128）| E3 |
| cutile tile 优化 | `gpu_utils.py` + tile sweep | cutile matmul 1.5-1.6x（32,64,32 默认）| E3(部分) |
| fp32 算子级基线 | `results/op_bench_20260901_003415.json` | Triton avg 1.44x / CUDA avg 2.15x（谱系含 0.11x 拖累，已知）| E3(基线) |

## 系统 / 复现

| Claim | Evidence Path | Status | Level |
|---|---|---|---|
| make reproduce 一键复现 | `tools/reproduce.py` + `Makefile` | build→test→bench→manifest 全自动（含 CUDA 工具链自动探测）| E3 |
| Manifest 完整（commit/dirty/GPU/deps） | `artifacts/<run-id>/manifest.json` | capture_metadata 补足 triton/cutile/nvcc/git_dirty | E3 |
| 多机构建（Ampere/Blackwell） | `tools/preflight.py` + `docs/compatibility-matrix.md` | 3070 实测 status=0（Ampere lane）；3080Ti/3090Ti/5070Ti 代码就绪待实机 | E4(待) |
| 构建隔离 | `setup.py` build_base | build/py312-torch2.11.0-sm86/（架构不互污染）| E2 |

## 失败案例 / 负结果（面试重点）

| Case | Root Cause | Where Documented |
|---|---|---|
| decode 融合 kernel 无效 | decode 小 M 是权重带宽 bound（9.4MB@80GB/s），非 launch | 实验报告更新 2 |
| cuda 大 shape 0.1x | dispatch 为精度弃 WMMA（历史 L2 0.75 bug）→ 已用 matmul_half 解锁（L2 2e-4）| 实验报告更新 3/6 + wmma.cu 注释 |
| cuda/cutile fp16 性能慢 | wmma32/ct.mma tile 不敌 cuBLAS fp16 | 实验报告更新 8 + README |
| Triton silu/gelu fp16 编译失败 | tl.sigmoid 限 fp32 → 已修（升 fp32 再回落）| commit 03f7756 |

## 30s / 3min / 15min 讲法（速记）

- **30s**: 为 Transformer SwiGLU MLP 做了多后端（eager/compile/triton/cuda/cutile）kernel 实验系统，fp16 Triton 在 prefill/train 达 4x（vs cuBLAS-FP32），并完成六后端 fp16 正确性闭环 + 可复现 manifest。
- **3min**: 加正确性矩阵（opcheck/gradcheck/训练对齐）、性能协议（CUDA Event/manifest）、失败案例（decode 带宽 bound、cuda WMMA 精度之谜）；数字全部可追溯。
- **15min**: 完整故事——从问题→方案→测量→优化→失败→边界，见 `docs/claim-matrix.md` + `docs/experiments/swiglu-sweep-20260831-3070.md`。
