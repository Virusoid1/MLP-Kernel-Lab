# 5070 Ti E4 原始证据（raw JSON）

从 5070 Ti（Windows+WSL2, ssh alias 5070ti, user virusoid）scp 拉回的原始 benchmark 输出，冻结进 git 用于 E4 跨代（Blackwell sm120）证据链。

## 文件与来源

| 文件 | 内容 | 远端运行时间 |
|---|---|---|
| `fp16_prefill_5070ti.json` | prefill 6 shape × fp16（eager/triton/compile），18 rows，correctness_failed=0 | 2026-09-02 ~02:36 |
| `fp32_baseline_5070ti.json` | prefill 6 shape × fp32 eager 基线，6 rows | 2026-09-02 ~02:37 |
| `fp16_prefill_cuda_5070ti.json` | prefill 6 shape × fp16 eager/triton/cuda（cuda 后端 sm120 单独编译后），18 rows，correctness_failed=0 | 2026-09-02 ~02:48 |
| `swiglu_bench_fp16_bf16.json` | prefill 6 shape × {fp16,bf16} × 7 后端 = 84 rows，**correctness 84/84**；含 **cutile 在 Blackwell 显著慢于 Ampere** 的负结果 | 2026-09-02 ~04:18 |

## 环境（metadata / preflight）

- GPU: NVIDIA GeForce RTX 5070 Ti（16GB），**cc 12.0 = Blackwell sm120**，70 SM
- torch 2.13.0+cu130 / triton 3.7.1 / cuda-tile 1.3.0 / nvcc 13.2 / driver 610.88
- lane=blackwell（preflight 首次跨代实测识别）

## 关键结论（由本 JSON 直接算出）

- **跨精度** fp16 triton / fp32 eager：大 shape 2.62x-3.00x（2048×4096×11008 = 3.00x）
- **同精度** fp16 triton / fp16 eager：大 shape 0.97-1.07x（Blackwell cuBLAS 极强，Triton 无显著同精度优势）
- decode 摊销 M1→M256 = **125.2x**（Blackwell 更高带宽）
- P1 opcheck/gradcheck 8/8；transformer_mlp 60 passed
- **cuda 后端闭环**：mlp_cuda 为本机 sm120 单独编译（TORCH_CUDA_ARCH_LIST=12.0，nvcc 13.2），import OK；
  transformer_mlp -k cuda 17p（fp16+bf16 块级全过）+ dtype matrix cuda 16/16 PASS；cuda fp16 性能仍慢于 cuBLAS（如实记录）
- **cutile 在 Blackwell 的负结果（2026-09-02 实测）**：M512×4096×11008 cutile fp16 **62.7ms / bf16 59.9ms**（3070 同 shape 11.4ms，Blackwell 反而 5x 慢）——ct.mma 在 sm120 未获得 tcgen05 加速（或 tile 不匹配），**推翻"Blackwell 才是 cuTile 主战场"的旧假设**，如实记录

> 对应结论见 docs/e4-runbook.md §5.8。
