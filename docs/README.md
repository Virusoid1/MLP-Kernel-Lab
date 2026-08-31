# Docs 导航

按"从哪看 → 看什么"组织。新读者建议从顶部一路向下。

## 入口顺序

1. **README**（../README.md）— 项目定位 + 当前主线结果摘要
2. **claim-matrix**（claim-matrix.md）— 每个主张 → 源码/证据路径 → 证据级别（E0–E4）。评审/面试/写简历前必读
3. **实验报告**（experiments/）— 每个实验的完整数据：协议、加速比矩阵、失败案例、复现命令
   - experiments/swiglu-sweep-20260831-3070.md — SwiGLU block 首个 E3 基线 + fp16 胜势 + decode/cuda 根因

## 工具

| 工具 | 用途 |
|---|---|
| tools/status.py | 一键项目状态总览（git/测试/manifest/最新 sweep） |
| tools/reproduce.py | make reproduce 驱动：build→test→bench→manifest 归档 |
| tools/render_swiglu.py | swiglu_bench.json → Markdown/CSV 渲染 |
| bench/run.py | SwiGLU block 性能基准（CUDA Event + manifest） |

## 历史 / 学习材料（legacy）

| 文档 | 内容 |
|---|---|
| optimization-guide.md | 测量-分析-优化方法论（P0.5 闭环） |
| optimization_log.md | 度量 + verify 命令日志（decision tree 风格） |
| cuda-guide / triton-guide / cutile-guide / pytorch-guide | 各后端学习笔记 |
| cuda_notes / triton_notes.md | 踩坑记录 |
| report.md | 历史项目报告 |

> Legacy 文档多为 MNIST 训练/算子合集时代产物，结论若与 claim-matrix 冲突，以 claim-matrix 为准。
