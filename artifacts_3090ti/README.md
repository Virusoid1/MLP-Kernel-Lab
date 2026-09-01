# 3090 Ti E4 原始证据（raw JSON）

从 3090 Ti 集群机（10.154.32.40:41035, user catlab）scp 拉回的原始 benchmark 输出，冻结进 git 用于 E4 复现证据链。

## 文件与来源

| 文件 | 内容 | 远端运行时间 |
|---|---|---|
| `fp16_prefill_3090ti.json` | prefill 6 shape × fp16（eager/triton/compile），18 rows，correctness_failed=0 | 2026-09-01 ~21:48 |
| `fp32_baseline_3090ti.json` | prefill 6 shape × fp32 eager 基线，6 rows | 2026-09-01 ~21:48 |

## 环境（metadata）

- GPU: NVIDIA GeForce RTX 3090 Ti（双卡 24GB，本归档 GPU0）
- torch 2.9.1+cu126 / triton 3.5.1 / Python 3.12.3 / driver 535.161.07

## 关键结论（由本 JSON 直接算出）

- fp16 triton vs fp32 eager（跨精度）：2.46x-3.40x
- 2048×4096×11008：fp32 22.0ms → fp16 triton 7.4ms（2.97x）
- 正确性 norm_l2 与 3070 同档

> 注：生成于"预打包/缓存编译/replicates"加固之前的 bench 版本；历史实测，如实冻结。
> 对应结论见 docs/e4-runbook.md §5.6。
