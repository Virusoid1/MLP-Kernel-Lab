# 3080 Ti E4 原始证据（raw JSON）

从 3080 Ti 集群机（10.154.32.34:41028, user catlab）scp 拉回的原始 benchmark 输出，冻结进 git 用于 E4 复现证据链（不再只是"文档声称"）。

## 文件与来源

| 文件 | 内容 | 远端运行时间 |
|---|---|---|
| `fp16_prefill_3080ti.json` | prefill 6 shape × fp16（eager/triton/compile 等），18 rows，correctness_failed=0 | 2026-09-01 ~18:59 |
| `fp32_baseline_3080ti.json` | prefill 6 shape × fp32 eager 基线，6 rows | 2026-09-01 ~19:00 |

## 环境（来自 metadata）

- GPU: NVIDIA GeForce RTX 3080 Ti（双卡，本归档用 GPU0）
- torch 2.9.1+cu126 / triton 3.5.1 / Python 3.12.3 / driver 535.104.05

## 关键结论（由本 JSON 直接算出）

- fp16 triton vs fp32 eager（跨精度）：2.16x-3.69x
- fp16 triton vs fp16 eager（同精度）：主 shape ~1.09x（见 same-precision 分析）
- 正确性 norm_l2 与 3070 同档（4.8e-4 级）

> 注：当时跑在"预打包/缓存编译/replicates"加固之前的 bench 版本；数据为历史实测，如实冻结。
> 对应结论见 docs/e4-runbook.md §5.5。
