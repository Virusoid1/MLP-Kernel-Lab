# 3070 E3 性能证据（raw JSON）

从 3070（sm86, WSL, torch 2.11.0+cu130 / triton 3.6.0 / cuda-tile 1.3.0 / nvcc 13.2 / driver 610.88）冻结的原始 bench 输出。

## 文件与来源

| 文件 | 内容 | 说明 |
|---|---|---|
| `swiglu_bench_fp16_bf16.json` | prefill 6 shape × {fp16, bf16} × 7 后端（eager/concat/triton/triton_fused/cuda/cutile/compile）= **84 rows** | 2026-09-02，git_sha=6e8662c（cuda fp16+bf16 全自定义后），`--warmup 5 --iters 20` |

## 关键结论（由本 JSON 直接算出）

- **正确性 100%**：84/84 correctness_passed=true，0 failed（fp16 l2~3-6e-4；bf16 l2~2-5e-3，bf16 尾数更粗符合预期）
- **M=512×4096×11008 fp16**：triton 4.52ms（1.12x vs eager）≈ eager 5.08 ≈ compile 5.07 > concat 5.72 > triton_fused 6.33 > cutile 11.4 > cuda 31.3ms（如实记录）
- **M=512×4096×11008 bf16**：eager 4.52 ≈ concat 4.39 ≈ compile 4.57 ≈ triton 4.56（同精度几乎打平）> triton_fused 6.14 > cutile 10.8 > cuda 30.5ms
- best_speedup（对同 dtype eager）：**3.454**（跨 shape 大 M 场景）
- cuda/cutile 自定义后端在 3070 上同精度仍慢于 cuBLAS（WMMA/ct.mma tile 不敌），如实记录

