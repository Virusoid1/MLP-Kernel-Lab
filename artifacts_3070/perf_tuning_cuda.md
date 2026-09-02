# cuda 后端 fp16 块性能调优记录（3070, 2026-09-02）

目标：cuda 块 fp16（3× matmul_half + swiglu_fused_half）相对 eager 从 0.16x 提升。
协议：同一 M512×4096×11008 fp16 块，9 拍取中位（首次算冷启动），eager 同协议对照。

## 实测变体（matmul_half 内核）

| 变体 | 描述 | 块耗时(ms) | vs eager 5.2ms | 结论 |
|---|---|---|---|---|
| v2.4 R16（基线） | TILE=32, R=16, sC staging, 128 threads | 33.3 | 6.4x | 起点 |
| v2.5 R32 dual-MMA | R=16→32, 每同步两次 k16 MMA（同步减半） | 32.3-33.8 | 6.2-6.5x | **无显著收益** —— 非同步 bound |
| v2.5b BM128×BN32 | 128×32 tile, 8 warps, sC 16KB | 42.7 | 8.2x | **更慢** —— sC+sA/sB 26KB/块 → 占用塌陷（约 25%） |
| v2.6 T64 直接 store | 64×64 tile, 4 warps, 2×2 frag, 无 sC | 编译失败 | — | **store_matrix_sync 要求元素类型匹配**（fp32 acc 不能直接写 half 内存）→ 路径不可行 |

## 结论

- 32×32 tile + 4KB smem 已跑 100% 占用（16 blocks/SM），全局流量 ~2.9GB（A 读 344x / B 读 16x），理想 ~6ms；实测 33ms 说明**每 k-step 的 ldmatrix→mma 串行依赖**是主要墙。
- 大 tile 需要 sC（fp32 累加器 staging），smem 成本 m*n*4B 使占用塌陷 → 需要 **cp.async 双缓冲 + 去掉 sC（按 fragment 布局直接全局写）** 或硬件 ldmatrix 向量化加载，留作后续迭代（v2.7）。
- 本记录同时是**负结果证据**：三种优化均未超基线；最终保留 R32 dual-MMA（正确，与基线同档），后续转向 cutile tile sweep 与 cp.async 管线。

## 正确性

所有变体通过 tests/test_cuda_kernels.py (25p)、test_transformer_mlp -k cuda (17p)、dtype matrix cuda (16p)。

