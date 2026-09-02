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

## v2.7 cp.async 双缓冲管线（2026-09-02 第二迭代）

对齐 shape（M/K/N %32==0）走 cp.async 2-stage 管线内核；非对齐回落同步内核。

| 指标（3070 fp16） | 同步（v2.5 R32） | cp.async 管线 | 提升 |
|---|---|---|---|
| matmul 512×4096×11008 单发 | 10.84ms | **6.85ms** | **1.58x** |
| 块 M512×4096×11008 | 33.3ms | **22.8ms** | **1.46x**（ratio vs eager 6.2x→3.6x） |
| 块 M2048×4096×11008 | 124.7ms | **85.3ms** | 1.46x |
| 块 M512×768×3072 | 1.86ms | **1.12ms** | 1.66x |

- 关键修复过程：cp.async 后缺少可见性 __syncthreads（他人写入不可见 → nan）；
  且不能预取 kt+2（与当前缓冲 (kt&1) 相同 → 覆写）。正确序：issue(kt+1) → wait_prior(1) → syncthreads → mma → syncthreads。
- 正确性：aligned/ragged fp16 rel_l2 2.1e-4/1.9e-4、bf16 1.7e-3；tests 全绿（块 17p、cuda_kernels 25p）。
- 待续：gate+up 融合（A 一次读）、向量化 ldmatrix 加载（v2.8）。
## v2.9 gate+up 融合 matmul（2026-09-02 第四迭代）

块 fp16 的 gate=X@Wg 与 up=X@Wu 共享 A=X：融合为单内核（A 一次读、双 B/双 acc、共享 sC 顺序 epilogue），块级 3→2 次 matmul 数据流。

| 指标（fp16） | 分离（v2.7 管线） | 融合 pair | 再提升 | 相对原始基线 |
|---|---|---|---|---|
| 块 M512×4096×11008 | 22.8ms | **18.6ms**（ratio 3.6x→2.8x） | 1.22x | **33.3→18.6 = 1.79x** |
| 块 M2048×4096×11008 | 85.3ms | **71.6ms** | 1.19x | 124.7→71.6 = 1.74x |
| 块 M512×768×3072 | 1.12ms | **0.95ms** | 1.18x | 1.86→0.95 |

- 绑定：`matmul_half_pair`/`matmul_bf16_pair`（非对齐 shape 内部回落两次单 matmul）。
- 正确性：块 cuda 17p、dtype matrix 16p、cuda_kernels 25p 全绿。
- 目标指标：cuda 块 fp16 vs eager **0.16x → ~0.35x**（累计 2.2x）。
## v2.8 STAGES 深度管线实验（2026-09-02 第三迭代）

管线内核模板化（STAGES 编译期实例 2/3/4）：

| STAGES | 块 med (M512×4096×11008 fp16) | vs eager |
|---|---|---|
| 2 | 24.4ms | 3.7x |
| 3 | 24.1ms | 3.7x |

**结论：2-stage 最优** —— 更深管线（smem 16KB+）占用损失抵消延迟隐藏收益。保持 STAGES=2。下一优化点：**gate+up 融合 matmul（A 一次读，块级 3→2 次 matmul 数据流）**。
