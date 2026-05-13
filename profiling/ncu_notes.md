# Nsight Compute Profiling Notes

> 用 `ncu --set full` 或 `ncu --set roofline` 分析 kernel 后，在此记录关键指标和观察。

## 关键指标速查

| 指标 | 含义 | 目标 |
|------|------|------|
| `achieved_occupancy` | 实际 Active warps / 理论最大 warps | >50% |
| `gld_efficiency` | Global Load 有效带宽利用率 | >80% |
| `gst_efficiency` | Global Store 效率 | >80% |
| `shared_load_transactions` | Shared memory bank conflict | 越少越好 |
| `stall_long_scoreboard` | 等 global memory 导致的 stall | 越低越好 |
| `stall_math_pipe_throttle` | 计算单元不够用 | 可与上面对比 |
| `sm_efficiency` | SM 利用率 | 越高越好 |
| `mem_pipe_to_fp64_ratio` | 内存带宽 vs 计算比率 | roofline 分析用 |

## 待记录

在 profiling 后填写以下格式：

```text
Kernel: matmul_naive_kernel
Shape: M=512, K=768, N=3072
Block: (16, 16)
Occupancy: __%
Global Load Efficiency: __%
Shared Memory: __ bytes/block
Registers: __ /thread
主要 stall 原因: ___
改进方向: ___
```

## 实际 profiling 结果

<!-- TODO: 粘贴你的 ncu 输出 -->

### matmul_naive

```
(待填写)
```

### matmul_tiled

```
(待填写)
```

### mlp_fused

```
(待填写)
```

## Roofline 分析

<!-- TODO: 绘制 arithmetic intensity vs TFLOPS 图 -->

```
(待填写)
```
