# CUDA & Triton MLP Kernel Optimization Report

> 项目最终报告。标记为 "TODO" 的部分需要填写实际数据。

## 1. Motivation

Transformer MLP (FFN) 是 LLM inference 中的核心计算模块。以 Llama 为例：

```
Y = SiLU(X @ W_gate) * (X @ W_up) @ W_down
```

对于 hidden_size=4096, intermediate_size=11008 的典型配置，
每个 token 需要 ~360M FLOPs。优化这部分 kernel 对 LLM serving 有直接价值。

本项目实现和 benchmark 了从 naive 到优化的 CUDA/Triton MLP kernel，
覆盖 matmul、fused activation、SwiGLU 等核心算子。

## 2. Implementations

| 实现 | 描述 | 文件 |
|------|------|------|
| PyTorch | `torch.matmul` / `F.gelu` baseline | `python/mlp_reference.py` |
| CUDA naive | 每个 thread 算一个 output element | `kernels/matmul_naive.cu` |
| CUDA tiled | Shared memory tiled matmul | `kernels/matmul_tiled.cu` |
| CUDA fused | Fused matmul + bias + GELU | `kernels/mlp_fused_first_layer.cu` |
| CUDA SwiGLU | Fused SiLU(gate) * up | `kernels/swiglu_fused.cu` |
| Triton matmul | `tl.dot` based block matmul | `triton_kernels/matmul_triton.py` |
| Triton MLP | Fused matmul + bias + GELU | `triton_kernels/mlp_triton.py` |
| Triton SwiGLU | Fused SiLU(gate) * up | `triton_kernels/swiglu_triton.py` |

## 3. Correctness

FP32:
- max_abs_error < 1e-3
- mean_abs_error < 1e-5

FP16:
- max_abs_error < 1e-1
- mean_abs_error < 1e-2

<!-- TODO: 填写实际 correctness 结果 -->

| Implementation | dtype | M | K | N | max_abs_err | mean_abs_err |
|---------------|-------|---|---|---|-------------|--------------|
| cuda_naive | fp32 | 512 | 768 | 3072 | ___ | ___ |
| cuda_tiled | fp32 | 512 | 768 | 3072 | ___ | ___ |
| triton_matmul | fp32 | 512 | 768 | 3072 | ___ | ___ |

## 4. Benchmark Setup

**GPU**: ___ (A100 / RTX 4090 / ...)
**CUDA**: ___ (12.x)
**PyTorch**: ___ (2.x)
**Triton**: ___ (2.x)
**Driver**: ___

**Benchmark 方法**:
- 每个 shape 预热 20 次
- 测量 100 次取 median + p95
- CUDA event 计时 + `torch.cuda.synchronize()`
- 固定随机种子 42

## 5. Results

### 5.1 Latency vs M

<!-- TODO: 将 plot 插入这里 -->
```
(See plots/latency_vs_hidden.png)
```

### 5.2 Speedup vs Naive CUDA

<!-- TODO: 填写表格 -->

| M | K | N | cuda_naive | cuda_tiled | speedup |
|---|---|---|------|------------|---------|
| 128 | 4096 | 11008 | ___ ms | ___ ms | ___x |
| 512 | 4096 | 11008 | ___ ms | ___ ms | ___x |
| 2048 | 4096 | 11008 | ___ ms | ___ ms | ___x |

### 5.3 Speedup vs PyTorch

| M | K | N | torch | cuda_tiled | triton | best_speedup |
|---|---|---|-------|------------|--------|-------------|
| 128 | 4096 | 11008 | ___ ms | ___ ms | ___ ms | ___x |
| 512 | 4096 | 11008 | ___ ms | ___ ms | ___ ms | ___x |
| 2048 | 4096 | 11008 | ___ ms | ___ ms | ___ ms | ___x |

### 5.4 TFLOPS

<!-- TODO: 填写 -->

| M | K | N | cuda_naive | cuda_tiled | triton |
|---|---|---|------------|------------|--------|
| 128 | 4096 | 11008 | ___ TFLOPS | ___ TFLOPS | ___ TFLOPS |
| 512 | 4096 | 11008 | ___ TFLOPS | ___ TFLOPS | ___ TFLOPS |
| 2048 | 4096 | 11008 | ___ TFLOPS | ___ TFLOPS | ___ TFLOPS |

## 6. Profiling

### 6.1 Nsight Compute Observations

<!-- TODO: 填写实际 profiling 结果 -->

**matmul_naive**:
- Occupancy: ___%
- Global Load Efficiency: ___%
- 主要 stall: ___
- 瓶颈: ___

**matmul_tiled**:
- Occupancy: ___%
- Global Load Efficiency: ___%
- Shared Memory: ___ bytes/block
- 主要 stall: ___

### 6.2 Roofline Analysis

<!-- TODO: 根据 ncu --set roofline 结果填写 -->

## 7. Optimization Analysis

### 7.1 Shared Memory Tiling

> 将 A 和 B 的 tile 加载到 shared memory，同一 block 内 thread 共用。
> 减少了重复 global memory 访问，提高了 arithmetic intensity。

效果: ___x speedup vs naive

### 7.2 Operator Fusion

> 将 bias add 和 GELU 融入 matmul kernel。
> 省去了中间 tensor 的 global memory 写回和重新读取。

效果: ___x speedup vs unfused

### 7.3 FP16

> FP16 减少了一半的 memory 带宽需求，同时现代 GPU 的 FP16 吞吐是 FP32 的 2x+。

效果: ___x speedup vs FP32

### 7.4 Triton vs CUDA

> Triton 用 Python DSL 实现相同逻辑，编译器自动生成高效 GPU 代码。
> 开发效率显著高于手写 CUDA，但精细控制不如 CUDA。

## 8. Limitations

1. **未使用 Tensor Core WMMA**: CUDA kernel 仍用 FMA 指令，未利用 Tensor Core
2. **未超过 cuBLAS**: PyTorch 底层调用 cuBLAS，我们还未达到这个水平
3. **有限的 shape 覆盖**: 仅测试 LLM-like shapes
4. **单卡**: 未测试多卡
5. **未做 war 级别的优化**: warp-level reduction, double buffering 未实现

## 9. Future Work

- 用 WMMA 或直接使用 Tensor Core intrinsics
- CUTLASS 对比
- Persistent kernel 减少 launch overhead
- 与 `torch.compile` / `torch.library` 集成
- Auto-tuning block size
- 扩展到 FP8 / INT8
- 扩展到完整的 MLP 两层 (SwiGLU + down projection)

## 10. References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Triton Documentation](https://triton-lang.org/)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
