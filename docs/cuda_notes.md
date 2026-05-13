# CUDA 学习笔记

> Day 1-2 的 CUDA 基础概念和动手记录。
> 参考: [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## Day 1: 基础概念

### Host vs Device

```
Host    = CPU + 系统内存
Device  = GPU + 显存
```

关键: CPU 不能直接访问 GPU 显存，GPU 不能直接访问系统内存。
需要通过 `cudaMalloc` / `cudaMemcpy` 管理。

### Kernel Launch

```cpp
kernel_name<<<grid, block, shared_mem, stream>>>(args);
```

- `grid`: block 数量 (dim3)
- `block`: 每个 block 的 thread 数量 (dim3)
- `shared_mem`: 动态 shared memory 大小 (可选)
- `stream`: CUDA stream (可选)

### Thread Hierarchy

```
Grid
  └── Block (0,0)
  │     ├── Thread (0,0,0)
  │     ├── Thread (1,0,0)
  │     └── ...
  └── Block (0,1)
        ├── Thread (0,0,0)
        └── ...
```

索引:
- `threadIdx` : thread 在 block 内的索引
- `blockIdx`  : block 在 grid 内的索引
- `blockDim`  : block 的尺寸 (= 每个 block 的 thread 数)
- `gridDim`   : grid 的尺寸 (= block 数量)

### Memory Hierarchy

| 类型 | 作用域 | 延迟 | 大小 |
|------|--------|------|------|
| Register | Thread | ~0 | 很少 (通常 255/thread) |
| Shared Memory | Block | ~20 cycles | 48-164 KB/SM |
| L1 Cache | SM | ~30 cycles | 128 KB/SM |
| L2 Cache | GPU | ~200 cycles | 几 MB |
| Global Memory (HBM) | GPU | ~600 cycles | 16-80 GB |

### 性能关键原则

1. **Coalesced Access**: 同一个 warp 的 thread 访问连续地址时，可合并为一次 transaction
2. **Occupancy**: 每个 SM 上同时运行的 warp 越多，隐藏延迟的能力越强
3. **Shared Memory**: block 内 thread 共享的快速缓存，用于 tiling

### CUDA Event Timing

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

## Day 2: Naive Matmul 分析

### 为什么 naive matmul 慢？

每个 thread 计算 C[row, col] 时:
- 需要 for (int k=0; k<K; k++)
- 每次迭代从 global memory 读 A[row, k] 和 B[k, col]
- 总共: 每个 thread 2*K 次 global memory 读取
- 但 A 的同一行被 N 个 thread 重复读取
- B 的同一列被 M 个 thread 重复读取

**改善方向**: 把 A/B 的 tile 缓存在 shared memory 中，让同一 block 的 thread 共享。

### 实际观察

<!-- TODO: 填写你的 benchmark 结果 -->

```
Shape: M=___, K=___, N=___
Naive latency: ___ ms
Naive TFLOPS: ___
```

## Day 4: Tiled Matmul

### 核心思想

```
把 A[M,K] 按 BLOCK_M x BLOCK_K 分块
把 B[K,N] 按 BLOCK_K x BLOCK_N 分块
每轮: 加载一对 tile 到 shared memory -> 同步 -> 计算 -> 同步 -> 下一对
```

### 代码结构

```cpp
for (int tile_k = 0; tile_k < K; tile_k += BLOCK_K) {
    // 1. 每个 thread 加载 A tile 的一部分 (注意 coalesced)
    // 2. 每个 thread 加载 B tile 的一部分
    // 3. __syncthreads()
    // 4. 计算 partial sum: for k in tile, acc += sA[ty][k] * sB[k][tx]
    // 5. __syncthreads()  // 等待所有 thread 用完 shared memory 再写下一轮
}
```

### `__syncthreads()` 位置要点

- 必须在 shared memory 写入后、读取前
- 必须在更新 shared memory 前完成上一轮读取
- 错误放置会导致数据竞争

### Debug 建议

- 先用小 shape (32x32x32) 验证正确性
- 检查 boundary: M % BLOCK_M, K % BLOCK_K, N % BLOCK_N 不为 0 时是否正确
- 用 `cuda-memcheck` 检查 out-of-bound access

## Day 5: Operator Fusion

### 为什么 fusion 有收益？

Unfused 流程:

```
Kernel 1: C = X @ W1        // 写 C 到 global memory
Kernel 2: C += bias          // 读 C, 写 C
Kernel 3: C = GELU(C)        // 读 C, 写 C
```

Fused 流程:

```
Kernel 1: C = GELU(X @ W1 + bias)  // 只写一次 C
```

省掉了 2 次 global memory 写入和 2 次读取。

### 在 tiled matmul 中融合

在 `acc` 计算完成后、写回 global memory 前:

```cpp
acc += bias[col];            // col 是当前 thread 的输出列
acc = gelu_device(acc);      // 在寄存器中完成，极快
C[row * N + col] = acc;
```

## Day 10: FP16 支持

### `half` 类型

```cpp
#include <cuda_fp16.h>
__half a = __float2half(1.0f);
float b = __half2float(a);
```

### `half2`

一次处理两个 half，提高吞吐：

```cpp
__half2 a2 = __floats2half2_rn(1.0f, 2.0f);
__half2 b2 = __hadd2(a2, a2);  // elementwise add
```

累加建议用 `float` 避免精度问题。

### Tensor Core (WMMA) - Future Work

```cpp
#include <mma.h>
// wmma::load_matrix_sync
// wmma::mma_sync
// wmma::store_matrix_sync
```

> WMMA 作为项目 Future Work，两周内不强求手写。
