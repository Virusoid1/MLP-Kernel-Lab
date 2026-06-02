# CUDA Kernel 编程指南

> 面向本项目的 C++ CUDA kernel 实现，从 naive 到 WMMA Tensor Core 的完整优化路径。

## 目录

- [1. CUDA kernel 基础](#1-cuda-kernel-基础)
- [2. Naive 矩阵乘法](#2-naive-矩阵乘法)
- [3. Shared Memory Tiled 矩阵乘法](#3-shared-memory-tiled-矩阵乘法)
- [4. 多配置 Dispatch](#4-多配置-dispatch)
- [5. WMMA FP16 Tensor Core](#5-wmma-fp16-tensor-core)
- [6. Activation Kernel 与向量化](#6-activation-kernel-与向量化)
- [7. LayerNorm: Warp Shuffle 归约](#7-layernorm-warp-shuffle-归约)
- [8. 融合 Kernel](#8-融合-kernel)
- [9. PyTorch C++ Extension 绑定](#9-pytorch-c-extension-绑定)
- [10. 编译与部署](#10-编译与部署)

---

## 1. CUDA kernel 基础

### 1.1 执行模型

```
Grid (gridDim.x, gridDim.y)
├── Block (0,0)   Block (1,0)   Block (2,0)
├── Block (0,1)   Block (1,1)   Block (2,1)
└── ...
    └── Thread (threadIdx.x, threadIdx.y)
```

- **Grid**：所有 block 的集合
- **Block**：独立执行的线程组，内含 shared memory
- **Thread**：最小执行单位

### 1.2 内存层次

```
全局内存 (Global Memory)     ~800 GB/s, ~400 cycles
  ↓
共享内存 (Shared Memory)     ~19 TB/s, ~20 cycles    ← 手动管理
  ↓
寄存器 (Registers)           ~数千 TB/s, ~1 cycle     ← 编译器自动
```

优化核心：**最大化数据在 shared memory / 寄存器中的复用，减少全局内存访问。**

## 2. Naive 矩阵乘法

最直观的实现：每个线程计算 C 的一个元素。

```cpp
__global__ void matmul_naive_kernel(
    const float* A, const float* B, float* C, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}
```

### 性能瓶颈

```
计算 C[0][0] 需要 K 次全局内存读取：
  A[0][0], A[0][1], ..., A[0][K-1]    ← K 次读
  B[0][0], B[1][0], ..., B[K-1][0]    ← K 次读

相邻线程 C[0][0] 和 C[0][1] 都读 A 的第 0 行，但各自独立读取。
→ 全局内存带宽成为瓶颈，计算单元空闲等待数据。
```

## 3. Shared Memory Tiled 矩阵乘法

### 3.1 核心思想

将矩阵分块（tile），每块加载到 shared memory 后复用：

```
A 的一个 tile (BLOCK_M x BLOCK_K) 被 BLOCK_M x BLOCK_N 个线程共享
B 的一个 tile (BLOCK_K x BLOCK_N) 被 BLOCK_M x BLOCK_N 个线程共享

全局内存读取次数：
  naive:   2 * M * N * K
  tiled:   2 * (M/BLOCK_M) * (N/BLOCK_N) * K * (BLOCK_M + BLOCK_N)
  加速比:  ~BLOCK_M * BLOCK_N / (BLOCK_M + BLOCK_N) 倍
```

### 3.2 完整实现

```cpp
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ __launch_bounds__(1024)
void matmul_tiled_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N)
{
    // shared memory：编译期大小确定
    __shared__ float sA[BLOCK_M][BLOCK_K];
    __shared__ float sB[BLOCK_K][BLOCK_N];

    int row = blockIdx.y * BLOCK_M + threadIdx.y;
    int col = blockIdx.x * BLOCK_N + threadIdx.x;

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < (K + BLOCK_K - 1) / BLOCK_K; ++k_tile) {
        int k_start = k_tile * BLOCK_K;

        // 协作加载 sA：threadIdx.y 对应行，threadIdx.x 遍历 K 方向
        for (int kk = threadIdx.x; kk < BLOCK_K; kk += BLOCK_N) {
            int a_k = k_start + kk;
            sA[threadIdx.y][kk] = (row < M && a_k < K) ? A[row * K + a_k] : 0.0f;
        }

        // 协作加载 sB：threadIdx.x 对应列，threadIdx.y 遍历 K 方向
        for (int kk = threadIdx.y; kk < BLOCK_K; kk += BLOCK_M) {
            int b_k = k_start + kk;
            sB[kk][threadIdx.x] = (b_k < K && col < N) ? B[b_k * N + col] : 0.0f;
        }

        __syncthreads();  // 等待所有线程完成加载

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            acc += sA[threadIdx.y][kk] * sB[kk][threadIdx.x];
        }

        __syncthreads();  // 确保计算完成后再覆盖 shared memory
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
```

### 3.3 协作加载

当 `BLOCK_K > blockDim` 时，一个线程需要加载多个元素：

```
BLOCK_M=32, BLOCK_N=32, BLOCK_K=32 → blockDim=(32,32)=1024 threads
  sA 有 32*32=1024 个元素，1024 个线程每个加载 1 个 → 刚好
  
BLOCK_M=32, BLOCK_N=32, BLOCK_K=64 → sA 有 32*64=2048 个元素
  threadIdx.y=0 的线程加载 sA[0][0], sA[0][32]
  → for (kk = threadIdx.x; kk < BLOCK_K; kk += BLOCK_N) 循环 2 次
```

### 3.4 `__launch_bounds__` 与 `#pragma unroll`

```cpp
__global__ __launch_bounds__(1024)   // 告知编译器最大线程数，优化寄存器分配
void matmul_tiled_kernel(...) {
    #pragma unroll                    // 强制展开循环，消除循环开销
    for (int kk = 0; kk < BLOCK_K; ++kk) {
        acc += sA[threadIdx.y][kk] * sB[kk][threadIdx.x];
    }
}
```

## 4. 多配置 Dispatch

C++ 模板要求编译期常量，但不同矩阵尺寸需要不同 tile 大小。解决方案：**模板实例化 + runtime dispatch**。

```cpp
void launch_matmul_tiled_auto(const float* A, const float* B, float* C,
                              int M, int K, int N, cudaStream_t stream)
{
    int max_dim = max({M, N, K});

    if (max_dim >= 512) {
        // 大矩阵：WMMA FP16 Tensor Core
        matmul_wmma_kernel<<<...>>>(A, B, C, M, K, N);
    } else if (max_dim >= 128) {
        // 中等矩阵：32x32x32 分块
        matmul_tiled_kernel<32, 32, 32><<<...>>>(A, B, C, M, K, N);
    } else {
        // 小矩阵：16x16x16 分块
        matmul_tiled_kernel<16, 16, 16><<<...>>>(A, B, C, M, K, N);
    }
}
```

与 Triton `@triton.autotune` 的区别：CUDA 需要手动写 dispatch 逻辑，Triton 自动 benchmark 选择最优配置。

## 5. WMMA FP16 Tensor Core

### 5.1 Tensor Core 原理

```
WMMA (Warp Matrix Multiply-Accumulate):
  一个 warp (32 threads) 协作计算 16x16x16 矩阵乘法
  输入: FP16 (half)
  累加: FP32 (float)
  吞吐: 每时钟周期完成 16x16x16 = 4096 次 FLOP
```

### 5.2 实现

```cpp
__global__ __launch_bounds__(128)
void matmul_wmma_kernel(const float* A, const float* B, float* C, int M, int K, int N)
{
    constexpr int TILE = 32;   // 每 block 输出 32x32
    constexpr int R = 16;      // WMMA fragment 大小
    __shared__ half sA[TILE][R];  // FP16 shared memory
    __shared__ half sB[R][TILE];
    __shared__ float sC[TILE][TILE]; // FP32 累加器输出

    // 4 个 warp 覆盖 32x32 区域（每个 warp 负责 16x16）
    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id / 2;  // 0 or 1 (行方向)
    int warp_n = warp_id % 2;  // 0 or 1 (列方向)

    // FP32 累加器
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int kt = 0; kt < (K + R - 1) / R; ++kt) {
        // 1. 协作加载 FP32 → FP16 到 shared memory
        for (int i = threadIdx.x; i < TILE * R; i += blockDim.x) {
            int r = i / R, c = i % R;
            sA[r][c] = __float2half(A[global_row * K + global_col]);
        }
        // sB 同理...

        __syncthreads();

        // 2. 从 shared memory 加载到 WMMA fragment
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::load_matrix_sync(a_frag, &sA[warp_row][0], R);

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::load_matrix_sync(b_frag, &sB[0][warp_col], TILE);

        // 3. Tensor Core 矩阵乘法
        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    // 4. 输出 FP32 结果
    nvcuda::wmma::store_matrix_sync(&sC[warp_row][warp_col], acc, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();

    // 5. 从 shared memory 写回全局内存
    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        C[global_idx] = sC[r][c];
    }
}
```

### 5.3 精度损失与收益

| 方面 | FP32 Tiled | WMMA FP16 |
|------|------------|-----------|
| 精度 | 全精度 | 输入 FP16（~3位有效数字），累加 FP32 |
| 速度 | ~1 TFLOPS | ~20 TFLOPS (Ampere) |
| 适用 | 任何场景 | 矩阵 ≥512，可容忍轻微精度损失 |

本项目策略：`max_dim >= 512` 时使用 WMMA，否则使用 FP32 tiled。

### 5.4 TransB/TransA 的 WMMA 变体

反向传播需要 `C = A @ B^T` 和 `C = A^T @ B`。通过在 shared memory 中转置存储来避免显式转置：

```cpp
// C = A @ B^T: sBT 存 B 的转置
sBT[n_local][k_local] = B[k * N + n]  // 原始 B 是按行存储，直接按列读取

// C = A^T @ B: sAT 存 A 的转置
sAT[k_local][m_local] = A[m * K + k]  // 原始 A 是按行存储，直接按列读取
```

## 6. Activation Kernel 与向量化

### 6.1 基本 Activation Kernel

```cpp
__device__ inline float gelu_device(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
    float tanh_inner = tanhf(inner);  // 使用 CUDA 内置 tanhf
    return 0.5f * x * (1.0f + tanh_inner);
}

__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = gelu_device(input[idx]);
}
```

### 6.2 float4 向量化

一个线程处理 4 个元素，通过 `float4` 类型一次读取/写入 128 位：

```cpp
__global__ void gelu_backward_vec4_kernel(
    const float* grad_output, const float* input, float* grad_input, int n)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        // 一次读取 4 个 float（128 位，单次内存事务）
        float4 go = *reinterpret_cast<const float4*>(grad_output + idx);
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = go.x * gelu_backward_device(in.x);
        out.y = go.y * gelu_backward_device(in.y);
        out.z = go.z * gelu_backward_device(in.z);
        out.w = go.w * gelu_backward_device(in.w);
        *reinterpret_cast<float4*>(grad_input + idx) = out;
    } else {
        // 尾部不足 4 个元素时逐个处理
        for (int i = 0; i < 4 && idx + i < n; i++)
            grad_input[idx + i] = grad_output[idx + i] * gelu_backward_device(input[idx + i]);
    }
}
```

向量化收益：
- 全局内存事务从 4 次减为 1 次（128 位对齐）
- 指令级并行更好（4 个独立计算）
- Grid 大小减为 1/4，减少调度开销

## 7. LayerNorm: Warp Shuffle 归约

### 7.1 Warp Shuffle 原理

```cpp
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

`__shfl_down_sync`：warp 内线程直接交换寄存器值，无需 shared memory。16→8→4→2→1，log2(32)=5 步完成 32 线程归约。

### 7.2 多 warp 归约

```
Warp 0: warp_reduce_sum → s_block[0]
Warp 1: warp_reduce_sum → s_block[1]
...
__syncthreads()
Warp 0: 读 s_block[0..n_warps-1] → 再次 warp_reduce_sum → 最终结果
```

```cpp
__global__ void layernorm_forward_kernel(
    const float* X, float* Y, const float* Gamma, const float* Beta,
    float* Mean, float* Rstd, int N, float eps)
{
    int row = blockIdx.x;  // 一个 block 处理一行
    const float* x_row = X + row * N;

    __shared__ float s_block[32];  // 最多 32 个 warp 的部分和
    __shared__ float s_mean, s_rstd;

    // 第一遍：求和 → mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sum += x_row[i];
    sum = warp_reduce_sum(sum);
    if (lane == 0) s_block[warp_id] = sum;
    __syncthreads();
    // Warp 0 归约 s_block → s_mean
    ...

    // 第二遍：variance → rstd（同上模式）
    ...

    // 第三遍：normalize + affine
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x_hat = (x_row[i] - mean) * rstd;
        y_row[i] = Gamma[i] * x_hat + Beta[i];
    }
}
```

### 7.3 Backward 的 atomicAdd

```cpp
// 多个 block（行）累加到同一 d_gamma/d_beta 数组
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float x_hat = (x_row[i] - mean) * rstd;
    float dg = dy_row[i] * Gamma[i];
    dx_row[i] = rstd * (dg - c1 - x_hat * c2);
    atomicAdd(&DGamma[i], dy_row[i] * x_hat);
    atomicAdd(&DBeta[i], dy_row[i]);
}
```

## 8. 融合 Kernel

### matmul + bias + GELU 融合

```cpp
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void mlp_fused_first_layer_kernel(
    const float* X, const float* W1, const float* bias, float* H,
    int M, int K, int N)
{
    __shared__ float sX[BLOCK_M][BLOCK_K];
    __shared__ float sW[BLOCK_K][BLOCK_N];

    float acc = 0.0f;

    // matmul 循环（同标准 tiled matmul）
    for (int k_tile = 0; ...) {
        // 加载 sX, sW
        __syncthreads();
        for (int kk = 0; kk < BLOCK_K; ++kk)
            acc += sX[threadIdx.y][kk] * sW[kk][threadIdx.x];
        __syncthreads();
    }

    // 在寄存器中完成 bias add + GELU
    if (row < M && col < N) {
        H[row * N + col] = gelu_device(acc + bias[col]);
    }
}
```

关键：`acc` 是寄存器变量，`bias[col]` 从全局内存读一次，`gelu_device()` 在寄存器中计算。相比分离实现减少了 1 次全局内存写入 + 1 次读取。

## 9. PyTorch C++ Extension 绑定

### 9.1 binding.cpp 结构

```cpp
#include <torch/extension.h>

// Forward 声明 CUDA launch 函数
void launch_matmul_tiled(const float* A, const float* B, float* C,
                         int M, int K, int N, int BM, int BN, int BK,
                         cudaStream_t stream);

// Python 包装函数
torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B,
                           int BLOCK_M, int BLOCK_N, int BLOCK_K)
{
    CHECK_CUDA(A); CHECK_CONTIGUOUS(A); CHECK_FLOAT32(A);
    CHECK_CUDA(B); CHECK_CONTIGUOUS(B); CHECK_FLOAT32(B);

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    launch_matmul_tiled(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N, BLOCK_M, BLOCK_N, BLOCK_K,
        c10::cuda::getCurrentCUDAStream(A.device().index()).stream());
    return C;
}

// 注册模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_tiled", &matmul_tiled, "Tiled CUDA matmul");
}
```

### 9.2 输入校验宏

```cpp
#define CHECK_CUDA(x)       TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x)  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x)     TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
```

### 9.3 数据指针

```cpp
A.data_ptr<float>()    // 获取底层 float* 指针
```

PyTorch tensor 的 `data_ptr<T>()` 返回底层数据指针，可直接传给 CUDA kernel。

## 10. 编译与部署

### 10.1 setup.py

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mlp_cuda',
    ext_modules=[
        CUDAExtension(
            name='mlp_cuda',
            sources=['kernels/binding.cpp',
                     'kernels/mlp/matmul.cu', 'kernels/mlp/wmma.cu',
                     'kernels/mlp/activation.cu', 'kernels/mlp/fused.cu',
                     'kernels/mlp/layernorm.cu', 'kernels/mlp/softmax.cu',
                     'kernels/mlp/pool_im2col.cu'],
            include_dirs=['kernels/mlp'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math',
                         '-gencode=arch=compute_86,code=sm_86',     # Ampere
                         '-gencode=arch=compute_120,code=sm_120'],   # Blackwell
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
```

### 10.2 `-gencode` 多架构支持

```
-gencode=arch=compute_86,code=sm_86      → RTX 3070 (Ampere)
-gencode=arch=compute_120,code=sm_120     → RTX 5070 Ti (Blackwell)
```

编译器为每个 `-gencode` 生成一份 PTX，driver 在 runtime 选择匹配的版本。`compute_XX` 是 PTX 虚拟架构，`sm_XX` 是真实架构。

### 10.3 编译命令

```bash
pip install -e .              # 开发模式（可编辑）
python setup.py install       # 标准安装
```

编译产物是 `mlp_cuda.cpython-3xx-linux-x86_64.so`，Python 中 `import mlp_cuda` 即可使用。
