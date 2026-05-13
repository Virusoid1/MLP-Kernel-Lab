// CUDA kernel launch wrappers
// 被 binding.cpp 调用, 内部启动实际 kernel
//
// 每个 launch_* 函数:
//   1. 计算 grid/block 大小
//   2. 启动 kernel
//   不负责内存分配

#include <cuda_runtime.h>
#include <cstdio>

// === matmul_naive ===
__global__ void matmul_naive_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N)
{
    // TODO: 与 matmul_naive.cu 中相同
}

void launch_matmul_naive(
    const float* A, const float* B, float* C,
    int M, int K, int N, cudaStream_t stream)
{
    // TODO: 计算 grid/block 并启动
    // dim3 block(16, 16);
    // dim3 grid(...);
    // matmul_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}

// === matmul_tiled ===
template<int BM, int BN, int BK>
__global__ void matmul_tiled_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N)
{
    // TODO: 与 matmul_tiled.cu 中相同
}

void launch_matmul_tiled(
    const float* A, const float* B, float* C,
    int M, int K, int N, int BLOCK_M, int BLOCK_N, int BLOCK_K,
    cudaStream_t stream)
{
    // TODO: 根据 BLOCK 参数选择模板实例化并启动
    // 建议先用固定 BLOCK=16x16x16 实例化
}

// === mlp_fused_first_layer ===
__global__ void mlp_fused_first_layer_kernel(
    const float* __restrict__ X, const float* __restrict__ W1,
    const float* __restrict__ bias, float* __restrict__ H,
    int M, int K, int N)
{
    // TODO: fused matmul + bias + GELU
}

void launch_mlp_fused_first_layer(
    const float* X, const float* W1, const float* bias, float* H,
    int M, int K, int N, cudaStream_t stream)
{
    // TODO: 计算 grid/block 并启动
}

// === swiglu_fused ===
__global__ void swiglu_fused_kernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ output, int total_elements)
{
    // TODO: fused SwiGLU
}

void launch_swiglu_fused(
    const float* gate, const float* up, float* output,
    int total_elements, cudaStream_t stream)
{
    // TODO: 计算 grid/block 并启动
}
