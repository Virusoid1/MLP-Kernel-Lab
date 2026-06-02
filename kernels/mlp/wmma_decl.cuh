/*
 * WMMA kernel 前向声明
 *
 * WMMA kernel 定义在 wmma.cu，由 matmul.cu 的 launch_matmul_tiled_auto /
 * launch_matmul_transA / launch_matmul_transB 通过 <<<>>> 启动语法引用。
 * nvcc 单文件编译 + 链接时通过本头文件解析符号。
 */

#pragma once

#include <cuda_runtime.h>

// C = A @ B, 32x32 tile
__global__ void matmul_wmma_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N);

// C = A @ B^T, 32x32 tile
__global__ void matmul_wmma_transB_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K);

// C = A^T @ B, 32x32 tile
__global__ void matmul_wmma_transA_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N);

// C = A @ B, 64x64 tile, 8 warp
__global__ void matmul_wmma64_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N);

// C = A @ B^T, 64x64 tile
__global__ void matmul_wmma64_transB_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K);

// C = A^T @ B, 64x64 tile
__global__ void matmul_wmma64_transA_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N);
