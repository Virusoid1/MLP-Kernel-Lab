/*
 * matmul kernels: naive + shared-memory tiled + transA + transB + bias_add
 * 拆分自原 mlp_cuda_kernels.cu。
 *
 * 大尺寸（max_dim >= 512/1024）通过本文件 dispatch 到 wmma.cu 中的
 * WMMA FP16 Tensor Core kernel；小尺寸退回 shared-memory tiled FP32。
 */

#include <cuda_runtime.h>
#include "wmma_decl.cuh"

// ============================================================
// matmul_naive
// ============================================================

__global__ void matmul_naive_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N)
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

void launch_matmul_naive(
    const float* A, const float* B, float* C,
    int M, int K, int N, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}

// ============================================================
// matmul_tiled (shared-memory tiled, 多配置 dispatch)
// ============================================================

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ __launch_bounds__(1024)
void matmul_tiled_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N)
{
    __shared__ float sA[BLOCK_M][BLOCK_K];
    __shared__ float sB[BLOCK_K][BLOCK_N];

    int row = blockIdx.y * BLOCK_M + threadIdx.y;
    int col = blockIdx.x * BLOCK_N + threadIdx.x;

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < (K + BLOCK_K - 1) / BLOCK_K; ++k_tile) {
        int k_start = k_tile * BLOCK_K;

        // 加载 sA: threadIdx.y → M 方向, threadIdx.x 作 K 方向偏移
        for (int kk = threadIdx.x; kk < BLOCK_K; kk += BLOCK_N) {
            int a_k = k_start + kk;
            sA[threadIdx.y][kk] = (row < M && a_k < K) ? A[row * K + a_k] : 0.0f;
        }

        // 加载 sB: threadIdx.x → N 方向, threadIdx.y 作 K 方向偏移
        for (int kk = threadIdx.y; kk < BLOCK_K; kk += BLOCK_M) {
            int b_k = k_start + kk;
            sB[kk][threadIdx.x] = (b_k < K && col < N) ? B[b_k * N + col] : 0.0f;
        }

        __syncthreads();

        // 计算 partial sum
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            acc += sA[threadIdx.y][kk] * sB[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

void launch_matmul_tiled(
    const float* A, const float* B, float* C,
    int M, int K, int N, int BLOCK_M, int BLOCK_N, int BLOCK_K,
    cudaStream_t stream)
{
    // 根据 BLOCK 参数 dispatch 到对应模板实例化
    if (BLOCK_M == 16 && BLOCK_N == 16 && BLOCK_K == 16) {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        matmul_tiled_kernel<16, 16, 16><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else if (BLOCK_M == 32 && BLOCK_N == 32 && BLOCK_K == 16) {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmul_tiled_kernel<32, 32, 16><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else if (BLOCK_M == 32 && BLOCK_N == 32 && BLOCK_K == 32) {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmul_tiled_kernel<32, 32, 32><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else if (BLOCK_M == 16 && BLOCK_N == 16 && BLOCK_K == 32) {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        matmul_tiled_kernel<16, 16, 32><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else if (BLOCK_M == 32 && BLOCK_N == 16 && BLOCK_K == 16) {
        dim3 block(16, 32);
        dim3 grid((N + 15) / 16, (M + 31) / 32);
        matmul_tiled_kernel<32, 16, 16><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else if (BLOCK_M == 16 && BLOCK_N == 32 && BLOCK_K == 16) {
        dim3 block(32, 16);
        dim3 grid((N + 31) / 32, (M + 15) / 16);
        matmul_tiled_kernel<16, 32, 16><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else if (BLOCK_M == 32 && BLOCK_N == 32 && BLOCK_K == 64) {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmul_tiled_kernel<32, 32, 64><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else if (BLOCK_M == 64 && BLOCK_N == 64 && BLOCK_K == 32) {
        dim3 block(64, 64);
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        matmul_tiled_kernel<64, 64, 32><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else if (BLOCK_M == 64 && BLOCK_N == 64 && BLOCK_K == 64) {
        dim3 block(64, 64);
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        matmul_tiled_kernel<64, 64, 64><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else if (BLOCK_M == 128 && BLOCK_N == 64 && BLOCK_K == 32) {
        dim3 block(64, 128);
        dim3 grid((N + 63) / 64, (M + 127) / 128);
        matmul_tiled_kernel<128, 64, 32><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else {
        // fallback: 最小安全配置
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        matmul_tiled_kernel<16, 16, 16><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    }
}


// 自适应 dispatch：按矩阵尺寸自动选择最优 tile
void launch_matmul_tiled_auto(
    const float* A, const float* B, float* C,
    int M, int K, int N, cudaStream_t stream)
{
    int max_dim = M > N ? M : N;
    max_dim = max_dim > K ? max_dim : K;

    if (max_dim >= 1024) {
        // WMMA64 FP16 Tensor Core: 每个 block 64x64 输出，8 warp
        constexpr int TILE = 64;
        dim3 block(TILE / 16 * TILE / 16 * 32);  // 256 threads
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        matmul_wmma64_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else if (max_dim >= 512) {
        // WMMA32 FP16 Tensor Core: 每个 block 32x32 输出，4 warp
        constexpr int TILE = 32;
        dim3 block(TILE / 16 * TILE / 16 * 32);  // 128 threads
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        matmul_wmma_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else if (max_dim >= 256) {
        // FIX: 原版用 block(64,64)=4096 threads 超出 CUDA 1024 上限,
        // 在 max_dim ∈ [256,511] 必然 cudaErrorInvalidValue。
        // 退回与 [128,255] 同一安全配置: 32x32 tile, block(32,32)=1024 threads.
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmul_tiled_kernel<32, 32, 32><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else if (max_dim >= 128) {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmul_tiled_kernel<32, 32, 32><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        matmul_tiled_kernel<16, 16, 16><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    }
}

// ============================================================
// bias_add
// ============================================================

__global__ void bias_add_kernel(
    const float* __restrict__ input, const float* __restrict__ bias,
    float* __restrict__ output, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        output[row * N + col] = input[row * N + col] + bias[col];
    }
}

void launch_bias_add(
    const float* input, const float* bias, float* output,
    int M, int N, cudaStream_t stream)
{
    dim3 block(32, 16);
    dim3 grid((N + 31) / 32, (M + 15) / 16);
    bias_add_kernel<<<grid, block, 0, stream>>>(input, bias, output, M, N);
}

// ============================================================
// matmul_transB: C = A @ B^T，B 不需要显式转置
// A: (M, N) B: (K, N) C: (M, K)
// C[m][k] = sum_n A[m][n] * B[k][n]
// ============================================================

template<int BLOCK_M, int BLOCK_K, int BLOCK_N>
__global__ __launch_bounds__(1024)
void matmul_transB_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float sA[BLOCK_M][BLOCK_N];
    __shared__ float sB[BLOCK_K][BLOCK_N];

    int row = blockIdx.y * BLOCK_M + threadIdx.y;
    int col = blockIdx.x * BLOCK_K + threadIdx.x;

    float acc = 0.0f;

    for (int n_tile = 0; n_tile < (N + BLOCK_N - 1) / BLOCK_N; ++n_tile) {
        int n_start = n_tile * BLOCK_N;

        for (int nn = threadIdx.x; nn < BLOCK_N; nn += BLOCK_K) {
            int a_n = n_start + nn;
            sA[threadIdx.y][nn] = (row < M && a_n < N) ? A[row * N + a_n] : 0.0f;
        }

        for (int nn = threadIdx.y; nn < BLOCK_N; nn += BLOCK_M) {
            int b_n = n_start + nn;
            sB[threadIdx.x][nn] = (col < K && b_n < N) ? B[col * N + b_n] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int nn = 0; nn < BLOCK_N; ++nn) {
            acc += sA[threadIdx.y][nn] * sB[threadIdx.x][nn];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = acc;
    }
}

void launch_matmul_transB(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream)
{
    int max_dim = M > K ? M : K;
    max_dim = max_dim > N ? max_dim : N;

    if (max_dim >= 1024) {
        constexpr int TILE = 64;
        dim3 block(TILE / 16 * TILE / 16 * 32);
        dim3 grid((K + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        matmul_wmma64_transB_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    } else if (max_dim >= 512) {
        constexpr int TILE = 32;
        dim3 block(TILE / 16 * TILE / 16 * 32);
        dim3 grid((K + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        matmul_wmma_transB_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    } else {
        dim3 block(16, 16);
        dim3 grid((K + 15) / 16, (M + 15) / 16);
        void* args[] = {(void*)&A, (void*)&B, (void*)&C, (void*)&M, (void*)&N, (void*)&K};
        cudaLaunchKernel((const void*)matmul_transB_kernel<16, 16, 16>, grid, block, args, 0, stream);
    }
}

// ============================================================
// matmul_transA: C = A^T @ B，A 不需要显式转置
// A: (M, K) B: (M, N) C: (K, N)
// C[k][n] = sum_m A[m][k] * B[m][n]
// ============================================================

template<int BLOCK_K, int BLOCK_N, int BLOCK_M>
__global__ __launch_bounds__(1024)
void matmul_transA_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N)
{
    __shared__ float sA[BLOCK_K][BLOCK_M];
    __shared__ float sB[BLOCK_M][BLOCK_N];

    int row = blockIdx.y * BLOCK_K + threadIdx.y;
    int col = blockIdx.x * BLOCK_N + threadIdx.x;

    float acc = 0.0f;

    for (int m_tile = 0; m_tile < (M + BLOCK_M - 1) / BLOCK_M; ++m_tile) {
        int m_start = m_tile * BLOCK_M;

        for (int mm = threadIdx.x; mm < BLOCK_M; mm += BLOCK_N) {
            int a_m = m_start + mm;
            sA[threadIdx.y][mm] = (row < K && a_m < M) ? A[a_m * K + row] : 0.0f;
        }

        for (int mm = threadIdx.y; mm < BLOCK_M; mm += BLOCK_K) {
            int b_m = m_start + mm;
            sB[mm][threadIdx.x] = (b_m < M && col < N) ? B[b_m * N + col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int mm = 0; mm < BLOCK_M; ++mm) {
            acc += sA[threadIdx.y][mm] * sB[mm][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < K && col < N) {
        C[row * N + col] = acc;
    }
}

void launch_matmul_transA(
    const float* A, const float* B, float* C,
    int M, int K, int N, cudaStream_t stream)
{
    int max_dim = K > N ? K : N;
    max_dim = max_dim > M ? max_dim : M;

    if (max_dim >= 1024) {
        constexpr int TILE = 64;
        dim3 block(TILE / 16 * TILE / 16 * 32);
        dim3 grid((N + TILE - 1) / TILE, (K + TILE - 1) / TILE);
        matmul_wmma64_transA_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else if (max_dim >= 512) {
        constexpr int TILE = 32;
        dim3 block(TILE / 16 * TILE / 16 * 32);
        dim3 grid((N + TILE - 1) / TILE, (K + TILE - 1) / TILE);
        matmul_wmma_transA_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    } else {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (K + 15) / 16);
        matmul_transA_kernel<16, 16, 16><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    }
}
