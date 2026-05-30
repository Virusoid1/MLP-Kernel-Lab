/*
 * MLP CUDA Kernel 集合
 *
 * 包含:
 *   - matmul (naive + tiled with shared memory)
 *   - activation (GELU/ReLU/SiLU forward + backward)
 *   - bias_add
 *   - fused MLP first layer (matmul + bias + GELU)
 *   - fused SwiGLU
 *
 * 针对 RTX 5070 Ti (Blackwell SM 12.0) 和 RTX 3070 Laptop (Ampere SM 8.6) 优化:
 *   - 多 tile 配置模板 dispatch
 *   - shared memory tiling
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cstdio>

// WMMA kernel 前向声明（定义在文件末尾，避免 nvcc 解析器冲突）
__global__ void matmul_wmma_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N);
__global__ void matmul_wmma_transB_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K);
__global__ void matmul_wmma_transA_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N);
__global__ void matmul_wmma64_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N);
__global__ void matmul_wmma64_transB_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K);
__global__ void matmul_wmma64_transA_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N);


// ============================================================
// Device 工具函数
// ============================================================

__device__ inline float gelu_device(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
    float tanh_inner = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_inner);
}

__device__ inline float gelu_backward_device(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float u = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
    float tanh_u = tanhf(u);
    float sech2_u = 1.0f - tanh_u * tanh_u;
    float du_dx = sqrt_2_over_pi * (1.0f + 0.134145f * x * x);
    return 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2_u * du_dx;
}

__device__ inline float silu_device(float x) {
    return __fdividef(x, 1.0f + expf(-x));
}

__device__ inline float silu_backward_device(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

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
        // 大 tile: FP32 shared memory tiled（Blackwell 新增）
        dim3 block(64, 64);
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        matmul_tiled_kernel<64, 64, 32><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
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
// Activation kernels
// ============================================================

// --- GELU ---
__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = gelu_device(input[idx]);
}
void launch_gelu(const float* input, float* output, int n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    gelu_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n);
}

// --- ReLU ---
__global__ void relu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = fmaxf(0.0f, input[idx]);
}
void launch_relu(const float* input, float* output, int n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    relu_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n);
}

// --- SiLU ---
__global__ void silu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = silu_device(input[idx]);
}
void launch_silu(const float* input, float* output, int n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    silu_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n);
}

// --- GELU backward ---
__global__ void gelu_backward_kernel(
    const float* grad_output, const float* input, float* grad_input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad_input[idx] = grad_output[idx] * gelu_backward_device(input[idx]);
}
void launch_gelu_backward(
    const float* grad_output, const float* input, float* grad_input,
    int n, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    gelu_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        grad_output, input, grad_input, n);
}

// --- ReLU backward ---
__global__ void relu_backward_kernel(
    const float* grad_output, const float* input, float* grad_input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
}
void launch_relu_backward(
    const float* grad_output, const float* input, float* grad_input,
    int n, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    relu_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        grad_output, input, grad_input, n);
}

// --- SiLU backward ---
__global__ void silu_backward_kernel(
    const float* grad_output, const float* input, float* grad_input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad_input[idx] = grad_output[idx] * silu_backward_device(input[idx]);
}
void launch_silu_backward(
    const float* grad_output, const float* input, float* grad_input,
    int n, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    silu_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        grad_output, input, grad_input, n);
}

// ============================================================
// mlp_fused_first_layer (fused matmul + bias + GELU)
// ============================================================

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ __launch_bounds__(1024)
void mlp_fused_first_layer_kernel(
    const float* __restrict__ X,
    const float* __restrict__ W1,
    const float* __restrict__ bias,
    float* __restrict__ H,
    int M, int K, int N)
{
    __shared__ float sX[BLOCK_M][BLOCK_K];
    __shared__ float sW[BLOCK_K][BLOCK_N];

    int row = blockIdx.y * BLOCK_M + threadIdx.y;
    int col = blockIdx.x * BLOCK_N + threadIdx.x;

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < (K + BLOCK_K - 1) / BLOCK_K; ++k_tile) {
        int k_start = k_tile * BLOCK_K;

        // 加载 sX (X 的 tile)
        for (int kk = threadIdx.x; kk < BLOCK_K; kk += BLOCK_N) {
            int x_k = k_start + kk;
            sX[threadIdx.y][kk] = (row < M && x_k < K) ? X[row * K + x_k] : 0.0f;
        }

        // 加载 sW (W1 的 tile)
        for (int kk = threadIdx.y; kk < BLOCK_K; kk += BLOCK_M) {
            int w_k = k_start + kk;
            sW[kk][threadIdx.x] = (w_k < K && col < N) ? W1[w_k * N + col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            acc += sX[threadIdx.y][kk] * sW[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        H[row * N + col] = gelu_device(acc + bias[col]);
    }
}

void launch_mlp_fused_first_layer(
    const float* X, const float* W1, const float* bias, float* H,
    int M, int K, int N, cudaStream_t stream)
{
    int max_dim = M > N ? M : N;
    max_dim = max_dim > K ? max_dim : K;

    if (max_dim >= 512) {
        dim3 block(64, 64);
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        mlp_fused_first_layer_kernel<64, 64, 32><<<grid, block, 0, stream>>>(
            X, W1, bias, H, M, K, N);
    } else if (max_dim >= 128) {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        mlp_fused_first_layer_kernel<32, 32, 32><<<grid, block, 0, stream>>>(
            X, W1, bias, H, M, K, N);
    } else {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        mlp_fused_first_layer_kernel<16, 16, 16><<<grid, block, 0, stream>>>(
            X, W1, bias, H, M, K, N);
    }
}

// ============================================================
// swiglu_fused
// ============================================================

__global__ void swiglu_fused_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float g = gate[idx];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[idx] = g * sigmoid_g * up[idx];
    }
}

void launch_swiglu_fused(
    const float* gate, const float* up, float* output,
    int total_elements, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    swiglu_fused_kernel<<<grid_size, block_size, 0, stream>>>(
        gate, up, output, total_elements);
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

// ============================================================
// activation backward (float4 向量化)
// ============================================================

__global__ void gelu_backward_vec4_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input, int n)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 go = *reinterpret_cast<const float4*>(grad_output + idx);
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = go.x * gelu_backward_device(in.x);
        out.y = go.y * gelu_backward_device(in.y);
        out.z = go.z * gelu_backward_device(in.z);
        out.w = go.w * gelu_backward_device(in.w);
        *reinterpret_cast<float4*>(grad_input + idx) = out;
    } else {
        for (int i = 0; i < 4 && idx + i < n; i++)
            grad_input[idx + i] = grad_output[idx + i] * gelu_backward_device(input[idx + i]);
    }
}

void launch_gelu_backward_vec4(
    const float* grad_output, const float* input, float* grad_input,
    int n, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (n + block_size * 4 - 1) / (block_size * 4);
    gelu_backward_vec4_kernel<<<grid_size, block_size, 0, stream>>>(
        grad_output, input, grad_input, n);
}

__global__ void relu_backward_vec4_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input, int n)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 go = *reinterpret_cast<const float4*>(grad_output + idx);
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = (in.x > 0.0f) ? go.x : 0.0f;
        out.y = (in.y > 0.0f) ? go.y : 0.0f;
        out.z = (in.z > 0.0f) ? go.z : 0.0f;
        out.w = (in.w > 0.0f) ? go.w : 0.0f;
        *reinterpret_cast<float4*>(grad_input + idx) = out;
    } else {
        for (int i = 0; i < 4 && idx + i < n; i++)
            grad_input[idx + i] = (input[idx + i] > 0.0f) ? grad_output[idx + i] : 0.0f;
    }
}

void launch_relu_backward_vec4(
    const float* grad_output, const float* input, float* grad_input,
    int n, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (n + block_size * 4 - 1) / (block_size * 4);
    relu_backward_vec4_kernel<<<grid_size, block_size, 0, stream>>>(
        grad_output, input, grad_input, n);
}

__global__ void silu_backward_vec4_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input, int n)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 go = *reinterpret_cast<const float4*>(grad_output + idx);
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = go.x * silu_backward_device(in.x);
        out.y = go.y * silu_backward_device(in.y);
        out.z = go.z * silu_backward_device(in.z);
        out.w = go.w * silu_backward_device(in.w);
        *reinterpret_cast<float4*>(grad_input + idx) = out;
    } else {
        for (int i = 0; i < 4 && idx + i < n; i++)
            grad_input[idx + i] = grad_output[idx + i] * silu_backward_device(input[idx + i]);
    }
}

void launch_silu_backward_vec4(
    const float* grad_output, const float* input, float* grad_input,
    int n, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (n + block_size * 4 - 1) / (block_size * 4);
    silu_backward_vec4_kernel<<<grid_size, block_size, 0, stream>>>(
        grad_output, input, grad_input, n);
}
// ============================================================
// LayerNorm forward + backward
// ============================================================

// --- warp reduce sum ---
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// --- forward: y = gamma * (x - mean) / sqrt(var + eps) + beta ---
// 一个 block 处理一行，blockDim.x 个线程协作
__global__ void layernorm_forward_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
    const float* __restrict__ Gamma,
    const float* __restrict__ Beta,
    float* __restrict__ Mean,
    float* __restrict__ Rstd,
    int N, float eps)
{
    int row = blockIdx.x;
    const float* x_row = X + row * N;
    float* y_row = Y + row * N;

    __shared__ float s_block[32];
    __shared__ float s_mean, s_rstd;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int n_warps = (blockDim.x + 31) / 32;

    // 第一遍：求和 → mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sum += x_row[i];
    sum = warp_reduce_sum(sum);
    if (lane == 0) s_block[warp_id] = sum;
    __syncthreads();
    sum = (threadIdx.x < n_warps) ? s_block[lane] : 0.0f;
    if (warp_id == 0) sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) s_mean = sum / N;
    __syncthreads();
    float mean = s_mean;

    // 第二遍：variance → rstd
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float d = x_row[i] - mean;
        var_sum += d * d;
    }
    var_sum = warp_reduce_sum(var_sum);
    if (lane == 0) s_block[warp_id] = var_sum;
    __syncthreads();
    var_sum = (threadIdx.x < n_warps) ? s_block[lane] : 0.0f;
    if (warp_id == 0) var_sum = warp_reduce_sum(var_sum);
    if (threadIdx.x == 0) s_rstd = 1.0f / sqrtf(var_sum / N + eps);
    __syncthreads();
    float rstd = s_rstd;

    // 第三遍：normalize + affine
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x_hat = (x_row[i] - mean) * rstd;
        y_row[i] = Gamma[i] * x_hat + Beta[i];
    }

    if (threadIdx.x == 0) {
        Mean[row] = mean;
        Rstd[row] = rstd;
    }
}

void launch_layernorm_forward(
    const float* X, float* Y,
    const float* Gamma, const float* Beta,
    float* Mean, float* Rstd,
    int B, int N, float eps, cudaStream_t stream)
{
    int block_size = (N + 31) / 32 * 32;
    if (block_size > 1024) block_size = 1024;
    layernorm_forward_kernel<<<B, block_size, 0, stream>>>(
        X, Y, Gamma, Beta, Mean, Rstd, N, eps);
}

// --- backward ---
__global__ void layernorm_backward_kernel(
    const float* __restrict__ DY,
    const float* __restrict__ X,
    const float* __restrict__ Gamma,
    const float* __restrict__ Mean,
    const float* __restrict__ Rstd,
    float* __restrict__ DX,
    float* __restrict__ DGamma,
    float* __restrict__ DBeta,
    int N)
{
    int row = blockIdx.x;
    const float* dy_row = DY + row * N;
    const float* x_row = X + row * N;
    float* dx_row = DX + row * N;
    float mean = Mean[row];
    float rstd = Rstd[row];

    __shared__ float s_c1_buf[32], s_c2_buf[32];
    __shared__ float s_c1, s_c2;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int n_warps = (blockDim.x + 31) / 32;

    // 第一遍：c1 = sum(dy*gamma) / N, c2 = sum(dy*gamma*x_hat) / N
    float c1_partial = 0.0f, c2_partial = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x_hat = (x_row[i] - mean) * rstd;
        float dg = dy_row[i] * Gamma[i];
        c1_partial += dg;
        c2_partial += dg * x_hat;
    }
    c1_partial = warp_reduce_sum(c1_partial);
    c2_partial = warp_reduce_sum(c2_partial);

    if (lane == 0) { s_c1_buf[warp_id] = c1_partial; s_c2_buf[warp_id] = c2_partial; }
    __syncthreads();
    c1_partial = (threadIdx.x < n_warps) ? s_c1_buf[lane] : 0.0f;
    c2_partial = (threadIdx.x < n_warps) ? s_c2_buf[lane] : 0.0f;
    if (warp_id == 0) {
        c1_partial = warp_reduce_sum(c1_partial);
        c2_partial = warp_reduce_sum(c2_partial);
    }
    if (threadIdx.x == 0) { s_c1 = c1_partial / N; s_c2 = c2_partial / N; }
    __syncthreads();
    float c1 = s_c1;
    float c2 = s_c2;

    // 第二遍：d_x + atomic_add d_gamma, d_beta
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x_hat = (x_row[i] - mean) * rstd;
        float dg = dy_row[i] * Gamma[i];
        dx_row[i] = rstd * (dg - c1 - x_hat * c2);
        atomicAdd(&DGamma[i], dy_row[i] * x_hat);
        atomicAdd(&DBeta[i], dy_row[i]);
    }
}

void launch_layernorm_backward(
    const float* DY, const float* X,
    const float* Gamma, const float* Mean, const float* Rstd,
    float* DX, float* DGamma, float* DBeta,
    int B, int N, cudaStream_t stream)
{
    int block_size = (N + 31) / 32 * 32;
    if (block_size > 1024) block_size = 1024;
    layernorm_backward_kernel<<<B, block_size, 0, stream>>>(
        DY, X, Gamma, Mean, Rstd, DX, DGamma, DBeta, N);
}

// ============================================================
// Softmax (逐行, 数值稳定)
// ============================================================

__global__ void softmax_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int M, int N)
{
    int row = blockIdx.x;
    if (row >= M) return;

    const float* x_row = input + row * N;
    float* y_row = output + row * N;

    int n_warps = (blockDim.x + 31) / 32;
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // 第一遍：找行最大值
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        max_val = fmaxf(max_val, x_row[i]);

    // warp reduce max
    for (int offset = 16; offset > 0; offset /= 2)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));

    // block reduce max: warp 结果写入 shared，再 reduce
    extern __shared__ float s_mem[];
    float* s_max_arr = s_mem;
    float* s_sum_arr = s_mem + n_warps;

    if (lane == 0) s_max_arr[wid] = max_val;
    __syncthreads();

    max_val = (threadIdx.x < n_warps) ? s_max_arr[threadIdx.x] : -FLT_MAX;
    for (int offset = n_warps / 2; offset > 0; offset /= 2)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    if (threadIdx.x == 0) s_max_arr[0] = max_val;
    __syncthreads();
    max_val = s_max_arr[0];

    // 第二遍：exp + sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sum += expf(x_row[i] - max_val);

    // warp reduce sum
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // block reduce sum
    if (lane == 0) s_sum_arr[wid] = sum;
    __syncthreads();

    sum = (threadIdx.x < n_warps) ? s_sum_arr[threadIdx.x] : 0.0f;
    for (int offset = n_warps / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) s_sum_arr[0] = sum;
    __syncthreads();
    float inv_sum = 1.0f / s_sum_arr[0];

    // 第三遍：写出
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        y_row[i] = expf(x_row[i] - max_val) * inv_sum;
}

void launch_softmax(
    const float* input, float* output,
    int M, int N, cudaStream_t stream)
{
    int block_size = (N + 31) / 32 * 32;
    if (block_size > 1024) block_size = 1024;
    int n_warps = (block_size + 31) / 32;
    int smem = n_warps * 2 * sizeof(float);
    softmax_kernel<<<M, block_size, smem, stream>>>(input, output, M, N);
}

// ============================================================
// MaxPool2D (NCHW)
// ============================================================

__global__ void maxpool2d_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int nc = idx / (H_out * W_out);
    int c = nc % C;
    int n = nc / C;

    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    float max_val = -FLT_MAX;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h_in = h_start + kh;
        if (h_in < 0 || h_in >= H) continue;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w_in = w_start + kw;
            if (w_in < 0 || w_in >= W) continue;
            float val = input[((n * C + c) * H + h_in) * W + w_in];
            max_val = fmaxf(max_val, val);
        }
    }
    output[((n * C + c) * H_out + h_out) * W_out + w_out] = max_val;
}

void launch_maxpool2d(
    const float* input, float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    cudaStream_t stream)
{
    int total = N * C * H_out * W_out;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    maxpool2d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W, H_out, W_out, kernel_size, stride, padding);
}

// ============================================================
// AvgPool2D (NCHW)
// ============================================================

__global__ void avgpool2d_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int nc = idx / (H_out * W_out);
    int c = nc % C;
    int n = nc / C;

    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    float acc = 0.0f;
    int count = 0;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h_in = h_start + kh;
        if (h_in < 0 || h_in >= H) continue;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w_in = w_start + kw;
            if (w_in < 0 || w_in >= W) continue;
            acc += input[((n * C + c) * H + h_in) * W + w_in];
            ++count;
        }
    }
    output[((n * C + c) * H_out + h_out) * W_out + w_out] = acc / count;
}

void launch_avgpool2d(
    const float* input, float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    cudaStream_t stream)
{
    int total = N * C * H_out * W_out;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    avgpool2d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W, H_out, W_out, kernel_size, stride, padding);
}

// ============================================================
// im2col (Conv2D 前置展开)
// ============================================================

__global__ void im2col_kernel(
    const float* __restrict__ input, float* __restrict__ col,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = N * H_out * W_out;
    int col_len = C * KH * KW;
    if (idx >= total_rows * col_len) return;

    int col_offset = idx % col_len;
    int row_idx = idx / col_len;

    int w_out = row_idx % W_out;
    int h_out = (row_idx / W_out) % H_out;
    int n = row_idx / (H_out * W_out);

    int kw = col_offset % KW;
    int kh = (col_offset / KW) % KH;
    int c = col_offset / (KH * KW);

    int h_in = h_out * stride_h - pad_h + kh;
    int w_in = w_out * stride_w - pad_w + kw;

    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        col[idx] = input[((n * C + c) * H + h_in) * W + w_in];
    } else {
        col[idx] = 0.0f;
    }
}

void launch_im2col(
    const float* input, float* col,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW,
    int stride, int padding,
    cudaStream_t stream)
{
    int total = N * H_out * W_out * C * KH * KW;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    im2col_kernel<<<grid_size, block_size, 0, stream>>>(
        input, col, N, C, H, W, H_out, W_out, KH, KW, stride, stride, padding, padding);
}

// ============================================================
// WMMA FP16 Tensor Core kernels (SM 8.6+)
// 每个 warp (32 threads) 协作计算 16x16 输出 tile
// Block 包含多个 warp，覆盖 TILE x TILE 输出区域
// ============================================================

// C = A @ B, A:(M,K) B:(K,N) C:(M,N)
__global__ __launch_bounds__(128)
void matmul_wmma_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N)
{
    constexpr int TILE = 32;
    constexpr int R = 16;
    __shared__ half sA[TILE][R];
    __shared__ half sB[R][TILE];
    __shared__ float sC[TILE][TILE];

    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id / (TILE / 16);
    int warp_n = warp_id % (TILE / 16);
    int warp_row = warp_m * 16;
    int warp_col = warp_n * 16;

    int block_row = blockIdx.y * TILE;
    int block_col = blockIdx.x * TILE;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int kt = 0; kt < (K + R - 1) / R; ++kt) {
        int k_start = kt * R;

        // 协作加载 sA: A[block_row+r][k_start+c] -> FP16
        for (int i = threadIdx.x; i < TILE * R; i += blockDim.x) {
            int r = i / R, c = i % R;
            int gr = block_row + r, gc = k_start + c;
            sA[r][c] = (gr < M && gc < K)
                        ? __float2half(A[gr * K + gc]) : __float2half(0.0f);
        }

        // 协作加载 sB: B[k_start+r][block_col+c] -> FP16
        for (int i = threadIdx.x; i < R * TILE; i += blockDim.x) {
            int r = i / TILE, c = i % TILE;
            int gr = k_start + r, gc = block_col + c;
            sB[r][c] = (gr < K && gc < N)
                        ? __float2half(B[gr * N + gc]) : __float2half(0.0f);
        }

        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::load_matrix_sync(a_frag, &sA[warp_row][0], R);
        nvcuda::wmma::load_matrix_sync(b_frag, &sB[0][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(&sC[warp_row][warp_col], acc, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        int r = i / TILE, c = i % TILE;
        int gr = block_row + r, gc = block_col + c;
        if (gr < M && gc < N)
            C[gr * N + gc] = sC[r][c];
    }
}

// C = A @ B^T, A:(M,N) B:(K,N) C:(M,K)
// sBT 存 B 的转置: sBT[n_local][k_local] = B[k][n]
__global__ __launch_bounds__(128)
void matmul_wmma_transB_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE = 32;
    constexpr int R = 16;
    __shared__ half sA[TILE][R];
    __shared__ half sBT[R][TILE];
    __shared__ float sC[TILE][TILE];

    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id / (TILE / 16);
    int warp_k = warp_id % (TILE / 16);
    int warp_row = warp_m * 16;
    int warp_col = warp_k * 16;

    int block_row = blockIdx.y * TILE;  // M 方向
    int block_col = blockIdx.x * TILE;  // K 方向

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int nt = 0; nt < (N + R - 1) / R; ++nt) {
        int n_start = nt * R;

        // sA: A[block_row+r][n_start+c]
        for (int i = threadIdx.x; i < TILE * R; i += blockDim.x) {
            int r = i / R, c = i % R;
            int gr = block_row + r, gc = n_start + c;
            sA[r][c] = (gr < M && gc < N)
                        ? __float2half(A[gr * N + gc]) : __float2half(0.0f);
        }

        // sBT[n_local][k_local] = B[(block_col+k_local)*N + (n_start+n_local)]
        for (int i = threadIdx.x; i < R * TILE; i += blockDim.x) {
            int nl = i / TILE, kl = i % TILE;
            int b_k = block_col + kl, b_n = n_start + nl;
            sBT[nl][kl] = (b_k < K && b_n < N)
                           ? __float2half(B[b_k * N + b_n]) : __float2half(0.0f);
        }

        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::load_matrix_sync(a_frag, &sA[warp_row][0], R);
        nvcuda::wmma::load_matrix_sync(b_frag, &sBT[0][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(&sC[warp_row][warp_col], acc, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        int r = i / TILE, c = i % TILE;
        int gr = block_row + r, gc = block_col + c;
        if (gr < M && gc < K)
            C[gr * K + gc] = sC[r][c];
    }
}

// C = A^T @ B, A:(M,K) B:(M,N) C:(K,N)
// sAT 存 A 的转置: sAT[k_local][m_local] = A[m][k]
__global__ __launch_bounds__(128)
void matmul_wmma_transA_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N)
{
    constexpr int TILE = 32;
    constexpr int R = 16;
    __shared__ half sAT[TILE][R];
    __shared__ half sB[R][TILE];
    __shared__ float sC[TILE][TILE];

    int warp_id = threadIdx.x / 32;
    int warp_k = warp_id / (TILE / 16);
    int warp_n = warp_id % (TILE / 16);
    int warp_row = warp_k * 16;
    int warp_col = warp_n * 16;

    int block_row = blockIdx.y * TILE;  // K 方向
    int block_col = blockIdx.x * TILE;  // N 方向

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int mt = 0; mt < (M + R - 1) / R; ++mt) {
        int m_start = mt * R;

        // sAT[k_local][m_local] = A[(m_start+m_local)*K + (block_row+k_local)]
        for (int i = threadIdx.x; i < TILE * R; i += blockDim.x) {
            int kl = i / R, ml = i % R;
            int a_m = m_start + ml, a_k = block_row + kl;
            sAT[kl][ml] = (a_m < M && a_k < K)
                           ? __float2half(A[a_m * K + a_k]) : __float2half(0.0f);
        }

        // sB: B[m_start+r][block_col+c]
        for (int i = threadIdx.x; i < R * TILE; i += blockDim.x) {
            int r = i / TILE, c = i % TILE;
            int gr = m_start + r, gc = block_col + c;
            sB[r][c] = (gr < M && gc < N)
                        ? __float2half(B[gr * N + gc]) : __float2half(0.0f);
        }

        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::load_matrix_sync(a_frag, &sAT[warp_row][0], R);
        nvcuda::wmma::load_matrix_sync(b_frag, &sB[0][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(&sC[warp_row][warp_col], acc, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        int r = i / TILE, c = i % TILE;
        int gr = block_row + r, gc = block_col + c;
        if (gr < K && gc < N)
            C[gr * N + gc] = sC[r][c];
    }
}

// ============================================================
// WMMA64 FP16 Tensor Core kernels (SM 8.0+, 大 tile 64x64)
// 每个 warp (32 threads) 协作计算 16x16 输出 tile
// Block 包含 8 warp (256 threads)，覆盖 64x64 输出区域
// R=32: shared memory 中 K 方向的内积步长
// ============================================================

// C = A @ B, A:(M,K) B:(K,N) C:(M,N)
__global__ __launch_bounds__(256)
void matmul_wmma64_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N)
{
    constexpr int TILE = 64;
    constexpr int R = 32;
    __shared__ half sA[TILE][R];
    __shared__ half sB[R][TILE];
    __shared__ float sC[TILE][TILE];

    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id / (TILE / 16);  // 0..3
    int warp_n = warp_id % (TILE / 16);  // 0..3
    int warp_row = warp_m * 16;
    int warp_col = warp_n * 16;

    int block_row = blockIdx.y * TILE;
    int block_col = blockIdx.x * TILE;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int kt = 0; kt < (K + R - 1) / R; ++kt) {
        int k_start = kt * R;

        for (int i = threadIdx.x; i < TILE * R; i += blockDim.x) {
            int r = i / R, c = i % R;
            int gr = block_row + r, gc = k_start + c;
            sA[r][c] = (gr < M && gc < K)
                        ? __float2half(A[gr * K + gc]) : __float2half(0.0f);
        }

        for (int i = threadIdx.x; i < R * TILE; i += blockDim.x) {
            int r = i / TILE, c = i % TILE;
            int gr = k_start + r, gc = block_col + c;
            sB[r][c] = (gr < K && gc < N)
                        ? __float2half(B[gr * N + gc]) : __float2half(0.0f);
        }

        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::load_matrix_sync(a_frag, &sA[warp_row][0], R);
        nvcuda::wmma::load_matrix_sync(b_frag, &sB[0][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(&sC[warp_row][warp_col], acc, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        int r = i / TILE, c = i % TILE;
        int gr = block_row + r, gc = block_col + c;
        if (gr < M && gc < N)
            C[gr * N + gc] = sC[r][c];
    }
}

// C = A @ B^T, A:(M,N) B:(K,N) C:(M,K)
__global__ __launch_bounds__(256)
void matmul_wmma64_transB_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE = 64;
    constexpr int R = 32;
    __shared__ half sA[TILE][R];
    __shared__ half sBT[R][TILE];
    __shared__ float sC[TILE][TILE];

    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id / (TILE / 16);
    int warp_k = warp_id % (TILE / 16);
    int warp_row = warp_m * 16;
    int warp_col = warp_k * 16;

    int block_row = blockIdx.y * TILE;  // M 方向
    int block_col = blockIdx.x * TILE;  // K 方向

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int nt = 0; nt < (N + R - 1) / R; ++nt) {
        int n_start = nt * R;

        for (int i = threadIdx.x; i < TILE * R; i += blockDim.x) {
            int r = i / R, c = i % R;
            int gr = block_row + r, gc = n_start + c;
            sA[r][c] = (gr < M && gc < N)
                        ? __float2half(A[gr * N + gc]) : __float2half(0.0f);
        }

        for (int i = threadIdx.x; i < R * TILE; i += blockDim.x) {
            int nl = i / TILE, kl = i % TILE;
            int b_k = block_col + kl, b_n = n_start + nl;
            sBT[nl][kl] = (b_k < K && b_n < N)
                           ? __float2half(B[b_k * N + b_n]) : __float2half(0.0f);
        }

        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::load_matrix_sync(a_frag, &sA[warp_row][0], R);
        nvcuda::wmma::load_matrix_sync(b_frag, &sBT[0][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(&sC[warp_row][warp_col], acc, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        int r = i / TILE, c = i % TILE;
        int gr = block_row + r, gc = block_col + c;
        if (gr < M && gc < K)
            C[gr * K + gc] = sC[r][c];
    }
}

// C = A^T @ B, A:(M,K) B:(M,N) C:(K,N)
__global__ __launch_bounds__(256)
void matmul_wmma64_transA_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N)
{
    constexpr int TILE = 64;
    constexpr int R = 32;
    __shared__ half sAT[TILE][R];
    __shared__ half sB[R][TILE];
    __shared__ float sC[TILE][TILE];

    int warp_id = threadIdx.x / 32;
    int warp_k = warp_id / (TILE / 16);
    int warp_n = warp_id % (TILE / 16);
    int warp_row = warp_k * 16;
    int warp_col = warp_n * 16;

    int block_row = blockIdx.y * TILE;  // K 方向
    int block_col = blockIdx.x * TILE;  // N 方向

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int mt = 0; mt < (M + R - 1) / R; ++mt) {
        int m_start = mt * R;

        for (int i = threadIdx.x; i < TILE * R; i += blockDim.x) {
            int kl = i / R, ml = i % R;
            int a_m = m_start + ml, a_k = block_row + kl;
            sAT[kl][ml] = (a_m < M && a_k < K)
                           ? __float2half(A[a_m * K + a_k]) : __float2half(0.0f);
        }

        for (int i = threadIdx.x; i < R * TILE; i += blockDim.x) {
            int r = i / TILE, c = i % TILE;
            int gr = m_start + r, gc = block_col + c;
            sB[r][c] = (gr < M && gc < N)
                        ? __float2half(B[gr * N + gc]) : __float2half(0.0f);
        }

        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::load_matrix_sync(a_frag, &sAT[warp_row][0], R);
        nvcuda::wmma::load_matrix_sync(b_frag, &sB[0][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(&sC[warp_row][warp_col], acc, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        int r = i / TILE, c = i % TILE;
        int gr = block_row + r, gc = block_col + c;
        if (gr < K && gc < N)
            C[gr * N + gc] = sC[r][c];
    }
}
