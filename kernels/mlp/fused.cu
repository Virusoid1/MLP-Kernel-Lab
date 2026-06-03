/*
 * Fused MLP kernels:
 *   - mlp_fused_first_layer: H = GELU(X @ W1 + bias)
 *   - swiglu_fused:          output = SiLU(gate) * up
 *
 * 拆分自原 mlp_cuda_kernels.cu。
 */

#include <cuda_runtime.h>
#include "device_utils.cuh"

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
        // FIX (P0): 原版用 block(64,64)=4096 threads 超出 CUDA 1024 上限,
        // 在 fused_mlp_first_layer (M,K,N)=(512,768,512) 时必然 cudaErrorInvalidValue.
        // 退到 32x32 tile + block(32,32)=1024 threads (与 matmul 修复同模式).
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        mlp_fused_first_layer_kernel<32, 32, 32><<<grid, block, 0, stream>>>(
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
