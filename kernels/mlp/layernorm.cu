/*
 * LayerNorm forward + backward
 *
 * 实现策略:
 *   - 每个 block 处理 X 的一行 (1 block <-> 1 sample)
 *   - block 内 blockDim.x 个线程协作做 warp_reduce + shared 二阶 reduce
 *   - backward 中 d_gamma / d_beta 用 atomic_add 累加到 (N,) 输出
 *
 * 拆分自原 mlp_cuda_kernels.cu。共用 warp_reduce_sum 来自 device_utils.cuh。
 */

#include <cuda_runtime.h>
#include <cmath>
#include "device_utils.cuh"

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
