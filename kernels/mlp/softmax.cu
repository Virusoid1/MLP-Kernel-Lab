/*
 * Softmax (逐行, 数值稳定)
 *
 * 一个 block 处理一行；动态共享内存 = n_warps * 2 floats，
 * 前半保存每个 warp 的 max,后半保存每个 warp 的 sum。
 *
 * 拆分自原 mlp_cuda_kernels.cu。
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>
#include <type_traits>

// 标量读写：内部一律 fp32 计算；fp16 走 upcast/downcast（与 matmul_half 数值约定一致）
template <typename T>
__device__ __forceinline__ float softmax_to_f(T v) {
    if constexpr (std::is_same_v<T, half>) return __half2float(v);
    else return v;
}
template <typename T>
__device__ __forceinline__ T softmax_from_f(float v) {
    if constexpr (std::is_same_v<T, half>) return __float2half(v);
    else return v;
}

template <typename T>
__global__ void softmax_kernel(
    const T* __restrict__ input, T* __restrict__ output,
    int M, int N)
{
    int row = blockIdx.x;
    if (row >= M) return;

    const T* x_row = input + row * N;
    T* y_row = output + row * N;

    int n_warps = (blockDim.x + 31) / 32;
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // 第一遍：找行最大值
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        max_val = fmaxf(max_val, softmax_to_f<T>(x_row[i]));

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
        sum += expf(softmax_to_f<T>(x_row[i]) - max_val);

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
        y_row[i] = softmax_from_f<T>(expf(softmax_to_f<T>(x_row[i]) - max_val) * inv_sum);
}

void launch_softmax(
    const float* input, float* output,
    int M, int N, cudaStream_t stream)
{
    int block_size = (N + 31) / 32 * 32;
    if (block_size > 1024) block_size = 1024;
    int n_warps = (block_size + 31) / 32;
    int smem = n_warps * 2 * sizeof(float);
    softmax_kernel<float><<<M, block_size, smem, stream>>>(input, output, M, N);
}

void launch_softmax_half(
    const half* input, half* output,
    int M, int N, cudaStream_t stream)
{
    int block_size = (N + 31) / 32 * 32;
    if (block_size > 1024) block_size = 1024;
    int n_warps = (block_size + 31) / 32;
    int smem = n_warps * 2 * sizeof(float);
    softmax_kernel<half><<<M, block_size, smem, stream>>>(input, output, M, N);
}
