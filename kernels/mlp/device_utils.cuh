/*
 * MLP CUDA kernel 公共 device 函数
 * 拆分自原 mlp_cuda_kernels.cu，多个 .cu 文件可共享同一份 inline 实现。
 */

#pragma once

#include <cuda_runtime.h>
#include <cmath>

// --- GELU (tanh 近似，与 Triton kernel 保持一致) ---
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

// --- SiLU ---
__device__ inline float silu_device(float x) {
    return __fdividef(x, 1.0f + expf(-x));
}

__device__ inline float silu_backward_device(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

// --- warp 内 reduce sum (LayerNorm / Softmax 共用) ---
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
