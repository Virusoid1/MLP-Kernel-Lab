/*
 * Elementwise activation kernels: GELU / ReLU / SiLU
 *   - forward
 *   - backward (标量版)
 *   - backward (float4 向量化版)
 *
 * 拆分自原 mlp_cuda_kernels.cu。device 实现复用 device_utils.cuh。
 */

#include <cuda_runtime.h>
#include "device_utils.cuh"

// ============================================================
// Activation forward
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

// ============================================================
// Activation backward (标量版)
// ============================================================

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
// Activation backward (float4 向量化版)
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
