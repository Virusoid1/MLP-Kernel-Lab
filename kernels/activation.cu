#include <cmath>
#include <cuda_runtime.h>

// Day 5: activation 函数的 CUDA 实现

// GELU 近似 (tanh 版本)
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ float gelu_device(float x) {
    // TODO: 实现 GELU activation
    return 0.0f;
}

// SiLU / Swish: x * sigmoid(x)
__device__ float silu_device(float x) {
    // TODO: 实现 SiLU activation
    return 0.0f;
}

// elementwise GELU kernel
__global__ void gelu_kernel(const float* input, float* output, int n) {
    // TODO: 对每个元素应用 GELU
}

// elementwise SiLU kernel
__global__ void silu_kernel(const float* input, float* output, int n) {
    // TODO: 对每个元素应用 SiLU
}

// fused SwiGLU kernel: output = SiLU(gate) * up
// gate 和 up 是两个同样 shape 的 tensor
__global__ void swiglu_kernel(
    const float* gate, const float* up, float* output, int n
) {
    // TODO: 实现 fused SwiGLU
    // output[i] = silu(gate[i]) * up[i]
}
