#include <cuda_runtime.h>

// Day 11: fused SwiGLU kernel
// 用于 Llama/Qwen 系列 FFN:
//   up   = X @ W_up     [M, N]
//   gate = X @ W_gate   [M, N]
//   hidden = SiLU(gate) * up   [M, N]   <-- 这个 kernel
//   out = hidden @ W_down       [M, K_down]
//
// 这个算子是 elementwise 的，相对简单但很贴近 LLM

__global__ void swiglu_fused_kernel(
    const float* __restrict__ gate,  // [M, N]
    const float* __restrict__ up,    // [M, N]
    float* __restrict__ output,      // [M, N]
    int total_elements
) {
    // TODO: 实现 fused SwiGLU
    // 提示:
    //   int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //   if (idx < total_elements) {
    //       float g = gate[idx];
    //       float u = up[idx];
    //       float sigmoid_g = 1.0f / (1.0f + expf(-g));
    //       output[idx] = g * sigmoid_g * u;
    //   }
}

// FP16 版本 (Day 10 扩展)
__global__ void swiglu_fused_half_kernel(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ output,
    int total_elements
) {
    // TODO: 实现 FP16 SwiGLU
    // 用 __half2 可以一次处理两个元素
}
