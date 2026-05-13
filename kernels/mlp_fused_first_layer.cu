#include <cstdio>
#include <cuda_runtime.h>

// Day 5: fused MLP first layer
// H = GELU(X @ W1 + bias)
//
// 对比两种实现:
//   unfused: matmul -> bias add -> GELU (三步，两次 global memory 读写)
//   fused:   matmul 输出时直接加 bias 并应用 GELU (一次 global memory 写)

// unfused 版本: 复用 matmul_tiled_kernel，然后单独 bias + GELU
// 不需要单独写，benchmark 时组合调用即可

// fused 版本: tiled matmul + bias + GELU 在同一个 kernel 中
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void mlp_fused_first_layer_kernel(
    const float* __restrict__ X,    // [M, K]
    const float* __restrict__ W1,   // [K, N]
    const float* __restrict__ bias, // [N]
    float* __restrict__ H,          // [M, N]
    int M, int K, int N
) {
    // TODO: 实现 fused matmul + bias + GELU
    //
    // 提示: 基于 matmul_tiled_kernel
    //   1. 同样的 tiled matmul 逻辑, 计算出 acc
    //   2. 写入前: acc += bias[col]
    //   3. 写入前: acc = gelu(acc)
    //   4. H[row * N + col] = acc
}
