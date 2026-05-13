#include <cstdio>
#include <cuda_runtime.h>

// Day 4: shared memory tiled 矩阵乘法
// C[M,N] = A[M,K] @ B[K,N]
//
// 核心思想:
//   将 A 和 B 按 tile 分块加载到 shared memory
//   多个 thread 复用同一 tile 的数据, 减少 global memory 访问
//
// BLOCK_M x BLOCK_N: 输出 tile 大小
// BLOCK_K: 每次 K 维度迭代的 tile 大小

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // TODO: 实现 tiled matmul
    //
    // 提示:
    //   __shared__ float sA[BLOCK_M][BLOCK_K];
    //   __shared__ float sB[BLOCK_K][BLOCK_N];
    //
    //   int row = blockIdx.y * BLOCK_M + threadIdx.y;  // thread 对应的输出行
    //   int col = blockIdx.x * BLOCK_N + threadIdx.x;  // thread 对应的输出列
    //
    //   float acc = 0.0f;
    //   for (int k_tile = 0; k_tile < (K + BLOCK_K - 1) / BLOCK_K; k_tile++) {
    //       // 1. 从 global memory 加载 tile 到 shared memory (注意越界检查)
    //       // 2. __syncthreads()
    //       // 3. 计算 partial sum: acc += sA[ty][kk] * sB[kk][tx]
    //       // 4. __syncthreads()
    //   }
    //   if (row < M && col < N) C[row * N + col] = acc;
}

// 编译: nvcc -o matmul_tiled matmul_tiled.cu
int main() {
    int M = 512, K = 768, N = 3072;

    // TODO: 同 matmul_naive, 分配/初始化/启动/验证/计时
    //   启动参数:
    //     dim3 block(BLOCK_N, BLOCK_M);  // 注意 x=N 方向, y=M 方向
    //     dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    //
    // 建议测试 BLOCK 配置: 16x16x16, 32x32x16, 32x32x32

    printf("matmul_tiled: M=%d K=%d N=%d\n", M, K, N);
    return 0;
}
