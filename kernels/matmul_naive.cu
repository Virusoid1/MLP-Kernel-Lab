#include <cstdio>
#include <cuda_runtime.h>

// Day 2: naive 矩阵乘法
// C[M,N] = A[M,K] @ B[K,N]
// A row-major, B row-major, C row-major
// 每个 thread 计算 C 中的一个元素
__global__ void matmul_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // TODO: 实现朴素矩阵乘法
    // 提示:
    //   int row = blockIdx.y * blockDim.y + threadIdx.y;
    //   int col = blockIdx.x * blockDim.x + threadIdx.x;
    //   if (row < M && col < N) {
    //       float acc = 0.0f;
    //       for (int k = 0; k < K; k++) {
    //           acc += A[row * K + k] * B[k * N + col];
    //       }
    //       C[row * N + col] = acc;
    //   }
}

// 编译: nvcc -o matmul_naive matmul_naive.cu
int main() {
    // 建议测试 shape (LLM-like):
    //   M=128, K=4096, N=11008  (Llama FFN up-proj)
    //   M=512, K=768,  N=3072   (BERT FFN)
    int M = 512, K = 768, N = 3072;

    // TODO: 分配 host/device 内存
    // TODO: 初始化 A, B (随机数)
    // TODO: 启动 kernel
    //   dim3 block(16, 16);
    //   dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    //   matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    // TODO: correctness check (与 CPU 或 PyTorch 结果比较)
    // TODO: CUDA event 计时
    // TODO: 计算 TFLOPS = 2 * M * N * K / (time_s * 1e12)

    printf("matmul_naive: M=%d K=%d N=%d\n", M, K, N);
    return 0;
}
