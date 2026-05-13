#include <cstdio>
#include <cuda_runtime.h>

// Day 1: vector add kernel
// 理解 grid/block/thread 组织和 global memory 访问
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    // TODO: 实现向量加法
    // 提示:
    //   int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //   if (idx < n) c[idx] = a[idx] + b[idx];
}

// CUDA event 计时工具函数
// 用法:
//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);
//   cudaEventRecord(start);
//   // ... kernel launch ...
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop);
//   float ms;
//   cudaEventElapsedTime(&ms, start, stop);
float cuda_event_elapsed(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// 编译: nvcc -o vector_add vector_add.cu
int main() {
    int n = 1024 * 1024;
    size_t bytes = n * sizeof(float);

    // TODO: 分配 host 内存, 初始化数据
    // TODO: 分配 device 内存, 拷贝数据
    // TODO: 计算 grid/block 大小, 启动 kernel
    // TODO: 拷贝结果回 host
    // TODO: correctness check (比较 c[i] 与 a[i]+b[i])
    // TODO: 用 CUDA event 计时并打印

    printf("vector_add: n=%d\n", n);
    // printf("max error: %.6f\n", max_err);
    // printf("latency: %.3f ms\n", elapsed_ms);

    return 0;
}
