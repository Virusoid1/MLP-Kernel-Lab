/**
 * 带 NVTX 标记的示例 CUDA kernel，用于演示 Nsight profiling 工作流。
 * 编译：nvcc -O2 -arch=sm_86 -lnvw -o example_kernel example_kernel.cu
 * 运行：./example_kernel [repeat_count]
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

// ---------- 工具宏 ----------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------- 向量加法 kernel ----------
__global__ void vec_add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

// ---------- 简单矩阵乘法（朴素实现，方便看到瓶颈） ----------
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ---------- 矩阵乘法（ tiled 版本，用于对比优化效果） ----------
#define TILE_SIZE 16
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ---------- 初始化数据 ----------
void init_data(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main(int argc, char** argv) {
    int repeat = (argc > 1) ? atoi(argv[1]) : 100;

    srand(42);

    // ---- 向量加法参数 ----
    int vec_n = 1 << 24;  // 16M 元素
    size_t vec_bytes = vec_n * sizeof(float);

    // ---- 矩阵乘法参数 ----
    int M = 1024, N = 1024, K = 1024;

    // ---- 分配内存 ----
    float *d_a, *d_b, *d_c, *d_A, *d_B, *d_C;
    float *h_a = (float*)malloc(vec_bytes);
    float *h_b = (float*)malloc(vec_bytes);
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));

    init_data(h_a, vec_n);
    init_data(h_b, vec_n);
    init_data(h_A, M * K);
    init_data(h_B, K * N);

    CUDA_CHECK(cudaMalloc(&d_a, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // ---- H2D 传输 ----
    nvtxRangePush("H2D Transfer");
    CUDA_CHECK(cudaMemcpy(d_a, h_a, vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    nvtxRangePop();

    // ---- warmup ----
    nvtxRangePush("Warmup");
    for (int i = 0; i < 5; i++) {
        vec_add_kernel<<<(vec_n + 255) / 256, 256>>>(d_a, d_b, d_c, vec_n);
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    // ---- 向量加法 benchmark ----
    nvtxRangePushA("VecAdd Benchmark");
    for (int i = 0; i < repeat; i++) {
        // 每个 iteration 也可以单独标记
        char name[64];
        snprintf(name, sizeof(name), "VecAdd iter %d", i);
        nvtxRangePushA(name);
        vec_add_kernel<<<(vec_n + 255) / 256, 256>>>(d_a, d_b, d_c, vec_n);
        nvtxRangePop();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    // ---- 朴素矩阵乘法 benchmark ----
    nvtxRangePushA("MatMul Naive Benchmark");
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    for (int i = 0; i < repeat; i++) {
        matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    // ---- Tiled 矩阵乘法 benchmark ----
    nvtxRangePushA("MatMul Tiled Benchmark");
    for (int i = 0; i < repeat; i++) {
        matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    // ---- D2H 传输 ----
    nvtxRangePush("D2H Transfer");
    float* h_c = (float*)malloc(vec_bytes);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, vec_bytes, cudaMemcpyDeviceToHost));
    nvtxRangePop();

    // ---- 计算有效带宽和 GFLOPS ----
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float ms;

    // 向量加法带宽
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        vec_add_kernel<<<(vec_n + 255) / 256, 256>>>(d_a, d_b, d_c, vec_n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float vec_bandwidth = (3.0f * vec_bytes * repeat) / (ms * 1e-3) / 1e9;

    // 朴素 matmul GFLOPS
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float naive_gflops = (2.0f * M * N * K * repeat) / (ms * 1e-3) / 1e9;

    // tiled matmul GFLOPS
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float tiled_gflops = (2.0f * M * N * K * repeat) / (ms * 1e-3) / 1e9;

    printf("=== Performance Summary ===\n");
    printf("VecAdd bandwidth:    %.2f GB/s\n", vec_bandwidth);
    printf("MatMul Naive:        %.2f GFLOPS\n", naive_gflops);
    printf("MatMul Tiled:        %.2f GFLOPS\n", tiled_gflops);
    printf("Tiled speedup:       %.2fx\n", tiled_gflops / naive_gflops);

    // ---- 清理 ----
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_a); free(h_b); free(h_c);
    free(h_A); free(h_B);

    return 0;
}
