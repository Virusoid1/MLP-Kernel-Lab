// matmul_half: fp16 WMMA TensorCore（fp32 累加，half 输出）
// 语义 = fp16 TensorCore 标准路径（fp16 in → fp32 acc → fp16 out）。
// perf 调优 v2.6 结论（3070 实测对比见 artifacts_3070/perf_tuning_cuda.md）：
//   R16 基线 33.3ms; R32 dual-MMA 32.3ms（同步减半无显著收益）;
//   BM128/BN32 42.7ms（sC 24KB 占用塌陷）; T64 直接-store 不可行（store_matrix_sync 要求元素类型匹配）。
// 保留形态：32x32 tile + R32 dual-MMA + fp32 sC staging（正确且与基线同档）。

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

__global__ __launch_bounds__(128)
void matmul_half_kernel(
    const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
    int M, int K, int N)
{
    constexpr int TILE = 32;
    constexpr int R = 32;
    __shared__ half sA[TILE][R];
    __shared__ half sB[R][TILE];
    __shared__ float sC[TILE][TILE];

    int warp_id = threadIdx.x / 32;
    int warp_row = (warp_id / 2) * 16;   // 2x2 warp 排布
    int warp_col = (warp_id % 2) * 16;
    int block_row = blockIdx.y * TILE;
    int block_col = blockIdx.x * TILE;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int kt = 0; kt < (K + R - 1) / R; ++kt) {
        int k_start = kt * R;
        for (int i = threadIdx.x; i < TILE * R; i += blockDim.x) {
            int r = i / R, c = i % R;
            int gr = block_row + r, gc = k_start + c;
            sA[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < R * TILE; i += blockDim.x) {
            int r = i / TILE, c = i % TILE;
            int gr = k_start + r, gc = block_col + c;
            sB[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }
        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a0, a1;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b0, b1;
        nvcuda::wmma::load_matrix_sync(a0, &sA[warp_row][0],  R);
        nvcuda::wmma::load_matrix_sync(a1, &sA[warp_row][16], R);
        nvcuda::wmma::load_matrix_sync(b0, &sB[0][warp_col],  TILE);
        nvcuda::wmma::load_matrix_sync(b1, &sB[16][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc, a0, b0, acc);
        nvcuda::wmma::mma_sync(acc, a1, b1, acc);
        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(&sC[warp_row][warp_col], acc, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        int r = i / TILE, c = i % TILE;
        int gr = block_row + r, gc = block_col + c;
        if (gr < M && gc < N)
            C[gr * N + gc] = __float2half(sC[r][c]);
    }
}

void launch_matmul_half(
    const half* A, const half* B, half* C,
    int M, int K, int N, cudaStream_t stream)
{
    dim3 block(128);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    matmul_half_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}
