// matmul_half: fp16 输入直通 WMMA TensorCore（fp32 累加，half 输出）
// 语义 = fp16 TensorCore 标准路径（fp16 in → fp32 acc → fp16 out）。
// 由 binding.matmul_half 暴露，用于 cuda fp16 block（v2 四后端 fp16 补全）。

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

__global__ __launch_bounds__(128)
void matmul_half_kernel(
    const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
    int M, int K, int N)
{
    constexpr int TILE = 32;
    constexpr int R = 16;
    __shared__ half sA[TILE][R];
    __shared__ half sB[R][TILE];
    __shared__ float sC[TILE][TILE];

    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id / (TILE / 16);
    int warp_n = warp_id % (TILE / 16);
    int warp_row = warp_m * 16;
    int warp_col = warp_n * 16;

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

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::load_matrix_sync(a_frag, &sA[warp_row][0], R);
        nvcuda::wmma::load_matrix_sync(b_frag, &sB[0][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
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
