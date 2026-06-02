/*
 * WMMA FP16 Tensor Core kernels
 * 拆分自原 mlp_cuda_kernels.cu，对应 32x32 / 64x64 两种 tile，
 * 含 normal / transB / transA 各 3 个，共 6 个 kernel。
 *
 * 需要 SM 7.5+ 支持 mma.sync 指令（RTX 20 系及以上）。
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// ============================================================
// WMMA FP16 Tensor Core kernels (SM 8.6+)
// 每个 warp (32 threads) 协作计算 16x16 输出 tile
// Block 包含多个 warp，覆盖 TILE x TILE 输出区域
// ============================================================

// C = A @ B, A:(M,K) B:(K,N) C:(M,N)
__global__ __launch_bounds__(128)
void matmul_wmma_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
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

        // 协作加载 sA: A[block_row+r][k_start+c] -> FP16
        for (int i = threadIdx.x; i < TILE * R; i += blockDim.x) {
            int r = i / R, c = i % R;
            int gr = block_row + r, gc = k_start + c;
            sA[r][c] = (gr < M && gc < K)
                        ? __float2half(A[gr * K + gc]) : __float2half(0.0f);
        }

        // 协作加载 sB: B[k_start+r][block_col+c] -> FP16
        for (int i = threadIdx.x; i < R * TILE; i += blockDim.x) {
            int r = i / TILE, c = i % TILE;
            int gr = k_start + r, gc = block_col + c;
            sB[r][c] = (gr < K && gc < N)
                        ? __float2half(B[gr * N + gc]) : __float2half(0.0f);
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
            C[gr * N + gc] = sC[r][c];
    }
}

// C = A @ B^T, A:(M,N) B:(K,N) C:(M,K)
// sBT 存 B 的转置: sBT[n_local][k_local] = B[k][n]
__global__ __launch_bounds__(128)
void matmul_wmma_transB_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE = 32;
    constexpr int R = 16;
    __shared__ half sA[TILE][R];
    __shared__ half sBT[R][TILE];
    __shared__ float sC[TILE][TILE];

    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id / (TILE / 16);
    int warp_k = warp_id % (TILE / 16);
    int warp_row = warp_m * 16;
    int warp_col = warp_k * 16;

    int block_row = blockIdx.y * TILE;  // M 方向
    int block_col = blockIdx.x * TILE;  // K 方向

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int nt = 0; nt < (N + R - 1) / R; ++nt) {
        int n_start = nt * R;

        // sA: A[block_row+r][n_start+c]
        for (int i = threadIdx.x; i < TILE * R; i += blockDim.x) {
            int r = i / R, c = i % R;
            int gr = block_row + r, gc = n_start + c;
            sA[r][c] = (gr < M && gc < N)
                        ? __float2half(A[gr * N + gc]) : __float2half(0.0f);
        }

        // sBT[n_local][k_local] = B[(block_col+k_local)*N + (n_start+n_local)]
        for (int i = threadIdx.x; i < R * TILE; i += blockDim.x) {
            int nl = i / TILE, kl = i % TILE;
            int b_k = block_col + kl, b_n = n_start + nl;
            sBT[nl][kl] = (b_k < K && b_n < N)
                           ? __float2half(B[b_k * N + b_n]) : __float2half(0.0f);
        }

        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::load_matrix_sync(a_frag, &sA[warp_row][0], R);
        nvcuda::wmma::load_matrix_sync(b_frag, &sBT[0][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(&sC[warp_row][warp_col], acc, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        int r = i / TILE, c = i % TILE;
        int gr = block_row + r, gc = block_col + c;
        if (gr < M && gc < K)
            C[gr * K + gc] = sC[r][c];
    }
}

// C = A^T @ B, A:(M,K) B:(M,N) C:(K,N)
// sAT 存 A 的转置: sAT[k_local][m_local] = A[m][k]
__global__ __launch_bounds__(128)
void matmul_wmma_transA_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N)
{
    constexpr int TILE = 32;
    constexpr int R = 16;
    __shared__ half sAT[TILE][R];
    __shared__ half sB[R][TILE];
    __shared__ float sC[TILE][TILE];

    int warp_id = threadIdx.x / 32;
    int warp_k = warp_id / (TILE / 16);
    int warp_n = warp_id % (TILE / 16);
    int warp_row = warp_k * 16;
    int warp_col = warp_n * 16;

    int block_row = blockIdx.y * TILE;  // K 方向
    int block_col = blockIdx.x * TILE;  // N 方向

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int mt = 0; mt < (M + R - 1) / R; ++mt) {
        int m_start = mt * R;

        // sAT[k_local][m_local] = A[(m_start+m_local)*K + (block_row+k_local)]
        for (int i = threadIdx.x; i < TILE * R; i += blockDim.x) {
            int kl = i / R, ml = i % R;
            int a_m = m_start + ml, a_k = block_row + kl;
            sAT[kl][ml] = (a_m < M && a_k < K)
                           ? __float2half(A[a_m * K + a_k]) : __float2half(0.0f);
        }

        // sB: B[m_start+r][block_col+c]
        for (int i = threadIdx.x; i < R * TILE; i += blockDim.x) {
            int r = i / TILE, c = i % TILE;
            int gr = m_start + r, gc = block_col + c;
            sB[r][c] = (gr < M && gc < N)
                        ? __float2half(B[gr * N + gc]) : __float2half(0.0f);
        }

        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::load_matrix_sync(a_frag, &sAT[warp_row][0], R);
        nvcuda::wmma::load_matrix_sync(b_frag, &sB[0][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(&sC[warp_row][warp_col], acc, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        int r = i / TILE, c = i % TILE;
        int gr = block_row + r, gc = block_col + c;
        if (gr < K && gc < N)
            C[gr * N + gc] = sC[r][c];
    }
}

// ============================================================
// WMMA64 FP16 Tensor Core kernels (SM 8.0+, 大 tile 64x64)
// 每个 warp (32 threads) 协作计算 16x16 输出 tile
// Block 包含 8 warp (256 threads)，覆盖 64x64 输出区域
// R=32: shared memory 中 K 方向的内积步长
// ============================================================

// C = A @ B, A:(M,K) B:(K,N) C:(M,N)
__global__ __launch_bounds__(512)
void matmul_wmma64_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N)
{
    constexpr int TILE = 64;
    constexpr int R = 32;
    __shared__ half sA[TILE][R];
    __shared__ half sB[R][TILE];
    __shared__ float sC[TILE][TILE];

    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id / (TILE / 16);  // 0..3
    int warp_n = warp_id % (TILE / 16);  // 0..3
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
            sA[r][c] = (gr < M && gc < K)
                        ? __float2half(A[gr * K + gc]) : __float2half(0.0f);
        }

        for (int i = threadIdx.x; i < R * TILE; i += blockDim.x) {
            int r = i / TILE, c = i % TILE;
            int gr = k_start + r, gc = block_col + c;
            sB[r][c] = (gr < K && gc < N)
                        ? __float2half(B[gr * N + gc]) : __float2half(0.0f);
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
            C[gr * N + gc] = sC[r][c];
    }
}

// C = A @ B^T, A:(M,N) B:(K,N) C:(M,K)
__global__ __launch_bounds__(512)
void matmul_wmma64_transB_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE = 64;
    constexpr int R = 32;
    __shared__ half sA[TILE][R];
    __shared__ half sBT[R][TILE];
    __shared__ float sC[TILE][TILE];

    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id / (TILE / 16);
    int warp_k = warp_id % (TILE / 16);
    int warp_row = warp_m * 16;
    int warp_col = warp_k * 16;

    int block_row = blockIdx.y * TILE;  // M 方向
    int block_col = blockIdx.x * TILE;  // K 方向

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int nt = 0; nt < (N + R - 1) / R; ++nt) {
        int n_start = nt * R;

        for (int i = threadIdx.x; i < TILE * R; i += blockDim.x) {
            int r = i / R, c = i % R;
            int gr = block_row + r, gc = n_start + c;
            sA[r][c] = (gr < M && gc < N)
                        ? __float2half(A[gr * N + gc]) : __float2half(0.0f);
        }

        for (int i = threadIdx.x; i < R * TILE; i += blockDim.x) {
            int nl = i / TILE, kl = i % TILE;
            int b_k = block_col + kl, b_n = n_start + nl;
            sBT[nl][kl] = (b_k < K && b_n < N)
                           ? __float2half(B[b_k * N + b_n]) : __float2half(0.0f);
        }

        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::load_matrix_sync(a_frag, &sA[warp_row][0], R);
        nvcuda::wmma::load_matrix_sync(b_frag, &sBT[0][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(&sC[warp_row][warp_col], acc, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        int r = i / TILE, c = i % TILE;
        int gr = block_row + r, gc = block_col + c;
        if (gr < M && gc < K)
            C[gr * K + gc] = sC[r][c];
    }
}

// C = A^T @ B, A:(M,K) B:(M,N) C:(K,N)
__global__ __launch_bounds__(512)
void matmul_wmma64_transA_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int K, int N)
{
    constexpr int TILE = 64;
    constexpr int R = 32;
    __shared__ half sAT[TILE][R];
    __shared__ half sB[R][TILE];
    __shared__ float sC[TILE][TILE];

    int warp_id = threadIdx.x / 32;
    int warp_k = warp_id / (TILE / 16);
    int warp_n = warp_id % (TILE / 16);
    int warp_row = warp_k * 16;
    int warp_col = warp_n * 16;

    int block_row = blockIdx.y * TILE;  // K 方向
    int block_col = blockIdx.x * TILE;  // N 方向

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int mt = 0; mt < (M + R - 1) / R; ++mt) {
        int m_start = mt * R;

        for (int i = threadIdx.x; i < TILE * R; i += blockDim.x) {
            int kl = i / R, ml = i % R;
            int a_m = m_start + ml, a_k = block_row + kl;
            sAT[kl][ml] = (a_m < M && a_k < K)
                           ? __float2half(A[a_m * K + a_k]) : __float2half(0.0f);
        }

        for (int i = threadIdx.x; i < R * TILE; i += blockDim.x) {
            int r = i / TILE, c = i % TILE;
            int gr = m_start + r, gc = block_col + c;
            sB[r][c] = (gr < M && gc < N)
                        ? __float2half(B[gr * N + gc]) : __float2half(0.0f);
        }

        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::load_matrix_sync(a_frag, &sAT[warp_row][0], R);
        nvcuda::wmma::load_matrix_sync(b_frag, &sB[0][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(&sC[warp_row][warp_col], acc, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        int r = i / TILE, c = i % TILE;
        int gr = block_row + r, gc = block_col + c;
        if (gr < K && gc < N)
            C[gr * N + gc] = sC[r][c];
    }
}
