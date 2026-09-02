// matmul_bf16: fp16 WMMA TensorCore（fp32 累加，__nv_bfloat16 输出）
// perf 调优 v2.7: 对齐 shape（M,K,N 均 %32==0）走 cp.async 双缓冲管线内核（消除
// ldmatrix->mma 串行依赖）; 非对齐 shape 走原同步 sC staging 内核（正确性兜底）。
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cuda_runtime.h>

// ---------- 同步内核（非对齐兜底, sC staging） ----------
__global__ __launch_bounds__(128)
void matmul_bf16_sync_kernel(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
    int M, int K, int N)
{
    constexpr int TILE = 32;
    constexpr int R = 32;
    __shared__ __nv_bfloat16 sA[TILE][R];
    __shared__ __nv_bfloat16 sB[R][TILE];
    __shared__ float sC[TILE][TILE];
    int warp_row = (threadIdx.x / 64) * 16;
    int warp_col = ((threadIdx.x / 32) % 2) * 16;
    int block_row = blockIdx.y * TILE, block_col = blockIdx.x * TILE;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);
    for (int kt = 0; kt < (K + R - 1) / R; ++kt) {
        int k_start = kt * R;
        for (int i = threadIdx.x; i < TILE * R; i += blockDim.x) {
            int r = i / R, c = i % R;
            int gr = block_row + r, gc = k_start + c;
            sA[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
        }
        for (int i = threadIdx.x; i < R * TILE; i += blockDim.x) {
            int r = i / TILE, c = i % TILE;
            int gr = k_start + r, gc = block_col + c;
            sB[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2bfloat16(0.0f);
        }
        __syncthreads();
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a0, a1;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> b0, b1;
        nvcuda::wmma::load_matrix_sync(a0, &sA[warp_row][0], R);
        nvcuda::wmma::load_matrix_sync(a1, &sA[warp_row][16], R);
        nvcuda::wmma::load_matrix_sync(b0, &sB[0][warp_col], TILE);
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
        if (gr < M && gc < N) C[gr * N + gc] = __float2bfloat16(sC[r][c]);
    }
}

// ---------- cp.async 多级管线内核（对齐 shape） ----------
template<int STAGES>
__global__ __launch_bounds__(128)
void matmul_bf16_pipe_kernel(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
    int M, int K, int N)
{
    constexpr int TILE = 32, R = 32;
    __shared__ __nv_bfloat16 sA[STAGES][TILE][R];
    __shared__ __nv_bfloat16 sB[STAGES][R][TILE];
    __shared__ float sC[TILE][TILE];
    int warp_row = (threadIdx.x / 64) * 16;
    int warp_col = ((threadIdx.x / 32) % 2) * 16;
    int block_row = blockIdx.y * TILE, block_col = blockIdx.x * TILE;
    const int num_tiles = K / R;   // 对齐保证整除
    const __nv_bfloat162* A2 = reinterpret_cast<const __nv_bfloat162*>(A);
    const __nv_bfloat162* B2 = reinterpret_cast<const __nv_bfloat162*>(B);

    auto issue = [&](int kt) {
        int buf = kt & 1, k_start = kt * R;
        for (int i = threadIdx.x; i < TILE * R / 2; i += blockDim.x) {
            int h = i * 2, r = h / R, cstart = h % R;
            const __nv_bfloat162* src = &A2[(block_row + r) * (K / 2) + (k_start + cstart) / 2];
            __pipeline_memcpy_async(&sA[buf][r][cstart], src, 4);
        }
        for (int i = threadIdx.x; i < R * TILE / 2; i += blockDim.x) {
            int h = i * 2, r = h / TILE, cstart = h % TILE;
            const __nv_bfloat162* src = &B2[(k_start + r) * (N / 2) + (block_col + cstart) / 2];
            __pipeline_memcpy_async(&sB[buf][r][cstart], src, 4);
        }
        __pipeline_commit();
    };

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);
    issue(0);
    for (int kt = 0; kt < num_tiles; ++kt) {
        if (kt + STAGES - 1 < num_tiles) issue(kt + STAGES - 1);   // 预取窗口内最远一 tile
        __pipeline_wait_prior((kt + STAGES - 1 < num_tiles) ? (STAGES - 1) : (num_tiles - kt - 1 < 0 ? 0 : num_tiles - kt - 1));   // 本 k-tile 数据就绪
        __syncthreads();   // 他人 cp.async 写入对本线程可见（关键）
        const int buf = kt & 1;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a0, a1;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> b0, b1;
        nvcuda::wmma::load_matrix_sync(a0, &sA[buf][warp_row][0],  R);
        nvcuda::wmma::load_matrix_sync(a1, &sA[buf][warp_row][16], R);
        nvcuda::wmma::load_matrix_sync(b0, &sB[buf][0][warp_col],  TILE);
        nvcuda::wmma::load_matrix_sync(b1, &sB[buf][16][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc, a0, b0, acc);
        nvcuda::wmma::mma_sync(acc, a1, b1, acc);
        __syncthreads();   // 所有 warp 读完 buf 后，下一轮预取才能覆写 (kt+1)&1
    }
    nvcuda::wmma::store_matrix_sync(&sC[warp_row][warp_col], acc, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        int r = i / TILE, c = i % TILE;
        int gr = block_row + r, gc = block_col + c;
        C[gr * N + gc] = __float2bfloat16(sC[r][c]);
    }
}

void launch_matmul_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int K, int N, cudaStream_t stream)
{
    const bool aligned = (M % 32 == 0) && (K % 32 == 0) && (N % 32 == 0) && (K >= 32);
    if (aligned) {
        dim3 block(128);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        const int stages = 2;  // 实测 2/3/4: 2 最优（3/4 占用损失抵消）；见 perf_tuning_cuda.md v2.8
        switch (stages) {
            case 3: matmul_bf16_pipe_kernel<3><<<grid, block, 0, stream>>>(A, B, C, M, K, N); break;
            case 4: matmul_bf16_pipe_kernel<4><<<grid, block, 0, stream>>>(A, B, C, M, K, N); break;
            default: matmul_bf16_pipe_kernel<2><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
        }
    } else {
        dim3 block(128);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmul_bf16_sync_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    }
}