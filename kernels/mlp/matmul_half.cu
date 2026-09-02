// matmul_half: fp16 WMMA TensorCore（fp32 累加，half 输出）
// perf 调优 v2.7: 对齐 shape（M,K,N 均 %32==0）走 cp.async 双缓冲管线内核（消除
// ldmatrix->mma 串行依赖）; 非对齐 shape 走原同步 sC staging 内核（正确性兜底）。
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cuda_runtime.h>

// ---------- 同步内核（非对齐兜底, sC staging） ----------
__global__ __launch_bounds__(128)
void matmul_half_sync_kernel(
    const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
    int M, int K, int N)
{
    constexpr int TILE = 32;
    constexpr int R = 32;
    __shared__ half sA[TILE][R];
    __shared__ half sB[R][TILE];
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
        if (gr < M && gc < N) C[gr * N + gc] = __float2half(sC[r][c]);
    }
}

// ---------- cp.async 多级管线内核（对齐 shape） ----------
template<int STAGES>
__global__ __launch_bounds__(128)
void matmul_half_pipe_kernel(
    const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
    int M, int K, int N)
{
    constexpr int TILE = 32, R = 32;
    __shared__ half sA[STAGES][TILE][R];
    __shared__ half sB[STAGES][R][TILE];
    __shared__ float sC[TILE][TILE];
    int warp_row = (threadIdx.x / 64) * 16;
    int warp_col = ((threadIdx.x / 32) % 2) * 16;
    int block_row = blockIdx.y * TILE, block_col = blockIdx.x * TILE;
    const int num_tiles = K / R;   // 对齐保证整除
    const half2* A2 = reinterpret_cast<const half2*>(A);
    const half2* B2 = reinterpret_cast<const half2*>(B);

    auto issue = [&](int kt) {
        int buf = kt & 1, k_start = kt * R;
        // 16B (8-half) 向量化 cp.async（v2.12）
        for (int i = threadIdx.x; i < TILE * R / 8; i += blockDim.x) {
            int h8 = i * 8, r = h8 / R, c8 = h8 % R;
            const float4* sa = reinterpret_cast<const float4*>(&A2[(block_row + r) * (K / 2) + (k_start + c8) / 2]);
            __pipeline_memcpy_async(&sA[buf][r][c8], sa, 16);
        }
        for (int i = threadIdx.x; i < R * TILE / 8; i += blockDim.x) {
            int h8 = i * 8, r = h8 / TILE, c8 = h8 % TILE;
            const float4* sb = reinterpret_cast<const float4*>(&B2[(k_start + r) * (N / 2) + (block_col + c8) / 2]);
            __pipeline_memcpy_async(&sB[buf][r][c8], sb, 16);
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
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a0, a1;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b0, b1;
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
        C[gr * N + gc] = __float2half(sC[r][c]);
    }
}


// ---------- gate+up 融合 cp.async 管线内核（对齐 shape; A 一次读, 双 B / 双 acc） ----------
template<int STAGES>
__global__ __launch_bounds__(128)
void matmul_half_pair_pipe_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B1, const half* __restrict__ B2,
    half* __restrict__ C1, half* __restrict__ C2,
    int M, int K, int N)
{
    constexpr int TILE = 32, R = 32;
    __shared__ half sA[STAGES][TILE][R];
    __shared__ half sB1[STAGES][R][TILE];
    __shared__ half sB2[STAGES][R][TILE];
    __shared__ float sC[TILE][TILE];
    int warp_row = (threadIdx.x / 64) * 16;
    int warp_col = ((threadIdx.x / 32) % 2) * 16;
    int block_row = blockIdx.y * TILE, block_col = blockIdx.x * TILE;
    const int num_tiles = K / R;
    const half2* A2 = reinterpret_cast<const half2*>(A);
    const half2* B1_2 = reinterpret_cast<const half2*>(B1);
    const half2* B2_2 = reinterpret_cast<const half2*>(B2);

    auto issue = [&](int kt) {
        int buf = kt & 1, k_start = kt * R;
        for (int i = threadIdx.x; i < TILE * R / 8; i += blockDim.x) {
            int h8 = i * 8, r = h8 / R, c8 = h8 % R;
            const float4* sa = reinterpret_cast<const float4*>(&A2[(block_row + r) * (K / 2) + (k_start + c8) / 2]);
            __pipeline_memcpy_async(&sA[buf][r][c8], sa, 16);
        }
        for (int i = threadIdx.x; i < R * TILE / 8; i += blockDim.x) {
            int h8 = i * 8, r = h8 / TILE, c8 = h8 % TILE;
            int off = (k_start + r) * (N / 2) + (block_col + c8) / 2;
            const float4* b1 = reinterpret_cast<const float4*>(&B1_2[off]);
            const float4* b2 = reinterpret_cast<const float4*>(&B2_2[off]);
            __pipeline_memcpy_async(&sB1[buf][r][c8], b1, 16);
            __pipeline_memcpy_async(&sB2[buf][r][c8], b2, 16);
        }
        __pipeline_commit();
    };

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc1, acc2;
    nvcuda::wmma::fill_fragment(acc1, 0.0f);
    nvcuda::wmma::fill_fragment(acc2, 0.0f);
    issue(0);
    for (int kt = 0; kt < num_tiles; ++kt) {
        if (kt + STAGES - 1 < num_tiles) issue(kt + STAGES - 1);
        {
            int rem = num_tiles - kt - 1;
            int wait = (rem < STAGES - 1) ? rem : (STAGES - 1);
            if (wait < 0) wait = 0;
            __pipeline_wait_prior(wait);
        }
        __syncthreads();
        const int buf = kt & 1;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a0, a1;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b0, b1;
        nvcuda::wmma::load_matrix_sync(a0, &sA[buf][warp_row][0],  R);
        nvcuda::wmma::load_matrix_sync(a1, &sA[buf][warp_row][16], R);
        nvcuda::wmma::load_matrix_sync(b0, &sB1[buf][0][warp_col],  TILE);
        nvcuda::wmma::load_matrix_sync(b1, &sB1[buf][16][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc1, a0, b0, acc1);
        nvcuda::wmma::mma_sync(acc1, a1, b1, acc1);
        nvcuda::wmma::load_matrix_sync(b0, &sB2[buf][0][warp_col],  TILE);
        nvcuda::wmma::load_matrix_sync(b1, &sB2[buf][16][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc2, a0, b0, acc2);
        nvcuda::wmma::mma_sync(acc2, a1, b1, acc2);
        __syncthreads();
    }
    // 顺序 epilogue：共享单个 sC（避免 2x sC 占用翻倍）
    nvcuda::wmma::store_matrix_sync(&sC[warp_row][warp_col], acc1, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        int r = i / TILE, c = i % TILE;
        C1[(block_row + r) * N + block_col + c] = __float2half(sC[r][c]);
    }
    __syncthreads();
    nvcuda::wmma::store_matrix_sync(&sC[warp_row][warp_col], acc2, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        int r = i / TILE, c = i % TILE;
        C2[(block_row + r) * N + block_col + c] = __float2half(sC[r][c]);
    }
}



// ---------- gate+up 融合 + swiglu epilogue（输出 hidden = SiLU(gate)*up, A 一次读） ----------
// 块级只需: 此内核(hidden) + matmul(hidden, wd) —— 免中间 gate/up 物化与独立 swiglu kernel
template<int STAGES>
__global__ __launch_bounds__(128)
void matmul_half_pair_swiglu_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B1, const half* __restrict__ B2,
    half* __restrict__ C,
    int M, int K, int N)
{
    constexpr int TILE = 32, R = 32;
    __shared__ half sA[STAGES][TILE][R];
    __shared__ half sB1[STAGES][R][TILE];
    __shared__ half sB2[STAGES][R][TILE];
    int warp_row = (threadIdx.x / 64) * 16;
    int warp_col = ((threadIdx.x / 32) % 2) * 16;
    int block_row = blockIdx.y * TILE, block_col = blockIdx.x * TILE;
    const int num_tiles = K / R;
    const half2* A2 = reinterpret_cast<const half2*>(A);
    const half2* B1_2 = reinterpret_cast<const half2*>(B1);
    const half2* B2_2 = reinterpret_cast<const half2*>(B2);

    auto issue = [&](int kt) {
        int buf = kt & 1, k_start = kt * R;
        for (int i = threadIdx.x; i < TILE * R / 8; i += blockDim.x) {
            int h8 = i * 8, r = h8 / R, c8 = h8 % R;
            const float4* sa = reinterpret_cast<const float4*>(&A2[(block_row + r) * (K / 2) + (k_start + c8) / 2]);
            __pipeline_memcpy_async(&sA[buf][r][c8], sa, 16);
        }
        for (int i = threadIdx.x; i < R * TILE / 8; i += blockDim.x) {
            int h8 = i * 8, r = h8 / TILE, c8 = h8 % TILE;
            int off = (k_start + r) * (N / 2) + (block_col + c8) / 2;
            const float4* b1 = reinterpret_cast<const float4*>(&B1_2[off]);
            const float4* b2 = reinterpret_cast<const float4*>(&B2_2[off]);
            __pipeline_memcpy_async(&sB1[buf][r][c8], b1, 16);
            __pipeline_memcpy_async(&sB2[buf][r][c8], b2, 16);
        }
        __pipeline_commit();
    };

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc1, acc2;
    nvcuda::wmma::fill_fragment(acc1, 0.0f);
    nvcuda::wmma::fill_fragment(acc2, 0.0f);
    issue(0);
    for (int kt = 0; kt < num_tiles; ++kt) {
        if (kt + STAGES - 1 < num_tiles) issue(kt + STAGES - 1);
        {
            int rem = num_tiles - kt - 1;
            int wait = (rem < STAGES - 1) ? rem : (STAGES - 1);
            if (wait < 0) wait = 0;
            __pipeline_wait_prior(wait);
        }
        __syncthreads();
        const int buf = kt & 1;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a0, a1;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b0, b1;
        nvcuda::wmma::load_matrix_sync(a0, &sA[buf][warp_row][0],  R);
        nvcuda::wmma::load_matrix_sync(a1, &sA[buf][warp_row][16], R);
        nvcuda::wmma::load_matrix_sync(b0, &sB1[buf][0][warp_col],  TILE);
        nvcuda::wmma::load_matrix_sync(b1, &sB1[buf][16][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc1, a0, b0, acc1);
        nvcuda::wmma::mma_sync(acc1, a1, b1, acc1);
        nvcuda::wmma::load_matrix_sync(b0, &sB2[buf][0][warp_col],  TILE);
        nvcuda::wmma::load_matrix_sync(b1, &sB2[buf][16][warp_col], TILE);
        nvcuda::wmma::mma_sync(acc2, a0, b0, acc2);
        nvcuda::wmma::mma_sync(acc2, a1, b1, acc2);
        __syncthreads();
    }
    // swiglu epilogue: hidden = SiLU(gate)*up（双 sC 缓冲，store_matrix_sync 布局无关, 保证正确）
    __shared__ float sC1[TILE][TILE];
    __shared__ float sC2[TILE][TILE];
    nvcuda::wmma::store_matrix_sync(&sC1[warp_row][warp_col], acc1, TILE, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync(&sC2[warp_row][warp_col], acc2, TILE, nvcuda::wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
        int r = i / TILE, c = i % TILE;
        float gg = sC1[r][c];
        float uu = sC2[r][c];
        float sig = 1.0f / (1.0f + expf(-gg));
        C[(block_row + r) * N + block_col + c] = __float2half(gg * sig * uu);
    }
}

void launch_matmul_half_pair_swiglu(
    const half* A, const half* B1, const half* B2, half* C,
    int M, int K, int N, cudaStream_t stream)
{
    dim3 block(128);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    matmul_half_pair_swiglu_kernel<2><<<grid, block, 0, stream>>>(A, B1, B2, C, M, K, N);
}

void launch_matmul_half_pair(

    const half* A, const half* B1, const half* B2,
    half* C1, half* C2, int M, int K, int N, cudaStream_t stream)
{
    dim3 block(128);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    matmul_half_pair_pipe_kernel<2><<<grid, block, 0, stream>>>(A, B1, B2, C1, C2, M, K, N);
}

void launch_matmul_half(

    const half* A, const half* B, half* C,
    int M, int K, int N, cudaStream_t stream)
{
    const bool aligned = (M % 32 == 0) && (K % 32 == 0) && (N % 32 == 0) && (K >= 32);
    if (aligned) {
        dim3 block(128);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        const int stages = 2;  // 实测 2/3/4: 2 最优（3/4 占用损失抵消）；见 perf_tuning_cuda.md v2.8
        switch (stages) {
            case 3: matmul_half_pipe_kernel<3><<<grid, block, 0, stream>>>(A, B, C, M, K, N); break;
            case 4: matmul_half_pipe_kernel<4><<<grid, block, 0, stream>>>(A, B, C, M, K, N); break;
            default: matmul_half_pipe_kernel<2><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
        }
    } else {
        dim3 block(128);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmul_half_sync_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    }
}