/*
 * Pooling + im2col kernels (Conv2D 路径前置算子)
 *   - maxpool2d (NCHW)
 *   - avgpool2d (NCHW)
 *   - im2col    (Conv2D 前置展开)
 *
 * 拆分自原 mlp_cuda_kernels.cu。Conv2D 主流程在 binding.cpp 中,
 * 顺序为 im2col -> launch_matmul_tiled_auto -> +bias -> reshape。
 */

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// ============================================================
// MaxPool2D (NCHW)
// ============================================================

__global__ void maxpool2d_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int nc = idx / (H_out * W_out);
    int c = nc % C;
    int n = nc / C;

    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    float max_val = -FLT_MAX;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h_in = h_start + kh;
        if (h_in < 0 || h_in >= H) continue;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w_in = w_start + kw;
            if (w_in < 0 || w_in >= W) continue;
            float val = input[((n * C + c) * H + h_in) * W + w_in];
            max_val = fmaxf(max_val, val);
        }
    }
    output[((n * C + c) * H_out + h_out) * W_out + w_out] = max_val;
}

void launch_maxpool2d(
    const float* input, float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    cudaStream_t stream)
{
    int total = N * C * H_out * W_out;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    maxpool2d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W, H_out, W_out, kernel_size, stride, padding);
}

// ============================================================
// AvgPool2D (NCHW)
// ============================================================

__global__ void avgpool2d_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int nc = idx / (H_out * W_out);
    int c = nc % C;
    int n = nc / C;

    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    float acc = 0.0f;
    int count = 0;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h_in = h_start + kh;
        if (h_in < 0 || h_in >= H) continue;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w_in = w_start + kw;
            if (w_in < 0 || w_in >= W) continue;
            acc += input[((n * C + c) * H + h_in) * W + w_in];
            ++count;
        }
    }
    output[((n * C + c) * H_out + h_out) * W_out + w_out] = acc / count;
}

void launch_avgpool2d(
    const float* input, float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    cudaStream_t stream)
{
    int total = N * C * H_out * W_out;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    avgpool2d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W, H_out, W_out, kernel_size, stride, padding);
}

// ============================================================
// im2col (Conv2D 前置展开)
// ============================================================

__global__ void im2col_kernel(
    const float* __restrict__ input, float* __restrict__ col,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = N * H_out * W_out;
    int col_len = C * KH * KW;
    if (idx >= total_rows * col_len) return;

    int col_offset = idx % col_len;
    int row_idx = idx / col_len;

    int w_out = row_idx % W_out;
    int h_out = (row_idx / W_out) % H_out;
    int n = row_idx / (H_out * W_out);

    int kw = col_offset % KW;
    int kh = (col_offset / KW) % KH;
    int c = col_offset / (KH * KW);

    int h_in = h_out * stride_h - pad_h + kh;
    int w_in = w_out * stride_w - pad_w + kw;

    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        col[idx] = input[((n * C + c) * H + h_in) * W + w_in];
    } else {
        col[idx] = 0.0f;
    }
}

void launch_im2col(
    const float* input, float* col,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW,
    int stride, int padding,
    cudaStream_t stream)
{
    int total = N * H_out * W_out * C * KH * KW;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    im2col_kernel<<<grid_size, block_size, 0, stream>>>(
        input, col, N, C, H, W, H_out, W_out, KH, KW, stride, stride, padding, padding);
}
