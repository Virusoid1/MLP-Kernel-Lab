/*
 * PyTorch C++ Extension binding
 *
 * 将 CUDA kernel 注册为 Python 可调用的 torch 函数。
 *
 * 编译: python setup.py install
 * 使用: import mlp_cuda; out = mlp_cuda.matmul_tiled(x, w)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

// 兼容不同 PyTorch 版本的 CUDA stream 获取
static cudaStream_t _get_cuda_stream(torch::Tensor t) {
    return c10::cuda::getCurrentCUDAStream(t.device().index()).stream();
}

// Forward 声明 CUDA kernel launch 函数 (定义在 kernels/mlp/*.cu)
void launch_matmul_naive(
    const float* A, const float* B, float* C,
    int M, int K, int N, cudaStream_t stream);

void launch_matmul_tiled(
    const float* A, const float* B, float* C,
    int M, int K, int N, int BLOCK_M, int BLOCK_N, int BLOCK_K,
    cudaStream_t stream);

void launch_matmul_tiled_auto(
    const float* A, const float* B, float* C,
    int M, int K, int N, cudaStream_t stream);

void launch_mlp_fused_first_layer(
    const float* X, const float* W1, const float* bias, float* H,
    int M, int K, int N, cudaStream_t stream);

void launch_swiglu_fused(
    const float* gate, const float* up, float* output,
    int total_elements, cudaStream_t stream);
void launch_swiglu_fused_half(
    const half* gate, const half* up, half* output,
    int total_elements, cudaStream_t stream);
void launch_swiglu_fused_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* output,
    int total_elements, cudaStream_t stream);

void launch_gelu(const float* input, float* output, int n, cudaStream_t stream);
void launch_relu(const float* input, float* output, int n, cudaStream_t stream);
void launch_silu(const float* input, float* output, int n, cudaStream_t stream);

// fp16 变体（cuda 算子级 fp16 解锁，v2 4.x）
void launch_gelu_half(const half* input, half* output, int n, cudaStream_t stream);
void launch_relu_half(const half* input, half* output, int n, cudaStream_t stream);
void launch_silu_half(const half* input, half* output, int n, cudaStream_t stream);

// bf16 变体（cuda 算子级 bf16 解锁，v2 4.x）
void launch_gelu_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int n, cudaStream_t stream);
void launch_relu_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int n, cudaStream_t stream);
void launch_silu_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int n, cudaStream_t stream);

void launch_gelu_backward(
    const float* grad_output, const float* input, float* grad_input,
    int n, cudaStream_t stream);
void launch_relu_backward(
    const float* grad_output, const float* input, float* grad_input,
    int n, cudaStream_t stream);
void launch_silu_backward(
    const float* grad_output, const float* input, float* grad_input,
    int n, cudaStream_t stream);

void launch_bias_add(
    const float* input, const float* bias, float* output,
    int M, int N, cudaStream_t stream);
void launch_bias_add_half(
    const half* input, const half* bias, half* output,
    int M, int N, cudaStream_t stream);
void launch_bias_add_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* bias, __nv_bfloat16* output,
    int M, int N, cudaStream_t stream);

void launch_matmul_transB(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream);

void launch_matmul_transA(
    const float* A, const float* B, float* C,
    int M, int K, int N, cudaStream_t stream);

// fp16 TensorCore matmul (v2: cuda fp16 block 解锁)
void launch_matmul_half(
    const half* A, const half* B, half* C,
    int M, int K, int N, cudaStream_t stream);

void launch_matmul_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int K, int N, cudaStream_t stream);

// gate+up 融合（A 一次读）: 定义在 matmul_half.cu / matmul_bf16.cu
void launch_matmul_half_pair(
    const half* A, const half* B1, const half* B2,
    half* C1, half* C2, int M, int K, int N, cudaStream_t stream);
void launch_matmul_bf16_pair(
    const __nv_bfloat16* A, const __nv_bfloat16* B1, const __nv_bfloat16* B2,
    __nv_bfloat16* C1, __nv_bfloat16* C2, int M, int K, int N, cudaStream_t stream);

// pair + swiglu epilogue（输出 hidden）
void launch_matmul_half_pair_swiglu(
    const half* A, const half* B1, const half* B2, half* C,
    int M, int K, int N, cudaStream_t stream);
void launch_matmul_bf16_pair_swiglu(
    const __nv_bfloat16* A, const __nv_bfloat16* B1, const __nv_bfloat16* B2,
    __nv_bfloat16* C, int M, int K, int N, cudaStream_t stream);

void launch_gelu_backward_vec4(
    const float* grad_output, const float* input, float* grad_input,
    int n, cudaStream_t stream);
void launch_relu_backward_vec4(
    const float* grad_output, const float* input, float* grad_input,
    int n, cudaStream_t stream);
void launch_silu_backward_vec4(
    const float* grad_output, const float* input, float* grad_input,
    int n, cudaStream_t stream);

void launch_layernorm_forward(
    const float* X, float* Y,
    const float* Gamma, const float* Beta,
    float* Mean, float* Rstd,
    int B, int N, float eps, cudaStream_t stream);

void launch_layernorm_backward(
    const float* DY, const float* X,
    const float* Gamma, const float* Mean, const float* Rstd,
    float* DX, float* DGamma, float* DBeta,
    int B, int N, cudaStream_t stream);

void launch_softmax(
    const float* input, float* output,
    int M, int N, cudaStream_t stream);
void launch_softmax_half(
    const half* input, half* output,
    int M, int N, cudaStream_t stream);
void launch_softmax_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int M, int N, cudaStream_t stream);

// 前向声明（pair_swiglu 非对齐回落使用，定义在文件后方）
torch::Tensor swiglu_fused(torch::Tensor gate, torch::Tensor up);
torch::Tensor matmul_half(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_bf16(torch::Tensor A, torch::Tensor B);

void launch_maxpool2d(
    const float* input, float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    cudaStream_t stream);
void launch_maxpool2d_half(
    const half* input, half* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    cudaStream_t stream);
void launch_maxpool2d_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    cudaStream_t stream);

void launch_avgpool2d(
    const float* input, float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    cudaStream_t stream);
void launch_avgpool2d_half(
    const half* input, half* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    cudaStream_t stream);
void launch_avgpool2d_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    cudaStream_t stream);

void launch_im2col(
    const float* input, float* col,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW,
    int stride, int padding,
    cudaStream_t stream);
void launch_im2col_half(
    const half* input, half* col,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW,
    int stride, int padding,
    cudaStream_t stream);
void launch_im2col_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* col,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int KH, int KW,
    int stride, int padding,
    cudaStream_t stream);

// ============================================================
// 输入校验辅助宏
// ============================================================

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")

// ============================================================
// matmul_naive
// ============================================================

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A); CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B);
    CHECK_FLOAT32(A); CHECK_FLOAT32(B);

    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(A.size(1) == B.size(0), "Shape mismatch: A(", M, ",", K, ") @ B(", B.size(0), ",", N, ")");

    auto C = torch::empty({M, N}, A.options());
    launch_matmul_naive(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N, _get_cuda_stream(A));
    return C;
}

// ============================================================
// matmul_tiled
// ============================================================

torch::Tensor matmul_tiled(
    torch::Tensor A, torch::Tensor B,
    int BLOCK_M, int BLOCK_N, int BLOCK_K)
{
    CHECK_CUDA(A); CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B);
    CHECK_FLOAT32(A); CHECK_FLOAT32(B);

    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(A.size(1) == B.size(0), "Shape mismatch: A(", M, ",", K, ") @ B(", B.size(0), ",", N, ")");

    auto C = torch::empty({M, N}, A.options());
    launch_matmul_tiled(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N, BLOCK_M, BLOCK_N, BLOCK_K,
        _get_cuda_stream(A));
    return C;
}

// matmul_tiled_auto: 按矩阵尺寸自动选择最优 tile
torch::Tensor matmul_tiled_auto(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A); CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B);
    CHECK_FLOAT32(A); CHECK_FLOAT32(B);

    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(A.size(1) == B.size(0), "Shape mismatch: A(", M, ",", K, ") @ B(", B.size(0), ",", N, ")");

    auto C = torch::empty({M, N}, A.options());
    launch_matmul_tiled_auto(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N, _get_cuda_stream(A));
    return C;
}

// matmul_half: fp16 TensorCore matmul（fp16 in → fp32 acc → fp16 out）
torch::Tensor matmul_half(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A); CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B);
    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be float16 for matmul_half");
    TORCH_CHECK(B.scalar_type() == torch::kHalf, "B must be float16 for matmul_half");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(A.size(1) == B.size(0), "Shape mismatch: A(", M, ",", K, ") @ B(", B.size(0), ",", N, ")");

    auto C = torch::empty({M, N}, A.options());
    launch_matmul_half(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, K, N, _get_cuda_stream(A));
    return C;
}

// matmul_bf16: bf16 TensorCore matmul（bf16 in → fp32 acc → bf16 out）
torch::Tensor matmul_bf16(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A); CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B);
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16, "A must be bfloat16 for matmul_bf16");
    TORCH_CHECK(B.scalar_type() == torch::kBFloat16, "B must be bfloat16 for matmul_bf16");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(A.size(1) == B.size(0), "Shape mismatch: A(", M, ",", K, ") @ B(", B.size(0), ",", N, ")");

    auto C = torch::empty({M, N}, A.options());
    launch_matmul_bf16(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
        M, K, N, _get_cuda_stream(A));
    return C;
}

// matmul_half_pair: gate=X@Wg 与 up=X@Wu 融合（A 一次读, 双 B/双 acc）
std::tuple<torch::Tensor, torch::Tensor> matmul_half_pair(
    torch::Tensor A, torch::Tensor B1, torch::Tensor B2)
{
    CHECK_CUDA(A); CHECK_CUDA(B1); CHECK_CUDA(B2);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B1); CHECK_CONTIGUOUS(B2);
    TORCH_CHECK(A.scalar_type() == torch::kHalf &&
                B1.scalar_type() == torch::kHalf && B2.scalar_type() == torch::kHalf,
                "matmul_half_pair requires fp16 inputs");

    int M = A.size(0), K = A.size(1), N = B1.size(1);
    TORCH_CHECK(A.size(1) == B1.size(0) && A.size(1) == B2.size(0), "K mismatch");
    TORCH_CHECK(B1.size(1) == B2.size(1), "N mismatch");

    auto C1 = torch::empty({M, N}, A.options());
    auto C2 = torch::empty({M, N}, A.options());
    const bool aligned = (M % 32 == 0) && (K % 32 == 0) && (N % 32 == 0) && (K >= 32);
    if (aligned) {
        dim3 block(128);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        launch_matmul_half_pair(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(B1.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(B2.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C1.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C2.data_ptr<at::Half>()),
            M, K, N, _get_cuda_stream(A));
    } else {
        auto g = ::matmul_half(A, B1);
        auto u = ::matmul_half(A, B2);
        C1.copy_(g); C2.copy_(u);
    }
    return std::make_tuple(C1, C2);
}

// matmul_bf16_pair: gate=X@Wg 与 up=X@Wu 融合（A 一次读, 双 B/双 acc）
std::tuple<torch::Tensor, torch::Tensor> matmul_bf16_pair(
    torch::Tensor A, torch::Tensor B1, torch::Tensor B2)
{
    CHECK_CUDA(A); CHECK_CUDA(B1); CHECK_CUDA(B2);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B1); CHECK_CONTIGUOUS(B2);
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16 &&
                B1.scalar_type() == torch::kBFloat16 && B2.scalar_type() == torch::kBFloat16,
                "matmul_bf16_pair requires bf16 inputs");

    int M = A.size(0), K = A.size(1), N = B1.size(1);
    TORCH_CHECK(A.size(1) == B1.size(0) && A.size(1) == B2.size(0), "K mismatch");
    TORCH_CHECK(B1.size(1) == B2.size(1), "N mismatch");

    auto C1 = torch::empty({M, N}, A.options());
    auto C2 = torch::empty({M, N}, A.options());
    const bool aligned = (M % 32 == 0) && (K % 32 == 0) && (N % 32 == 0) && (K >= 32);
    if (aligned) {
        dim3 block(128);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        launch_matmul_bf16_pair(
            reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(B1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(B2.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(C1.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(C2.data_ptr<at::BFloat16>()),
            M, K, N, _get_cuda_stream(A));
    } else {
        auto g = matmul_bf16(A, B1);
        auto u = matmul_bf16(A, B2);
        C1.copy_(g); C2.copy_(u);
    }
    return std::make_tuple(C1, C2);
}

// matmul_half_pair_swiglu: hidden = SiLU(X@Wg) * (X@Wu)（A 一次读 + swiglu epilogue 融合）
torch::Tensor matmul_half_pair_swiglu(
    torch::Tensor A, torch::Tensor B1, torch::Tensor B2)
{
    CHECK_CUDA(A); CHECK_CUDA(B1); CHECK_CUDA(B2);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B1); CHECK_CONTIGUOUS(B2);
    TORCH_CHECK(A.scalar_type() == torch::kHalf &&
                B1.scalar_type() == torch::kHalf && B2.scalar_type() == torch::kHalf,
                "requires fp16");
    int M = A.size(0), K = A.size(1), N = B1.size(1);
    TORCH_CHECK(A.size(1) == B1.size(0) && A.size(1) == B2.size(0), "K mismatch");
    TORCH_CHECK(B1.size(1) == B2.size(1), "N mismatch");
    auto C = torch::empty({M, N}, A.options());
    const bool aligned = (M % 32 == 0) && (K % 32 == 0) && (N % 32 == 0) && (K >= 32);
    if (aligned) {
        dim3 block(128);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        launch_matmul_half_pair_swiglu(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(B1.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(B2.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            M, K, N, _get_cuda_stream(A));
        return C;
    }
    // 非对齐：pair + swiglu_fused 组合
    auto [g, u] = matmul_half_pair(A, B1, B2);
    return swiglu_fused(g, u);
}

// matmul_bf16_pair_swiglu: hidden = SiLU(X@Wg) * (X@Wu)（A 一次读 + swiglu epilogue）
torch::Tensor matmul_bf16_pair_swiglu(
    torch::Tensor A, torch::Tensor B1, torch::Tensor B2)
{
    CHECK_CUDA(A); CHECK_CUDA(B1); CHECK_CUDA(B2);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B1); CHECK_CONTIGUOUS(B2);
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16 &&
                B1.scalar_type() == torch::kBFloat16 && B2.scalar_type() == torch::kBFloat16,
                "requires bf16");
    int M = A.size(0), K = A.size(1), N = B1.size(1);
    TORCH_CHECK(A.size(1) == B1.size(0) && A.size(1) == B2.size(0), "K mismatch");
    TORCH_CHECK(B1.size(1) == B2.size(1), "N mismatch");
    auto C = torch::empty({M, N}, A.options());
    const bool aligned = (M % 32 == 0) && (K % 32 == 0) && (N % 32 == 0) && (K >= 32);
    if (aligned) {
        dim3 block(128);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        launch_matmul_bf16_pair_swiglu(
            reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(B1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(B2.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
            M, K, N, _get_cuda_stream(A));
        return C;
    }
    auto [g, u] = matmul_bf16_pair(A, B1, B2);
    return swiglu_fused(g, u);
}

// ============================================================
// bias_add
// ============================================================

torch::Tensor bias_add(torch::Tensor x, torch::Tensor bias) {
    CHECK_CUDA(x); CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(bias);
    TORCH_CHECK(x.scalar_type() == bias.scalar_type(),
                "x and bias must have same dtype; got ", x.scalar_type(), " vs ", bias.scalar_type());

    TORCH_CHECK(x.dim() == 2, "x must be 2D (M, N)");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == x.size(1),
                "bias must be 1D with shape (N,)");

    int M = x.size(0), N = x.size(1);
    auto output = torch::empty({M, N}, x.options());
    if (x.scalar_type() == torch::kHalf) {
        launch_bias_add_half(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            M, N, _get_cuda_stream(x));
    } else if (x.scalar_type() == torch::kBFloat16) {
        launch_bias_add_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            M, N, _get_cuda_stream(x));
    } else {
        CHECK_FLOAT32(x); CHECK_FLOAT32(bias);
        launch_bias_add(
            x.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
            M, N, _get_cuda_stream(x));
    }
    return output;
}

// ============================================================
// 激活函数 (elementwise, 支持任意形状)
// ============================================================

// --- GELU ---
torch::Tensor gelu(torch::Tensor x) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
    auto output = torch::empty_like(x);
    if (x.scalar_type() == torch::kHalf) {
        launch_gelu_half(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            x.numel(), _get_cuda_stream(x));
    } else if (x.scalar_type() == torch::kBFloat16) {
        launch_gelu_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            x.numel(), _get_cuda_stream(x));
    } else {
        CHECK_FLOAT32(x);
        launch_gelu(x.data_ptr<float>(), output.data_ptr<float>(), x.numel(),
                    _get_cuda_stream(x));
    }
    return output;
}

// --- ReLU ---
torch::Tensor relu(torch::Tensor x) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
    auto output = torch::empty_like(x);
    if (x.scalar_type() == torch::kHalf) {
        launch_relu_half(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            x.numel(), _get_cuda_stream(x));
    } else if (x.scalar_type() == torch::kBFloat16) {
        launch_relu_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            x.numel(), _get_cuda_stream(x));
    } else {
        CHECK_FLOAT32(x);
        launch_relu(x.data_ptr<float>(), output.data_ptr<float>(), x.numel(),
                    _get_cuda_stream(x));
    }
    return output;
}

// --- SiLU ---
torch::Tensor silu(torch::Tensor x) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
    auto output = torch::empty_like(x);
    if (x.scalar_type() == torch::kHalf) {
        launch_silu_half(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            x.numel(), _get_cuda_stream(x));
    } else if (x.scalar_type() == torch::kBFloat16) {
        launch_silu_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            x.numel(), _get_cuda_stream(x));
    } else {
        CHECK_FLOAT32(x);
        launch_silu(x.data_ptr<float>(), output.data_ptr<float>(), x.numel(),
                    _get_cuda_stream(x));
    }
    return output;
}

// ============================================================
// 激活函数 backward
// ============================================================

torch::Tensor gelu_backward(torch::Tensor grad_output, torch::Tensor x) {
    CHECK_CUDA(grad_output); CHECK_CUDA(x);
    CHECK_CONTIGUOUS(grad_output); CHECK_CONTIGUOUS(x);
    CHECK_FLOAT32(grad_output); CHECK_FLOAT32(x);

    auto grad_input = torch::empty_like(grad_output);
    launch_gelu_backward(
        grad_output.data_ptr<float>(), x.data_ptr<float>(),
        grad_input.data_ptr<float>(), x.numel(),
        _get_cuda_stream(grad_output));
    return grad_input;
}

torch::Tensor relu_backward(torch::Tensor grad_output, torch::Tensor x) {
    CHECK_CUDA(grad_output); CHECK_CUDA(x);
    CHECK_CONTIGUOUS(grad_output); CHECK_CONTIGUOUS(x);
    CHECK_FLOAT32(grad_output); CHECK_FLOAT32(x);

    auto grad_input = torch::empty_like(grad_output);
    launch_relu_backward(
        grad_output.data_ptr<float>(), x.data_ptr<float>(),
        grad_input.data_ptr<float>(), x.numel(),
        _get_cuda_stream(grad_output));
    return grad_input;
}

torch::Tensor silu_backward(torch::Tensor grad_output, torch::Tensor x) {
    CHECK_CUDA(grad_output); CHECK_CUDA(x);
    CHECK_CONTIGUOUS(grad_output); CHECK_CONTIGUOUS(x);
    CHECK_FLOAT32(grad_output); CHECK_FLOAT32(x);

    auto grad_input = torch::empty_like(grad_output);
    launch_silu_backward(
        grad_output.data_ptr<float>(), x.data_ptr<float>(),
        grad_input.data_ptr<float>(), x.numel(),
        _get_cuda_stream(grad_output));
    return grad_input;
}

// ============================================================
// mlp_fused_first_layer: H = GELU(X @ W1 + bias)
// ============================================================

torch::Tensor mlp_fused_first_layer(
    torch::Tensor X, torch::Tensor W1, torch::Tensor bias)
{
    CHECK_CUDA(X); CHECK_CUDA(W1); CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(X); CHECK_CONTIGUOUS(W1); CHECK_CONTIGUOUS(bias);
    TORCH_CHECK(X.scalar_type() == W1.scalar_type() && X.scalar_type() == bias.scalar_type(),
                "X/W1/bias must have same dtype; got ", X.scalar_type());

    TORCH_CHECK(X.dim() == 2 && W1.dim() == 2, "X and W1 must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(X.size(1) == W1.size(0), "X(", X.size(1), ") != W1 row(", W1.size(0), ")");
    TORCH_CHECK(bias.size(0) == W1.size(1), "bias(", bias.size(0), ") != W1 col(", W1.size(1), ")");

    // fp16/bf16: 组合现有 WMMA matmul + bias_add + gelu（同一 stream 顺序执行）
    // 显式 :: 限定避免与 at::gelu ADL 重载歧义
    if (X.scalar_type() == torch::kHalf) {
        return ::gelu(::bias_add(::matmul_half(X, W1), bias));
    }
    if (X.scalar_type() == torch::kBFloat16) {
        return ::gelu(::bias_add(::matmul_bf16(X, W1), bias));
    }

    int M = X.size(0), K = X.size(1), N = W1.size(1);
    auto H = torch::empty({M, N}, X.options());
    launch_mlp_fused_first_layer(
        X.data_ptr<float>(), W1.data_ptr<float>(), bias.data_ptr<float>(),
        H.data_ptr<float>(), M, K, N,
        _get_cuda_stream(X));
    return H;
}

// ============================================================
// swiglu_fused: output = SiLU(gate) * up
// ============================================================

torch::Tensor swiglu_fused(torch::Tensor gate, torch::Tensor up) {
    CHECK_CUDA(gate); CHECK_CUDA(up);
    CHECK_CONTIGUOUS(gate); CHECK_CONTIGUOUS(up);
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(),
                "gate and up must have same dtype; got ", gate.scalar_type(), " vs ", up.scalar_type());

    TORCH_CHECK(gate.sizes() == up.sizes(),
                "gate and up must have same shape; got ", gate.sizes(), " vs ", up.sizes());

    auto output = torch::empty_like(gate);
    if (gate.scalar_type() == torch::kHalf) {
        launch_swiglu_fused_half(
            reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            gate.numel(), _get_cuda_stream(gate));
    } else if (gate.scalar_type() == torch::kBFloat16) {
        launch_swiglu_fused_bf16(
            reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(up.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            gate.numel(), _get_cuda_stream(gate));
    } else {
        CHECK_FLOAT32(gate); CHECK_FLOAT32(up);
        launch_swiglu_fused(
            gate.data_ptr<float>(), up.data_ptr<float>(), output.data_ptr<float>(),
            gate.numel(), _get_cuda_stream(gate));
    }
    return output;
}

// ============================================================
// matmul_transB: C = A @ B^T (B 不转置)
// A: (M, N) B: (K, N) C: (M, K)
// ============================================================

torch::Tensor matmul_transB(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A); CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B);
    CHECK_FLOAT32(A); CHECK_FLOAT32(B);

    int M = A.size(0), N = A.size(1), K = B.size(0);
    TORCH_CHECK(A.size(1) == B.size(1),
                "N mismatch: A(", M, ",", N, ") B(", K, ",", B.size(1), ")");

    auto C = torch::empty({M, K}, A.options());
    launch_matmul_transB(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K, _get_cuda_stream(A));
    return C;
}

// ============================================================
// matmul_transA: C = A^T @ B (A 不转置)
// A: (M, K) B: (M, N) C: (K, N)
// ============================================================

torch::Tensor matmul_transA(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A); CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B);
    CHECK_FLOAT32(A); CHECK_FLOAT32(B);

    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(A.size(0) == B.size(0),
                "M mismatch: A(", M, ",", K, ") B(", B.size(0), ",", N, ")");

    auto C = torch::empty({K, N}, A.options());
    launch_matmul_transA(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N, _get_cuda_stream(A));
    return C;
}

// ============================================================
// activation backward (float4 向量化)
// ============================================================

torch::Tensor gelu_backward_vec4(torch::Tensor grad_output, torch::Tensor x) {
    CHECK_CUDA(grad_output); CHECK_CUDA(x);
    CHECK_CONTIGUOUS(grad_output); CHECK_CONTIGUOUS(x);
    CHECK_FLOAT32(grad_output); CHECK_FLOAT32(x);

    auto grad_input = torch::empty_like(grad_output);
    launch_gelu_backward_vec4(
        grad_output.data_ptr<float>(), x.data_ptr<float>(),
        grad_input.data_ptr<float>(), x.numel(),
        _get_cuda_stream(grad_output));
    return grad_input;
}

torch::Tensor relu_backward_vec4(torch::Tensor grad_output, torch::Tensor x) {
    CHECK_CUDA(grad_output); CHECK_CUDA(x);
    CHECK_CONTIGUOUS(grad_output); CHECK_CONTIGUOUS(x);
    CHECK_FLOAT32(grad_output); CHECK_FLOAT32(x);

    auto grad_input = torch::empty_like(grad_output);
    launch_relu_backward_vec4(
        grad_output.data_ptr<float>(), x.data_ptr<float>(),
        grad_input.data_ptr<float>(), x.numel(),
        _get_cuda_stream(grad_output));
    return grad_input;
}

torch::Tensor silu_backward_vec4(torch::Tensor grad_output, torch::Tensor x) {
    CHECK_CUDA(grad_output); CHECK_CUDA(x);
    CHECK_CONTIGUOUS(grad_output); CHECK_CONTIGUOUS(x);
    CHECK_FLOAT32(grad_output); CHECK_FLOAT32(x);

    auto grad_input = torch::empty_like(grad_output);
    launch_silu_backward_vec4(
        grad_output.data_ptr<float>(), x.data_ptr<float>(),
        grad_input.data_ptr<float>(), x.numel(),
        _get_cuda_stream(grad_output));
    return grad_input;
}

// ============================================================
// LayerNorm forward / backward
// ============================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> layernorm_forward(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps)
{
    CHECK_CUDA(x); CHECK_CUDA(gamma); CHECK_CUDA(beta);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(gamma); CHECK_CONTIGUOUS(beta);
    CHECK_FLOAT32(x); CHECK_FLOAT32(gamma); CHECK_FLOAT32(beta);

    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    int B = x.size(0), N = x.size(1);

    auto y = torch::empty_like(x);
    auto mean = torch::empty({B}, x.options());
    auto rstd = torch::empty({B}, x.options());

    launch_layernorm_forward(
        x.data_ptr<float>(), y.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(),
        B, N, (float)eps, _get_cuda_stream(x));
    return std::make_tuple(y, mean, rstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> layernorm_backward(
    torch::Tensor dy, torch::Tensor x, torch::Tensor gamma,
    torch::Tensor mean, torch::Tensor rstd)
{
    CHECK_CUDA(dy); CHECK_CUDA(x); CHECK_CUDA(gamma);
    CHECK_CUDA(mean); CHECK_CUDA(rstd);
    CHECK_CONTIGUOUS(dy); CHECK_CONTIGUOUS(x);
    CHECK_FLOAT32(dy); CHECK_FLOAT32(x); CHECK_FLOAT32(gamma);

    TORCH_CHECK(dy.dim() == 2, "dy must be 2D");
    int B = dy.size(0), N = dy.size(1);

    auto dx = torch::empty_like(dy);
    auto d_gamma = torch::zeros_like(gamma);
    auto d_beta = torch::zeros_like(gamma);

    launch_layernorm_backward(
        dy.data_ptr<float>(), x.data_ptr<float>(),
        gamma.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        dx.data_ptr<float>(), d_gamma.data_ptr<float>(), d_beta.data_ptr<float>(),
        B, N, _get_cuda_stream(dy));
    return std::make_tuple(dx, d_gamma, d_beta);
}

// ============================================================
// softmax
// ============================================================

torch::Tensor softmax(torch::Tensor x) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D (M, N)");

    int M = x.size(0), N = x.size(1);
    auto output = torch::empty_like(x);
    if (x.scalar_type() == torch::kHalf) {
        launch_softmax_half(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            M, N, _get_cuda_stream(x));
    } else if (x.scalar_type() == torch::kBFloat16) {
        launch_softmax_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            M, N, _get_cuda_stream(x));
    } else {
        CHECK_FLOAT32(x);
        launch_softmax(
            x.data_ptr<float>(), output.data_ptr<float>(),
            M, N, _get_cuda_stream(x));
    }
    return output;
}

// ============================================================
// maxpool2d
// ============================================================

torch::Tensor maxpool2d(
    torch::Tensor x, int kernel_size, int stride, int padding)
{
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    // -65504 = -(fp16 最大有限值)，fp16/bf16/fp32 均可表示（padding 位置语义安全）
    auto output = torch::full({N, C, H_out, W_out}, -6.5e4, x.options());
    if (x.scalar_type() == torch::kHalf) {
        launch_maxpool2d_half(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            N, C, H, W, H_out, W_out,
            kernel_size, stride, padding,
            _get_cuda_stream(x));
    } else if (x.scalar_type() == torch::kBFloat16) {
        launch_maxpool2d_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            N, C, H, W, H_out, W_out,
            kernel_size, stride, padding,
            _get_cuda_stream(x));
    } else {
        CHECK_FLOAT32(x);
        launch_maxpool2d(
            x.data_ptr<float>(), output.data_ptr<float>(),
            N, C, H, W, H_out, W_out,
            kernel_size, stride, padding,
            _get_cuda_stream(x));
    }
    return output;
}

// ============================================================
// avgpool2d
// ============================================================

torch::Tensor avgpool2d(
    torch::Tensor x, int kernel_size, int stride, int padding)
{
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({N, C, H_out, W_out}, x.options());
    if (x.scalar_type() == torch::kHalf) {
        launch_avgpool2d_half(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            N, C, H, W, H_out, W_out,
            kernel_size, stride, padding,
            _get_cuda_stream(x));
    } else if (x.scalar_type() == torch::kBFloat16) {
        launch_avgpool2d_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            N, C, H, W, H_out, W_out,
            kernel_size, stride, padding,
            _get_cuda_stream(x));
    } else {
        CHECK_FLOAT32(x);
        launch_avgpool2d(
            x.data_ptr<float>(), output.data_ptr<float>(),
            N, C, H, W, H_out, W_out,
            kernel_size, stride, padding,
            _get_cuda_stream(x));
    }
    return output;
}

// ============================================================
// im2col (Conv2D 前置展开)
// ============================================================

torch::Tensor im2col(
    torch::Tensor x, int KH, int KW, int stride, int padding)
{
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int H_out = (H + 2 * padding - KH) / stride + 1;
    int W_out = (W + 2 * padding - KW) / stride + 1;

    auto col = torch::zeros({N * H_out * W_out, C * KH * KW}, x.options());
    if (x.scalar_type() == torch::kHalf) {
        launch_im2col_half(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(col.data_ptr<at::Half>()),
            N, C, H, W, H_out, W_out, KH, KW, stride, padding,
            _get_cuda_stream(x));
    } else if (x.scalar_type() == torch::kBFloat16) {
        launch_im2col_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(col.data_ptr<at::BFloat16>()),
            N, C, H, W, H_out, W_out, KH, KW, stride, padding,
            _get_cuda_stream(x));
    } else {
        CHECK_FLOAT32(x);
        launch_im2col(
            x.data_ptr<float>(), col.data_ptr<float>(),
            N, C, H, W, H_out, W_out, KH, KW, stride, padding,
            _get_cuda_stream(x));
    }
    return col;
}

// ============================================================
// conv2d (im2col + tiled matmul + bias)
// ============================================================

torch::Tensor conv2d(
    torch::Tensor input, torch::Tensor weight,
    torch::Tensor bias, int stride, int padding)
{
    CHECK_CUDA(input); CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(input); CHECK_CONTIGUOUS(weight);
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
                "input and weight must have same dtype");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C_in, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (C_out, C_in, KH, KW)");

    int N = input.size(0), C_in = input.size(1), H = input.size(2), W = input.size(3);
    int C_out = weight.size(0), KH = weight.size(2), KW = weight.size(3);
    int H_out = (H + 2 * padding - KH) / stride + 1;
    int W_out = (W + 2 * padding - KW) / stride + 1;

    // im2col
    auto col = torch::zeros({N * H_out * W_out, C_in * KH * KW}, input.options());
    if (input.scalar_type() == torch::kHalf) {
        launch_im2col_half(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(col.data_ptr<at::Half>()),
            N, C_in, H, W, H_out, W_out, KH, KW, stride, padding,
            _get_cuda_stream(input));
    } else if (input.scalar_type() == torch::kBFloat16) {
        launch_im2col_bf16(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(col.data_ptr<at::BFloat16>()),
            N, C_in, H, W, H_out, W_out, KH, KW, stride, padding,
            _get_cuda_stream(input));
    } else {
        CHECK_FLOAT32(input); CHECK_FLOAT32(weight);
        launch_im2col(
            input.data_ptr<float>(), col.data_ptr<float>(),
            N, C_in, H, W, H_out, W_out, KH, KW, stride, padding,
            _get_cuda_stream(input));
    }

    // weight reshape + transpose: (C_out, C_in*KH*KW) -> (C_in*KH*KW, C_out)
    auto w_flat = weight.reshape({C_out, -1}).transpose(0, 1).contiguous();

    // matmul（fp16/bf16 走 WMMA TensorCore；fp32 走 tiled_auto）
    auto output = torch::empty({N * H_out * W_out, C_out}, input.options());
    if (input.scalar_type() == torch::kHalf) {
        output = ::matmul_half(col, w_flat);
    } else if (input.scalar_type() == torch::kBFloat16) {
        output = ::matmul_bf16(col, w_flat);
    } else {
        launch_matmul_tiled_auto(
            col.data_ptr<float>(), w_flat.data_ptr<float>(), output.data_ptr<float>(),
            N * H_out * W_out, C_in * KH * KW, C_out,
            _get_cuda_stream(input));
    }

    // add bias（字段对齐：output (M, C_out) + bias (C_out,)）
    if (bias.numel() > 0) {
        output = ::bias_add(output, bias);
    }

    // reshape to (N, C_out, H_out, W_out)
    output = output.reshape({N, H_out * W_out, C_out})
                 .permute({0, 2, 1})
                 .reshape({N, C_out, H_out, W_out})
                 .contiguous();
    return output;
}

// ============================================================
// 注册模块
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_naive", &matmul_naive, "Naive CUDA matmul C = A @ B");
    m.def("matmul_tiled", &matmul_tiled,
          "Tiled CUDA matmul C = A @ B",
          py::arg("A"), py::arg("B"),
          py::arg("BLOCK_M") = 16, py::arg("BLOCK_N") = 16, py::arg("BLOCK_K") = 16);
    m.def("matmul_half", &matmul_half, "fp16 TensorCore matmul (fp16 in -> fp32 acc -> fp16 out)");
    m.def("matmul_half_pair", &matmul_half_pair, "fused gate+up: (A@B1, A@B2) shared-A fp16");
    m.def("matmul_bf16_pair", &matmul_bf16_pair, "fused gate+up: (A@B1, A@B2) shared-A bf16");
    m.def("matmul_half_pair_swiglu", &matmul_half_pair_swiglu, "hidden = SiLU(X@Wg)*(X@Wu), shared-A + swiglu epilogue fp16");
    m.def("matmul_bf16_pair_swiglu", &matmul_bf16_pair_swiglu, "hidden = SiLU(X@Wg)*(X@Wu), shared-A + swiglu epilogue bf16");
    m.def("matmul_bf16", &matmul_bf16, "bf16 TensorCore matmul (bf16 in -> fp32 acc -> bf16 out)");
    m.def("matmul_tiled_auto", &matmul_tiled_auto,
          "Auto-tiled CUDA matmul C = A @ B (adaptive block sizes)");
    m.def("matmul_transB", &matmul_transB,
          "CUDA matmul C = A @ B^T (no transpose needed)");
    m.def("matmul_transA", &matmul_transA,
          "CUDA matmul C = A^T @ B (no transpose needed)");
    m.def("bias_add", &bias_add, "CUDA bias add: Y = X + bias");

    m.def("gelu", &gelu, "CUDA GELU activation");
    m.def("relu", &relu, "CUDA ReLU activation");
    m.def("silu", &silu, "CUDA SiLU activation");

    m.def("gelu_backward", &gelu_backward, "CUDA GELU backward");
    m.def("relu_backward", &relu_backward, "CUDA ReLU backward");
    m.def("silu_backward", &silu_backward, "CUDA SiLU backward");

    m.def("gelu_backward_vec4", &gelu_backward_vec4, "CUDA GELU backward (float4)");
    m.def("relu_backward_vec4", &relu_backward_vec4, "CUDA ReLU backward (float4)");
    m.def("silu_backward_vec4", &silu_backward_vec4, "CUDA SiLU backward (float4)");

    m.def("mlp_fused_first_layer", &mlp_fused_first_layer,
          "Fused MLP first layer: H = GELU(X @ W1 + bias)");
    m.def("swiglu_fused", &swiglu_fused,
          "Fused SwiGLU: output = SiLU(gate) * up");

    m.def("layernorm_forward", &layernorm_forward,
          "CUDA LayerNorm forward: returns (y, mean, rstd)",
          py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5);
    m.def("layernorm_backward", &layernorm_backward,
          "CUDA LayerNorm backward: returns (dx, d_gamma, d_beta)");

    m.def("softmax", &softmax, "CUDA row-wise softmax: (M, N) -> (M, N)");
    m.def("maxpool2d", &maxpool2d, "CUDA MaxPool2D",
          py::arg("x"), py::arg("kernel_size"),
          py::arg("stride"), py::arg("padding") = 0);
    m.def("avgpool2d", &avgpool2d, "CUDA AvgPool2D",
          py::arg("x"), py::arg("kernel_size"),
          py::arg("stride"), py::arg("padding") = 0);
    m.def("im2col", &im2col, "CUDA im2col for Conv2D",
          py::arg("x"), py::arg("KH"), py::arg("KW"),
          py::arg("stride"), py::arg("padding") = 0);
    m.def("conv2d", &conv2d, "CUDA Conv2D (im2col + tiled matmul)",
          py::arg("input"), py::arg("weight"),
          py::arg("bias") = torch::Tensor(),
          py::arg("stride") = 1, py::arg("padding") = 0);
}