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
#include <c10/cuda/CUDAStream.h>

// 兼容不同 PyTorch 版本的 CUDA stream 获取
static cudaStream_t _get_cuda_stream(torch::Tensor t) {
    return c10::cuda::getCurrentCUDAStream(t.device().index()).stream();
}

// Forward 声明 CUDA kernel launch 函数 (定义在 mlp_cuda_kernels.cu)
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

void launch_gelu(const float* input, float* output, int n, cudaStream_t stream);
void launch_relu(const float* input, float* output, int n, cudaStream_t stream);
void launch_silu(const float* input, float* output, int n, cudaStream_t stream);

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

void launch_matmul_transB(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream);

void launch_matmul_transA(
    const float* A, const float* B, float* C,
    int M, int K, int N, cudaStream_t stream);

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

// ============================================================
// bias_add
// ============================================================

torch::Tensor bias_add(torch::Tensor x, torch::Tensor bias) {
    CHECK_CUDA(x); CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(bias);
    CHECK_FLOAT32(x); CHECK_FLOAT32(bias);

    TORCH_CHECK(x.dim() == 2, "x must be 2D (M, N)");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == x.size(1),
                "bias must be 1D with shape (N,)");

    int M = x.size(0), N = x.size(1);
    auto output = torch::empty({M, N}, x.options());
    launch_bias_add(
        x.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        M, N, _get_cuda_stream(x));
    return output;
}

// ============================================================
// 激活函数 (elementwise, 支持任意形状)
// ============================================================

// --- GELU ---
torch::Tensor gelu(torch::Tensor x) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x);
    auto output = torch::empty_like(x);
    launch_gelu(x.data_ptr<float>(), output.data_ptr<float>(), x.numel(),
                _get_cuda_stream(x));
    return output;
}

// --- ReLU ---
torch::Tensor relu(torch::Tensor x) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x);
    auto output = torch::empty_like(x);
    launch_relu(x.data_ptr<float>(), output.data_ptr<float>(), x.numel(),
                _get_cuda_stream(x));
    return output;
}

// --- SiLU ---
torch::Tensor silu(torch::Tensor x) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x);
    auto output = torch::empty_like(x);
    launch_silu(x.data_ptr<float>(), output.data_ptr<float>(), x.numel(),
                _get_cuda_stream(x));
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
    CHECK_FLOAT32(X); CHECK_FLOAT32(W1); CHECK_FLOAT32(bias);

    TORCH_CHECK(X.dim() == 2 && W1.dim() == 2, "X and W1 must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(X.size(1) == W1.size(0), "X(", X.size(1), ") != W1 row(", W1.size(0), ")");
    TORCH_CHECK(bias.size(0) == W1.size(1), "bias(", bias.size(0), ") != W1 col(", W1.size(1), ")");

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
    CHECK_FLOAT32(gate); CHECK_FLOAT32(up);

    TORCH_CHECK(gate.sizes() == up.sizes(),
                "gate and up must have same shape; got ", gate.sizes(), " vs ", up.sizes());

    auto output = torch::empty_like(gate);
    launch_swiglu_fused(
        gate.data_ptr<float>(), up.data_ptr<float>(), output.data_ptr<float>(),
        gate.numel(), _get_cuda_stream(gate));
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
// 注册模块
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_naive", &matmul_naive, "Naive CUDA matmul C = A @ B");
    m.def("matmul_tiled", &matmul_tiled,
          "Tiled CUDA matmul C = A @ B",
          py::arg("A"), py::arg("B"),
          py::arg("BLOCK_M") = 16, py::arg("BLOCK_N") = 16, py::arg("BLOCK_K") = 16);
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
}
