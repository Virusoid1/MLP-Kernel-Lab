// Day 3: PyTorch C++ Extension binding
// 让 CUDA kernel 能被 Python 调用
//
// 编译: python setup.py install
// 使用: import mlp_cuda; out = mlp_cuda.matmul_naive(x, w)

#include <torch/extension.h>
#include <cuda_runtime.h>

// 声明 CUDA kernel 函数 (定义在 .cu 文件中)
// matmul_naive
void launch_matmul_naive(
    const float* A, const float* B, float* C,
    int M, int K, int N, cudaStream_t stream);

// matmul_tiled
void launch_matmul_tiled(
    const float* A, const float* B, float* C,
    int M, int K, int N, int BLOCK_M, int BLOCK_N, int BLOCK_K,
    cudaStream_t stream);

// mlp_fused_first_layer
void launch_mlp_fused_first_layer(
    const float* X, const float* W1, const float* bias, float* H,
    int M, int K, int N, cudaStream_t stream);

// swiglu_fused
void launch_swiglu_fused(
    const float* gate, const float* up, float* output,
    int total_elements, cudaStream_t stream);

// ---- Python 可调用的函数 ----

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B) {
    // TODO: 实现 binding
    // 提示:
    //   TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA tensors");
    //   TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "inputs must be contiguous");
    //   TORCH_CHECK(A.dtype() == torch::kFloat32, "only float32 supported");
    //   int M = A.size(0), K = A.size(1), N = B.size(1);
    //   auto C = torch::empty({M, N}, A.options());
    //   launch_matmul_naive(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
    //                       M, K, N, at::cuda::getCurrentCUDAStream());
    //   return C;
    TORCH_CHECK(false, "matmul_naive binding not implemented yet");
    return A;
}

torch::Tensor matmul_tiled(
    torch::Tensor A, torch::Tensor B,
    int BLOCK_M = 16, int BLOCK_N = 16, int BLOCK_K = 16
) {
    // TODO: 实现 binding (同上模式, 额外传 BLOCK 参数)
    TORCH_CHECK(false, "matmul_tiled binding not implemented yet");
    return A;
}

torch::Tensor mlp_fused_first_layer(
    torch::Tensor X, torch::Tensor W1, torch::Tensor bias
) {
    // TODO: 实现 binding
    TORCH_CHECK(false, "mlp_fused_first_layer binding not implemented yet");
    return X;
}

torch::Tensor swiglu_fused(torch::Tensor gate, torch::Tensor up) {
    // TODO: 实现 binding
    TORCH_CHECK(false, "swiglu_fused binding not implemented yet");
    return gate;
}

// 注册模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_naive", &matmul_naive, "Naive CUDA matmul");
    m.def("matmul_tiled", &matmul_tiled, "Tiled CUDA matmul",
          py::arg("A"), py::arg("B"),
          py::arg("BLOCK_M") = 16, py::arg("BLOCK_N") = 16, py::arg("BLOCK_K") = 16);
    m.def("mlp_fused_first_layer", &mlp_fused_first_layer, "Fused MLP first layer (matmul+bias+GELU)");
    m.def("swiglu_fused", &swiglu_fused, "Fused SwiGLU kernel");
}
