/**
 * CUDA kernel C++ 单元测试
 *
 * 直接调用 CUDA kernel 函数，与 CPU 参考实现对比验证正确性。
 * 编译: make test-cuda
 * 运行: ./build/test_kernels
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>
#include <random>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 被测函数:本文件内联独立 kernel 实现验证核心算法正确性,
// 不依赖 kernels/mlp/*.cu 链接(那些 launch 函数走 PyTorch extension 路径)。

// ============================================================
// 工具函数
// ============================================================

static int g_pass = 0;
static int g_fail = 0;

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                 \
    do {                                                                        \
        double _a = (double)(a), _b = (double)(b);                             \
        double _diff = fabs(_a - _b);                                          \
        if (_diff > (tol)) {                                                    \
            fprintf(stderr, "  FAIL: %.6f vs %.6f (diff=%.6e, tol=%.6e)\n",    \
                    _a, _b, _diff, (double)(tol));                              \
            return false;                                                       \
        }                                                                       \
    } while (0)

#define TEST_CASE(name)                                                        \
    static bool test_##name();                                                 \
    static struct Test_##name {                                                 \
        Test_##name() {                                                         \
            tests.push_back({#name, test_##name});                              \
        }                                                                       \
    } reg_##name;                                                               \
    static bool test_##name()

struct TestEntry {
    const char* name;
    bool (*fn)();
};
static std::vector<TestEntry> tests;

// CPU 参考 GELU (tanh 近似)
static float ref_gelu_tanh(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
    float tanh_inner = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_inner);
}

// CPU 参考 SiLU
static float ref_silu(float x) {
    return x / (1.0f + expf(-x));
}

// CPU 参考 ReLU
static float ref_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// ============================================================
// CUDA kernel：elementwise activation
// ============================================================

__global__ void relu_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
    }
}

__global__ void gelu_tanh_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float inner = sqrt_2_over_pi * (v + 0.044715f * v * v * v);
        float tanh_inner = tanhf(inner);
        out[idx] = 0.5f * v * (1.0f + tanh_inner);
    }
}

__global__ void silu_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        out[idx] = v / (1.0f + expf(-v));
    }
}

// ============================================================
// CUDA kernel：bias_add
// ============================================================

__global__ void bias_add_kernel(const float* x, const float* b,
                                 float* out, int rows, int cols) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        out[row * cols + col] = x[row * cols + col] + b[col];
    }
}

// ============================================================
// CUDA kernel：layernorm forward
// ============================================================

__global__ void layernorm_forward_kernel(const float* x, const float* gamma,
                                          const float* beta, float* y,
                                          float* mean_out, float* rstd_out,
                                          int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * cols;
    float* y_row = y + row * cols;

    // warp reduce 求 mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += x_row[i];
    }

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum);
    __syncthreads();

    float mean = s_sum / cols;

    // warp reduce 求 var
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float d = x_row[i] - mean;
        var_sum += d * d;
    }

    __shared__ float s_var;
    if (threadIdx.x == 0) s_var = 0.0f;
    __syncthreads();
    atomicAdd(&s_var, var_sum);
    __syncthreads();

    float var = s_var / cols;
    float rstd = rsqrtf(var + eps);

    if (threadIdx.x == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float x_hat = (x_row[i] - mean) * rstd;
        y_row[i] = gamma[i] * x_hat + beta[i];
    }
}

// ============================================================
// 测试用例
// ============================================================

TEST_CASE(relu_forward) {
    const int N = 1024;
    std::vector<float> h_x(N), h_ref(N);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; i++) {
        h_x[i] = dist(rng);
        h_ref[i] = ref_relu(h_x[i]);
    }

    float *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    relu_kernel<<<(N + 255) / 256, 256>>>(d_x, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        max_err = fmaxf(max_err, fabsf(h_out[i] - h_ref[i]));
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));

    ASSERT_NEAR(max_err, 0.0f, 1e-6f);
    return true;
}

TEST_CASE(gelu_tanh_forward) {
    const int N = 1024;
    std::vector<float> h_x(N), h_ref(N);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; i++) {
        h_x[i] = dist(rng);
        h_ref[i] = ref_gelu_tanh(h_x[i]);
    }

    float *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    gelu_tanh_kernel<<<(N + 255) / 256, 256>>>(d_x, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        max_err = fmaxf(max_err, fabsf(h_out[i] - h_ref[i]));
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));

    ASSERT_NEAR(max_err, 0.0f, 1e-5f);
    return true;
}

TEST_CASE(silu_forward) {
    const int N = 1024;
    std::vector<float> h_x(N), h_ref(N);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; i++) {
        h_x[i] = dist(rng);
        h_ref[i] = ref_silu(h_x[i]);
    }

    float *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    silu_kernel<<<(N + 255) / 256, 256>>>(d_x, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        max_err = fmaxf(max_err, fabsf(h_out[i] - h_ref[i]));
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));

    ASSERT_NEAR(max_err, 0.0f, 1e-5f);
    return true;
}

TEST_CASE(bias_add) {
    const int ROWS = 64, COLS = 128;
    std::vector<float> h_x(ROWS * COLS), h_b(COLS), h_ref(ROWS * COLS);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < ROWS * COLS; i++) h_x[i] = dist(rng);
    for (int i = 0; i < COLS; i++) h_b[i] = dist(rng);
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            h_ref[r * COLS + c] = h_x[r * COLS + c] + h_b[c];

    float *d_x, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), COLS * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid((COLS + 127) / 128, ROWS);
    bias_add_kernel<<<grid, 128>>>(d_x, d_b, d_out, ROWS, COLS);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(ROWS * COLS);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < ROWS * COLS; i++) {
        max_err = fmaxf(max_err, fabsf(h_out[i] - h_ref[i]));
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));

    ASSERT_NEAR(max_err, 0.0f, 1e-5f);
    return true;
}

TEST_CASE(layernorm_forward) {
    const int ROWS = 8, COLS = 256;
    std::vector<float> h_x(ROWS * COLS), h_gamma(COLS, 1.0f), h_beta(COLS, 0.0f);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < ROWS * COLS; i++) h_x[i] = dist(rng);

    // CPU 参考
    std::vector<float> h_ref(ROWS * COLS);
    for (int r = 0; r < ROWS; r++) {
        const float* row = h_x.data() + r * COLS;
        float* ref_row = h_ref.data() + r * COLS;
        float mean = 0.0f;
        for (int c = 0; c < COLS; c++) mean += row[c];
        mean /= COLS;
        float var = 0.0f;
        for (int c = 0; c < COLS; c++) {
            float d = row[c] - mean;
            var += d * d;
        }
        var /= COLS;
        float rstd = 1.0f / sqrtf(var + 1e-5f);
        for (int c = 0; c < COLS; c++) {
            float x_hat = (row[c] - mean) * rstd;
            ref_row[c] = h_gamma[c] * x_hat + h_beta[c];
        }
    }

    float *d_x, *d_gamma, *d_beta, *d_y, *d_mean, *d_rstd;
    CUDA_CHECK(cudaMalloc(&d_x, ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, ROWS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rstd, ROWS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), COLS * sizeof(float), cudaMemcpyHostToDevice));

    layernorm_forward_kernel<<<ROWS, 256>>>(d_x, d_gamma, d_beta, d_y,
                                             d_mean, d_rstd, ROWS, COLS, 1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(ROWS * COLS);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < ROWS * COLS; i++) {
        max_err = fmaxf(max_err, fabsf(h_out[i] - h_ref[i]));
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_rstd));

    ASSERT_NEAR(max_err, 0.0f, 1e-3f);
    return true;
}

TEST_CASE(layernorm_learnable) {
    const int ROWS = 4, COLS = 64;
    std::vector<float> h_x(ROWS * COLS), h_gamma(COLS), h_beta(COLS);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < ROWS * COLS; i++) h_x[i] = dist(rng);
    for (int i = 0; i < COLS; i++) {
        h_gamma[i] = dist(rng) * 0.5f + 1.0f;
        h_beta[i] = dist(rng) * 0.1f;
    }

    // CPU 参考
    std::vector<float> h_ref(ROWS * COLS);
    for (int r = 0; r < ROWS; r++) {
        const float* row = h_x.data() + r * COLS;
        float* ref_row = h_ref.data() + r * COLS;
        float mean = 0.0f;
        for (int c = 0; c < COLS; c++) mean += row[c];
        mean /= COLS;
        float var = 0.0f;
        for (int c = 0; c < COLS; c++) {
            float d = row[c] - mean;
            var += d * d;
        }
        var /= COLS;
        float rstd = 1.0f / sqrtf(var + 1e-5f);
        for (int c = 0; c < COLS; c++) {
            float x_hat = (row[c] - mean) * rstd;
            ref_row[c] = h_gamma[c] * x_hat + h_beta[c];
        }
    }

    float *d_x, *d_gamma, *d_beta, *d_y, *d_mean, *d_rstd;
    CUDA_CHECK(cudaMalloc(&d_x, ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, ROWS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rstd, ROWS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), COLS * sizeof(float), cudaMemcpyHostToDevice));

    layernorm_forward_kernel<<<ROWS, 64>>>(d_x, d_gamma, d_beta, d_y,
                                             d_mean, d_rstd, ROWS, COLS, 1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(ROWS * COLS);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < ROWS * COLS; i++) {
        max_err = fmaxf(max_err, fabsf(h_out[i] - h_ref[i]));
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_rstd));

    ASSERT_NEAR(max_err, 0.0f, 1e-3f);
    return true;
}

// ============================================================
// main
// ============================================================

int main() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s\n\n", prop.name);

    printf("Running %zu tests...\n\n", tests.size());

    for (auto& t : tests) {
        printf("[ RUN  ] %s\n", t.name);
        bool ok = t.fn();
        if (ok) {
            printf("[ PASS ] %s\n", t.name);
            g_pass++;
        } else {
            printf("[ FAIL ] %s\n", t.name);
            g_fail++;
        }
    }

    printf("\n%d passed, %d failed, %d total\n",
           g_pass, g_fail, g_pass + g_fail);

    return g_fail > 0 ? 1 : 0;
}
