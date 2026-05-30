/**
 * Naive C++ 实现：完整 MLP 训练流程 + CNN 算子
 *
 * 纯 CPU 实现，不依赖任何深度学习库，用于理解深度学习的数学原理。
 *
 * 包含:
 *   Part 1: 基础数据结构 (Matrix 2D, Tensor 4D)
 *   Part 2: MLP 算子 — Linear, ReLU, Sigmoid, Softmax, CrossEntropy (含反向传播)
 *   Part 3: MLP 模型 + 训练循环
 *   Part 4: CNN 算子 — Conv2D, MaxPool2D, AvgPool2D
 *   Part 5: 演示与正确性验证
 *
 * 编译: g++ -std=c++17 -O2 -o naive_ops naive/operators.cpp
 * 运行: ./naive_ops
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

// ============================================================
// Part 1: 基础数据结构
// ============================================================

// ---------- 2D 矩阵 (MLP 核心数据结构) ----------
// row-major: data[i * cols + j]

struct Matrix {
    int rows, cols;
    std::vector<float> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}

    int size() const { return rows * cols; }

    float at(int i, int j) const { return data[i * cols + j]; }
    float& at(int i, int j) { return data[i * cols + j]; }

    // 随机初始化 (Xavier/Glorot)
    void xavier_init(int fan_in, int fan_out) {
        float sd = std::sqrt(2.0f / static_cast<float>(fan_in + fan_out));
        for (auto& v : data) {
            float u1 = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 0.998f + 0.001f;
            float u2 = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 0.998f + 0.001f;
            v = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530f * u2) * sd;
        }
    }

    void zero() { std::fill(data.begin(), data.end(), 0.0f); }

    void print(const char* name, int max_rows = -1, int max_cols = -1) const {
        int mr = (max_rows < 0) ? rows : std::min(rows, max_rows);
        int mc = (max_cols < 0) ? cols : std::min(cols, max_cols);
        printf("  %s (%dx%d):\n", name, rows, cols);
        for (int i = 0; i < mr; ++i) {
            printf("    ");
            for (int j = 0; j < mc; ++j) {
                printf("%8.4f", at(i, j));
            }
            if (mc < cols) printf("  ...");
            printf("\n");
        }
        if (mr < rows) printf("    ... (%d more rows)\n", rows - mr);
    }
};

// ---------- 4D 张量 (CNN 数据结构, NCHW) ----------

struct Tensor {
    int n, c, h, w;
    std::vector<float> data;

    Tensor() : n(0), c(0), h(0), w(0) {}
    Tensor(int n_, int c_, int h_, int w_)
        : n(n_), c(c_), h(h_), w(w_),
          data(n_ * c_ * h_ * w_, 0.0f) {}

    int size() const { return n * c * h * w; }
    int offset(int ni, int ci, int hi, int wi) const {
        return ((ni * c + ci) * h + hi) * w + wi;
    }
    float at(int ni, int ci, int hi, int wi) const {
        return data[offset(ni, ci, hi, wi)];
    }
    float& at(int ni, int ci, int hi, int wi) {
        return data[offset(ni, ci, hi, wi)];
    }
    void fill_random() {
        for (auto& v : data)
            v = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    }
    void print_shape(const char* name) const {
        printf("  %s: (%d, %d, %d, %d)\n", name, n, c, h, w);
    }
};

// ============================================================
// Part 2: MLP 算子 (前向 + 反向)
// ============================================================

// ---------- 矩阵乘法: C = A @ B ----------
// A: (M, K), B: (K, N), C: (M, N)

Matrix matmul(const Matrix& a, const Matrix& b) {
    Matrix c(a.rows, b.cols);
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < b.cols; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < a.cols; ++p) {
                sum += a.at(i, p) * b.at(p, j);
            }
            c.at(i, j) = sum;
        }
    }
    return c;
}

// 矩阵转置: B = A^T
Matrix transpose(const Matrix& a) {
    Matrix b(a.cols, a.rows);
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            b.at(j, i) = a.at(i, j);
    return b;
}

// ---------- ReLU ----------
// 前向: y = max(0, x)
Matrix relu_forward(const Matrix& x) {
    Matrix y(x.rows, x.cols);
    for (int i = 0; i < x.size(); ++i)
        y.data[i] = std::max(0.0f, x.data[i]);
    return y;
}

// 反向: dx = dy * (x > 0)
Matrix relu_backward(const Matrix& dy, const Matrix& x) {
    Matrix dx(dy.rows, dy.cols);
    for (int i = 0; i < dy.size(); ++i)
        dx.data[i] = dy.data[i] * (x.data[i] > 0.0f ? 1.0f : 0.0f);
    return dx;
}

// ---------- Sigmoid ----------
// 前向: y = 1 / (1 + exp(-x))
Matrix sigmoid_forward(const Matrix& x) {
    Matrix y(x.rows, x.cols);
    for (int i = 0; i < x.size(); ++i)
        y.data[i] = 1.0f / (1.0f + std::exp(-x.data[i]));
    return y;
}

// 反向: dx = dy * y * (1 - y)
Matrix sigmoid_backward(const Matrix& dy, const Matrix& y) {
    Matrix dx(dy.rows, dy.cols);
    for (int i = 0; i < dy.size(); ++i)
        dx.data[i] = dy.data[i] * y.data[i] * (1.0f - y.data[i]);
    return dx;
}

// ---------- Softmax (逐行, 数值稳定) ----------
// softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))

Matrix softmax_forward(const Matrix& x) {
    Matrix y(x.rows, x.cols);
    for (int i = 0; i < x.rows; ++i) {
        // 找行最大值
        float max_val = x.at(i, 0);
        for (int j = 1; j < x.cols; ++j)
            max_val = std::max(max_val, x.at(i, j));

        // exp + sum
        float sum = 0.0f;
        for (int j = 0; j < x.cols; ++j) {
            y.at(i, j) = std::exp(x.at(i, j) - max_val);
            sum += y.at(i, j);
        }

        // 归一化
        for (int j = 0; j < x.cols; ++j)
            y.at(i, j) /= sum;
    }
    return y;
}

// ---------- Cross-Entropy Loss ----------
// loss = -mean(log(softmax_prob[label]))
// 输入: probs (B, C) softmax 输出, labels (B) 整数标签

float cross_entropy_loss(const Matrix& probs, const std::vector<int>& labels) {
    float loss = 0.0f;
    for (int i = 0; i < probs.rows; ++i) {
        float p = probs.at(i, labels[i]);
        p = std::max(p, 1e-7f);  // 防止 log(0)
        loss -= std::log(p);
    }
    return loss / probs.rows;
}

// Cross-Entropy + Softmax 反向传播的合并梯度:
// d_logits = (softmax_prob - one_hot(label)) / batch_size
Matrix softmax_cross_entropy_backward(const Matrix& probs, const std::vector<int>& labels) {
    Matrix dx(probs.rows, probs.cols);
    for (int i = 0; i < probs.rows; ++i) {
        for (int j = 0; j < probs.cols; ++j) {
            dx.at(i, j) = probs.at(i, j);
        }
        dx.at(i, labels[i]) -= 1.0f;
    }
    float scale = 1.0f / probs.rows;
    for (auto& v : dx.data) v *= scale;
    return dx;
}

// ---------- MSE Loss (用于回归任务如 XOR) ----------
// loss = mean((pred - target)^2)

float mse_loss(const Matrix& pred, const Matrix& target) {
    float sum = 0.0f;
    for (int i = 0; i < pred.size(); ++i) {
        float d = pred.data[i] - target.data[i];
        sum += d * d;
    }
    return sum / pred.rows;
}

// MSE 反向: dpred = 2 * (pred - target) / batch_size
Matrix mse_backward(const Matrix& pred, const Matrix& target) {
    Matrix dx(pred.rows, pred.cols);
    float scale = 2.0f / pred.rows;
    for (int i = 0; i < pred.size(); ++i)
        dx.data[i] = scale * (pred.data[i] - target.data[i]);
    return dx;
}

// ============================================================
// Part 3: MLP 模型
//
// 架构: Linear → ReLU → Linear → ReLU → ... → Linear → Output
// 支持分类 (softmax + cross_entropy) 和回归 (sigmoid + mse)
// ============================================================

// Linear 层: Y = X @ W + b
// 参数: W (in_dim, out_dim), b (out_dim)
struct LinearLayer {
    Matrix weight;    // (in_dim, out_dim)
    Matrix bias;      // (1, out_dim)

    // 梯度
    Matrix grad_weight;
    Matrix grad_bias;

    // 缓存 (反向传播需要)
    Matrix input_cache;

    LinearLayer() = default;
    LinearLayer(int in_dim, int out_dim)
        : weight(in_dim, out_dim), bias(1, out_dim),
          grad_weight(in_dim, out_dim), grad_bias(1, out_dim)
    {
        weight.xavier_init(in_dim, out_dim);
        bias.zero();
    }

    // 前向: Y = X @ W + b
    Matrix forward(const Matrix& x) {
        input_cache = x;  // 保存输入用于反向传播
        Matrix y = matmul(x, weight);
        // 加 bias (广播到每行)
        for (int i = 0; i < y.rows; ++i)
            for (int j = 0; j < y.cols; ++j)
                y.at(i, j) += bias.at(0, j);
        return y;
    }

    // 反向: 给定 dY, 计算 dX 并累积 dW, db
    Matrix backward(const Matrix& dy) {
        // dW = X^T @ dY
        Matrix xt = transpose(input_cache);
        Matrix dw = matmul(xt, dy);
        for (int i = 0; i < grad_weight.size(); ++i)
            grad_weight.data[i] += dw.data[i];

        // db = sum(dY, axis=0)
        for (int j = 0; j < dy.cols; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < dy.rows; ++i)
                sum += dy.at(i, j);
            grad_bias.at(0, j) += sum;
        }

        // dX = dY @ W^T
        Matrix wt = transpose(weight);
        return matmul(dy, wt);
    }

    // SGD 参数更新
    void update(float lr) {
        for (int i = 0; i < weight.size(); ++i)
            weight.data[i] -= lr * grad_weight.data[i];
        for (int i = 0; i < bias.size(); ++i)
            bias.data[i] -= lr * grad_bias.data[i];
    }

    void zero_grad() {
        grad_weight.zero();
        grad_bias.zero();
    }
};

// MLP 模型
enum class OutputMode { CLASSIFICATION, REGRESSION };

struct MLP {
    std::vector<LinearLayer> layers;
    std::vector<Matrix> activations;   // 缓存每层激活值 (反向传播需要)
    OutputMode mode;
    int num_layers;

    // hidden_sizes: 各隐藏层大小, 例如 {64, 32}
    MLP(int input_dim, const std::vector<int>& hidden_sizes, int output_dim,
        OutputMode m = OutputMode::CLASSIFICATION)
        : mode(m)
    {
        std::vector<int> dims = {input_dim};
        dims.insert(dims.end(), hidden_sizes.begin(), hidden_sizes.end());
        dims.push_back(output_dim);
        num_layers = static_cast<int>(dims.size()) - 1;

        layers.reserve(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            layers.emplace_back(dims[i], dims[i + 1]);
        }
    }

    // 前向传播
    Matrix forward(const Matrix& x) {
        activations.clear();
        Matrix out = x;

        for (int i = 0; i < num_layers - 1; ++i) {
            // 隐藏层: Linear → ReLU
            out = layers[i].forward(out);
            activations.push_back(out);       // 保存线性输出 (ReLU 反向需要)
            out = relu_forward(out);
            activations.push_back(out);       // 保存 ReLU 输出
        }

        // 输出层: Linear → (softmax 或 sigmoid)
        out = layers[num_layers - 1].forward(out);
        activations.push_back(out);

        if (mode == OutputMode::CLASSIFICATION) {
            out = softmax_forward(out);
        } else {
            out = sigmoid_forward(out);
        }
        activations.push_back(out);
        return out;
    }

    // 反向传播
    void backward(const Matrix& d_output) {
        Matrix grad = d_output;

        // 输出层反向
        grad = layers[num_layers - 1].backward(grad);

        // 隐藏层反向 (从倒数第一层到第 0 层)
        for (int i = num_layers - 2; i >= 0; --i) {
            // activations 的布局: [linear0, relu0, linear1, relu1, ..., linear_out, output]
            // 索引: linear_i = i*2, relu_i = i*2+1
            int relu_idx = i * 2 + 1;
            int linear_idx = i * 2;

            // ReLU 反向
            grad = relu_backward(grad, activations[linear_idx]);
            // Linear 反向
            grad = layers[i].backward(grad);
        }
    }

    // SGD 更新
    void update(float lr) {
        for (auto& layer : layers)
            layer.update(lr);
    }

    void zero_grad() {
        for (auto& layer : layers)
            layer.zero_grad();
    }

    // 训练一步 (分类, 交叉熵): 前向 → loss → 反向 → 更新
    float train_step(const Matrix& x, const std::vector<int>& labels, float lr) {
        zero_grad();
        Matrix output = forward(x);
        float loss = cross_entropy_loss(output, labels);
        Matrix d_output = softmax_cross_entropy_backward(output, labels);
        backward(d_output);
        update(lr);
        return loss;
    }
};

// ============================================================
// Part 4: CNN 算子 (NCHW)
// ============================================================

// im2col: 输入 patch → 二维矩阵
static std::vector<float> im2col(
    const Tensor& x, int kh, int kw,
    int stride, int pad, int h_out, int w_out)
{
    int rows = x.n * h_out * w_out;
    int cols = x.c * kh * kw;
    std::vector<float> col(rows * cols, 0.0f);

    for (int ni = 0; ni < x.n; ++ni) {
        for (int ho = 0; ho < h_out; ++ho) {
            for (int wo = 0; wo < w_out; ++wo) {
                int row = (ni * h_out + ho) * w_out + wo;
                for (int ci = 0; ci < x.c; ++ci) {
                    for (int r = 0; r < kh; ++r) {
                        for (int s = 0; s < kw; ++s) {
                            int hi = ho * stride - pad + r;
                            int wi = wo * stride - pad + s;
                            int col_idx = (ci * kh + r) * kw + s;
                            if (hi >= 0 && hi < x.h && wi >= 0 && wi < x.w)
                                col[row * cols + col_idx] = x.at(ni, ci, hi, wi);
                        }
                    }
                }
            }
        }
    }
    return col;
}

// Conv2D: im2col + matmul
Tensor conv2d(const Tensor& input, const Tensor& weight,
              const std::vector<float>& bias, int stride, int pad)
{
    int c_out = weight.n, kh = weight.h, kw = weight.w;
    int h_out = (input.h + 2 * pad - kh) / stride + 1;
    int w_out = (input.w + 2 * pad - kw) / stride + 1;

    auto col = im2col(input, kh, kw, stride, pad, h_out, w_out);
    int m = input.n * h_out * w_out;
    int k = input.c * kh * kw;
    int n = c_out;

    // weight 展平转置
    std::vector<float> w_flat(k * n);
    for (int co = 0; co < c_out; ++co)
        for (int ci = 0; ci < input.c; ++ci)
            for (int r = 0; r < kh; ++r)
                for (int s = 0; s < kw; ++s)
                    w_flat[((ci * kh + r) * kw + s) * n + co] = weight.data[weight.offset(co, ci, r, s)];

    // 朴素 matmul
    std::vector<float> out_flat(m * n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p)
                sum += col[i * k + p] * w_flat[p * n + j];
            out_flat[i * n + j] = sum + bias[j];
        }

    Tensor output(input.n, c_out, h_out, w_out);
    for (int ni = 0; ni < input.n; ++ni)
        for (int ho = 0; ho < h_out; ++ho)
            for (int wo = 0; wo < w_out; ++wo)
                for (int co = 0; co < c_out; ++co)
                    output.at(ni, co, ho, wo) = out_flat[((ni * h_out + ho) * w_out + wo) * n + co];
    return output;
}

// MaxPool2D
Tensor maxpool2d(const Tensor& input, int kernel_size, int stride, int pad) {
    int h_out = (input.h + 2 * pad - kernel_size) / stride + 1;
    int w_out = (input.w + 2 * pad - kernel_size) / stride + 1;
    Tensor output(input.n, input.c, h_out, w_out);
    for (int ni = 0; ni < input.n; ++ni)
        for (int ci = 0; ci < input.c; ++ci)
            for (int ho = 0; ho < h_out; ++ho)
                for (int wo = 0; wo < w_out; ++wo) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int kh = 0; kh < kernel_size; ++kh)
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int hi = ho * stride - pad + kh;
                            int wi = wo * stride - pad + kw;
                            if (hi >= 0 && hi < input.h && wi >= 0 && wi < input.w)
                                max_val = std::max(max_val, input.at(ni, ci, hi, wi));
                        }
                    output.at(ni, ci, ho, wo) = max_val;
                }
    return output;
}

// AvgPool2D
Tensor avgpool2d(const Tensor& input, int kernel_size, int stride, int pad) {
    int h_out = (input.h + 2 * pad - kernel_size) / stride + 1;
    int w_out = (input.w + 2 * pad - kernel_size) / stride + 1;
    Tensor output(input.n, input.c, h_out, w_out);
    for (int ni = 0; ni < input.n; ++ni)
        for (int ci = 0; ci < input.c; ++ci)
            for (int ho = 0; ho < h_out; ++ho)
                for (int wo = 0; wo < w_out; ++wo) {
                    float sum = 0.0f;
                    int count = 0;
                    for (int kh = 0; kh < kernel_size; ++kh)
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int hi = ho * stride - pad + kh;
                            int wi = wo * stride - pad + kw;
                            if (hi >= 0 && hi < input.h && wi >= 0 && wi < input.w) {
                                sum += input.at(ni, ci, hi, wi);
                                ++count;
                            }
                        }
                    output.at(ni, ci, ho, wo) = (count > 0) ? sum / count : 0.0f;
                }
    return output;
}

// ============================================================
// Part 5: 演示与验证
// ============================================================

static float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    return max_err;
}

// ---------- Demo 1: XOR 分类 ----------
void demo_xor() {
    printf("=== Demo 1: XOR Classification ===\n\n");

    // XOR 数据集: 4 个样本
    Matrix x(4, 2);
    x.at(0, 0) = 0.0f; x.at(0, 1) = 0.0f;
    x.at(1, 0) = 0.0f; x.at(1, 1) = 1.0f;
    x.at(2, 0) = 1.0f; x.at(2, 1) = 0.0f;
    x.at(3, 0) = 1.0f; x.at(3, 1) = 1.0f;

    // 标签: class 0 或 class 1
    std::vector<int> labels = {0, 1, 1, 0};

    // MLP: 2 → 16 → 8 → 2
    MLP model(2, {16, 8}, 2, OutputMode::CLASSIFICATION);

    printf("  Architecture: 2 → 16 (ReLU) → 8 (ReLU) → 2 (Softmax)\n");
    printf("  Training for 500 epochs...\n\n");

    float lr = 0.5f;
    for (int epoch = 0; epoch <= 500; ++epoch) {
        float loss = model.train_step(x, labels, lr);

        if (epoch % 100 == 0) {
            Matrix out = model.forward(x);
            int correct = 0;
            for (int i = 0; i < 4; ++i) {
                int pred = (out.at(i, 1) > out.at(i, 0)) ? 1 : 0;
                if (pred == labels[i]) ++correct;
            }
            printf("  epoch %3d  loss=%.6f  accuracy=%d/4\n", epoch, loss, correct);
        }
    }

    // 最终预测
    Matrix out = model.forward(x);
    printf("\n  Final predictions:\n");
    printf("    Input     →  P(class0)  P(class1)  →  Pred  True\n");
    for (int i = 0; i < 4; ++i) {
        int pred = (out.at(i, 1) > out.at(i, 0)) ? 1 : 0;
        printf("    [%.0f, %.0f]    %.4f     %.4f     →  %d     %d\n",
               x.at(i, 0), x.at(i, 1), out.at(i, 0), out.at(i, 1), pred, labels[i]);
    }
    printf("\n");
}

// ---------- Demo 2: 螺旋分类 (2D 3-class) ----------
// 生成三类螺旋数据，训练 MLP 学习决策边界

void generate_spiral(Matrix& x, std::vector<int>& labels, int points_per_class) {
    int n = points_per_class * 3;
    x = Matrix(n, 2);
    labels.resize(n);

    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < points_per_class; ++i) {
            float t = static_cast<float>(i) / points_per_class;
            float r = t * 4.0f;
            float angle = t * 6.0f + c * 2.094f;  // 2π/3 间隔
            int idx = c * points_per_class + i;
            x.at(idx, 0) = r * std::sin(angle) + 0.1f * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f);
            x.at(idx, 1) = r * std::cos(angle) + 0.1f * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f);
            labels[idx] = c;
        }
    }
}

void demo_spiral() {
    printf("=== Demo 2: Spiral Classification (3-class) ===\n\n");

    Matrix x;
    std::vector<int> labels;
    generate_spiral(x, labels, /*points_per_class=*/50);

    printf("  Dataset: %d samples, 3 classes, 2D input\n", x.rows);

    // MLP: 2 → 64 → 32 → 3
    MLP model(2, {64, 32}, 3, OutputMode::CLASSIFICATION);
    printf("  Architecture: 2 → 64 (ReLU) → 32 (ReLU) → 3 (Softmax)\n");
    printf("  Training for 500 epochs...\n\n");

    float lr = 0.5f;
    for (int epoch = 0; epoch <= 500; ++epoch) {
        float loss = model.train_step(x, labels, lr);

        if (epoch % 100 == 0) {
            Matrix out = model.forward(x);
            int correct = 0;
            for (int i = 0; i < x.rows; ++i) {
                int pred = 0;
                float max_p = out.at(i, 0);
                for (int c = 1; c < 3; ++c) {
                    if (out.at(i, c) > max_p) { max_p = out.at(i, c); pred = c; }
                }
                if (pred == labels[i]) ++correct;
            }
            printf("  epoch %3d  loss=%.4f  accuracy=%d/%d (%.0f%%)\n",
                   epoch, loss, correct, x.rows, 100.0f * correct / x.rows);
        }
    }
    printf("\n");
}

// ---------- Demo 3: CNN 算子验证 ----------
void demo_cnn() {
    printf("=== Demo 3: CNN Operators ===\n\n");

    // Conv2D
    {
        printf("--- Conv2D ---\n");
        Tensor input(1, 1, 5, 5);
        Tensor weight(2, 1, 3, 3);
        std::vector<float> bias(2, 0.0f);
        std::iota(input.data.begin(), input.data.end(), 1.0f);
        std::fill(weight.data.begin(), weight.data.end(), 1.0f);
        input.print_shape("input");
        weight.print_shape("weight");

        auto output = conv2d(input, weight, bias, 1, 0);
        output.print_shape("output");
        printf("  output[0][0] (每个位置 = 9 个输入值之和):\n");
        for (int h = 0; h < output.h; ++h) {
            printf("    ");
            for (int w = 0; w < output.w; ++w)
                printf("%7.1f", output.at(0, 0, h, w));
            printf("\n");
        }
        printf("\n");
    }

    // MaxPool2D
    {
        printf("--- MaxPool2D ---\n");
        Tensor input(1, 1, 4, 4);
        std::iota(input.data.begin(), input.data.end(), 1.0f);
        printf("  input[0][0]:\n");
        for (int h = 0; h < input.h; ++h) {
            printf("    ");
            for (int w = 0; w < input.w; ++w)
                printf("%5.1f", input.at(0, 0, h, w));
            printf("\n");
        }

        auto output = maxpool2d(input, 2, 2, 0);
        output.print_shape("output");
        printf("  output[0][0] (2x2 max):\n");
        for (int h = 0; h < output.h; ++h) {
            printf("    ");
            for (int w = 0; w < output.w; ++w)
                printf("%5.1f", output.at(0, 0, h, w));
            printf("\n");
        }
        printf("\n");
    }

    // AvgPool2D
    {
        printf("--- AvgPool2D ---\n");
        Tensor input(1, 1, 4, 4);
        std::iota(input.data.begin(), input.data.end(), 1.0f);
        auto output = avgpool2d(input, 2, 2, 0);
        output.print_shape("output");
        printf("  output[0][0] (2x2 avg):\n");
        for (int h = 0; h < output.h; ++h) {
            printf("    ");
            for (int w = 0; w < output.w; ++w)
                printf("%7.2f", output.at(0, 0, h, w));
            printf("\n");
        }
        printf("\n");
    }

    // Softmax
    {
        printf("--- Softmax ---\n");
        const int m = 3, n = 4;
        Matrix x(m, n);
        float vals[] = {1,2,3,4, 1,0,-1,-2, 100,101,102,103};
        for (int i = 0; i < m * n; ++i) x.data[i] = vals[i];
        auto y = softmax_forward(x);
        printf("  input (3x4, 第 3 行大数测试数值稳定性):\n");
        for (int i = 0; i < m; ++i) {
            printf("    ");
            for (int j = 0; j < n; ++j) printf("%8.2f", x.at(i, j));
            printf("\n");
        }
        printf("  softmax output (每行和 ≈ 1.0):\n");
        for (int i = 0; i < m; ++i) {
            printf("    ");
            float row_sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                printf("%8.4f", y.at(i, j));
                row_sum += y.at(i, j);
            }
            printf("  sum=%.6f\n", row_sum);
        }
        printf("\n");
    }
}

int main() {
    std::srand(42);

    demo_cnn();
    demo_xor();
    demo_spiral();

    printf("=== All Done ===\n");
    return 0;
}
