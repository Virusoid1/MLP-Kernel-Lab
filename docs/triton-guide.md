# Triton Kernel 编程指南

> 基于 OpenAI Triton，面向本项目实际实现的完整教程。

## 目录

- [1. Triton 是什么](#1-triton-是什么)
- [2. 核心编程模型](#2-核心编程模型)
- [3. Elementwise kernel](#3-elementwise-kernel)
- [4. 分块矩阵乘法](#4-分块矩阵乘法)
- [5. LayerNorm](#5-layernorm)
- [6. Backward kernel](#6-backward-kernel)
- [7. 融合 kernel](#7-融合-kernel)
- [8. autograd 集成](#8-autograd-集成)
- [9. Autotune 优化](#9-autotune-优化)
- [10. 精度控制](#10-精度控制)
- [11. 常见陷阱](#11-常见陷阱)

---

## 1. Triton 是什么

Triton 是 OpenAI 开发的 Python GPU 编程语言。它提供类似 NumPy 的编程接口，但编译为高效 GPU kernel。与 CUDA C++ 相比，Triton 无需手动管理 shared memory、线程索引、同步等底层细节。

核心思想：
- **Block 为基本单位**：每个 program 处理一块数据，block 大小在编译期确定
- **指针算术**：通过 `base_ptr + offsets` 访问数据，比 cuTile 的 tile-level index 更灵活
- **自动优化**：编译器自动处理 shared memory 管理、指令调度

## 2. 核心编程模型

### 2.1 kernel 定义与启动

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                        # 当前 block 的 ID
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)

# 启动
n = 4096
BLOCK_SIZE = 1024
grid = (triton.cdiv(n, BLOCK_SIZE),)
my_kernel[grid](x, y, n, BLOCK_SIZE=BLOCK_SIZE)
```

关键要素：

| 概念 | Triton | cuTile |
|------|--------|--------|
| kernel 定义 | `@triton.jit` | `@ct.kernel` |
| block ID | `tl.program_id(0)` | `ct.bid(0)` |
| 数据加载 | `tl.load(ptr + offsets, mask=...)` | `ct.load(array, index=..., shape=...)` |
| 数据存储 | `tl.store(ptr + offsets, val, mask=...)` | `ct.store(array, index=..., tile=...)` |
| 越界处理 | `mask` + `other=0.0` | `ct.PaddingMode.ZERO` |
| 编译期常量 | `tl.constexpr` | `ct.Constant[int]` |

### 2.2 指针算术 vs Tile 索引

Triton 使用**指针偏移**访问数据：

```python
# Triton: 指针 + 元素级偏移
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(ptr + offsets, mask=offsets < n)
```

cuTile 使用**tile 级索引**：

```python
# cuTile: tile-level index
x = ct.load(array, index=(pid,), shape=(TILE,))
```

Triton 的指针模型更灵活（可以任意 stride），cuTile 的 tile 模型更结构化（自动处理 2D 分块）。

## 3. Elementwise kernel

### 3.1 ReLU

```python
@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, tl.where(x > 0, x, 0.0), mask=mask)
```

模式：每个 program 处理 BLOCK_SIZE 个元素，用 mask 处理尾部不足一个 block 的情况。

### 3.2 GELU（tanh 近似）

```python
@triton.jit
def gelu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    output = 0.5 * x * (1.0 + tanh_inner)

    tl.store(output_ptr + offsets, output, mask=mask)
```

注意：使用 `2.0 * tl.sigmoid(2.0 * inner) - 1.0` 近似 `tl.tanh(inner)`，兼容更多 Triton 版本。

### 3.3 BiasAdd（2D 数据）

```python
@triton.jit
def bias_add_kernel(input_ptr, bias_ptr, output_ptr, n_cols,
                    input_row_stride, output_row_stride,
                    BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x = tl.load(input_ptr + row_idx * input_row_stride + col_offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    tl.store(output_ptr + row_idx * output_row_stride + col_offsets, x + b, mask=mask)
```

2D kernel 的关键：用 `row_idx * stride` 定位行，`col_offsets` 遍历列。Stride 参数使 kernel 不依赖连续内存布局。

### 3.4 融合 BiasAdd + ReLU

```python
@triton.jit
def bias_add_relu_kernel(input_ptr, bias_ptr, output_ptr, n_cols,
                         input_row_stride, output_row_stride,
                         BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(input_ptr + row_idx * input_row_stride + col_offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    y = x + b
    tl.store(output_ptr + row_idx * output_row_stride + col_offsets,
             tl.where(y > 0, y, 0.0), mask=mask)
```

融合优势：原本需要 2 次 global memory 读写（bias_add 写一次，relu 写一次），融合后只写 1 次。

## 4. 分块矩阵乘法

### 4.1 L2 缓存优化的 super-grouping

```
标准 grid 遍历:
  block(0,0) → block(0,1) → block(1,0) → block(1,1) → ...
  问题：block(0,0) 读 A 的第 0 行，block(1,0) 也读 A 的第 0 行，
  但 block(0,1) 和 block(1,0) 之间 block(0,0) 的数据可能已被逐出 L2

super-grouping (GROUP_SIZE_M=8):
  把相邻 8 个行 block 编为一组，同一组内按列优先遍历
  → 同一组的 block 共享 A 的行数据，在 L2 中保持热
```

### 4.2 完整实现

```python
@triton.jit
def tiled_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 当前 block 负责的输出区域
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    # 初始化指针
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 沿 K 维度循环
    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K
        a = tl.load(a_ptrs, mask=m_mask & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 写回
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask & n_mask)
```

### 4.3 指针递增优化

注意 `a_ptrs += BLOCK_K * stride_ak`：K 维度循环中，指针通过增量而非重新计算来推进。这比每次循环重新算 `base + (k_start + offs_k) * stride` 更高效。

## 5. LayerNorm

### 5.1 Forward

```python
@triton.jit
def layernorm_forward_kernel(X_ptr, Y_ptr, Gamma_ptr, Beta_ptr,
                              Mean_ptr, Rstd_ptr, stride, N, eps,
                              BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride + cols, mask=mask, other=0.0)

    # 单行内计算 mean、var
    x_sum = tl.sum(x, axis=0)
    mean = x_sum / N
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # normalize + affine
    x_hat = x_centered * rstd
    gamma = tl.load(Gamma_ptr + cols, mask=mask, other=1.0)
    beta = tl.load(Beta_ptr + cols, mask=mask, other=0.0)
    y = gamma * x_hat + beta

    tl.store(Y_ptr + row * stride + cols, y, mask=mask)
    tl.store(Mean_ptr + row, mean)
    tl.store(Rstd_ptr + row, rstd)
```

Triton LayerNorm 的优势：一行用一个 program，`tl.sum` 直接做行内归约，无需手动管理 shared memory。

### 5.2 Backward（含 atomic_add）

```python
@triton.jit
def layernorm_backward_kernel(DY_ptr, X_ptr, Gamma_ptr, Mean_ptr, Rstd_ptr,
                               DX_ptr, DGamma_ptr, DBeta_ptr, stride, N,
                               BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    dy = tl.load(DY_ptr + row * stride + cols, mask=mask, other=0.0)
    x = tl.load(X_ptr + row * stride + cols, mask=mask, other=0.0)
    gamma = tl.load(Gamma_ptr + cols, mask=mask, other=1.0)
    mean = tl.load(Mean_ptr + row)
    rstd = tl.load(Rstd_ptr + row)

    x_hat = (x - mean) * rstd
    dy_gamma = dy * gamma
    c1 = tl.sum(dy_gamma, axis=0) / N
    c2 = tl.sum(dy_gamma * x_hat, axis=0) / N
    dx = rstd * (dy_gamma - c1 - x_hat * c2)

    tl.store(DX_ptr + row * stride + cols, dx, mask=mask)
    # d_gamma, d_beta 需要跨行累加 → atomic_add
    tl.atomic_add(DGamma_ptr + cols, dy * x_hat, mask=mask)
    tl.atomic_add(DBeta_ptr + cols, dy, mask=mask)
```

`tl.atomic_add` 跨多个 program（行）累加到同一 `d_gamma`/`d_beta` 数组。

## 6. Backward kernel

### 6.1 MatMul Backward

前向 `C = A @ B` 的反向：

```
dA = dC @ B^T    dC: (M, N), B: (K, N), dA: (M, K)
dB = A^T @ dC    A: (M, K), dC: (M, N), dB: (K, N)
```

```python
# dA = dC @ B^T
@triton.jit
def matmul_backward_a_kernel(dC_ptr, B_ptr, dA_ptr, M, N, K, ...):
    # B^T 的 tile: 加载 B 后转置（通过交换 stride 实现）
    B_ptrs = B_ptr + offs_k[None, :] * stride_bk + offs_n[:, None] * stride_bn
    # 注意 offs_n 是列方向，这里放在行 → 实现隐式转置
    B_tile = tl.load(B_ptrs, mask=k_mask[:, None] & n_mask, other=0.0)
    acc += tl.dot(dC, B_tile, allow_tf32=ALLOW_TF32)
```

关键技巧：**通过交换指针的 stride 实现隐式转置**。`B_ptrs` 中 `offs_n` 放在第一个维度（行），`offs_k` 放在第二个维度（列），相当于读取 `B^T` 的一个 tile。

### 6.2 Activation Backward

```python
# GELU backward（autotune 版）
@triton.autotune(configs=_ACT_BWD_CONFIGS, key=["n_elements"])
@triton.jit
def gelu_backward_kernel(grad_out_ptr, input_ptr, grad_in_ptr, n_elements,
                          BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    sqrt_2_over_pi = 0.7978845608028654
    u = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    tanh_u = 2.0 * tl.sigmoid(2.0 * u) - 1.0
    sech2_u = 1.0 - tanh_u * tanh_u
    du_dx = sqrt_2_over_pi * (1.0 + 0.134145 * x * x)
    gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx

    tl.store(grad_in_ptr + offsets, grad_out * gelu_grad, mask=mask)
```

## 7. 融合 kernel

### 融合 matmul + bias + GELU

```python
@triton.jit
def mlp_first_layer_kernel(X_ptr, W1_ptr, bias_ptr, H_ptr, M, N, K, ...):
    # ... matmul 循环同标准 matmul ...
    acc += tl.dot(x, w, allow_tf32=ALLOW_TF32)

    # bias add + GELU，在寄存器中完成，无额外 global memory 写入
    acc = acc + b[None, :]
    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (acc + 0.044715 * acc * acc * acc)
    tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    result = 0.5 * acc * (1.0 + tanh_inner)

    tl.store(h_ptrs, result, mask=m_mask & n_mask)
```

性能收益：matmul 结果留在寄存器中直接做 activation，避免一次 global memory round-trip。对于 `[784, 1024]` 这样的小矩阵，减少 global memory 写入的收益尤其明显。

## 8. autograd 集成

### 8.1 TritonLinearFunction

```python
class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        input_f = input.float()
        weight_f = weight.float()
        output = tiled_matmul(input_f, weight_f)
        if bias is not None:
            output = bias_add(output, bias.float())
        ctx.save_for_backward(input_f, weight_f)
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_f, weight_f = ctx.saved_tensors
        grad_output_f = grad_output.contiguous().float()
        grad_input = matmul_backward_a(grad_output_f, weight_f)
        grad_weight = matmul_backward_b(input_f, grad_output_f)
        grad_bias = torch.sum(grad_output_f, dim=0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias
```

### 8.2 TritonLinear nn.Module

```python
class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # 与 nn.Linear 一致
        init.zeros_(self.bias)

    def forward(self, input):
        return TritonLinearFunction.apply(input, self.weight, self.bias)
```

关键：`weight` 形状为 `(in_features, out_features)`（与 `nn.Linear` 的 `(out, in)` 不同），因为 Triton kernel 直接做 `input @ weight`，不需要转置。

## 9. Autotune 优化

### 9.1 配置设计

```python
_MATMUL_CONFIGS = [
    # 大 tile：Blackwell 最优（充足 shared memory）
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                  num_stages=3, num_warps=8),
    # 中 tile：Ampere 平衡
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                  num_stages=4, num_warps=4),
    # 小 tile：MNIST 等小矩阵
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                  num_stages=4, num_warps=4),
]
```

| 参数 | 含义 | 影响 |
|------|------|------|
| `BLOCK_M/N/K` | 输出/输入 tile 大小 | 大 tile → 更好的计算/访存比，但占用更多 shared memory |
| `num_stages` | Pipeline 深度 | 更多 stages → 更好地掩盖延迟，但占用更多 shared memory |
| `num_warps` | 每 block 的 warp 数 | 影响 SM 占用率 |
| `GROUP_SIZE_M` | L2 缓存 super-group 大小 | 相邻 N 个行 block 编为一组，共享 A 的行数据 |

### 9.2 Autotune 工作原理

```python
@triton.autotune(configs=_MATMUL_CONFIGS, key=["M", "N", "K"])
```

- 首次调用时，对每个配置 benchmark 一小段数据
- 按 `key` 参数分组：相同 `(M, N, K)` 复用同一最优配置
- 结果缓存：后续调用直接使用最优配置，无额外开销

## 10. 精度控制

### 全局精度单例

```python
# triton_kernels/precision.py
class _PrecisionConfig:
    def __init__(self):
        self._allow_tf32 = True

    @property
    def allow_tf32(self) -> bool:
        return self._allow_tf32

precision = _PrecisionConfig()
```

### 在 kernel 中使用

```python
@triton.jit
def matmul_kernel(..., ALLOW_TF32: tl.constexpr):
    acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
```

### 统一控制

```python
# run_compare.py
if args.precision == "fp32":
    torch.backends.cuda.matmul.allow_tf32 = False   # PyTorch
    precision.allow_tf32 = False                     # Triton
else:
    torch.backends.cuda.matmul.allow_tf32 = True
    precision.allow_tf32 = True
```

## 11. 常见陷阱

### 11.1 Grid 大小硬编码导致梯度爆炸

```python
# 错误：grid 硬编码为 cdiv(n, 2048)
relu_backward_kernel[(triton.cdiv(n, 2048),)](grad_out, x, grad_in, n)

# 但 autotune 可能选择 BLOCK_SIZE=512
# → grid 只覆盖 n/2048 个 block，其余元素未被写入 → 梯度爆炸

# 正确：grid 使用 autotune 实际选择的 BLOCK_SIZE
grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
relu_backward_kernel[grid](grad_out, x, grad_in, n)
```

### 11.2 Dropout 种子共享

```python
# 错误：所有行共享同一个 dropout mask
mask = tl.rand(seed, col_offsets)

# 正确：每行不同的 mask
mask = tl.rand(seed, row_idx * n_cols + col_offsets)
```

### 11.3 `tl.tanh` 兼容性

部分 Triton 版本不支持 `tl.tanh`，用 `2.0 * tl.sigmoid(2.0 * x) - 1.0` 替代。

### 11.4 BLOCK_SIZE 必须是 2 的幂

`triton.next_power_of_2(N)` 自动向上取整到最近的 2 的幂。
