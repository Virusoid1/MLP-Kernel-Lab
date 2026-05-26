# NVIDIA cuTile Python 编程指南

> 基于 cuTile 1.3+，面向有 PyTorch/Triton 经验的开发者

## 目录

- [1. cuTile 是什么](#1-cutile-是什么)
- [2. 安装与环境](#2-安装与环境)
- [3. 核心概念](#3-核心概念)
- [4. 第一个 kernel：向量加法](#4-第一个-kernel向量加法)
- [5. 数据模型详解](#5-数据模型详解)
- [6. kernel 编写参考](#6-kernel-编写参考)
- [7. 完整示例：分块矩阵乘法](#7-完整示例分块矩阵乘法)
- [8. 完整示例：LayerNorm](#8-完整示例layernorm)
- [9. 融合 kernel 示例](#9-融合-kernel-示例)
- [10. 高级模式](#10-高级模式)
- [11. 常见陷阱与调试技巧](#11-常见陷阱与调试技巧)
- [12. API 速查表](#12-api-速查表)
- [13. cuTile vs Triton 对比](#13-cutile-vs-triton-对比)

---

## 1. cuTile 是什么

**cuTile** 是 NVIDIA 推出的 Python tile-based GPU 编程模型。它提供了一套以 **Tile**（分块）为第一类抽象的 API，让开发者用 Python 编写高性能 GPU kernel，无需手写 CUDA C++。

核心理念：
- **Tile 是基本单位**：kernel 操作的不是单个元素，而是一小块数据（tile），形状在编译期确定
- **编译期常量**：tile 大小、数组维度等必须标注为 `ct.Constant[int]`，编译器据此生成高效代码
- **不可变 Tile**：所有 tile 运算产生新 tile，不会就地修改
- **PyTorch 互操作**：输入/输出是 `torch.Tensor`，可无缝嵌入 PyTorch 训练循环

适用场景：
- 编写融合 kernel（如 matmul+bias+activation 单次 kernel）
- 替代手写 CUDA C++ kernel 的 Python 方案
- 与 Triton 互补：cuTile 更贴近硬件 tile 语义，Triton 更贴近 Python 数组语义

---

## 2. 安装与环境

### 2.1 安装

```bash
pip install cuda-tile          # cuTile 核心
pip install torch              # PyTorch（必须，cuTile 依赖 torch.Tensor）
```

### 2.2 环境要求

| 组件 | 最低版本 |
|------|----------|
| CUDA Driver | 535+（推荐 596+） |
| Python | 3.8+ |
| PyTorch | 2.0+ |
| GPU | compute capability 7.5+（Turing/Ampere/Ada/Blackwell） |

### 2.3 验证安装

```python
import cuda.tile as ct
import torch

# 确认 cuTile 版本
print(ct.__version__)  # 应输出 1.3+

# 确认 CUDA 可用
assert torch.cuda.is_available()
```

---

## 3. 核心概念

### 3.1 程序结构

一个 cuTile kernel 由三部分组成：

```
1. kernel 定义  —— @ct.kernel 装饰的函数，描述每个 thread block 做什么
2. host 函数    —— 分配输出、计算 grid 大小、调用 ct.launch
3. 调用入口     —— host 函数被 PyTorch autograd 层调用
```

### 3.2 Tile

Tile 是 cuTile 的核心数据类型，表示一小块连续的 GPU 数据。Tile 类似 NumPy 数组，但有两个关键区别：

1. **形状在编译期确定**：tile 形状来自 `ct.Constant[int]` 参数，编译器据此优化
2. **不可变**：所有操作返回新 tile，原始 tile 不变

```python
# Tile 的创建
acc = ct.full((32, 32), 0.0, dtype=ct.float32)  # 32x32 全零 tile
z = ct.zeros((1, 256), dtype=ct.float32)          # 1x256 零 tile
```

### 3.3 Array

Array 是 `torch.Tensor` 的 wrapper。kernel 接收的数组参数就是 array。通过 `ct.load` 从 array 读取数据到 tile，通过 `ct.store` 将 tile 写回 array。

### 3.4 执行模型

```
ct.launch(stream, grid, kernel, args)
         │       │      │       │
         │       │      │       └─ kernel 参数 tuple
         │       │      └───────── @ct.kernel 函数
         │       └──────────────── (x, y, z) grid 大小
         └──────────────────────── CUDA stream（通常用 torch.cuda.current_stream()）
```

- `grid` 是 3-tuple，定义 thread block 的网格大小
- `ct.bid(dim)` 返回当前 block 在指定维度的 ID（类似 CUDA 的 blockIdx）
- 每个 block 执行 kernel 函数体一次

### 3.5 ct.Constant

编译期常量。所有 tile 大小和数组维度参数必须标注为 `ct.Constant[int]`（或 `ct.Constant[float]`）：

```python
@ct.kernel
def my_kernel(A, Out,
              N: ct.Constant[int],      # 编译期常量
              TILE: ct.Constant[int]):   # 编译期常量
    ...
```

cuTile 编译器会为每组不同的常量值生成专门的 kernel 代码，实现类似 C++ 模板的效果。

---

## 4. 第一个 kernel：向量加法

从最简单的例子开始理解 cuTile 的工作方式：

```python
import cuda.tile as ct
import torch
from math import ceil

# 1. 定义 kernel
@ct.kernel
def vector_add_kernel(A, B, C, N: ct.Constant[int], TILE: ct.Constant[int]):
    pid = ct.bid(0)                                    # 当前 block 的 ID
    a = ct.load(A, index=(pid,), shape=(TILE,))        # 加载一个 tile
    b = ct.load(B, index=(pid,), shape=(TILE,))
    ct.store(C, index=(pid,), tile=a + b)              # 写回结果

# 2. host 函数
def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.shape[0]
    TILE = 512
    c = torch.empty_like(a)
    grid = (ceil(n / TILE), 1, 1)                      # ceil(n/TILE) 个 block
    ct.launch(torch.cuda.current_stream(), grid, vector_add_kernel,
              (a, b, c, n, TILE))
    return c

# 3. 测试
a = torch.randn(4096, device="cuda")
b = torch.randn(4096, device="cuda")
c = vector_add(a, b)
assert torch.allclose(c, a + b)
```

逐步解析：

| 步骤 | 代码 | 说明 |
|------|------|------|
| 获取 block ID | `pid = ct.bid(0)` | block 在第 0 维的索引 |
| 加载数据 | `ct.load(A, index=(pid,), shape=(TILE,))` | 从 array 的 `pid*TILE` 位置加载 TILE 个元素 |
| 计算 | `a + b` | 逐元素加法，返回新 tile |
| 写回结果 | `ct.store(C, index=(pid,), tile=result)` | 将 tile 写到 array 的 `pid*TILE` 位置 |
| 启动 | `ct.launch(stream, grid, kernel, args)` | 启动 `ceil(n/TILE)` 个 block |

---

## 5. 数据模型详解

### 5.1 ct.load

```python
ct.load(array, index, shape, padding_mode=ct.PaddingMode.ZERO)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `array` | torch.Tensor | 输入数组 |
| `index` | tuple[int/tile] | tile 在数组中的位置（tile-level 索引） |
| `shape` | tuple[int] | tile 的形状（各维度大小） |
| `padding_mode` | ct.PaddingMode | 越界处理方式 |

**关键规则：`index` 的长度必须等于 array 的 rank（维度数）。**

```python
# 1D array (rank=1): index 是 1-tuple
x = ct.load(X_1d, index=(pid,), shape=(TILE,))

# 2D array (rank=2): index 是 2-tuple
a = ct.load(A_2d, index=(pid_m, pid_n), shape=(TM, TN))

# 加载 2D array 的一行: index=(row, tile_col)
x = ct.load(X_2d, index=(row, j), shape=(1, TILE))
```

`index` 中的每个值表示 tile 在该维度的 **tile 级别索引**，不是元素级别。例如 `index=(pid_m, k), shape=(TM, TK)` 会从第 `pid_m * TM` 行、第 `k * TK` 列开始加载 `TM x TK` 大小的 tile。

### 5.2 ct.store

```python
ct.store(array, index, tile)
```

与 `ct.load` 对称：将 tile 写入 array 的指定位置。

### 5.3 ct.PaddingMode

```python
ct.PaddingMode.ZERO   # 越界位置填充 0.0
```

当 tile 超出 array 边界时（如最后一组 tile 不足 TILE 个元素），自动用 0 填充。这是最常用的模式。

### 5.4 数据类型

```python
ct.float32    # 单精度（默认）
ct.float16    # 半精度
ct.int32      # 32 位整数
```

### 5.5 Broadcasting

tile 运算支持 NumPy 风格的 broadcasting：

```python
# (1, TILE) + (TILE,) -> (1, TILE)  ← 自动 broadcast
result = x_tile + bias_tile

# (TM, TN) + (TN,) -> (TM, TN)     ← 标量 bias broadcast 到每行
acc = acc + bias
```

---

## 6. kernel 编写参考

### 6.1 工厂函数

```python
ct.full(shape, value, dtype=ct.float32)   # 用 value 填充
ct.zeros(shape, dtype=ct.float32)          # 全零
ct.ones(shape, dtype=ct.float32)           # 全一
```

### 6.2 数学运算

```python
# 算术（逐元素）
a + b       # 加
a - b       # 减
a * b       # 乘（逐元素，非矩阵乘）
a / b       # 除

# 数学函数
ct.exp(x)       # 指数
ct.log(x)       # 对数
ct.sqrt(x)      # 平方根
ct.tanh(x)      # 双曲正切
ct.sigmoid(x)   # 逻辑函数（注意：需要手动实现 1/(1+exp(-x))）

# 手动 sigmoid（cuTile 无 ct.sigmoid）
sigmoid_x = 1.0 / (1.0 + ct.exp(-x))
```

### 6.3 条件运算

```python
ct.where(condition, true_value, false_value)

# 示例：ReLU
relu = ct.where(x > 0.0, x, 0.0)
```

### 6.4 矩阵乘法

```python
ct.mma(a, b, acc)    # acc = a @ b + acc（融合乘加）
```

`ct.mma` 是 cuTile 的核心运算，等价于 `acc += a @ b`。三个参数：
- `a`: 左矩阵 tile
- `b`: 右矩阵 tile
- `acc`: 累加器 tile（同时是输出）

### 6.5 形状操作

```python
ct.transpose(tile)           # 转置最后两个维度
ct.reshape(tile, new_shape)  # 改变形状
ct.arange(n, dtype=ct.int32) # 生成 [0, 1, 2, ..., n-1]
```

### 6.6 归约

```python
ct.sum(tile, axis=1, keepdims=True)   # 沿指定轴求和
```

- `axis`: 归约的轴（整数）
- `keepdims=True`: 保持维度（用于后续 broadcasting）

### 6.7 原子操作

```python
ct.atomic_add(array, indices, update)
```

两种使用方式：

**连续模式**（indices 是 tile-level 索引）：
```python
ct.atomic_add(DGamma, (j,), update)  # 写到 array[j*TILE : (j+1)*TILE]
```

**Scatter 模式**（indices 是元素级别偏移）：
```python
offsets = j * TILE + ct.arange(TILE, dtype=ct.int32)  # [j*TILE, j*TILE+1, ...]
ct.atomic_add(DGamma, offsets, update)
```

Scatter 模式在多个 block 向同一数组累加时使用（如 LayerNorm backward 的 `d_gamma`）。

---

## 7. 完整示例：分块矩阵乘法

矩阵乘法 `C = A @ B` 是理解 cuTile tile 编程的最佳案例。

### 7.1 算法思路

```
A: (M, K) 分为 (M/TM, K/TK) 个 tile
B: (K, N) 分为 (K/TK, N/TN) 个 tile
C: (M, N) 分为 (M/TM, N/TN) 个 tile

每个 block 负责计算 C 的一个 tile：
  C[pid_m, pid_n] = Σ_k  A[pid_m, k] @ B[k, pid_n]
```

### 7.2 完整代码

```python
import cuda.tile as ct
import torch
from math import ceil

@ct.kernel
def matmul_kernel(A, B, C,
                  M: ct.Constant[int], N: ct.Constant[int], K: ct.Constant[int],
                  TM: ct.Constant[int], TN: ct.Constant[int], TK: ct.Constant[int]):
    pid_m = ct.bid(0)           # C 的行 tile ID
    pid_n = ct.bid(1)           # C 的列 tile ID
    num_tiles_k = ct.cdiv(K, TK) # K 维度有多少个 tile

    # FP32 累加器，初始为 0
    acc = ct.full((TM, TN), 0.0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # 沿 K 维度循环，累加部分乘积
    for k in range(num_tiles_k):
        a = ct.load(A, index=(pid_m, k), shape=(TM, TK), padding_mode=zero_pad)
        b = ct.load(B, index=(k, pid_n), shape=(TK, TN), padding_mode=zero_pad)
        acc = ct.mma(a, b, acc)     # acc += a @ b

    # 写回结果
    ct.store(C, index=(pid_m, pid_n), tile=acc)


def cutile_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    TM, TN, TK = 32, 32, 32     # tile 大小
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = (ceil(M / TM), ceil(N / TN), 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel,
              (a, b, c, M, N, K, TM, TN, TK))
    return c
```

### 7.3 数据流图

```
Array A (M, K)           Array B (K, N)
    │                         │
    │ load(pid_m, k)          │ load(k, pid_n)
    │ shape=(TM,TK)           │ shape=(TK,TN)
    ▼                         ▼
  Tile a  ──── ct.mma ──── Tile b
                │
                │ acc = a @ b + acc
                ▼
            Tile acc (TM, TN)
                │
                │ 循环 K/TK 次后
                ▼
          ct.store → Array C (M, N)
```

---

## 8. 完整示例：LayerNorm

LayerNorm 展示了 cuTile 的归约、循环累加、原子操作等高级模式。

### 8.1 算法

```
forward:
  mean_i = Σ_j x[i,j] / N
  var_i  = Σ_j (x[i,j] - mean_i)^2 / N
  rstd_i = 1 / sqrt(var_i + eps)
  y[i,j] = gamma[j] * (x[i,j] - mean_i) * rstd_i + beta[j]

backward:
  c1 = mean(dy * gamma)
  c2 = mean(dy * gamma * x_hat)
  dx = rstd * (dy * gamma - c1 - x_hat * c2)
  d_gamma += dy * x_hat    (跨 block 累加 → atomic_add)
  d_beta  += dy             (跨 block 累加 → atomic_add)
```

### 8.2 Forward kernel

```python
@ct.kernel
def layernorm_forward_kernel(X, Gamma, Beta, Y, Mean, Rstd,
                              B: ct.Constant[int], N: ct.Constant[int],
                              TILE: ct.Constant[int], eps: ct.Constant[float]):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    # ── 计算 mean ──
    # 关键：累加器必须初始化为 tile，不能是标量 0.0
    mean_acc = ct.zeros((1, TILE), dtype=ct.float32)
    for j in range(num_tiles):
        x_tile = ct.load(X, index=(row, j), shape=(1, TILE),
                         padding_mode=ct.PaddingMode.ZERO)
        mean_acc = mean_acc + x_tile
    mean_val = ct.sum(mean_acc, axis=1, keepdims=True) / N   # (1, 1)

    # ── 计算 variance ──
    var_acc = ct.zeros((1, TILE), dtype=ct.float32)
    for j in range(num_tiles):
        x_tile = ct.load(X, index=(row, j), shape=(1, TILE),
                         padding_mode=ct.PaddingMode.ZERO)
        centered = x_tile - mean_val
        var_acc = var_acc + centered * centered
    var_val = ct.sum(var_acc, axis=1, keepdims=True) / N     # (1, 1)
    rstd_val = 1.0 / ct.sqrt(var_val + eps)                  # (1, 1)

    # 保存 mean 和 rstd 供 backward 使用
    ct.store(Mean, index=(row,), tile=ct.reshape(mean_val, (1,)))
    ct.store(Rstd, index=(row,), tile=ct.reshape(rstd_val, (1,)))

    # ── normalize + affine ──
    for j in range(num_tiles):
        x_tile = ct.load(X, index=(row, j), shape=(1, TILE),
                         padding_mode=ct.PaddingMode.ZERO)
        gamma_tile = ct.load(Gamma, index=(j,), shape=(TILE,),
                             padding_mode=ct.PaddingMode.ZERO)
        beta_tile = ct.load(Beta, index=(j,), shape=(TILE,),
                            padding_mode=ct.PaddingMode.ZERO)
        x_hat = (x_tile - mean_val) * rstd_val
        y = x_hat * gamma_tile + beta_tile
        ct.store(Y, index=(row, j), tile=y)
```

### 8.3 Backward kernel（含 atomic_add scatter）

```python
@ct.kernel
def layernorm_backward_kernel(DY, X, Gamma, Mean, Rstd, DX, DGamma, DBeta,
                               B: ct.Constant[int], N: ct.Constant[int],
                               TILE: ct.Constant[int]):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    # 加载 mean, rstd（标量），reshape 为 (1,1) 用于 broadcasting
    mean_val = ct.reshape(ct.load(Mean, index=(row,), shape=(1,)), (1, 1))
    rstd_val = ct.reshape(ct.load(Rstd, index=(row,), shape=(1,)), (1, 1))

    # ── 第一遍：计算 c1, c2 ──
    c1_acc = ct.zeros((1, TILE), dtype=ct.float32)
    c2_acc = ct.zeros((1, TILE), dtype=ct.float32)
    for j in range(num_tiles):
        dy_tile = ct.load(DY, index=(row, j), shape=(1, TILE),
                          padding_mode=ct.PaddingMode.ZERO)
        x_tile = ct.load(X, index=(row, j), shape=(1, TILE),
                         padding_mode=ct.PaddingMode.ZERO)
        gamma_tile = ct.load(Gamma, index=(j,), shape=(TILE,),
                             padding_mode=ct.PaddingMode.ZERO)
        x_hat = (x_tile - mean_val) * rstd_val
        dy_gamma = dy_tile * gamma_tile
        c1_acc = c1_acc + dy_gamma
        c2_acc = c2_acc + dy_gamma * x_hat

    c1 = ct.sum(c1_acc, axis=1, keepdims=True) / N
    c2 = ct.sum(c2_acc, axis=1, keepdims=True) / N

    # ── 第二遍：计算 dx + 累加 d_gamma/d_beta ──
    for j in range(num_tiles):
        dy_tile = ct.load(DY, index=(row, j), shape=(1, TILE),
                          padding_mode=ct.PaddingMode.ZERO)
        x_tile = ct.load(X, index=(row, j), shape=(1, TILE),
                         padding_mode=ct.PaddingMode.ZERO)
        gamma_tile = ct.load(Gamma, index=(j,), shape=(TILE,),
                             padding_mode=ct.PaddingMode.ZERO)
        x_hat = (x_tile - mean_val) * rstd_val
        dy_gamma = dy_tile * gamma_tile
        dx = rstd_val * (dy_gamma - c1 - x_hat * c2)

        ct.store(DX, index=(row, j), tile=dx)

        # scatter atomic_add：多个 block 累加到同一个 d_gamma/d_beta
        offsets = j * TILE + ct.arange(TILE, dtype=ct.int32)
        ct.atomic_add(DGamma, offsets, ct.reshape(dy_tile * x_hat, (TILE,)))
        ct.atomic_add(DBeta, offsets, ct.reshape(dy_tile, (TILE,)))
```

### 8.4 关键模式总结

| 模式 | 用法 | 说明 |
|------|------|------|
| 循环累加 | `acc = ct.zeros(shape); for j in ...: acc = acc + ...` | 累加器必须初始化为 tile |
| 归约 | `ct.sum(acc, axis=1, keepdims=True)` | `keepdims=True` 保持 broadcasting 兼容 |
| 标量广播 | `(1,1)` tile 与 `(1,TILE)` tile 运算 | broadcasting 自动处理 |
| 跨 block 累加 | `ct.atomic_add(array, scatter_indices, update)` | scatter 模式解决多 block 写冲突 |

---

## 9. 融合 kernel 示例

cuTile 的一个核心优势是轻松编写融合 kernel，将多个算子合并到一次 kernel launch 中。

### 9.1 融合 Matmul + Bias + GELU

```python
@ct.kernel
def mlp_first_layer_kernel(A, B, Bias, C,
                            M: ct.Constant[int], N: ct.Constant[int],
                            K: ct.Constant[int], TM: ct.Constant[int],
                            TN: ct.Constant[int], TK: ct.Constant[int]):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    num_tiles_k = ct.cdiv(K, TK)

    # matmul
    acc = ct.full((TM, TN), 0.0, dtype=ct.float32)
    for k in range(num_tiles_k):
        a = ct.load(A, index=(pid_m, k), shape=(TM, TK),
                    padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, pid_n), shape=(TK, TN),
                    padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc)

    # bias add
    bias = ct.load(Bias, index=(pid_n,), shape=(TN,),
                   padding_mode=ct.PaddingMode.ZERO)
    acc = acc + bias

    # GELU (tanh 近似)
    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (acc + 0.044715 * acc * acc * acc)
    result = 0.5 * acc * (1.0 + ct.tanh(inner))

    ct.store(C, index=(pid_m, pid_n), tile=result)
```

### 9.2 融合 SwiGLU

```python
@ct.kernel
def swiglu_kernel(X, Out, N: ct.Constant[int], TILE: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(X, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    sigmoid_x = 1.0 / (1.0 + ct.exp(-x))
    ct.store(Out, index=(pid,), tile=x * sigmoid_x)
```

融合的优势：减少全局内存读写次数。分离实现需要 matmul 结果写回显存再读取做 activation；融合后在寄存器/共享内存中直接完成全部运算。

---

## 10. 高级模式

### 10.1 MatMul Backward

反向传播的矩阵乘法利用 `ct.transpose` 实现：

```python
# dA = dC @ B^T
@ct.kernel
def matmul_backward_a_kernel(dC, B, dA, ...):
    for n_tile in range(num_tiles_n):
        dc = ct.load(dC, ...)
        b = ct.load(B, ...)
        b_T = ct.transpose(b)   # B 的 tile 转置
        acc = ct.mma(dc, b_T, acc)

# dB = A^T @ dC
@ct.kernel
def matmul_backward_b_kernel(A, dC, dB, ...):
    for m_tile in range(num_tiles_m):
        a = ct.load(A, ...)
        dc = ct.load(dC, ...)
        a_T = ct.transpose(a)   # A 的 tile 转置
        acc = ct.mma(a_T, dc, acc)
```

### 10.2 Activation Backward

激活函数的梯度通过 `ct.where` 和数学运算实现：

```python
# ReLU backward: dx = dy if x > 0 else 0
ct.where(x > 0.0, dy, 0.0)

# GELU backward (tanh 近似)
u = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
tanh_u = ct.tanh(u)
sech2_u = 1.0 - tanh_u * tanh_u
du_dx = sqrt_2_over_pi * (1.0 + 0.134145 * x * x)
gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx

# SiLU backward: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
sig = 1.0 / (1.0 + ct.exp(-x))
silu_grad = sig * (1.0 + x * (1.0 - sig))
```

### 10.3 与 PyTorch autograd 集成

通过 `torch.autograd.Function` 将 cuTile kernel 包装为可微分层：

```python
class CUTILELinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        # 调用 cuTile matmul kernel
        output = cutile_matmul(x, weight)
        if bias is not None:
            output = bias_add(output, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        # 调用 cuTile backward kernel
        grad_input = matmul_backward_a(grad_output, weight)
        grad_weight = matmul_backward_b(x, grad_output)
        return grad_input, grad_weight, None

class CUTILELinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return CUTILELinearFunction.apply(x, self.weight, self.bias)
```

---

## 11. 常见陷阱与调试技巧

### 11.1 必须从文件加载源码

cuTile 使用 `inspect.getsourcelines()` 读取 kernel 源码进行编译。因此：

```bash
# 正确：从 .py 文件运行
python my_script.py

# 错误：使用 -c 参数（无源文件）
python -c "import cuda.tile as ct; ..."   # 报错：无法获取源码
```

在 Jupyter Notebook 中也可能遇到此问题，建议将 kernel 写入 .py 文件再 import。

### 11.2 index 维度必须匹配 array rank

```python
# X 是 2D tensor (B, N)
# 错误：index 只有 1 个维度
x = ct.load(X, index=(row,), shape=(N,))          # 报错！

# 正确：index 必须有 2 个维度
x = ct.load(X, index=(row, 0), shape=(1, TILE))   # 正确
```

规则：`len(index) == array.dim()`。

### 11.3 循环变量类型必须一致

cuTile 要求循环体内的变量在每次迭代中保持相同类型：

```python
# 错误：第一次 acc 是标量 0.0，之后变成 tile
acc = 0.0
for j in range(num_tiles):
    tile = ct.load(...)
    acc = acc + tile    # 类型不一致！

# 正确：始终使用 tile 类型
acc = ct.zeros((1, TILE), dtype=ct.float32)
for j in range(num_tiles):
    tile = ct.load(...)
    acc = acc + tile    # 类型一致
```

### 11.4 atomic_add 的两种模式

```python
# 连续模式：indices 是 tile-level 索引
ct.atomic_add(Array, (j,), update)    # 写到 [j*TILE : (j+1)*TILE]

# Scatter 模式：indices 是元素级偏移
offsets = j * TILE + ct.arange(TILE, dtype=ct.int32)
ct.atomic_add(Array, offsets, update) # 写到 [offsets[0], offsets[1], ...]
```

当多个 block 向同一数组的不同位置累加时，使用 scatter 模式。

### 11.5 Tile 大小选择

| 场景 | 推荐 TILE 大小 |
|------|----------------|
| Elementwise (1D) | 256, 512, 1024 |
| Matmul (2D) | TM=32, TN=32, TK=32 |
| LayerNorm | 256（匹配常见 hidden_dim） |

注意：TILE 过大导致过多 padding 会降低精度。当 `N < TILE` 时，大部分 tile 是填充的零，归约结果会偏小。

### 11.6 调试流程

1. **先用小尺寸测试**：如 4x4 矩阵、16 维向量
2. **对比 PyTorch 参考实现**：`torch.allclose(out, ref, rtol=1e-2, atol=1e-2)`
3. **检查 NaN/Inf**：`assert not torch.isnan(out).any()`
4. **逐步增加尺寸**：确认大尺寸也正确

---

## 12. API 速查表

### Kernel 定义与启动

| API | 说明 |
|-----|------|
| `@ct.kernel` | 定义 kernel 函数 |
| `ct.bid(dim)` | 当前 block 在 dim 维的 ID |
| `ct.cdiv(a, b)` | `ceil(a / b)`，向上取整除法 |
| `ct.launch(stream, grid, kernel, args)` | 启动 kernel |

### 数据加载/存储

| API | 说明 |
|-----|------|
| `ct.load(array, index, shape, padding_mode)` | 从 array 加载 tile |
| `ct.store(array, index, tile)` | 将 tile 写入 array |
| `ct.PaddingMode.ZERO` | 越界填零 |

### 工厂函数

| API | 说明 |
|-----|------|
| `ct.full(shape, value, dtype)` | 用 value 填充 |
| `ct.zeros(shape, dtype)` | 全零 tile |
| `ct.ones(shape, dtype)` | 全一 tile |
| `ct.arange(n, dtype)` | [0, 1, ..., n-1] |

### 数学运算

| API | 说明 |
|-----|------|
| `ct.mma(a, b, acc)` | 融合乘加 acc += a @ b |
| `ct.exp(x)` | 指数 |
| `ct.log(x)` | 对数 |
| `ct.sqrt(x)` | 平方根 |
| `ct.tanh(x)` | 双曲正切 |
| `ct.where(cond, a, b)` | 条件选择 |

### 形状与归约

| API | 说明 |
|-----|------|
| `ct.sum(tile, axis, keepdims)` | 沿轴求和 |
| `ct.transpose(tile)` | 转置最后两维 |
| `ct.reshape(tile, shape)` | 改变形状 |

### 原子操作

| API | 说明 |
|-----|------|
| `ct.atomic_add(array, indices, update)` | 原子加 |

### 类型注解

| API | 说明 |
|-----|------|
| `ct.Constant[int]` | 编译期整数常量 |
| `ct.Constant[float]` | 编译期浮点常量 |
| `ct.float32` | 32 位浮点 |
| `ct.float16` | 16 位浮点 |
| `ct.int32` | 32 位整数 |

---

## 13. cuTile vs Triton 对比

| 特性 | cuTile | Triton |
|------|--------|--------|
| **编程模型** | Tile-based（显式 tile 操作） | Block-based（类似 numpy 的指针算术） |
| **Kernel 定义** | `@ct.kernel` | `@triton.jit` |
| **数据加载** | `ct.load(array, index=(i, j), shape=(TM, TN))` | `tl.load(ptr + offsets)` |
| **矩阵乘法** | `ct.mma(a, b, acc)` 一等公民 | `tl.dot(a, b)` |
| **Tile 大小** | `ct.Constant[int]` 编译期确定 | `tl.constexpr` 或 `BLOCK_SIZE: tl.constexpr` |
| **Autotune** | 无（需手动调参） | `@triton.autotune` 自动搜索最优配置 |
| **原子操作** | `ct.atomic_add(array, indices, update)` | `tl.atomic_add(ptr, value)` |
| **越界处理** | `ct.PaddingMode.ZERO` | 手动 mask：`tl.load(..., mask=..., other=0.0)` |
| **数据类型** | `ct.float32`, `ct.int32` | `tl.float32`, `tl.int32` |
| **生态成熟度** | 早期（1.3） | 成熟（3.x） |
| **文档** | NVIDIA 官方文档 | 开源社区 + 论文 |

选择建议：
- **需要显式 tile 控制**（如精确控制 shared memory 使用）→ cuTile
- **需要快速开发、autotune** → Triton
- **需要与 CUDA C++ kernel 等价物对照** → cuTile（API 更贴近 SM 级别操作）
- **生产环境、社区支持** → Triton（更成熟）

---

## 参考资源

- [NVIDIA cuTile 官方文档](https://docs.nvidia.com/cuda/cutile-python/)
- [cuTile GitHub 示例](https://github.com/NVIDIA/cutile-python)
- 本项目 `cutile_kernels/` 目录包含 7 个完整实现的 kernel
- 本项目 `tests/test_cutile_kernels.py` 包含 18 项正确性测试
