# Triton 学习笔记

> Day 6 的 Triton 入门和 kernel 开发记录。
> 参考: [Triton 官方文档](https://triton-lang.org/main/index.html)

## 核心概念

### Triton vs CUDA

| | CUDA | Triton |
|---|---|---|
| 编程粒度 | Thread-level | Block-level |
| 语言 | C++ | Python DSL |
| 编译 | nvcc | Triton compiler (MLIR/LLVM) |
| 优势 | 精细控制 | 开发效率高 |
| 劣势 | 开发慢 | 控制粒度粗 |

### 基本结构

```python
@triton.jit
def my_kernel(
    x_ptr, y_ptr, output_ptr,  # 指针参数
    n_elements,                 # 标量参数
    BLOCK_SIZE: tl.constexpr,   # 编译期常量
):
    pid = tl.program_id(0)  # 获取 block 索引
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def my_func(x, y):
    output = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    my_kernel[grid](x, y, output, x.numel(), BLOCK_SIZE=1024)
    return output
```

### 关键 API

- `tl.program_id(axis)` — block 在 grid 中的索引
- `tl.arange(start, end)` — 生成 1D 范围的 indices
- `tl.load(ptr, mask=..., other=0.0)` — 从 global memory 加载
- `tl.store(ptr, value, mask=...)` — 存回 global memory
- `tl.dot(a, b)` — block-level 矩阵乘法
- `tl.cdiv(a, b)` — ceil division
- `tl.zeros(shape, dtype)` — 零初始化

### Block Pointer (Advanced)

```python
# 比手动计算 offset 更高效
a_ptrs = tl.make_block_ptr(
    base=A, shape=(M, K), strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_M, 0),
    block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
)
```

## Matmul 实现要点

### 数据布局

Triton 中矩阵通常 row-major:
- `A[M, K]` strides: `(K, 1)`
- `B[K, N]` strides: `(N, 1)`
- `C[M, N]` strides: `(N, 1)`

### tl.dot 要求

- 输入维度: `(M, K)` 和 `(K, N)`
- 自动使用 Tensor Core (如可用)
- 累加精度: `tl.float32`

### 典型 Grid

```python
grid = (
    triton.cdiv(M, BLOCK_M),
    triton.cdiv(N, BLOCK_N),
)
```

### Block Size 选择

| BLOCK_M | BLOCK_N | BLOCK_K | 适用 GPU |
|---------|---------|---------|----------|
| 64 | 64 | 32 | 大多数 |
| 128 | 128 | 32 | 大显存 |
| 32 | 32 | 32 | 调试 |

## Activation 实现

```python
# SiLU
@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

# GELU (tanh approx)
@triton.jit
def gelu(x):
    return 0.5 * x * (1.0 + tl.libdevice.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
```

## Autotune (可选, Day 12+)

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    ...
```

## 实际开发体验

<!-- TODO: 记录你的 Triton 开发体验 -->

```
写第一个 matmul kernel 用时: ___ 小时
调试主要遇到的问题: ___
与 CUDA 开发效率对比: ___
```

## 实际 Benchmark 结果

<!-- TODO: 填写你的 benchmark 结果 -->

```
Shape: M=512, K=768, N=3072, FP32
Triton matmul:  ___ ms, ___ TFLOPS
CUDA tiled:     ___ ms, ___ TFLOPS
PyTorch:        ___ ms, ___ TFLOPS
```
