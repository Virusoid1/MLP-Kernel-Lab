# MLP-Kernel-Lab 优化思路详解

> 基于项目实际改动，记录每次优化的动机、方法和效果。

## 目录

- [1. 项目优化全景](#1-项目优化全景)
- [2. 权重初始化对齐](#2-权重初始化对齐)
- [3. 精度控制统一](#3-精度控制统一)
- [4. Triton Grid 大小 Bug 修复](#4-triton-grid-大小-bug-修复)
- [5. Triton Autotune 配置设计](#5-triton-autotune-配置设计)
- [6. CUDA 推理路径优化](#6-cuda-推理路径优化)
- [7. CUDA WMMA 自适应 Dispatch](#7-cuda-wmma-自适应-dispatch)
- [8. CUDA float4 向量化](#8-cuda-float4-向量化)
- [9. 融合 Kernel 设计](#9-融合-kernel-设计)
- [10. LayerNorm 三端实现对比](#10-layernorm-三端实现对比)
- [11. 性能数据总结](#11-性能数据总结)

---

## 1. 项目优化全景

优化分三个阶段：

```
阶段 1：正确性（5/25）
  → 四个后端分别实现，确保训练收敛
  → 修复 dropout、GELU 等正确性 bug

阶段 2：公平对比（5/26）
  → 统一精度控制（TF32/FP32 全局切换）
  → 统一权重初始化
  → 统一模型架构（Linear→LayerNorm→ReLU→Dropout）

阶段 3：性能优化（5/26-27）
  → Autotune 配置针对 GPU 架构调优
  → CUDA 推理路径优化
  → 融合 kernel 减少内存往返
```

## 2. 权重初始化对齐

### 问题

初始实现中，自定义层和 PyTorch `nn.Linear` 的权重初始化不一致：

```python
# nn.Linear 默认
nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

# 初始的自定义层（缺少 a=math.sqrt(5)）
nn.init.kaiming_uniform_(self.weight)
```

### 影响

`a` 参数影响 `kaiming_uniform_` 的分布范围。LeakyReLU 的 negative slope `a=math.sqrt(5)` 对应 `nn.Linear` 的默认 fan_in 模式。不一致的初始化导致各后端从不同的起点开始训练，准确率差异无法归因于 kernel 实现差异。

### 修复

在 `triton_layers.py` 和 `cuda_layers.py` 中统一使用 `a=math.sqrt(5)`：

```python
def reset_parameters(self):
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        init.zeros_(self.bias)
```

## 3. 精度控制统一

### 问题

三个后端使用不同精度，结果不可比：

| 后端 | 默认精度 | 原因 |
|------|----------|------|
| PyTorch | TF32 | Ampere+ GPU 默认 `allow_tf32=True` |
| Triton | 硬编码 TF32 | `tl.dot(..., allow_tf32=True)` 写死 4 处 |
| CUDA | FP32 | 自定义 tiled kernel 无 TF32，cuBLAS 才有 |

### 解决方案

**新建 `triton_kernels/precision.py`**——全局精度单例：

```python
class _PrecisionConfig:
    def __init__(self):
        self._allow_tf32 = True

    @property
    def allow_tf32(self) -> bool:
        return self._allow_tf32

precision = _PrecisionConfig()
```

**修改 3 个 Triton kernel 文件**，将 `allow_tf32` 从硬编码改为读取全局配置：

```python
# 修改前
acc += tl.dot(a, b, allow_tf32=True)

# 修改后
@triton.jit
def matmul_kernel(..., ALLOW_TF32: tl.constexpr):
    acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

# 调用时传入
ALLOW_TF32=precision.allow_tf32
```

**`run_compare.py` 统一控制**：

```python
if args.precision == "fp32":
    torch.backends.cuda.matmul.allow_tf32 = False  # PyTorch
    precision.allow_tf32 = False                     # Triton
    CUDALinearFunction.use_cublas = False            # CUDA（禁用 cuBLAS，使用自定义 FP32 kernel）
else:
    torch.backends.cuda.matmul.allow_tf32 = True
    precision.allow_tf32 = True
    CUDALinearFunction.use_cublas = True             # CUDA（启用 cuBLAS TF32）
```

### 效果

FP32 严格模式下，三方精度差距从 1%+ 降到 0.04%：

```
PyTorch 98.44% | Triton 98.40% | CUDA 98.40%
```

## 4. Triton Grid 大小 Bug 修复

### 问题

Triton MLP 训练不收敛（准确率卡在 16%），原因是 backward kernel 的 grid 大小硬编码。

```python
# backward.py 中的错误代码
relu_backward_kernel[(triton.cdiv(n, 2048),)](grad_out, x, grad_in, n)
```

kernel 使用 `@triton.autotune` 可选 `BLOCK_SIZE=512/1024/2048/4096`。当 autotune 选择 `BLOCK_SIZE=512` 时：

```
需要 block 数 = ceil(n / 512)
实际 block 数 = ceil(n / 2048)   ← 只有需要的 1/4
→ 75% 的输出元素未被写入
→ torch.empty_like 返回垃圾值
→ 梯度爆炸
```

### 修复

```python
# 修改前：grid 硬编码
relu_backward_kernel[(triton.cdiv(n, 2048),)](...)

# 修改后：grid 使用 autotune 实际选择的 BLOCK_SIZE
grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
relu_backward_kernel[grid](...)
```

### 效果

训练从 16% → 96.87%，正常收敛。

### 教训

**任何使用 `@triton.autotune` 的 kernel，grid 必须使用 `lambda meta: ...` 引用 `meta["BLOCK_SIZE"]`，绝对不能硬编码。**

## 5. Triton Autotune 配置设计

### 问题

单组配置无法同时在大（RTX 5070 Ti, Blackwell）和小（RTX 3070 Laptop, Ampere）GPU 上达到最优。

### 优化策略

按 GPU 架构分层设计配置：

```python
_MATMUL_CONFIGS = [
    # --- 大 tile：Blackwell 最优 ---
    # SM 12.0: 更多 shared memory → 大 tile + 多 pipeline stages
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                  num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                  num_stages=3, num_warps=8),
    # --- 中 tile：Ampere 平衡 ---
    # SM 8.6: 较少 shared memory → 中等 tile + 适度 pipeline
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                  num_stages=4, num_warps=4),
    # --- 小 tile：MNIST 等小矩阵 ---
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                  num_stages=4, num_warps=4),
]
```

### 配置参数解释

| 参数 | 作用 | 大 tile | 小 tile |
|------|------|---------|---------|
| `BLOCK_M/N` | 输出 tile 大小 | 128x128 | 32x32 |
| `BLOCK_K` | 内积 tile 大小 | 32-64 | 32 |
| `num_stages` | 软件流水线深度 | 3 | 4 |
| `num_warps` | 每 block warp 数 | 8 | 4 |
| `GROUP_SIZE_M` | L2 super-group | 8 | 8 |

### 效果

训练步时从 10.1ms 降至 4.64ms（2.18x 加速），相对 PyTorch 从 2.4x 慢降到 1.36x 慢。

## 6. CUDA 推理路径优化

### 问题

初始 CUDA 层在推理时也经过 `torch.autograd.Function`，产生了不必要的开销（保存中间值用于反向、autograd 图构建等）。

### 优化

在 `CUDALinear.forward` 中添加推理模式快速路径：

```python
def forward(self, input):
    if not torch.is_grad_enabled():
        # 推理模式：跳过 autograd.Function 包装
        input_f = input.float()
        weight_f = self.weight.float()
        output = mlp_cuda.matmul_tiled_auto(input_f, weight_f)
        if self.bias is not None:
            output = output + self.bias.float()
        return output
    return CUDALinearFunction.apply(input, self.weight, self.bias)
```

同理，`CUDAActivation.forward` 也跳过 autograd：

```python
def forward(self, input):
    if not torch.is_grad_enabled():
        return _ACT_FORWARD[self.activation](input.float())
    return CUDAActivationFunction.apply(input, self.activation)
```

### 效果

CUDA 推理延迟从约 2.6x PyTorch 降到 1.20ms（0.82x PyTorch），成为最快后端。

## 7. CUDA WMMA 自适应 Dispatch

### 问题

不同矩阵尺寸的最优策略不同：
- 小矩阵（<128）：FP32 tiled 足够，FP16 反而引入不必要的精度损失和转换开销
- 大矩阵（≥512）：Tensor Core 的吞吐远超 FP32 CUDA core

### 解决方案

三级自适应 dispatch：

```cpp
void launch_matmul_tiled_auto(...) {
    int max_dim = max({M, N, K});

    if (max_dim >= 512) {
        // 大矩阵：WMMA FP16 Tensor Core
        // FP32 输入 → FP16 shared memory → FP32 累加 → FP32 输出
        matmul_wmma_kernel<<<...>>>(A, B, C, M, K, N);
    } else if (max_dim >= 128) {
        // 中等矩阵：FP32 shared memory tiled
        matmul_tiled_kernel<32, 32, 32><<<...>>>(A, B, C, M, K, N);
    } else {
        // 小矩阵：最小安全配置
        matmul_tiled_kernel<16, 16, 16><<<...>>>(A, B, C, M, K, N);
    }
}
```

### WMMA TransB/TransA 的特殊处理

反向传播需要 `dA = dC @ B^T` 和 `dB = A^T @ dC`。不能显式转置矩阵（额外显存和带宽），而是在 shared memory 中按转置方式加载：

```cpp
// C = A @ B^T: shared memory 存 B 的转置
sBT[n_local][k_local] = B[k * N + n]  // 按列读取 B，存为行

// C = A^T @ B: shared memory 存 A 的转置
sAT[k_local][m_local] = A[m * K + k]  // 按列读取 A，存为行
```

## 8. CUDA float4 向量化

### 问题

Activation backward 是访存密集型操作（读 2 个数组，写 1 个数组，计算简单）。每个线程处理 1 个元素时，内存带宽利用率低。

### 优化

使用 `float4`（128 位）一次读写 4 个 float：

```cpp
__global__ void relu_backward_vec4_kernel(
    const float* grad_output, const float* input, float* grad_input, int n)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 go = *reinterpret_cast<const float4*>(grad_output + idx);
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = (in.x > 0.0f) ? go.x : 0.0f;
        out.y = (in.y > 0.0f) ? go.y : 0.0f;
        out.z = (in.z > 0.0f) ? go.z : 0.0f;
        out.w = (in.w > 0.0f) ? go.w : 0.0f;
        *reinterpret_cast<float4*>(grad_input + idx) = out;
    }
}
```

### 效果

- 全局内存事务减为 1/4
- Grid 大小减为 1/4（减少调度开销）
- Activation backward 整体达到 PyTorch 的 2.6-6.8x 加速

## 9. 融合 Kernel 设计

### 动机

标准 MLP forward 的分离实现：

```
Linear:  output = input @ weight + bias    → 写 M*N 个 float 到显存
GELU:    output = GELU(output)             → 读 M*N，写 M*N

总计：1 次写 + 1 次读 + 1 次写 = 3 次显存事务（M*N 个 float）
```

融合实现：

```
融合:    output = GELU(input @ weight + bias)  → 写 1 次

总计：1 次写 = 1 次显存事务
节省：2 次显存事务（读+写 M*N 个 float）
```

### 实现关键

matmul 的累加器 `acc` 是寄存器变量，直接在寄存器中完成 bias add + activation：

```cpp
// 融合 kernel（CUDA 版）
float acc = 0.0f;
for (int k_tile = 0; ...;) {
    // ... tiled matmul 循环 ...
    acc += sX[threadIdx.y][kk] * sW[kk][threadIdx.x];
}
// acc 在寄存器中，直接做 activation
H[row * N + col] = gelu_device(acc + bias[col]);
```

```python
# 融合 kernel（Triton 版）
acc += tl.dot(x, w, allow_tf32=ALLOW_TF32)
# acc 在寄存器中，直接做 activation
acc = acc + b[None, :]
result = 0.5 * acc * (1.0 + tanh_inner)
tl.store(h_ptrs, result, ...)
```

### 融合收益数据

```
Fused MLP first layer vs 分离实现（benchmark_ops.py 数据）：
  CUDA:  0.87-1.78x 加速
  Triton: 0.98-1.68x 加速
```

收益在小矩阵上更大（MNIST 的 784x1024 算是小矩阵），因为小矩阵的计算时间短，内存延迟占比更高。

## 10. LayerNorm 三端实现对比

三端 LayerNorm 使用不同的归约策略：

### Triton

```python
# 单行一个 program，tl.sum 直接归约
x_sum = tl.sum(x, axis=0)
mean = x_sum / N
```

Triton 编译器自动处理归约的 shared memory/warp shuffle。代码最简洁。

### CUDA

```cpp
// Warp shuffle 归约
float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
// 多 warp → shared memory 中间结果 → 再次 warp 归约
```

手动管理归约层次：thread → warp → block。代码量大但控制精细。

### cuTile

```python
# 循环累加 + ct.sum 归约
mean_acc = ct.zeros((1, TILE), dtype=ct.float32)
for j in range(num_tiles):
    x_tile = ct.load(X, index=(row, j), shape=(1, TILE), ...)
    mean_acc = mean_acc + x_tile
mean_val = ct.sum(mean_acc, axis=1, keepdims=True) / N
```

需要手动分 tile 累加，再用 `ct.sum` 做 tile 内归约。代码量居中。

### 关键差异

| 方面 | Triton | CUDA | cuTile |
|------|--------|------|--------|
| 归约方式 | `tl.sum` 编译器自动 | warp shuffle 手动 | `ct.sum` tile 归约 |
| 代码复杂度 | 低 | 高 | 中 |
| 控制精细度 | 低 | 高 | 中 |
| Backward atomic_add | `tl.atomic_add` | `atomicAdd` | `ct.atomic_add` (scatter) |

## 11. 性能数据总结

### 端到端训练对比（FP32, 15 epochs）

| 指标 | PyTorch | Triton | CUDA |
|------|---------|--------|------|
| 准确率 | 98.71% | 98.61% | 98.64% |
| 训练时间 | 89.9s | 100.9s | 89.8s |
| 训练步延迟 | 3.30ms | 12.35ms | 8.85ms |
| 推理延迟 | 1.46ms | 2.13ms | **1.20ms** |
| 推理吞吐量 | 174.8K | 119.7K | **213.2K** |

### 算子级对比（48 项，PyTorch=1.0x）

| 算子类别 | CUDA | Triton |
|----------|------|--------|
| Matmul forward | 0.33-1.01x | 0.64-0.96x |
| Activation backward | 2.6-6.8x | 4.0-4.85x |
| Matmul backward | 1.2-6.5x | 2.0-4.0x |
| Fused MLP | 0.87-1.78x | 0.98-1.68x |
| **整体平均** | **2.53x** | **2.04x** |

### 关键发现

1. **自定义 kernel 在 backward 上优势明显**：绕过 PyTorch autograd 图开销，直接调用专用 kernel
2. **自定义 kernel 在 forward matmul 上不如 cuBLAS**：cuBLAS 的多年优化积累难以匹敌
3. **融合 kernel 是最有价值的优化方向**：减少显存带宽瓶颈
4. **推理路径优化效果显著**：跳过 autograd 封装，CUDA 推理延迟降至 PyTorch 的 0.82x
5. **精度对齐是公平对比的前提**：统一 TF32/FP32 后，精度差异从 1%+ 降到 0.04%

---

## Decision Tree / Playbook

把"何时选哪个 backend / kernel"压缩成一棵决策树,避免每次直觉摇号。
按场景从上至下匹配:

| 场景 | 推荐 backend | 备注 |
|------|--------------|------|
| shape 是 MNIST-class MLP weight,dtype=fp32,大 batch | `matmul_wmma64` (auto dispatch) > Triton autotune > torch.matmul | WMMA64 实测推理延迟 0.30ms,显著领先 |
| K 不能被 32 整除 | cuTile(Python 端切 K)或 Triton(autotune 自适应) | WMMA64 要求 K 对齐 16 |
| dtype=fp16 / bf16 | Triton > cuTile > torch autocast | 仓库 WMMA fp16 路径未完成 |
| batch B ≥ 4096 forward | cuTile(Python loop tile + 少 L2 压力)| ct.mma 单 thread block 容量饱和 |
| 需要 LayerNorm 融合到 GEMM | 新写专用 fused kernel(参考 `mlp_fused_first_layer`)| cuTile 是 per-op,不便融合 |
| 单算子推理延迟优先 | CUDA(若已修好 WMMA64 K-loop) | LayerNorm 与 GEMM 之间无融合机会 |
| 训练步整体延迟优先 | PyTorch cuBLAS | 4-backend 实测仍最快(2026-06-03) |
| 在 MLP_LAYERS 上观察到 correctness drift | `python tools/analyze_bench.py BASELINE CAND --shape MLP_LAYERS --warn-l2 1e-3` | 锁定到具体 (op,shape) |

> 这棵树的输入是"约束(shape+dtype+目标)",输出是"先试哪个 backend"。
> 任何选择前后必须经过 `make gate` 验证;若 gate fail,立刻回退或修。

---

## cuTile 优化指南

cuTile (Python `cuda.tile` API) 与 CUDA / Triton 的区别:

| 维度 | cuTile | CUDA | Triton |
|------|--------|------|--------|
| Tile 描述 | Python `ct.Tile` 对象 | C++ 模板参数 | Triton `tl.constexpr` |
| K 维循环 | Python 端(host) | kernel 内 for-loop | Triton 自动 |
| Reduction | `ct.sum` + `ct.atomic_add` | warp shuffle + shared | `tl.sum` |
| TMA 用法 | `ct.load`/`ct.store` 隐式 TMA | 手动 cp.async | Triton 自动 |
| 编译时间 | 首次 ~3-8s | 一次性 nvcc | 首次 + autotune cache |
| 适用尺寸 | 中等 (256-4096) | 全尺寸 | 全尺寸 |

**Tile shape 选择**:
- 输出 tile 大小通常等于 `get_arch_params()["cutile_mma_tile"]`(SM_120=64×64,SM_86=32×32)
- K 维步长用 `cutile_mma_tile_k`(SM_120=32);较小 K 时 Python 端循环切片
- elementwise:用 `cutile_elementwise_tile` (默认 4096);超大时 Python 切

**Python 循环 vs kernel 循环**:
- Python 循环(host 端) → 灵活,容易调试,kernel launch 多
- Kernel 循环(device 端) → 启动开销小,但要 unroll 控制 register

实测对比(RTX 5070 Ti, M=512 K=768 N=3072):
- cuTile matmul: 1.09 ms ≈ CUDA WMMA32 1.10 ms,Triton autotune 1.6-2.1 ms

**如何读 `results/cutile_bench.json`**:
- 顶层 `metadata`: GPU/torch/git_sha 等
- `ops.<op_name>`: 单算子 4 轮取后 3 均值
- `mlp.mlp_train_step` / `mlp_infer_step`: 端到端 1 step
- `discarded`: 第 1 轮丢弃(因 lazy compile / autotune)

---

## torch.compile / fp16 / bf16 recipes

### `torch.compile`

```python
import torch
torch.set_float32_matmul_precision("high")  # 启用 TF32 + bf16 自动转换
model = MLP(config).cuda()
model = torch.compile(model, mode="reduce-overhead")  # 适合训练 step
```

适用条件:
- 模型结构稳定(无 dynamic control flow)
- 训练步 ≥ 200 次(摊销 compile 开销)
- 不与自定义 `torch.autograd.Function` 混用(Triton/CUDA backend 不兼容)

### Mixed Precision (autocast + GradScaler)

```python
from torch.amp import autocast, GradScaler
scaler = GradScaler()
with autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(x)
    loss = criterion(logits, y)
scaler.scale(loss).backward()
scaler.step(optimizer); scaler.update()
```

注意:
- `bf16` 在 SM_80+ (Ampere/Blackwell) 上推荐(更宽 dynamic range);`fp16` 上易溢出需 GradScaler
- WMMA fp16 路径在本仓库未实现,autocast + 自定义 kernel 会 fall back 到 fp32

### Numerical Verification After AMP

```python
fp32_out = model(x.float())
bf16_out = autocast_model(x)
torch.testing.assert_close(fp32_out, bf16_out, atol=1e-2, rtol=1e-2)
```

工具:`python tools/analyze_bench.py BASELINE CAND --metric tflops --warn-l2 1e-2` 可一次性看 perf + numerics 双指标。
