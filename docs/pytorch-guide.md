# PyTorch MLP 基线实现指南

> 本项目的 PyTorch 基线：一个可配置的多层 MLP，作为所有自定义 kernel 的对比参照。

## 目录

- [1. 模型架构](#1-模型架构)
- [2. MLPConfig 配置](#2-mlpconfig-配置)
- [3. MLP 模型实现](#3-mlp-模型实现)
- [4. 训练流程](#4-训练流程)
- [5. PyTorch 自动优化机制](#5-pytorch-自动优化机制)
- [6. 为什么 PyTorch 很难被打败](#6-为什么-pytorch-很难被打败)

---

## 1. 模型架构

本项目测试架构为 `[784, 1024, 512, 256, 10]`：

```
输入 (B, 784)
  → Linear(784, 1024) → LayerNorm(1024) → ReLU → Dropout(0.1)
  → Linear(1024, 512) → LayerNorm(512)  → ReLU → Dropout(0.1)
  → Linear(512, 256)  → LayerNorm(256)  → ReLU → Dropout(0.1)
  → Linear(256, 10)
输出 (B, 10) logits
```

每层结构为 `Linear → LayerNorm → ReLU → Dropout`，最后一层无激活。

## 2. MLPConfig 配置

```python
@dataclass
class MLPConfig:
    hidden_dims: list[int]     # 各层维度，如 [784, 1024, 512, 256, 10]
    activation: str            # relu | gelu | silu
    dropout: float             # dropout 率
    use_layernorm: bool        # 是否使用 LayerNorm
```

`hidden_dims` 列表长度决定层数。列表中相邻两个元素构成一层 `Linear(dim[i], dim[i+1])`。

## 3. MLP 模型实现

```python
class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_activation = []

        dims = config.hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = (i == len(dims) - 2)
            if not is_last:
                self.norms.append(nn.LayerNorm(dims[i + 1]) if config.use_layernorm else nn.Identity())
                self.activations.append(_ACTIVATIONS[config.activation]())
                self.use_activation.append(True)
            else:
                self.norms.append(nn.Identity())
                self.activations.append(nn.Identity())
                self.use_activation.append(False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(1)
        for i, linear in enumerate(self.layers):
            x = linear(x)
            x = self.norms[i](x)
            x = self.activations[i](x)
            if self.use_activation[i]:
                x = self.dropout(x)
        return x
```

关键设计决策：

| 决策 | 原因 |
|------|------|
| `nn.ModuleList` 而非 `nn.Sequential` | 需要在每层之间插入 LayerNorm、Dropout 等操作 |
| `use_activation` 列表 | 最后一层不需要激活，但 norm 和 activation 都需要占位以保持索引对齐 |
| GELU 使用 `approximate="tanh"` | 与 Triton/CUDA/cuTile 实现对齐，三者均使用 tanh 近似 |
| `x.flatten(1)` | MNIST 输入为 (B, 1, 28, 28)，展平为 (B, 784) |

## 4. 训练流程

```
Trainer.fit(train_loader, test_loader, epochs=15)
  ├── optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
  ├── scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
  └── 对每个 epoch:
        ├── train: 前向 → CrossEntropyLoss → 反向 → optimizer.step()
        └── eval:  torch.no_grad() 下计算准确率
```

训练器使用 `AdamW + CosineAnnealingLR`，对所有后端统一。

## 5. PyTorch 自动优化机制

理解 PyTorch 的内部优化有助于理解为什么自定义 kernel 很难超越它：

### 5.1 cuBLAS / cuDNN 集成

PyTorch 的 `torch.matmul` 在 GPU 上直接调用 NVIDIA cuBLAS：

```
torch.matmul(A, B)
  → at::matmul
    → at::cuda::blas::gemm          // cuBLAS GEMM
      → cublasSgemm / cublasGemmEx   // 自动选择 TF32/FP32/FP16
```

cuBLAS 内部有：
- **Tensor Core 自动调度**：Ampere+ 自动使用 TF32 tensor core
- **多算法搜索**：`cublasGemmEx` 内部有 heuristic 选择最优算法
- **高度优化的内存布局**：针对连续/转置输入有不同优化路径

### 5.2 autograd 优化

```python
output = F.linear(input, weight, bias)  # 内部 = input @ weight.T + bias
loss = F.cross_entropy(output, target)
loss.backward()
```

PyTorch autograd 引擎的优化：
- **反向计算图融合**：多个小操作可自动融合
- **内存高效反向**：`grad_input` 和 `grad_weight` 可就地计算
- **异步执行**：CPU 和 GPU 流水线化，`loss.backward()` 返回时反向可能还在 GPU 上执行

### 5.3 推理优化

`torch.no_grad()` 下的推理路径：
- 跳过 autograd 计算图构建
- `nn.Linear.forward` 直接调用 `F.linear`
- 无需保存中间激活用于反向

## 6. 为什么 PyTorch 很难被打败

本项目实测数据（FP32, 15 epochs）：

| 指标 | PyTorch | Triton | CUDA |
|------|---------|--------|------|
| 准确率 | 98.71% | 98.61% | 98.64% |
| 训练时间 | 89.9s | 100.9s | 89.8s |
| 推理延迟 | 1.46ms | 2.13ms | **1.20ms** |

PyTorch 在训练时间上几乎最优（与自定义 CUDA 持平），原因是：

1. **cuBLAS 几十年优化积累**：针对每种 GPU 架构、每种矩阵尺寸都有专门调优
2. **autograd 引擎开销极低**：C++ 实现，计算图构建几乎零开销
3. **融合 bias add**：cuBLAS 的 GEMM 可以融合 bias（`cublasGemmStridedBatchedEx`）

自定义 kernel 能胜出的场景：
- **推理延迟**：CUDA 后端 1.20ms vs PyTorch 1.46ms（自定义 kernel 跳过 autograd 包装）
- **融合 kernel**：matmul+bias+activation 合并为单次 kernel launch
- **专用小矩阵**：cuBLAS 对小矩阵的 heuristic 可能不够优，自定义 kernel 可以针对性调优
