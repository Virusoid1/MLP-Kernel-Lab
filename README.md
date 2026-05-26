# MLP-Kernel-Lab

面向 Transformer MLP 推理的自定义 CUDA & Triton kernel 实验，以性能分析驱动优化，并与 PyTorch 基线进行对比。支持 MNIST MLP 的 PyTorch / Triton / CUDA 三后端端到端训练、推理及详细性能对比。

## 项目亮点

- **三后端对比**：PyTorch (cuBLAS) / Triton / CUDA 自定义 kernel 端到端训练对比
- **CUDA kernel**：naive / shared-memory tiled / fused activation / WMMA FP16 / LayerNorm 多版本实现
- **Triton kernel**：完整 MLP 训练算子（matmul、elementwise、backward、dropout、loss、layernorm、fused SwiGLU）
- **精度控制**：TF32 / FP32 全局切换，三方后端公平对比
- **LayerNorm**：Triton + CUDA 双端实现，forward + backward
- **autograd 集成**：TritonLinear/CUDALinear 等 `torch.autograd.Function` 层，可直接嵌入 PyTorch 训练循环
- **完整测试**：37 项 Python 测试 + 6 项 C++ CUDA 测试
- **Nsight Compute profiling** & Chrome trace 导出

## 项目结构

```
MLP-Kernel-Lab/
├── kernels/                    # CUDA kernel 实现
│   ├── vector_add.cu           #   CUDA 基础 & 计时
│   ├── matmul_naive.cu         #   朴素矩阵乘法
│   ├── matmul_tiled.cu         #   shared memory 分块矩阵乘法
│   ├── activation.cu           #   GELU / SiLU device 函数
│   ├── mlp_fused_first_layer.cu#   融合 matmul+bias+GELU
│   ├── swiglu_fused.cu         #   融合 SwiGLU
│   ├── mlp_cuda_kernels.cu     #   kernel launch 封装（含 LayerNorm）
│   └── binding.cpp             #   PyTorch C++ extension 绑定
├── triton_kernels/             # Triton kernel 实现
│   ├── matmul.py               #   分块矩阵乘法（L2 缓存优化）
│   ├── elementwise.py          #   BiasAdd、ReLU、GELU、SiLU、融合 BiasAdd+ReLU
│   ├── backward.py             #   MatMul backward、Activation backward
│   ├── layernorm.py            #   LayerNorm forward/backward
│   ├── dropout.py              #   Inverted dropout
│   ├── loss.py                 #   融合 CrossEntropy
│   ├── mlp_triton.py           #   融合 matmul+bias+GELU
│   ├── swiglu_triton.py        #   融合 SwiGLU
│   ├── precision.py            #   全局精度控制（TF32/FP32）
│   └── gpu_utils.py            #   GPU 检测模块
├── python/mnist/               # MNIST 训练子项目
│   ├── model.py                #   PyTorch MLP 模型
│   ├── triton_model.py         #   Triton MLP 模型
│   ├── cuda_model.py           #   CUDA MLP 模型
│   ├── triton_layers.py        #   TritonLinear/TritonLayerNorm autograd 层
│   ├── cuda_layers.py          #   CUDALinear/CUDALayerNorm autograd 层
│   ├── trainer.py              #   训练器（AdamW + CosineAnnealing）
│   ├── benchmark.py            #   CUDA Event 精确基准测试
│   └── dashboard.py            #   控制台输出 & JSON 导出
├── tests/                      # 测试
│   ├── test_triton_kernels.py  #   21 项 Triton kernel 正确性测试
│   ├── test_cuda_kernels.py    #   16 项 CUDA kernel 正确性测试
│   └── test_kernels.cu         #   6 项 C++ CUDA 单元测试
├── configs/
│   └── mnist_mlp.yaml          # MNIST 训练配置
├── run_compare.py              # 三后端对比训练（PyTorch/Triton/CUDA）
├── benchmark_ops.py            # 算子级横向对比
├── profiling/                  # Nsight Compute 脚本
├── setup.py                    # CUDA extension 构建配置
├── Makefile                    # install / test / bench / profile
└── requirements.txt
```

## 快速开始

### 环境要求

- NVIDIA GPU（compute capability 7.5+）
- CUDA Toolkit 11.8+
- Python 3.8+
- PyTorch 2.0+
- Triton 2.0+

### 安装

```bash
pip install -r requirements.txt
make install          # 构建并安装 CUDA extension
```

### 检查环境

```bash
make check
```

### 三后端对比训练

```bash
python run_compare.py --cuda --precision fp32 --epochs 15    # 三后端 FP32 对比
python run_compare.py --cuda --precision tf32 --epochs 10    # 三后端 TF32 对比
python run_compare.py --epochs 5                             # PyTorch vs Triton
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 配置文件 |
| `--batch-size` | 批大小 | 256 |
| `--lr` | 学习率 | 0.001 |
| `--hidden` | 隐藏层维度 | 784,1024,512,256,10 |
| `--activation` | 激活函数 (relu/gelu/silu) | relu |
| `--dropout` | Dropout 率 | 0.1 |
| `--precision` | 精度模式 (tf32/fp32) | tf32 |
| `--cuda` | 启用 CUDA 后端（三方对比） | - |
| `--no-bench` | 跳过基准测试 | - |

### 算子级基准测试

```bash
python benchmark_ops.py --precision fp32 --warmup 20 --iters 100
```

### 测试

```bash
make test-python       # Python 测试（37 项）
make test-cuda         # C++ CUDA 测试（6 项）
```

### Profiling

```bash
bash profiling/run_ncu.sh triton          # Nsight 分析 Triton
bash profiling/run_ncu.sh compare         # torch.profiler 对比
```

## 实现状态

| 实现 | 说明 | 状态 |
|------|------|------|
| `torch` | `torch.matmul` + `F.gelu` 基线 | 已完成 |
| `triton_mlp` | TritonLinear + TritonLayerNorm 端到端训练 | 已完成 |
| `triton_matmul` | `tl.dot` 分块矩阵乘法（L2 缓存优化 + autotune） | 已完成 |
| `triton_elementwise` | BiasAdd, ReLU, GELU, SiLU + backward | 已完成 |
| `triton_layernorm` | LayerNorm forward + backward | 已完成 |
| `triton_loss` | 融合 CrossEntropy | 已完成 |
| `triton_fused` | 融合 matmul + bias + GELU | 已完成 |
| `triton_swiglu` | 融合 SwiGLU | 已完成 |
| `cuda_tiled` | shared memory 分块矩阵乘法 | 已完成 |
| `cuda_fused` | 融合 matmul + bias + GELU | 已完成 |
| `cuda_layernorm` | LayerNorm forward + backward (warp shuffle) | 已完成 |
| `cuda_activation` | ReLU / GELU / SiLU forward + backward (vec4) | 已完成 |

## 性能参考

FP32 模式，模型 `[784,1024,512,256,10]`，ReLU + LayerNorm + Dropout=0.1，15 epochs：

| 指标 | PyTorch | Triton | CUDA |
|------|---------|--------|------|
| 准确率 | 98.71% | 98.61% | 98.64% |
| 训练时间 | 89.9s | 100.9s | 89.8s |
| 训练步延迟 | 3.30ms | 12.35ms | 8.85ms |
| 推理延迟 | 1.46ms | 2.13ms | **1.20ms** |
| 推理吞吐量 | 174.8K | 119.7K | **213.2K** |

## 许可证

MIT
