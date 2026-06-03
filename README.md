# MLP-Kernel-Lab

面向 Transformer MLP 推理的自定义 CUDA & Triton & cuTile kernel 实验，以性能分析驱动优化，并与 PyTorch 基线进行对比。支持 MNIST MLP 的 PyTorch / Triton / CUDA / cuTile 四后端端到端训练、推理及详细性能对比。

## 项目亮点

- **四后端对比**：PyTorch (cuBLAS) / Triton / CUDA / cuTile 端到端训练对比
- **CUDA kernel**：naive / shared-memory tiled / fused activation / WMMA FP16 / LayerNorm 多版本实现
- **Triton kernel**：完整 MLP 训练算子（matmul、elementwise、backward、dropout、loss、layernorm、fused SwiGLU）
- **cuTile kernel**：NVIDIA cuTile Python tile-based GPU 编程（matmul、elementwise、backward、layernorm、fused MLP、SwiGLU）
- **精度控制**：TF32 / FP32 全局切换，多后端公平对比
- **LayerNorm**：Triton + CUDA + cuTile 三端实现，forward + backward
- **autograd 集成**：TritonLinear/CUDALinear/CUTILELinear 等 `torch.autograd.Function` 层，可直接嵌入 PyTorch 训练循环
- **完整测试**：55 项 Python 测试 + 6 项 C++ CUDA 测试
- **Nsight Compute profiling** & Chrome trace 导出

## 项目结构

```
MLP-Kernel-Lab/
├── kernels/                    # CUDA kernel 实现
│   ├── vector_add.cu           #   CUDA 基础 & 计时
│   ├── matmul_naive.cu         #   朴素矩阵乘法 (CMake standalone)
│   ├── matmul_tiled.cu         #   shared memory 分块矩阵乘法 (CMake standalone)
│   ├── activation.cu           #   GELU / SiLU device 函数 (CMake standalone)
│   ├── mlp_fused_first_layer.cu#   融合 matmul+bias+GELU (CMake standalone)
│   ├── swiglu_fused.cu         #   融合 SwiGLU (CMake standalone)
│   ├── binding.cpp             #   PyTorch C++ extension 绑定
│   └── mlp/                    # PyTorch extension kernel (按算子族拆分)
│       ├── device_utils.cuh    #   公共 device 函数 (gelu/silu/warp_reduce)
│       ├── wmma_decl.cuh       #   WMMA kernel 前向声明
│       ├── matmul.cu           #   naive + tiled + transA + transB + bias_add
│       ├── wmma.cu             #   WMMA FP16 Tensor Core 6 个 kernel
│       ├── activation.cu       #   GELU/ReLU/SiLU forward + backward (+ vec4)
│       ├── fused.cu            #   mlp_fused_first_layer + swiglu_fused
│       ├── layernorm.cu        #   LayerNorm forward + backward (warp shuffle)
│       ├── softmax.cu          #   逐行 softmax (数值稳定)
│       └── pool_im2col.cu      #   MaxPool / AvgPool / im2col
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
├── cutile_kernels/             # cuTile kernel 实现
│   ├── matmul.py               #   分块矩阵乘法（ct.mma）
│   ├── elementwise.py          #   BiasAdd、ReLU、GELU、SiLU、融合 BiasAdd+ReLU
│   ├── backward.py             #   MatMul backward、Activation backward
│   ├── layernorm.py            #   LayerNorm forward/backward
│   ├── mlp_cutile.py           #   融合 matmul+bias+GELU
│   └── swiglu_cutile.py        #   融合 SwiGLU
├── python/mnist/               # MNIST 训练子项目
│   ├── model.py                #   PyTorch MLP 模型
│   ├── triton_model.py         #   Triton MLP 模型
│   ├── cuda_model.py           #   CUDA MLP 模型
│   ├── cutile_model.py         #   cuTile MLP 模型
│   ├── triton_layers.py        #   TritonLinear/TritonLayerNorm autograd 层
│   ├── cuda_layers.py          #   CUDALinear/CUDALayerNorm autograd 层
│   ├── cutile_layers.py        #   CUTILELinear/CUTILELayerNorm autograd 层
│   ├── trainer.py              #   训练器（AdamW + CosineAnnealing）
│   ├── benchmark.py            #   CUDA Event 精确基准测试
│   └── dashboard.py            #   控制台输出 & JSON 导出
├── tests/                      # 测试
│   ├── test_triton_kernels.py  #   21 项 Triton kernel 正确性测试
│   ├── test_cuda_kernels.py    #   16 项 CUDA kernel 正确性测试
│   ├── test_cutile_kernels.py  #   18 项 cuTile kernel 正确性测试
│   └── test_kernels.cu         #   6 项 C++ CUDA 单元测试
├── configs/
│   └── mnist_mlp.yaml          # MNIST 训练配置
├── run_compare.py              # 四后端对比训练（PyTorch/Triton/CUDA/cuTile）
├── benchmark_ops.py            #   算子级横向对比
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
- cuTile 1.3+（可选，`pip install cuda-tile`）

> **WSL 用户注意**：仓库放在 `\\wsl.localhost\Ubuntu\...` 下时，Windows 端 git 会报 `dubious ownership`。在 Windows shell 执行一次：
>
> ```bash
> git config --global --add safe.directory '%(prefix)///wsl.localhost/Ubuntu/home/<user>/projects/MLP-Kernel-Lab'
> ```

### 安装

```bash
pip install -r requirements.txt
pip install cuda-tile            # 可选：启用 cuTile 后端
make install                     # 构建并安装 CUDA extension
```

### 检查环境

```bash
make check
```

### 四后端对比训练

```bash
python run_compare.py --cuda --cutile --precision fp32 --epochs 15   # 四后端 FP32 对比
python run_compare.py --cuda --cutile --precision tf32 --epochs 10   # 四后端 TF32 对比
python run_compare.py --cuda --epochs 5                              # PyTorch vs Triton vs CUDA
python run_compare.py --epochs 5                                     # PyTorch vs Triton
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
| `--cuda` | 启用 CUDA 后端 | - |
| `--cutile` | 启用 cuTile 后端 | - |
| `--no-bench` | 跳过基准测试 | - |

### 算子级基准测试

```bash
python benchmark_ops.py --precision fp32 --warmup 20 --iters 100
```

### 测试

```bash
make test-python       # Python 测试（55 项）
make test-cuda         # C++ CUDA 测试（6 项）
```

### Profiling

仓库提供 3 个互补 nsight 入口（详见 `profiling/README.md`）：

```bash
# 1. ncu (kernel 级 micro-arch 指标)
bash profiling/run_ncu.sh tiled              # CUDA tiled matmul auto-dispatch
bash profiling/run_ncu.sh cuda               # CUDA matmul + LayerNorm + activation
bash profiling/run_ncu.sh cutile             # cuTile matmul (需 cuda-tile)
bash profiling/run_ncu.sh mlp-cuda           # CUDAMLP 端到端 1 step
bash profiling/run_ncu.sh mlp-cutile         # CUTILEMLP 端到端 1 step
bash profiling/run_ncu.sh triton             # Triton MLP 1 step
bash profiling/run_ncu.sh compare            # PyTorch vs Triton torch.profiler

# 2. nsys (时间线 / NVTX 折叠 / 跨 step)
bash profiling/profile_nsys.sh tiled         # 单 kernel 重复 timeline
bash profiling/profile_nsys.sh mlp-cuda      # CUDAMLP 5 step timeline
bash profiling/profile_nsys.sh compare       # PyTorch vs Triton vs CUDA

# 3. 算子级 driver (NVTX 包裹, 任意维度, 跨 backend)
python profiling/profile_ops.py                          # 全 backend 全算子
python profiling/profile_ops.py --backend cuda triton    # 仅指定 backend
python profiling/profile_ops.py --ops matmul --M 2048 --K 2048 --N 2048
nsys profile -t cuda,nvtx -o results/nsys_ops python profiling/profile_ops.py
```

Makefile 短路:`make profile-tiled` / `make profile-nsys` / `make profile-ops` / `make profile-compare`。

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
| `cutile_matmul` | ct.mma 分块矩阵乘法 | 已完成 |
| `cutile_elementwise` | BiasAdd, ReLU, GELU, SiLU + backward | 已完成 |
| `cutile_layernorm` | LayerNorm forward + backward (ct.sum + atomic_add) | 已完成 |
| `cutile_fused` | 融合 matmul + bias + GELU | 已完成 |
| `cutile_swiglu` | 融合 SwiGLU | 已完成 |

## 性能参考

FP32 模式，模型 `[784,1024,512,256,10]`，ReLU + LayerNorm + Dropout=0.1，15 epochs：

| 指标 | PyTorch | Triton | CUDA | cuTile |
|------|---------|--------|------|--------|
| 准确率 | 98.71% | 98.61% | 98.64% | - |
| 训练时间 | 89.9s | 100.9s | 89.8s | - |
| 训练步延迟 | 3.30ms | 12.35ms | 8.85ms | - |
| 推理延迟 | 1.46ms | 2.13ms | **1.20ms** | - |
| 推理吞吐量 | 174.8K | 119.7K | **213.2K** | - |

> cuTile 列 `-` 表示尚未采集数据(需 cuTile 1.3+ 安装)。复现命令:
>
> ```bash
> pip install cuda-tile
> python run_compare.py --cuda --cutile --precision fp32 --epochs 15 \
>     | tee results/four_backend_fp32.txt
> ```
>
> 完成后将结果填入上表(参考 CHANGELOG 2026-05-27 #1 测得的正确性数据,只缺端到端延迟)。

### cuTile 单 step 微基准 (RTX 5070 Ti, FP32, B=256, 2026-06-02)

`python profiling/bench_cutile.py` 实测,每项 4 轮取后 3 轮均值(详见 `results/cutile_bench.json`):

| 算子 | mean ms | std |
|------|---------|-----|
| matmul (512×768·768×3072) | 1.091 | 0.001 |
| matmul_backward_a (dA = dC@B^T) | 1.375 | 0.001 |
| matmul_backward_b (dB = A^T@dC) | 2.033 | 0.003 |
| bias_add (512×3072) | 0.011 | 0.001 |
| gelu / silu / relu | 0.009–0.015 | — |
| gelu_backward | 0.010 | 0.000 |
| layernorm forward | 0.020 | 0.000 |
| layernorm backward | 0.259 | 0.001 |
| mlp_fused_first_layer (matmul+bias+GELU) | 1.069 | 0.001 |
| swiglu | 0.010 | 0.003 |

端到端 CUTILEMLP `[784,1024,512,256,10]` + LayerNorm + Dropout 0.1,batch=256:

| 指标 | cuTile (RTX 5070 Ti, 单 step) |
|------|----|
| 训练步延迟 | **2.256 ms** |
| 推理延迟 | **0.612 ms** |
| 推理吞吐量 | **418,556 samples/sec** |

> 注:此小节是 1 step 微基准,**未跑 15-epoch 完整训练**,不与上表同列直接比较 GPU 差异 — 原表在不同 GPU/不同 batch 上测得。要并列对照,请用 `run_compare.py --cuda --cutile --precision fp32 --epochs 15`(同机)采一份完整数据,再填回上表。

### 完整 4-backend 实测 (RTX 5070 Ti, FP32, 15 epoch, 2026-06-02)

同机同 config 跑 `python run_compare.py --cuda --cutile --precision fp32 --epochs 15`,模型 `[784,1024,512,256,10]` + LayerNorm + Dropout 0.1,batch=256。完整日志 `results/four_backend_fp32_v2.log`,benchmark JSON `results/compare_20260602_235625.json`。

| 指标 | PyTorch | Triton | CUDA | cuTile |
|------|---------|--------|------|--------|
| best val_acc | 98.71% | 98.62% | 98.65% | 98.52% |
| best at epoch | 12 | 15 | 15 | 14 |
| 训练时间(总) | 27.4s | 45.0s | **27.0s** | 30.4s |
| 训练步延迟 (median) | **1.500 ms** | 2.083 ms | 1.719 ms | 2.616 ms |
| 训练步延迟 (p95) | 3.522 ms | 5.517 ms | 4.594 ms | 6.047 ms |
| 训练吞吐量 (samples/s) | **170,698** | 122,889 | 148,936 | 97,845 |
| 推理延迟 (median) | 0.340 ms | 0.649 ms | **0.298 ms** | 0.711 ms |
| 推理吞吐量 (samples/s) | 753,331 | 394,487 | **858,139** | 360,044 |

**观察:**
- 4 后端精度差 ≤ 0.19pp,全部在 FP16 WMMA + TF32 + tanh-GELU 近似的合理噪声范围内。
- **CUDA backend**: 训练总时长最短(27.0s),推理延迟最低(0.298 ms / 858K samples/s)。WMMA64 FP16 Tensor Core 路径修复后启用,L0 / L1 大 matmul 在 64×64 tile 上提速。
- **PyTorch baseline (cuBLAS)**: 训练 step 最快(1.500ms),依赖高度优化的 cuBLAS GEMM。
- **Triton / cuTile**: 训练步比 PyTorch 慢 39–74%,主要来自 autotune cache miss + autograd graph 开销;推理路径同样差距(0.65–0.71ms vs 0.30–0.34ms)。
- **cuTile bench_cutile.py 单 step 0.61ms** 与此处端到端 0.71ms 一致(差距来自 dataloader + autograd 包装)。

### 跨模型 4-backend 对比 (RTX 5070 Ti, FP32 strict, 15 epoch, 4 轮末轮)

测试 3 个不同 MLP 拓扑对 4 backend 表现的影响,数据落 `results/full_eval_20260604_011723/model_*/`。运行:

```bash
MODELS=default,deep_narrow,wide_skip bash tools/run_full_eval.sh
```

| 模型 | 架构 | 参数量 | 测试目的 |
|------|------|--------|----------|
| `default` | 784→1024→512→256→10 + LN | 1.46M | 现行 baseline |
| `deep_narrow` | 784→256×8→10 + 每层 LN | 0.60M | 深度 + LN 多次摊销 |
| `wide_skip` | 784→1024×3→10 + 残差(后 2 层) + LN | 2.92M | 残差 + 大 hidden |

**default (4-layer, 1.46M):**

| backend | val_acc | 训练时间 | step_med | 吞吐 |
|---------|---------|----------|----------|------|
| PyTorch | **0.9871** | **25.1s** | 1.16ms | 221,395 |
| Triton  | 0.9865 | 45.3s | 2.27ms | 112,780 |
| CUDA    | 0.9867 | 26.8s | **1.08ms** | **237,248** |
| cuTile  | 0.9864 | 27.3s | 2.39ms | 107,014 |

**deep_narrow (8×256 + LN, 0.60M):**

| backend | val_acc | 训练时间 | step_med | 吞吐 |
|---------|---------|----------|----------|------|
| PyTorch | 0.9849 | 25.4s | 1.94ms | 132,232 |
| Triton  | 0.9853 | 40.9s | 2.62ms | 97,633 |
| CUDA    | **0.9858** | **26.1s** | **1.86ms** | **137,670** |
| cuTile  | 0.9847 | 27.6s | 2.10ms | 122,174 |

**wide_skip (3×1024 + 残差 + LN, 2.92M):**

| backend | val_acc | 训练时间 | step_med | 吞吐 |
|---------|---------|----------|----------|------|
| PyTorch | **0.9871** | **24.8s** | 1.30ms | 197,095 |
| Triton  | 0.9851 | 38.6s | 1.69ms | 151,216 |
| CUDA    | 0.9861 | 25.2s | 1.30ms | 196,437 |
| cuTile  | 0.9863 | 26.4s | 4.03ms | 63,477 |

**关键观察:**

1. **Triton 永远最慢 (38.6-45.3s)**,3 个模型一致。Python autograd + autotune 是固有问题,不是 kernel 不好。
2. **cuTile 在 deep_narrow 上追平 CUDA** (27.6 vs 26.1s,差距 <6%):(16,16,16) mma fragment 摊销好。
3. **cuTile 在 wide_skip 上 step_med 退化到 4.03ms**(default 2.39 的 1.7x):大 hidden 残差路径触发 `torch.add` 同步点。
4. **PyTorch (cuBLAS) 跨模型稳定** (24.8-25.4s):hidden dim 切换不敏感。
5. **val_acc 跨模型**:default ≈ wide_skip (0.987) > deep_narrow (0.985)。8 层窄 + 多次 LN 在 MNIST 上轻微欠拟合。

**场景化 backend 选择:**

| 场景 | 推荐 | 理由 |
|------|------|------|
| 浅 + 任意 hidden | PyTorch / CUDA | cuBLAS 已充分优化 |
| 深 + 多次小 matmul + LN | cuTile | 16×16×16 fragment 摊销 |
| 残差 + 大 hidden | PyTorch / CUDA | 残差 add 同步 + cuTile 大 tile 退化 |
| 生产路径(无教学需求) | PyTorch | 最快 + 最高 val_acc |
| 教学 / 算子对比 | 全部 4 backend | 暴露 kernel 设计取舍 |

## 许可证

MIT
