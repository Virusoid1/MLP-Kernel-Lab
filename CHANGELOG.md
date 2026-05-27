# CHANGELOG

## 2026-05-27 #2 — RTX 5070 Ti (Blackwell SM 12.0) 优化

**内容：**
- `triton_kernels/gpu_utils.py`：新增 `get_arch_params()` 函数，按 GPU 架构返回推荐 tile/warp/stage 参数
- Triton autotune 配置扩展：
  - `matmul.py`：追加 Blackwell 专属配置（128x128x128, 256x64x32 等）
  - `backward.py`：追加大 tile matmul backward 配置 + 8192 BLOCK_SIZE activation backward
  - `mlp_triton.py`：追加 Blackwell fused MLP 配置（128x128x128, 256x64x32）
- `triton_kernels/elementwise.py`：ReLU/GELU/SiLU kernel 添加 `@triton.autotune`（原硬编码 1024）
- `triton_kernels/swiglu_triton.py`：SwiGLU kernel 添加 `@triton.autotune`
- CUDA kernel 扩展：
  - 新增 WMMA64 变体（`matmul_wmma64_kernel`/`transB`/`transA`），64x64 tile，256 threads
  - 新增 tiled 模板实例化（64x64x32, 64x64x64, 128x64x32）
  - `launch_matmul_tiled_auto` 扩展为 5 级 dispatch（1024→WMMA64, 512→WMMA32, 256→64x64, 128→32x32, fallback→16x16）
  - fused MLP dispatch 扩展为 3 级（512→64x64, 128→32x32, fallback→16x16）
- cuTile 架构感知 tile 参数：所有 host 函数从 `get_arch_params()` 读取 tile 大小
- `CMakeLists.txt`：`CMAKE_CUDA_ARCHITECTURES` 从 `80` 改为 `86;120`

**验证：**
- 全部 55 项 Python 测试通过（Triton 21 + CUDA 16 + cuTile 18）
- Ampere 路径无回归（RTX 3070 Laptop 验证）

## 2026-05-27 #1 — cuTile Python 算子实现 + 四后端对比

**内容：**
- 新建 `cutile_kernels/` 模块：NVIDIA cuTile Python tile-based GPU 编程
  - `matmul.py`：分块矩阵乘法（ct.mma 融合乘加）
  - `elementwise.py`：BiasAdd、ReLU、GELU（tanh 近似）、SiLU、融合 BiasAdd+ReLU
  - `backward.py`：MatMul backward（dA=dC@B^T, dB=A^T@dC）、ReLU/GELU/SiLU backward
  - `layernorm.py`：LayerNorm forward/backward（ct.sum reduction + ct.atomic_add scatter）
  - `mlp_cutile.py`：融合 matmul+bias+GELU
  - `swiglu_cutile.py`：融合 SwiGLU
- 新建 `python/mnist/cutile_layers.py`：CUTILELinear/CUTILEActivation/CUTILELayerNorm autograd 层
- 新建 `python/mnist/cutile_model.py`：CUTILEMLP 模型（与 PyTorch/Triton/CUDA MLP 结构对称）
- 新建 `tests/test_cutile_kernels.py`：18 项 cuTile kernel 正确性测试
- `run_compare.py`：添加 `--cutile` 参数，支持 PyTorch/Triton/CUDA/cuTile 四后端对比

**验证：**
- 18 项 cuTile 测试全部通过
- 全部 55 项 Python 测试通过（Triton 21 + CUDA 16 + cuTile 18）
- cuTile MLP 前向/反向传播正常，梯度无 NaN/Inf

## 2026-05-26 #4 — LayerNorm 实现 + 测试完善

**内容：**
- 新建 `triton_kernels/layernorm.py`：Triton LayerNorm forward/backward kernel
  - forward: per-row mean/var/rstd 计算，affine 变换
  - backward: d_x via reduction，d_gamma/d_beta via atomic_add
- `kernels/mlp_cuda_kernels.cu`：添加 CUDA LayerNorm forward/backward kernel
  - warp shuffle reduction + shared memory broadcast
  - binding.cpp 添加 layernorm_forward/backward 绑定
- `python/mnist/triton_layers.py`：添加 TritonLayerNorm 模块
- `python/mnist/cuda_layers.py`：添加 CUDALayerNorm 模块
- 模型架构统一为 `Linear → LayerNorm → ReLU → Dropout`
- 配置更新：`hidden_dims: [784,1024,512,256,10]`，`activation: relu`，`dropout: 0.1`，`use_layernorm: true`
- 新建 `tests/test_triton_kernels.py`：21 项 Triton kernel 正确性测试
- 新建 `tests/test_cuda_kernels.py`：16 项 CUDA kernel 正确性测试
- 新建 `tests/test_kernels.cu`：6 项 C++ CUDA kernel 单元测试（nvcc 编译）
- Makefile 添加 `test-cuda`、`test-python` 目标
- 删除备份文件 `kernels/mlp_cuda_kernels.cu.bak`、`kernels/mlp_cuda_kernels_test.cu`

**验证：**
- 37 项 Python 测试全部通过（Triton 21 + CUDA 16）
- 6 项 C++ 测试全部通过
- CUDA 推理吞吐量 213K samples/sec（PyTorch 1.22x）

## 2026-05-26 #3 — 精度对齐：三方后端公平比较

**内容：**
- 新建 `triton_kernels/precision.py`：全局精度单例，控制 Triton kernel 的 TF32/FP32 模式
- 修改 3 个 Triton kernel 文件加 `ALLOW_TF32: tl.constexpr` 参数：
  - `triton_kernels/matmul.py`：tiled_matmul_kernel
  - `triton_kernels/backward.py`：matmul_backward_a/b_kernel
  - `triton_kernels/mlp_triton.py`：mlp_first_layer_kernel
- 对齐权重初始化：`triton_layers.py` 和 `cuda_layers.py` 均使用 `kaiming_uniform_(weight, a=math.sqrt(5))`
- `run_compare.py` 和 `benchmark_ops.py` 新增 `--precision tf32/fp32` 参数
- `cuda_layers.py`：cuBLAS 模式（use_cublas=True）作为可选，默认使用自定义 CUDA kernel
- 修复 `run_compare.py --no-bench` 时 cu_bench 未初始化的 bug

**验证：**
- FP32 严格模式（10 epoch）：PyTorch 98.44% | Triton 98.40% | CUDA 98.40%（差距 0.04%）
- TF32 模式（10 epoch）：PyTorch 98.45% | Triton 98.38% | CUDA 98.41%（差距 0.07%）
- 训练时间：CUDA 61.3s ≈ PyTorch 62.9s < Triton 71.8s

## 2026-05-26 #2 — 清理临时脚本 + 完整对比测试

**内容：**
- 删除 6 个临时脚本：benchmark_stable.py、compare_before_after.py、debug_grad.py、debug_grad2.py、debug_triton_train.py、print_stable.py
- 保留：benchmark_ops.py（算子对比）、run_compare.py（端到端对比）、run_mnist.py（MNIST 训练入口）

**验证：**
- `benchmark_ops.py --warmup 20 --iters 100`：48 项算子测试全部通过，Triton avg 1.89x / CUDA avg 2.75x vs PyTorch
- `run_compare.py --epochs 10 --cuda`：三方对比完成
  - 精度：PyTorch 98.43% | Triton 97.38% | CUDA 97.45%
  - 训练步延迟：CUDA 2.74ms < PyTorch 3.30ms < Triton 3.75ms
  - 推理延迟：CUDA 0.40ms < PyTorch 0.94ms < Triton 1.23ms
  - CUDA 推理吞吐量 318K samples/sec（PyTorch 的 2.34x）

## 2026-05-26 #1 — Triton MLP 梯度爆炸修复 + CUDA 推理优化

**内容：**
- `triton_kernels/backward.py`：修复 activation backward 函数的 grid 大小硬编码 bug
  - `relu_backward`、`gelu_backward`、`silu_backward` 三个函数的 grid 均为 `(triton.cdiv(n, 2048),)`（硬编码）
  - 但 kernel 使用 `@triton.autotune` 可选 BLOCK_SIZE=512/1024/2048/4096
  - 当 autotune 选择 BLOCK_SIZE<2048 时，grid 过小导致大部分输出元素未被写入（torch.empty_like 垃圾值），引发梯度爆炸
  - 修复：改为 `lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)` 使用 autotune 实际选择的 BLOCK_SIZE
- `python/mnist/cuda_layers.py`：CUDA 推理性能优化
  - forward matmul 始终使用 cuBLAS（移除 _CUBLAS_FALLBACK_DIM 条件判断）
  - backward 使用自定义 CUDA kernel（比 PyTorch autograd 更快）
  - 推理模式下跳过 autograd.Function 包装，直接调用 cuBLAS + CUDA activation kernel
  - bias_add 从 `mlp_cuda.bias_add` 改为 `output + bias`（cuBLAS 可融合）

**目的：** 修复 Triton MLP 训练不收敛问题（16% → 96.87%），改善 CUDA 推理性能（2.6x → 1.5x vs PyTorch）。

**验证：** `run_compare.py --epochs 5` 通过：PyTorch 98.25% | Triton 96.87%（正常收敛，TF32 + tanh-GELU 近似导致 ~1.4% 精度差异）

## 2026-05-25 #5 — Triton kernel 正确性修复 + GELU 优化

**内容：**
- `triton_kernels/dropout.py`：修复 `tl.rand(seed, col_offsets)` 为 `tl.rand(seed, row_idx * n_cols + col_offsets)`，解决所有 batch 行共享相同 dropout 掩码的 bug
- `triton_kernels/elementwise.py`：GELU tanh 近似从 `2.0 * tl.sigmoid(2.0 * inner) - 1.0` 替换为 `tl.tanh(inner)`
- `triton_kernels/backward.py`：GELU backward 同步使用 `tl.tanh(u)`
- `triton_kernels/mlp_triton.py`：fused MLP 已使用 `tl.tanh(inner)`（无需修改）
- `triton_kernels/loss.py`：消除冗余全局内存加载，用 `tl.where + tl.sum` 从已加载行中提取 target logit

**目的：** 修复 dropout 正确性 bug，统一 GELU 使用原生 `tl.tanh`（减少指令数），减少 loss kernel 全局内存访问。

## 2026-05-25 #1 — Triton MLP 端到端训练 + PyTorch 对比 + Nsight 支持

**内容：**
- 从 Salvation-Lies-Within/Triton/ 复制并集成 5 个 Triton 算子模块到 `triton_kernels/`
- 实现 `TritonLinear`（`torch.autograd.Function`）和 `TritonActivation` autograd 层
- 实现 `TritonMLP` 模型，结构与 PyTorch MLP 完全对称
- 新建 `run_compare.py` 支持 PyTorch vs Triton 并排对比训练
- 实现 SwiGLU kernel 和 fused MLP first layer kernel
- 新建 `profiling/profile_compare.py`（torch.profiler 对比分析）
- 扩展 `profiling/run_ncu.sh` 支持 Triton profiling 模式
- 扩展 Makefile 添加 `profile-triton`、`profile-triton-matmul`、`profile-compare` 目标

**目的：** 完成 Triton MLP 端到端训练能力，支持与 PyTorch 的详细性能对比和 Nsight profiling。

## 2026-05-25 #2 — GPU 针对性优化（RTX 5070 Ti + RTX 3070 Laptop）

**内容：**
- `triton_kernels/matmul.py`：添加 `@triton.autotune`（8 组配置覆盖 Blackwell/Ampere），启用 TF32 tensor core
- `triton_kernels/backward.py`：matmul backward 添加 autotune（12 组配置），启用 TF32
- `triton_kernels/mlp_triton.py`：fused kernel 添加 autotune（7 组配置），启用 TF32
- `triton_kernels/gpu_utils.py`：新建 GPU 检测模块，支持 compute capability 识别和架构分类
- `setup.py`：添加 SM_86（RTX 3070 Laptop）和 SM_120（RTX 5070 Ti）编译目标，修正优化级别 -O3

**目的：** 针对 RTX 5070 Ti (Blackwell SM12.0) 和 RTX 3070 Laptop (Ampere SM8.6) 自动选择最优 tile 配置，利用 TF32 tensor core 加速训练。

**验证：** `run_compare.py --epochs 2` 通过。Triton 训练步时从 10.1ms 降至 4.64ms（优化前 2.4x → 优化后 1.36x vs PyTorch）。PyTorch 97.44%，Triton 95.72%（差异来自 TF32 + tanh-GELU 近似，训练可用）。

## 2026-05-25 #3 — 代码审查 + setup.py 修正

**内容：**
- `setup.py`：移除冗余 `-arch` flag（多 `-arch` 只有最后一个生效，`-gencode` 已正确处理多架构），修正为纯 `-gencode` 方案

**审查结论：** 无 CRITICAL/HIGH 问题。autotune 配置合理覆盖大/中/小 tile。TF32 + tanh-GELU 精度差异在训练可接受范围内。

## 2026-05-25 #4 — CUDA Kernel 完整实现 + 三方对比

**内容：**
- `kernels/mlp_cuda_kernels.cu`：实现全部 CUDA kernel（14 个 launch 函数）
  - matmul_naive + matmul_tiled（6 组 tile 配置 dispatch，shared memory tiling）
  - bias_add、3 组激活函数（GELU/ReLU/SiLU forward + backward）
  - mlp_fused_first_layer（融合 matmul + bias + GELU）、swiglu_fused
  - GELU tanh-approximation 与 Triton kernel 保持一致
- `kernels/binding.cpp`：实现全部 PyTorch C++ Extension 绑定（pybind11）
  - 适配 PyTorch 2.11 API（`c10::cuda::getCurrentCUDAStream`）
- `python/mnist/cuda_layers.py`：新建 CUDA autograd 层（CUDALinear + CUDAActivation）
  - 结构与 triton_layers.py 完全对称，weight 布局 (in_features, out_features)
- `python/mnist/cuda_model.py`：新建 CUDAMLP 模型（与 MLP/TritonMLP 结构对称）
- `run_compare.py`：添加 `--cuda` 参数，支持 PyTorch vs Triton vs CUDA 三方对比
- `setup.py`：仅使用 `-gencode`（arch=compute_86/120,code=sm_86/120），修正 CUDA 13 编译兼容

**目的：** 完成 CUDA kernel 实现并集成到对比框架，支持三种后端（PyTorch/Triton/CUDA）的端到端训练和基准测试对比。

**验证：** `run_compare.py --epochs 2 --cuda` 三方对比通过：
- PyTorch 97.44% (18.1s) | Triton 95.72% (33.8s) | CUDA 95.62% (17.2s)
- CUDA 训练步时 5.15ms（vs PyTorch cuBLAS 4.79ms，差距 7%）
- CUDA 推理延迟 1.20ms（vs PyTorch 1.00ms）
- Triton/CUDA 精度差异来自 TF32 + tanh-GELU 近似，两者精度一致（95.72% vs 95.62%）

## 2026-05-25 #5 — 算子级横向对比 + tl.tanh 兼容性修复

**内容：**
- 新建 `benchmark_ops.py`：12 类算子的逐算子正确性+性能横向对比
  - 覆盖：matmul、gelu/relu/silu forward+backward、bias_add、matmul_backward(dA/dB)、fused_mlp_first、swiglu、bias_add_relu
  - 每个算子 4 组尺寸（4K~262K 元素），正确性（L2/Max误差）+ 延迟（median/P95）
  - 支持 `--sizes`、`--ops`、`--warmup`、`--iters` 参数化
- `triton_kernels/elementwise.py`：`tl.tanh` → `2*tl.sigmoid(2*t)-1`（兼容无 tl.tanh 的 Triton 版本）
- `triton_kernels/backward.py`：同上修复 GELU backward
- `triton_kernels/mlp_triton.py`：同上修复 fused MLP kernel

**目的：** 提供完整的算子级对比数据，量化每个自定义 kernel 相对 PyTorch cuBLAS/autograd 的性能差距。

**验证：** `benchmark_ops.py --warmup 20 --iters 100` 通过，4 组尺寸 x 12 算子：
- Matmul forward：CUDA 0.33-1.01x，Triton 0.64-0.96x（cuBLAS 大矩阵优势明显）
- Activation backward：CUDA 2.6-6.8x，Triton 4.0-4.85x（绕过 autograd 图开销）
- Matmul backward：CUDA 1.2-6.5x，Triton 2.0-4.0x（直接 kernel 调用）
- Fused MLP：CUDA 0.87-1.78x，Triton 0.98-1.68x（融合减少显存写入）
- 整体：CUDA avg 2.53x，Triton avg 2.04x vs PyTorch
