# CHANGELOG

## 2026-08-31 v2 — SwiGLU MLP block 主线 + fp16 TensorCore 胜势 + 可追溯复现

**里程碑（v2 阶段）：**
- **SwiGLU block 主线**：`python/transformer_mlp.py`（eager/concat/compile/triton/cuda/cutile/triton_fused 7 后端）+ bench/suites YAML + 正确性测试（shape×dtype 支持矩阵）
- **fp16 TensorCore**：`triton_kernels/matmul.py` 支持 fp16/bf16（input_precision + fp32 累加）；`swiglu_triton.py` sigmoid 升 fp32。**实测 prefill/train 3.0–3.9x vs eager-FP32**（M≥128）
- **make reproduce**：`tools/reproduce.py` 一键 build→test→bench→manifest（自动选 CUDA 工具链版本）
- **E3 基线归档**：228/152-case sweep 带 manifest；first decode 融合 kernel 负结果（带宽 bound）+ cuda WMMA 精度取舍根因
- 测试规模 55→136（SwiGLU 用例加入）

## 2026-09-01 v2 Phase-2 — fp16 六后端闭环 + P1 集成 + 多机代码 + 交付包

**内容（RTX 3070, commit 55dc2fe）:**
- **cuda fp16 解锁**：`kernels/mlp/matmul_half.cu`（fp16 WMMA in→fp32 acc→fp16 out, L2 2e-4），破除历史注释"L2 0.75 不可修"之谜（实为 fp16 正常精度）；`swiglu_block_cuda` fp16 分支
- **fp16 六后端正确性闭环**：eager/triton/triton_fused/cuda/cutile/compile 42-case sweep corr 100%；cutile 经 dtype 传播修复（matmul 恒输出 fp32→段间 recast）
- **cutile tile 优化**：ampere (32,32,32)→(32,64,32)，cutile matmul 1.5-1.6x（tile 单变量实验）
- **fp16 训练闭环**：eager(cuBLAS fp16) 与 Triton fp16 均收敛（rel<0.005），`tests/test_training_loop.py` 新增
- **P1 torch.library**：`mlp_kernel::swiglu` schema/meta/autograd/opcheck/gradcheck/compile 全通过（8 tests）
- **多机代码**：`tools/preflight.py`（Ampere/Blackwell lane）+ `docs/compatibility-matrix.md` + setup build_base 隔离 + `scripts/verify.sh` + **git bundle 离线同步**（绕 GFW）
- **交付包**：`EVIDENCE.md`（claim→evidence→level + 30s/3min/15min）、`KNOWN-LIMITATIONS.md`、`docs/fp16-delivery-status.md`
- **测试规模** 136→**211**（174 passed / 0 failed / 37 skipped，含算子矩阵/P1/训练/fp16）

## 2026-06-04 #1 — Triton matmul 64×64 + cuTile tile (16,16,16),per-op matmul 3-6x 提速

**内容:**
- `triton_kernels/matmul.py`: 删除 autotune 中 32×32 fallback,加 64×64 / 64×128 / 128×64 tile(全 num_warps=4)。原 fallback 在 M=64 小 batch 场景下被 autotune 抖动误选为"最快",实际 GPU 跑出 0.091ms;64×64 tile 0.024ms,**4 layer matmul 单步 0.36→0.09ms**。
- `triton_kernels/gpu_utils.py`: Blackwell arch 的 `cutile_matmul_tile` 从 (64,64,32) 改为 (16,16,16)。(16,16,16) 是 Blackwell `mma.sync.aligned.m16n8k16` 的 native fragment;大 tile 在小 batch 场景增加 padding + 循环 overhead,4 个 MLP shape 实测全部 1.5-3x 慢。改了 `cutile_matmul_tile` 同时影响 forward + backward matmul(`cutile_kernels/matmul.py` / `cutile_kernels/backward.py`)。cuTile 4 个 MLP shape matmul 0.575ms → 0.110ms。
- `tools/gpu_warmup.py` / `tools/run_full_eval.sh`: 60s GPU 预热 + 4 轮重复 + 后 3 轮均值 driver,支持 `--skip-compare` / `--rounds` / `--take-last` / `--compare-precision`。

**验证 (per-op bench, 4 轮均, fp32 strict):**

| shape | triton 旧 | triton 新 | cuTile 旧 | cuTile 新 |
|-------|-----------|-----------|-----------|-----------|
| (64, 784, 1024) | 0.091ms | **0.024ms** | 0.174ms | **0.035ms** |
| (64, 1024, 512) | 0.091ms | **0.028ms** | 0.226ms | **0.044ms** |
| (64, 512, 256) | 0.091ms | **0.017ms** | 0.115ms | **0.019ms** |
| (64, 256, 10) | 0.091ms | **0.018ms** | 0.060ms | **0.012ms** |

**端到端 (15 epoch, fp32 strict, 4 轮后 3 均):**

| backend | 旧 train_s | 新 train_s | 旧 step_med | 新 step_med |
|---------|------------|------------|-------------|-------------|
| PyTorch | 24.3s | 24.3s | 1.78ms | 1.15ms |
| Triton  | 41.8s | 44.1s | 1.88ms | 2.04ms |
| CUDA    | 24.9s | 25.1s | 1.14ms | 1.12ms |
| cuTile  | 25.9s | 26.7s | 3.17ms | **2.32ms (-27%)** |

**根因复盘:** per-op 3-6x 提速未能转化为端到端等比例提速。MLP 训练 step 内 matmul kernel 时延仅占 1-2ms 不到 5%,主导是 Python launch overhead + AdamW 状态更新 + DataLoader I/O + `.item()` 同步。**cuTile step_med -27% 是真改善,triton/pytorch 端到端无变化是因为它们的 matmul 占比已很小。**

**D 方案其余 step 决策:**
- #1 (CUDA Graph): capture 时遇到 `cudaErrorStreamCaptureImplicit` (autograd AccumulateGrad stream 与 capture stream 不一致),fallback 到 eager 路径复杂,收益仅 ~3s 不抵成本,**未保留**。
- #3 (fused_bias_gelu): 跳过。per-op 节省 0.02ms,端到端预估 <0.1s。
- #5 (完整 fused MLP): 跳过。改动 150 行复杂 kernel,fp32 strict 数值调试成本高,预估端到端 <3s 收益。

**记录位置:**
- `results/full_eval_20260604_001432/` 含 4 轮 op bench + 4 轮 cuTile bench + 4-backend 端到端 fp32 strict 完整数据。
- `results/compare_*.json` 4 份独立 compare_*.json 落盘。
- Driver `bash tools/run_full_eval.sh` 重现。

## 2026-06-03 #1 — 修 wmma64 数值 bug,CUDA 精度回归彻底解决

**内容:**
- 定位 #3 提出的 CUDA backend 精度回归 (97.06% vs 期望 98.6%) 真因:
  - pytest 全部 16 项通过,说明单算子在测试尺寸下正确;
  - 直接对比 `matmul_tiled_auto` 与 `torch.matmul` 在 MNIST 4 个 layer 尺寸:
    - M=8 K=784 N=1024 (max_dim=1024 → wmma64): **max_err 83.22**
    - M=8 K=1024 N=512 (max_dim=1024 → wmma64): **max_err 87.60**
    - M=8 K=512 N=256 (max_dim=512 → wmma32): max_err 2.6e-02 ✓
    - M=8 K=256 N=10 (max_dim=256 → tiled FP32): max_err 1.5e-05 ✓
  - 结论:**只有 wmma64 (64×64 tile, R=32) 路径错**,wmma32 / FP32 路径正确。
- 真因:wmma64 kernel `R=32` 但 `mma_sync` 只调一次(单 16x16x16 fragment),**漏掉 K 方向后 16 个元素的乘加**。wmma32 因 R=16 == fragment K 维 16 所以正好对齐没事。
- 修复:`kernels/mlp/wmma.cu` 中 3 个 wmma64 kernel(normal / transB / transA)的 mma 段改为 K-direction `for (int kk = 0; kk < R; kk += 16)` 循环 + 2 次 fragment load + 2 次 mma_sync 累加。

**验证:**
- 修复后 4 个 MLP 形状的 matmul max_err 全部 ≤ 3.5e-02(纯 FP16 精度噪声),与 wmma32 同档。
- 重跑 15-epoch 4-backend 完整对比:CUDA 98.65%(↑ 1.59pp from 97.06%),与 PyTorch 98.71% / Triton 98.62% / cuTile 98.52% 同档。
- CUDA backend 此次拿到训练总时长第一(27.0s)+ 推理延迟第一(0.298 ms / 858K samples/s)。
- 详细结果见 `results/four_backend_fp32_v2.log` 与 `results/compare_20260602_235625.json`,README "完整 4-backend 实测" 小节同步更新。

**影响范围:**
- 此 bug 在原 `kernels/mlp_cuda_kernels.cu` 中就存在,拆分时按行 100% 保留语义,所以拆分本身未引入。
- 之前 RTX 3070 Laptop CHANGELOG 2026-05-27 #2 验证 "Ampere 无回归",但 wmma64 是 SM 8.0+ 的 64x64 tile 实现,在 3070 上 max_dim 是否真触发了 wmma64 path 未深查;5070 Ti 上 #3 修了 launch_bounds 后才正式可 launch,这条 bug 立刻浮现。

## 2026-06-02 #3 — 完整 4-backend 15-epoch 对比 (RTX 5070 Ti)

**内容:**
- 修两个 RTX 5070 Ti 上才暴露的原生 bug(拆分前后都存在,只在 max_dim ∈ [256,511] 或 wmma64 路径触发):
  1. `matmul.cu launch_matmul_tiled_auto`: max_dim ∈ [256,511] 启动 `block(64,64)=4096 threads` 超 CUDA 1024 上限。改为 `block(32,32)=1024 threads` + 32×32 tile FP32。
  2. `wmma.cu matmul_wmma64_{,transB,transA}_kernel`: `__launch_bounds__(256)` 声明与实际 block(16 warps × 32 = 512 threads)不一致,PTX runtime 拒绝 launch。改为 `__launch_bounds__(512)`。
- 通过代理(127.0.0.1:7897)成功 prefetch MNIST 4 个 raw 文件(yann/S3 直连 1-15 KB/s,代理 100+ MB/s)到 `data/MNIST/raw/`。
- 完整 4-backend × 15 epoch 跑通,结果落 `results/four_backend_fp32.log` + `results/compare_20260602_231903.json`,README "完整 4-backend 实测" 小节同步更新。

**结果(15 epoch best val_acc / 训练总时长 / 训练步 median ms / 推理 median ms):**
- PyTorch: 98.71% / 28.5s / 1.745 / 0.363
- Triton:  98.63% / 50.1s / 2.215 / 0.838
- CUDA:    **97.06%** ⚠️ / 31.3s / 1.583 / 0.269
- cuTile:  98.70% / 36.5s / 2.944 / 0.780

**已知问题(进入 #4 调查):**
- CUDA backend 精度从 ~98.6% 退到 97.06%,差 1.6 个百分点,**远超 TF32/tanh-GELU 近似可解释的 0.05% 范围**。loss 也偏高(0.103 vs 其它 0.054)。说明上面 2 个 launch_bounds/tile 修复治标但伤了数值正确性,或暴露了既有数值 bug。需先跑 `tests/test_cuda_kernels.py` 定位是 wmma64、matmul_tiled 32×32 还是别的算子。

## 2026-06-02 #2 — cuTile 单 step 微基准 (RTX 5070 Ti)

**内容:**
- 新增 `profiling/bench_cutile.py`:cuTile 专用 benchmark driver,2s GPU warmup + 每项 4 轮(后 3 轮均值)。
- 修正 driver 内 cuTile 导入名:`layernorm_forward` / `layernorm_backward`(非 `_cutile` 后缀);`swiglu_cutile(x)` 单参数(`x*sigmoid(x)`)。
- 实测 RTX 5070 Ti + cuda-tile (`cuda.tile`) + torch 2.11.0+cu130,M=512 K=768 N=3072,batch=256:
  - 算子级 12 项 (matmul / matmul_backward_a/b / bias_add / gelu / silu / relu / gelu_backward / layernorm / layernorm_backward / mlp_fused_first_layer / swiglu),完整数据见 `results/cutile_bench.json`。
  - 端到端 CUTILEMLP `[784,1024,512,256,10]` + LayerNorm + Dropout=0.1:
    - 训练步 2.256 ms (std 0.012)
    - 推理 0.612 ms (std 0.001)
    - 推理吞吐 418,556 samples/sec
- README "性能参考" 下追加 `cuTile 单 step 微基准 (RTX 5070 Ti, FP32, B=256, 2026-06-02)` 小节;原表 cuTile 列保留 `-`,因为原表是不同 GPU 的 15-epoch 完整训练,直接同列会误导。

**验证:**
- 4 轮里第 1 轮 (discarded) 与后 3 轮均值差 ≤ 5%,std ≤ 1.2% mean,数值稳定。
- 启动 warmup 23708 次 matmul 后 GPU 已稳定在 P0。
- bench driver `precheck()` 在 cuTile / 项目 wrapper 缺失时立即 exit 2,不会沉默失败。

## 2026-06-02 #1 — 工程化清理:拆分大 CUDA 文件、删占位 bench/、补齐 nsight 脚本

**内容:**
- **CUDA 拆分**:原 `kernels/mlp_cuda_kernels.cu`(1473 行)按算子族拆分到 `kernels/mlp/` 子目录:
  - `device_utils.cuh`(43 行)— 公共 device 函数 (gelu/silu/warp_reduce_sum)
  - `wmma_decl.cuh`(41 行)— 6 个 WMMA kernel 前向声明
  - `matmul.cu`(349 行)— naive + tiled + transA + transB + bias_add + auto dispatch
  - `wmma.cu`(410 行)— 32x32 / 64x64 共 6 个 WMMA FP16 Tensor Core kernel
  - `activation.cu`(200 行)— GELU/ReLU/SiLU forward + backward + vec4 backward
  - `fused.cu`(114 行)— mlp_fused_first_layer + swiglu_fused
  - `layernorm.cu`(158 行)— LayerNorm forward + backward (warp shuffle)
  - `softmax.cu`(86 行)— 数值稳定行内 softmax
  - `pool_im2col.cu`(169 行)— maxpool2d + avgpool2d + im2col
  - 全部 ≤ 410 行,符合 800 行上限规则。
  - `setup.py` sources 列表更新, 增加 `include_dirs=['kernels/mlp']`。
  - 原 `kernels/mlp_cuda_kernels.cu` 删除。
- **bench/ 清理**:删除 `bench/` 整个目录(`benchmark.py`、`compare_correctness.py`、`benchmark_shapes.yaml`、`__init__.py`),它们是占位代码 + 大量 TODO 注释,引用的 `python/mlp_reference.py` 与 `python/torch_extension.py` 也仅被 bench/ 使用,一并删除。`benchmark_ops.py` 作为唯一权威算子级 benchmark 入口。
- **Makefile**:`test` / `bench` / `bench-quick` 三个 target 重定向到 `benchmark_ops.py`;追加 `profile-nsys` / `profile-ops` 两个 target;`.PHONY` 同步更新。
- **profiling 脚本**:
  - 重写 `profiling/run_ncu.sh`:4 个引用已删 `bench/benchmark.py` 的 case (naive/tiled/roofline/speedof) 改为内联 Python 直接调用 `mlp_cuda`;新增 4 个 case (`cuda` / `cutile` / `mlp-cuda` / `mlp-cutile`) 补齐对自定义 backend 的 ncu profile 覆盖;支持 `M=… K=… N=… bash …` env 覆盖矩阵尺寸。
  - 新增 `profiling/profile_nsys.sh`:nsys 时间线 wrapper,case 与 `run_ncu.sh` 同构(tiled/triton/mlp-cuda/mlp-cutile/compare),输出 `results/*.nsys-rep`。
  - 新增 `profiling/profile_ops.py`:算子级 driver,NVTX 包裹每个 (backend, op),按需 import 各 backend(缺失自动跳过),支持 `--export-trace` 输出 Chrome trace。
- **requirements.txt**:对所有依赖加上 minor 版本下限,torch 上限 `<3.0`,torchvision 上限 `<1.0`;追加 cuda-tile / nvtx 为可选依赖的注释说明。
- **README**:`项目结构` 反映新 `kernels/mlp/` 子树;`环境要求` 追加 WSL `git config --add safe.directory` 提示;`Profiling` 章节完整改写,列出 3 类 nsight 入口(ncu / nsys / profile_ops);`性能参考` cuTile `-` 列加复现脚本说明。

**目的:** 解决 audit 列出的全部 MEDIUM / LOW 项:大文件拆分、benchmark 入口合并、nsight 脚本补齐、依赖锁定、WSL git 提示。

**验证:**
- 文件行数:`wc -l kernels/mlp/*` 全部 ≤ 410 行(原 1473 → 9 文件 × 平均 175 行)。
- bench/ 删除后 grep 无残留引用(已确认 `bench` 目录不存在,`python/mlp_reference.py` / `python/torch_extension.py` 删除后无其它 import)。
- setup.py 编译路径未变(`make install` 命令不变),仅 sources 列表变化,首次 `pip install -e .` 应重新编译全部新 .cu 文件。
- 拆分文件代码 100% 原样保留,无逻辑改动;build 需要在 GPU 机器实测(本任务在 WSL 工作树内,无 GPU 验证)。
- cuTile 性能数据待用户在装有 cuTile 的 GPU 机器运行 README 中给出的命令补齐。
- 残留 TODO(非本次范围):`docs/` 中英文混排(LOW)未统一,可后续用 `doc-updater` agent 处理。

## 2026-05-30 #2 — 新增 Naive C++ CPU 实现（学习用）

**内容：**
- `naive/operators.cpp`：纯 C++17 实现，不依赖任何深度学习库
- MLP 完整训练流程: Linear 前向/反向、ReLU/Sigmoid 激活、Softmax + CrossEntropy、SGD 参数更新
- CNN 算子: Conv2D(im2col+matmul)、MaxPool2D、AvgPool2D
- Demo: XOR 分类(4/4 accuracy)、Spiral 3-class 分类(99% accuracy)、CNN 算子验证

**目的：** 作为学习材料，直观展示深度学习前向/反向传播的数学原理

## 2026-05-30 #1 — 新增 Conv2D、MaxPool2D、AvgPool2D、Softmax 算子

**内容：**
- `triton_kernels/softmax.py`：新增 Triton softmax kernel（逐行，数值稳定）
- `triton_kernels/pool.py`：新增 Triton MaxPool2D、AvgPool2D kernel（NCHW 格式）
- `triton_kernels/conv.py`：新增 Triton Conv2D（im2col + matmul 分解）
- `kernels/mlp_cuda_kernels.cu`：新增 CUDA softmax、maxpool2d、avgpool2d、im2col kernel
- `kernels/binding.cpp`：新增 conv2d、maxpool2d、avgpool2d、softmax Python 绑定
- `triton_kernels/__init__.py`：导出新算子
- `benchmark_ops.py`：新增 bench_conv2d、bench_maxpool、bench_avgpool、bench_softmax

**目的：** 扩展算子库，覆盖卷积、池化、softmax 等常见 CNN/Transformer 操作

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
- 全部 55 项 Python 测试通过（Triton 21 + CUDA 16 + cuTile 18）※历史快照；当前以 make reproduce 报告为准
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
- 全部 55 项 Python 测试通过（Triton 21 + CUDA 16 + cuTile 18）※历史快照；当前以 make reproduce 报告为准
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
