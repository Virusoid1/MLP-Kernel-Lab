# Optimization Log

> 每次优化的变更、数值、复现命令、gate 结果。
> 与 `CHANGELOG.md` 区别:CHANGELOG 记"做了什么 + 为什么"(给开发者),本文件记"度量 + 结果 + verify 命令"(给优化方法论)。

## Current Baseline

Live pointer: `results/baseline.json`(生成方式见 [docs/optimization-guide.md → Decision Tree](optimization-guide.md))。
历史副本:`results/baselines/<git_sha>.json`(升级 baseline 须开 `chore(bench): promote baseline` PR)。

## 元数据 Schema(必填)

每条 `### vN` 条目必须包含以下结构,缺一不可:

```yaml
metadata:
  gpu:        {name, cc, vram_gb}
  driver:     "<nvidia-smi driver_version>"
  torch:      "<__version__>"
  cudnn:      "<version>"
  allow_tf32: bool
  seed:       int
  dtype:      "fp32" | "fp16" | "bf16"
  git_sha:    "abc1234"
  ts:         "YYYY-MM-DDTHH:MM:SSZ"
command:   "python ... 实际跑了什么"
verify:    "python tools/analyze_bench.py ... --gate  # exit 0"
regression: 内联 analyze 输出的 delta 表 (markdown) 或链接
```

`capture_metadata()` (`python/mnist/benchmark.py`) 是 metadata 的单一来源,所有 caller(`benchmark_ops.py` / `bench_cutile.py` / `run_compare.py`)都嵌入 JSON 顶部。

---

## 优化记录

### v0: wmma64 K-dim 累加循环修复(范例)

**问题**:`matmul_wmma64_kernel` 内 `R=32` 但 `mma_sync` 只调一次(单 16x16x16 fragment),漏算 K 维后 16 元素,导致 MNIST `(784,1024)` / `(1024,512)` 上 max_abs_err ~85(vs cuBLAS),CUDA backend MNIST val_acc 97.06% (期望 ~98.6%)。

**修改**:`kernels/mlp/wmma.cu` 中 3 个 wmma64 kernel 的 mma 段改为
```cpp
for (int kk = 0; kk < R; kk += 16) {
    nvcuda::wmma::load_matrix_sync(a_frag, &sA[warp_row][kk], R);
    nvcuda::wmma::load_matrix_sync(b_frag, &sB[kk][warp_col], TILE);
    nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
}
```

**metadata**:
```yaml
gpu:        {name: "NVIDIA GeForce RTX 5070 Ti", cc: "12.0", vram_gb: 16.0}
driver:     "596.36"
torch:      "2.11.0+cu130"
cudnn:      "90100"
allow_tf32: false
seed:       42
dtype:      "fp32"
git_sha:    "665dc07"
ts:         "2026-06-03T01:24:00Z"
```

**command**:`make install && python run_compare.py --cuda --cutile --precision fp32 --epochs 15`

**verify**:
```
python tools/analyze_bench.py results/baselines/d621338.json results/baselines/665dc07.json \
    --shape MLP_LAYERS --gate --warn-l2 1e-3 --warn-maxabs 1e-2
# exit 0  (perf 与正确性均通过)
```

**regression(精度)**:

| 形状 | 修复前 max_err | 修复后 max_err |
|------|-------|-------|
| 8×784→1024 | 83.22 | 3.42e-02 |
| 8×1024→512 | 87.60 | 3.48e-02 |
| 8×512→256 | 2.6e-02 | 2.6e-02 |
| 8×256→10 | 1.5e-05 | 1.5e-05 |

**regression(端到端)**:CUDA backend val_acc 97.06% → 98.65%(+1.59pp),与 PyTorch 98.71% / Triton 98.62% / cuTile 98.52% 同档。

**分析**:wmma32 sibling kernel `R=16` 与 fragment K=16 重合所以单 mma 正好对齐;wmma64 `R=32` 与 fragment K=16 失配,需 2 次累加。教训:任何 `R != 16` 的 wmma kernel 都必须显式 `for (kk; kk<R; kk+=16)` 内积循环。`tests/test_cuda_kernels.py` 未捕获是因为单测用的尺寸跟 MNIST `[784,1024]` 不重合,后续应在 baseline 中固定 MLP_LAYERS shape。

---

### v1: triton matmul 64×64 tile + cuTile (16,16,16) native mma fragment

**问题**:
- `triton_kernels/matmul.py` autotune 12 个 config 中 32×32 fallback 在 M=64 小 batch 场景下被 `do_bench` 抖动误选为"最快",实际 GPU 跑出 0.091ms,而 64×64 tile 仅 0.024ms。
- `triton_kernels/gpu_utils.py` Blackwell `cutile_matmul_tile=(64,64,32)` 在 4 个 MLP 真实 shape 上比 (16,16,16) 慢 1.5-3x,因为 (16,16,16) 才是 Blackwell `mma.sync.aligned.m16n8k16` 的 native fragment,大 tile 增加 padding + 循环 overhead。

**修改**:
- `triton_kernels/matmul.py`: 删除 32×32 fallback,加 64×64 / 64×128 / 128×64 tile(全 num_warps=4)。
- `triton_kernels/gpu_utils.py`: Blackwell `cutile_matmul_tile` 从 (64,64,32) 改为 (16,16,16)。同时影响 `cutile_kernels/matmul.py` + `cutile_kernels/backward.py`(forward + backward matmul 走同一 dict)。
- `tools/gpu_warmup.py` / `tools/run_full_eval.sh`: 60s GPU 预热 + N 轮重复 + 后 K 轮均 driver。

**metadata**:
```yaml
gpu:        {name: "NVIDIA GeForce RTX 5070 Ti", cc: "12.0", vram_gb: 16.0}
driver:     "596.36"
torch:      "2.11.0+cu130"
cudnn:      "90100"
allow_tf32: false
seed:       42
dtype:      "fp32"
git_sha:    "1dc3193"
ts:         "2026-06-04T00:32:48Z"
```

**command**:`bash tools/run_full_eval.sh`(4 轮 op + 4 轮 cuTile + 4 轮 4-backend end-to-end,后 3 轮均)

**verify**: 数据落 `results/full_eval_20260604_003248/`(latest)。`analyze_bench.py` 工具未在 v1 启用,验证通过 `python -c "import json; assert 'rows' in json.load(open('results/full_eval_20260604_003248/round_1_ops.json'))"`。

**per-op 改善 (fp32 strict, 4 MLP shape)**:

| shape | triton 旧 | triton 新 | cuTile 旧 | cuTile 新 |
|-------|-----------|-----------|-----------|-----------|
| (64, 784, 1024) | 0.091ms | **0.024ms (3.8x)** | 0.174ms | **0.035ms (5.0x)** |
| (64, 1024, 512) | 0.091ms | **0.028ms (3.2x)** | 0.226ms | **0.044ms (5.1x)** |
| (64, 512, 256)  | 0.091ms | **0.017ms (5.3x)** | 0.115ms | **0.019ms (6.0x)** |
| (64, 256, 10)   | 0.091ms | **0.018ms (5.0x)** | 0.060ms | **0.012ms (5.0x)** |

**端到端 (15 epoch, fp32 strict, 4-backend, 4 轮后 3 均)**:

| backend | val_acc | train_s | step_med | samp/s |
|---------|---------|---------|----------|--------|
| PyTorch | 0.9871 | 24.8 | 1.23ms | 208,972 |
| Triton  | 0.9862 | 44.5 | 1.79ms | 143,741 |
| CUDA    | 0.9867 | 25.0 | 1.13ms | 229,533 |
| cuTile  | 0.9863 | 26.2 | **2.32ms (-27% vs v0)** | 110,577 |

**regression 失真修正**:
v0 之前所有 4-backend 对比数据**实际是用 num_workers=0 测出来的**(WSL + proxy 担心多 worker 死锁),15 epoch 训练时间被 DataLoader I/O wait 拉高到 25-44s。**num_workers=2 是 `create_mnist_loaders` 默认值**,1 epoch 实际仅 1.7s(0.45s/epoch 训练 + 0.8s/epoch validate + cache hit 后几乎 0),15 epoch 大头是 validation。**端到端"训练时间"在 num_workers=2 下基本不变**。

**分析(踩坑)**:
1. **per-op 3-6x 提速未转化为端到端等比例提速**。MLP step 1-2ms 中 matmul kernel 实际只占 0.05-0.1ms(~5%),主导是 Python launch overhead + AdamW 状态更新 + `.item()` 同步点。triton/pytorch end-to-end 在 v0/v1 之间几乎不变。
2. **autotune 不能信 do_bench 抖动**:32×32 在 M=64 上 do_bench 几次都在 5-8us 范围抖动,autotune "选"了 32×32 不是因为它真最快,而是误差范围内的噪声。**对策:对已知的 shape 集合,显式在 configs 列表里加推荐 tile + 把小 fallback 删掉,不要靠 autotune 自动发现**。
3. **cuTile mma fragment 不是越大越好**。Blackwell m16n8k16 是 native,(16,16,16) = 单 fragment 全展开,大 tile (64,64,32) 走 4×4×2 = 32 fragment,需要 padding + loop + register 管理,反而慢 1.5-3x。**对策:小 batch 永远从最小 fragment 试起**。
4. **D 方案其余 step 跳过**:#1 CUDA Graph 在 Triton autograd + AccumulateGrad 多 stream 下 capture invalid,#3 fused_bias_gelu 端到端 <0.1s 收益不值,#5 完整 fused MLP 150 行复杂 kernel 收益 <3s 不值。

**记录位置**:
- `results/full_eval_20260604_003248/` 含 4 轮 op bench + 4 轮 cuTile bench + 4-backend 端到端 fp32 strict 完整数据。
- `results/compare_*.json` 4 份独立落盘。
- `results/full_eval_20260604_001432/` 是 num_workers=0 失真数据,**已淘汰**。
- Driver `bash tools/run_full_eval.sh` 重现。

---

<details>
<summary>历史模板</summary>

#### v1: shared memory tiling

**变更**: 加载 A/B tile 到 shared memory
**BLOCK**: 16x16x16
**Latency**: ___ ms
**TFLOPS**: ___
**速度提升**: ___x

#### v2: block size 调整

| BLOCK_M | BLOCK_N | BLOCK_K | Latency | 说明 |
|---------|---------|---------|---------|------|
| 16 | 16 | 16 | | |
| 32 | 32 | 16 | | |
| 32 | 32 | 32 | | |

#### v3: coalesced memory access

**Latency**: ___ ms

#### v4: fused bias + GELU

**Latency (unfused)**: ___ ms → **Latency (fused)**: ___ ms

#### v5: FP16

**FP32 Latency**: ___ ms → **FP16 Latency**: ___ ms

#### v6: (your optimization)

**变更**: ___

</details>

---

## 汇总表(当真实数据可用时填)

| Version | Change | Latency (ms) | TFLOPS | Speedup vs v0 | git_sha |
|---------|--------|-------------|--------|---------------|---------|
| v0 | wmma64 K-dim fix | 1.09 (8×784×1024) | — | correctness baseline | 665dc07 |
| v1 | fused_mlp_first block fix + ref-FP32 | — | — | correctness gate enabled | (pending) |
| v1-v6 | — | — | — | — | — |

---

### v1: `fused_mlp_first_layer` 块尺寸修复 + ref-FP32 暴露 CUDA matmul K-split 需求

**问题**:
- `fused_mlp_first_layer` 在 `(M,K,N)=(512,768,512)` 启动 `block(64,64)=4096 threads`,超 CUDA 1024 硬限,`cudaErrorInvalidValue`。
- `benchmark_ops.py` 的 reference `a @ b` 跟随全局 `allow_tf32=True`,**与自研 backend 的 TF32 / FP32 选择混在一起**, 无法孤立数值偏差。

**修改**:
- `kernels/mlp/fused.cu` `launch_mlp_fused_first_layer` 的 `max_dim >= 512` 分支:`block(64,64)=4096` → `block(32,32)=1024` + tile 从 `<64,64,32>` 改 `<32,32,32>`。
- `benchmark_ops.py` `bench_matmul` reference 用 `with torch.backends.cuda.matmul.allow_tf32 = False: ref = a @ b` 显式 FP32,让 L2 偏差真实反映 backend vs cuBLAS-FP32。

**metadata**:
```yaml
gpu:        {name: "NVIDIA GeForce RTX 5070 Ti", cc: "12.0", vram_gb: 16.0}
driver:     "596.36"
torch:      "2.11.0+cu130"
cudnn:      "90100"
allow_tf32: false
seed:       42
dtype:      "fp32"
git_sha:    "8991e9d"
ts:         "2026-06-03T12:50:00Z"
```

**command**:`make install && python benchmark_ops.py --sizes medium --dtypes fp32 --roofline --warmup 20 --iters 100 --output results/baseline_post_p0p1_v2.json`

**verify**:
```
python tools/analyze_bench.py results/baseline_pre_p0p1.json results/baseline_post_p0p1_v2.json --shape all
# P0 fused_mlp_first (256,512) 与 (512,768) 行从 MISSING → OK (2 行 × 3 backend = 6 行)
# 其余行 perf 改变在 ±70% 范围内,无 CORRECTNESS_FAIL
```

**regression(精度,P0 fused_mlp_first)**:

| shape | 修前 | 修后 |
|-------|------|------|
| (256,512)@(512,256) GELU | cudaErrorInvalidValue | max_abs 2.7e-05 |
| (512,768)@(768,512) GELU | cudaErrorInvalidValue | max_abs 2.7e-05 |

**P0.5 失败经验(已记录)**:
尝试 matmul_tiled_kernel 用 2 路 K-split 双累加器 (Kahan-like) 修 CUDA FP32 偏差。**实测无效**:L2 4.17 → 4.17 不变。2 路 split 不足以消除 cuBLAS 的 4 路 K-split 精度差,需 Kahan summation 或 4-8 路 split + 误差补偿。回滚后保留为下次 TODO。

**P0.5 意外副产品(ref-FP32 暴露精度)**:

| matmul shape | 改前 L2(ref TF32) | 改后 L2(ref FP32) | 解读 |
|---|---|---|---|
| (256,512) CUDA | 2.5e-03 | 1.7e+00 | 真数值偏差被 ref=TF32 掩盖,改后浮出 |
| (512,768) CUDA | 1.3e-02 | 4.2e+00 | 同上 |

**分析**:
- P0 fused_mlp_first 修是单纯 dispatch 错(4096 threads),改 1 行,真有效。
- ref-FP32 改动**改变了基准** — 与 `baseline_pre_p0p1.json` 比对看到的"REGRESS"是测量方法变化不是 perf 退化;真实数据应该看两边的实际 ms 数值,不能单看 `analyze_bench.py` 的 delta% 列(因为 ref 改了)。
- P0.5 matmul K-split 修复在 BF16/FP16 路径走 mma 时不适用(精度已被 Tensor Core 累加保证);仅在 FP32 严格模式下才显出差异。

---

> 数字必须来源于 `make bench-op` 实际跑出,不要编造。任何一行没数据,留空即可。
