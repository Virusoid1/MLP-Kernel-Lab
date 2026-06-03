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

### v1-v6: (历史 matmul 优化系列,占位待填)

下方为原模板,仅当真实有数据时再填。请勿编造。

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
