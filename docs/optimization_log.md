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
| v1-v6 | — | — | — | — | — |

> 数字必须来源于 `make bench-op` 实际跑出,不要编造。任何一行没数据,留空即可。
