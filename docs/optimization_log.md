# Optimization Log

> Day 9 起记录每次优化的变更、效果和分析。

## 格式

```text
### vN: 版本描述
**变更**: 
**预期**: 
**实际**: 
**分析**: 
```

## 优化记录

### v0: naive matmul

**Shape**: M=512, K=768, N=3072 (FP32)
**Latency**: ___ ms
**TFLOPS**: ___
**速度提升**: 1.0x (baseline)

---

### v1: shared memory tiling

**变更**: 加载 A/B tile 到 shared memory
**BLOCK**: 16x16x16
**Latency**: ___ ms
**TFLOPS**: ___
**速度提升**: ___x

**分析**: (待填写)

---

### v2: block size 调整

**变更**: 测试不同的 BLOCK 配置
**测试配置**:

| BLOCK_M | BLOCK_N | BLOCK_K | Latency | 说明 |
|---------|---------|---------|---------|------|
| 16 | 16 | 16 | | |
| 32 | 32 | 16 | | |
| 32 | 32 | 32 | | |

**分析**: (待填写)

---

### v3: coalesced memory access

**变更**: 调整 shared memory 存储布局
**Latency**: ___ ms
**速度提升**: ___x

**分析**: (待填写)

---

### v4: fused bias + GELU

**变更**: 在 tiled matmul kernel 中直接计算 bias + GELU
**Latency (unfused)**: ___ ms → **Latency (fused)**: ___ ms
**速度提升**: ___x

**分析**: (待填写)

---

### v5: FP16

**变更**: 支持 FP16 (half)
**FP32 Latency**: ___ ms → **FP16 Latency**: ___ ms
**速度提升**: ___x

**分析**: (待填写)

---

### v6: (your optimization)

**变更**: ___
**Latency**: ___ ms
**速度提升**: ___x

**分析**: ___

---

## 汇总表

| Version | Change | Latency (ms) | TFLOPS | Speedup vs v0 |
|---------|--------|-------------|--------|---------------|
| v0 | naive | | | 1.0x |
| v1 | shared memory tile | | | __x |
| v2 | block size tuning | | | __x |
| v3 | coalesced access | | | __x |
| v4 | fused activation | | | __x |
| v5 | FP16 | | | __x |
| v6 | ___ | | | __x |

> 数字来源于实际 benchmark，不要编造。
