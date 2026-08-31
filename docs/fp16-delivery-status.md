# v2 fp16/bf16 交付状态（2026-09-01, RTX 3070）

## 一句话

SwiGLU MLP block 的 fp16 正确性在 **六后端全闭环**（eager/triton/triton_fused/cuda/cutile/compile），
3070 性能主路径 = **Triton**（vs eager-fp32 至 4.07x）。

## 正确性（块级，norm_l2 2.4e-4~6e-4，42-case sweep corr 100%）

| 后端 | fp16 | bf16 |
|---|---|---|
| eager（参考） | ✅ | ✅ |
| triton | ✅ | ✅ |
| triton_fused | ✅ | ✅ |
| cuda（matmul_half） | ✅ | fp32-only（binding）|
| cutile（dtype 传播修复后）| ✅ | ✅ |
| compile | ✅ | ✅ |

## 性能地图（M=512×4096×11008, fp16, median ms）

triton 4.60 < compile 6.42 < triton_fused 7.02 < cutile 11.8（tile 优化后）< cuda 29-118

（cuda/cutile 慢于 eager16 5-7x —— wmma32/ct.mma tile 不敌 cuBLAS，诚实记录）

## 边界（已知限制）

1. **算子级 CUDA fp16**：swiglu/softmax 等算子仍被 binding CHECK_FLOAT32 阻塞；块级用它不依赖（epilogue 用 PyTorch F.silu，matmul 用 matmul_half）
2. **cuda bf16**：matmul_half 仅 fp16，bf16 走 fp32 tiled（正确但慢）
3. **Triton bf16 sigmoid**：已升 fp32 修复（silu/swiglu 均 OK）

## 复现

```bash
make reproduce PYTHON=<venv>/bin/python   # 209 tests (172p/0f/37s)
python bench/run.py --suite prefill --dtypes fp16 --backends eager,concat,triton,triton_fused,cuda,cutile,compile
```
