# v2 fp16/bf16 交付状态（2026-09-01, RTX 3070）

## 一句话

SwiGLU MLP block 的 fp16 正确性在 **六后端全闭环**（eager/triton/triton_fused/cuda/cutile/compile），
3070 性能主路径 = **Triton**（vs eager-fp32 至 3.52x，all-suite 266-case）。

## 正确性（块级，norm_l2 2.4e-4~6e-4，42-case sweep corr 100%）

| 后端 | fp16 | bf16 |
|---|---|---|
| eager（参考） | ✅ | ✅ |
| triton | ✅ | ✅ |
| triton_fused | ✅ | ✅ |
| cuda（matmul_half/matmul_bf16） | ✅ | ✅（算子级 + 块级 matmul_half/matmul_bf16 全自定义，2026-09-02） |
| cutile（dtype 传播修复后）| ✅ | ✅ |
| compile | ✅ | ✅ |

## 性能地图（M=512×4096×11008, fp16, median ms）

triton 4.46 < eager 5.07 ≈ compile 5.24 < concat 5.70 < triton_fused 6.30 < **cutile 6.50（tile64+pair, 0.77x）** < **cuda 13.4ms（cp.async 16B 管线+pair, 0.38x）**（2026-09-02 最终冻结协议）

（cuda/cutile 同精度仍低于 cuBLAS，但已从 0.16x/0.55x 分别提升至 0.38x/0.77x，诚实记录）

## 边界（已知限制）

1. ~~**算子级 CUDA fp16**~~：~~swiglu/softmax 等算子仍被 binding CHECK_FLOAT32 阻塞；块级用它不依赖（epilogue 用 PyTorch F.silu，matmul 用 matmul_half）~~ **已解锁（2026-09-02）**：binding fp16 分派 + half kernels（swiglu_fused/softmax/relu/gelu/silu，fp16 in→fp32 math→fp16 out）；cuda fp16 block 全自定义（matmul_half + swiglu_fused_half），不再回退 F.silu，norm_l2 5e-4（dtype matrix cuda-fp16 5 行 PASSED）
2. **cuda bf16**：matmul_half 仅 fp16，bf16 走 fp32 tiled（正确但慢）
3. **Triton bf16 sigmoid**：已升 fp32 修复（silu/swiglu 均 OK）

## 复现

```bash
make reproduce PYTHON=<venv>/bin/python   # 235 tests (216p/0f/19s, 2110876 冻结)
python bench/run.py --suite prefill --dtypes fp16 --backends eager,concat,triton,triton_fused,cuda,cutile,compile
```
