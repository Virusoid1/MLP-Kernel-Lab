# KNOWN-LIMITATIONS.md

> 未实现 / 未验证 / 硬件限制的明确清单。写简历时：只有 E2 能写正确性、E3/E4 才能写性能数字，
> 本清单内容不得当作已完成主张。

## 未实现

| 项 | 状态 | 影响 |
|---|---|---|
| ~~CUDA 算子级 fp16/bf16~~ | **全解锁（2026-09-02）：swiglu_fused / softmax / relu / gelu / silu / bias_add / matmul / fused_mlp_first 的 fp16+bf16 变体**（低精度 in→fp32 math→同精度 out；fused_first 走 WMMA 组合） | ~~blocked~~ ✅ 已解锁：**dtype matrix cuda 16/16 行 PASSED（3070 + 5070 Ti 双架构）**；cuda fp16 block 不再回退 F.silu |
| ~~cuda bf16（算子级 vs 块级）~~ | **全自定义（2026-09-02）：算子级 bf16（swiglu_fused/softmax/relu/gelu/silu）+ 块级 matmul_bf16（WMMA，bf16 in→fp32 acc→bf16 out）** | ✅ 短板消除：cuda fp16+bf16 块级+算子级全闭环（dtype matrix 32p/16s；cuda 块 17p，3070 + 5070 Ti 双架构） |
| ~~torch.library 集成仅覆盖 swiglu~~ | **已推广（2026-09-02）：mlp_kernel::matmul + mlp_kernel::layernorm（schema/CPU+CUDA/Meta/autograd + opcheck/gradcheck/compile）**，21 registration tests 全绿 | ~~工作量扩展~~ ✅ 已实现（3070 验证） |
| 多 GPU / FP8 | 有意不做 | 超出 kernel 项目范围 |

## 未验证（需其它硬件/权限）

| 项 | 状态 | 需要的条件 |
|---|---|---|
| E4 跨设备复现 | 3070 + 3080 Ti + 3090 Ti + 5070 Ti 均已测且 raw JSON 入 git（artifacts_3080ti/ + artifacts_3090ti/ + artifacts_5070ti/）；5070 Ti 上 mlp_cuda 已为 sm120 编译并验证（transformer_mlp -k cuda 15p/2s，cuda_kernels 16p，P1 8/8） | **闭环达成（全四机 + Blackwell cuda 后端）**；cuda 后端 fp16 性能在 Blackwell 仍慢于 cuBLAS（如实记录，见 artifacts_5070ti/fp16_prefill_cuda_5070ti.json） |
| Blackwell（sm120）工具链 | 5070 Ti 已实机（nvcc 13.2 / torch 2.13+cu130 / triton 3.7.1）；preflight lane=blackwell + cutile_probe=True | ~~5070 Ti + CUDA ≥13 + 匹配 torch~~ ✅ 已跑 |
| cuTile 在 Blackwell 的性能（2026-09-02 实测，负结果） | **M512×4096×11008 cutile fp16 62.7ms / bf16 59.9ms —— 3070 同 shape 11.4ms，Blackwell 反而 5x 慢**（ct.mma 在 sm120 未获 tcgen05 加速或 tile 不匹配） | **推翻「Blackwell 才是 cuTile 主战场」旧假设**；证据 = artifacts_5070ti/swiglu_bench_fp16_bf16.json（84 rows correctness 100%） |
| ncu/nsys 硬件计数器 | WSL 无 sudo 权限（ERR_NVGPUCTRPERM） | sudo 或非 WSL 环境 |
| ~~训练闭环在 fp16 全工况~~ | **已消除（2026-09）：新增 tests/test_training_loop.py 的 fp16 用例** —— eager(cuBLAS fp16) 与 Triton 自定义后端 fp16 均收敛（loss rel<0.005），2/2 测试通过 | ✅ 已验证 |

## 硬件限制（RTX 3070 Laptop）

| 限制 | 影响 |
|---|---|
| 8GB VRAM | 大 shape（M≥4096×4096×11008）压力测试受限；留给 3090 Ti |
| 显存带宽（3070 Laptop 理论 448 GB/s；decode 实测利用仅 13-20%） | 单 token decode 物理限制（每次重读全部权重）；多 token 摊销 ↓82.9x 是解法 |
| WSL2 环境 | ncu/nsys 权限、热漂移、性能波动需插电+固定模式 |
| 40 SM | 大 tile 配置受 SM 数限制（与 5070 Ti 96 SM 不同最优解） |

## 已知数值边界

| 边界 | 说明 |
|---|---|
| fp16 scale=1.0 时 K=3072 累加溢出 | kb16 matmul 需 scale≤0.1 稳定输入（eager 实测） |
| bf16 尾数 | norm_l2 ~4e-3（合理但比 fp16 粗） |
| TF32 不稳定 | 3070 上 2048-shape 0.78x（autotune 噪声），FP16 才是稳定主路径 |
