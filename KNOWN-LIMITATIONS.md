# KNOWN-LIMITATIONS.md

> 未实现 / 未验证 / 硬件限制的明确清单。写简历时：只有 E2 能写正确性、E3/E4 才能写性能数字，
> 本清单内容不得当作已完成主张。

## 未实现

| 项 | 状态 | 影响 |
|---|---|---|
| CUDA 算子级 fp16（swiglu/softmax 等） | binding CHECK_FLOAT32 阻塞；仅 matmul_half 解锁 | 不影响块级主线（fp16 epilogue 用 PyTorch F.silu）；MNIST 历史算子集受限 |
| cuda bf16 | matmul_half 仅 fp16；bf16 走 fp32 tiled（正确但慢） | 四后端 bf16 中 cuda 是短板 |
| torch.library 集成仅覆盖 swiglu | 未推广到 matmul/layernorm | 证明链已完整，推广是工作量扩展 |
| 多 GPU / FP8 | 有意不做 | 超出 kernel 项目范围 |

## 未验证（需其它硬件/权限）

| 项 | 状态 | 需要的条件 |
|---|---|---|
| E4 跨设备复现 | 3070 + 3080 Ti + 3090 Ti + 5070 Ti 均已测且 raw JSON 入 git（artifacts_3080ti/ + artifacts_3090ti/ + artifacts_5070ti/） | **闭环达成**；唯一剩余项 = 5070 Ti 上的 cuda 后端未编译（见下） |
| Blackwell（sm120）工具链 | 5070 Ti 已实机（nvcc 13.2 / torch 2.13+cu130 / triton 3.7.1）；preflight lane=blackwell + cutile_probe=True | ~~5070 Ti + CUDA ≥13 + 匹配 torch~~ ✅ 已跑 |
| cuTile 在 5070 Ti 的性能 | 3070 慢（11.8ms）；5070 Ti 上 cutile 可 import（driver 610 满足 r580+）但完整 sweep 未跑 | 5070 Ti 实机（cutile_probe=True 已确认，性能 sweep 未做） |
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
