# 多机兼容矩阵（v2 其它机器）

目标: 同一 commit 在 3070 / 3080 Ti / 3090 Ti / 5070 Ti 上都可构建、可测、可比较。
原则: 每台机器与它自己的 PyTorch eager baseline 比 speedup，跨 GPU 不比毫秒；Ampere 三卡同为 sm_86 视为同架构复现。

## Lane / 构建

| GPU | CC | Lane | TORCH_CUDA_ARCH_LIST | 定位 |
|---|---|---|---|---|
| RTX 3070 Laptop | 8.6 | Ampere | 8.6 | 主开发（本机实测） |
| RTX 3080 Ti | 8.6 | Ampere | 8.6 | 同架构独立复现 |
| RTX 3090 Ti | 8.6 | Ampere | 8.6 | 大显存压力 / 最终 Ampere 数据 |
| RTX 5070 Ti | 12.0 | Blackwell | 12.0 | 跨架构移植 / cuTile / Blackwell 调优 |

> \`setup.py\` 与 \`Makefile\` 已动态探测（先 \`TORCH_CUDA_ARCH_LIST\`，否则 \`torch.cuda.get_device_capability()\`）。
> Blackwell 机器不要直接用 Ampere 的 \`.so\`；\`make install\` 会在本机重编译。

## 每台机器流程

\`\`\`bash
python tools/preflight.py          # 检查 lane/arch/toolchain，status=0 才能继续
pip install -r requirements.txt    # cuTile 可选: pip install cuda-tile
make reproduce PYTHON=<venv>/bin/python   # 构建 -> 136 测试 -> bench -> manifest 归档
python bench/run.py --suite all --dtypes fp32,fp16   # SwiGLU 全 shape sweep
python bench/run.py --suite decode --dtypes fp16     # decode 档
\`\`\`

## 已知机器差异（实测/推断）

| 差异 | Ampere (3070/3080Ti/3090Ti) | Blackwell (5070 Ti) |
|---|---|---|
| WMMA64 路径 | 需 launch_bounds(512)（已修，见 CHANGELOG 2026-06-03） | 原生可用 |
| cuTile tile | gpu_utils 用 (64,64,32) 族配置 | (16,16,16) native fragment（gpu_utils 自动） |
| fp16 TensorCore | Triton input_precision 可用（本机 3.0-3.9x） | 同理 + cuTile 原生支持 |
| CUDA matmul strict FP32 | 走 matmul_tiled 32×32 + Kahan（精度取舍，见 claim-matrix） | 同（cuda 大 shape 仍慢） |
| 3070 Laptop 热漂移 | 需插电/固定性能模式；bench 用小 M 多拍降低噪声 | 桌面级更稳定 |

## 复现声明策略

- 3070: 开发/正确性主战场 → 数字均带 manifest（artifacts/<run-id>/）
- 3080 Ti/3090 Ti: 跑同一 make reproduce + 全 sweep，验证"同架构第二设备"一致性（E4 复现）
- **E4 实操手册：[docs/e4-runbook.md](e4-runbook.md)**（事前准备→到机验证→全量复现→对比模板→Blackwell 注意）
- 5070 Ti: 先跑 preflight + smoke（Blackwell portability），再独立 autotune cache（不做跨架构对数字）

## 已知环境注意事项

1. **工具链匹配**: torch 2.11.0+cu130 需要 nvcc ≥13。若 PATH 里是旧 nvcc（如 apt CUDA 12），
   \`tools/preflight.py\` 会警告；\`tools/reproduce.py\` 构建时会自动在 /usr/local/cuda* 找匹配版本。
2. **cuTile import 名**: \`import cuda.tile\`（pip 包 cuda-tile）。
3. **GPU 性能计数器**: WSL2 下 ncu/nsys 需 sudo（ERR_NVGPUCTRPERM）；无 sudo 时用软件拆段测时替代。
4. **3070 小 M decode**: 融合 kernel 无效（权重带宽 bound），大 decode batch 摊销 ↓80x / token（见实验报告）。
