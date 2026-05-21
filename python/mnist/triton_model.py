"""
Triton MLP 占位文件

后续实现：使用手写 Triton kernel（tiled_matmul, bias_add, gelu, 等）
替换 PyTorch 的 nn.Linear + nn.GELU，组成与 model.py:MLP 同架构的网络。

接口约定：
- 接受同样的 MLPConfig
- forward 输出与 PyTorch MLP 同 shape
- 所有计算使用 float32
- 训练使用自定义 backward kernel（matmul_backward_a/b, gelu_backward 等）

Triton kernel 来源：
    ~/projects/Salvation-Lies-Within/Triton/tiled_matmul.py   → tiled_matmul(a,b)
    ~/projects/Salvation-Lies-Within/Triton/elementwise.py    → bias_add, gelu, relu, silu
    ~/projects/Salvation-Lies-Within/Triton/dropout.py        → triton_dropout
    ~/projects/Salvation-Lies-Within/Triton/loss.py           → cross_entropy
    ~/projects/Salvation-Lies-Within/Triton/backward.py       → matmul_backward_a/b, gelu_backward
"""

# 预留 — 待 Triton kernel 集成后实现
