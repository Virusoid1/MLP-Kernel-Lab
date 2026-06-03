"""
MLP 模型定义

可配置层数/宽度/激活函数/dropout，后续供 Triton kernel 版本对比。
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    """MLP 架构配置，可序列化到 JSON 用于结果对比。"""

    hidden_dims: list[int] = field(default_factory=lambda: [784, 256, 128, 10])
    activation: str = "gelu"  # relu | gelu | silu
    dropout: float = 0.0
    use_layernorm: bool = False
    use_residual: bool = False  # 残差连接(post-LN,要求相邻层宽相同)

    def to_dict(self) -> dict:
        return {
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
            "dropout": self.dropout,
            "use_layernorm": self.use_layernorm,
            "use_residual": self.use_residual,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MLPConfig":
        return cls(
            hidden_dims=list(d.get("hidden_dims", [784, 256, 128, 10])),
            activation=d.get("activation", "gelu"),
            dropout=d.get("dropout", 0.0),
            use_layernorm=d.get("use_layernorm", False),
            use_residual=d.get("use_residual", False),
        )

    @classmethod
    def deep_narrow(cls) -> "MLPConfig":
        """8 层 256 hidden,深 + 多次 LayerNorm(每层 LN).
        用于测试 LN 摊销和深度对 4 backend 的差异化影响.
        """
        return cls(
            hidden_dims=[784, 256, 256, 256, 256, 256, 256, 256, 10],
            activation="gelu",
            use_layernorm=True,
            use_residual=False,
        )

    @classmethod
    def wide_skip(cls) -> "MLPConfig":
        """3 层 1024 hidden + 残差(post-LN).
        第一层 784→1024,后两层 1024→1024 形成残差块;最后一层 1024→10 无残差.
        """
        return cls(
            hidden_dims=[784, 1024, 1024, 1024, 10],
            activation="gelu",
            use_layernorm=True,
            use_residual=True,
        )


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": lambda: nn.GELU(approximate="tanh"),
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    """可配置多层 MLP。最后一层无激活，输出 logits。"""

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_activation = []  # 标记哪些层需要激活

        dims = config.hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last_hidden = (i == len(dims) - 2)
            if not is_last_hidden:
                if config.use_layernorm:
                    self.norms.append(nn.LayerNorm(dims[i + 1]))
                else:
                    self.norms.append(nn.Identity())
                self.activations.append(_ACTIVATIONS[config.activation]())
                self.use_activation.append(True)
            else:
                self.norms.append(nn.Identity())
                self.activations.append(nn.Identity())
                self.use_activation.append(False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 28, 28) 或 (B, 784)
        返回: (B, num_classes) logits

        残差块 (use_residual=True 时): 相邻两个同宽 layer 形成 post-LN 残差.
        形如: h = act(linear_a(x)); x = LN(x + linear_b(h))
        要求 layer_a.in == layer_a.out == layer_b.in == layer_b.out.
        """
        if x.dim() == 4:
            x = x.flatten(1)
        elif x.dim() == 2 and x.shape[1] != self.config.hidden_dims[0]:
            x = x.flatten(1)

        i = 0
        n_layers = len(self.layers)
        while i < n_layers:
            linear = self.layers[i]
            # 判断是否能和下一个 layer 形成同宽残差块
            # 残差块: linear_a(in → mid) + linear_b(mid → mid) 要求 in == mid
            # 这样 x (in-dim) 加上 block 输出 (mid-dim) shape 一致
            if (
                self.config.use_residual
                and i + 1 < n_layers
                and linear.in_features == linear.out_features  # mid == in, 残差维度匹配
                and self.layers[i + 1].in_features == self.layers[i + 1].out_features  # 同样 mid
                and linear.in_features == self.layers[i + 1].in_features  # 衔接处 mid 一致
                and i + 1 < n_layers - 1  # 不在最后一层形成残差
            ):
                # post-LN 残差块: x = LN(x + linear_b(act(linear_a(x))))
                linear_a = linear
                linear_b = self.layers[i + 1]
                h = linear_a(x)
                h = self.norms[i](h)
                h = self.activations[i](h)
                h = linear_b(h)
                x = x + h
                # 输出 norm: 残差块的 norm 归到下一个 index
                # 简化: 不加额外 norm,直接用 self.norms[i+1] 作为 block 输出 norm
                if self.config.use_layernorm and isinstance(self.norms[i + 1], nn.LayerNorm):
                    x = self.norms[i + 1](x)
                i += 2
            else:
                x = linear(x)
                x = self.norms[i](x)
                x = self.activations[i](x)
                if self.use_activation[i] and self.dropout is not None:
                    x = self.dropout(x)
                i += 1

        return x

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
