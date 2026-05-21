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

    def to_dict(self) -> dict:
        return {
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MLPConfig":
        return cls(
            hidden_dims=list(d.get("hidden_dims", [784, 256, 128, 10])),
            activation=d.get("activation", "gelu"),
            dropout=d.get("dropout", 0.0),
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
        self.use_activation = []  # 标记哪些层需要激活

        dims = config.hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last_hidden = (i == len(dims) - 2)
            if not is_last_hidden:
                self.activations.append(_ACTIVATIONS[config.activation]())
                self.use_activation.append(True)
            else:
                self.activations.append(nn.Identity())
                self.use_activation.append(False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 28, 28) 或 (B, 784)
        返回: (B, num_classes) logits
        """
        if x.dim() == 4:
            x = x.flatten(1)
        elif x.dim() == 2 and x.shape[1] != self.config.hidden_dims[0]:
            x = x.flatten(1)

        for i, linear in enumerate(self.layers):
            x = linear(x)
            x = self.activations[i](x)
            if self.use_activation[i] and self.dropout is not None:
                x = self.dropout(x)

        return x

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
