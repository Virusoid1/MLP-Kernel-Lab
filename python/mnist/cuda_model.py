"""
CUDA MLP 模型

使用 CUDA kernel（CUDALinear + CUDAActivation）构建 MLP，
接口与 PyTorch MLP（model.py）和 Triton MLP（triton_model.py）完全对称。
"""

from python.mnist.model import MLPConfig
from python.mnist.cuda_layers import CUDALinear, CUDAActivation, CUDALayerNorm

import torch.nn as nn


class CUDAMLP(nn.Module):
    """使用 CUDA kernel 的可配置多层 MLP。结构与 model.py:MLP 对称。"""

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_activation = []

        dims = config.hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(CUDALinear(dims[i], dims[i + 1]))
            is_last = (i == len(dims) - 2)
            if not is_last:
                if config.use_layernorm:
                    self.norms.append(CUDALayerNorm(dims[i + 1]))
                else:
                    self.norms.append(nn.Identity())
                self.activations.append(CUDAActivation(config.activation))
                self.use_activation.append(True)
            else:
                self.norms.append(nn.Identity())
                self.activations.append(nn.Identity())
                self.use_activation.append(False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(1)
        elif x.dim() == 2 and x.shape[1] != self.config.hidden_dims[0]:
            x = x.flatten(1)

        for i, linear in enumerate(self.layers):
            x = linear(x)
            x = self.norms[i](x)
            x = self.activations[i](x)
            if self.use_activation[i] and self.dropout is not None:
                x = self.dropout(x)

        return x

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
