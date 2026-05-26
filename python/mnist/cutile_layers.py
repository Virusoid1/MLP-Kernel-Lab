"""
cuTile autograd 层：CUTILELinear + CUTILEActivation + CUTILELayerNorm

使用 torch.autograd.Function 包装 cuTile kernel，使其可嵌入 PyTorch 训练循环。
"""

import math

import torch
import torch.nn as nn
import torch.nn.init as init

from cutile_kernels.matmul import cutile_matmul
from cutile_kernels.backward import matmul_backward_a, matmul_backward_b
from cutile_kernels.backward import relu_backward, gelu_backward, silu_backward
from cutile_kernels.elementwise import bias_add, relu, gelu, silu
from cutile_kernels.layernorm import layernorm_forward, layernorm_backward


class CUTILELinearFunction(torch.autograd.Function):
    """cuTile 线性层前向/反向。forward: output = input @ weight + bias"""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        input_f = input.float().contiguous()
        weight_f = weight.float().contiguous()

        output = cutile_matmul(input_f, weight_f)
        if bias is not None:
            output = bias_add(output, bias.float())

        ctx.save_for_backward(input_f, weight_f)
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_f, weight_f = ctx.saved_tensors
        grad_output_f = grad_output.contiguous().float()

        grad_input = matmul_backward_a(grad_output_f, weight_f)
        grad_weight = matmul_backward_b(input_f, grad_output_f)
        grad_bias = torch.sum(grad_output_f, dim=0) if ctx.has_bias else None

        if grad_input.dtype != grad_output.dtype:
            grad_input = grad_input.to(grad_output.dtype)
        if grad_weight.dtype != grad_output.dtype:
            grad_weight = grad_weight.to(grad_output.dtype)
        if grad_bias is not None and grad_bias.dtype != grad_output.dtype:
            grad_bias = grad_bias.to(grad_output.dtype)

        return grad_input, grad_weight, grad_bias


class CUTILELinear(nn.Module):
    """使用 cuTile kernel 的线性层。接口与 nn.Linear 一致。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CUTILELinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


_ACT_FORWARD = {
    'relu': relu,
    'gelu': gelu,
    'silu': silu,
}

_ACT_BACKWARD = {
    'relu': relu_backward,
    'gelu': gelu_backward,
    'silu': silu_backward,
}


class CUTILEActivationFunction(torch.autograd.Function):
    """cuTile 激活函数前向/反向。"""

    @staticmethod
    def forward(ctx, input: torch.Tensor, activation: str) -> torch.Tensor:
        input_f = input.float()
        output = _ACT_FORWARD[activation](input_f)
        ctx.save_for_backward(input_f)
        ctx.activation = activation
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input_f,) = ctx.saved_tensors
        grad_output_f = grad_output.contiguous().float()
        grad_input = _ACT_BACKWARD[ctx.activation](grad_output_f, input_f)

        if grad_input.dtype != grad_output.dtype:
            grad_input = grad_input.to(grad_output.dtype)
        return grad_input, None


class CUTILEActivation(nn.Module):
    """使用 cuTile kernel 的激活函数模块。"""

    def __init__(self, activation: str = 'gelu'):
        super().__init__()
        assert activation in _ACT_FORWARD, f"Unsupported activation: {activation}"
        self.activation = activation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CUTILEActivationFunction.apply(input, self.activation)

    def extra_repr(self) -> str:
        return f'activation={self.activation}'


class CUTILELayerNormFunction(torch.autograd.Function):
    """cuTile LayerNorm 前向/反向。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        x_f = x.float().contiguous()
        w_f = weight.float().contiguous()
        b_f = bias.float().contiguous()
        y, mean, rstd = layernorm_forward(x_f, w_f, b_f, eps)
        ctx.save_for_backward(x_f, w_f, mean, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_f, w_f, mean, rstd = ctx.saved_tensors
        grad_output_f = grad_output.contiguous().float()
        dx, d_weight, d_bias = layernorm_backward(grad_output_f, x_f, w_f, mean, rstd)

        if dx.dtype != grad_output.dtype:
            dx = dx.to(grad_output.dtype)
        if d_weight.dtype != grad_output.dtype:
            d_weight = d_weight.to(grad_output.dtype)
        if d_bias.dtype != grad_output.dtype:
            d_bias = d_bias.to(grad_output.dtype)
        return dx, d_weight, d_bias, None


class CUTILELayerNorm(nn.Module):
    """使用 cuTile kernel 的 LayerNorm。"""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        return CUTILELayerNormFunction.apply(x, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}'
