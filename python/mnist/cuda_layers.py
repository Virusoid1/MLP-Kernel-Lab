"""
CUDA autograd 层：CUDALinear + CUDAActivation

使用 torch.autograd.Function 包装 CUDA kernel（编译为 mlp_cuda 模块），
使其可嵌入 PyTorch 训练循环。

- CUDALinearFunction: forward(mlp_cuda.matmul_tiled + bias_add), backward(mlp_cuda.matmul_tiled)
- CUDALinear: nn.Module 封装
- CUDAActivationFunction: forward(activation), backward(activation_backward)

与 triton_layers.py 结构完全对称。
"""

import math

import torch
import torch.nn as nn
import torch.nn.init as init

try:
    import mlp_cuda
    _HAS_CUDA = True
except ImportError:
    mlp_cuda = None
    _HAS_CUDA = False


def _check_cuda_available():
    if not _HAS_CUDA:
        raise RuntimeError(
            "mlp_cuda 模块未安装。请运行: pip install -e . 或 python setup.py install"
        )


class CUDALinearFunction(torch.autograd.Function):
    """CUDA 线性层前向/反向。forward: output = input @ weight + bias

    默认使用自定义 CUDA kernel；use_cublas=True 时回退到 cuBLAS。
    """

    # 类级别开关：True 时 forward/backward 全部走 cuBLAS
    use_cublas = False

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        _check_cuda_available()

        input_f = input.float()
        weight_f = weight.float()

        if CUDALinearFunction.use_cublas:
            output = torch.matmul(input_f, weight_f)
        else:
            output = mlp_cuda.matmul_tiled_auto(input_f, weight_f)

        if bias is not None:
            output = output + bias.float()

        ctx.save_for_backward(input_f, weight_f)
        ctx.has_bias = bias is not None
        ctx.used_cublas = CUDALinearFunction.use_cublas
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        _check_cuda_available()
        input_f, weight_f = ctx.saved_tensors
        grad_output_f = grad_output.contiguous().float()

        if ctx.used_cublas:
            grad_input = torch.matmul(grad_output_f, weight_f.t())
            grad_weight = torch.matmul(input_f.t(), grad_output_f)
        else:
            grad_input = mlp_cuda.matmul_transB(grad_output_f, weight_f)
            grad_weight = mlp_cuda.matmul_transA(input_f, grad_output_f)

        grad_bias = grad_output_f.sum(0) if ctx.has_bias else None

        if grad_input.dtype != grad_output.dtype:
            grad_input = grad_input.to(grad_output.dtype)
        if grad_weight.dtype != grad_output.dtype:
            grad_weight = grad_weight.to(grad_output.dtype)
        if grad_bias is not None and grad_bias.dtype != grad_output.dtype:
            grad_bias = grad_bias.to(grad_output.dtype)

        return grad_input, grad_weight, grad_bias


class CUDALinear(nn.Module):
    """
    使用 CUDA kernel 的线性层。接口与 nn.Linear 一致。
    weight 存储为 (in_features, out_features)，与 TritonLinear 一致。
    """

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
        if not input.is_contiguous():
            input = input.contiguous()
        # 推理模式：跳过 autograd.Function 包装，减少开销
        if not torch.is_grad_enabled():
            input_f = input.float()
            weight_f = self.weight.float()
            if CUDALinearFunction.use_cublas:
                output = torch.matmul(input_f, weight_f)
            else:
                output = mlp_cuda.matmul_tiled_auto(input_f, weight_f)
            if self.bias is not None:
                output = output + self.bias.float()
            return output
        return CUDALinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


# 激活函数前向/反向映射
_ACT_FORWARD = {
    'relu': lambda x: mlp_cuda.relu(x) if _HAS_CUDA else None,
    'gelu': lambda x: mlp_cuda.gelu(x) if _HAS_CUDA else None,
    'silu': lambda x: mlp_cuda.silu(x) if _HAS_CUDA else None,
}

_ACT_BACKWARD = {
    'relu': lambda g, x: mlp_cuda.relu_backward_vec4(g, x) if _HAS_CUDA else None,
    'gelu': lambda g, x: mlp_cuda.gelu_backward_vec4(g, x) if _HAS_CUDA else None,
    'silu': lambda g, x: mlp_cuda.silu_backward_vec4(g, x) if _HAS_CUDA else None,
}


class CUDAActivationFunction(torch.autograd.Function):
    """CUDA 激活函数前向/反向。"""

    @staticmethod
    def forward(ctx, input: torch.Tensor, activation: str) -> torch.Tensor:
        _check_cuda_available()
        input_f = input.float()
        output = _ACT_FORWARD[activation](input_f)
        ctx.save_for_backward(input_f)
        ctx.activation = activation
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        _check_cuda_available()
        (input_f,) = ctx.saved_tensors
        grad_output_f = grad_output.contiguous().float()
        grad_input = _ACT_BACKWARD[ctx.activation](grad_output_f, input_f)

        if grad_input.dtype != grad_output.dtype:
            grad_input = grad_input.to(grad_output.dtype)
        return grad_input, None


class CUDAActivation(nn.Module):
    """使用 CUDA kernel 的激活函数模块。"""

    def __init__(self, activation: str = 'gelu'):
        super().__init__()
        assert activation in _ACT_FORWARD, f"Unsupported activation: {activation}"
        self.activation = activation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_contiguous():
            input = input.contiguous()
        # 推理模式：直接用 CUDA kernel，跳过 autograd.Function 开销
        if not torch.is_grad_enabled():
            return _ACT_FORWARD[self.activation](input.float())
        return CUDAActivationFunction.apply(input, self.activation)

    def extra_repr(self) -> str:
        return f'activation={self.activation}'


class CUDALayerNormFunction(torch.autograd.Function):
    """CUDA LayerNorm 前向/反向。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        _check_cuda_available()
        x_f = x.float()
        w_f = weight.float()
        b_f = bias.float()
        y, mean, rstd = mlp_cuda.layernorm_forward(x_f, w_f, b_f, eps)
        ctx.save_for_backward(x_f, w_f, mean, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        _check_cuda_available()
        x_f, w_f, mean, rstd = ctx.saved_tensors
        grad_output_f = grad_output.contiguous().float()
        dx, d_weight, d_bias = mlp_cuda.layernorm_backward(
            grad_output_f, x_f, w_f, mean, rstd
        )

        if dx.dtype != grad_output.dtype:
            dx = dx.to(grad_output.dtype)
        if d_weight.dtype != grad_output.dtype:
            d_weight = d_weight.to(grad_output.dtype)
        if d_bias.dtype != grad_output.dtype:
            d_bias = d_bias.to(grad_output.dtype)
        return dx, d_weight, d_bias, None


class CUDALayerNorm(nn.Module):
    """使用 CUDA kernel 的 LayerNorm。接口与 nn.LayerNorm 一致。"""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        return CUDALayerNormFunction.apply(x, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}'
