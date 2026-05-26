"""
Triton kernel 集合

matmul:      分块矩阵乘法
elementwise: BiasAdd, ReLU, GELU, SiLU, 融合 BiasAdd+ReLU
backward:    MatMul backward, Activation backward
dropout:     Inverted dropout
loss:        融合 CrossEntropy
mlp_triton:  融合 MLP first layer (matmul+bias+GELU)
swiglu:      融合 SwiGLU
"""

from triton_kernels.matmul import tiled_matmul
from triton_kernels.elementwise import bias_add, relu, gelu, silu, bias_add_relu
from triton_kernels.backward import (
    matmul_backward_a, matmul_backward_b,
    relu_backward, gelu_backward, silu_backward,
)
from triton_kernels.dropout import triton_dropout
from triton_kernels.loss import cross_entropy
from triton_kernels.mlp_triton import mlp_first_layer_triton
from triton_kernels.swiglu_triton import swiglu_triton
from triton_kernels.precision import precision, set_precision
from triton_kernels.layernorm import layernorm_forward, layernorm_backward
