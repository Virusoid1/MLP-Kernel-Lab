"""
PyTorch MLP reference implementation

用于正确性对比和 benchmark baseline
"""

import torch
import torch.nn.functional as F


def matmul_ref(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """矩阵乘法 baseline: C = A @ B"""
    return A @ B


def mlp_first_layer_ref(
    X: torch.Tensor, W1: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """MLP first layer: H = GELU(X @ W1 + bias)"""
    return F.gelu(X @ W1 + bias)


def mlp_two_layer_ref(
    X: torch.Tensor,
    W1: torch.Tensor, b1: torch.Tensor,
    W2: torch.Tensor, b2: torch.Tensor,
) -> torch.Tensor:
    """完整 MLP: Y = GELU(X @ W1 + b1) @ W2 + b2"""
    h = F.gelu(X @ W1 + b1)
    return h @ W2 + b2


def swiglu_ref(
    gate: torch.Tensor, up: torch.Tensor,
    W_down: torch.Tensor, b_down: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU FFN (Llama-style):
    hidden = SiLU(gate) * up
    out = hidden @ W_down + b_down
    """
    hidden = F.silu(gate) * up
    return hidden @ W_down + b_down


def correctness_check(
    output: torch.Tensor, reference: torch.Tensor, name: str = "kernel"
) -> dict:
    """比较两个 tensor 的数值误差"""
    abs_diff = (output - reference).abs()
    max_err = abs_diff.max().item()
    mean_err = abs_diff.mean().item()
    rel_err = (abs_diff / (reference.abs() + 1e-8)).mean().item()

    print(f"[{name}] max_abs_err={max_err:.6e}  mean_abs_err={mean_err:.6e}  mean_rel_err={rel_err:.6e}")

    return {
        "name": name,
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "mean_rel_error": rel_err,
    }
