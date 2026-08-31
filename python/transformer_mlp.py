"""SwiGLU MLP block 统一执行层（v2 主线）。

标准 Transformer FFN block:

    gate   = X @ W_gate          (M, K) @ (K, F)
    up     = X @ W_up            (M, K) @ (K, F)
    hidden = SiLU(gate) * up     elementwise, (M, F)
    Y      = hidden @ W_down     (M, F) @ (F, K)

支持的 backend（尽量复用仓库现有算子，避免重写 kernel）:
    eager   : PyTorch eager（两个独立 Linear + SiLU + mul + Linear）
    concat  : PyTorch eager，拼接后单次 gate/up GEMM（区分"算法重写"收益）
    compile : torch.compile 包裹 eager（Inductor）
    triton  : Triton matmul + swiglu_triton + Triton matmul
    cuda    : mlp_cuda (matmul_tiled_auto + swiglu_fused + matmul_tiled_auto)
    cutile  : cuTile (cutile_matmul + swiglu_cutile + cutile_matmul)

shape 配置见 bench/suites/transformer_mlp.yaml（decode/prefill/train 三档）。

设计要点：
- 每个 backend 产出 (Y, meta)，meta 记录后端名/是否走 TensorCore/是否正确性已验。
- correctness: 以 eager FP32 为 reference，逐 backend 对比 Y 的 max_abs/max_rel/norm_l2。
- 本模块不直接做计时（计时交给 bench/run.py 用 CUDA Event + warmup），只负责算。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# PyTorch baselines
# ---------------------------------------------------------------------------

def swiglu_block_eager(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,
                       w_down: torch.Tensor) -> torch.Tensor:
    """PyTorch eager：两个独立 Linear + SiLU * up + Linear。"""
    gate = x @ w_gate
    up = x @ w_up
    hidden = silu(gate) * up
    return hidden @ w_down


def swiglu_block_concat(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,
                        w_down: torch.Tensor) -> torch.Tensor:
    """PyTorch eager：先拼 W=[W_gate; W_up]，单次大 GEMM，再按列切分。

    与 eager 的差别只在"如何组织 GEMM"——用于分离 fusion 收益与算法重写收益。
    """
    w_gate_up = torch.cat([w_gate, w_up], dim=-1)          # (K, 2F)
    gate_up = x @ w_gate_up                                 # (M, 2F)
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = silu(gate) * up
    return hidden @ w_down


# ---------------------------------------------------------------------------
# 自定义后端（复用仓库现有算子）
# ---------------------------------------------------------------------------

def swiglu_block_triton(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,
                        w_down: torch.Tensor) -> torch.Tensor:
    """Triton：matmul -> swiglu_triton -> matmul。"""
    from triton_kernels.matmul import tiled_matmul
    from triton_kernels.swiglu_triton import swiglu_triton
    gate = tiled_matmul(x, w_gate)
    up = tiled_matmul(x, w_up)
    hidden = swiglu_triton(gate, up)
    return tiled_matmul(hidden, w_down)


def swiglu_block_triton_fused(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,
                                 w_down: torch.Tensor) -> torch.Tensor:
    """Triton 融合 gate+up：单 launch 完成 gate/up GEMM + SiLU×up epilogue，down 另用 matmul。

    decode 小 M 时 3 个 kernel launch 是主要开销；融合后 1 次 launch 出 hidden。
    """
    from triton_kernels.fused_swiglu_gateup import fused_gateup_swiglu
    from triton_kernels.matmul import tiled_matmul
    w_gate_up = torch.cat([w_gate, w_up], dim=-1)
    hidden = fused_gateup_swiglu(x, w_gate_up)
    return tiled_matmul(hidden, w_down)


def swiglu_block_cuda(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,
                      w_down: torch.Tensor) -> torch.Tensor:
    """CUDA：matmul_tiled_auto（fp32）或 matmul_half（fp16 TensorCore）→ swiglu_fused → matmul。

    fp16 输入走 WMMA TensorCore（matmul_half, fp16 in→fp32 acc→fp16 out, L2 2e-4）；
    fp32 输入走 strict FP32 tiled+Kahan（精度优先，注释见 matmul.cu）。
    """
    import mlp_cuda
    if x.dtype == torch.float16:
        mm = mlp_cuda.matmul_half
        gate = mm(x, w_gate)
        up = mm(x, w_up)
        hidden = torch.nn.functional.silu(gate) * up  # F.silu 支持 fp16（mlp_cuda.silu 仅 fp32）
        return mm(hidden, w_down)
    gate = mlp_cuda.matmul_tiled_auto(x, w_gate)
    up = mlp_cuda.matmul_tiled_auto(x, w_up)
    hidden = mlp_cuda.swiglu_fused(gate, up)
    return mlp_cuda.matmul_tiled_auto(hidden, w_down)


def swiglu_block_cutile(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,
                        w_down: torch.Tensor) -> torch.Tensor:
    """cuTile：cutile_matmul -> silu(gate)*up（复用 cuTile silu 语义）-> cutile_matmul。

    注：仓库 cutile swiglu_cutile 只是单参数 silu(x)；SwiGLU 双参数 (gate, up) 语义
    在此用 Python 层 silu(gate)*up 表达（gate/up 均已来自 cuTile matmul）。
    """
    from cutile_kernels.matmul import cutile_matmul
    from cutile_kernels.swiglu_cutile import swiglu_cutile
    in_dtype = x.dtype
    # cutile_matmul 恒输出 fp32：中间量转回输入 dtype，保证下游 ct.mma 输入同名
    gate = cutile_matmul(x, w_gate).to(in_dtype)
    up = cutile_matmul(x, w_up).to(in_dtype)
    hidden = swiglu_cutile(gate) * up
    return cutile_matmul(hidden, w_down).to(in_dtype)


def swiglu_block_compile(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,
                         w_down: torch.Tensor) -> torch.Tensor:
    """torch.compile(Inductor) 包裹 eager。"""
    return torch.compile(swiglu_block_eager)(x, w_gate, w_up, w_down)


BACKENDS = {
    "eager": swiglu_block_eager,
    "concat": swiglu_block_concat,
    "compile": swiglu_block_compile,
    "triton": swiglu_block_triton,
    "triton_fused": swiglu_block_triton_fused,
    "cuda": swiglu_block_cuda,
    "cutile": swiglu_block_cutile,
}


def available_backends() -> list[str]:
    """返回当前环境可用的 backend（依 import 探测）。"""
    out = ["eager", "concat", "compile"]
    try:
        from triton_kernels.matmul import tiled_matmul  # noqa: F401
        from triton_kernels.swiglu_triton import swiglu_triton  # noqa: F401
        out.append("triton")
        from triton_kernels.fused_swiglu_gateup import fused_gateup_swiglu  # noqa: F401
        out.append("triton_fused")
    except Exception:
        pass
    try:
        import mlp_cuda  # noqa: F401
        out.append("cuda")
    except Exception:
        pass
    try:
        import cuda.tile  # noqa: F401
        from cutile_kernels.matmul import cutile_matmul  # noqa: F401
        out.append("cutile")
    except Exception:
        pass
    return out
