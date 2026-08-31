"""Fused gate+up GEMM with SiLU×up epilogue — decode 小 M 专用（v2 signature optimization）。

背景（bench 数据）:
  decode (M<=32) 时 triton-fp16 只有 0.5x —— 3 个独立 kernel launch（gate, up, silu×up）
  的 launch/autotune 开销压倒 TensorCore 收益。
  本 kernel 把 gate GEMM + up GEMM + SiLU(gate)*up 融合为一次 launch:

        gate = X @ W_gate ; up = X @ W_up ; hidden = SiLU(gate) * up

  做法: 每个 program 处理 (BLOCK_M 行, BLOCK_N 输出的 F 列区间)。
  - 一次性加载 W_gate[:, n] 与 W_up[:, n]（同列区间）
  - 两个 tl.dot 分别算 gate/up 块（fp32 累加器）
  - epilogue: SiLU(gate)*up（sigmoid 升 fp32 计算规避 Triton math 限制）
  - 写 hidden = (M,F)，无需 gate/up 中间张量

输入:
  x: (M, K) ; w_gate_up: (K, 2F) = concat([W_gate, W_up], dim=-1)
输出: hidden: (M, F)
"""

import torch
import triton
import triton.language as tl

from triton_kernels.precision import precision


_FUSED_CONFIGS = [
    # decode 专用：小 BLOCK_N 提升并行度 + 低 shared memory
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=_FUSED_CONFIGS, key=["M", "N", "K"])
@triton.jit
def fused_gateup_swiglu_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr, OUT_DTYPE: tl.constexpr,
):
    """X (M,K) @ W_gate_up (K,2F) -> hidden (M,F) = SiLU(gate)*up。

    W 布局: w[:, 0:F] = W_gate, w[:, F:2F] = W_up；N 表示 F（半宽）。
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # hidden 列 (0..F)
    offs_k = tl.arange(0, BLOCK_K)

    # a: (BLOCK_M, BLOCK_K), wg/wu: (BLOCK_K, BLOCK_N)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N
    k_mask_row = offs_k[None, :] < K         # (1, BLOCK_K) 给 a
    k_mask_col = offs_k[:, None] < K         # (BLOCK_K, 1) 给 w

    # gate 列 = offs_n (同 hidden 列), up 列 = F + offs_n
    offs_n_up = offs_n + N

    a_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    wg_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    wu_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n_up[None, :] * stride_wn

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        # mask 需叠加当前 k_start（ptr 每次前进 BLOCK_K）
        a = tl.load(a_ptrs, mask=m_mask & ((k_start + offs_k)[None, :] < K), other=0.0)
        wg = tl.load(wg_ptrs, mask=((k_start + offs_k)[:, None] < K) & n_mask, other=0.0)
        wu = tl.load(wu_ptrs, mask=((k_start + offs_k)[:, None] < K) & n_mask, other=0.0)
        if a.dtype == tl.float32:
            acc_g += tl.dot(a, wg, allow_tf32=ALLOW_TF32)
            acc_u += tl.dot(a, wu, allow_tf32=ALLOW_TF32)
        else:
            acc_g = tl.dot(a, wg, acc_g, input_precision="ieee" if not ALLOW_TF32 else "tf32")
            acc_u = tl.dot(a, wu, acc_u, input_precision="ieee" if not ALLOW_TF32 else "tf32")
        a_ptrs += BLOCK_K * stride_xk
        wg_ptrs += BLOCK_K * stride_wk
        wu_ptrs += BLOCK_K * stride_wk

    # epilogue: hidden = SiLU(gate) * up（sigmoid 升 fp32 规避 tl.sigmoid 的 fp16 限制）
    gate_f32 = acc_g.to(tl.float32)
    sigmoid_g = tl.sigmoid(gate_f32)
    silu_g = gate_f32 * sigmoid_g
    hidden = silu_g * acc_u
    hidden = hidden.to(OUT_DTYPE)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, hidden, mask=m_mask & n_mask)


def fused_gateup_swiglu(x: torch.Tensor, w_gate_up: torch.Tensor) -> torch.Tensor:
    """fused gate+up GEMM + SiLU epilogue。

    x: (M, K); w_gate_up: (K, 2F)（concat 后的权重）
    返回 hidden: (M, F)
    """
    M, K = x.shape
    _, N2 = w_gate_up.shape
    N = N2 // 2
    hidden = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    fused_gateup_swiglu_kernel[grid](
        x, w_gate_up, hidden,
        M, N, K,
        x.stride(0), x.stride(1),
        w_gate_up.stride(0), w_gate_up.stride(1),
        hidden.stride(0), hidden.stride(1),
        ALLOW_TF32=precision.allow_tf32,
        OUT_DTYPE=tl.float16 if x.dtype == torch.float16 else (tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float32),
    )
    return hidden
