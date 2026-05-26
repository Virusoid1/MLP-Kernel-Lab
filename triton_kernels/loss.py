"""
Triton 融合 CrossEntropy kernel

融合实现：单 kernel 完成 softmax + log + gather。
数值稳定：减去行最大值防止 exp 溢出。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr, targets_ptr, losses_ptr,
    n_cols,
    logits_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合交叉熵 kernel。每个 program 处理一个样本。
    loss = log_sum_exp - logits[target]
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(logits_ptr + row_idx * logits_row_stride + col_offsets,
                  mask=mask, other=-float('inf'))

    row_max = tl.max(row, axis=0)
    exp_row = tl.exp(row - row_max)
    log_sum_exp = row_max + tl.log(tl.sum(exp_row, axis=0))

    target = tl.load(targets_ptr + row_idx)
    target_logit = tl.sum(tl.where(col_offsets == target, row, 0.0), axis=0)

    tl.store(losses_ptr + row_idx, log_sum_exp - target_logit)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Triton 融合交叉熵。
    logits: (M, N)  targets: (M,)  ->  (M,) 每个样本的交叉熵损失
    """
    assert logits.dim() == 2
    assert targets.dim() == 1
    assert logits.shape[0] == targets.shape[0]
    M, N = logits.shape

    losses = torch.empty(M, device=logits.device, dtype=torch.float32)
    BLOCK_SIZE = triton.next_power_of_2(N)

    cross_entropy_kernel[(M,)](
        logits, targets, losses,
        N, logits.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return losses
