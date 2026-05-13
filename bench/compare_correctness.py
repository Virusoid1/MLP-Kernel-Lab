"""
Correctness 检查: 比较各实现与 PyTorch reference 的数值误差

用法:
    python bench/compare_correctness.py
    python bench/compare_correctness.py --all
"""

import argparse
import torch

from python.mlp_reference import (
    matmul_ref, mlp_first_layer_ref, correctness_check
)


def test_matmul_correctness(M=512, K=768, N=3072, dtype=torch.float32, atol=1e-3):
    """测试 matmul 实现与 PyTorch 的一致性"""
    device = 'cuda'
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)

    ref = matmul_ref(A, B)
    results = []

    # PyTorch baseline (self-check)
    results.append(correctness_check(ref, ref, "torch_matmul"))

    # TODO: 取消注释以下行来测试你的实现
    # from python.torch_extension import matmul_naive, matmul_tiled
    # cuda_naive = matmul_naive(A.float(), B.float()).to(dtype)
    # results.append(correctness_check(cuda_naive, ref, "cuda_matmul_naive"))
    #
    # cuda_tiled = matmul_tiled(A.float(), B.float())
    # results.append(correctness_check(cuda_tiled, ref, "cuda_matmul_tiled"))
    #
    # from triton_kernels.matmul_triton import matmul_triton
    # triton_out = matmul_triton(A, B)
    # results.append(correctness_check(triton_out, ref, "triton_matmul"))

    return results


def test_mlp_correctness(M=512, K=768, N=3072, dtype=torch.float32):
    """测试 MLP first layer 与 PyTorch 的一致性"""
    device = 'cuda'
    X = torch.randn(M, K, device=device, dtype=dtype)
    W1 = torch.randn(K, N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)

    ref = mlp_first_layer_ref(X, W1, bias)
    results = []
    results.append(correctness_check(ref, ref, "torch_mlp"))

    # TODO: 取消注释以下行来测试你的实现
    # from python.torch_extension import mlp_fused_first_layer
    # cuda_fused = mlp_fused_first_layer(X.float(), W1.float(), bias.float()).to(dtype)
    # results.append(correctness_check(cuda_fused, ref, "cuda_mlp_fused"))
    #
    # from triton_kernels.mlp_triton import mlp_first_layer_triton
    # triton_out = mlp_first_layer_triton(X, W1, bias)
    # results.append(correctness_check(triton_out, ref, "triton_mlp"))

    return results


def test_swiglu_correctness(M=512, N=4096, dtype=torch.float32):
    """测试 SwiGLU 与 PyTorch 的一致性"""
    device = 'cuda'
    gate = torch.randn(M, N, device=device, dtype=dtype)
    up = torch.randn(M, N, device=device, dtype=dtype)

    ref = torch.nn.functional.silu(gate) * up
    results = []
    results.append(correctness_check(ref, ref, "torch_swiglu"))

    # TODO: 取消注释以下行来测试你的实现
    # from python.torch_extension import swiglu_fused
    # cuda_out = swiglu_fused(gate.float(), up.float()).to(dtype)
    # results.append(correctness_check(cuda_out, ref, "cuda_swiglu"))
    #
    # from triton_kernels.swiglu_triton import swiglu_triton
    # triton_out = swiglu_triton(gate, up)
    # results.append(correctness_check(triton_out, ref, "triton_swiglu"))

    return results


def main():
    parser = argparse.ArgumentParser(description='Correctness Check for MLP Kernels')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--matmul', action='store_true', help='Test matmul only')
    parser.add_argument('--mlp', action='store_true', help='Test MLP first layer')
    parser.add_argument('--swiglu', action='store_true', help='Test SwiGLU')
    args = parser.parse_args()

    if not any([args.all, args.matmul, args.mlp, args.swiglu]):
        args.all = True

    print("=" * 60)
    print("Correctness Check")
    print("=" * 60)

    all_results = []

    if args.all or args.matmul:
        print("\n--- Matmul ---")
        all_results.extend(test_matmul_correctness())

    if args.all or args.mlp:
        print("\n--- MLP First Layer ---")
        all_results.extend(test_mlp_correctness())

    if args.all or args.swiglu:
        print("\n--- SwiGLU ---")
        all_results.extend(test_swiglu_correctness())

    # 汇总
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    fmt = "{:<20} {:<14} {:<14} {:<14}"
    print(fmt.format("kernel", "max_abs_err", "mean_abs_err", "mean_rel_err"))
    print("-" * 62)
    for r in all_results:
        print(fmt.format(
            r['name'],
            f"{r['max_abs_error']:.6e}",
            f"{r['mean_abs_error']:.6e}",
            f"{r['mean_rel_error']:.6e}",
        ))


if __name__ == '__main__':
    main()
