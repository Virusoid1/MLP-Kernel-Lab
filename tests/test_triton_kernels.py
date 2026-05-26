"""
Triton kernel 正确性测试

测试所有 Triton kernel 的前向/反向输出与 PyTorch 参考实现的对齐程度。
运行: pytest tests/test_triton_kernels.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(42)


# ============================================================
# matmul
# ============================================================

class TestMatmul:
    def _check(self, M, K, N, rtol=1e-2, atol=1e-2):
        from triton_kernels.matmul import tiled_matmul
        A = torch.randn(M, K, device="cuda")
        B = torch.randn(K, N, device="cuda")
        ref = torch.matmul(A, B)
        out = tiled_matmul(A, B)
        assert torch.allclose(out, ref, rtol=rtol, atol=atol), \
            f"matmul ({M},{K})@({K},{N}) L2={torch.norm(out - ref).item():.4f}"

    def test_small(self):
        self._check(32, 64, 32)

    def test_medium(self):
        self._check(128, 256, 128, rtol=1e-1, atol=1e-1)

    def test_large(self):
        self._check(512, 512, 512, rtol=1e-1, atol=1e-1)


# ============================================================
# elementwise
# ============================================================

class TestElementwise:
    @pytest.mark.parametrize("fn_name,torch_fn", [
        ("relu", torch.relu),
        ("gelu", lambda x: torch.nn.functional.gelu(x, approximate="tanh")),
        ("silu", torch.nn.functional.silu),
    ])
    def test_forward(self, fn_name, torch_fn):
        import triton_kernels.elementwise as ew
        fn = getattr(ew, fn_name)
        x = torch.randn(1024, device="cuda")
        ref = torch_fn(x)
        out = fn(x)
        assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)

    def test_bias_add(self):
        from triton_kernels.elementwise import bias_add
        x = torch.randn(64, 128, device="cuda")
        b = torch.randn(128, device="cuda")
        ref = x + b
        out = bias_add(x, b)
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)

    def test_bias_add_relu(self):
        from triton_kernels.elementwise import bias_add_relu
        x = torch.randn(64, 128, device="cuda")
        b = torch.randn(128, device="cuda")
        ref = torch.relu(x + b)
        out = bias_add_relu(x, b)
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)


# ============================================================
# backward: activation backward
# ============================================================

class TestActivationBackward:
    @pytest.mark.parametrize("act_name", ["relu", "silu"])
    def test_backward(self, act_name):
        import triton_kernels.backward as bw
        fn = getattr(bw, f"{act_name}_backward")
        x = torch.randn(1024, device="cuda", requires_grad=True)
        torch_fn = getattr(torch.nn.functional, act_name)
        ref = torch_fn(x)
        dy = torch.randn_like(ref)
        ref.backward(dy)
        ref_grad = x.grad.clone()
        x.grad = None
        out = fn(dy, x)
        assert torch.allclose(out, ref_grad, rtol=1e-3, atol=1e-3)

    def test_gelu_backward(self):
        import triton_kernels.backward as bw
        x = torch.randn(1024, device="cuda", requires_grad=True)
        ref = torch.nn.functional.gelu(x, approximate="tanh")
        dy = torch.randn_like(ref)
        ref.backward(dy)
        ref_grad = x.grad.clone()
        x.grad = None
        out = bw.gelu_backward(dy, x)
        assert torch.allclose(out, ref_grad, rtol=1e-3, atol=1e-3)


# ============================================================
# backward: matmul backward
# ============================================================

class TestMatmulBackward:
    def test_backward_a(self):
        from triton_kernels.backward import matmul_backward_a
        A = torch.randn(64, 128, device="cuda", requires_grad=True)
        B = torch.randn(128, 64, device="cuda")
        C = torch.matmul(A, B)
        dC = torch.randn_like(C)
        C.backward(dC)
        ref = A.grad.clone()
        out = matmul_backward_a(dC, B)
        assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2)

    def test_backward_b(self):
        from triton_kernels.backward import matmul_backward_b
        A = torch.randn(64, 128, device="cuda", requires_grad=False)
        B = torch.randn(128, 64, device="cuda", requires_grad=True)
        C = torch.matmul(A, B)
        dC = torch.randn_like(C)
        C.backward(dC)
        ref = B.grad.clone()
        out = matmul_backward_b(A, dC)
        assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2)


# ============================================================
# layernorm
# ============================================================

class TestLayerNorm:
    def test_forward(self):
        from triton_kernels.layernorm import layernorm_forward
        x = torch.randn(8, 256, device="cuda")
        gamma = torch.ones(256, device="cuda")
        beta = torch.zeros(256, device="cuda")
        y, mean, rstd = layernorm_forward(x, gamma, beta)
        ref = torch.nn.functional.layer_norm(x, [256], gamma, beta)
        assert torch.allclose(y, ref, rtol=1e-4, atol=1e-4)

    def test_backward(self):
        from triton_kernels.layernorm import layernorm_forward, layernorm_backward
        x = torch.randn(8, 256, device="cuda")
        gamma = torch.ones(256, device="cuda", requires_grad=False)
        beta = torch.zeros(256, device="cuda")
        y, mean, rstd = layernorm_forward(x, gamma, beta)
        dy = torch.randn_like(y)
        dx, dg, db = layernorm_backward(dy, x, gamma, mean, rstd)

        # PyTorch 参考
        x_ref = x.clone().requires_grad_(True)
        g_ref = gamma.clone().requires_grad_(True)
        b_ref = beta.clone().requires_grad_(True)
        ref_y = torch.nn.functional.layer_norm(x_ref, [256], g_ref, b_ref)
        ref_y.backward(dy)
        assert torch.allclose(dx, x_ref.grad, rtol=1e-3, atol=1e-3)
        assert torch.allclose(dg, g_ref.grad, rtol=1e-3, atol=1e-3)
        assert torch.allclose(db, b_ref.grad, rtol=1e-5, atol=1e-5)

    def test_learnable_affine(self):
        from triton_kernels.layernorm import layernorm_forward
        x = torch.randn(4, 64, device="cuda")
        gamma = torch.randn(64, device="cuda") * 0.5 + 1.0
        beta = torch.randn(64, device="cuda") * 0.1
        y, _, _ = layernorm_forward(x, gamma, beta)
        ref = torch.nn.functional.layer_norm(x, [64], gamma, beta)
        assert torch.allclose(y, ref, rtol=1e-4, atol=1e-4)


# ============================================================
# dropout
# ============================================================

class TestDropout:
    def test_shape_and_range(self):
        from triton_kernels.dropout import triton_dropout
        x = torch.randn(64, 128, device="cuda")
        out = triton_dropout(x, p=0.5)
        assert out.shape == x.shape
        # 约 50% 元素为 0
        zero_frac = (out == 0).float().mean().item()
        assert 0.2 < zero_frac < 0.8

    def test_expectation_preserved(self):
        from triton_kernels.dropout import triton_dropout
        x = torch.ones(256, 512, device="cuda")
        # 多次采样取均值，期望应接近 1.0
        acc = torch.zeros_like(x)
        for _ in range(20):
            acc += triton_dropout(x, p=0.3)
        mean_val = acc.mean().item() / 20
        assert abs(mean_val - 1.0) < 0.1


# ============================================================
# loss
# ============================================================

class TestLoss:
    def test_cross_entropy(self):
        from triton_kernels.loss import cross_entropy
        logits = torch.randn(16, 10, device="cuda")
        targets = torch.randint(0, 10, (16,), device="cuda")
        out = cross_entropy(logits, targets)
        ref = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2)


# ============================================================
# swiglu
# ============================================================

class TestSwiGLU:
    def test_forward(self):
        from triton_kernels.swiglu_triton import swiglu_triton
        gate = torch.randn(64, 128, device="cuda")
        up = torch.randn(64, 128, device="cuda")
        out = swiglu_triton(gate, up)
        ref = torch.nn.functional.silu(gate) * up
        assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)


# ============================================================
# fused mlp first layer
# ============================================================

class TestFusedMLP:
    def test_forward(self):
        from triton_kernels.mlp_triton import mlp_first_layer_triton
        x = torch.randn(32, 128, device="cuda")
        w = torch.randn(128, 64, device="cuda")
        b = torch.randn(64, device="cuda")
        out = mlp_first_layer_triton(x, w, b)
        ref = torch.nn.functional.gelu(torch.matmul(x, w) + b, approximate="tanh")
        assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2)
