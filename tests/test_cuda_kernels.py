"""
CUDA kernel 正确性测试

测试所有 CUDA kernel 的前向/反向输出与 PyTorch 参考实现的对齐程度。
运行: pytest tests/test_cuda_kernels.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import mlp_cuda
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False


@pytest.fixture(autouse=True)
def _setup():
    if not _HAS_CUDA:
        pytest.skip("mlp_cuda not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(42)


# ============================================================
# matmul
# ============================================================

class TestMatmul:
    def test_tiled_auto(self):
        A = torch.randn(64, 128, device="cuda")
        B = torch.randn(128, 64, device="cuda")
        ref = torch.matmul(A, B)
        out = mlp_cuda.matmul_tiled_auto(A, B)
        assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2)

    def test_transB(self):
        A = torch.randn(32, 64, device="cuda")
        B = torch.randn(16, 64, device="cuda")
        ref = torch.matmul(A, B.t())
        out = mlp_cuda.matmul_transB(A, B)
        assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2)

    def test_transA(self):
        A = torch.randn(64, 32, device="cuda")
        B = torch.randn(64, 16, device="cuda")
        ref = torch.matmul(A.t(), B)
        out = mlp_cuda.matmul_transA(A, B)
        assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2)


# ============================================================
# activation
# ============================================================

class TestActivation:
    @pytest.mark.parametrize("name,torch_fn", [
        ("relu", torch.relu),
        ("gelu", lambda x: torch.nn.functional.gelu(x, approximate="tanh")),
        ("silu", torch.nn.functional.silu),
    ])
    def test_forward(self, name, torch_fn):
        x = torch.randn(1024, device="cuda")
        cuda_fn = getattr(mlp_cuda, name)
        ref = torch_fn(x)
        out = cuda_fn(x)
        assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("name", ["relu", "silu"])
    def test_backward_vec4(self, name):
        x = torch.randn(1024, device="cuda", requires_grad=True)
        torch_fn = getattr(torch.nn.functional, name)
        ref = torch_fn(x)
        dy = torch.randn_like(ref)
        ref.backward(dy)
        ref_grad = x.grad.clone()
        x.grad = None

        cuda_fn = getattr(mlp_cuda, f"{name}_backward_vec4")
        out = cuda_fn(dy, x)
        assert torch.allclose(out, ref_grad, rtol=1e-3, atol=1e-3)

    def test_gelu_backward_vec4(self):
        x = torch.randn(1024, device="cuda", requires_grad=True)
        ref = torch.nn.functional.gelu(x, approximate="tanh")
        dy = torch.randn_like(ref)
        ref.backward(dy)
        ref_grad = x.grad.clone()
        x.grad = None

        out = mlp_cuda.gelu_backward_vec4(dy, x)
        assert torch.allclose(out, ref_grad, rtol=1e-3, atol=1e-3)


# ============================================================
# bias_add
# ============================================================

class TestBiasAdd:
    def test_forward(self):
        x = torch.randn(64, 128, device="cuda")
        b = torch.randn(128, device="cuda")
        ref = x + b
        out = mlp_cuda.bias_add(x, b)
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)


# ============================================================
# layernorm
# ============================================================

class TestLayerNorm:
    def test_forward(self):
        x = torch.randn(8, 256, device="cuda")
        gamma = torch.ones(256, device="cuda")
        beta = torch.zeros(256, device="cuda")
        y, mean, rstd = mlp_cuda.layernorm_forward(x, gamma, beta, 1e-5)
        ref = torch.nn.functional.layer_norm(x, [256], gamma, beta)
        assert torch.allclose(y, ref, rtol=1e-4, atol=1e-4)

    def test_backward(self):
        x = torch.randn(8, 256, device="cuda")
        gamma = torch.ones(256, device="cuda")
        beta = torch.zeros(256, device="cuda")
        y, mean, rstd = mlp_cuda.layernorm_forward(x, gamma, beta, 1e-5)
        dy = torch.randn_like(y)
        dx, dg, db = mlp_cuda.layernorm_backward(dy, x, gamma, mean, rstd)

        x_ref = x.clone().requires_grad_(True)
        g_ref = gamma.clone().requires_grad_(True)
        b_ref = beta.clone().requires_grad_(True)
        ref_y = torch.nn.functional.layer_norm(x_ref, [256], g_ref, b_ref)
        ref_y.backward(dy)

        assert torch.allclose(dx, x_ref.grad, rtol=1e-3, atol=1e-3)
        assert torch.allclose(dg, g_ref.grad, rtol=1e-3, atol=1e-3)
        assert torch.allclose(db, b_ref.grad, rtol=1e-5, atol=1e-5)

    def test_learnable_affine(self):
        x = torch.randn(4, 64, device="cuda")
        gamma = torch.randn(64, device="cuda") * 0.5 + 1.0
        beta = torch.randn(64, device="cuda") * 0.1
        y, _, _ = mlp_cuda.layernorm_forward(x, gamma, beta, 1e-5)
        ref = torch.nn.functional.layer_norm(x, [64], gamma, beta)
        assert torch.allclose(y, ref, rtol=1e-4, atol=1e-4)

    def test_large_dims(self):
        x = torch.randn(32, 1024, device="cuda")
        gamma = torch.ones(1024, device="cuda")
        beta = torch.zeros(1024, device="cuda")
        y, mean, rstd = mlp_cuda.layernorm_forward(x, gamma, beta, 1e-5)
        ref = torch.nn.functional.layer_norm(x, [1024], gamma, beta)
        assert torch.allclose(y, ref, rtol=1e-4, atol=1e-4)


# ============================================================
# fused kernels
# ============================================================

class TestFused:
    def test_mlp_first_layer(self):
        x = torch.randn(32, 128, device="cuda")
        w = torch.randn(128, 64, device="cuda")
        b = torch.randn(64, device="cuda")
        out = mlp_cuda.mlp_fused_first_layer(x, w, b)
        ref = torch.nn.functional.gelu(torch.matmul(x, w) + b, approximate="tanh")
        assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3)

    def test_swiglu(self):
        x = torch.randn(64, 128, device="cuda")
        out = mlp_cuda.swiglu_fused(x, x)
        ref = torch.nn.functional.silu(x) * x
        assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)


# ============================================================
# conv2d / pool fp16 / bf16（v2 4.x：全算子 dtype 完备）
# ============================================================

class TestConvPoolDtypes:
    DTYPES = [torch.float32, torch.float16, torch.bfloat16]

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_maxpool2d(self, dtype):
        x = torch.randn(2, 4, 8, 8, device="cuda", dtype=dtype)
        out = mlp_cuda.maxpool2d(x, 2, 2, 0)
        ref = torch.nn.functional.max_pool2d(x.float(), 2, 2, 0)
        l2 = ((out.float() - ref).pow(2).sum() / ref.pow(2).sum()).sqrt().item()
        assert l2 < 5e-2, f"{dtype} maxpool l2={l2:.3e}"

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_avgpool2d(self, dtype):
        x = torch.randn(2, 4, 8, 8, device="cuda", dtype=dtype)
        out = mlp_cuda.avgpool2d(x, 2, 2, 0)
        ref = torch.nn.functional.avg_pool2d(x.float(), 2, 2, 0)
        l2 = ((out.float() - ref).pow(2).sum() / ref.pow(2).sum()).sqrt().item()
        assert l2 < 5e-2, f"{dtype} avgpool l2={l2:.3e}"

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_conv2d(self, dtype):
        x = torch.randn(1, 3, 8, 8, device="cuda", dtype=dtype)
        w = torch.randn(6, 3, 3, 3, device="cuda", dtype=dtype)
        b = torch.randn(6, device="cuda", dtype=dtype)
        out = mlp_cuda.conv2d(x, w, b, 1, 0)
        ref = torch.nn.functional.conv2d(x.float(), w.float(), b.float(), stride=1, padding=0)
        l2 = ((out.float() - ref).pow(2).sum() / ref.pow(2).sum()).sqrt().item()
        assert l2 < 5e-2, f"{dtype} conv2d l2={l2:.3e}"

