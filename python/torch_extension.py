"""
PyTorch CUDA Extension 调用入口

用法:
    python setup.py install
    python -c "from python.torch_extension import *; ..."
"""

# CUDA extension 安装后才能 import
# 安装: python setup.py install
try:
    import mlp_cuda
    CUDA_EXT_AVAILABLE = True
except ImportError:
    CUDA_EXT_AVAILABLE = False
    print("Warning: mlp_cuda extension not installed. Run: python setup.py install")


def matmul_naive(A, B):
    """调用 CUDA naive matmul"""
    if not CUDA_EXT_AVAILABLE:
        raise RuntimeError("CUDA extension not available")
    return mlp_cuda.matmul_naive(A, B)


def matmul_tiled(A, B, BLOCK_M=16, BLOCK_N=16, BLOCK_K=16):
    """调用 CUDA tiled matmul"""
    if not CUDA_EXT_AVAILABLE:
        raise RuntimeError("CUDA extension not available")
    return mlp_cuda.matmul_tiled(A, B, BLOCK_M, BLOCK_N, BLOCK_K)


def mlp_fused_first_layer(X, W1, bias):
    """调用 CUDA fused matmul+bias+GELU"""
    if not CUDA_EXT_AVAILABLE:
        raise RuntimeError("CUDA extension not available")
    return mlp_cuda.mlp_fused_first_layer(X, W1, bias)


def swiglu_fused(gate, up):
    """调用 CUDA fused SwiGLU"""
    if not CUDA_EXT_AVAILABLE:
        raise RuntimeError("CUDA extension not available")
    return mlp_cuda.swiglu_fused(gate, up)
