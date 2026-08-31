"""CUDA extension 构建配置（v2：动态架构探测，不再硬编码 sm_86/sm_120）。

架构来源优先级：
1. 环境变量 TORCH_CUDA_ARCH_LIST（如 "8.6" / "12.0" / "8.6;12.0" / "8.6+PTX"）— 显式指定，跨机器可移植；
2. 运行时 torch.cuda.get_device_capability() — 当前 GPU；
3. 兜底 Ampere "8.6"（主开发机 RTX 3070 Laptop）。
"""

import os
import platform

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _detect_arch_list() -> list[str]:
    """返回架构列表，元素形如 "8.6" 或 "8.6+PTX"。"""
    env = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if env:
        parts = [p.strip() for p in env.replace(";", ",").split(",") if p.strip()]
        archs = [p for p in parts if p.split("+")[0].strip()[:1].isdigit()]
        if archs:
            return archs
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return [f"{major}.{minor}"]
    return ["8.6"]


def _gencode_flags(arch_list: list[str]) -> list[str]:
    """把 ["8.6", "12.0+PTX"] 转成 nvcc -gencode 参数。"""
    flags: list[str] = []
    for entry in arch_list:
        name, _, ptx = entry.partition("+")
        cc = name.strip().replace(".", "")
        if not cc.isdigit():
            continue
        flags.append(f"-gencode=arch=compute_{cc},code=sm_{cc}")
        if ptx.strip().upper() == "PTX":
            flags.append(f"-gencode=arch=compute_{cc},code=compute_{cc}")
    if not flags:
        raise RuntimeError("无法从 TORCH_CUDA_ARCH_LIST 解析任何架构: %r" % (arch_list,))
    return flags


# 构建时暴露给外部工具（make check / preflight 可复用）
CUDA_ARCH_LIST = _detect_arch_list()

# 构建目录按 环境+架构 隔离: build/py<MAJOR><MINOR>-torch<X>-sm<CC>/
# 避免 Ampere/Blackwell 或不同 Python 的产物互相污染（v2 多机代码）。
_py = f"py{platform.python_version_tuple()[0]}{platform.python_version_tuple()[1]}"
_torch_ver = torch.__version__.split("+")[0]
_cc = (CUDA_ARCH_LIST[0] if CUDA_ARCH_LIST else "8.6").replace(".", "")
_build_dir = f"build/{_py}-torch{_torch_ver}-sm{_cc}"

setup(
    name='mlp_cuda',
    ext_modules=[
        CUDAExtension(
            name='mlp_cuda',
            sources=[
                'kernels/binding.cpp',
                # CUDA kernels (按算子族拆分自原 mlp_cuda_kernels.cu)
                'kernels/mlp/matmul.cu',
                'kernels/mlp/wmma.cu',
                'kernels/mlp/activation.cu',
                'kernels/mlp/fused.cu',
                'kernels/mlp/layernorm.cu',
                'kernels/mlp/softmax.cu',
                'kernels/mlp/pool_im2col.cu',
            ],
            include_dirs=['kernels/mlp'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    *_gencode_flags(CUDA_ARCH_LIST),
                ],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
    # 构建产物按 环境+架构 隔离（多机: build/py312-torch2.11.0-8.6 / ...-12.0）
    # build_base 会折叠 build/lib.* 与 build/temp.* 到该目录下
    options={
        'build': {'build_base': _build_dir},
    },
    python_requires='>=3.8',
)
