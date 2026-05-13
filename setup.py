from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mlp_cuda',
    ext_modules=[
        CUDAExtension(
            name='mlp_cuda',
            sources=[
                'kernels/binding.cpp',
                'kernels/mlp_cuda_kernels.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    # '-arch=sm_80',  # A100
                    # '-arch=sm_86',  # RTX 3090
                    # '-arch=sm_89',  # RTX 4090
                    # '-arch=sm_75',  # T4
                ],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
    python_requires='>=3.8',
)
