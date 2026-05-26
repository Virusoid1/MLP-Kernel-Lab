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
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_120,code=sm_120',
                ],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
    python_requires='>=3.8',
)
