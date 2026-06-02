from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
