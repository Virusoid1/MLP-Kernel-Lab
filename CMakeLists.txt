# 可选: 纯 CUDA 独立编译 (不依赖 PyTorch extension)
# 用于快速迭代 kernel 开发和 nsight compute profiling
#
# 用法:
#   mkdir build && cd build
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
#   make

cmake_minimum_required(VERSION 3.18)
project(mlp-kernel-lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 设置 CUDA 架构 (根据你的 GPU 修改)
# T4: 75, A100: 80, RTX 3090: 86, RTX 4090: 89
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86;120)
endif()

# vector_add
add_executable(vector_add kernels/vector_add.cu)

# matmul_naive
add_executable(matmul_naive kernels/matmul_naive.cu)

# matmul_tiled
add_executable(matmul_tiled kernels/matmul_tiled.cu)
