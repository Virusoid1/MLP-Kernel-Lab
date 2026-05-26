"""
GPU 检测与最优参数选择

根据当前 GPU 的 compute capability 自动选择最优配置：
- RTX 5070 Ti (Blackwell, SM 12.0): 40 SMs, 16GB GDDR7
- RTX 3070 Laptop (Ampere, SM 8.6): 40 SMs, 8GB GDDR6

影响 Triton kernel 性能的关键硬件参数：
- Shared memory 容量：决定可用的 tile 大小
- L2 cache 容量：影响 GROUP_SIZE_M 选择
- Tensor core 代数：影响 TF32/FP16 加速
- 内存带宽：影响 memory-bound kernel 的 BLOCK_SIZE 选择
"""

import torch


def get_gpu_info() -> dict:
    """返回当前 GPU 的关键参数。"""
    if not torch.cuda.is_available():
        return {"name": "CPU", "cc": (0, 0), "sms": 0, "memory_gb": 0}

    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "cc": (props.major, props.minor),
        "sms": props.multi_processor_count,
        "memory_gb": round(props.total_memory / 1e9, 1),
    }


def get_gpu_arch() -> str:
    """返回简化的架构标识。"""
    cc = get_gpu_info()["cc"]
    if cc >= (12, 0):
        return "blackwell"
    elif cc >= (8, 9):
        return "ada_loom"
    elif cc >= (8, 6):
        return "ampere"
    elif cc >= (8, 0):
        return "ampere_80"
    elif cc >= (7, 5):
        return "turing"
    else:
        return "volta_or_older"


def supports_tf32() -> bool:
    """当前 GPU 是否支持 TF32 tensor core 运算（SM 8.0+）。"""
    return get_gpu_info()["cc"] >= (8, 0)


def get_cuda_arch_flags() -> list[str]:
    """返回当前 GPU 的 nvcc -arch 编译标志。"""
    cc = get_gpu_info()["cc"]
    if cc == (0, 0):
        return []
    return [f"-arch=sm_{cc[0]}{cc[1]}"]
