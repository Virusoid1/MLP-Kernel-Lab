# Nsight Profiling 使用指南

## 环境

- WSL2 Ubuntu, CUDA 13.2, RTX 3070 (sm_86)
- ncu: `/usr/local/cuda/bin/ncu`
- nsys: `/usr/local/cuda/bin/nsys`
- Windows GUI: Nsight Compute 2025.2.1 / Nsight Systems 2025.1.3

## 快速开始

### CUDA C++ Kernel

```bash
cd profiling

# 完整工作流（步骤 1-3）
./profile_workflow.sh all

# 或单独执行
./profile_workflow.sh step1    # nsys 时间线
./profile_workflow.sh step2    # ncu 快速分析
./profile_workflow.sh step3    # ncu 详细分析

# 优化后对比
./profile_workflow.sh step5    # 需要先有 before/after 报告
```

### PyTorch/Triton Kernel

```bash
# nsys 时间线
nsys profile --trace=cuda,nvtx -o reports/nsys_pytorch \
    python profile_pytorch.py --nsys

# ncu 详细分析
ncu --set full --launch-skip 10 --launch-count 1 \
    -o reports/ncu_pytorch \
    python profile_pytorch.py --ncu
```

## 查看报告

报告文件生成在 `profiling/reports/` 下。

### 方法 1：命令行查看

```bash
# nsys 报告
nsys stats --report cuda_gpu_kernsum reports/nsys_timeline.nsys-rep

# ncu 报告
ncu --import reports/ncu_full.ncu-rep --page details --section Occupancy --csv
```

### 方法 2：Windows GUI 查看

在 PowerShell 中：

```powershell
# Nsight Systems GUI
Start-Process "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\host-windows-x64\nsys-ui.exe" -ArgumentList "\\wsl$\Ubuntu\home\virusoid\projects\MLP-Kernel-Lab\profiling\reports\nsys_timeline.nsys-rep"

# Nsight Compute GUI
Start-Process "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.1\host-windows-x64\ncu-ui.exe" -ArgumentList "\\wsl$\Ubuntu\home\virusoid\projects\MLP-Kernel-Lab\profiling\reports\ncu_full.ncu-rep"
```

也可以直接在 Windows 资源管理器地址栏输入 `\\wsl$\Ubuntu\home\virusoid\projects\MLP-Kernel-Lab\profiling\reports\`，双击 `.nsys-rep` 或 `.ncu-rep` 文件。

## profiling 目录工具索引

| 脚本 | 用途 | 示例 |
|---|---|---|
| profile_workflow.sh | nsys/ncu 完整工作流（step1-5） | `./profile_workflow.sh all` |
| profile_pytorch.py | PyTorch/Triton 的 nsys/ncu 输出 | `python profile_pytorch.py --nsys` |
| profile_compare.py | before/after 对比 | `python profile_compare.py --before ... --after ...` |
| profile_ops.py | 算子级 profile 定位 | `python profile_ops.py` |
| bench_cutile.py | cuTile 算子 + MLP 端到端基准 | `python bench_cutile.py` |
| bench_cutile.py --tile-sweep | **cuTile matmul tile 单变量搜索（E4/Blackwell 选最优 tile）** | `python bench_cutile.py --tile-sweep --M 512 --K 4096 --N 11008` |

> 无 ncu/nsys 权限（WSL 无 sudo，ERR_NVGPUCTRPERM）时：用软件计时（CUDA Event）
> 代替硬件计数器——见 bench/run.py 的 protocol；热状态用 scripts/gpu_telemetry.py。

## 工作流详解

```
步骤 1: nsys → 时间线全局视图
    输出: kernel 执行顺序、持续时间、CPU/GPU 并行度
    关注: 哪个 kernel 占时最多？是否有不必要的同步？内存传输是否是瓶颈？

步骤 2: ncu --set quick → 瓶颈分类
    输出: kernel 是否受限于计算、带宽或 occupancy
    关注: "Compute Bound" vs "Memory Bound" vs "Latency Bound"

步骤 3: ncu --set full → 详细指标
    输出: 所有微架构指标
    关注: occupancy, L1/L2 cache hit rate, warp stall reasons, 指令吞吐

步骤 4: 优化代码（手动）

步骤 5: ncu 对比 → 验证效果
    对比优化前后的关键指标变化
```

## 关键指标速查

| 指标 | 理想值 | 含义 |
|------|--------|------|
| Occupancy | > 50% | SM 上活跃 warp 占比 |
| Achieved Bandwidth | > 70% peak | 实际内存带宽利用率 |
| L1/TEX Hit Rate | 越高越好 | L1 缓存命中率 |
| Warp Stall: Not Selected | < 20% | warp 调度等待 |
| SM Efficiency | > 80% | SM 利用率 |

## 常用 ncu metrics

```bash
# GPU 执行时间
--metrics gpu__time_duration.sum

# Occupancy
--metrics sm__warps_active.avg.pct_of_peak

# 全局内存读取吞吐
--metrics lts__t_sectors_pipe_lsu_mem_global_op_read.sum

# 全局内存写入吞吐
--metrics lts__t_sectors_pipe_lsu_mem_global_op_write.sum

# DRAM 吞吐
--metrics dram__throughput.avg.pct_of_peak_sustained_elapsed

# FP32 FMA 吞吐
--metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum
```
