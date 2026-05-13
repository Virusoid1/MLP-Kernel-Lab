#!/bin/bash
# Nsight Compute profiling 脚本
#
# 用法:
#   chmod +x profiling/run_ncu.sh
#   bash profiling/run_ncu.sh naive
#   bash profiling/run_ncu.sh tiled
#   bash profiling/run_ncu.sh roofline
#
# 输出: results/*.ncu-rep

MODE=${1:-tiled}

case $MODE in
    naive)
        echo "=== Profiling CUDA naive matmul ==="
        ncu --set full \
            -o results/ncu_naive \
            python bench/benchmark.py --impl cuda_naive --M 512 --K 768 --N 3072
        ;;
    tiled)
        echo "=== Profiling CUDA tiled matmul ==="
        ncu --set full \
            -o results/ncu_tiled \
            python bench/benchmark.py --impl cuda_tiled --M 512 --K 768 --N 3072
        ;;
    roofline)
        echo "=== Roofline analysis for tiled matmul ==="
        ncu --set roofline \
            -o results/ncu_roofline \
            python bench/benchmark.py --impl cuda_tiled --M 512 --K 768 --N 3072
        ;;
    speedof)
        echo "=== SpeedOfLight (利用率和瓶颈) ==="
        ncu --set speed_of_light \
            -o results/ncu_sol \
            python bench/benchmark.py --impl cuda_tiled --M 512 --K 768 --N 3072
        ;;
    *)
        echo "Usage: $0 {naive|tiled|roofline|speedof}"
        exit 1
        ;;
esac

echo "Profiling done. Results in results/"
echo "View with: ncu-ui results/ncu_*.ncu-rep"
