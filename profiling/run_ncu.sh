#!/bin/bash
# Nsight Compute profiling 脚本
#
# 用法:
#   bash profiling/run_ncu.sh naive       # CUDA naive matmul
#   bash profiling/run_ncu.sh tiled       # CUDA tiled matmul
#   bash profiling/run_ncu.sh roofline    # Roofline 分析
#   bash profiling/run_ncu.sh triton      # Triton MLP 训练
#   bash profiling/run_ncu.sh triton-matmul  # Triton matmul kernel
#   bash profiling/run_ncu.sh compare     # PyTorch vs Triton timeline
#
# 输出: results/*.ncu-rep, results/*.trace.json.gz

MODE=${1:-tiled}

mkdir -p results

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
    triton)
        echo "=== Profiling Triton MLP training ==="
        ncu --set full \
            -o results/ncu_triton_mlp \
            python -c "
from python.mnist.triton_model import TritonMLP
from python.mnist.model import MLPConfig
from python.mnist.trainer import Trainer, create_mnist_loaders
import torch

config = MLPConfig(hidden_dims=[784, 256, 128, 10], activation='gelu')
model = TritonMLP(config).cuda()
train_loader, test_loader = create_mnist_loaders(batch_size=128)
trainer = Trainer(model, lr=1e-3)
# 只训练 2 步用于 profiling
for x, y in train_loader:
    x, y = x.cuda(), y.cuda()
    trainer.optimizer.zero_grad(set_to_none=True)
    loss = trainer.criterion(model(x), y)
    loss.backward()
    trainer.optimizer.step()
    break
print('Triton MLP profiling step done')
"
        ;;
    triton-matmul)
        echo "=== Profiling Triton matmul kernel ==="
        ncu --set full \
            -o results/ncu_triton_matmul \
            python -c "
import torch
from triton_kernels.matmul import tiled_matmul

M, K, N = 512, 768, 3072
a = torch.randn(M, K, device='cuda', dtype=torch.float32)
b = torch.randn(K, N, device='cuda', dtype=torch.float32)
# warmup
for _ in range(3):
    _ = tiled_matmul(a, b)
# profiled run
c = tiled_matmul(a, b)
torch.cuda.synchronize()
print(f'Triton matmul profiling done: {c.shape}')
"
        ;;
    compare)
        echo "=== PyTorch vs Triton timeline comparison ==="
        python profiling/profile_compare.py --epochs 2
        ;;
    *)
        echo "Usage: $0 {naive|tiled|roofline|speedof|triton|triton-matmul|compare}"
        exit 1
        ;;
esac

echo "Profiling done. Results in results/"
echo "View with: ncu-ui results/ncu_*.ncu-rep"
