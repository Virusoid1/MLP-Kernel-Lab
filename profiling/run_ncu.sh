#!/bin/bash
# Nsight Compute profiling 脚本
#
# 用法:
#   bash profiling/run_ncu.sh naive          # CUDA naive matmul
#   bash profiling/run_ncu.sh tiled          # CUDA tiled matmul (auto dispatch)
#   bash profiling/run_ncu.sh roofline       # Roofline 分析 (tiled matmul)
#   bash profiling/run_ncu.sh speedof        # Speed-of-Light 利用率/瓶颈
#   bash profiling/run_ncu.sh triton         # Triton MLP 训练 1 步
#   bash profiling/run_ncu.sh triton-matmul  # Triton matmul kernel
#   bash profiling/run_ncu.sh cuda           # CUDA matmul + LayerNorm + activation
#   bash profiling/run_ncu.sh cutile         # cuTile matmul (需安装 cuda-tile)
#   bash profiling/run_ncu.sh mlp-cuda       # CUDAMLP 训练 1 步 (端到端)
#   bash profiling/run_ncu.sh mlp-cutile     # CUTILEMLP 训练 1 步
#   bash profiling/run_ncu.sh compare        # PyTorch vs Triton timeline
#
# 输出: results/*.ncu-rep, results/*.trace.json.gz

MODE=${1:-tiled}
M=${M:-512}
K=${K:-768}
N=${N:-3072}

mkdir -p results

# 通用: 单 kernel ncu 包装. 调用方式: ncu_run <report_name> <python_one_liner>
ncu_run() {
    local report="$1"; shift
    ncu --set full \
        --launch-skip 5 --launch-count 1 \
        -o "results/${report}" \
        python -c "$*"
}

ncu_roofline() {
    local report="$1"; shift
    ncu --set roofline \
        --launch-skip 5 --launch-count 1 \
        -o "results/${report}" \
        python -c "$*"
}

ncu_sol() {
    local report="$1"; shift
    ncu --set speed_of_light \
        --launch-skip 5 --launch-count 1 \
        -o "results/${report}" \
        python -c "$*"
}

case $MODE in
    naive)
        echo "=== Profiling CUDA naive matmul (M=$M K=$K N=$N) ==="
        ncu_run ncu_naive "
import torch, mlp_cuda
a = torch.randn($M, $K, device='cuda', dtype=torch.float32)
b = torch.randn($K, $N, device='cuda', dtype=torch.float32)
for _ in range(8):
    c = mlp_cuda.matmul_naive(a, b)
torch.cuda.synchronize()
print('matmul_naive done:', c.shape)
"
        ;;
    tiled)
        echo "=== Profiling CUDA tiled matmul (auto dispatch, M=$M K=$K N=$N) ==="
        ncu_run ncu_tiled "
import torch, mlp_cuda
a = torch.randn($M, $K, device='cuda', dtype=torch.float32)
b = torch.randn($K, $N, device='cuda', dtype=torch.float32)
for _ in range(8):
    c = mlp_cuda.matmul_tiled_auto(a, b)
torch.cuda.synchronize()
print('matmul_tiled_auto done:', c.shape)
"
        ;;
    roofline)
        echo "=== Roofline analysis for CUDA tiled matmul ==="
        ncu_roofline ncu_roofline "
import torch, mlp_cuda
a = torch.randn($M, $K, device='cuda', dtype=torch.float32)
b = torch.randn($K, $N, device='cuda', dtype=torch.float32)
for _ in range(8):
    c = mlp_cuda.matmul_tiled_auto(a, b)
torch.cuda.synchronize()
"
        ;;
    speedof)
        echo "=== Speed-of-Light (利用率/瓶颈) ==="
        ncu_sol ncu_sol "
import torch, mlp_cuda
a = torch.randn($M, $K, device='cuda', dtype=torch.float32)
b = torch.randn($K, $N, device='cuda', dtype=torch.float32)
for _ in range(8):
    c = mlp_cuda.matmul_tiled_auto(a, b)
torch.cuda.synchronize()
"
        ;;
    triton)
        echo "=== Profiling Triton MLP training (1 step) ==="
        ncu_run ncu_triton_mlp "
from python.mnist.triton_model import TritonMLP
from python.mnist.model import MLPConfig
from python.mnist.trainer import Trainer, create_mnist_loaders
import torch
config = MLPConfig(hidden_dims=[784, 256, 128, 10], activation='gelu')
model = TritonMLP(config).cuda()
train_loader, _ = create_mnist_loaders(batch_size=128)
trainer = Trainer(model, lr=1e-3)
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
        ncu_run ncu_triton_matmul "
import torch
from triton_kernels.matmul import tiled_matmul
a = torch.randn($M, $K, device='cuda', dtype=torch.float32)
b = torch.randn($K, $N, device='cuda', dtype=torch.float32)
for _ in range(3):
    _ = tiled_matmul(a, b)
c = tiled_matmul(a, b)
torch.cuda.synchronize()
print('triton matmul done:', c.shape)
"
        ;;
    cuda)
        echo "=== Profiling CUDA algo bundle (matmul + layernorm + activation) ==="
        ncu_run ncu_cuda_bundle "
import torch, mlp_cuda
x = torch.randn($M, $K, device='cuda', dtype=torch.float32)
w = torch.randn($K, $N, device='cuda', dtype=torch.float32)
gamma = torch.ones($N, device='cuda', dtype=torch.float32)
beta = torch.zeros($N, device='cuda', dtype=torch.float32)
for _ in range(3):
    h = mlp_cuda.matmul_tiled_auto(x, w)
    y, mean, rstd = mlp_cuda.layernorm_forward(h, gamma, beta, 1e-5)
    out = mlp_cuda.gelu(y)
torch.cuda.synchronize()
print('cuda bundle done:', out.shape)
"
        ;;
    cutile)
        echo "=== Profiling cuTile matmul (requires cuda-tile) ==="
        ncu_run ncu_cutile_matmul "
import torch
try:
    from cutile_kernels.matmul import tiled_matmul
except ImportError as e:
    print('cuTile not installed:', e)
    raise SystemExit
a = torch.randn($M, $K, device='cuda', dtype=torch.float32)
b = torch.randn($K, $N, device='cuda', dtype=torch.float32)
for _ in range(3):
    _ = tiled_matmul(a, b)
c = tiled_matmul(a, b)
torch.cuda.synchronize()
print('cutile matmul done:', c.shape)
"
        ;;
    mlp-cuda)
        echo "=== Profiling CUDAMLP training (1 step, end-to-end) ==="
        ncu_run ncu_mlp_cuda "
from python.mnist.cuda_model import CUDAMLP
from python.mnist.model import MLPConfig
from python.mnist.trainer import Trainer, create_mnist_loaders
import torch
config = MLPConfig(hidden_dims=[784, 256, 128, 10], activation='gelu')
model = CUDAMLP(config).cuda()
train_loader, _ = create_mnist_loaders(batch_size=128)
trainer = Trainer(model, lr=1e-3)
for x, y in train_loader:
    x, y = x.cuda(), y.cuda()
    trainer.optimizer.zero_grad(set_to_none=True)
    loss = trainer.criterion(model(x), y)
    loss.backward()
    trainer.optimizer.step()
    break
print('CUDAMLP profiling step done')
"
        ;;
    mlp-cutile)
        echo "=== Profiling CUTILEMLP training (1 step, end-to-end) ==="
        ncu_run ncu_mlp_cutile "
from python.mnist.cutile_model import CUTILEMLP
from python.mnist.model import MLPConfig
from python.mnist.trainer import Trainer, create_mnist_loaders
import torch
config = MLPConfig(hidden_dims=[784, 256, 128, 10], activation='gelu')
model = CUTILEMLP(config).cuda()
train_loader, _ = create_mnist_loaders(batch_size=128)
trainer = Trainer(model, lr=1e-3)
for x, y in train_loader:
    x, y = x.cuda(), y.cuda()
    trainer.optimizer.zero_grad(set_to_none=True)
    loss = trainer.criterion(model(x), y)
    loss.backward()
    trainer.optimizer.step()
    break
print('CUTILEMLP profiling step done')
"
        ;;
    compare)
        echo "=== PyTorch vs Triton timeline comparison ==="
        python profiling/profile_compare.py --epochs 2
        ;;
    *)
        echo "Usage: $0 {naive|tiled|roofline|speedof|triton|triton-matmul|cuda|cutile|mlp-cuda|mlp-cutile|compare}"
        echo "Override matrix size: M=512 K=768 N=3072 bash profiling/run_ncu.sh tiled"
        exit 1
        ;;
esac

echo "Profiling done. Results in results/"
echo "View with: ncu-ui results/ncu_*.ncu-rep"
