#!/bin/bash
# Nsight Systems (nsys) 时间线 profiling 脚本
#
# nsys 用于"看 timeline / 找瓶颈段",ncu 用于"逐 kernel 细看".
# 本文件是 run_ncu.sh 的同构镜像,case 与之一一对应,便于交叉对照.
#
# 用法:
#   bash profiling/profile_nsys.sh tiled       # CUDA tiled matmul timeline
#   bash profiling/profile_nsys.sh triton      # Triton MLP training timeline
#   bash profiling/profile_nsys.sh mlp-cuda    # CUDAMLP 端到端
#   bash profiling/profile_nsys.sh mlp-cutile  # CUTILEMLP 端到端
#   bash profiling/profile_nsys.sh compare     # PyTorch vs Triton (5 steps)
#
# 输出: results/*.nsys-rep (可在 Nsight Systems GUI 中打开)

MODE=${1:-tiled}
M=${M:-512}
K=${K:-768}
N=${N:-3072}
STEPS=${STEPS:-5}

mkdir -p results

# 通用 nsys 包装. 调用方式: nsys_run <report_name> <python_one_liner>
nsys_run() {
    local report="$1"; shift
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        --output="results/${report}" \
        python -c "$*"
}

case $MODE in
    tiled)
        echo "=== nsys: CUDA tiled matmul (M=$M K=$K N=$N) ==="
        nsys_run nsys_tiled "
import torch, mlp_cuda
a = torch.randn($M, $K, device='cuda', dtype=torch.float32)
b = torch.randn($K, $N, device='cuda', dtype=torch.float32)
for _ in range(20):
    c = mlp_cuda.matmul_tiled_auto(a, b)
torch.cuda.synchronize()
"
        ;;
    triton)
        echo "=== nsys: Triton MLP training ($STEPS steps) ==="
        nsys_run nsys_triton_mlp "
from python.mnist.triton_model import TritonMLP
from python.mnist.model import MLPConfig
from python.mnist.trainer import Trainer, create_mnist_loaders
import torch
config = MLPConfig(hidden_dims=[784, 256, 128, 10], activation='gelu')
model = TritonMLP(config).cuda()
train_loader, _ = create_mnist_loaders(batch_size=128)
trainer = Trainer(model, lr=1e-3)
it = iter(train_loader)
for step in range($STEPS):
    x, y = next(it)
    x, y = x.cuda(), y.cuda()
    trainer.optimizer.zero_grad(set_to_none=True)
    loss = trainer.criterion(model(x), y)
    loss.backward()
    trainer.optimizer.step()
torch.cuda.synchronize()
"
        ;;
    mlp-cuda)
        echo "=== nsys: CUDAMLP training ($STEPS steps) ==="
        nsys_run nsys_mlp_cuda "
from python.mnist.cuda_model import CUDAMLP
from python.mnist.model import MLPConfig
from python.mnist.trainer import Trainer, create_mnist_loaders
import torch
config = MLPConfig(hidden_dims=[784, 256, 128, 10], activation='gelu')
model = CUDAMLP(config).cuda()
train_loader, _ = create_mnist_loaders(batch_size=128)
trainer = Trainer(model, lr=1e-3)
it = iter(train_loader)
for step in range($STEPS):
    x, y = next(it)
    x, y = x.cuda(), y.cuda()
    trainer.optimizer.zero_grad(set_to_none=True)
    loss = trainer.criterion(model(x), y)
    loss.backward()
    trainer.optimizer.step()
torch.cuda.synchronize()
"
        ;;
    mlp-cutile)
        echo "=== nsys: CUTILEMLP training ($STEPS steps) ==="
        nsys_run nsys_mlp_cutile "
from python.mnist.cutile_model import CUTILEMLP
from python.mnist.model import MLPConfig
from python.mnist.trainer import Trainer, create_mnist_loaders
import torch
config = MLPConfig(hidden_dims=[784, 256, 128, 10], activation='gelu')
model = CUTILEMLP(config).cuda()
train_loader, _ = create_mnist_loaders(batch_size=128)
trainer = Trainer(model, lr=1e-3)
it = iter(train_loader)
for step in range($STEPS):
    x, y = next(it)
    x, y = x.cuda(), y.cuda()
    trainer.optimizer.zero_grad(set_to_none=True)
    loss = trainer.criterion(model(x), y)
    loss.backward()
    trainer.optimizer.step()
torch.cuda.synchronize()
"
        ;;
    compare)
        echo "=== nsys: PyTorch vs Triton vs CUDA (5 steps each) ==="
        nsys_run nsys_compare "
from python.mnist.model import MLP, MLPConfig
from python.mnist.triton_model import TritonMLP
from python.mnist.cuda_model import CUDAMLP
from python.mnist.trainer import create_mnist_loaders
import torch
config = MLPConfig(hidden_dims=[784, 256, 128, 10], activation='gelu')
train_loader, _ = create_mnist_loaders(batch_size=128)
crit = torch.nn.CrossEntropyLoss()

for name, ModelCls in [('pytorch', MLP), ('triton', TritonMLP), ('cuda', CUDAMLP)]:
    m = ModelCls(config).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    torch.cuda.nvtx.range_push(f'backend:{name}')
    it = iter(train_loader)
    for _ in range($STEPS):
        x, y = next(it); x, y = x.cuda(), y.cuda()
        opt.zero_grad(set_to_none=True)
        loss = crit(m(x), y); loss.backward(); opt.step()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
"
        ;;
    *)
        echo "Usage: $0 {tiled|triton|mlp-cuda|mlp-cutile|compare}"
        echo "Override: M, K, N, STEPS env vars"
        exit 1
        ;;
esac

echo "nsys done. Report at results/${MODE/-/_}.nsys-rep"
echo "Open with: nsys-ui results/*.nsys-rep   (Linux)"
echo "Or Windows: \\\\wsl\$\\Ubuntu\$(pwd)/results/"
