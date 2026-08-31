#!/usr/bin/env bash
# 一键验证（其它机器用）：preflight -> build -> tests -> bench smoke -> status
# 用法: bash scripts/verify.sh [PYTHON]
set -euo pipefail

PY="${1:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "===== [1/4] preflight (lane / toolchain) ====="
"$PY" tools/preflight.py || echo "preflight 警告 — 检查 GPU/CUDA 工具链 (Triton/GPU 推理仍可用)"

echo
echo "===== [2/4] import smoke ====="
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
"$PY" -c "import triton_kernels; print('triton_kernels OK')"
if "$PY" -c "import mlp_cuda" 2>/dev/null; then echo "mlp_cuda OK"; else echo "mlp_cuda NOT BUILT (make install 后可用)"; fi

echo
echo "===== [3/4] tests (quick) ====="
"$PY" -m pytest tests/test_transformer_mlp.py tests/test_torch_registration.py -q --tb=line || echo "TESTS FAILED"

echo
echo "===== [4/4] SwiGLU block smoke bench (with GPU thermal capture) ====="
"$PY" scripts/gpu_telemetry.py --interval 2 --cmd "$PY bench/run.py --suite prefill --dtypes fp16 --backends eager,concat,triton --warmup 5 --iters 20 --max-cases 4" || echo "BENCH/THERMAL FAILED"

echo
echo "===== 验证完成 ====="
"$PY" tools/status.py 2>/dev/null || true
