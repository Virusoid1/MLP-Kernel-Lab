.PHONY: install test bench profile clean plots test-cuda test-python profile-nsys profile-ops bench-op bench-cu analyze gate

# Python 测试
test-python:
	python -m pytest tests/ -v

# 当前 GPU compute capability（与 setup.py 同一策略；可被 TORCH_CUDA_ARCH_LIST 覆盖）
CUDA_ARCH ?= $(shell python -c "import torch; print(f'{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')" 2>/dev/null || echo 86)
GENCODE = -gencode=arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)

# C++ CUDA 测试
test-cuda: build/test_kernels
	@echo "Running CUDA C++ tests... (arch=$(CUDA_ARCH))"
	./build/test_kernels

build/test_kernels: tests/test_kernels.cu
	@mkdir -p build
	nvcc -O3 -o $@ $< --use_fast_math \
		$(GENCODE)

# 安装 CUDA extension
install:
	python setup.py install

# 检查 CUDA 环境
check:
	@echo "=== CUDA Version ==="
	@nvcc --version || echo "nvcc not found"
	@echo ""
	@echo "=== GPU Info ==="
	@nvidia-smi || echo "nvidia-smi not found"
	@echo ""
	@echo "=== PyTorch CUDA ==="
	@python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
	@echo ""
	@echo "=== Triton ==="
	@python -c "import triton; print(f'Triton {triton.__version__}')" 2>/dev/null || echo "Triton not installed"

# 测试 (算子级横向对比, 唯一权威 benchmark 入口)
test:
	python benchmark_ops.py

# benchmark (算子级横向对比, 全尺寸默认配置)
bench:
	python benchmark_ops.py --warmup 20 --iters 100

# bench 快速测试 (小尺寸)
bench-quick:
	python benchmark_ops.py --sizes small --warmup 5 --iters 20

# profiling (需要 Nsight Compute)
profile-naive:
	bash profiling/run_ncu.sh naive

profile-tiled:
	bash profiling/run_ncu.sh tiled

profile-roofline:
	bash profiling/run_ncu.sh roofline

profile-triton:
	bash profiling/run_ncu.sh triton

profile-triton-matmul:
	bash profiling/run_ncu.sh triton-matmul

profile-compare:
	bash profiling/run_ncu.sh compare

# nsys 时间线 profiling (新增, 与 run_ncu.sh 同构)
profile-nsys:
	bash profiling/profile_nsys.sh tiled

# 算子级 profiling driver (NVTX 包裹, 跨 backend)
profile-ops:
	python profiling/profile_ops.py

# ============================================================
# 测量-分析-优化 闭环 (Phase 1-3 of plan)
# ============================================================
TS := $(shell date +%Y%m%d_%H%M%S)
BASELINE ?= results/baseline.json
LATEST   ?= $(shell ls -t results/op_bench_*.json 2>/dev/null | head -1)

# Phase 1 Measure: 跑算子级 bench (dtype sweep + roofline + metadata)
bench-op:
	python benchmark_ops.py --dtypes fp32,fp16,bf16 --roofline \
	    --output results/op_bench_$(TS).json

# Phase 1 Measure: cuTile 专用 bench (4 轮取后 3 + warmup)
bench-cu:
	python profiling/bench_cutile.py $(ARGS)

# Phase 2 Analyze: 跟 baseline 对比, 默认只看 MLP_LAYERS shape
analyze:
	@if [ -z "$(LATEST)" ]; then echo "No candidate JSON found. Run 'make bench-op' first."; exit 1; fi
	@if [ ! -f "$(BASELINE)" ]; then echo "No baseline at $(BASELINE). First-time setup: cp $(LATEST) $(BASELINE)"; exit 1; fi
	python tools/analyze_bench.py $(BASELINE) $(LATEST) --shape MLP_LAYERS

# 完整闭环 gate: bench + analyze --gate (perf regress or correctness fail -> exit 1)
gate:
	$(MAKE) bench-op
	@LATEST=$$(ls -t results/op_bench_*.json | head -1); \
	if [ ! -f "$(BASELINE)" ]; then echo "Promoting first run to baseline: $$LATEST"; cp $$LATEST $(BASELINE); fi; \
	python tools/analyze_bench.py $(BASELINE) $$LATEST --shape MLP_LAYERS --gate

# 生成图表
plots:
	python plots/plot_results.py

# 清理
clean:
	rm -rf build/ dist/ *.egg-info/ results/*.ncu-rep results/*.csv
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 全部流程: 安装 -> 测试 -> benchmark -> 画图
all: install test bench plots
	@echo "Done!"
