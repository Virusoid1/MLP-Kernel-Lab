.PHONY: install test bench profile clean plots test-cuda test-python

# Python 测试
test-python:
	python -m pytest tests/ -v

# C++ CUDA 测试
test-cuda: build/test_kernels
	@echo "Running CUDA C++ tests..."
	./build/test_kernels

build/test_kernels: tests/test_kernels.cu
	@mkdir -p build
	nvcc -O3 -o $@ $< --use_fast_math \
		-gencode=arch=compute_86,code=sm_86

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

# 测试
test:
	python bench/compare_correctness.py --all

# benchmark
bench:
	python bench/benchmark.py --config bench/benchmark_shapes.yaml --output results/benchmark_results.csv

# bench 快速测试
bench-quick:
	python bench/benchmark.py --config bench/benchmark_shapes.yaml

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
