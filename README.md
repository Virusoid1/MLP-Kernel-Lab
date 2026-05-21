# MLP-Kernel-Lab

Custom CUDA & Triton kernels for Transformer MLP inference, with profiling-driven optimization and benchmark against PyTorch baselines.

## Highlights

- Naive / shared-memory tiled / fused activation CUDA kernels
- Triton block-level matmul & MLP kernels
- PyTorch C++ extension integration
- Unified benchmark: latency, TFLOPS, speedup, numerical correctness
- Nsight Compute profiling & optimization log

## Project Structure

```
MLP-Kernel-Lab/
├── kernels/                    # CUDA kernel implementations
│   ├── vector_add.cu           #   Day 1: CUDA basics & timing
│   ├── matmul_naive.cu         #   Day 2: naive matmul
│   ├── matmul_tiled.cu         #   Day 4: shared memory tiled matmul
│   ├── activation.cu           #   Day 5: GELU / SiLU device functions
│   ├── mlp_fused_first_layer.cu#   Day 5: fused matmul+bias+GELU
│   ├── swiglu_fused.cu         #   Day 11: fused SwiGLU
│   ├── mlp_cuda_kernels.cu     #   kernel launch wrappers
│   └── binding.cpp             #   PyTorch C++ extension binding
├── triton_kernels/             # Triton kernel implementations
│   ├── matmul_triton.py        #   tl.dot based matmul
│   ├── mlp_triton.py           #   fused matmul+bias+GELU
│   └── swiglu_triton.py        #   fused SwiGLU
├── python/                     # Python entry points
│   ├── mlp_reference.py        #   PyTorch baseline & correctness check
│   └── torch_extension.py      #   CUDA extension wrapper
├── bench/                      # Benchmark framework
│   ├── benchmark.py            #   unified benchmark runner
│   ├── compare_correctness.py  #   numerical error validation
│   └── benchmark_shapes.yaml   #   shape / dtype / impl config
├── profiling/                  # Nsight Compute scripts
│   ├── run_ncu.sh              #   profiling launcher
│   └── ncu_notes.md            #   metric templates
├── docs/                       # Reports & notes
│   ├── report.md               #   final optimization report
│   ├── cuda_notes.md           #   CUDA learning notes
│   ├── triton_notes.md         #   Triton learning notes
│   └── optimization_log.md     #   per-version optimization log
├── plots/                      # Result visualization
│   └── plot_results.py         #   generate benchmark plots
├── setup.py                    # CUDA extension build config
├── CMakeLists.txt              # standalone CUDA build (optional)
├── Makefile                    # install / test / bench / profile
└── requirements.txt
```

## Quick Start

### Prerequisites

- NVIDIA GPU (compute capability 7.5+)
- CUDA Toolkit 11.8+
- Python 3.8+
- PyTorch 2.0+
- Triton 2.0+

### Install

```bash
pip install -r requirements.txt
make install          # build & install CUDA extension
```

### Check Environment

```bash
make check
```

### Run Benchmark

```bash
make bench                             # full benchmark
make bench-quick                       # quick test
python bench/benchmark.py --impl torch # single implementation
```

### Correctness Check

```bash
make test
python bench/compare_correctness.py --all
```

### Profiling

```bash
bash profiling/run_ncu.sh naive       # profile naive kernel
bash profiling/run_ncu.sh tiled       # profile tiled kernel
bash profiling/run_ncu.sh roofline    # roofline analysis
```

### Generate Plots

```bash
make plots
```

## Benchmark Shapes

Configured in `bench/benchmark_shapes.yaml`, defaults to LLM-like shapes:

| Config | M (batch*seq) | K (hidden) | N (intermediate) | Model |
|--------|---------------|------------|------------------|-------|
| BERT-like | 128, 512, 2048 | 768 | 3072 | BERT-base |
| LLM-like | 128, 512, 2048 | 4096 | 11008 | Llama-7B |

## Implementations

| Impl | Description | Status |
|------|-------------|--------|
| `torch` | `torch.matmul` + `F.gelu` baseline | Done |
| `cuda_naive` | 1 thread = 1 output element | TODO |
| `cuda_tiled` | Shared memory tiled matmul | TODO |
| `cuda_fused` | Fused matmul + bias + GELU | TODO |
| `triton_matmul` | `tl.dot` block matmul | TODO |
| `triton_mlp` | Fused matmul + bias + GELU | TODO |

## Optimization Path

```
v0  naive matmul          → baseline
v1  shared memory tiling  → reduce global memory traffic
v2  block size tuning     → improve occupancy
v3  coalesced access      → improve memory throughput
v4  fused activation      → eliminate intermediate writes
v5  FP16 support          → halve memory bandwidth
```

See `docs/optimization_log.md` for detailed per-version results.

## License

MIT
