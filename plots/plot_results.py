"""
Benchmark 结果绘图脚本

生成图片:
  1. latency_vs_M.png       - 延迟随 M (batch*sqlen) 变化
  2. speedup_vs_pytorch.png  - 相对 PyTorch 的加速比
  3. speedup_vs_naive.png   - 相对 naive CUDA 的加速比
  4. bandwidth_or_tflops.png - TFLOPS 随 shape 变化

用法:
    python plots/plot_results.py --csv results/benchmark_results.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 无 GUI 环境


plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_data(csv_path: str) -> pd.DataFrame:
    """加载 benchmark CSV"""
    df = pd.read_csv(csv_path)
    return df


def plot_latency_vs_M(df: pd.DataFrame, output_dir: str = 'plots'):
    """延迟随 M 变化"""
    fig, ax = plt.subplots()

    for dtype in df['dtype'].unique():
        df_dtype = df[df['dtype'] == dtype]
        for impl in df_dtype['impl'].unique():
            df_impl = df_dtype[df_dtype['impl'] == impl]
            ax.plot(df_impl['M'], df_impl['latency_median_ms'],
                    marker='o', label=f'{impl} ({dtype})')

    ax.set_xlabel('M (batch * seq_len)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('MLP Matmul Latency vs M')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xscale('log', base=2)

    fig.tight_layout()
    fig.savefig(f'{output_dir}/latency_vs_M.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/latency_vs_M.png")


def plot_speedup_vs_pytorch(df: pd.DataFrame, output_dir: str = 'plots'):
    """相对 PyTorch 的加速比"""
    fig, ax = plt.subplots()

    df_pivot = df.pivot_table(
        index=['M', 'K', 'N', 'dtype'],
        columns='impl',
        values='latency_median_ms'
    ).reset_index()

    if 'torch' not in df_pivot.columns:
        print("Warning: 'torch' baseline not in data, skipping speedup vs pytorch plot")
        return

    impls = [c for c in df_pivot.columns if c not in ('M', 'K', 'N', 'dtype', 'torch')]

    x_labels = df_pivot['M'].astype(str)
    x = range(len(x_labels))

    bar_width = 0.8 / len(impls) if impls else 0.4

    for i, impl in enumerate(impls):
        speedup = df_pivot['torch'] / df_pivot[impl]
        ax.bar([xi + i * bar_width for xi in x], speedup, bar_width, label=impl)

    ax.set_xlabel('M')
    ax.set_ylabel('Speedup vs PyTorch')
    ax.set_title('Speedup vs PyTorch Baseline')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks([xi + bar_width * (len(impls) - 1) / 2 for xi in x])
    ax.set_xticklabels(x_labels, rotation=45)
    ax.legend()

    fig.tight_layout()
    fig.savefig(f'{output_dir}/speedup_vs_pytorch.png', dpi=150)
    print(f"Saved: {output_dir}/speedup_vs_pytorch.png")


def plot_tflops(df: pd.DataFrame, output_dir: str = 'plots'):
    """TFLOPS 随 shape 变化"""
    fig, ax = plt.subplots()

    for dtype in df['dtype'].unique():
        df_dtype = df[df['dtype'] == dtype]
        for impl in df_dtype['impl'].unique():
            df_impl = df_dtype[df_dtype['impl'] == impl]
            ax.plot(df_impl['M'], df_impl['TFLOPS'],
                    marker='s', label=f'{impl} ({dtype})')

    ax.set_xlabel('M (batch * seq_len)')
    ax.set_ylabel('TFLOPS')
    ax.set_title('TFLOPS vs M')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xscale('log', base=2)

    fig.tight_layout()
    fig.savefig(f'{output_dir}/bandwidth_or_tflops.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/bandwidth_or_tflops.png")


def main():
    parser = argparse.ArgumentParser(description='Plot benchmark results')
    parser.add_argument('--csv', type=str, default='results/benchmark_results.csv',
                        help='Path to benchmark CSV')
    parser.add_argument('--output', type=str, default='plots',
                        help='Output directory')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Benchmark CSV not found: {csv_path}")
        print("Run 'python benchmark_ops.py' first.")
        print("Generating demo plot with sample data structure...")
        return

    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    plot_latency_vs_M(df, args.output)
    plot_speedup_vs_pytorch(df, args.output)
    plot_tflops(df, args.output)

    print("All plots generated.")


if __name__ == '__main__':
    main()
