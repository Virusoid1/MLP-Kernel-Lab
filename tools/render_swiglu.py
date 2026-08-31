#!/usr/bin/env python3
"""把 swiglu_bench.json 渲染成可读 Markdown / CSV（离线，不需要 GPU）。

用法:
    python tools/render_swiglu.py artifacts/<run-id>/swiglu_bench.json [--format md|csv] [--opts]
"""

from __future__ import annotations

import argparse
import json
import csv
import sys
from pathlib import Path


def render_md(payload: dict) -> str:
    md = payload["metadata"]
    out = [f"# SwiGLU MLP Block Benchmark — {payload.get('suite', '?')}", ""]
    out.append(f"- git: {md.get('git', {}).get('short_commit')} ({md.get('git', {}).get('branch')}, dirty={md.get('git', {}).get('dirty')})")
    out.append(f"- gpu: {md.get('gpu', {}).get('name')} (cc {md.get('gpu', {}).get('cc')})")
    out.append(f"- driver: {md.get('driver')} | cuda: {md.get('cuda')} | torch {md.get('torch')} / triton {md.get('triton')}")
    out.append(f"- protocol: warmup={md.get('benchmark_protocol', {}).get('warmup')} iters={md.get('benchmark_protocol', {}).get('iters')}")
    out.append("")
    out.append(f"- cases: {payload['summary']['cases_total']} | errors: {payload['summary']['cases_with_error']} | corr-fail: {payload['summary']['correctness_failed']} | best speedup: {payload['summary'].get('best_speedup')}")
    out.append("")

    rows = [r for r in payload["rows"] if r.get("median_ms") is not None]
    if not rows:
        out.append("_no benchmark rows_")
        return "\n".join(out)

    # 按 dtype / suite 分块
    for suite_group in ["decode", "prefill"]:
        pass
    out.append("| backend | dtype | M | K | F | median ms | p95 ms | TFLOPS | vs eager | corr l2 | peak MB |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x.get("M", 0), x.get("K", 0), x.get("F", 0), x.get("backend", ""), x.get("dtype", ""))):
        sp = r.get("speedup_vs_eager")
        sp_s = f"{sp:.2f}x" if sp else "-"
        corr = r.get("correctness_norm_l2")
        corr_s = f"{corr:.1e}" if corr is not None else "-"
        out.append(
            f"| {r.get('backend','?'):7s} | {r.get('dtype','?'):4s} | {r.get('M','?'):5d} | {r.get('K','?'):5d} | {r.get('F','?'):6d} "
            f"| {r.get('median_ms',0):9.4f} | {r.get('p95_ms',0):9.4f} | {r.get('tflops',0):7.2f} | {sp_s:>7s} | {corr_s:>8s} | {r.get('peak_gpu_mem_mb','-')} |"
        )
    out.append("")
    return "\n".join(out)


def render_csv(payload: dict, path: Path):
    rows = payload["rows"]
    keys = ["backend", "dtype", "M", "K", "F", "median_ms", "mean_ms", "p95_ms", "std_ms",
            "tflops", "speedup_vs_eager", "correctness_norm_l2", "correctness_passed",
            "peak_gpu_mem_mb", "n_repeat", "error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="swiglu_bench.json path")
    ap.add_argument("--format", default="md", choices=["md", "csv"])
    args = ap.parse_args()
    p = Path(args.json)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if args.format == "md":
        print(render_md(payload))
    else:
        out = p.with_suffix(".csv")
        render_csv(payload, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
