#!/usr/bin/env python3
"""v2 项目状态总览（UI/展示）：一键输出当前证据状态。

用法: python tools/status.py [--json]
展示: git 状态 / 测试总数 / 最新 manifest / 最新 swiglu sweep / 关键证据链接
"""

from __future__ import annotations

import argparse
import json
import glob
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "artifacts"


def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=str(REPO))
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _latest(dir_path, pattern):
    if not dir_path.exists():
        return None
    m = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return m[0] if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    git = {
        "commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }

    # 测试数（junit 不在手时用静态统计）
    total_tests = 0
    for f in glob.glob(str(REPO / "tests" / "test_*.py")):
        txt = Path(f).read_text(encoding="utf-8", errors="ignore")
        total_tests += txt.count("def test_")

    # 最新 reproduce manifest + swiglu sweep
    latest_manifest = _latest(ARTIFACTS, "*/*/manifest.json")
    manifests = sorted(ARTIFACTS.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True) if ARTIFACTS.exists() else []
    latest_manifest = manifests[0] if manifests else None

    sweeps = sorted(ARTIFACTS.glob("swiglu_*/swiglu_bench.json"), key=lambda p: p.stat().st_mtime, reverse=True) if ARTIFACTS.exists() else []
    latest_sweep = sweeps[0] if sweeps else None

    sweep_summary = None
    if latest_sweep:
        try:
            d = json.loads(latest_sweep.read_text(encoding="utf-8"))
            sweep_summary = d.get("summary")
        except Exception:
            sweep_summary = None

    if args.json:
        print(json.dumps({
            "git": git,
            "test_functions_approx": total_tests,
            "latest_manifest": str(latest_manifest) if latest_manifest else None,
            "latest_sweep": str(latest_sweep) if latest_sweep else None,
            "latest_sweep_summary": sweep_summary,
        }, ensure_ascii=False, indent=2))
        return 0

    # 人类可读
    print("=" * 66)
    print(f"  MLP-Kernel-Lab v2 — 项目状态")
    print("=" * 66)
    print("  git   : %s @ %s (%s)" % (git["branch"], git["commit"], "dirty" if git["dirty"] else "clean"))
    print("  测试  : %d 项 pytest *源码函数*（parametrize 展开后 ~136 用例，以 make reproduce 报告为准）" % total_tests)
    if latest_manifest:
        print("  manifest: %s" % latest_manifest.relative_to(REPO))
    if latest_sweep:
        print("  最新 sweep: %s" % latest_sweep.relative_to(REPO))
        if sweep_summary:
            s = sweep_summary
            print("    cases=%s errors=%s corr_fail=%s best_speedup=%s"
                  % (s.get("cases_total"), s.get("cases_with_error"),
                     s.get("correctness_failed"), s.get("best_speedup")))
    print("")
    print("  证据入口:")
    print("    - claim-matrix      : docs/claim-matrix.md")
    print("    - 实验报告          : docs/experiments/swiglu-sweep-20260831-3070.md")
    print("    - 一键复现          : make reproduce (PYTHON=.../venv/bin/python)")
    print("    - 基准              : python bench/run.py --suite all --dtypes fp32,fp16")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
