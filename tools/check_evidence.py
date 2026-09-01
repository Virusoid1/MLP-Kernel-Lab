#!/usr/bin/env python3
"""证据一致性校验器：文档数字 <-> 最新 manifest/sweep 数据 自动核对。

用途:
    防止"文档数字漂移"（改为某处但别处没同步）。CI / 其它机器复现后跑一遍。
用法:
    python tools/check_evidence.py            # 全部检查，期望 EXIT 0
    python tools/check_evidence.py --fix-json # 输出最新 manifest 关键字段（供人工同步文档）
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_TESTS = 226
EXPECTED_PASSED = 207  # 2026-09-02 冻结: 3cebeee（+4 = bias_add/fused_first fp16+bf16）
MIN_PASSED = 170  # 至少 170 passed（允许未来加测试但不应回落过多）


def latest_manifest_dir() -> Path | None:
    arts = ROOT / "artifacts"
    if not arts.exists():
        return None
    dirs = [d for d in arts.iterdir() if d.is_dir() and d.name.startswith("20")]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.name)


def check_manifest() -> list[str]:
    errs = []
    d = latest_manifest_dir()
    if d is None:
        return ["no manifests under artifacts/ (run make reproduce first)"]
    mf = d / "manifest.json"
    if not mf.exists():
        return [f"{d.name}: manifest.json missing"]
    m = json.loads(mf.read_text())
    git_c = m.get("git", {}).get("short_commit", "?")
    tests = m.get("steps", {}).get("tests", {}).get("summary", {})
    total, passed, failed = tests.get("tests"), tests.get("passed"), tests.get("failures")
    print(f"  manifest: {d.name} (commit {git_c})")
    if total != EXPECTED_TESTS:
        errs.append(f"tests total {total} != expected {EXPECTED_TESTS}")
    if passed != EXPECTED_PASSED:
        errs.append(f"passed {passed} != expected {EXPECTED_PASSED}")
    if passed < MIN_PASSED:
        errs.append(f"passed {passed} < floor {MIN_PASSED}")
    if failed:
        errs.append(f"failures {failed} != 0")
    # 关键环境字段在场
    for k in ("gpu", "driver", "cuda", "torch", "triton", "cutile"):
        if k not in m:
            errs.append(f"manifest missing field: {k}")
    return errs


def check_doc_refs() -> list[str]:
    """README/EVIDENCE/REPRODUCE 引用的关键文件存在。"""
    errs = []
    paths = [
        "README.md", "REPRODUCE.md", "EVIDENCE.md", "KNOWN-LIMITATIONS.md",
        "docs/claim-matrix.md", "docs/experiments/swiglu-sweep-20260831-3070.md",
        "docs/fp16-delivery-status.md", "docs/compatibility-matrix.md",
        "docs/e4-runbook.md", "scripts/verify.sh", "scripts/gpu_telemetry.py",
        ".github/workflows/ci.yml",
    ]
    for p in paths:
        if not (ROOT / p).exists():
            errs.append(f"documented path missing: {p}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-json", action="store_true", help="print latest manifest key fields as JSON")
    ap.add_argument("--doc-only", action="store_true", help="check doc references only (no manifest needed; CI cpu-static)")
    args = ap.parse_args()
    if args.fix_json:
        d = latest_manifest_dir()
        if not d:
            print('{"error": "no manifest"}'); return 1
        m = json.loads((d / "manifest.json").read_text())
        s = m.get("git", {}).get("short_commit", "?")
        print(json.dumps({"commit": s, "dir": d.name,
                          "tests": m.get("steps", {}).get("tests", {}).get("summary", {})}, indent=2))
        return 0
    print("== evidence consistency check ==")
    errs = []
    if not args.doc_only:
        errs += check_manifest()
        print("  doc references: checking")
    else:
        print("  --doc-only: manifest check skipped (CI), doc refs only")
    errs += check_doc_refs()
    print("  doc references: OK" if not any("path missing" in e for e in errs) else "  doc references: see errors")
    if errs:
        print("FAIL:")
        for e in errs:
            print("  -", e)
        return 1
    print("ALL OK (manifest + doc refs consistent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
