#!/usr/bin/env python3
"""一键复现入口（make reproduce 的驱动）：构建 -> 55 测试 -> bench -> manifest 汇总。

流程（每个步骤失败即退出非零，便于 gate）:
  1. preflight : 收集环境/git/依赖版本（复用 python.mnist.benchmark.capture_metadata + 扩展）
  2. build     : setup.py build_ext --inplace（CUDA extension 动态架构，见 setup.py）
  3. tests     : pytest tests/ --junitxml -> 解析 pass/fail/skip（真实用例数，不硬编码）
  4. bench     : benchmark_ops.py（默认 --sizes small --warmup 5 --iters 20）
  5. summary   : 写 manifest.json / correctness.jsonl / benchmark.json / summary.md / env.lock.txt

用法:
    python tools/reproduce.py                 # 全流程，输出到 artifacts/<run-id>/
    python tools/reproduce.py --quick         # 同默认（bench 已是 small smoke）
    python tools/reproduce.py --skip-build    # 跳过 CUDA extension 构建（用已装 .so）
    python tools/reproduce.py --test-only     # 只跑测试 + manifest（不构建不 bench）
    python tools/reproduce.py --out DIR       # 指定输出目录（默认 artifacts/<run-id>）
    python tools/reproduce.py -- ARGS         # 传给 benchmark_ops.py 的额外参数

输出结构:
    artifacts/<YYYYMMDD-HHMMSS-<commit>-<gpu>>/
        manifest.json         # 环境 + git + 每步状态 + 产 files
        correctness.jsonl     # 每条测试用例一行 {name, status, duration}
        benchmark.json        # benchmark_ops.py 原始输出（{metadata, rows}）
        summary.md            # 人类可读汇总表
        environment.lock.txt  # pip freeze + nvcc/driver 摘要
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """执行命令并捕获输出；cwd 默认为仓库根目录。"""
    kw.setdefault("cwd", str(REPO_ROOT))
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    kw.setdefault("timeout", 3600)
    return subprocess.run(cmd, **kw)


def git_info() -> dict:
    """git commit / branch / dirty。"""
    out = {"commit": "unknown", "short_commit": "unknown", "branch": "unknown", "dirty": True}
    for key, argv in {
        "commit": ["git", "rev-parse", "HEAD"],
        "short_commit": ["git", "rev-parse", "--short", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    }.items():
        r = run(argv)
        if r.returncode == 0:
            out[key] = r.stdout.strip()
    r = run(["git", "status", "--porcelain"])
    out["dirty"] = not (r.returncode == 0 and not r.stdout.strip())
    return out


def env_manifest(args) -> dict:
    """复用 capture_metadata 并叠加 git / 构建相关字段。"""
    from python.mnist.benchmark import capture_metadata
    md = capture_metadata(args)
    md["git"] = git_info()
    return md


def _cuda_toolchain_env() -> dict:
    """探测并返回能匹配 torch 的 CUDA 工具链环境（PATH/CUDA_HOME）。

    若环境中 nvcc 的主版本与 torch.version.cuda 不一致（例如 apt 装了 12.0、
    torch 要 13.0），则优先 /usr/local/cuda*/bin 里的新版 nvcc。
    找不到就原样返回（让 setup.py 报错并给出明确信息）。
    """
    base = dict(os.environ)
    try:
        import torch
    except Exception:
        return base
    want = (getattr(torch.version, "cuda", None) or "").split(".")[0]
    if not want:
        return base
    # 当前 nvcc 主版本
    cur = None
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
        m = re.search(r"release\s+(\d+)\.", r.stdout)
        if m:
            cur = m.group(1)
    except Exception:
        pass
    if cur == want:
        return base
    # 在 /usr/local/cuda* 里找匹配的
    import glob as _glob
    for d in sorted(_glob.glob("/usr/local/cuda*"), reverse=True):
        nvcc = os.path.join(d, "bin", "nvcc")
        if not os.path.exists(nvcc):
            continue
        try:
            r = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=10)
            m = re.search(r"release\s+(\d+)\.", r.stdout)
            if m and m.group(1) == want:
                base["PATH"] = os.path.join(d, "bin") + os.pathsep + base.get("PATH", "")
                base["CUDA_HOME"] = d
                base["CUDA_PATH"] = d
                print(f"[reproduce] selected CUDA {want} toolchain: {d} (was nvcc {cur})")
                return base
        except Exception:
            continue
    print(f"[reproduce] WARNING: found no CUDA {want} nvcc in /usr/local; using default PATH (may fail)")
    return base


def step_build(env: dict) -> dict:
    """构建 CUDA extension（inplace），自动匹配 CUDA 工具链版本。"""
    r = run([sys.executable, "setup.py", "build_ext", "--inplace"], env=env)
    return {"command": "setup.py build_ext --inplace", "ok": r.returncode == 0,
            "stdout_tail": r.stdout[-1200:], "stderr_tail": r.stderr[-1200:]}


def step_tests() -> dict:
    """pytest：用 junitxml 得到真实用例数与状态分布。"""
    junit = REPO_ROOT / "artifacts" / f"_junit_{int(time.time())}.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    r = run([sys.executable, "-m", "pytest", "tests/", "-v",
             "--junitxml", str(junit), "-q"])
    tests, failures, errors, skipped = 0, 0, 0, 0
    cases: list[dict] = []
    if junit.exists():
        root = ET.parse(junit).getroot()
        for ts in root.iter("testsuite"):
            tests += int(ts.get("tests", 0))
            failures += int(ts.get("failures", 0))
            errors += int(ts.get("errors", 0))
            skipped += int(ts.get("skipped", 0))
        for tc in root.iter("testcase"):
            status = "passed"
            if tc.find("failure") is not None:
                status = "failed"
            elif tc.find("error") is not None:
                status = "error"
            elif tc.find("skipped") is not None:
                status = "skipped"
            cases.append({
                "name": tc.get("classname", "") + "::" + tc.get("name", ""),
                "status": status,
                "time": float(tc.get("time", 0.0)),
            })
    junit.unlink(missing_ok=True)
    return {
        "command": "pytest tests/ -v --junitxml",
        "exit_code": r.returncode,
        "ok": r.returncode == 0,
        "summary": {"tests": tests, "failures": failures, "errors": errors,
                    "passed": tests - failures - errors - skipped, "skipped": skipped},
        "cases": cases,
    }


def step_bench(extra_args: list[str]) -> dict:
    """benchmark_ops.py 单次运行（默认 small smoke），返回其 JSON。"""
    out = REPO_ROOT / "artifacts" / f"_bench_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "benchmark_ops.py", "--sizes", "small",
           "--warmup", "5", "--iters", "20", "--output", str(out)] + extra_args
    r = run(cmd)
    data = None
    if out.exists():
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            data = None
        out.unlink(missing_ok=True)
    return {"command": " ".join(cmd), "exit_code": r.returncode, "ok": r.returncode == 0,
            "rows": len(data.get("rows", [])) if data else 0, "data": data}


def write_summary(run_dir: Path, manifest: dict, tests: dict, bench: dict, skip_bench: bool) -> str:
    t = tests["summary"]
    md = [f"# Reproduce Report — {run_dir.name}", ""]
    md.append(f"- git: {manifest['git']['short_commit']} ({manifest['git']['branch']}, dirty={manifest['git']['dirty']})")
    md.append(f"- gpu: {manifest['gpu']['name']} (cc {manifest['gpu']['cc']}, {manifest['gpu']['vram_gb']} GB)")
    md.append(f"- driver: {manifest['driver']} | cuda: {manifest['cuda']}")
    md.append(f"- torch {manifest['torch']} / triton {manifest['triton']} / cutile {manifest['cutile']} / python {manifest['python']}")
    md.append("")
    md.append("## Tests")
    md.append(f"- **{t['tests']} tests**: {t['passed']} passed, {t['failures']} failed, {t['errors']} errors, {t['skipped']} skipped")
    md.append("")
    if t["failures"] or t["errors"]:
        md.append("### Failures")
        for c in tests["cases"]:
            if c["status"] in ("failed", "error"):
                md.append(f"- {c['name']} [{c['status']}] ({c['time']:.2f}s)")
        md.append("")
    md.append("## Benchmark (small smoke)")
    if skip_bench or bench.get("data") is None:
        md.append("- (skipped / failed)")
    else:
        md.append(f"- rows: {bench['rows']}, backend-fields: pytorch/triton/cuda(+cutile if available)")
        first = bench["data"]["rows"][0] if bench["data"]["rows"] else {}
        md.append(f"- sample row: {json.dumps(first, ensure_ascii=False)}")
    md.append("")
    md.append("## Files")
    for p in sorted(run_dir.iterdir()):
        md.append(f"- {p.name} ({p.stat().st_size} bytes)")
    md.append("")
    return "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser(description="一键复现：build -> tests -> bench -> manifest")
    ap.add_argument("--skip-build", action="store_true", help="跳过 CUDA extension 构建")
    ap.add_argument("--test-only", action="store_true", help="只跑测试 + manifest")
    ap.add_argument("--out", default=None, help="输出目录（默认 artifacts/<run-id>）")
    ap.add_argument("extra", nargs="*", help="透传给 benchmark_ops.py 的参数")
    args = ap.parse_args()

    git = git_info()
    short = git["short_commit"]
    gpu_slug = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).replace(" ", "_")
            gpu_slug = re.sub(r"[^A-Za-z0-9_]", "", name)
    except Exception:
        pass
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{short}-{gpu_slug}"
    out_root = Path(args.out) if args.out else (REPO_ROOT / "artifacts" / run_id)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[reproduce] run-id: {run_id}")
    print(f"[reproduce] output: {out_root}")

    manifest = env_manifest(args)
    steps: dict = {}

    # 1. build
    if args.test_only:
        steps["build"] = {"ok": True, "skipped": "test-only"}
    elif args.skip_build:
        steps["build"] = {"ok": True, "skipped": "user-flag"}
    else:
        print("[reproduce] step 1/3: build CUDA extension ...")
        steps["build"] = step_build(_cuda_toolchain_env())
        if not steps["build"]["ok"]:
            print("[reproduce] BUILD FAILED", file=sys.stderr)
            print(steps["build"]["stderr_tail"], file=sys.stderr)
            manifest["steps"] = steps
            (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            return 1
        print("[reproduce] build OK")

    # 2. tests
    print("[reproduce] step 2/3: pytest ...")
    steps["tests"] = step_tests()
    print(f"[reproduce] tests: {steps['tests']['summary']}")
    (out_root / "correctness.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in steps["tests"]["cases"]),
        encoding="utf-8")

    # 3. bench
    if args.test_only:
        steps["bench"] = {"ok": True, "skipped": "test-only"}
    else:
        print("[reproduce] step 3/3: benchmark_ops smoke ...")
        steps["bench"] = step_bench(args.extra)
        print(f"[reproduce] bench rows: {steps['bench']['rows']}")
        if steps["bench"]["data"] is not None:
            (out_root / "benchmark.json").write_text(
                json.dumps(steps["bench"]["data"], indent=2, ensure_ascii=False), encoding="utf-8")

    manifest["steps"] = steps
    manifest["artifacts_dir"] = str(out_root)

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # env lock
    env_lock = []
    r = run([sys.executable, "-m", "pip", "list", "--format=freeze"])
    env_lock.append("# pip freeze (filtered, key packages)")
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        low = line.lower()
        if any(k in low for k in ("torch", "triton", "numpy", "pytest", "cuda-tile", "tilelang")):
            env_lock.append(line)
    env_lock.append("")
    env_lock.append("# nvcc / driver")
    r2 = run(["nvcc", "--version"])
    env_lock.append((r2.stdout.strip().splitlines()[-1] if r2.stdout.strip() else "nvcc not found"))
    (out_root / "environment.lock.txt").write_text("\n".join(env_lock), encoding="utf-8")

    # summary
    (out_root / "summary.md").write_text(
        write_summary(out_root, manifest, steps["tests"], steps.get("bench", {}), args.test_only),
        encoding="utf-8")

    print(f"[reproduce] manifest -> {out_root / 'manifest.json'}")
    print(f"[reproduce] summary  -> {out_root / 'summary.md'}")
    t = steps["tests"]["summary"]
    print(f"[reproduce] RESULT: {t['passed']}/{t['tests']} passed ({t['failures']} failed, {t['errors']} errors, {t['skipped']} skipped)")
    if t["failures"] or t["errors"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
