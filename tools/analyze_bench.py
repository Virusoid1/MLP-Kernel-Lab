"""Compare two benchmark JSON files; emit a markdown diff and optionally gate.

Usage:
    python tools/analyze_bench.py BASELINE.json CANDIDATE.json
    python tools/analyze_bench.py BASELINE.json CANDIDATE.json --shape MLP_LAYERS --gate
    python tools/analyze_bench.py base.json cand.json --metric tflops --tolerance 0.10

Supports both schemas:
    - benchmark_ops.py: {metadata, rows: [{name, shapes, dtype, *_ms, *_l2_err, ...}]}
    - profiling/bench_cutile.py: {metadata, ops: {op_name: {mean_last_n_ms,...}}, mlp: {...}}

Gate rules (default tolerance 0.05 / 5%):
    perf REGRESS         : candidate {metric} > baseline * (1 + tol)
    perf IMPROVE         : candidate {metric} < baseline * (1 - tol)
    CORRECTNESS_FAIL     : baseline l2_err < warn_l2 and candidate l2_err > warn_l2
                           OR baseline max_abs < warn_maxabs and candidate max_abs > warn_maxabs
    MISSING              : key in baseline absent in candidate
With --gate, any REGRESS / CORRECTNESS_FAIL / MISSING --> exit non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# --- MLP_LAYERS shape filter (MNIST [784,1024,512,256,10] 真实 weight 形状) ---
MLP_LAYER_SHAPES = {
    "(784,1024)", "(1024,512)", "(512,256)", "(256,10)",
    "(784,1024)@(1024,512)", "(1024,512)@(512,256)", "(512,256)@(256,10)",
}

# 当 candidate row 的 shapes 串包含以下之一,视为 MLP_LAYERS
_MLP_LAYERS_PATTERNS = ["(784,", "(1024,512)", "(512,256)", "(256,10)"]

STATUS_REGRESS = "REGRESS"
STATUS_OK = "OK"
STATUS_IMPROVE = "IMPROVE"
STATUS_MISSING = "MISSING"
STATUS_CORRECTNESS_FAIL = "CORRECTNESS_FAIL"

_FAIL_STATUSES = {STATUS_REGRESS, STATUS_CORRECTNESS_FAIL}


@dataclass
class Row:
    """Normalized comparison row across both schemas."""
    op: str
    backend: str
    size: str
    dtype: str
    baseline_metric: float
    candidate_metric: float
    delta_pct: float
    status: str
    metric_name: str
    baseline_l2: float | None = None
    candidate_l2: float | None = None
    baseline_maxabs: float | None = None
    candidate_maxabs: float | None = None
    note: str = ""

    def as_md(self) -> str:
        b = f"{self.baseline_metric:.4f}" if self.baseline_metric > 0 else "-"
        c = f"{self.candidate_metric:.4f}" if self.candidate_metric > 0 else "-"
        d = f"{self.delta_pct:+.1f}%" if self.baseline_metric > 0 else "n/a"
        return (f"| {self.op} | {self.backend} | {self.size} | {self.dtype} "
                f"| {b} | {c} | {d} | {self.status} | {self.note} |")


# --- JSON schema detection / normalization ----------------------------------

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"benchmark JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_schema(data: Any) -> str:
    """Return 'op_bench' | 'cutile_bench' | 'list_legacy' | 'unknown'."""
    if isinstance(data, list):
        return "list_legacy"  # old benchmark_ops dumped a bare list
    if isinstance(data, dict):
        if "rows" in data:
            return "op_bench"
        if "ops" in data and "mlp" in data:
            return "cutile_bench"
    return "unknown"


def extract_metadata(data: Any) -> dict:
    if isinstance(data, dict):
        return data.get("metadata", {}) or {}
    return {}


def normalize_rows(data: Any) -> list[dict]:
    """Both schemas -> a uniform list of dicts (one per (op, backend, size, dtype))."""
    schema = detect_schema(data)
    out: list[dict] = []

    if schema == "op_bench":
        rows = data.get("rows", [])
        for r in rows:
            base = {
                "op": r.get("name", "?"),
                "size": r.get("shapes", "?"),
                "dtype": r.get("dtype", "fp32"),
            }
            for backend in ("pytorch", "triton", "cuda", "cutile"):
                ms = r.get(f"{backend}_ms", 0.0)
                if ms <= 0:
                    continue
                out.append({
                    **base,
                    "backend": backend,
                    "median_ms": ms,
                    "p95_ms": r.get(f"{backend}_p95_ms", 0.0),
                    "tflops": r.get(f"{backend}_tflops", 0.0),
                    "gbps": r.get(f"{backend}_gbps", 0.0),
                    "l2_err": r.get(f"{backend}_l2_err", 0.0) if backend != "pytorch" else 0.0,
                    "max_err": r.get(f"{backend}_max_err", 0.0) if backend != "pytorch" else 0.0,
                })

    elif schema == "list_legacy":
        for r in data:
            base = {
                "op": r.get("name", "?"),
                "size": r.get("shapes", "?"),
                "dtype": r.get("dtype", "fp32"),
            }
            for backend in ("pytorch", "triton", "cuda", "cutile"):
                ms = r.get(f"{backend}_ms", 0.0)
                if ms <= 0:
                    continue
                out.append({
                    **base,
                    "backend": backend,
                    "median_ms": ms,
                    "p95_ms": r.get(f"{backend}_p95_ms", 0.0),
                    "tflops": 0.0, "gbps": 0.0,
                    "l2_err": r.get(f"{backend}_l2_err", 0.0) if backend != "pytorch" else 0.0,
                    "max_err": r.get(f"{backend}_max_err", 0.0) if backend != "pytorch" else 0.0,
                })

    elif schema == "cutile_bench":
        ops = data.get("ops", {})
        cfg = data.get("metadata", data.get("config", {}))
        shape = f"M={cfg.get('M','?')} K={cfg.get('K','?')} N={cfg.get('N','?')}"
        for op_name, info in ops.items():
            if not isinstance(info, dict):
                continue
            out.append({
                "op": op_name,
                "size": shape,
                "dtype": "fp32",
                "backend": "cutile",
                "median_ms": info.get("mean_last_n_ms", info.get("mean_last_3_ms", 0.0)),
                "p95_ms": 0.0,
                "tflops": 0.0, "gbps": 0.0,
                "l2_err": 0.0, "max_err": 0.0,
            })
        mlp = data.get("mlp", {})
        for op_name, info in mlp.items():
            if not isinstance(info, dict):
                continue
            out.append({
                "op": op_name,
                "size": "MLP_end_to_end",
                "dtype": "fp32",
                "backend": "cutile",
                "median_ms": info.get("mean_last_n_ms", info.get("mean_last_3_ms", 0.0)),
                "p95_ms": 0.0,
                "tflops": 0.0, "gbps": 0.0,
                "l2_err": 0.0, "max_err": 0.0,
            })

    return out


def row_key(r: dict) -> tuple:
    return (r["op"], r["backend"], r["size"], r["dtype"])


# --- shape filter ------------------------------------------------------------

def shape_matches(size_str: str, shape_filter: str) -> bool:
    if shape_filter == "all":
        return True
    if shape_filter == "MLP_LAYERS":
        return any(p in size_str for p in _MLP_LAYERS_PATTERNS)
    if shape_filter == "MLP_FWD":
        return any(p in size_str for p in ("784", "1024", "512", "256"))
    return True


# --- metadata mismatch check -------------------------------------------------

def check_metadata(baseline_md: dict, candidate_md: dict, force: bool) -> list[str]:
    """Return list of warning strings; if non-empty + not force, gate must abort."""
    warnings: list[str] = []
    for key in ("gpu", "torch", "driver"):
        b = baseline_md.get(key)
        c = candidate_md.get(key)
        if b and c and b != c:
            warnings.append(f"metadata.{key} mismatch: baseline={b!r} candidate={c!r}")
    return warnings


# --- gate logic --------------------------------------------------------------

def compare(
    baseline: list[dict], candidate: list[dict], *,
    metric: str, tolerance: float, warn_l2: float, warn_maxabs: float,
    shape_filter: str,
) -> list[Row]:
    b_map = {row_key(r): r for r in baseline if shape_matches(r["size"], shape_filter)}
    c_map = {row_key(r): r for r in candidate if shape_matches(r["size"], shape_filter)}

    rows: list[Row] = []
    all_keys = sorted(set(b_map) | set(c_map))

    for key in all_keys:
        op, backend, size, dtype = key
        b = b_map.get(key)
        c = c_map.get(key)

        b_val = float(b.get(metric, 0.0)) if b else 0.0
        c_val = float(c.get(metric, 0.0)) if c else 0.0

        # status
        if not b and not c:
            continue
        if b and not c:
            status = STATUS_MISSING
            note = "absent in candidate"
            delta = 0.0
        elif c and not b:
            status = STATUS_MISSING
            note = "absent in baseline"
            delta = 0.0
        else:
            # both exist
            if b_val <= 0 or c_val <= 0:
                status = STATUS_MISSING
                note = f"{metric}=0 in one side"
                delta = 0.0
            else:
                # for ms metric: bigger candidate = REGRESS; for tflops/gbps: bigger = IMPROVE
                bigger_is_better = metric in ("tflops", "gbps")
                ratio = c_val / b_val
                delta = (ratio - 1.0) * 100.0
                if bigger_is_better:
                    if ratio < 1 - tolerance:
                        status = STATUS_REGRESS
                    elif ratio > 1 + tolerance:
                        status = STATUS_IMPROVE
                    else:
                        status = STATUS_OK
                else:
                    if ratio > 1 + tolerance:
                        status = STATUS_REGRESS
                    elif ratio < 1 - tolerance:
                        status = STATUS_IMPROVE
                    else:
                        status = STATUS_OK
                note = ""

            # correctness gate (overrides perf status if numerically broken)
            b_l2 = float(b.get("l2_err", 0.0)) if b else 0.0
            c_l2 = float(c.get("l2_err", 0.0)) if c else 0.0
            b_max = float(b.get("max_err", 0.0)) if b else 0.0
            c_max = float(c.get("max_err", 0.0)) if c else 0.0
            if (b_l2 < warn_l2 and c_l2 >= warn_l2) or (b_max < warn_maxabs and c_max >= warn_maxabs):
                status = STATUS_CORRECTNESS_FAIL
                note = f"l2 {b_l2:.1e}->{c_l2:.1e} maxabs {b_max:.1e}->{c_max:.1e}"

        rows.append(Row(
            op=op, backend=backend, size=size, dtype=dtype,
            baseline_metric=b_val, candidate_metric=c_val, delta_pct=delta,
            status=status, metric_name=metric,
            baseline_l2=float(b.get("l2_err", 0.0)) if b else None,
            candidate_l2=float(c.get("l2_err", 0.0)) if c else None,
            baseline_maxabs=float(b.get("max_err", 0.0)) if b else None,
            candidate_maxabs=float(c.get("max_err", 0.0)) if c else None,
            note=note,
        ))

    # Sort: regressions first, then correctness fail, then improve, then ok
    order = {STATUS_REGRESS: 0, STATUS_CORRECTNESS_FAIL: 1, STATUS_MISSING: 2,
             STATUS_IMPROVE: 3, STATUS_OK: 4}
    rows.sort(key=lambda r: (order.get(r.status, 9), -abs(r.delta_pct)))
    return rows


# --- markdown emission -------------------------------------------------------

def print_table(rows: list[Row], metric: str) -> None:
    if not rows:
        print("_(no rows match filter)_")
        return
    print(f"\n| op | backend | size | dtype | baseline {metric} | candidate {metric} | delta% | status | note |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(r.as_md())
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"\n**Summary**: {summary}")


# --- main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--shape", default="all", choices=["all", "MLP_LAYERS", "MLP_FWD"])
    ap.add_argument("--metric", default="median_ms",
                    choices=["median_ms", "p95_ms", "tflops", "gbps"])
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="Relative threshold (e.g. 0.05 = 5%%)")
    ap.add_argument("--warn-l2", type=float, default=1e-3)
    ap.add_argument("--warn-maxabs", type=float, default=1e-2)
    ap.add_argument("--gate", action="store_true",
                    help="Exit non-zero on REGRESS / CORRECTNESS_FAIL / MISSING")
    ap.add_argument("--force-baseline-update", action="store_true",
                    help="Ignore metadata mismatch warnings")
    ap.add_argument("--no-correctness-gate", action="store_true",
                    help="Disable l2/max_abs correctness gating")
    args = ap.parse_args()

    b_data = load_json(args.baseline)
    c_data = load_json(args.candidate)
    print(f"# Bench diff: {args.baseline.name}  ->  {args.candidate.name}")
    print(f"_metric={args.metric}  shape={args.shape}  tolerance={args.tolerance:.0%}_")

    md_b = extract_metadata(b_data)
    md_c = extract_metadata(c_data)
    warns = check_metadata(md_b, md_c, args.force_baseline_update)
    if warns:
        print("\n**Metadata warnings:**")
        for w in warns:
            print(f"- {w}")
        if args.gate and not args.force_baseline_update:
            print("\nRefusing to gate due to metadata mismatch (pass --force-baseline-update to override).")
            return 2

    rows = compare(
        normalize_rows(b_data), normalize_rows(c_data),
        metric=args.metric, tolerance=args.tolerance,
        warn_l2=args.warn_l2, warn_maxabs=args.warn_maxabs,
        shape_filter=args.shape,
    )
    if args.no_correctness_gate:
        for r in rows:
            if r.status == STATUS_CORRECTNESS_FAIL:
                r.status = STATUS_OK

    print_table(rows, args.metric)

    if args.gate:
        has_fail = any(r.status in _FAIL_STATUSES or r.status == STATUS_MISSING for r in rows)
        return 1 if has_fail else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
