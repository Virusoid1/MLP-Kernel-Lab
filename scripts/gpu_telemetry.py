#!/usr/bin/env python3
"""GPU 热稳定/节流采集（v2 计划 3.5）。
在 benchmark 运行期间后台采样 nvidia-smi，记录温度/时钟/功耗/利用率。
用法:
    python scripts/gpu_telemetry.py --interval 2 --seconds 30
    python scripts/gpu_telemetry.py --cmd "python bench/run.py --suite prefill --dtypes fp16"
输出: artifacts/gpu_telemetry_<ts>.csv / .json
"""
import argparse, csv, json, subprocess, sys, time
from pathlib import Path

FIELDS = ["temperature.gpu", "clocks.gr", "clocks.max.gr", "clocks.mem",
          "power.draw", "power.limit", "utilization.gpu", "utilization.memory"]

def sample():
    q = ",".join(FIELDS)
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        vals = out.stdout.strip().splitlines()[0].split(",")
        d = {}
        for f, v in zip(FIELDS, vals):
            v = v.strip()
            d[f] = float(v) if v != "[N/A]" else None
        return d
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--cmd", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = args.out or ("artifacts/gpu_telemetry_" + ts)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.time()
    proc = None
    if args.cmd:
        proc = subprocess.Popen(args.cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = start + (args.seconds if not args.cmd else 6 * 3600)
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            break
        s = sample()
        if s:
            s["t_rel"] = round(time.time() - start, 1)
            rows.append(s)
        time.sleep(args.interval)
    if proc is not None:
        proc.wait()
    if not rows:
        print("no samples (nvidia-smi unavailable?)", file=sys.stderr)
        return 1
    with open(prefix + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    temps = [r["temperature.gpu"] for r in rows if r["temperature.gpu"] is not None]
    clocks = [r["clocks.gr"] for r in rows if r["clocks.gr"] is not None]
    powers = [r["power.draw"] for r in rows if r["power.draw"] is not None]
    utils = [r["utilization.gpu"] for r in rows if r["utilization.gpu"] is not None]
    summary = {
        "samples": len(rows),
        "max_temp_c": max(temps) if temps else None,
        "min_clock_mhz": min(clocks) if clocks else None,
        "max_clock_mhz": max(clocks) if clocks else None,
        "throttled": bool(clocks and min(clocks) < 0.8 * max(clocks)) if max(clocks) else None,
        "max_power_w": max(powers) if powers else None,
        "max_util_pct": max(utils) if utils else None,
    }
    with open(prefix + ".json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("saved: " + prefix + ".csv / .json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
