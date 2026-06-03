#!/bin/bash
# Full evaluation driver:
#   1. GPU 预热 60s (tools/gpu_warmup.py)
#   2. 4 轮重复 bench_cutile / benchmark_ops / run_compare
#   3. 取后 3 轮均值, 输出 results/full_eval_<ts>/summary.json + 控制台 markdown 汇总
#
# 用法:
#   bash tools/run_full_eval.sh
#   bash tools/run_full_eval.sh --rounds 4 --take-last 3   # 默认
#   bash tools/run_full_eval.sh --rounds 3 --take-last 2   # 减
#   bash tools/run_full_eval.sh --skip-compare             # 跳 15-epoch 端到端, ~5x 加速
#
# 输出: results/full_eval_<ts>/round_{1..N}_cutile.json + _ops.json + _compare.json
#       + summary.json (后 3 轮均值) + summary.md (人类可读)

set -euo pipefail

source /home/virusoid/projects/venv/bin/activate
cd /home/virusoid/projects/MLP-Kernel-Lab

# ---------- 参数 ----------
ROUNDS=${ROUNDS:-4}
TAKE_LAST=${TAKE_LAST:-3}
WARMUP_SECS=${WARMUP_SECS:-60}
SKIP_COMPARE=0
MODELS=${MODELS:-default}
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/full_eval_${TS}"

# 把 model 名字映射到 yaml 路径
declare -A MODEL_YAML
MODEL_YAML[default]="configs/mnist_mlp.yaml"
MODEL_YAML[deep_narrow]="configs/mnist_mlp_deep_narrow.yaml"
MODEL_YAML[wide_skip]="configs/mnist_mlp_wide_skip.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds)        ROUNDS=$2; shift 2;;
        --take-last)     TAKE_LAST=$2; shift 2;;
        --warmup-secs)   WARMUP_SECS=$2; shift 2;;
        --skip-compare)  SKIP_COMPARE=1; shift;;
        --out-dir)       OUT_DIR=$2; shift 2;;
        --models)        MODELS=$2; shift 2;;
        *) echo "unknown arg $1" >&2; exit 2;;
    esac
done

# 校验 model 名字
for m in $(echo "$MODELS" | tr ',' ' '); do
    if [[ -z "${MODEL_YAML[$m]:-}" ]]; then
        echo "unknown model '$m' (valid: ${!MODEL_YAML[*]})" >&2
        exit 2
    fi
done

mkdir -p "$OUT_DIR"
echo "== Full Eval =="
echo "  rounds=$ROUNDS  take_last=$TAKE_LAST  warmup=${WARMUP_SECS}s  out=$OUT_DIR"
echo "  skip_compare=$SKIP_COMPARE"
echo

# ---------- 1. GPU 预热 ----------
echo "[1/3] GPU warmup (${WARMUP_SECS}s) ..."
python tools/gpu_warmup.py --seconds "$WARMUP_SECS" --matmul 2048
echo

# ---------- 2. 4 轮 op 评测 ----------
echo "[2/3] per-op benchmark x $ROUNDS rounds ..."
for r in $(seq 1 $ROUNDS); do
    echo "  --- op round $r/$ROUNDS ---"
    python benchmark_ops.py \
        --sizes medium --dtypes fp32 --roofline \
        --warmup 20 --iters 100 \
        --output "$OUT_DIR/round_${r}_ops.json" 2>&1 | tail -2
done

echo "  --- cuTile bench x $ROUNDS rounds ---"
for r in $(seq 1 $ROUNDS); do
    echo "  --- cuTile round $r/$ROUNDS ---"
    # 跑默认 4 rounds 取 3 mean, 但每 round 自身已经采 4 轮 = 4*4=16 个内部测点.
    # 外部 ROUNDS=4 让 driver 自己再 wrap 4 轮, 即每 op 64 个测点 (极稳).
    python profiling/bench_cutile.py --output "$OUT_DIR/round_${r}_cutile.json" 2>&1 | tail -2
done
echo

# ---------- 3. 4 轮 end-to-end (可选) ----------
if [[ $SKIP_COMPARE -eq 0 ]]; then
    for model in $(echo "$MODELS" | tr ',' ' '); do
        CONFIG="${MODEL_YAML[$model]}"
        MODEL_OUT="$OUT_DIR/model_${model}"
        mkdir -p "$MODEL_OUT"
        echo "[3/3] model=$model  config=$CONFIG  end-to-end 4-backend x $ROUNDS rounds ..."
        for r in $(seq 1 $ROUNDS); do
            echo "  --- $model compare round $r/$ROUNDS ---"
            python run_compare.py --config "$CONFIG" --cuda --cutile \
                --precision "${COMPARE_PRECISION:-fp32}" --epochs 15 \
                2>&1 | tail -3
        done
        # 复制该 model 最后 1 份 compare 进 MODEL_OUT
        LATEST_ONE=$(ls -t results/compare_*.json 2>/dev/null | head -1)
        if [[ -n "$LATEST_ONE" ]]; then
            cp "$LATEST_ONE" "$MODEL_OUT/last_compare.json"
        fi
    done
else
    echo "[3/3] end-to-end skipped (--skip-compare)"
fi
echo

# ---------- 4. 汇总 (后 3 轮均值) ----------
echo "== Summary (mean of last $TAKE_LAST of $ROUNDS) =="
python - << PYEOF
import json
import statistics
import sys
from pathlib import Path

out_dir = Path("$OUT_DIR")
rounds = $ROUNDS
take_last = $TAKE_LAST

# 找 round 1..N 的 JSON
op_files = sorted(out_dir.glob("round_*_ops.json"))
cu_files = sorted(out_dir.glob("round_*_cutile.json"))
print(f"  ops:     {len(op_files)} rounds")
print(f"  cuTile:  {len(cu_files)} rounds")

# 抽出后 N 轮
def read_jsonl_files(files, take_last):
    rows = []
    for f in files[-take_last:]:
        try:
            d = json.load(open(f))
            rows.append(d)
        except Exception as e:
            print(f"  warn: skip {f}: {e}")
    return rows

ops = read_jsonl_files(op_files, take_last)
cutile = read_jsonl_files(cu_files, take_last)

# 汇总 per-op 关键指标
def stat(records, key):
    vals = []
    for r in records:
        rows = r.get("rows", [])
        for row in rows:
            v = row.get(key, 0)
            if v > 0:
                vals.append(v)
    if not vals: return None
    return statistics.mean(vals)

def stat_per_op(records, key):
    """按 op 分组, 取后 N 轮均值"""
    agg = {}
    for r in records:
        for row in r.get("rows", []):
            op = row.get("name", "?")
            v = row.get(key, 0)
            if v <= 0: continue
            agg.setdefault(op, []).append(v)
    return {op: statistics.mean(v) for op, v in agg.items()}

# Print op perf
print()
header_metric = "median_ms" if (op_files and "median_ms" in json.load(open(op_files[0])).get("rows", [{}])[0]) else "p50_ms"
print(f"{'op':<26}{'backend':<10}{'mean '+header_metric}")
for backend in ("pytorch","triton","cuda"):
    means = stat_per_op(ops, f"{backend}_ms")
    if not means: continue
    for op, ms in sorted(means.items(), key=lambda t: -t[1])[:5]:
        print(f"  {op:<24} {backend:<10} {ms:8.4f} ms")

# cuTile
if cutile:
    print()
    print("cuTile (per-op mean over last N rounds):")
    agg = {}
    for r in cutile:
        for k, v in (r.get("ops") or {}).items():
            if not isinstance(v, dict): continue
            x = v.get("mean_last_n_ms", v.get("mean_last_3_ms", 0))
            if x > 0:
                agg.setdefault(k, []).append(x)
    for k, vs in sorted(agg.items(), key=lambda t: -t[1])[:5]:
        print(f"  {k:<24} {statistics.mean(vs):8.4f} ms")

# Write summary JSON
summary = {
    "rounds": rounds,
    "take_last": take_last,
    "warmup_secs": $WARMUP_SECS,
    "op_files": [str(f) for f in op_files],
    "cuTile_files": [str(f) for f in cu_files],
    "ops_p50_mean_per_op": stat_per_op(ops, "pytorch_ms"),
    "ops_triton_mean_per_op": stat_per_op(ops, "triton_ms"),
    "ops_cuda_mean_per_op": stat_per_op(ops, "cuda_ms"),
}
out = out_dir / "summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"\nSummary saved: {out}")
PYEOF

echo
echo "== Done. Artifacts under: $OUT_DIR =="
ls -la "$OUT_DIR"
