"""Ad-hoc analysis of baseline.json (闭环演练 step 2)."""
import json
from pathlib import Path

d = json.load(open("results/baseline.json"))
rows = d["rows"]
md = d["metadata"]

print(f"## GPU: {md['gpu']['name']} (cc={md['gpu']['cc']}), torch={md['torch']}, git={md['git_sha']}")
print(f"## {len(rows)} rows, allow_tf32_matmul={md['allow_tf32_matmul']}\n")


# ============================================================
# 角度 1: TFLOPS 排名 (matmul-class only)
# ============================================================
print("### 角度 1: 计算密集型 TFLOPS 排名 (matmul-class only)\n")
print(f"{'op':<26}{'shape':<32}{'PT':>9}{'Tr':>9}{'CU':>9}  TFLOPS")
print("-" * 92)
mr = [r for r in rows if r.get("flops", 0) > 0 and "matmul" in r["name"]]
mr.sort(key=lambda r: -max(r.get("pytorch_tflops", 0), r.get("triton_tflops", 0), r.get("cuda_tflops", 0)))
for r in mr:
    print(f"{r['name']:<26}{r['shapes']:<32}"
          f"{r.get('pytorch_tflops', 0):>9.1f}"
          f"{r.get('triton_tflops', 0):>9.1f}"
          f"{r.get('cuda_tflops', 0):>9.1f}")
print()


# ============================================================
# 角度 2: CUDA 比 PyTorch 慢的 op (优化目标)
# ============================================================
print("### 角度 2: CUDA backend 比 PyTorch 慢的 op (优化目标)\n")
slow = [(r["name"], r["shapes"], r.get("cuda_speedup", 0))
        for r in rows
        if 0 < r.get("cuda_speedup", 0) < 1.0]
slow.sort(key=lambda t: t[2])
print(f"{'op':<26}{'shape':<32}{'CU/PT':>10}   <-- 越小越糟")
print("-" * 80)
for n, s, sp in slow:
    print(f"{n:<26}{s:<32}{sp:>10.2f}x")
if not slow:
    print("  (none — CUDA 全面领先)")
print()


# ============================================================
# 角度 3: L2 误差排名 (数值正确性)
# ============================================================
print("### 角度 3: L2 误差排名 (vs PyTorch reference, 大 = 怀疑数值 bug 或 FP16)\n")
err = [(r["name"], r["shapes"], r.get("triton_l2_err", 0), r.get("cuda_l2_err", 0))
       for r in rows if r.get("triton_l2_err", 0) > 0 or r.get("cuda_l2_err", 0) > 0]
err.sort(key=lambda t: -max(t[2], t[3]))
print(f"{'op':<26}{'shape':<32}{'Triton L2':>14}{'CUDA L2':>14}")
print("-" * 86)
for n, s, t, c in err[:10]:
    print(f"{n:<26}{s:<32}{t:>14.2e}{c:>14.2e}")
print()


# ============================================================
# 角度 4: backward 加速比 (自研 kernel 强项)
# ============================================================
print("### 角度 4: backward 加速比 (绕过 autograd, 自研 kernel 强项)\n")
bw = [r for r in rows if "backward" in r["name"]]
bw.sort(key=lambda r: -r.get("cuda_speedup", 0))
print(f"{'op':<26}{'shape':<32}{'Tr/PT':>9}{'CU/PT':>9}")
print("-" * 80)
for r in bw:
    print(f"{r['name']:<26}{r['shapes']:<32}"
          f"{r.get('triton_speedup', 0):>8.2f}x"
          f"{r.get('cuda_speedup', 0):>8.2f}x")
print()


# ============================================================
# 角度 5: 带宽利用率 (elementwise GB/s)
# ============================================================
print("### 角度 5: elementwise 带宽利用率 (GB/s, RTX 5070 Ti 理论 896 GB/s)\n")
ew = [r for r in rows if r.get("flops", 0) == 0 and r.get("bytes_io", 0) > 0]
ew.sort(key=lambda r: -max(r.get("pytorch_gbps", 0), r.get("triton_gbps", 0), r.get("cuda_gbps", 0)))
print(f"{'op':<26}{'shape':<24}{'PT':>9}{'Tr':>9}{'CU':>9}  GB/s")
print("-" * 84)
for r in ew[:14]:
    print(f"{r['name']:<26}{r['shapes']:<24}"
          f"{r.get('pytorch_gbps', 0):>9.1f}"
          f"{r.get('triton_gbps', 0):>9.1f}"
          f"{r.get('cuda_gbps', 0):>9.1f}")
print()
