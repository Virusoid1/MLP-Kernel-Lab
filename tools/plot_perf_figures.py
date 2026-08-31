#!/usr/bin/env python3
"""Generate performance figures from archived benchmark data (no re-measurement). Outputs: figures/*.png"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTS = Path("artifacts")
FIGDIR = Path("figures"); FIGDIR.mkdir(exist_ok=True)

def load_bench(sub):
    matches = sorted(ARTS.glob('swiglu_*' + sub + '*'), reverse=True)
    if not matches: return []
    jsons = sorted(matches[0].glob('*.json'))
    bench = [j for j in jsons if 'swiglu_bench' in j.name]
    if not bench: return []
    return json.loads(bench[0].read_text()).get('rows', [])

# ---- Fig1: decode per-token amortization (log) ----
rows = load_bench("034658-decode")
if rows:
    tr = sorted([(r['M'], r['median_ms']/max(r['M'],1)) for r in rows if r['dtype']=='fp16' and r['backend']=='triton'], key=lambda x: x[0])
    if tr:
        ms_ = [x[0] for x in tr]; pts = [x[1] for x in tr]
        plt.figure(figsize=(7,4.5))
        plt.semilogy(ms_, pts, "o", label="triton fp16 per-token")
        plt.xlabel("batch M (tokens)"); plt.ylabel("per-token ms (log)")
        plt.title("decode amortization: per-token cost sheds with batch (K=4096 F=11008)")
        plt.grid(True, which="both", alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig("figures/decode_amortization.png", dpi=130)
        print("fig1 saved: %d points, M1 per-token=%.4f ms" % (len(tr), pts[0]))

# ---- Fig2: fp16 triton speedup vs eager-fp16 (all-suite shapes) ----
rows = load_bench("013853-all")
if rows:
    pts = []
    for r in rows:
        if r['dtype']=='fp16' and r['backend']=='triton':
            eg = [x for x in rows if x['M']==r['M'] and x['K']==r['K'] and x['F']==r['F'] and x['dtype']=='fp16' and x['backend']=='eager']
            if eg and eg[0]['median_ms'] > 0:
                pts.append((str(r['M']), eg[0]['median_ms']/r['median_ms']))
    if pts:
        pts.sort(key=lambda x: x[1], reverse=True)
        labels = [x[0] for x in pts]; vals = [x[1] for x in pts]
        plt.figure(figsize=(7,4.5))
        plt.bar(range(len(vals)), vals, color="#4C72B0")
        plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
        plt.axhline(1.0, color="red", ls="--", lw=1, label="eager-fp16 = 1x")
        plt.ylabel("triton-fp16 / eager-fp16"); plt.title("triton fp16 vs eager-fp16 by M (prefill/train)")
        plt.legend(); plt.tight_layout()
        plt.savefig("figures/fp16_speedup_shapes.png", dpi=130)
        print("fig2 saved: %d shapes; max=%.2fx" % (len(vals), max(vals)))

# ---- Fig3: roofline efficiency by backend (fp16, 512x4096x11008) ----
if rows:
    F16_PEAK = 31.0
    d = []
    for backend in ["eager","triton","cutile","cuda"]:
        r0 = [x for x in rows if x['backend']==backend and x['dtype']=='fp16' and x['M']==512 and x['K']==4096 and x['F']==11008]
        if r0:
            tf = 6*512*4096*11008/(r0[0]['median_ms']*1e-3)/1e12
            d.append((backend, tf/F16_PEAK*100))
    if d:
        names=[x[0] for x in d]; eff=[x[1] for x in d]
        plt.figure(figsize=(6,4))
        plt.bar(names, eff, color=["#55A868","#4C72B0","#C44E52","#CCB974"])
        plt.ylabel("% of fp16-TC peak (31 TFLOPS)")
        plt.title("roofline efficiency (fp16, 512x4096x11008)")
        for i,v in enumerate(eff): plt.text(i, v+1, "%.0f%%" % v, ha="center")
        plt.ylim(0, max(eff)*1.25); plt.tight_layout()
        plt.savefig("figures/roofline_efficiency.png", dpi=130)
        print("fig3 saved:", dict(zip(names, [round(e,1) for e in eff])))

print("done")
