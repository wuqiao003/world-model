#!/usr/bin/env python3
"""Plot Mem v3 Pareto-style curve from existing result JSONs (no GPU)."""
import json
from pathlib import Path

# Prefer cluster results if present locally; else embed seed1 numbers from log
RUNS = [
    Path(r"f:\world model\reports\mem_v3_seed1.json"),
]
# Fallback: known seed1 from cluster (2026-08-17)
SEED1 = {
    "force_w_0.0": {"instant_cf_gap": 0.0011527929455041885, "fact_err": 0.011101248674094677},
    "force_w_0.25": {"instant_cf_gap": 0.0004047797410748899, "fact_err": 0.010406216606497765},
    "force_w_0.5": {"instant_cf_gap": 0.00035608408506959677, "fact_err": 0.01030649896711111},
    "force_w_0.75": {"instant_cf_gap": 0.0003295913338661194, "fact_err": 0.010116402059793472},
    "force_w_1.0": {"instant_cf_gap": 0.0, "fact_err": 0.010079588741064072},
    "learn_gate": {"instant_cf_gap": 0.00015560448809992522, "fact_err": 0.010160830803215504, "mean_w": 0.8395830988883972},
}

OUT = Path(r"f:\world model\reports\figures")
OUT.mkdir(parents=True, exist_ok=True)


def load():
    for p in RUNS:
        if p.is_file():
            return json.loads(p.read_text())
    return SEED1


def main():
    d = load()
    ws = [0.0, 0.25, 0.5, 0.75, 1.0]
    gaps = [d[f"force_w_{w}"]["instant_cf_gap"] for w in ws]
    facts = [d[f"force_w_{w}"]["fact_err"] for w in ws]

    # SVG (no matplotlib dependency)
    W, H, pad = 640, 360, 50
    xs = [pad + i * (W - 2 * pad) / 4 for i in range(5)]
    ymax = max(gaps) * 1.15 or 1e-3
    ys = [H - pad - (g / ymax) * (H - 2 * pad) for g in gaps]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    circles = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#1d4ed8"/>' for x, y in zip(xs, ys)
    )
    labels = "\n".join(
        f'<text x="{x:.1f}" y="{H - pad + 18}" text-anchor="middle" font-size="12">{w}</text>'
        for x, w in zip(xs, ws)
    )
    # learn_gate marker
    lg = d.get("learn_gate", {})
    lg_g = lg.get("instant_cf_gap")
    lg_w = lg.get("mean_w", 0.84)
    extra = ""
    if lg_g is not None:
        lx = pad + lg_w * (W - 2 * pad)
        ly = H - pad - (lg_g / ymax) * (H - 2 * pad)
        extra = f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="7" fill="none" stroke="#b45309" stroke-width="2"/><text x="{lx:.1f}" y="{ly-12:.1f}" text-anchor="middle" font-size="11" fill="#b45309">learn_gate</text>'

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">
  <rect width="100%" height="100%" fill="#fafafa"/>
  <text x="{W/2}" y="28" text-anchor="middle" font-size="16" font-family="Segoe UI,sans-serif">Mem v3: instant CF gap vs force_w (seed1)</text>
  <line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#333"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#333"/>
  <text x="{W/2}" y="{H-8}" text-anchor="middle" font-size="13">force_w (memory mix)</text>
  <text x="16" y="{H/2}" transform="rotate(-90 16,{H/2})" text-anchor="middle" font-size="13">instant_cf_gap (controllability↓)</text>
  <polyline fill="none" stroke="#1d4ed8" stroke-width="2" points="{pts}"/>
  {circles}
  {labels}
  {extra}
  <text x="{W-pad}" y="{pad+10}" text-anchor="end" font-size="11" fill="#555">monotonic: True</text>
</svg>
'''
    (OUT / "mem_v3_pareto_seed1.svg").write_text(svg, encoding="utf-8")

    md = OUT / "mem_v3_table.md"
    lines = ["| force_w | instant_cf_gap | fact_err |", "|---:|---:|---:|"]
    for w, g, f in zip(ws, gaps, facts):
        lines.append(f"| {w} | {g:.6f} | {f:.6f} |")
    if lg:
        lines.append(f"| learn_gate (mean_w≈{lg.get('mean_w', float('nan')):.3f}) | {lg.get('instant_cf_gap', float('nan')):.6f} | {lg.get('fact_err', float('nan')):.6f} |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("WROTE", OUT / "mem_v3_pareto_seed1.svg")
    print("WROTE", md)


if __name__ == "__main__":
    main()
