#!/usr/bin/env python3
"""SVG for excitation_v3: PV on high_exc eval by train mode."""
from pathlib import Path

OUT = Path(r"f:\world model\reports\figures")
OUT.mkdir(parents=True, exist_ok=True)

# from cluster excitation_v3 seed0/1 (approx from reports)
# seed0: demo 0.0505, high_exc 0.0370; seed1: demo 0.0498, high_exc 0.0410
rows = [
    ("seed0", 0.050505, 0.037001),
    ("seed1", 0.049840, 0.041005),
]

W, H, pad = 560, 320, 55
bar_w = 36
gap = 80
svg_bars = []
ymax = 0.06
for i, (name, demo, he) in enumerate(rows):
    x0 = pad + 40 + i * 220
    for j, (val, color, label) in enumerate([
        (demo, "#b91c1c", "train_demo"),
        (he, "#15803d", "train_high_exc"),
    ]):
        x = x0 + j * (bar_w + 18)
        bh = (val / ymax) * (H - 2 * pad)
        y = H - pad - bh
        svg_bars.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bh:.1f}" fill="{color}"/>')
        svg_bars.append(f'<text x="{x+bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="11">{val:.3f}</text>')
    svg_bars.append(f'<text x="{x0+bar_w+9}" y="{H-pad+20}" text-anchor="middle" font-size="13">{name}</text>')

svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">
  <rect width="100%" height="100%" fill="#fafafa"/>
  <text x="{W/2}" y="28" text-anchor="middle" font-size="15" font-family="Segoe UI,sans-serif">Excitation v3: oracle PV on high_exc eval (lower better)</text>
  <line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#333"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#333"/>
  {''.join(svg_bars)}
  <rect x="{W-200}" y="48" width="12" height="12" fill="#b91c1c"/><text x="{W-182}" y="58" font-size="12">train demo</text>
  <rect x="{W-200}" y="68" width="12" height="12" fill="#15803d"/><text x="{W-182}" y="78" font-size="12">train high_exc</text>
  <text x="{W/2}" y="{H-12}" text-anchor="middle" font-size="12">pass: high_exc training improves PV (not self_cf_gap)</text>
</svg>
'''
(OUT / "excitation_v3_pv.svg").write_text(svg, encoding="utf-8")
(OUT / "excitation_v3_table.md").write_text(
    "| seed | PV train_demo | PV train_high_exc | pass |\n|---|---:|---:|---|\n"
    "| 0 | 0.0505 | 0.0370 | yes |\n| 1 | 0.0498 | 0.0410 | yes |\n",
    encoding="utf-8",
)
print("WROTE", OUT / "excitation_v3_pv.svg")
