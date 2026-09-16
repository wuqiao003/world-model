#!/usr/bin/env python3
"""
K3: memory <-> controllability on *trained* context-length variants.

The inference-time version of this experiment (code/k3_mem_ctrl_real.py) was a
negative result: blending context slots in a frozen checkpoint barely moved
either axis. The only legitimate remedy is to train the sweep, so train_zoo_spec
carries mem1/mem2/mem3/mem5/mem8 -- identical apart from context length, and
deliberately sharing one batch size and epoch budget because SIGReg is a batch
statistic and would otherwise vary along the memory axis too.

train_zoo_replicates.json re-trains the same five context lengths under two
further seeds. That distinction drives the whole analysis below:

  evaluation seed  only randomises planning (mppi_cos, gt_rank_pct, cl_progress);
                   the offline metrics are deterministic given a checkpoint, so
                   agreement across evaluation seeds is not replication.
  training seed    gives a genuinely independent 5-point memory curve, so the
                   spread of the trend across training seeds is the error bar
                   the claim actually rests on.

Reports, per context length:
  fidelity        pred_id_ratio_h1   (lower is better; what memory should buy)
  sensitivity     gap_ratio          (does the action still move the prediction)
  controllability caca               (is the action-attributable part correct)
  closed loop     cl_progress        (does planning with it actually control)
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict

import numpy as np

CTX = [1, 2, 3, 5, 8]
NAME_RE = re.compile(r"^mem(\d+)(?:_s(\d+))?$")
COLS = ["pred_id_ratio_h1", "gap_ratio", "pv_ratio", "caca", "cf_align",
        "fact_align", "mppi_cos", "gt_rank_pct", "cl_progress"]


def parse(name: str):
    """'mem5:id' -> (5, 0); 'mem5_s2:id' -> (5, 2); anything else -> None."""
    m = NAME_RE.match(name.split(":")[0])
    if not m:
        return None
    ctx = int(m.group(1))
    return (ctx, int(m.group(2) or 0)) if ctx in CTX else None


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/final/memrep_seed*.json")
    ap.add_argument("--out", default="runs/final/k3_mem_pareto.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no files match {args.glob}")
    runs = [json.load(open(f)) for f in files]
    print(f"{len(runs)} evaluation seeds: {[f.split('/')[-1] for f in files]}")

    # (ctx, train_seed) -> metric -> values, pooled over evaluation seeds and
    # strata. Strata differ only in which anchors they use, evaluation seeds only
    # in planning noise, so both are nuisance dimensions for K3.
    cell = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for sv in r["strata"].values():
            for name, m in sv["systems"].items():
                key = parse(name)
                if key is None:
                    continue
                for c in COLS:
                    v = m.get(c, np.nan)
                    if v is not None and np.isfinite(v):
                        cell[key][c].append(float(v))

    if not cell:
        raise SystemExit("no mem* systems found in the evaluation output")
    seeds = sorted({s for _, s in cell})
    print(f"{len(seeds)} training seeds per context length: {seeds}")

    # Curve per training seed: seed -> metric -> {ctx: value}.
    curves = defaultdict(lambda: defaultdict(dict))
    for (ctx, s), mm in cell.items():
        for c, vals in mm.items():
            if vals:
                curves[s][c][ctx] = float(np.mean(vals))

    table = {}
    for ctx in CTX:
        row = {"ctx": ctx, "n_runs": sum(1 for s in seeds if (ctx, s) in cell)}
        for c in COLS:
            vals = [curves[s][c][ctx] for s in seeds if ctx in curves[s].get(c, {})]
            row[c] = float(np.mean(vals)) if vals else float("nan")
            row[c + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        if row["n_runs"]:
            table[ctx] = row

    print(f"\nmean over training runs (sd across runs in brackets)")
    print(f"{'ctx':>3s} {'n':>2s} " + " ".join(f"{c[:11]:>18s}" for c in COLS))
    for ctx, row in table.items():
        cells = []
        for c in COLS:
            sd = row[c + "_sd"]
            cells.append(f"{row[c]:9.4f}" + (f" ({sd:6.4f})" if np.isfinite(sd) else " " * 9))
        print(f"{ctx:3d} {row['n_runs']:2d} " + " ".join(cells))

    # The trend is estimated once per training seed. With one seed this is a
    # single coarse 5-point correlation; with three it has a spread, and the
    # sign agreement across seeds is the claim's actual support.
    trends = {}
    for c in COLS:
        rhos = []
        for s in seeds:
            pts = sorted(curves[s].get(c, {}).items())
            if len(pts) >= 3:
                rhos.append(spearman([p[0] for p in pts], [p[1] for p in pts]))
        pooled_pts = [(ctx, table[ctx][c]) for ctx in table if np.isfinite(table[ctx][c])]
        if len(pooled_pts) < 3:
            continue
        pooled = spearman([p[0] for p in pooled_pts], [p[1] for p in pooled_pts])
        y0, y1 = pooled_pts[0][1], pooled_pts[-1][1]
        trends[c] = {
            "spearman_pooled": pooled,
            "spearman_per_train_seed": [round(v, 3) for v in rhos],
            "n_train_seeds": len(rhos),
            "n_agree_sign": int(sum(np.sign(v) == np.sign(pooled) for v in rhos)),
            "at_ctx_min": y0,
            "at_ctx_max": y1,
            "rel_change": float((y1 - y0) / (abs(y0) + 1e-12)),
        }

    print("\ntrend vs context length, estimated once per training seed")
    for c, t in trends.items():
        per = ",".join(f"{v:+.2f}" for v in t["spearman_per_train_seed"])
        print(f"  {c:20s} pooled rho={t['spearman_pooled']:+.2f}  "
              f"per-run [{per}]  sign {t['n_agree_sign']}/{t['n_train_seeds']}  "
              f"{t['at_ctx_min']:+.4f} -> {t['at_ctx_max']:+.4f} "
              f"({t['rel_change']*100:+.0f}%)")

    weak = [c for c, t in trends.items()
            if t["n_train_seeds"] > 1 and t["n_agree_sign"] < t["n_train_seeds"]]
    if weak:
        print("\nsign flips across training seeds (do not claim a trend): "
              + ", ".join(weak))
    if len(seeds) < 2:
        print("\nONE training run per context length: the trends above have no "
              "error bars. Train code/train_zoo_replicates.json before claiming "
              "a memory-controllability trade-off.")

    with open(args.out, "w") as f:
        json.dump(
            {
                "table": {str(k): v for k, v in table.items()},
                "trends": trends,
                "n_eval_seeds": len(runs),
                "n_train_seeds": len(seeds),
                "sign_flip_metrics": weak,
            },
            f, indent=2,
        )
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
