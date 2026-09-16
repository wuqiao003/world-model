#!/usr/bin/env python3
"""
How much of each metric is signal?

Everything downstream -- the K1/K2 rank correlations, the K3 memory trend, the
claim that the ranking depends on which utility axis you pick -- is a statement
about how systems are ordered by these numbers. That is only meaningful if the
ordering reproduces. A metric whose per-system values are re-drawn noise will
still produce correlations, and they will still look like findings.

Two independent checks:

  cross-seed     rank agreement of the per-system values between evaluation
                 seeds. Anchors are fixed across seeds, so the offline metrics
                 are deterministic and must come out at 1.00 -- that is a sanity
                 check on the harness, not evidence. The planning metrics
                 (mppi_cos, gt_rank_pct, cl_progress) are re-sampled per seed,
                 and this is the real test for them.

  cross-stratum  rank agreement between strata. Strata hold different anchors,
                 so this asks whether a system's score is a property of the
                 system or of the anchors it happened to be scored on. This is
                 the only reliability test the deterministic metrics get.

A utility axis that fails both cannot support a claim about which metric
predicts it, however tight the bootstrap interval around the correlation looks.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from itertools import combinations

import numpy as np

COLS = ["pred_id_ratio_h1", "gap_ratio", "pv_ratio", "caca", "cf_align",
        "fact_align", "mppi_cos", "gt_rank_pct", "gt_rank_id_pct", "ood_credit",
        "cl_progress", "plan_cost_corr"]
UTILS = ["mppi_cos", "gt_rank_pct", "gt_rank_id_pct", "cl_progress",
         "plan_cost_corr"]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok]))
    rb = np.argsort(np.argsort(b[ok]))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pairwise(vectors):
    """Mean Spearman over every pair of score vectors, on their common systems."""
    rhos = []
    for va, vb in combinations(vectors, 2):
        common = sorted(set(va) & set(vb))
        if len(common) < 3:
            continue
        r = spearman([va[k] for k in common], [vb[k] for k in common])
        if np.isfinite(r):
            rhos.append(r)
    return rhos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/final/primary_seed*.json")
    ap.add_argument("--out", default="runs/final/metric_reliability.json")
    ap.add_argument("--floor", type=float, default=0.5,
                    help="reliability below this is reported as unusable")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no files match {args.glob}")
    runs = [json.load(open(f)) for f in files]
    print(f"{len(runs)} evaluation seeds, "
          f"{len(runs[0]['strata'])} strata: {[f.split('/')[-1] for f in files]}")

    # metric -> seed -> stratum -> {system: value}
    v = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for si, r in enumerate(runs):
        for sk, sv in r["strata"].items():
            for name, m in sv["systems"].items():
                for c in COLS:
                    x = m.get(c)
                    if isinstance(x, (int, float)) and np.isfinite(x):
                        v[c][si][sk][name] = float(x)

    rows = {}
    for c in COLS:
        # cross-seed, holding the stratum fixed
        seed_rhos = []
        for sk in runs[0]["strata"]:
            vecs = [v[c][si][sk] for si in range(len(runs)) if v[c][si][sk]]
            seed_rhos += pairwise(vecs)
        # cross-stratum, holding the seed fixed
        strat_rhos = []
        for si in range(len(runs)):
            vecs = [d for d in v[c][si].values() if d]
            strat_rhos += pairwise(vecs)
        rows[c] = {
            "cross_seed": float(np.mean(seed_rhos)) if seed_rhos else float("nan"),
            "cross_seed_n": len(seed_rhos),
            "cross_stratum": float(np.mean(strat_rhos)) if strat_rhos else float("nan"),
            "cross_stratum_n": len(strat_rhos),
        }

    print(f"\n{'metric':20s} {'cross-seed':>12s} {'cross-stratum':>14s}")
    for c, r in rows.items():
        cs = f"{r['cross_seed']:+.2f}" if np.isfinite(r["cross_seed"]) else "  n/a"
        ct = f"{r['cross_stratum']:+.2f}" if np.isfinite(r["cross_stratum"]) else "  n/a"
        tag = ""
        if np.isfinite(r["cross_stratum"]) and r["cross_stratum"] < args.floor:
            tag = "  <- unreliable"
        print(f"{c:20s} {cs:>12s} {ct:>14s}{tag}")

    print("\nutility axes")
    bad = []
    for u in UTILS:
        r = rows[u]
        worst = np.nanmin([r["cross_seed"], r["cross_stratum"]])
        verdict = "usable" if worst >= args.floor else "TOO NOISY TO SUPPORT A CLAIM"
        print(f"  {u:15s} worst reliability {worst:+.2f}   {verdict}")
        if worst < args.floor:
            bad.append(u)
    if bad:
        print("\nAny K1/K2 statement about " + ", ".join(bad) + " is a statement "
              "about noise. Either raise the planning budget / anchor count until "
              "reliability clears the floor, or drop the axis -- do not report a "
              "correlation against it.")

    with open(args.out, "w") as f:
        json.dump({"reliability": rows, "unreliable_utilities": bad,
                   "floor": args.floor, "n_eval_seeds": len(runs)}, f, indent=2)
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
