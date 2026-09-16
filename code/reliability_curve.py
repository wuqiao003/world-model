#!/usr/bin/env python3
"""
How many anchors does each metric need before its system ranking is stable?

metric_reliability.py can only compare whole evaluation runs, which conflates
two things: sampling noise, and systems genuinely behaving differently on
different anchors. This works one stratum at a time on the raw per-unit values
in <result>.units.npz, so the anchors are drawn from a single population and any
disagreement is sampling noise alone.

Protocol, per metric and per stratum:
  * draw two disjoint subsets of n units,
  * score every system on each subset,
  * Spearman between the two score vectors, over systems,
  * average over repeats and strata.

That is a split-half reliability at size n. Spearman-Brown converts it to the
reliability of the full run, and inverts to the n that would be needed to reach
a target -- which turns "this axis is noisy" into a budget.
"""
from __future__ import annotations

import argparse
import glob
import re
from collections import defaultdict

import numpy as np

# metric -> (unit group, function of a subsampled unit dict)
GROUP = {
    "pred_id_ratio_h1": "anchor", "gap_ratio": "anchor", "pv_ratio": "anchor",
    "caca": "anchor", "cf_align": "anchor", "fact_align": "anchor",
    "mppi_cos": "plan", "gt_rank_pct": "plan", "gt_rank_id_pct": "plan",
    "plan_cost_corr": "plan", "cl_progress": "closed",
}
EPS = 1e-12


def score(metric, u):
    """Recompute a system's metric from a subsample of its per-unit values."""
    v = u.get("valid")
    if metric == "pred_id_ratio_h1":
        return u["fact1"].mean() / (u["id1"].mean() + EPS)
    if metric == "gap_ratio":
        return u["gap"].mean() / (u["id1"].mean() + EPS)
    if metric == "pv_ratio":
        return (u["pv"][v].mean() / (u["state_gap"][v].mean() + EPS)
                if v.any() else np.nan)
    if metric in ("caca", "cf_align"):
        return u[metric][v].mean() if v.any() else np.nan
    if metric == "fact_align":
        return u["fact_align"].mean()
    if metric == "mppi_cos":
        return u["cos"].mean()
    if metric == "gt_rank_pct":
        return u["rank"].mean()
    if metric == "gt_rank_id_pct":
        return u["rank_id"].mean()
    if metric == "plan_cost_corr":
        return np.nanmean(u["pcc"]) if np.isfinite(u["pcc"]).any() else np.nan
    if metric == "cl_progress":
        return u["progress"].mean()
    raise KeyError(metric)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_brown(r, k):
    """Reliability after scaling the sample by k."""
    r = min(max(r, -0.999), 0.999)
    return k * r / (1 + (k - 1) * r)


def needed_k(r_half, target):
    """Scale factor on a half-sample needed to reach `target` reliability."""
    r = min(max(r_half, 1e-6), 0.999)
    if target >= 1:
        return np.inf
    return target * (1 - r) / (r * (1 - target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/final/primary_seed0.units.npz")
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(
            f"no files match {args.glob} -- re-run k1k2_real_eval.py with --dump-units"
        )
    rng = np.random.default_rng(args.seed)

    # stratum -> system -> field -> array
    data = defaultdict(lambda: defaultdict(dict))
    for f in files:
        z = np.load(f)
        for k in z.files:
            sk, name, field = k.split("|")
            data[f"{f}:{sk}"][name][field] = z[k]

    out = {}
    for metric, grp in GROUP.items():
        halves, sizes = [], []
        for sk, sysd in data.items():
            names = sorted(sysd)
            lens = {len(sysd[n][{"anchor": "fact1", "plan": "cos",
                                 "closed": "progress"}[grp]]) for n in names}
            if len(lens) != 1:
                continue
            n_units = lens.pop()
            half = n_units // 2
            if half < 4 or len(names) < 3:
                continue
            fields = [f for f in sysd[names[0]]
                      if len(sysd[names[0]][f]) == n_units]
            for _ in range(args.repeats):
                perm = rng.permutation(n_units)
                ia, ib = perm[:half], perm[half:2 * half]
                sa, sb = [], []
                for n in names:
                    u = sysd[n]
                    sa.append(score(metric, {f: u[f][ia] for f in fields}))
                    sb.append(score(metric, {f: u[f][ib] for f in fields}))
                r = spearman(sa, sb)
                if np.isfinite(r):
                    halves.append(r)
            sizes.append(n_units)
        if not halves:
            continue
        r_half = float(np.mean(halves))
        r_full = spearman_brown(r_half, 2.0)
        k = needed_k(r_half, args.target)
        n_now = int(np.mean(sizes))
        out[metric] = {
            "units": grp, "n_per_stratum_now": n_now,
            "split_half_r": r_half, "full_run_r": r_full,
            "n_needed_per_stratum": (int(np.ceil(k * n_now / 2))
                                     if np.isfinite(k) else None),
        }

    print(f"{'metric':20s} {'unit':>7s} {'n now':>6s} {'half r':>7s} "
          f"{'full r':>7s} {'n for ' + str(args.target):>10s}")
    for m, d in out.items():
        need = d["n_needed_per_stratum"]
        flag = ""
        if need and need > d["n_per_stratum_now"]:
            flag = f"  x{need / d['n_per_stratum_now']:.0f}"
        print(f"{m:20s} {d['units']:>7s} {d['n_per_stratum_now']:6d} "
              f"{d['split_half_r']:+7.2f} {d['full_run_r']:+7.2f} "
              f"{str(need):>10s}{flag}")

    print("\nn is per stratum; the eval splits its budget across strata, so the "
          "run-level flag is that number times the stratum count.")
    hopeless = [m for m, d in out.items()
                if d["n_needed_per_stratum"] is None
                or d["n_needed_per_stratum"] > 50 * d["n_per_stratum_now"]]
    if hopeless:
        print("Out of reach by sampling alone (the estimator itself is the "
              "problem, not the budget): " + ", ".join(hopeless))


if __name__ == "__main__":
    main()
