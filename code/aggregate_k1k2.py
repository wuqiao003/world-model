#!/usr/bin/env python3
"""
Aggregate multi-seed K1/K2 results: pooled system metrics, rank correlations
against both planning utilities, and a paired bootstrap over systems that tests
whether the best controllability metric beats the best fidelity metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 1e-12 else float("nan")


OURS = ["caca", "caf_caca", "caf_caca_mag", "cf_align", "caf_align", "caf", "caf_soft", "dual", "neg_pv"]
FIDELITY = ["fact_align", "neg_fact_h1", "neg_fact_hH", "neg_pred_id_ratio_h1", "neg_pred_id_ratio_hH"]
# mppi_cos and cl_progress are kept only so the paper can show what they do:
# code/utility_oracle.py finds a near-perfect dynamics model scoring no better
# than the zoo on either, so neither can order models. plan_cost_corr replaces
# them with a target the goal actually identifies.
UTILS = ["mppi_cos", "neg_gt_rank", "neg_gt_rank_id", "cl_progress",
         "plan_cost_corr"]


def split_name(name: str):
    """'lewm_v2/w0.05:rot30' -> (model_with_variant, pipeline)."""
    model, _, pipe = name.rpartition(":")
    return model, pipe


def cluster_bootstrap(pooled, names, util_key, key_a, key_b, cluster_by, n_boot=4000, seed=0):
    """
    Paired bootstrap resampling whole clusters rather than individual systems.

    Systems are not independent: all pipelines of one checkpoint share an
    encoder, and all checkpoints under one pipeline share the same corruption.
    Resampling clusters is the conservative accounting.
    """
    rng = np.random.default_rng(seed)
    if cluster_by == "system":
        clusters = [[n] for n in names]
    else:
        pos = 0 if cluster_by == "model" else 1
        buckets = {}
        for n in names:
            buckets.setdefault(split_name(n)[pos], []).append(n)
        clusters = list(buckets.values())
    if len(clusters) < 2:
        # One cluster cannot be resampled; reporting a zero-width interval here
        # would read as certainty. Happens for cluster_by="pipeline" when the
        # analysis uses a single pipeline.
        return {"n_clusters": len(clusters), "degenerate": True}
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(clusters), len(clusters))
        sub = [n for i in pick for n in clusters[i]]
        u = [pooled[n][util_key] for n in sub]
        a = spearman([pooled[n][key_a] for n in sub], u)
        b = spearman([pooled[n][key_b] for n in sub], u)
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(a - b)
    diffs = np.asarray(diffs)
    if diffs.size == 0:
        return {"n_clusters": len(clusters), "ci95": [float("nan")] * 2, "p_ours_worse": float("nan")}
    return {
        "n_clusters": len(clusters),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_ours_worse": float((diffs <= 0).mean()),
    }


def analyse(runs, get_systems, label, report):
    names = sorted(get_systems(runs[0]))
    # pool metrics across seeds (metrics are deterministic except the planning ones)
    pooled = {}
    for n in names:
        keys = set()
        for r in runs:
            keys |= set(get_systems(r)[n])
        pooled[n] = {
            k: float(np.nanmean([get_systems(r)[n].get(k, np.nan) for r in runs]))
            for k in keys
        }
    report[label] = {"systems": pooled}
    print(f"\n########## {label} ##########")

    for util in UTILS:
        if util not in pooled[names[0]] or not np.isfinite(
            [pooled[n].get(util, np.nan) for n in names]
        ).any():
            continue
        u = [pooled[n][util] for n in names]
        corr = {k: spearman([pooled[n][k] for n in names], u) for k in OURS + FIDELITY}
        best_ours = max((corr[k], k) for k in OURS if np.isfinite(corr[k]))
        best_fid = max((corr[k], k) for k in FIDELITY if np.isfinite(corr[k]))

        boots = {
            cb: cluster_bootstrap(pooled, names, util, best_ours[1], best_fid[1], cb)
            for cb in ("system", "pipeline", "model")
        }
        report[label][f"vs_{util}"] = {
            "n_systems": len(names),
            "spearman": corr,
            "best_ours": {"metric": best_ours[1], "rho": best_ours[0]},
            "best_fidelity": {"metric": best_fid[1], "rho": best_fid[0]},
            "delta": best_ours[0] - best_fid[0],
            "bootstrap": boots,
        }
        d = report[label][f"vs_{util}"]
        print(f"\n== utility = {util} ==  (n_systems={len(names)})")
        for k in OURS + FIDELITY:
            tag = "OURS" if k in OURS else "fid "
            print(f"  [{tag}] {k:24s} rho={corr[k]:+.3f}")
        print(
            f"  best ours    : {best_ours[1]} rho={best_ours[0]:+.3f}\n"
            f"  best fidelity: {best_fid[1]} rho={best_fid[0]:+.3f}\n"
            f"  delta={d['delta']:+.3f}"
        )
        for cb, b in boots.items():
            if b.get("degenerate"):
                continue
            print(
                f"    boot[{cb:8s}] n_clusters={b['n_clusters']:3d} "
                f"CI95={b['ci95'][0]:+.3f}..{b['ci95'][1]:+.3f} "
                f"P(ours<=fid)={b['p_ours_worse']:.3f}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/result_seed*.json")
    ap.add_argument("--out", default="runs/k1k2_aggregate.json")
    args = ap.parse_args()

    files = sorted(Path().glob(args.glob))
    if not files:
        raise SystemExit(f"no files match {args.glob}")
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    print(f"seeds={len(runs)}: {[p.name for p in files]}")

    report = {
        "n_seeds": len(runs),
        "harness_sanity": runs[0].get("harness_sanity", {}),
    }

    if "strata" in runs[0] and len(runs[0]["strata"]) > 1:
        for sk in sorted(runs[0]["strata"]):
            mean_a = runs[0]["strata"][sk]["net_action_mean"]
            analyse(
                runs,
                lambda r, sk=sk: r["strata"][sk]["systems"],
                f"stratum_{sk}_meanA{mean_a:.4f}",
                report,
            )
    else:
        analyse(runs, lambda r: r["systems"], "all", report)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nWROTE", args.out)


if __name__ == "__main__":
    main()
