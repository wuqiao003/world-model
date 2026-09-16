#!/usr/bin/env python3
"""
Ceiling check: what would a near-perfect dynamics model score on each utility axis?

metric_reliability.py showed that mppi_cos barely reproduces across evaluation
seeds and that the three utility axes disagree with each other. Two explanations
are still open, and they call for opposite responses:

  budget        the estimator is fine, we simply average too few anchors
                -> spend more compute
  specification the axis asks for something the goal does not determine, so even
                a perfect model scores near zero and the axis cannot rank models
                -> the axis has to go, and no budget rescues it

The way to separate them is to run the same three measurements with a dynamics
model that is as close to correct as this dataset allows: ProprioNNSim, the
nearest-neighbour transition model in end-effector space, planning against the
true recorded goal position. It is not literally perfect -- it inherits the
dataset's coverage -- but it is far better than any checkpoint in the zoo, so it
bounds what the axis can deliver.

An axis on which the oracle scores no better than the collapsed model (lam0) is
broken by construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from k1k2_real_eval import EPS, ProprioNNSim, load_proprio, load_subset


def mppi_proprio(sim, p0, p_goal, horizon, k=128, iters=10, beta=3.0, var=None):
    """MPPI in raw action space against the proprio simulator."""
    dev = p0.device
    scale = var if var is not None else float(sim.A.norm(dim=-1).mean()) + EPS
    mean = torch.zeros(horizon, 2, device=dev)
    std = torch.full_like(mean, scale)
    for _ in range(iters):
        cand = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(k, horizon, 2, device=dev)
        cost = torch.empty(k, device=dev)
        for c in range(k):
            p = p0.clone()
            for h in range(horizon):
                p, _ = sim.step(p, cand[c, h])
            cost[c] = (p - p_goal).pow(2).mean()
        w = F.softmax(-beta * cost / (cost.std() + EPS), dim=0)
        mean = (w.view(-1, 1, 1) * cand).sum(0)
    return mean[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pusht_heldout.h5")
    ap.add_argument("--out", default="runs/final/utility_oracle.json")
    ap.add_argument("--n", type=int, default=60, help="anchors per axis")
    ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    _, actions_raw, chain, block, horizon = load_subset(Path(args.data))
    proprio = load_proprio(Path(args.data))
    sim = ProprioNNSim(torch.as_tensor(proprio, dtype=torch.float32),
                       torch.as_tensor(chain, dtype=torch.long),
                       torch.as_tensor(actions_raw, dtype=torch.float32))
    A = len(chain)
    sel = np.linspace(0, A - 1, min(args.n, A)).astype(int)
    print(f"{A} anchors, horizon={horizon}, block={block}; oracle on {len(sel)}")

    # net raw displacement per plan step: (A, horizon, 2)
    a_expert = torch.as_tensor(actions_raw, dtype=torch.float32).sum(2)
    coss, ranks, progs = [], [], []
    for i in sel:
        p0 = sim.p_frame[chain[i, 0]]
        p_goal = sim.p_frame[chain[i, -1]]

        # axis 1: recover the expert's first action from the terminal goal
        a_hat = mppi_proprio(sim, p0, p_goal, horizon, k=args.k, iters=args.iters)
        a_true = a_expert[i, 0]
        coss.append(float(F.cosine_similarity(a_hat, a_true, dim=0)))

        # axis 2: does the expert plan beat random plans under the oracle's cost
        cand = torch.randn(args.k, horizon, 2) * (float(sim.A.norm(dim=-1).mean()) + EPS)
        allc = torch.cat([cand, a_expert[i].unsqueeze(0)], 0)
        cost = torch.empty(allc.size(0))
        for c in range(allc.size(0)):
            p = p0.clone()
            for h in range(horizon):
                p, _ = sim.step(p, allc[c, h])
            cost[c] = (p - p_goal).pow(2).mean()
        c_rand, c_exp = cost[:-1], cost[-1]
        tol = 1e-6 * (c_rand.abs().mean() + EPS)
        ranks.append(float((c_rand < c_exp - tol).float().mean()
                           + 0.5 * ((c_rand - c_exp).abs() <= tol).float().mean()))

        # axis 3: closed-loop receding horizon, oracle planning and oracle stepping
        p, d0 = p0.clone(), float((p0 - p_goal).norm()) + EPS
        for _ in range(horizon):
            a = mppi_proprio(sim, p, p_goal, horizon, k=args.k,
                             iters=max(2, args.iters // 2))
            p, _ = sim.step(p, a)
        progs.append(1.0 - float((p - p_goal).norm()) / d0)

    res = {
        "n": len(sel),
        "mppi_cos": {"mean": float(np.mean(coss)), "sd": float(np.std(coss))},
        "gt_rank_pct": {"mean": float(np.mean(ranks)), "sd": float(np.std(ranks))},
        "cl_progress": {"mean": float(np.mean(progs)), "sd": float(np.std(progs))},
    }
    print("\noracle (ProprioNNSim planning against the recorded goal)")
    for kk, v in res.items():
        if isinstance(v, dict):
            print(f"  {kk:14s} {v['mean']:+.4f}  sd={v['sd']:.4f}")

    print("\nfor reference, the zoo on the same axes (primary_seed0, all strata):")
    try:
        r = json.load(open("runs/final/primary_seed0.json"))
        for axis in ("mppi_cos", "gt_rank_pct", "cl_progress"):
            vals = [m[axis] for sv in r["strata"].values()
                    for m in sv["systems"].values()
                    if isinstance(m.get(axis), (int, float)) and np.isfinite(m[axis])]
            best = max(vals) if axis != "gt_rank_pct" else min(vals)
            print(f"  {axis:14s} zoo best {best:+.4f}   oracle "
                  f"{res[axis]['mean']:+.4f}")
    except FileNotFoundError:
        pass

    print("\nAn axis where the oracle lands near the zoo is not measuring model "
          "quality: the goal does not identify what the axis asks for, and no "
          "amount of averaging will make it rank models.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
