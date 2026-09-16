#!/usr/bin/env python3
"""
K2v2: CF-sensitive downstream proxy for main-track predictive validity.

Downstream = open-loop rollout error under HIGH-MAGNITUDE / negated action
sequences (off-policy relative to weak_data). On-policy fact_err should NOT
dominate; CAF or pred_validity should.
"""
from __future__ import annotations

import json, os, time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(os.environ.get("OUT_DIR", "runs/k2v2_cf_pv"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

H, W, T, A = 24, 24, 8, 3
BATCH = int(os.environ.get("BATCH", "48"))
STEPS = int(os.environ.get("STEPS", "700"))
LR = 3e-4
HPLAN = 5


def env_step(frame, action):
    vx, vy, ang = action.unbind(-1)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=frame.device),
        torch.linspace(-1, 1, W, device=frame.device),
        indexing="ij",
    )
    amp = 0.5 * torch.tanh(ang)
    c, s = torch.cos(amp), torch.sin(amp)
    xr = c.view(-1, 1, 1) * xx - s.view(-1, 1, 1) * yy + 0.3 * torch.tanh(vx).view(-1, 1, 1)
    yr = s.view(-1, 1, 1) * xx + c.view(-1, 1, 1) * yy + 0.3 * torch.tanh(vy).view(-1, 1, 1)
    warped = F.grid_sample(frame, torch.stack([xr, yr], -1), align_corners=True, padding_mode="border")
    blob = torch.exp(-(xr ** 2 + yr ** 2) * 6).unsqueeze(1)
    return torch.tanh(warped + 0.45 * torch.tanh(ang).view(-1, 1, 1, 1) * blob)


def make_traj(n, scale=1.0):
    frames = [torch.randn(n, 3, H, W, device=DEVICE) * 0.2]
    acts = scale * torch.randn(n, T - 1, A, device=DEVICE)
    for t in range(T - 1):
        frames.append(env_step(frames[-1], acts[:, t]))
    return torch.stack(frames, 1), acts


def oracle_roll(frame0, acts):
    frames = [frame0]
    for t in range(acts.size(1)):
        frames.append(env_step(frames[-1], acts[:, t]))
    return torch.stack(frames, 1)


class Pred(nn.Module):
    def __init__(self, listen=True):
        super().__init__()
        self.listen = listen
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 96),
        )
        self.act = nn.Linear(A, 96)
        self.film = nn.Linear(96, 192)
        self.trunk = nn.Sequential(nn.Linear(96, 96), nn.GELU())
        self.fc = nn.Linear(96, 64 * 6 * 6)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, frames, acts):
        b, tlen = acts.size(0), acts.size(1)
        z = self.enc(frames[:, :tlen].reshape(b * tlen, 3, H, W)).view(b, tlen, -1)
        if self.listen:
            g, be = self.film(self.act(acts)).chunk(2, -1)
            h = g * self.trunk(z) + be
        else:
            h = self.trunk(z)
        y = self.fc(h).view(b * tlen, 64, 6, 6)
        return torch.tanh(self.deconv(y)).view(b, tlen, 3, H, W)


def train(model, steps=STEPS, scale=1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(steps):
        fr, ac = make_traj(BATCH, scale=scale)
        loss = F.mse_loss(model(fr[:, :-1], ac), fr[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def metrics(model, n=80) -> Dict[str, float]:
    model.eval()
    fr, ac = make_traj(n, scale=1.3)
    pred = model(fr[:, :-1], ac)
    pred_cf = model(fr[:, :-1], -ac)
    fr_cf = oracle_roll(fr[:, 0], -ac)
    fact = F.mse_loss(pred, fr[:, 1:]).item()
    gap = F.mse_loss(pred, pred_cf).item()
    pv = F.mse_loss(pred_cf, fr_cf[:, 1:]).item()
    # calibrated score: reward sensitivity only when PV is good
    caf_cal = gap / (pv + 1e-8) * (1.0 / (1.0 + pv))
    return {
        "fact_err": fact,
        "self_cf_gap": gap,
        "pred_validity": pv,
        "caf": gap / (pv + 1e-8),
        "caf_cal": caf_cal,
        "neg_pv": -pv,
    }


@torch.no_grad()
def cf_plan_err(model, n=48) -> float:
    """Off-policy: large + negated action open-loop error (CF-sensitive utility)."""
    model.eval()
    fr, _ = make_traj(n, scale=1.0)
    acts = 1.5 * torch.randn(n, HPLAN, A, device=DEVICE)
    # mix: half negated patterns
    acts[n // 2 :] = -acts[n // 2 :]
    cur = fr[:, 0]
    preds = []
    for t in range(HPLAN):
        pred = model(cur.unsqueeze(1), acts[:, t : t + 1])[:, 0]
        preds.append(pred)
        cur = pred
    pred_seq = torch.stack(preds, 1)
    oracle = oracle_roll(fr[:, 0], acts)[:, 1:]
    return F.mse_loss(pred_seq, oracle).item()


def spearman(x: List[float], y: List[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for ri, i in enumerate(order):
            r[i] = float(ri)
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den > 1e-12 else float("nan")


def main():
    t0 = time.time()
    seed = int(os.environ.get("SEED", "0"))
    torch.manual_seed(seed)

    variants = {
        "film": Pred(True).to(DEVICE),
        "ignore": Pred(False).to(DEVICE),
        "weak_data": Pred(True).to(DEVICE),
    }
    train(variants["film"], scale=1.0)
    train(variants["ignore"], scale=1.0)
    train(variants["weak_data"], scale=0.05)

    scores, plan = {}, {}
    for name, m in variants.items():
        scores[name] = metrics(m)
        plan[name] = cf_plan_err(m)

    names = list(variants.keys())
    util = [-plan[n] for n in names]  # higher better
    corr = {
        "spearman_caf_cal": spearman([scores[n]["caf_cal"] for n in names], util),
        "spearman_caf": spearman([scores[n]["caf"] for n in names], util),
        "spearman_neg_pv": spearman([scores[n]["neg_pv"] for n in names], util),
        "spearman_neg_fact": spearman([-scores[n]["fact_err"] for n in names], util),
        "spearman_self_gap": spearman([scores[n]["self_cf_gap"] for n in names], util),
    }
    # pass: calibrated CAF or neg_pv beats fact_err on CF planning
    best_ours = max(corr["spearman_caf_cal"], corr["spearman_neg_pv"])
    pass_k2 = best_ours > corr["spearman_neg_fact"] + 0.05 and best_ours > 0.3

    out = {
        "seed": seed,
        "device": str(DEVICE),
        "seconds": time.time() - t0,
        "scores": scores,
        "cf_plan_errors": plan,
        "correlations": corr,
        "pass_k2v2": pass_k2,
        "note": "CF-sensitive downstream; caf_cal = caf/(1+pv)",
    }
    path = OUT / f"result_seed{seed}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"correlations": corr, "cf_plan_errors": plan, "pass_k2v2": pass_k2}, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
