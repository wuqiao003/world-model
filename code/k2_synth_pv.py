#!/usr/bin/env python3
"""
K2-style predictive validity on a controlled suite (main-track intermediate).

Train several WM variants; score CAF / fact_err / self_gap; downstream =
open-loop planning MSE to oracle under random action sequences (higher error = worse).

Claim to stress-test: corr(CAF, -plan_err) > corr(-fact_err, -plan_err).
When LeWM/Matrix arrive, swap adapters — same table schema.
"""
from __future__ import annotations

import json, os, time, math
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(os.environ.get("OUT_DIR", "runs/k2_synth_pv"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

H, W, T, A = 24, 24, 8, 3
BATCH = int(os.environ.get("BATCH", "48"))
STEPS = int(os.environ.get("STEPS", "600"))
LR = 3e-4
PLAN_HORIZON = 5


def env_step(frame, action):
    vx, vy, ang = action.unbind(-1)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=frame.device),
        torch.linspace(-1, 1, W, device=frame.device),
        indexing="ij",
    )
    amp = 0.45 * torch.tanh(ang)
    c, s = torch.cos(amp), torch.sin(amp)
    xr = c.view(-1, 1, 1) * xx - s.view(-1, 1, 1) * yy + 0.28 * torch.tanh(vx).view(-1, 1, 1)
    yr = s.view(-1, 1, 1) * xx + c.view(-1, 1, 1) * yy + 0.28 * torch.tanh(vy).view(-1, 1, 1)
    warped = F.grid_sample(frame, torch.stack([xr, yr], -1), align_corners=True, padding_mode="border")
    blob = torch.exp(-(xr ** 2 + yr ** 2) * 6).unsqueeze(1)
    return torch.tanh(warped + 0.4 * torch.tanh(ang).view(-1, 1, 1, 1) * blob)


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
    def __init__(self, listen=True, noise_act=0.0):
        super().__init__()
        self.listen = listen
        self.noise_act = noise_act
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
        b = frames.size(0)
        tlen = acts.size(1)
        z = self.enc(frames[:, :tlen].reshape(b * tlen, 3, H, W)).view(b, tlen, -1)
        if self.noise_act > 0:
            acts = acts + self.noise_act * torch.randn_like(acts)
        if self.listen:
            a = self.act(acts)
            g, be = self.film(a).chunk(2, -1)
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
def score_metrics(model, n=80) -> Dict[str, float]:
    model.eval()
    fr, ac = make_traj(n, scale=1.2)
    pred = model(fr[:, :-1], ac)
    pred_cf = model(fr[:, :-1], -ac)
    fr_cf = oracle_roll(fr[:, 0], -ac)
    fact = F.mse_loss(pred, fr[:, 1:]).item()
    gap = F.mse_loss(pred, pred_cf).item()
    pv = F.mse_loss(pred_cf, fr_cf[:, 1:]).item()
    return {
        "fact_err": fact,
        "self_cf_gap": gap,
        "pred_validity": pv,
        "caf": gap / (pv + 1e-8),
    }


@torch.no_grad()
def plan_err(model, n=40) -> float:
    """Downstream proxy: open-loop WM rollout vs oracle under held-out actions."""
    model.eval()
    fr, ac = make_traj(n, scale=1.2)
    # use only first frame + actions of length PLAN_HORIZON
    h = min(PLAN_HORIZON, ac.size(1))
    acts = ac[:, :h]
    # teacher-forced one-step cascade from predicted frames (open loop)
    cur = fr[:, 0]
    preds = []
    for t in range(h):
        # model expects frames (B,tlen,C,H,W) and acts (B,tlen,A)
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
        "film": Pred(listen=True).to(DEVICE),
        "ignore": Pred(listen=False).to(DEVICE),
        "noisy_act": Pred(listen=True, noise_act=0.5).to(DEVICE),
        "weak_data": Pred(listen=True).to(DEVICE),  # train on tiny actions
    }
    train(variants["film"], scale=1.0)
    train(variants["ignore"], scale=1.0)
    train(variants["noisy_act"], scale=1.0)
    train(variants["weak_data"], scale=0.05)

    model_scores = {}
    downstream = {}  # higher better → use negative plan_err
    plan_errors = {}
    for name, m in variants.items():
        model_scores[name] = score_metrics(m)
        pe = plan_err(m)
        plan_errors[name] = pe
        downstream[name] = -pe

    names = list(variants.keys())
    caf = [model_scores[n]["caf"] for n in names]
    fact = [-model_scores[n]["fact_err"] for n in names]  # higher better after negate
    util = [downstream[n] for n in names]
    corr = {
        "spearman_caf_vs_neg_plan_err": spearman(caf, util),
        "spearman_neg_fact_vs_neg_plan_err": spearman(fact, util),
        "spearman_neg_pv_vs_neg_plan_err": spearman(
            [-model_scores[n]["pred_validity"] for n in names], util
        ),
    }
    # pass if CAF correlates with utility at least as well as fact_err (and preferably better)
    pass_k2 = (
        corr["spearman_caf_vs_neg_plan_err"] >= corr["spearman_neg_fact_vs_neg_plan_err"] - 0.05
        and corr["spearman_caf_vs_neg_plan_err"] > 0.2
    )
    out = {
        "seed": seed,
        "device": str(DEVICE),
        "seconds": time.time() - t0,
        "model_scores": model_scores,
        "plan_errors": plan_errors,
        "correlations": corr,
        "pass_k2_proxy": pass_k2,
        "note": "Intermediate K2 on controlled suite; replace variants with real WM adapters for main-track K1/K2.",
    }
    path = OUT / f"result_seed{seed}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("correlations", "plan_errors", "pass_k2_proxy", "seconds")}, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
