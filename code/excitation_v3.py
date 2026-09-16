#!/usr/bin/env python3
"""Excitation v3: ID-matched CF + predictive validity; detect demo extrapolation explosion."""
import json, os, time, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(os.environ.get("OUT_DIR", "/mnt/group/jxdong/wm_exp/runs/excitation_v3"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)

H, W, T, A = 28, 28, 12, 3
BATCH = 48
STEPS = 800
LR = 3e-4


def env_step(frame, action):
    vx, vy, ang = action.unbind(-1)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=frame.device),
        torch.linspace(-1, 1, W, device=frame.device),
        indexing="ij",
    )
    c, s = torch.cos(0.5 * ang), torch.sin(0.5 * ang)
    xr = c.view(-1, 1, 1) * xx - s.view(-1, 1, 1) * yy + 0.25 * torch.tanh(vx).view(-1, 1, 1)
    yr = s.view(-1, 1, 1) * xx + c.view(-1, 1, 1) * yy + 0.25 * torch.tanh(vy).view(-1, 1, 1)
    grid = torch.stack([xr, yr], -1)
    warped = F.grid_sample(frame, grid, align_corners=True, padding_mode="border")
    blob = torch.exp(-(xr ** 2 + yr ** 2) * 6).unsqueeze(1)
    return torch.tanh(warped + 0.4 * torch.tanh(ang).view(-1, 1, 1, 1) * blob)


def sample_actions(n, mode):
    if mode == "demo":
        return 0.05 * torch.randn(n, T - 1, A, device=DEVICE)
    if mode == "random":
        return torch.randn(n, T - 1, A, device=DEVICE)
    if mode == "high_exc":
        base = torch.randn(n, T - 1, A, device=DEVICE)
        t = torch.linspace(0, 2 * math.pi, T - 1, device=DEVICE)
        sweep = torch.stack([torch.sin(t), torch.cos(t), torch.sin(2 * t)], -1).unsqueeze(0).expand(n, -1, -1)
        return 1.5 * base + 0.8 * sweep
    raise ValueError(mode)


def make_traj(n, mode):
    frames = [torch.randn(n, 3, H, W, device=DEVICE) * 0.2]
    acts = sample_actions(n, mode)
    for t in range(T - 1):
        frames.append(env_step(frames[-1], acts[:, t]))
    return torch.stack(frames, 1), acts


def true_cf_frames(frames0, acts):
    """Re-simulate trajectory from first frame with given actions (oracle CF)."""
    frames = [frames0]
    for t in range(acts.size(1)):
        frames.append(env_step(frames[-1], acts[:, t]))
    return torch.stack(frames, 1)


class Pred(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 96),
        )
        self.act = nn.Linear(A, 96)
        self.film = nn.Linear(96, 192)
        self.trunk = nn.Sequential(nn.Linear(96, 96), nn.GELU())
        self.fc = nn.Linear(96, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, frames, acts):
        b = frames.size(0)
        x = frames[:, :-1].reshape(b * (T - 1), 3, H, W)
        z = self.enc(x).view(b, T - 1, -1)
        a = self.act(acts)
        g, be = self.film(a).chunk(2, -1)
        h = g * self.trunk(z) + be
        y = self.fc(h).view(b * (T - 1), 64, 7, 7)
        return torch.tanh(self.deconv(y)).view(b, T - 1, 3, H, W)


def train_on(mode, steps=STEPS):
    model = Pred().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(steps):
        fr, ac = make_traj(BATCH, mode)
        loss = F.mse_loss(model(fr, ac), fr[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_suite(model, n=80):
    model.eval()
    out = {}
    for mode in ["demo", "random", "high_exc"]:
        fr, ac = make_traj(n, mode)
        pred = model(fr, ac)
        fact = F.mse_loss(pred, fr[:, 1:]).item()
        # naive self-gap (can explode OOD)
        self_gap = F.mse_loss(pred, model(fr, -ac)).item()
        # predictive validity: match oracle CF trajectory under -ac
        fr_cf = true_cf_frames(fr[:, 0], -ac)
        # teacher-forced CF from original frames with -ac vs oracle next frames
        pred_cf = model(fr, -ac)
        pv = F.mse_loss(pred_cf, fr_cf[:, 1:]).item()
        # oracle gap magnitude (how much world actually changes)
        oracle_gap = F.mse_loss(fr[:, 1:], fr_cf[:, 1:]).item()
        out[mode] = {
            "fact_err": fact,
            "self_cf_gap": self_gap,
            "pred_validity_vs_oracle_cf": pv,
            "oracle_cf_gap": oracle_gap,
            "pv_ratio": pv / (oracle_gap + 1e-8),  # lower better relative to difficulty
        }
    return out


def main():
    t0 = time.time()
    results = {}
    for train_mode in ["demo", "random", "high_exc"]:
        m = train_on(train_mode)
        results[f"train_{train_mode}"] = eval_suite(m)
    # Primary claim: on high_exc eval, high_exc training has better (lower) pred_validity than demo
    pv_demo = results["train_demo"]["high_exc"]["pred_validity_vs_oracle_cf"]
    pv_he = results["train_high_exc"]["high_exc"]["pred_validity_vs_oracle_cf"]
    # Secondary: demo self_gap on high_exc may be LARGE (explosion) while PV is worse
    results["meta"] = {
        "device": str(DEVICE), "seed": SEED, "seconds": time.time() - t0,
        "hypothesis": "train_high_exc better PV on high_exc than train_demo",
        "pv_demo_on_highexc": pv_demo,
        "pv_highexc_on_highexc": pv_he,
        "pass": pv_he < pv_demo * 0.85,
        "note": "self_cf_gap alone is rejected as primary metric (OOD explosion)",
    }
    path = OUT / f"result_seed{SEED}.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
