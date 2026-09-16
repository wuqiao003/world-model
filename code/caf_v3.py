#!/usr/bin/env python3
"""CAF v3: stronger action coupling + absolute thresholds + predictive validity vs oracle CF."""
import json, os, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(os.environ.get("OUT_DIR", "/mnt/group/jxdong/wm_exp/runs/caf_v3"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)

H, W, T, A = 32, 32, 8, 4
BATCH = 64
STEPS = 1000
LR = 3e-4


def env_step(frame, action):
    ax, ay, sc, hue = action.unbind(-1)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=frame.device),
        torch.linspace(-1, 1, W, device=frame.device),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], -1).unsqueeze(0).expand(frame.size(0), -1, -1, -1).clone()
    # STRONG warp
    grid[..., 0] = grid[..., 0] + 0.35 * torch.tanh(ax).view(-1, 1, 1)
    grid[..., 1] = grid[..., 1] + 0.35 * torch.tanh(ay).view(-1, 1, 1)
    warped = F.grid_sample(frame, grid, align_corners=True, padding_mode="border")
    r = ((xx - 0.4 * torch.tanh(ax).view(-1, 1, 1)) ** 2 + (yy - 0.4 * torch.tanh(ay).view(-1, 1, 1)) ** 2)
    blob = torch.exp(-6 * r).unsqueeze(1)
    scale = (1 + 0.6 * torch.tanh(sc).view(-1, 1, 1, 1)) * blob
    shift = 0.4 * torch.tanh(hue).view(-1, 1, 1, 1) * blob
    return torch.tanh(warped * (1 + scale) + shift)


def make_traj(n):
    frames = [torch.randn(n, 3, H, W, device=DEVICE) * 0.3]
    acts = []
    for _ in range(T - 1):
        a = torch.randn(n, A, device=DEVICE)
        acts.append(a)
        frames.append(env_step(frames[-1], a))
    return torch.stack(frames, 1), torch.stack(acts, 1)


def oracle_cf(frames0, acts):
    frames = [frames0]
    for t in range(acts.size(1)):
        frames.append(env_step(frames[-1], acts[:, t]))
    return torch.stack(frames, 1)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 128),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        return self.net(x.reshape(b * t, c, h, w)).view(b, t, -1)


class FilmPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.act = nn.Linear(A, 128)
        self.film = nn.Linear(128, 256)
        self.trunk = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
        self.dec = nn.Linear(128, 64 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, frames, acts):
        z = self.enc(frames[:, :-1])
        a = self.act(acts)
        gamma, beta = self.film(a).chunk(2, -1)
        h = gamma * self.trunk(z) + beta
        x = self.dec(h).view(-1, 64, 8, 8)
        return torch.tanh(self.deconv(x)).view(frames.size(0), T - 1, 3, H, W)


class IgnorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.trunk = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
        self.dec = nn.Linear(128, 64 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        self.dead = nn.Linear(A, 1)

    def forward(self, frames, acts):
        z = self.enc(frames[:, :-1])
        _ = self.dead(acts) * 0
        h = self.trunk(z)
        x = self.dec(h).view(-1, 64, 8, 8)
        return torch.tanh(self.deconv(x)).view(frames.size(0), T - 1, 3, H, W)


def train(model, steps=STEPS):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    hist = []
    for i in range(steps):
        fr, ac = make_traj(BATCH)
        loss = F.mse_loss(model(fr, ac), fr[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 200 == 0:
            hist.append((i, float(loss)))
    return hist


@torch.no_grad()
def caf_scores(model, n=128):
    model.eval()
    fr, ac = make_traj(n)
    pred_f = model(fr, ac)
    pred_cf = model(fr, -ac)
    fr_cf = oracle_cf(fr[:, 0], -ac)
    fact_err = F.mse_loss(pred_f, fr[:, 1:]).item()
    self_gap = F.mse_loss(pred_f, pred_cf).item()
    pv = F.mse_loss(pred_cf, fr_cf[:, 1:]).item()
    oracle_gap = F.mse_loss(fr[:, 1:], fr_cf[:, 1:]).item()
    # CAF composite: want large self_gap AND low pv (faithful CF)
    caf = self_gap / (pv + 1e-8)
    return {
        "fact_err": fact_err,
        "self_cf_gap": self_gap,
        "pred_validity_cf": pv,
        "oracle_cf_gap": oracle_gap,
        "caf_score": caf,
    }


def main():
    t0 = time.time()
    film = FilmPredictor().to(DEVICE)
    ign = IgnorePredictor().to(DEVICE)
    hist_f = train(film)
    hist_i = train(ign)
    scores = {
        "film": caf_scores(film),
        "ignore": caf_scores(ign),
        "train_hist_film": hist_f,
        "train_hist_ignore": hist_i,
        "device": str(DEVICE),
        "seed": SEED,
        "seconds": time.time() - t0,
    }
    # absolute: film should have higher CAF and higher self_gap than ignore; ignore pv ~ fact (can't do CF)
    scores["pass"] = (
        scores["film"]["self_cf_gap"] > 0.01
        and scores["film"]["caf_score"] > 2 * scores["ignore"]["caf_score"]
        and scores["film"]["self_cf_gap"] > 5 * max(scores["ignore"]["self_cf_gap"], 1e-8)
    )
    path = OUT / f"result_seed{SEED}.json"
    path.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
