#!/usr/bin/env python3
"""CAF hard: nonlinear env, pixel-only loss, no delta aux."""
import json, os, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(os.environ.get("OUT_DIR", "/mnt/group/jxdong/wm_exp/runs/caf_hard"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)

H, W, T, A = 32, 32, 8, 4
BATCH = 64
STEPS = 800
LR = 3e-4


def env_step(frame, action):
    """Nonlinear action-conditioned dynamics (no explicit delta target)."""
    # action -> soft spatial warp field + local nonlinear color shift
    ax, ay, sc, hue = action.unbind(-1)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=frame.device),
        torch.linspace(-1, 1, W, device=frame.device),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], -1).unsqueeze(0).expand(frame.size(0), -1, -1, -1).clone()
    grid[..., 0] = grid[..., 0] + 0.15 * torch.tanh(ax).view(-1, 1, 1)
    grid[..., 1] = grid[..., 1] + 0.15 * torch.tanh(ay).view(-1, 1, 1)
    warped = F.grid_sample(frame, grid, align_corners=True, padding_mode="border")
    # nonlinear local modulation
    r = ((xx - 0.3 * torch.tanh(ax).view(-1, 1, 1)) ** 2 + (yy - 0.3 * torch.tanh(ay).view(-1, 1, 1)) ** 2)
    blob = torch.exp(-8 * r).unsqueeze(1)
    scale = (1 + 0.4 * torch.tanh(sc).view(-1, 1, 1, 1)) * blob
    shift = 0.25 * torch.tanh(hue).view(-1, 1, 1, 1) * blob
    nxt = torch.tanh(warped * (1 + scale) + shift)
    return nxt


def make_traj(n):
    frames = [torch.randn(n, 3, H, W, device=DEVICE) * 0.3]
    acts = []
    for _ in range(T - 1):
        a = torch.randn(n, A, device=DEVICE)
        acts.append(a)
        frames.append(env_step(frames[-1], a))
    return torch.stack(frames, 1), torch.stack(acts, 1)  # (B,T,3,H,W), (B,T-1,A)


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
        self.dec = nn.Sequential(
            nn.Linear(128, 64 * 8 * 8), nn.GELU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, frames, acts):
        # predict next from current frame + action
        z = self.enc(frames[:, :-1])  # B,T-1,128
        a = self.act(acts)
        gamma, beta = self.film(a).chunk(2, -1)
        h = self.trunk(z)
        h = gamma * h + beta
        x = self.dec(h).view(-1, 64, 8, 8)
        pred = torch.tanh(self.deconv(x)).view(frames.size(0), T - 1, 3, H, W)
        return pred


class IgnorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.trunk = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
        self.dec = nn.Sequential(nn.Linear(128, 64 * 8 * 8), nn.GELU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        # absorb action but zero it
        self.dead = nn.Linear(A, 1)

    def forward(self, frames, acts):
        z = self.enc(frames[:, :-1])
        _ = self.dead(acts) * 0
        h = self.trunk(z)
        x = self.dec(h).view(-1, 64, 8, 8)
        pred = torch.tanh(self.deconv(x)).view(frames.size(0), T - 1, 3, H, W)
        return pred


def train(model, steps=STEPS):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    losses = []
    model.train()
    for i in range(steps):
        fr, ac = make_traj(BATCH)
        pred = model(fr, ac)
        loss = F.mse_loss(pred, fr[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 100 == 0:
            losses.append((i, float(loss)))
    return losses


@torch.no_grad()
def caf_scores(model, n=128):
    model.eval()
    fr, ac = make_traj(n)
    # factual
    pred_f = model(fr, ac)
    # counterfactual: negate action
    pred_cf = model(fr, -ac)
    # random action
    pred_rand = model(fr, torch.randn_like(ac))
    # same action twice (stability)
    pred_f2 = model(fr, ac)

    def mse(a, b):
        return ((a - b) ** 2).mean().item()

    fact_err = mse(pred_f, fr[:, 1:])
    cf_gap = mse(pred_f, pred_cf)  # should be large if listens
    rand_gap = mse(pred_f, pred_rand)
    stab = mse(pred_f, pred_f2)
    # sensitivity: response to action vs response to frame noise
    fr_n = fr.clone(); fr_n[:, :-1] = fr_n[:, :-1] + 0.05 * torch.randn_like(fr_n[:, :-1])
    pred_noise = model(fr_n, ac)
    noise_gap = mse(pred_f, pred_noise)
    resp_ratio = cf_gap / (noise_gap + 1e-8)
    insens_neg = 1.0 / (1.0 + cf_gap)  # high if ignores negate
    gain = cf_gap - fact_err  # rough: action-induced change beyond error
    return {
        "fact_err": fact_err,
        "cf_gap_negate": cf_gap,
        "cf_gap_rand": rand_gap,
        "stability": stab,
        "noise_gap": noise_gap,
        "resp_ratio": resp_ratio,
        "insens_negate": insens_neg,
        "gain": gain,
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
        "train_hist_film": hist_f[-3:],
        "train_hist_ignore": hist_i[-3:],
        "device": str(DEVICE),
        "seed": SEED,
        "seconds": time.time() - t0,
    }
    # pass criterion: film resp_ratio >> ignore, film cf_gap >> ignore
    scores["pass"] = (
        scores["film"]["resp_ratio"] > 2 * scores["ignore"]["resp_ratio"]
        and scores["film"]["cf_gap_negate"] > 3 * max(scores["ignore"]["cf_gap_negate"], 1e-6)
    )
    path = OUT / f"result_seed{SEED}.json"
    path.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
