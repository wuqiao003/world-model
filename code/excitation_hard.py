#!/usr/bin/env python3
"""Excitation-aware data hard: pixel-only loss; compare demo vs random vs high-excitation."""
import json, os, time, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(os.environ.get("OUT_DIR", "/mnt/group/jxdong/wm_exp/runs/excitation_hard"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)

H, W, T, A = 28, 28, 12, 3
BATCH = 48
STEPS = 700
LR = 3e-4


def env_step(frame, action):
    # nonlinear: action rotates a blob and shifts color; hard to learn from demos with tiny actions
    vx, vy, ang = action.unbind(-1)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=frame.device),
        torch.linspace(-1, 1, W, device=frame.device),
        indexing="ij",
    )
    c, s = torch.cos(0.4 * ang), torch.sin(0.4 * ang)
    xr = c.view(-1, 1, 1) * xx - s.view(-1, 1, 1) * yy + 0.2 * torch.tanh(vx).view(-1, 1, 1)
    yr = s.view(-1, 1, 1) * xx + c.view(-1, 1, 1) * yy + 0.2 * torch.tanh(vy).view(-1, 1, 1)
    grid = torch.stack([xr, yr], -1)
    warped = F.grid_sample(frame, grid, align_corners=True, padding_mode="border")
    blob = torch.exp(-(xr ** 2 + yr ** 2) * 6).unsqueeze(1)
    nxt = torch.tanh(warped + 0.35 * torch.tanh(ang).view(-1, 1, 1, 1) * blob)
    return nxt


def excitation(actions):
    # log det of action cov proxy + energy
    # actions: B,T-1,A
    flat = actions.reshape(-1, A)
    energy = flat.pow(2).mean()
    centered = flat - flat.mean(0, keepdim=True)
    cov = (centered.T @ centered) / max(flat.size(0) - 1, 1) + 1e-3 * torch.eye(A, device=flat.device)
    sign, logdet = torch.slogdet(cov)
    return (energy + logdet).item()


def sample_actions(n, mode):
    if mode == "demo":
        # small near-zero actions (behavior cloning demos)
        return 0.05 * torch.randn(n, T - 1, A, device=DEVICE)
    if mode == "random":
        return torch.randn(n, T - 1, A, device=DEVICE)
    if mode == "high_exc":
        # cover extremes + orthogonal directions
        base = torch.randn(n, T - 1, A, device=DEVICE)
        # amplify and add structured sweeps
        t = torch.linspace(0, 2 * math.pi, T - 1, device=DEVICE)
        sweep = torch.stack([torch.sin(t), torch.cos(t), torch.sin(2 * t)], -1)
        sweep = sweep.unsqueeze(0).expand(n, -1, -1)
        return 1.5 * base + 0.8 * sweep
    raise ValueError(mode)


def make_traj(n, mode):
    frames = [torch.randn(n, 3, H, W, device=DEVICE) * 0.2]
    acts = sample_actions(n, mode)
    for t in range(T - 1):
        frames.append(env_step(frames[-1], acts[:, t]))
    return torch.stack(frames, 1), acts


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
        h = self.trunk(z)
        h = g * h + be
        y = self.fc(h).view(b * (T - 1), 64, 7, 7)
        pred = torch.tanh(self.deconv(y)).view(b, T - 1, 3, H, W)
        return pred


def train_on(mode, steps=STEPS):
    model = Pred().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for i in range(steps):
        fr, ac = make_traj(BATCH, mode)
        pred = model(fr, ac)
        # PIXEL ONLY — no delta/action aux
        loss = F.mse_loss(pred, fr[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_model(model, n=96):
    model.eval()
    # evaluate on held-out high-excitation rollouts + counterfactual
    fr, ac = make_traj(n, "high_exc")
    pred = model(fr, ac)
    fact = F.mse_loss(pred, fr[:, 1:]).item()
    pred_cf = model(fr, -ac)
    cf_gap = F.mse_loss(pred, pred_cf).item()
    # excitation of training-like batches for reporting
    exc = {m: excitation(sample_actions(n, m)) for m in ["demo", "random", "high_exc"]}
    return {"fact_err_on_highexc": fact, "cf_gap": cf_gap, "train_mode_exc_ref": exc}


def main():
    t0 = time.time()
    out = {}
    for mode in ["demo", "random", "high_exc"]:
        m = train_on(mode)
        out[mode] = eval_model(m)
        out[mode]["train_exc_sample"] = excitation(sample_actions(128, mode))
    out["meta"] = {"device": str(DEVICE), "seed": SEED, "seconds": time.time() - t0, "loss": "pixel_only"}
    # pass if high_exc has higher cf_gap than demo
    out["pass"] = out["high_exc"]["cf_gap"] > out["demo"]["cf_gap"] * 1.5
    path = OUT / f"result_seed{SEED}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
