#!/usr/bin/env python3
"""Mem-Ctrl clean: no FiLM bypass; memory and action enter only via gated mix."""
import json, os, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(os.environ.get("OUT_DIR", "/mnt/group/jxdong/wm_exp/runs/mem_clean"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)

H, W, T, A, D = 24, 24, 10, 4, 64
BATCH = 64
STEPS = 600
LR = 3e-4


def env_step(frame, action, hidden):
    # dynamics depend on BOTH last action AND a latent memory of past actions
    ax, ay, sc, hue = action.unbind(-1)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=frame.device),
        torch.linspace(-1, 1, W, device=frame.device),
        indexing="ij",
    )
    # memory modulates drift center
    mx = torch.tanh(hidden[:, 0]).view(-1, 1, 1)
    my = torch.tanh(hidden[:, 1]).view(-1, 1, 1)
    grid = torch.stack([xx, yy], -1).unsqueeze(0).expand(frame.size(0), -1, -1, -1).clone()
    grid[..., 0] = grid[..., 0] + 0.12 * torch.tanh(ax).view(-1, 1, 1) + 0.08 * mx
    grid[..., 1] = grid[..., 1] + 0.12 * torch.tanh(ay).view(-1, 1, 1) + 0.08 * my
    warped = F.grid_sample(frame, grid, align_corners=True, padding_mode="border")
    r = (xx - 0.25 * torch.tanh(ax).view(-1, 1, 1)) ** 2 + (yy - 0.25 * torch.tanh(ay).view(-1, 1, 1)) ** 2
    blob = torch.exp(-10 * r).unsqueeze(1)
    nxt = torch.tanh(warped + 0.2 * torch.tanh(hue).view(-1, 1, 1, 1) * blob)
    # update hidden with EMA of actions
    hidden2 = 0.7 * hidden + 0.3 * F.pad(action, (0, D - A))[:, :D]
    return nxt, hidden2


def make_traj(n):
    frames = [torch.randn(n, 3, H, W, device=DEVICE) * 0.25]
    hidden = torch.zeros(n, D, device=DEVICE)
    acts = []
    for _ in range(T - 1):
        a = torch.randn(n, A, device=DEVICE)
        acts.append(a)
        fr, hidden = env_step(frames[-1], a, hidden)
        frames.append(fr)
    return torch.stack(frames, 1), torch.stack(acts, 1)


class CleanWM(nn.Module):
    """
    Only pathway for action and memory is gated mix into latent.
    No separate FiLM that can bypass memory.
    force_w: if not None, force memory gate weight.
    """

    def __init__(self, force_w=None, learn_gate=False):
        super().__init__()
        self.force_w = force_w
        self.learn_gate = learn_gate
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, D),
        )
        self.act_enc = nn.Linear(A, D)
        self.mem_cell = nn.GRUCell(D, D)
        if learn_gate:
            self.gate = nn.Sequential(nn.Linear(D * 2, 64), nn.GELU(), nn.Linear(64, 1))
        self.pred = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 64 * 6 * 6))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def encode(self, frames):
        b, t, c, h, w = frames.shape
        return self.enc(frames.reshape(b * t, c, h, w)).view(b, t, -1)

    def forward(self, frames, acts, return_w=False):
        b = frames.size(0)
        z = self.encode(frames[:, :-1])  # B,T-1,D
        a = self.act_enc(acts)
        mem = torch.zeros(b, D, device=frames.device)
        preds = []
        ws = []
        for t in range(acts.size(1)):
            mem = self.mem_cell(a[:, t], mem)
            if self.force_w is not None:
                w = torch.full((b, 1), self.force_w, device=frames.device)
            elif self.learn_gate:
                w = torch.sigmoid(self.gate(torch.cat([z[:, t], mem], -1)))
            else:
                w = torch.zeros(b, 1, device=frames.device)
            # CLEAN mix: only this channel mixes mem vs action-current
            mixed = (1 - w) * a[:, t] + w * mem
            h = z[:, t] + mixed  # additive, no FiLM bypass
            x = self.pred(h).view(b, 64, 6, 6)
            preds.append(torch.tanh(self.deconv(x)))
            ws.append(w)
        pred = torch.stack(preds, 1)
        if return_w:
            return pred, torch.stack(ws, 1).mean().item()
        return pred


def train(model, steps=STEPS):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    last = 0.0
    for i in range(steps):
        fr, ac = make_traj(BATCH)
        pred = model(fr, ac)
        loss = F.mse_loss(pred, fr[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss)
    return last


@torch.no_grad()
def metrics(model, n=96):
    model.eval()
    fr, ac = make_traj(n)
    pred, mean_w = model(fr, ac, return_w=True)
    fact = F.mse_loss(pred, fr[:, 1:]).item()
    pred_cf = model(fr, -ac)
    cf_gap = F.mse_loss(pred, pred_cf).item()
    # response ratio vs frame noise
    fr_n = fr.clone()
    fr_n[:, :-1] = fr_n[:, :-1] + 0.05 * torch.randn_like(fr_n[:, :-1])
    pred_n = model(fr_n, ac)
    noise = F.mse_loss(pred, pred_n).item()
    return {
        "fact_err": fact,
        "cf_gap": cf_gap,
        "noise_gap": noise,
        "resp_ratio": cf_gap / (noise + 1e-8),
        "mean_w": mean_w,
    }


def main():
    t0 = time.time()
    results = {}
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        m = CleanWM(force_w=w).to(DEVICE)
        loss = train(m)
        results[f"force_w_{w}"] = {**metrics(m), "train_loss": loss}
    lg = CleanWM(learn_gate=True).to(DEVICE)
    loss = train(lg)
    results["learn_gate"] = {**metrics(lg), "train_loss": loss}
    results["meta"] = {"device": str(DEVICE), "seed": SEED, "seconds": time.time() - t0}
    # expected: resp_ratio decreases as force_w increases (more memory, less instant action)
    rr = [results[f"force_w_{w}"]["resp_ratio"] for w in [0.0, 0.25, 0.5, 0.75, 1.0]]
    results["monotonic_decreasing_resp"] = all(rr[i] >= rr[i + 1] - 0.05 for i in range(len(rr) - 1))
    path = OUT / f"result_seed{SEED}.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
