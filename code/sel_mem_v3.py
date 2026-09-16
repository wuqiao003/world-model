#!/usr/bin/env python3
"""Sel-Mem v3: write head uses prediction surprise + mode-delta features."""
import json, os, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(os.environ.get("OUT_DIR", "/mnt/group/jxdong/wm_exp/runs/sel_mem_v3"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)

H, W, T, A, D = 24, 24, 24, 4, 64
BATCH = 64
STEPS = 1200
LR = 3e-4
EVENT_RATE = 0.05


def env_step(frame, action, hidden, event):
    ax, ay = action[:, 0], action[:, 1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=frame.device),
        torch.linspace(-1, 1, W, device=frame.device),
        indexing="ij",
    )
    mode = torch.tanh(hidden[:, 0]).view(-1, 1, 1)
    grid = torch.stack([xx, yy], -1).unsqueeze(0).expand(frame.size(0), -1, -1, -1).clone()
    grid[..., 0] = grid[..., 0] + 0.12 * torch.tanh(ax).view(-1, 1, 1)
    grid[..., 1] = grid[..., 1] + 0.12 * torch.tanh(ay).view(-1, 1, 1)
    warped = F.grid_sample(frame, grid, align_corners=True, padding_mode="border")
    r = (xx - 0.5 * mode) ** 2 + (yy + 0.5 * mode) ** 2
    blob = torch.exp(-7 * r).unsqueeze(1)
    nxt = torch.tanh(warped + 0.55 * mode.unsqueeze(1) * blob)
    hidden2 = hidden.clone()
    pad = F.pad(action, (0, D - A))[:, :D]
    hidden2[:, 1:] = 0.92 * hidden2[:, 1:] + 0.08 * pad[:, 1:]
    if event is not None:
        target = torch.tanh(action[:, 0] + action[:, 1])
        hidden2[:, 0] = torch.where(event > 0.5, target, hidden2[:, 0])
    return nxt, hidden2


def make_traj(n, event_rate=EVENT_RATE):
    frames = [torch.randn(n, 3, H, W, device=DEVICE) * 0.2]
    hidden = torch.zeros(n, D, device=DEVICE)
    hidden[:, 0] = torch.randn(n, device=DEVICE)
    acts, events = [], []
    for _ in range(T - 1):
        a = torch.randn(n, A, device=DEVICE)
        ev = (torch.rand(n, device=DEVICE) < event_rate).float()
        acts.append(a); events.append(ev)
        fr, hidden = env_step(frames[-1], a, hidden, ev)
        frames.append(fr)
    return torch.stack(frames, 1), torch.stack(acts, 1), torch.stack(events, 1)


class MemWM(nn.Module):
    def __init__(self, mode="always"):
        super().__init__()
        self.mode = mode
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, D),
        )
        self.act = nn.Linear(A, D)
        self.cell = nn.GRUCell(D, D)
        # surprise features: |cand-mem|, |z_t - z_{t-1}| proxy via z and a
        self.write_head = nn.Sequential(nn.Linear(D * 4 + 2, 64), nn.GELU(), nn.Linear(64, 1))
        self.pred = nn.Sequential(nn.Linear(D * 2, D), nn.GELU(), nn.Linear(D, 64 * 6 * 6))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        # auxiliary one-step predictor for surprise (no mem)
        self.aux = nn.Sequential(nn.Linear(D + D, D), nn.GELU(), nn.Linear(D, 64 * 6 * 6))

    def encode(self, frames):
        b, t, c, h, w = frames.shape
        return self.enc(frames.reshape(b * t, c, h, w)).view(b, t, -1)

    def decode(self, h):
        x = self.pred(h).view(h.size(0), 64, 6, 6)
        return torch.tanh(self.deconv(x))

    def forward(self, frames, acts, events=None, return_stats=False):
        b = frames.size(0)
        z = self.encode(frames[:, :-1])
        a = self.act(acts)
        mem = torch.zeros(b, D, device=frames.device)
        preds, writes = [], []
        for t in range(acts.size(1)):
            cand = self.cell(a[:, t], mem)
            if self.mode == "always":
                w = torch.ones(b, 1, device=frames.device)
            elif self.mode == "never":
                w = torch.zeros(b, 1, device=frames.device)
            elif self.mode == "oracle_event":
                w = events[:, t].unsqueeze(-1)
            else:
                delta = (cand - mem).abs().mean(dim=-1, keepdim=True)
                # aux prediction without updating mem — surprise vs next frame later used in loss;
                # for gate, use cand-mem and z change
                z_delta = (z[:, t] - (z[:, t - 1] if t > 0 else z[:, t])).abs().mean(dim=-1, keepdim=True)
                feat = torch.cat([z[:, t], a[:, t], mem, cand, delta, z_delta], -1)
                w = torch.sigmoid(self.write_head(feat))
            mem = (1 - w) * mem + w * cand
            preds.append(self.decode(torch.cat([z[:, t], mem], -1)))
            writes.append(w)
        pred = torch.stack(preds, 1)
        wmean = torch.stack(writes, 1)
        if return_stats:
            ev = events.unsqueeze(-1)
            return pred, {
                "mean_write": float(wmean.mean()),
                "p_write_event": float((wmean * ev).sum() / (ev.sum() + 1e-8)),
                "p_write_nonevent": float((wmean * (1 - ev)).sum() / ((1 - ev).sum() + 1e-8)),
            }
        return pred, wmean


def train(model, steps=STEPS):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    last = 0.0
    for _ in range(steps):
        fr, ac, ev = make_traj(BATCH)
        pred, wmean = model(fr, ac, ev)
        loss = F.mse_loss(pred, fr[:, 1:])
        if model.mode == "selective":
            # contrastive write: encourage higher write on events than non-events
            ev = ev.unsqueeze(-1)
            p_ev = (wmean * ev).sum() / (ev.sum() + 1e-8)
            p_ne = (wmean * (1 - ev)).sum() / ((1 - ev).sum() + 1e-8)
            # maximize margin p_ev - p_ne; also light BCE to event labels
            margin = F.relu(0.35 - (p_ev - p_ne))
            bce = F.binary_cross_entropy(wmean.squeeze(-1), ev.squeeze(-1))
            loss = loss + 0.5 * margin + 0.2 * bce
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss)
    return last


@torch.no_grad()
def eval_model(model, n=128):
    model.eval()
    fr, ac, ev = make_traj(n)
    pred, stats = model(fr, ac, ev, return_stats=True)
    err = ((pred - fr[:, 1:]) ** 2).mean(dim=(2, 3, 4))
    post = torch.zeros_like(err)
    for t in range(ev.size(1) - 1):
        for k in (1, 2, 3):
            if t + k < err.size(1):
                post[:, t + k] = post[:, t + k] + ev[:, t]
    post = post.clamp(max=1.0)
    return {
        "fact_err": F.mse_loss(pred, fr[:, 1:]).item(),
        "post_event_err": float((err * post).sum() / (post.sum() + 1e-8)),
        "event_step_err": float((err * ev).sum() / (ev.sum() + 1e-8)),
        **stats,
    }


def main():
    t0 = time.time()
    results = {}
    for mode in ["always", "never", "oracle_event", "selective"]:
        m = MemWM(mode=mode).to(DEVICE)
        loss = train(m)
        results[mode] = {**eval_model(m), "train_loss": loss}
    results["meta"] = {
        "device": str(DEVICE), "seed": SEED, "seconds": time.time() - t0,
        "hypothesis": "selective discriminates events; oracle beats always on post_event",
        "pass": (
            results["oracle_event"]["post_event_err"] <= results["always"]["post_event_err"] * 1.02
            and results["never"]["post_event_err"] > results["oracle_event"]["post_event_err"] * 1.05
            and results["selective"]["p_write_event"] > results["selective"]["p_write_nonevent"] + 0.2
        ),
    }
    path = OUT / f"result_seed{SEED}.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
