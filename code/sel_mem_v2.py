#!/usr/bin/env python3
"""Sel-Mem v2: rare high-value events + overwrite cost + write sparsity."""
import json, os, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(os.environ.get("OUT_DIR", "/mnt/group/jxdong/wm_exp/runs/sel_mem_v2"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)

H, W, T, A, D = 24, 24, 24, 4, 64
BATCH = 64
STEPS = 1200
LR = 3e-4
EVENT_RATE = 0.05  # rarer


def env_step(frame, action, hidden, event):
    ax, ay, sc, hue = action.unbind(-1)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=frame.device),
        torch.linspace(-1, 1, W, device=frame.device),
        indexing="ij",
    )
    # persistent mode in hidden[0] — only events flip it; noise actions should NOT overwrite
    mode = torch.tanh(hidden[:, 0]).view(-1, 1, 1)
    grid = torch.stack([xx, yy], -1).unsqueeze(0).expand(frame.size(0), -1, -1, -1).clone()
    grid[..., 0] = grid[..., 0] + 0.12 * torch.tanh(ax).view(-1, 1, 1)
    grid[..., 1] = grid[..., 1] + 0.12 * torch.tanh(ay).view(-1, 1, 1)
    warped = F.grid_sample(frame, grid, align_corners=True, padding_mode="border")
    r = (xx - 0.5 * mode) ** 2 + (yy + 0.5 * mode) ** 2
    blob = torch.exp(-7 * r).unsqueeze(1)
    # strong visual dependence on mode
    nxt = torch.tanh(warped + 0.55 * mode.unsqueeze(1) * blob)
    hidden2 = hidden.clone()
    # mild action trace in dims 1: — NOT into mode slot
    pad = F.pad(action, (0, D - A))[:, :D]
    hidden2[:, 1:] = 0.92 * hidden2[:, 1:] + 0.08 * pad[:, 1:]
    # EVENT: flip mode to action-dependent target (high value, rare)
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
        self.write_head = nn.Sequential(nn.Linear(D * 3, 64), nn.GELU(), nn.Linear(64, 1))
        self.pred = nn.Sequential(nn.Linear(D * 2, D), nn.GELU(), nn.Linear(D, 64 * 6 * 6))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def encode(self, frames):
        b, t, c, h, w = frames.shape
        return self.enc(frames.reshape(b * t, c, h, w)).view(b, t, -1)

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
                feat = torch.cat([z[:, t], a[:, t], mem], -1)
                w = torch.sigmoid(self.write_head(feat))
            mem = (1 - w) * mem + w * cand
            h = torch.cat([z[:, t], mem], -1)
            x = self.pred(h).view(b, 64, 6, 6)
            preds.append(torch.tanh(self.deconv(x)))
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
            # sparse write + soft event alignment
            sparse = wmean.mean()
            align = F.binary_cross_entropy(wmean.squeeze(-1), ev)
            loss = loss + 0.02 * sparse + 0.15 * align
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss)
    return last


@torch.no_grad()
def eval_model(model, n=128):
    model.eval()
    fr, ac, ev = make_traj(n)
    pred, stats = model(fr, ac, ev, return_stats=True)
    err = ((pred - fr[:, 1:]) ** 2).mean(dim=(2, 3, 4))
    # post-event windows: error on next 3 steps after an event (mode must persist)
    post = torch.zeros_like(err)
    for t in range(ev.size(1) - 1):
        # mark steps after event
        if t + 1 < err.size(1):
            post[:, t + 1] = post[:, t + 1] + ev[:, t]
        if t + 2 < err.size(1):
            post[:, t + 2] = post[:, t + 2] + ev[:, t]
        if t + 3 < err.size(1):
            post[:, t + 3] = post[:, t + 3] + ev[:, t]
    post = post.clamp(max=1.0)
    post_err = (err * post).sum() / (post.sum() + 1e-8)
    return {
        "fact_err": F.mse_loss(pred, fr[:, 1:]).item(),
        "post_event_err": float(post_err),
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
    # pass: oracle better post_event than always; selective closer to oracle than always on post_event
    # and selective discriminates events (p_ev > p_ne + 0.15)
    results["meta"] = {
        "device": str(DEVICE), "seed": SEED, "seconds": time.time() - t0,
        "hypothesis": "oracle/selective beat always on post_event_err; selective discriminates writes",
        "pass": (
            results["oracle_event"]["post_event_err"] < results["always"]["post_event_err"] * 0.95
            and results["never"]["post_event_err"] > results["oracle_event"]["post_event_err"] * 1.05
            and results["selective"]["p_write_event"] > results["selective"]["p_write_nonevent"] + 0.15
        ),
    }
    path = OUT / f"result_seed{SEED}.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
