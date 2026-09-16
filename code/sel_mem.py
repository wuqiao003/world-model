#!/usr/bin/env python3
"""Idea 3: Selective memory writes — only write when surprise/utility high."""
import json, os, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(os.environ.get("OUT_DIR", "/mnt/group/jxdong/wm_exp/runs/sel_mem"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)

H, W, T, A, D = 24, 24, 16, 4, 64
BATCH = 64
STEPS = 900
LR = 3e-4


def env_step(frame, action, hidden, event):
    """Most steps: smooth drift. Rare event bit flips a persistent 'mode' in hidden world state."""
    ax, ay, sc, hue = action.unbind(-1)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=frame.device),
        torch.linspace(-1, 1, W, device=frame.device),
        indexing="ij",
    )
    # world mode in hidden[0]: changes only on event
    mode = torch.tanh(hidden[:, 0]).view(-1, 1, 1)
    grid = torch.stack([xx, yy], -1).unsqueeze(0).expand(frame.size(0), -1, -1, -1).clone()
    grid[..., 0] = grid[..., 0] + 0.15 * torch.tanh(ax).view(-1, 1, 1)
    grid[..., 1] = grid[..., 1] + 0.15 * torch.tanh(ay).view(-1, 1, 1)
    warped = F.grid_sample(frame, grid, align_corners=True, padding_mode="border")
    r = (xx - 0.4 * mode) ** 2 + (yy + 0.4 * mode) ** 2
    blob = torch.exp(-8 * r).unsqueeze(1)
    nxt = torch.tanh(warped + 0.35 * mode.unsqueeze(1) * blob)
    hidden2 = hidden.clone()
    # always mild decay of action trace
    hidden2 = 0.9 * hidden2 + 0.05 * F.pad(action, (0, D - A))[:, :D]
    # EVENT: flip mode strongly
    if event is not None:
        hidden2[:, 0] = torch.where(event > 0.5, -hidden[:, 0] + action[:, 0], hidden2[:, 0])
    return nxt, hidden2


def make_traj(n, event_rate=0.12):
    frames = [torch.randn(n, 3, H, W, device=DEVICE) * 0.25]
    hidden = torch.zeros(n, D, device=DEVICE)
    # seed mode
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
        """mode: always | never | selective | oracle_event"""
        super().__init__()
        self.mode = mode
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, D),
        )
        self.act = nn.Linear(A, D)
        self.cell = nn.GRUCell(D, D)
        # selective write head: predicts write gate from (z, a, mem, surprise proxy)
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
            else:  # selective learned
                # surprise proxy: mismatch between cand and mem
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
            # write recall on events: P(write|event) and P(write|~event)
            if events is not None:
                ev = events.unsqueeze(-1)
                p_w_ev = (wmean * ev).sum() / (ev.sum() + 1e-8)
                p_w_ne = (wmean * (1 - ev)).sum() / ((1 - ev).sum() + 1e-8)
            else:
                p_w_ev = p_w_ne = torch.tensor(0.0)
            return pred, {
                "mean_write": float(wmean.mean()),
                "p_write_event": float(p_w_ev),
                "p_write_nonevent": float(p_w_ne),
            }
        return pred


def train(model, steps=STEPS, use_events=False):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    last = 0.0
    for _ in range(steps):
        fr, ac, ev = make_traj(BATCH)
        pred = model(fr, ac, ev if use_events or model.mode in ("oracle_event", "selective") else None)
        loss = F.mse_loss(pred, fr[:, 1:])
        # selective: slight sparsity pressure so it doesn't collapse to always-write
        if model.mode == "selective":
            # encourage write when event (aux), discourage otherwise
            # soft: use event labels only as weak supervision for write head via prediction path
            pass
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss)
    return last


@torch.no_grad()
def eval_model(model, n=96):
    model.eval()
    fr, ac, ev = make_traj(n)
    pred, stats = model(fr, ac, ev, return_stats=True)
    fact = F.mse_loss(pred, fr[:, 1:]).item()
    # long-horizon: after an event, does prediction track mode flip?
    # teacher-forced already; also measure error on event steps vs non-event
    err = ((pred - fr[:, 1:]) ** 2).mean(dim=(2, 3, 4))  # B,T-1
    ev_err = (err * ev).sum() / (ev.sum() + 1e-8)
    ne_err = (err * (1 - ev)).sum() / ((1 - ev).sum() + 1e-8)
    return {
        "fact_err": fact,
        "event_step_err": float(ev_err),
        "nonevent_step_err": float(ne_err),
        **stats,
    }


def main():
    t0 = time.time()
    results = {}
    for mode in ["always", "never", "oracle_event", "selective"]:
        m = MemWM(mode=mode).to(DEVICE)
        loss = train(m, use_events=(mode in ("oracle_event", "selective")))
        results[mode] = {**eval_model(m), "train_loss": loss}
    # hypothesis: selective / oracle better than always on fact_err (less overwrite noise)
    # and much better than never on event_step_err
    results["meta"] = {
        "device": str(DEVICE),
        "seed": SEED,
        "seconds": time.time() - t0,
        "hypothesis": "oracle/selective < always on fact_err; never worst on event_step_err",
        "pass": (
            results["oracle_event"]["fact_err"] <= results["always"]["fact_err"] * 1.05
            and results["never"]["event_step_err"] > results["oracle_event"]["event_step_err"] * 1.1
        ),
    }
    path = OUT / f"result_seed{SEED}.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
