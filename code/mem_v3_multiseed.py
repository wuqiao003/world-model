#!/usr/bin/env python3
"""Mem v3 multi-seed sweep — same frozen-memory CF protocol, report monotonic rate."""
import json, os, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse logic compactly
OUT = Path(os.environ.get("OUT_DIR", "/mnt/group/jxdong/wm_exp/runs/mem_v3_multiseed"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

H, W, T, A, D = 24, 24, 12, 4, 64
BATCH = int(os.environ.get("BATCH", "64"))
STEPS = int(os.environ.get("STEPS", "800"))
LR = 3e-4


def env_step(frame, action, hidden):
    ax, ay = action[:, 0], action[:, 1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=frame.device),
        torch.linspace(-1, 1, W, device=frame.device),
        indexing="ij",
    )
    mx = torch.tanh(hidden[:, 0]).view(-1, 1, 1)
    my = torch.tanh(hidden[:, 1]).view(-1, 1, 1)
    mamp = torch.tanh(hidden[:, 2]).view(-1, 1, 1, 1)
    grid = torch.stack([xx, yy], -1).unsqueeze(0).expand(frame.size(0), -1, -1, -1).clone()
    grid[..., 0] = grid[..., 0] + 0.22 * torch.tanh(ax).view(-1, 1, 1)
    grid[..., 1] = grid[..., 1] + 0.22 * torch.tanh(ay).view(-1, 1, 1)
    warped = F.grid_sample(frame, grid, align_corners=True, padding_mode="border")
    r = (xx - 0.5 * mx) ** 2 + (yy - 0.5 * my) ** 2
    nxt = torch.tanh(warped + 0.45 * mamp * torch.exp(-8 * r).unsqueeze(1))
    hidden2 = 0.85 * hidden + 0.15 * F.pad(action, (0, D - A))[:, :D]
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
    def __init__(self, force_w=None, learn_gate=False):
        super().__init__()
        self.force_w = force_w
        self.learn_gate = learn_gate
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, D),
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

    def encode_frames(self, frames_bt):
        b, t, c, h, w = frames_bt.shape
        return self.enc(frames_bt.reshape(b * t, c, h, w)).view(b, t, -1)

    def step_mix(self, z_t, a_t, mem):
        if self.force_w is not None:
            w = torch.full((z_t.size(0), 1), float(self.force_w), device=z_t.device)
        elif self.learn_gate:
            w = torch.sigmoid(self.gate(torch.cat([z_t, mem], -1)))
        else:
            w = torch.zeros(z_t.size(0), 1, device=z_t.device)
        h = z_t + (1 - w) * a_t + w * mem
        return torch.tanh(self.deconv(self.pred(h).view(z_t.size(0), 64, 6, 6))), w

    @torch.no_grad()
    def instant_cf(self, frames, acts):
        b = frames.size(0)
        z = self.encode_frames(frames[:, :-1])
        a = self.act_enc(acts)
        a_neg = self.act_enc(-acts)
        mem = torch.zeros(b, D, device=frames.device)
        preds_f, preds_cf = [], []
        for t in range(acts.size(1)):
            mem = self.mem_cell(a[:, t], mem)
            pf, _ = self.step_mix(z[:, t], a[:, t], mem)
            pcf, _ = self.step_mix(z[:, t], a_neg[:, t], mem.detach())
            preds_f.append(pf); preds_cf.append(pcf)
        return torch.stack(preds_f, 1), torch.stack(preds_cf, 1)


def train(model, steps=STEPS):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(steps):
        fr, ac = make_traj(BATCH)
        # forward train path
        b = fr.size(0)
        z = model.encode_frames(fr[:, :-1])
        a = model.act_enc(ac)
        mem = torch.zeros(b, D, device=fr.device)
        preds = []
        for t in range(ac.size(1)):
            mem = model.mem_cell(a[:, t], mem)
            pred, _ = model.step_mix(z[:, t], a[:, t], mem)
            preds.append(pred)
        loss = F.mse_loss(torch.stack(preds, 1), fr[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def metrics(model, n=96):
    model.eval()
    fr, ac = make_traj(n)
    pf, pcf = model.instant_cf(fr, ac)
    return {
        "fact_err": F.mse_loss(pf, fr[:, 1:]).item(),
        "instant_cf_gap": F.mse_loss(pf, pcf).item(),
    }


def run_seed(seed):
    torch.manual_seed(seed)
    results = {}
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        m = CleanWM(force_w=w).to(DEVICE)
        train(m)
        results[f"force_w_{w}"] = metrics(m)
    lg = CleanWM(learn_gate=True).to(DEVICE)
    train(lg)
    results["learn_gate"] = metrics(lg)
    gaps = [results[f"force_w_{w}"]["instant_cf_gap"] for w in [0.0, 0.25, 0.5, 0.75, 1.0]]
    results["instant_gaps"] = gaps
    results["monotonic"] = all(gaps[i] + 1e-6 >= gaps[i + 1] for i in range(len(gaps) - 1))
    results["w1_zero"] = gaps[-1] < 1e-6
    return results


def main():
    t0 = time.time()
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
    per = {f"seed{s}": run_seed(s) for s in seeds}
    summary = {
        "seeds": seeds,
        "frac_monotonic": sum(per[f"seed{s}"]["monotonic"] for s in seeds) / len(seeds),
        "frac_w1_zero": sum(per[f"seed{s}"]["w1_zero"] for s in seeds) / len(seeds),
        "pass": sum(per[f"seed{s}"]["monotonic"] for s in seeds) >= max(1, len(seeds) - 1),
        "device": str(DEVICE),
        "seconds": time.time() - t0,
    }
    out = {"per_seed": per, "summary": summary}
    path = OUT / "result.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
