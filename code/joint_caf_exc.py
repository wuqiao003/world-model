#!/usr/bin/env python3
"""Joint demo: CAF separates film/ignore AND excitation improves PV — one script, multi-seed."""
import json, os, time, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(os.environ.get("OUT_DIR", "/mnt/group/jxdong/wm_exp/runs/joint_caf_exc"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

H, W, T, A = 28, 28, 10, 3
BATCH = int(os.environ.get("BATCH", "48"))
STEPS = int(os.environ.get("STEPS", "700"))
LR = 3e-4


def env_step(frame, action):
    vx, vy, ang = action.unbind(-1)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=frame.device),
        torch.linspace(-1, 1, W, device=frame.device),
        indexing="ij",
    )
    c, s = torch.cos(0.45 * ang), torch.sin(0.45 * ang)
    xr = c.view(-1, 1, 1) * xx - s.view(-1, 1, 1) * yy + 0.22 * torch.tanh(vx).view(-1, 1, 1)
    yr = s.view(-1, 1, 1) * xx + c.view(-1, 1, 1) * yy + 0.22 * torch.tanh(vy).view(-1, 1, 1)
    warped = F.grid_sample(frame, torch.stack([xr, yr], -1), align_corners=True, padding_mode="border")
    blob = torch.exp(-(xr ** 2 + yr ** 2) * 6).unsqueeze(1)
    return torch.tanh(warped + 0.35 * torch.tanh(ang).view(-1, 1, 1, 1) * blob)


def sample_actions(n, mode):
    if mode == "demo":
        return 0.05 * torch.randn(n, T - 1, A, device=DEVICE)
    if mode == "high_exc":
        t = torch.linspace(0, 2 * math.pi, T - 1, device=DEVICE)
        sweep = torch.stack([torch.sin(t), torch.cos(t), torch.sin(2 * t)], -1).unsqueeze(0).expand(n, -1, -1)
        return 1.4 * torch.randn(n, T - 1, A, device=DEVICE) + 0.7 * sweep
    return torch.randn(n, T - 1, A, device=DEVICE)


def make_traj(n, mode):
    frames = [torch.randn(n, 3, H, W, device=DEVICE) * 0.2]
    acts = sample_actions(n, mode)
    for t in range(T - 1):
        frames.append(env_step(frames[-1], acts[:, t]))
    return torch.stack(frames, 1), acts


def oracle_cf(frames0, acts):
    frames = [frames0]
    for t in range(acts.size(1)):
        frames.append(env_step(frames[-1], acts[:, t]))
    return torch.stack(frames, 1)


class Pred(nn.Module):
    def __init__(self, listen=True):
        super().__init__()
        self.listen = listen
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 96),
        )
        self.act = nn.Linear(A, 96)
        self.film = nn.Linear(96, 192)
        self.trunk = nn.Sequential(nn.Linear(96, 96), nn.GELU())
        self.fc = nn.Linear(96, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        self.dead = nn.Linear(A, 1)

    def forward(self, frames, acts):
        b = frames.size(0)
        z = self.enc(frames[:, :-1].reshape(b * (T - 1), 3, H, W)).view(b, T - 1, -1)
        if self.listen:
            a = self.act(acts)
            g, be = self.film(a).chunk(2, -1)
            h = g * self.trunk(z) + be
        else:
            _ = self.dead(acts) * 0
            h = self.trunk(z)
        y = self.fc(h).view(b * (T - 1), 64, 7, 7)
        return torch.tanh(self.deconv(y)).view(b, T - 1, 3, H, W)


def train(model, mode, steps=STEPS):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(steps):
        fr, ac = make_traj(BATCH, mode)
        loss = F.mse_loss(model(fr, ac), fr[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def caf_bundle(model, n=96, eval_mode="high_exc"):
    model.eval()
    fr, ac = make_traj(n, eval_mode)
    pred = model(fr, ac)
    pred_cf = model(fr, -ac)
    fr_cf = oracle_cf(fr[:, 0], -ac)
    self_gap = F.mse_loss(pred, pred_cf).item()
    pv = F.mse_loss(pred_cf, fr_cf[:, 1:]).item()
    fact = F.mse_loss(pred, fr[:, 1:]).item()
    return {
        "fact_err": fact,
        "self_cf_gap": self_gap,
        "pred_validity_cf": pv,
        "caf": self_gap / (pv + 1e-8),
    }


def run_seed(seed):
    torch.manual_seed(seed)
    film_demo = Pred(True).to(DEVICE)
    film_he = Pred(True).to(DEVICE)
    ign = Pred(False).to(DEVICE)
    train(film_demo, "demo")
    train(film_he, "high_exc")
    train(ign, "high_exc")
    return {
        "film_train_demo": caf_bundle(film_demo),
        "film_train_high_exc": caf_bundle(film_he),
        "ignore": caf_bundle(ign),
    }


def main():
    t0 = time.time()
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
    all_res = {}
    for s in seeds:
        all_res[f"seed{s}"] = run_seed(s)
    # aggregate claims
    caf_film = [all_res[f"seed{s}"]["film_train_high_exc"]["caf"] for s in seeds]
    caf_ign = [all_res[f"seed{s}"]["ignore"]["caf"] for s in seeds]
    pv_demo = [all_res[f"seed{s}"]["film_train_demo"]["pred_validity_cf"] for s in seeds]
    pv_he = [all_res[f"seed{s}"]["film_train_high_exc"]["pred_validity_cf"] for s in seeds]
    summary = {
        "mean_caf_film_highexc": sum(caf_film) / len(caf_film),
        "mean_caf_ignore": sum(caf_ign) / len(caf_ign),
        "mean_pv_demo": sum(pv_demo) / len(pv_demo),
        "mean_pv_highexc_train": sum(pv_he) / len(pv_he),
        "pass_caf_separates": all(
            all_res[f"seed{s}"]["film_train_high_exc"]["self_cf_gap"]
            > 5 * max(all_res[f"seed{s}"]["ignore"]["self_cf_gap"], 1e-8)
            for s in seeds
        ),
        "pass_excitation_pv": all(pv_he[i] < pv_demo[i] * 0.9 for i in range(len(seeds))),
        "seconds": time.time() - t0,
        "device": str(DEVICE),
        "seeds": seeds,
    }
    out = {"per_seed": all_res, "summary": summary}
    path = OUT / "result.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
