#!/usr/bin/env python3
"""Excitation v4: stronger dynamics + joint criteria (PV + fact_err), multi-seed."""
import json, os, time, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(os.environ.get("OUT_DIR", "/mnt/group/jxdong/wm_exp/runs/excitation_v4"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

H, W, T, A = 32, 32, 16, 3
BATCH = int(os.environ.get("BATCH", "48"))
STEPS = int(os.environ.get("STEPS", "1000"))
LR = 3e-4


def env_step(frame, action):
    """Stronger nonlinear action effect so excitation matters more."""
    vx, vy, ang = action.unbind(-1)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=frame.device),
        torch.linspace(-1, 1, W, device=frame.device),
        indexing="ij",
    )
    # large rotation + translation; magnitude saturates less for big |a|
    amp = 0.55 * torch.tanh(ang) + 0.25 * torch.tanh(3 * ang)
    c, s = torch.cos(amp), torch.sin(amp)
    tx = 0.35 * torch.tanh(vx) + 0.15 * torch.tanh(2 * vx)
    ty = 0.35 * torch.tanh(vy) + 0.15 * torch.tanh(2 * vy)
    xr = c.view(-1, 1, 1) * xx - s.view(-1, 1, 1) * yy + tx.view(-1, 1, 1)
    yr = s.view(-1, 1, 1) * xx + c.view(-1, 1, 1) * yy + ty.view(-1, 1, 1)
    warped = F.grid_sample(frame, torch.stack([xr, yr], -1), align_corners=True, padding_mode="border")
    blob = torch.exp(-(xr ** 2 + yr ** 2) * 5).unsqueeze(1)
    return torch.tanh(warped + 0.5 * torch.tanh(ang).view(-1, 1, 1, 1) * blob)


def sample_actions(n, mode):
    if mode == "demo":
        # near-zero BC-like actions (low excitation)
        return 0.04 * torch.randn(n, T - 1, A, device=DEVICE)
    if mode == "high_exc":
        t = torch.linspace(0, 2 * math.pi, T - 1, device=DEVICE)
        sweep = torch.stack([torch.sin(t), torch.cos(t), torch.sin(2 * t)], -1).unsqueeze(0).expand(n, -1, -1)
        # cover extremes
        return 1.8 * torch.randn(n, T - 1, A, device=DEVICE) + 1.0 * sweep
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
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 128),
        )
        self.act = nn.Linear(A, 128)
        self.film = nn.Linear(128, 256)
        self.trunk = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
        self.fc = nn.Linear(128, 64 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, frames, acts):
        b = frames.size(0)
        z = self.enc(frames[:, :-1].reshape(b * (T - 1), 3, H, W)).view(b, T - 1, -1)
        a = self.act(acts)
        g, be = self.film(a).chunk(2, -1)
        h = g * self.trunk(z) + be
        y = self.fc(h).view(b * (T - 1), 64, 8, 8)
        return torch.tanh(self.deconv(y)).view(b, T - 1, 3, H, W)


def train(model, mode, steps=STEPS):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(steps):
        fr, ac = make_traj(BATCH, mode)
        loss = F.mse_loss(model(fr, ac), fr[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def eval_on(model, mode, n=96):
    model.eval()
    fr, ac = make_traj(n, mode)
    pred = model(fr, ac)
    pred_cf = model(fr, -ac)
    fr_cf = oracle_cf(fr[:, 0], -ac)
    fact = F.mse_loss(pred, fr[:, 1:]).item()
    self_gap = F.mse_loss(pred, pred_cf).item()
    pv = F.mse_loss(pred_cf, fr_cf[:, 1:]).item()
    return {
        "fact_err": fact,
        "self_cf_gap": self_gap,
        "pred_validity_cf": pv,
        "caf": self_gap / (pv + 1e-8),
    }


def run_seed(seed):
    torch.manual_seed(seed)
    demo = Pred().to(DEVICE)
    he = Pred().to(DEVICE)
    train(demo, "demo")
    train(he, "high_exc")
    return {
        "train_demo_eval_highexc": eval_on(demo, "high_exc"),
        "train_highexc_eval_highexc": eval_on(he, "high_exc"),
        "train_demo_eval_demo": eval_on(demo, "demo"),
        "train_highexc_eval_demo": eval_on(he, "demo"),
    }


def main():
    t0 = time.time()
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
    per = {f"seed{s}": run_seed(s) for s in seeds}
    # primary: on high_exc eval, high_exc train has lower PV AND lower fact_err
    pv_ok = []
    fact_ok = []
    for s in seeds:
        d = per[f"seed{s}"]
        a, b = d["train_demo_eval_highexc"], d["train_highexc_eval_highexc"]
        pv_ok.append(b["pred_validity_cf"] < a["pred_validity_cf"])
        fact_ok.append(b["fact_err"] < a["fact_err"])
        # secondary signal: demo has larger self_gap (OOD inflation)
        d["ood_self_gap_inflation"] = a["self_cf_gap"] > b["self_cf_gap"] * 1.2
    summary = {
        "seeds": seeds,
        "frac_pv_better": sum(pv_ok) / len(pv_ok),
        "frac_fact_better": sum(fact_ok) / len(fact_ok),
        "frac_ood_gap_inflation": sum(
            1 for s in seeds if per[f"seed{s}"]["ood_self_gap_inflation"]
        ) / len(seeds),
        "pass": (sum(pv_ok) >= 2 and sum(fact_ok) >= 2),  # majority of 3
        "device": str(DEVICE),
        "seconds": time.time() - t0,
        "criterion": "majority: highexc-train better PV and fact_err on highexc-eval",
    }
    out = {"per_seed": per, "summary": summary}
    path = OUT / "result.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
