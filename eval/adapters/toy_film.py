"""Toy FiLM / Ignore adapters — smoke-test the harness (same spirit as CAF synth)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.metrics import Batch, WorldModelAdapter, compute_caf, save_json
from pathlib import Path


class ToyFilmWM(nn.Module, WorldModelAdapter):
    name = "toy_film"

    def __init__(self, a_dim=4, z_dim=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(32, z_dim), nn.GELU())
        self.act = nn.Linear(a_dim, z_dim)
        self.film = nn.Linear(z_dim, z_dim * 2)
        self.head = nn.Linear(z_dim, 32)

    def predict(self, obs, actions):
        # obs: (B,T,32), actions: (B,T-1,A) — predict each next from obs[:,t]
        B, T, D = obs.shape
        outs = []
        for t in range(T - 1):
            z = self.enc(obs[:, t])
            a = self.act(actions[:, t])
            g, b = self.film(a).chunk(2, -1)
            outs.append(self.head(g * z + b))
        return torch.stack(outs, 1)


class ToyIgnoreWM(ToyFilmWM):
    name = "toy_ignore"

    def predict(self, obs, actions):
        B, T, D = obs.shape
        outs = []
        for t in range(T - 1):
            z = self.enc(obs[:, t])
            _ = self.act(actions[:, t]) * 0
            g = torch.ones_like(z)
            b = torch.zeros_like(z)
            outs.append(self.head(g * z + b))
        return torch.stack(outs, 1)


def _make_batch(n=64, T=6, A=4, device="cpu"):
    obs = torch.randn(n, T, 32, device=device)
    actions = torch.randn(n, T - 1, A, device=device)
    # synthetic dynamics: next = obs + 0.2*pad(action)
    next_obs = []
    for t in range(T - 1):
        pad = F.pad(actions[:, t], (0, 32 - A))[:, :32]
        next_obs.append(torch.tanh(obs[:, t] + 0.3 * pad))
    next_obs = torch.stack(next_obs, 1)
    # rebuild obs consistent-ish
    return Batch(obs=obs, actions=actions, next_obs=next_obs)


def run_smoke(out_dir: str = "runs/harness_smoke"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = _make_batch(device=device)
    film = ToyFilmWM().to(device)
    ign = ToyIgnoreWM().to(device)
    # quick train film to listen
    opt = torch.optim.Adam(film.parameters(), lr=1e-3)
    for _ in range(200):
        b = _make_batch(device=device)
        pred = film.predict(b.obs, b.actions)
        loss = F.mse_loss(pred, b.next_obs)
        opt.zero_grad(); loss.backward(); opt.step()

    acts_cf = -batch.actions
    # oracle under CF
    oracle = []
    for t in range(batch.actions.size(1)):
        pad = F.pad(acts_cf[:, t], (0, 32 - 4))[:, :32]
        oracle.append(torch.tanh(batch.obs[:, t] + 0.3 * pad))
    oracle = torch.stack(oracle, 1)

    rf = compute_caf(film, batch, acts_cf, oracle_next=oracle)
    ri = compute_caf(ign, batch, acts_cf, oracle_next=oracle)
    out = {
        "film": rf.to_dict(),
        "ignore": ri.to_dict(),
        "pass_separates": rf.self_cf_gap > 5 * max(ri.self_cf_gap, 1e-8),
        "device": device,
    }
    save_json(out, Path(out_dir) / "result.json")
    print(out)
    return out


if __name__ == "__main__":
    run_smoke()
