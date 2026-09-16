#!/usr/bin/env python3
"""
K1 smoke: LeWM latent CAF on synthetic RGB (no PushT env required).

Compares fact rollout vs negated-action CF in latent space.
Does NOT claim predictive validity vs downstream — that needs real demos / MPPI proxy.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

# repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.adapters.lewm_stub import LeWMAdapter, resolve_ckpt_dir
from eval.metrics import Batch, compute_caf, save_json

OUT = Path(os.environ.get("OUT_DIR", "runs/k1_lewm_latent_caf"))
OUT.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = resolve_ckpt_dir()
    print("CKPT", ckpt, "DEVICE", device)

    adapter = LeWMAdapter(ckpt_dir=ckpt, device=device)
    B, T = int(os.environ.get("BATCH", "4")), 3
    # synthetic frames — protocol smoke only
    obs = torch.randint(0, 255, (B, T, 224, 224, 3), dtype=torch.uint8)
    # raw Δxy ~ N(0, 8mm)
    actions = 0.008 * torch.randn(B, T - 1, 2)
    actions_cf = -actions

    pixels = adapter.preprocess_rgb(obs.to(device) if False else obs)
    emb = adapter.encode_pixels(pixels.to(adapter.device))
    # teacher: encode next frames as oracle latents for on-policy fact
    # CF oracle unavailable without env → report self_gap + fact_err only; mark PV weak
    batch = Batch(obs=emb[:, :-1].contiguous(), actions=actions, next_obs=emb[:, 1:].contiguous())
    # For CF we only have self-gap unless env; use negated next as non-oracle stand-in disabled
    res = compute_caf(adapter, batch, actions_cf, oracle_next=None)

    # Also compute pure self-gap under CF from same start emb
    pred_f = adapter.predict(emb[:, :1], actions[:, :1])
    pred_cf = adapter.predict(emb[:, :1], actions_cf[:, :1])
    gap1 = float(torch.nn.functional.mse_loss(pred_f, pred_cf).item())

    out = {
        "device": str(adapter.device),
        "ckpt_dir": str(ckpt),
        "seconds": time.time() - t0,
        "caf_result": res.to_dict(),
        "one_step_self_gap": gap1,
        "note": "Synthetic RGB smoke; PV without env is weak stand-in. Next: real PushT frames + MPPI proxy.",
        "pass_load": True,
        "pass_action_sensitive": gap1 > 1e-6,
    }
    path = OUT / "result.json"
    save_json(out, path)
    print(json.dumps(out, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
