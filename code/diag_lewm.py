#!/usr/bin/env python3
"""
Acceptance test for the LeWM eval path.

The released card reports pred/id ~= 0.465 at one plan-step. If our harness
cannot reproduce that order of magnitude, the harness is wrong, not the model.
Sweeps channel order and action-context handling to find the correct convention.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from eval.adapters.lewm_stub import LeWMAdapter, find_bundle  # noqa: E402
from k1k2_real_eval import load_subset  # noqa: E402


@torch.no_grad()
def encode(adapter, pixels, batch=64):
    outs = []
    for i in range(0, len(pixels), batch):
        chunk = torch.from_numpy(np.ascontiguousarray(pixels[i : i + batch])).unsqueeze(1)
        outs.append(adapter.encode_pixels(adapter.preprocess_rgb(chunk))[:, 0])
    return torch.cat(outs, 0)


@torch.no_grad()
def step(adapter, z_ctx, act_ctx_norm, hist_actions=None):
    """One plan-step. z_ctx (B,D). act_ctx_norm (B,BLOCK,2)."""
    hs = adapter.history_size
    b = z_ctx.size(0)
    ctx = z_ctx.unsqueeze(1).expand(b, hs, -1)
    flat = adapter.flatten_plan_step(act_ctx_norm.unsqueeze(1))
    emb = adapter.model.action_encoder(flat)
    ctx_act = torch.zeros(b, hs, emb.size(-1), device=z_ctx.device)
    if hist_actions is not None:
        h_flat = adapter.flatten_plan_step(hist_actions)
        h_emb = adapter.model.action_encoder(h_flat)
        k = min(hs - 1, h_emb.size(1))
        if k > 0:
            ctx_act[:, hs - 1 - k : hs - 1] = h_emb[:, -k:]
    ctx_act[:, -1:] = emb
    return adapter.model.predict(ctx, ctx_act)[:, -1]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pixels, actions, chain, block, horizon = load_subset(Path(sys.argv[1] if len(sys.argv) > 1 else "data/pusht_subset_jpg.h5"))
    print(f"frames={len(pixels)} anchors={len(chain)}", flush=True)

    report = {}
    for mkey in ("lewm_v2", "lewm_v3", "lewm_v4b"):
        try:
            _, ckpt = find_bundle(mkey)
        except FileNotFoundError as e:
            print("skip", mkey, e)
            continue
        adapter = LeWMAdapter(ckpt_path=ckpt, device=device, a_block=block, name=mkey)
        acts = adapter.normalize_actions(torch.from_numpy(actions))
        ch = torch.from_numpy(chain).to(adapter.device)

        for order in ("as_stored_rgb", "flipped_bgr"):
            px = pixels if order == "as_stored_rgb" else pixels[..., ::-1]
            Z = encode(adapter, px)
            z0, z1 = Z[ch[:, 0]], Z[ch[:, 1]]
            id_err = (z0 - z1).pow(2).mean()
            pred = step(adapter, z0, acts[:, 0])
            pred_zero = step(adapter, z0, torch.zeros_like(acts[:, 0]))
            fact = (pred - z1).pow(2).mean()
            fact_zero = (pred_zero - z1).pow(2).mean()
            key = f"{mkey}|{order}"
            report[key] = {
                "latent_rms": float(Z.pow(2).mean().sqrt()),
                "latent_std": float(Z.std()),
                "id_err": float(id_err),
                "fact_err": float(fact),
                "pred_id_ratio": float(fact / id_err),
                "fact_err_zeroact": float(fact_zero),
                "pred_id_ratio_zeroact": float(fact_zero / id_err),
                "action_effect": float((pred - pred_zero).pow(2).mean() / id_err),
            }
            r = report[key]
            print(
                f"{key:28s} rms={r['latent_rms']:.3f} id={r['id_err']:.5f} "
                f"fact={r['fact_err']:.5f} pid={r['pred_id_ratio']:.3f} "
                f"pid0={r['pred_id_ratio_zeroact']:.3f} aeff={r['action_effect']:.3f}",
                flush=True,
            )
        del adapter
        torch.cuda.empty_cache()

    out = Path("runs/diag_lewm/result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("WROTE", out)


if __name__ == "__main__":
    main()
