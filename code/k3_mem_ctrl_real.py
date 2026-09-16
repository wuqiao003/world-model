#!/usr/bin/env python3
"""
K3: memory <-> controllability on real LeWM checkpoints (inference-time intervention).

Released checkpoints are frozen, so the memory *weight* cannot be retrained. We
instead intervene on the predictor's context: the history slots are blended
towards a stale latent with weight w, and the number of live context slots L is
varied. This measures whether relying more on stale memory suppresses
instantaneous action sensitivity, which is what the synthetic Mem experiments
predicted (w -> 1 kills instant controllability).

Reported per (ckpt, L, w):
  instant_gap_ratio : frozen-context, flip current action only, / no-op baseline
  pred_id_ratio     : one-step fidelity vs the no-op predictor
  mppi_cos          : downstream open-loop agreement with the expert action

Honest limitation: this is an inference-time ablation of a frozen model, not a
trained memory-capacity sweep.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.adapters.lewm_stub import LeWMAdapter, find_bundle  # noqa: E402
from eval.metrics import save_json  # noqa: E402

EPS = 1e-8


@torch.no_grad()
def encode_all(adapter, pixels, batch=64):
    outs = []
    for i in range(0, len(pixels), batch):
        chunk = torch.from_numpy(pixels[i : i + batch]).unsqueeze(1)
        outs.append(adapter.encode_pixels(adapter.preprocess_rgb(chunk))[:, 0])
    return torch.cat(outs, 0)


@torch.no_grad()
def step_with_memory(
    adapter: LeWMAdapter,
    z_cur: torch.Tensor,
    z_stale: torch.Tensor,
    acts_norm_step: torch.Tensor,
    live_slots: int,
    w: float,
) -> torch.Tensor:
    """
    One plan-step prediction with a memory-blended context.

    The predictor sees `history_size` slots. The newest `live_slots` carry the
    current latent; the rest carry `z_stale`. Every slot is additionally blended
    towards z_stale with weight w, so w=0 is the stock model and w=1 removes all
    information about the current state.
    """
    hs = adapter.history_size
    b = z_cur.size(0)
    slots = []
    for k in range(hs):
        is_live = k >= hs - live_slots
        base = z_cur if is_live else z_stale
        slots.append((1.0 - w) * base + w * z_stale)
    ctx = torch.stack(slots, dim=1)                                  # (B,hs,D)

    act_flat = adapter.flatten_plan_step(acts_norm_step.unsqueeze(1))  # (B,1,block*2)
    act_emb = adapter.model.action_encoder(act_flat)
    ctx_act = torch.zeros(b, hs, act_emb.size(-1), device=z_cur.device)
    ctx_act[:, -1:] = act_emb
    return adapter.model.predict(ctx, ctx_act)[:, -1]


@torch.no_grad()
def mppi_cos_with_memory(
    adapter, z0, z_stale, z_goal, a_true, horizon, block, live_slots, w,
    k=64, iters=6, beta=3.0,
):
    dev = z0.device
    mean = torch.zeros(horizon, block, 2, device=dev)
    std = torch.ones_like(mean)
    for _ in range(iters):
        cand = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            k, horizon, block, 2, device=dev
        )
        z = z0.unsqueeze(0).expand(k, -1)
        stale = z_stale.unsqueeze(0).expand(k, -1)
        for h in range(horizon):
            z = step_with_memory(adapter, z, stale, cand[:, h], live_slots, w)
        cost = (z - z_goal.unsqueeze(0)).pow(2).mean(-1)
        wts = F.softmax(-beta * cost, dim=0)
        mean = (wts.view(-1, 1, 1, 1) * cand).sum(0)
    return float(F.cosine_similarity(mean[0, 0], a_true, dim=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pusht_subset.h5")
    ap.add_argument("--out", default="runs/k3_mem_real/result.json")
    ap.add_argument("--models", default="lewm_v2,lewm_v3,lewm_v4b")
    ap.add_argument("--weights", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--live-slots", default="1,2,3")
    ap.add_argument("--n-plan", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(0)
    sys.path.insert(0, str(ROOT / "code"))
    from k1k2_real_eval import load_subset

    pixels, actions, chain, block, horizon = load_subset(Path(args.data))

    grid: Dict[str, Dict] = {}
    for mkey in args.models.split(","):
        _, ckpt = find_bundle(mkey)
        adapter = LeWMAdapter(ckpt_path=ckpt, device=args.device, a_block=block, name=mkey)
        Z = encode_all(adapter, pixels)
        acts = adapter.normalize_actions(torch.from_numpy(actions))
        dev = Z.device
        ch = torch.from_numpy(chain).to(dev)
        z0, z1 = Z[ch[:, 0]], Z[ch[:, 1]]
        z_goal = Z[ch[:, -1]]
        # stale memory = the anchor's own oldest available latent surrogate:
        # shift anchors by one so the stale slot is a *different* real state.
        z_stale = torch.roll(z0, shifts=1, dims=0)
        id_err = (z0 - z1).pow(2).mean(-1).mean()
        sel = torch.linspace(0, z0.size(0) - 1, min(args.n_plan, z0.size(0))).long()

        print(f"[{mkey}] latents {tuple(Z.shape)} {time.time()-t0:.0f}s", flush=True)
        for L in [int(x) for x in args.live_slots.split(",")]:
            if L > adapter.history_size:
                continue
            for w in [float(x) for x in args.weights.split(",")]:
                a0 = acts[:, 0]
                p_f = step_with_memory(adapter, z0, z_stale, a0, L, w)
                p_cf = step_with_memory(adapter, z0, z_stale, -a0, L, w)
                gap = (p_f - p_cf).pow(2).mean(-1).mean()
                fact = (p_f - z1).pow(2).mean(-1).mean()

                coss: List[float] = []
                for i in sel.tolist():
                    coss.append(
                        mppi_cos_with_memory(
                            adapter, z0[i], z_stale[i], z_goal[i], acts[i, 0, 0],
                            horizon, block, L, w,
                        )
                    )
                key = f"{mkey}|L{L}|w{w}"
                grid[key] = {
                    "model": mkey,
                    "live_slots": L,
                    "w": w,
                    "instant_gap_ratio": float(gap / (id_err + EPS)),
                    "pred_id_ratio": float(fact / (id_err + EPS)),
                    "mppi_cos": float(np.mean(coss)),
                }
                g = grid[key]
                print(
                    f"  L={L} w={w:.2f}  gapR={g['instant_gap_ratio']:.4f} "
                    f"pid={g['pred_id_ratio']:.3f} cos={g['mppi_cos']:+.3f}",
                    flush=True,
                )
        del adapter, Z
        torch.cuda.empty_cache()

    # endpoint claim: w=1 should collapse instantaneous controllability
    per_model = {}
    for mkey in args.models.split(","):
        rows = [v for v in grid.values() if v["model"] == mkey]
        if not rows:
            continue
        by_w = {}
        for r in rows:
            by_w.setdefault(r["w"], []).append(r["instant_gap_ratio"])
        ws = sorted(by_w)
        means = [float(np.mean(by_w[w])) for w in ws]
        per_model[mkey] = {
            "w_grid": ws,
            "instant_gap_ratio_mean": means,
            "w1_over_w0": means[-1] / (means[0] + EPS),
            "monotone_decreasing": all(
                means[i + 1] <= means[i] + 1e-6 for i in range(len(means) - 1)
            ),
        }

    out = {
        "seconds": time.time() - t0,
        "data": args.data,
        "grid": grid,
        "per_model": per_model,
        "pass_k3": bool(
            per_model and all(v["w1_over_w0"] < 0.2 for v in per_model.values())
        ),
        "note": "inference-time memory-blend intervention on frozen released ckpts",
    }
    save_json(out, Path(args.out))
    print(json.dumps({"per_model": per_model, "pass_k3": out["pass_k3"]}, indent=2))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
