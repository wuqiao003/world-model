#!/usr/bin/env python3
"""
Train a LeWM-style action-conditioned JEPA world model on PushT-FR3.

Why train our own: the released checkpoints give only 3 independent models, and
a bootstrap that resamples checkpoints (the honest unit of independence) is
therefore powerless -- see reports/2026-08-31.md section 9.0. A zoo of models
trained across capacity / data / regularization / memory settings gives both
independent checkpoints and a real quality spectrum, instead of a spectrum
manufactured by corrupting actions.

Reproduces the official recipe (github.com/lucas-maes/le-wm, train.py):
    loss = ||predict(emb[:, :ctx], act[:, :ctx]) - emb[:, n_preds:]||^2
           + lambda * SIGReg(emb)
end-to-end, no EMA and no stop-gradient -- SIGReg alone prevents collapse.

Checkpoints are written in the pickled-JEPA layout the eval adapter already
loads, alongside jepa.py / module.py / action_scaler.json / meta.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class SeqDataset(torch.utils.data.Dataset):
    """Frameskip-strided frame sequences with their per-step action blocks."""

    def __init__(self, root, ep_ids, ctx_len, n_preds, frameskip, a_mean, a_std, frac=1.0, seed=0):
        self.root = str(root)
        meta = np.load(os.path.join(self.root, "meta.npz"))
        self.shape = tuple(meta["shape"])
        self.action = meta["action"][:, :2].astype(np.float32)  # dx, dy
        ep_len, ep_offset = meta["ep_len"], meta["ep_offset"]
        self.ctx_len, self.n_preds, self.fs = ctx_len, n_preds, frameskip
        self.a_mean, self.a_std = a_mean.astype(np.float32), a_std.astype(np.float32)
        self._mm = None

        T = ctx_len + n_preds
        # last pixel touched, and last action tick touched by the ctx actions
        span_px = (T - 1) * frameskip
        span_act = (ctx_len - 1) * frameskip + frameskip - 1
        need = max(span_px, span_act)

        starts = []
        for e in ep_ids:
            off, ln = int(ep_offset[e]), int(ep_len[e])
            if ln <= need:
                continue
            starts.append(off + np.arange(0, ln - need, dtype=np.int64))
        self.starts = np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)

        if frac < 1.0:
            rng = np.random.default_rng(seed)
            k = max(1, int(len(self.starts) * frac))
            self.starts = np.sort(rng.choice(self.starts, size=k, replace=False))

    def _pixels(self):
        if self._mm is None:  # open per worker
            self._mm = np.memmap(
                os.path.join(self.root, "pixels.u8"), dtype=np.uint8, mode="r",
                shape=self.shape,
            )
        return self._mm

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = int(self.starts[i])
        T = self.ctx_len + self.n_preds
        fidx = t + self.fs * np.arange(T)
        px = np.asarray(self._pixels()[fidx])                    # (T,H,W,3) uint8

        # Action for frame f = the next `fs` raw (dx,dy) ticks, standardized.
        # Only the first ctx_len frames are conditioned on (the model consumes
        # act_emb[:, :ctx_len]); later frames are zero-padded, which also keeps
        # the action window inside the episode.
        acts = np.zeros((T, self.fs, 2), dtype=np.float32)
        for i, f in enumerate(fidx[: self.ctx_len]):
            blk = self.action[f : f + self.fs]
            acts[i, : len(blk)] = (blk - self.a_mean) / self.a_std
        return {
            "pixels": torch.from_numpy(px.copy()),
            "action": torch.from_numpy(acts.reshape(T, self.fs * 2)),
        }


def preprocess(px_u8, mean, std):
    """(B,T,H,W,3) uint8 BGR -> (B,T,3,224,224) normalized, order preserved."""
    x = px_u8.float().div_(255.0).permute(0, 1, 4, 2, 3)
    b, t, c, h, w = x.shape
    x = x.reshape(b * t, c, h, w)
    if (h, w) != (224, 224):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    return x.view(b, t, c, 224, 224)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_model(cfg, src_dir):
    sys.path.insert(0, str(src_dir))
    import jepa
    import module
    from transformers import ViTConfig, ViTModel

    H = cfg["enc_hidden"]
    encoder = ViTModel(
        ViTConfig(
            image_size=224,
            patch_size=cfg["patch"],
            hidden_size=H,
            num_hidden_layers=cfg["enc_layers"],
            num_attention_heads=cfg["enc_heads"],
            intermediate_size=4 * H,
        ),
        add_pooling_layer=False,
    )
    return jepa.JEPA(
        encoder=encoder,
        predictor=module.ARPredictor(
            num_frames=cfg["ctx_len"],
            depth=cfg["pred_depth"],
            heads=cfg["pred_heads"],
            mlp_dim=cfg["pred_mlp"],
            input_dim=H,
            hidden_dim=H,
            dim_head=64,
        ),
        action_encoder=module.Embedder(
            input_dim=cfg["frameskip"] * 2,
            smoothed_dim=cfg["frameskip"] * 2,
            emb_dim=H,
            mlp_scale=4,
        ),
        projector=module.MLP(H, 4 * H, H),
        pred_proj=module.MLP(H, 4 * H, H),
    ), module


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="/root/wm_data")
    ap.add_argument("--src", default="/mnt/group/jxdong/wm_exp/ckpts/lewm_pusht")
    ap.add_argument("--out", default="/mnt/group/jxdong/wm_exp/ckpts/trained")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=0.1, help="SIGReg weight")
    ap.add_argument("--ctx-len", type=int, default=3)
    ap.add_argument("--n-preds", type=int, default=1)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--enc-hidden", type=int, default=192)
    ap.add_argument("--enc-layers", type=int, default=12)
    ap.add_argument("--enc-heads", type=int, default=3)
    ap.add_argument("--patch", type=int, default=14)
    ap.add_argument("--pred-depth", type=int, default=6)
    ap.add_argument("--pred-heads", type=int, default=16)
    ap.add_argument("--pred-mlp", type=int, default=2048)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-eps", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epochs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda")
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = np.load(os.path.join(args.data, "meta.npz"))
    n_ep = len(meta["ep_len"])
    rng = np.random.default_rng(12345)  # split fixed across the whole zoo
    perm = rng.permutation(n_ep)
    val_ids, train_ids = perm[: args.val_eps], perm[args.val_eps :]

    # action scaler from training episodes only
    tr_mask = np.isin(meta["episode_idx"], train_ids)
    a_tr = meta["action"][tr_mask][:, :2]
    a_mean, a_std = a_tr.mean(0), a_tr.std(0).clip(1e-8)
    print(f"[{args.name}] action mean={a_mean} std={a_std}", flush=True)

    cfg = {k: getattr(args, k) for k in (
        "ctx_len", "n_preds", "frameskip", "enc_hidden", "enc_layers", "enc_heads",
        "patch", "pred_depth", "pred_heads", "pred_mlp", "lam", "epochs", "frac",
        "seed", "lr",
    )}

    ds_kw = dict(ctx_len=args.ctx_len, n_preds=args.n_preds, frameskip=args.frameskip,
                 a_mean=a_mean, a_std=a_std)
    train_ds = SeqDataset(args.data, train_ids, frac=args.frac, seed=args.seed, **ds_kw)
    val_ds = SeqDataset(args.data, val_ids, **ds_kw)
    print(f"[{args.name}] train={len(train_ds)} val={len(val_ds)}", flush=True)

    dl_kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=True,
                 persistent_workers=args.workers > 0, drop_last=True)
    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, **dl_kw)
    val_dl = torch.utils.data.DataLoader(val_ds, shuffle=False, **dl_kw)

    model, module_mod = build_model(cfg, args.src)
    model.to(dev)
    sigreg = module_mod.SIGReg(knots=17, num_proj=1024).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{args.name}] params {n_par/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.max_steps or args.epochs * len(train_dl)
    warm = max(1, int(args.warmup * total))

    def lr_at(step):
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    mean, std = IMAGENET_MEAN.to(dev), IMAGENET_STD.to(dev)
    ctx, npred = args.ctx_len, args.n_preds

    def forward(batch):
        x = preprocess(batch["pixels"].to(dev, non_blocking=True), mean, std)
        info = {"pixels": x, "action": torch.nan_to_num(
            batch["action"].to(dev, non_blocking=True), 0.0)}
        out = model.encode(info)
        emb, act_emb = out["emb"], out["act_emb"]
        pred = model.predict(emb[:, :ctx], act_emb[:, :ctx])
        pred_loss = (pred - emb[:, npred:]).pow(2).mean()
        sig = sigreg(emb.transpose(0, 1))
        return pred_loss, sig, emb

    hist, step, t0 = [], 0, time.time()
    # lr_hi is expected to diverge; keep the best finite-val weights so a
    # diverged run contributes a genuinely bad model rather than NaN metrics.
    best = {"val": float("inf"), "state": None, "epoch": -1}
    n_nonfinite = 0
    for ep in range(args.epochs):
        model.train()
        run = {"pred": 0.0, "sig": 0.0, "n": 0}
        for batch in train_dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_loss, sig, _ = forward(batch)
                loss = pred_loss + args.lam * sig
            if not torch.isfinite(loss):
                n_nonfinite += 1
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run["pred"] += float(pred_loss)
            run["sig"] += float(sig)
            run["n"] += 1
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        model.eval()
        vp, vn, rms = 0.0, 0, 0.0
        with torch.no_grad():
            for batch in val_dl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pl, _, emb = forward(batch)
                vp += float(pl)
                rms += float(emb.float().pow(2).mean().sqrt())
                vn += 1
                if vn >= 40:
                    break
        rec = {
            "epoch": ep,
            "train_pred": run["pred"] / max(1, run["n"]),
            "train_sig": run["sig"] / max(1, run["n"]),
            "val_pred": vp / max(1, vn),
            "latent_rms": rms / max(1, vn),
            "secs": time.time() - t0,
            "nonfinite_steps": n_nonfinite,
        }
        hist.append(rec)
        if math.isfinite(rec["val_pred"]) and rec["val_pred"] < best["val"]:
            best = {
                "val": rec["val_pred"],
                "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "epoch": ep,
            }
        print(
            f"[{args.name}] ep{ep:03d} pred={rec['train_pred']:.5f} "
            f"sig={rec['train_sig']:.4f} val={rec['val_pred']:.5f} "
            f"rms={rec['latent_rms']:.3f} {rec['secs']:.0f}s",
            flush=True,
        )
        with open(out_dir / "history.json", "w") as f:
            json.dump(hist, f, indent=2)
        if args.max_steps and step >= args.max_steps:
            break

    # save in the layout eval/adapters/lewm_stub.py loads
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"[{args.name}] restored best epoch {best['epoch']} val={best['val']:.5f}", flush=True)
    model.eval().to("cpu")
    for p in model.parameters():
        p.requires_grad_(False)
    torch.save(model, out_dir / f"{args.name}.ckpt")
    for fn in ("jepa.py", "module.py"):
        shutil.copy(Path(args.src) / fn, out_dir / fn)
    with open(out_dir / "action_scaler.json", "w") as f:
        json.dump({"mean": a_mean.tolist(), "std": a_std.tolist()}, f)
    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "name": args.name, "history_size": args.ctx_len, "a_block": args.frameskip,
            "params_M": n_par / 1e6, "config": cfg,
            "val_pred_final": hist[-1]["val_pred"] if hist else None,
            "latent_rms_final": hist[-1]["latent_rms"] if hist else None,
            "best_epoch": best["epoch"],
            "best_val_pred": best["val"] if best["state"] is not None else None,
            "nonfinite_steps": n_nonfinite,
            "val_episodes": val_ids.tolist(),
        }, f, indent=2)
    print(f"[{args.name}] SAVED {out_dir}", flush=True)


if __name__ == "__main__":
    main()
