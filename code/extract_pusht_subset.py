#!/usr/bin/env python3
"""
Extract a compact PushT-FR3 subset directly over HTTP range reads.

The upstream file `pusht_lewm_fr3.h5` is 10.4 GB; the K1/K2 eval only needs
anchor transitions plus the frames a frameskip-5 / H=5 rollout touches.

Output layout (gzip HDF5, a few hundred MB):
  pixels   (M, 224, 224, 3) uint8   RGB (converted from upstream BGR)
  actions  (A, H, BLOCK, 6) float32 raw EE deltas per plan-step
  anchors  (A,) int64               index into pixels of frame t
  chain    (A, H+1) int64           pixel indices for t, t+5, ..., t+H*5
  proprio  (M, 7) float32
  meta     attrs
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

URL = (
    "https://huggingface.co/datasets/Rongxuan-Zhou/pusht_lewm_fr3/"
    "resolve/main/pusht_lewm_fr3.h5"
)
BLOCK = 5   # frameskip / action block
H = 5       # plan-steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pusht_subset.h5")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--anchors-per-ep", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--local", default="data/pusht_lewm_fr3.h5",
                    help="use this local h5 if present instead of HTTP")
    ap.add_argument(
        "--episode-list", default="",
        help="comma-separated episode ids to draw from. Use this to build an "
             "evaluation subset from episodes the trained zoo never saw; without "
             "it the subset spans all episodes, which leaks training data into "
             "the evaluation of any model we trained ourselves",
    )
    ap.add_argument(
        "--val-split", type=int, default=0,
        help="alternative to --episode-list: reproduce train_wm.py's held-out "
             "split (default_rng(12345).permutation(n_ep)[:N]) and use those N",
    )
    args = ap.parse_args()

    import h5py

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.local and os.path.exists(args.local):
        print(f"reading local {args.local}", flush=True)
        src = h5py.File(args.local, "r")
    else:
        import fsspec

        fs_file = fsspec.open(args.url, mode="rb", block_size=16 * 1024 * 1024).open()
        src = h5py.File(fs_file, "r")
    print("keys:", list(src.keys()), flush=True)

    ep_len = src["ep_len"][:]
    ep_offset = src["ep_offset"][:]
    n_ep = len(ep_len)
    print(f"episodes={n_ep} frames={int(ep_len.sum())}", flush=True)

    span = H * BLOCK  # frames consumed by one anchor chain
    if args.episode_list:
        pool = np.array([int(x) for x in args.episode_list.split(",")])
    elif args.val_split:
        # must match train_wm.py exactly
        pool = np.random.default_rng(12345).permutation(n_ep)[: args.val_split]
        print(f"held-out episodes: {sorted(pool.tolist())}", flush=True)
    else:
        pool = np.arange(n_ep)
    ep_ids = pool[np.linspace(0, len(pool) - 1, min(args.episodes, len(pool))).astype(int)]

    # Collect (global_frame_idx chains, action blocks)
    chains, act_blocks = [], []
    for e in ep_ids:
        off, ln = int(ep_offset[e]), int(ep_len[e])
        if ln < span + BLOCK + 2:
            continue
        hi = ln - span - 1
        starts = rng.choice(np.arange(0, hi), size=min(args.anchors_per_ep, hi), replace=False)
        for s in sorted(starts.tolist()):
            idx = off + s + BLOCK * np.arange(H + 1)
            chains.append(idx)
            act_blocks.append(off + s + np.arange(H * BLOCK))
    chains = np.asarray(chains, dtype=np.int64)
    act_blocks = np.asarray(act_blocks, dtype=np.int64)
    print(f"anchors={len(chains)}", flush=True)

    # Unique frames to fetch
    uniq = np.unique(chains.reshape(-1))
    remap = {int(g): i for i, g in enumerate(uniq)}
    print(f"unique frames={len(uniq)} (~{len(uniq)*224*224*3/1e6:.0f} MB raw)", flush=True)

    px_src = src["pixels"]
    prop_src = src["proprio"]
    act_src = src["action"]

    with h5py.File(args.out, "w") as dst:
        d_px = dst.create_dataset(
            "pixels", shape=(len(uniq), 224, 224, 3), dtype="uint8",
            compression="gzip", compression_opts=4, chunks=(8, 224, 224, 3),
        )
        # fetch in monotonically increasing runs to keep range reads sequential
        step = 64
        for i in range(0, len(uniq), step):
            gidx = uniq[i : i + step]
            block = px_src[gidx[0] : gidx[-1] + 1]
            sel = block[gidx - gidx[0]]
            # Keep the upstream OpenCV BGR order: that is what the released
            # LeWM encoders were trained on and what their inference code feeds.
            d_px[i : i + len(gidx)] = sel
            if (i // step) % 5 == 0:
                print(f"  frames {i}/{len(uniq)}  {time.time()-t0:.0f}s", flush=True)

        dst.create_dataset("proprio", data=prop_src[:][uniq].astype(np.float32))
        dst.create_dataset("episode_idx", data=src["episode_idx"][:][uniq])
        acts = act_src[:][act_blocks.reshape(-1)].astype(np.float32)
        acts = acts.reshape(len(chains), H, BLOCK, 6)
        dst.create_dataset("actions", data=acts)
        dst.create_dataset(
            "chain",
            data=np.vectorize(lambda g: remap[int(g)])(chains).astype(np.int64),
        )
        dst.create_dataset("anchor_global", data=chains[:, 0])
        dst.attrs["block"] = BLOCK
        dst.attrs["horizon"] = H
        dst.attrs["source"] = "Rongxuan-Zhou/pusht_lewm_fr3"
        dst.attrs["channel_order"] = "BGR"
        dst.attrs["episodes"] = len(ep_ids)
        dst.attrs["episode_ids"] = np.asarray(sorted(ep_ids.tolist()), dtype=np.int64)
        dst.attrs["val_split"] = args.val_split

    src.close()
    sz = os.path.getsize(args.out) / 1e6
    print(f"WROTE {args.out}  {sz:.0f} MB  in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
