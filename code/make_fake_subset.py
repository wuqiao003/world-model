#!/usr/bin/env python3
"""Schema-identical fake PushT subset for plumbing tests (no real data needed)."""
from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pusht_fake.h5")
    ap.add_argument("--anchors", type=int, default=48)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()

    import h5py

    rng = np.random.default_rng(0)
    A, H, B = args.anchors, args.horizon, args.block
    n_frames = A * (H + 1)

    # smooth-ish frames so the encoder sees structure rather than pure noise
    base = rng.integers(40, 200, size=(A, 224, 224, 3), dtype=np.uint16)
    frames = np.empty((n_frames, 224, 224, 3), dtype=np.uint8)
    for a in range(A):
        for h in range(H + 1):
            shift = 3 * h
            img = np.roll(base[a], shift, axis=0)
            img = np.roll(img, shift, axis=1)
            frames[a * (H + 1) + h] = np.clip(img, 0, 255).astype(np.uint8)

    chain = np.arange(n_frames, dtype=np.int64).reshape(A, H + 1)
    actions = np.zeros((A, H, B, 6), dtype=np.float32)
    actions[..., 0] = rng.normal(0, 0.005, size=(A, H, B))
    actions[..., 1] = rng.normal(0, 0.006, size=(A, H, B))

    with h5py.File(args.out, "w") as f:
        f.create_dataset("pixels", data=frames, compression="gzip", compression_opts=1)
        f.create_dataset("actions", data=actions)
        f.create_dataset("chain", data=chain)
        f.create_dataset("proprio", data=rng.normal(0, 1, (n_frames, 7)).astype(np.float32))
        f.attrs["block"] = B
        f.attrs["horizon"] = H
        f.attrs["source"] = "FAKE"
        f.attrs["channel_order"] = "RGB"
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
