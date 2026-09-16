#!/usr/bin/env python3
"""
Re-encode a PushT subset with JPEG frames, optionally split into shards.

kubectl's exec channel drops long transfers (a 205 MB file truncated at ~106 MB),
so frames are JPEG-encoded and the result is split into self-contained shards
small enough to copy in one piece. JPEG q=95 on these frames costs ~1.4/255 mean
absolute pixel error, far below the effects the CAF metrics measure.

Each shard holds its own frames plus anchors whose chains are remapped to local
frame indices, so shards can be loaded and concatenated independently.
"""
from __future__ import annotations

import argparse
import io
import os

import numpy as np


def write_shard(src, dst, anchor_idx, quality, tag):
    from PIL import Image

    chain = src["chain"][:][anchor_idx]
    uniq = np.unique(chain.reshape(-1))
    remap = {int(g): i for i, g in enumerate(uniq)}
    local_chain = np.vectorize(lambda g: remap[int(g)])(chain).astype(np.int64)

    import h5py

    vlen = h5py.vlen_dtype(np.uint8)
    d = dst.create_dataset("pixels_jpeg", shape=(len(uniq),), dtype=vlen)
    px = src["pixels"]
    err, n_err = 0.0, 0
    for i, g in enumerate(uniq):
        buf = io.BytesIO()
        frame = px[int(g)]
        Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
        raw = buf.getvalue()
        d[i] = np.frombuffer(raw, dtype=np.uint8)
        if i % 400 == 0:
            back = np.asarray(Image.open(io.BytesIO(raw)))
            err += float(np.abs(back.astype(np.int16) - frame.astype(np.int16)).mean())
            n_err += 1
            print(f"  [{tag}] {i}/{len(uniq)}", flush=True)

    dst.create_dataset("chain", data=local_chain)
    dst.create_dataset("actions", data=src["actions"][:][anchor_idx])
    for k in ("proprio", "episode_idx"):
        if k in src:
            dst.create_dataset(k, data=src[k][:][uniq])
    if "anchor_global" in src:
        dst.create_dataset("anchor_global", data=src["anchor_global"][:][anchor_idx])
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    dst.attrs["encoding"] = "jpeg"
    dst.attrs["jpeg_quality"] = quality
    dst.attrs["jpeg_mean_abs_err"] = err / max(1, n_err)
    return len(uniq), len(anchor_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/pusht_subset.h5")
    ap.add_argument("--out", default="data/pusht_subset_jpg.h5")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    import h5py

    with h5py.File(args.inp, "r") as src:
        n_anchor = src["chain"].shape[0]
        groups = np.array_split(np.arange(n_anchor), max(1, args.shards))
        base, ext = os.path.splitext(args.out)
        for i, g in enumerate(groups):
            path = args.out if args.shards == 1 else f"{base}.shard{i}{ext}"
            with h5py.File(path, "w") as dst:
                nf, na = write_shard(src, dst, g, args.quality, f"shard{i}")
            mb = os.path.getsize(path) / 1e6
            print(f"WROTE {path}  frames={nf} anchors={na}  {mb:.0f} MB")


if __name__ == "__main__":
    main()
