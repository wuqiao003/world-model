#!/usr/bin/env python3
"""
Convert the PushT-FR3 HDF5 into a raw memmap plus a small metadata npz.

Eight training jobs run concurrently on one node. Reading pixels through HDF5
in each job serialises on decompression and would need ~10 GB of RAM per job;
a raw uint8 memmap on local NVMe lets all jobs share the OS page cache instead,
which makes the epochs GPU-bound.

Writes:
  <out>/pixels.u8    (N, 224, 224, 3) uint8, native BGR (the order LeWM trained on)
  <out>/meta.npz     action, proprio, episode_idx, ep_len, ep_offset, shape
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="/mnt/group/jxdong/wm_exp/data/pusht_lewm_fr3.h5")
    ap.add_argument("--out", default="/root/wm_data")
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()

    import h5py

    os.makedirs(args.out, exist_ok=True)
    px_path = os.path.join(args.out, "pixels.u8")
    t0 = time.time()

    with h5py.File(args.inp, "r") as f:
        px = f["pixels"]
        n, h, w, c = px.shape
        print(f"pixels {px.shape} {px.dtype} -> {px.nbytes/1e9:.1f} GB", flush=True)

        mm = np.memmap(px_path, dtype=np.uint8, mode="w+", shape=(n, h, w, c))
        for i in range(0, n, args.chunk):
            j = min(n, i + args.chunk)
            mm[i:j] = px[i:j]
            if (i // args.chunk) % 10 == 0:
                done = j / n
                el = time.time() - t0
                print(
                    f"  {j}/{n} ({done*100:.1f}%) {el:.0f}s "
                    f"eta {el/max(done,1e-6)*(1-done):.0f}s",
                    flush=True,
                )
        mm.flush()
        del mm

        np.savez(
            os.path.join(args.out, "meta.npz"),
            action=f["action"][:].astype(np.float32),
            proprio=f["proprio"][:].astype(np.float32),
            episode_idx=f["episode_idx"][:],
            ep_len=f["ep_len"][:],
            ep_offset=f["ep_offset"][:],
            shape=np.asarray([n, h, w, c], dtype=np.int64),
        )

    print(f"DONE {px_path}  {os.path.getsize(px_path)/1e9:.1f} GB  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
