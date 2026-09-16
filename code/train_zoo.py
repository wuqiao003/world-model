#!/usr/bin/env python3
"""
Schedule the pre-registered training zoo across the node's GPUs.

Runs one training job per GPU and starts the next queued config as soon as a
GPU frees up. Skips configs whose checkpoint already exists so the launcher is
restartable after a dropped connection.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def resolve(val, epochs):
    if isinstance(val, str) and val.startswith("EPOCHS"):
        _, _, denom = val.partition("/")
        return max(1, epochs // int(denom)) if denom else epochs
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="code/train_zoo_spec.json")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--out", default="/mnt/group/jxdong/wm_exp/ckpts/trained")
    ap.add_argument("--logs", default="/mnt/group/jxdong/wm_exp/logs/train")
    ap.add_argument("--data", default="/root/wm_data")
    ap.add_argument("--src", default="/mnt/group/jxdong/wm_exp/ckpts/lewm_pusht")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--only", default="", help="comma list of names to run")
    ap.add_argument("--extra", default="", help="extra flags appended to every job")
    args = ap.parse_args()

    spec = json.load(open(args.spec))
    models = spec["models"]
    if args.only:
        want = set(args.only.split(","))
        models = [m for m in models if m["name"] in want]

    Path(args.logs).mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    queue = []
    for m in models:
        ck = Path(args.out) / m["name"] / f"{m['name']}.ckpt"
        if ck.exists():
            print(f"skip {m['name']} (exists)")
            continue
        queue.append(m)
    print(f"queued {len(queue)} jobs on {len(gpus)} gpus, epochs={args.epochs}")

    running = {}  # gpu -> (proc, name, t0)
    t_start = time.time()
    while queue or running:
        for g in gpus:
            if g in running or not queue:
                continue
            m = queue.pop(0)
            cmd = [
                "python3", "code/train_wm.py",
                "--name", m["name"], "--data", args.data, "--src", args.src,
                "--out", args.out, "--epochs", str(args.epochs),
                "--batch", str(args.batch), "--workers", str(args.workers),
            ]
            for k, v in m["args"].items():
                cmd += [f"--{k}", str(resolve(v, args.epochs))]
            if args.extra:
                cmd += args.extra.split()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
            log = open(Path(args.logs) / f"{m['name']}.log", "w")
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            running[g] = (p, m["name"], time.time())
            print(f"  gpu{g} <- {m['name']}  ({' '.join(cmd[2:])})", flush=True)

        time.sleep(20)
        for g, (p, name, t0) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            mins = (time.time() - t0) / 60
            print(f"  gpu{g} done {name} rc={rc} {mins:.0f}min", flush=True)
            del running[g]

    print(f"ZOO DONE in {(time.time()-t_start)/60:.0f} min")


if __name__ == "__main__":
    main()
