#!/usr/bin/env python3
"""
Run the three pre-registered K1/K2 analyses across seeds and GPUs.

  primary   : self-trained checkpoints, `id` pipeline only
              -> quality varies only through the models themselves, and every
                 system is scored on episodes none of them trained on
  released  : primary + the 3 released checkpoints
              -> spans self-trained vs official, but the released models did see
                 these episodes during training; reported separately for that reason
  corrupt   : all checkpoints x 5 action pipelines
              -> the old design, kept to quantify how much action corruption
                 inflates the apparent advantage of fidelity metrics
  memrep    : the memory sweep plus its seed replicates, `id` pipeline only
              -> K3 only, kept out of the analyses above on purpose

Membership comes from the pre-registered spec files, not from whatever happens to
be on disk. The seed replicates of the memory sweep would otherwise raise the
memory axis from 5/18 to 15/28 of the primary zoo, so a K3 power increase would
silently rewrite the K1/K2 model population.

Each analysis runs one process per seed, one seed per GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

RELEASED = ["lewm_v2", "lewm_v3", "lewm_v4b"]
SPEC = "code/train_zoo_spec.json"
REPLICATE_SPEC = "code/train_zoo_replicates.json"

# `corrupt` has 5x the systems, so it gets a smaller planning budget per system;
# its role is the comparison against the old design, not maximum precision.
ANALYSES = {
    "corrupt":  {"models": "zoo+released", "pipelines": "id,zero,neg,swap,atten",
                 "n_plan": 120, "n_closed": 20},
    "released": {"models": "zoo+released", "pipelines": "id"},
    "primary":  {"models": "zoo",          "pipelines": "id"},
    "memrep":   {"models": "memory",       "pipelines": "id"},
}


def spec_names(path: str):
    with open(path) as f:
        return [m["name"] for m in json.load(f)["models"]]


def existing(root: Path, names):
    """Keep the spec order; report anything the spec promised but never trained."""
    have, missing = [], []
    for n in names:
        d = root / n
        (have if (d / f"{n}.ckpt").exists() and (d / "meta.json").exists()
         else missing).append(n)
    return have, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained-root", default="/mnt/group/jxdong/wm_exp/ckpts/trained")
    ap.add_argument("--data", default="data/pusht_heldout.h5")
    ap.add_argument("--out", default="runs/final")
    ap.add_argument("--analyses", default="primary,released,corrupt,memrep")
    ap.add_argument("--spec", default=SPEC)
    ap.add_argument("--replicate-spec", default=REPLICATE_SPEC)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--strata", type=int, default=5)
    ap.add_argument("--n-plan", type=int, default=250)
    ap.add_argument("--n-closed", type=int, default=50)
    ap.add_argument("--mppi-iters", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.trained_root)
    zoo, zoo_missing = existing(root, spec_names(args.spec))
    reps, rep_missing = existing(root, spec_names(args.replicate_spec))
    memory = [n for n in zoo if n.startswith("mem")] + reps

    print(f"zoo ({len(zoo)}): {','.join(zoo)}")
    print(f"memory sweep incl. replicates ({len(memory)}): {','.join(memory)}")
    for tag, miss in (("zoo", zoo_missing), ("replicates", rep_missing)):
        if miss:
            print(f"MISSING {tag}: {','.join(miss)} -- these are pre-registered "
                  f"and must be reported; train them before the final run")
    if not zoo:
        raise SystemExit("no trained checkpoints found")

    pools = {"zoo": zoo, "zoo+released": zoo + RELEASED, "memory": memory}

    Path(args.out).mkdir(parents=True, exist_ok=True)
    logs = Path(args.out) / "logs"
    logs.mkdir(exist_ok=True)

    jobs = []
    for aname in args.analyses.split(","):
        cfg = ANALYSES[aname]
        models = pools[cfg["models"]]
        if not models:
            print(f"skip {aname} (no checkpoints in its pool)")
            continue
        for seed in args.seeds.split(","):
            out = Path(args.out) / f"{aname}_seed{seed}.json"
            if out.exists():
                print(f"skip {out.name} (exists)")
                continue
            jobs.append((aname, seed, [
                "python3", "code/k1k2_real_eval.py",
                "--data", args.data,
                "--models", ",".join(models),
                "--pipelines", cfg["pipelines"],
                "--strata", str(args.strata),
                "--n-plan", str(cfg.get("n_plan", args.n_plan)),
                "--n-closed", str(cfg.get("n_closed", args.n_closed)),
                "--mppi-iters", str(args.mppi_iters),
                "--seed", seed,
                "--out", str(out),
                "--dump-units",
            ]))

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    running, t0 = {}, time.time()
    print(f"{len(jobs)} jobs on {len(gpus)} gpus")
    while jobs or running:
        for g in gpus:
            if g in running or not jobs:
                continue
            aname, seed, cmd = jobs.pop(0)
            log = open(logs / f"{aname}_seed{seed}.log", "w")
            p = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=g),
            )
            running[g] = (p, f"{aname}/seed{seed}", time.time())
            print(f"  gpu{g} <- {aname} seed{seed}", flush=True)
        time.sleep(20)
        for g, (p, tag, t) in list(running.items()):
            if p.poll() is None:
                continue
            print(f"  gpu{g} done {tag} rc={p.returncode} "
                  f"{(time.time()-t)/60:.0f}min", flush=True)
            del running[g]

    print(f"ALL EVALS DONE in {(time.time()-t0)/60:.0f} min")

    # quality spectrum of the zoo, for the paper's model table
    spec = {}
    for n in zoo + reps:
        with open(Path(args.trained_root) / n / "meta.json") as f:
            m = json.load(f)
        spec[n] = {
            k: m.get(k) for k in
            ("params_M", "best_val_pred", "latent_rms_final", "nonfinite_steps")
        }
        spec[n]["ctx"] = m["config"]["ctx_len"]
    with open(Path(args.out) / "zoo_quality.json", "w") as f:
        json.dump(spec, f, indent=2)
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
