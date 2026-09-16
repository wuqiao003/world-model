#!/usr/bin/env python3
"""Fetch LeWM PushT bundle via hf-mirror (company HF often blocked)."""
from __future__ import annotations

import os
import traceback
from pathlib import Path

OUT = Path(os.environ.get("LEWM_DIR", "/mnt/group/jxdong/wm_exp/ckpts/lewm_pusht"))
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

REPO = os.environ.get("LEWM_REPO", "YuhaiW/lewm-pusht-fr3-v2")


def main():
    print("ENDPOINT", os.environ.get("HF_ENDPOINT"))
    print("OUT", OUT)
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except Exception as e:
        print("NO_HF_HUB", e)
        return 2

    # Try snapshot first; fall back to individual files
    try:
        p = snapshot_download(
            REPO,
            local_dir=str(OUT),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print("SNAPSHOT_OK", p)
    except Exception:
        traceback.print_exc()
        print("SNAPSHOT_FAIL; trying file-by-file")
        files = [
            "lewm_pusht_fr3_v2.ckpt",
            "action_scaler.json",
            "jepa.py",
            "module.py",
            "pusht_lewm_inference.py",
            "requirements.txt",
            "README.md",
        ]
        for f in files:
            try:
                hf_hub_download(REPO, f, local_dir=str(OUT), local_dir_use_symlinks=False)
                print("OK", f)
            except Exception as e:
                print("FAIL", f, e)

    for p in sorted(OUT.rglob("*")):
        if p.is_file() and ".huggingface" not in str(p) and ".hf_home" not in str(p):
            print("FILE", p.name, p.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
