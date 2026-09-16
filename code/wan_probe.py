#!/usr/bin/env python3
"""Wan2.2-TI2V-5B dependency/path probe (no full generation unless cheap)."""
import json, os, sys
from pathlib import Path

OUT = Path("/mnt/group/jxdong/wm_exp/runs/wan_probe")
OUT.mkdir(parents=True, exist_ok=True)
root = Path("/mnt/public/Wan2.2_models/Wan2.2-TI2V-5B")
info = {"root": str(root), "exists": root.is_dir()}
if root.is_dir():
    info["top"] = sorted([p.name for p in root.iterdir()])[:40]
    # common config files
    for name in ["config.json", "model_index.json", "configuration.json"]:
        p = root / name
        if p.is_file():
            info["config_file"] = name
            try:
                info["config_head"] = p.read_text()[:800]
            except Exception as e:
                info["config_err"] = str(e)
            break
# python deps
deps = {}
for m in ["torch", "diffusers", "transformers", "accelerate", "einops", "imageio"]:
    try:
        mod = __import__(m)
        deps[m] = getattr(mod, "__version__", "ok")
    except Exception as e:
        deps[m] = f"FAIL:{e}"
info["deps"] = deps
info["cuda"] = False
try:
    import torch
    info["cuda"] = bool(torch.cuda.is_available())
    info["torch"] = torch.__version__
except Exception as e:
    info["torch_err"] = str(e)
# look for example scripts nearby
cands = []
for base in [Path("/mnt/public"), Path("/mnt/group")]:
    for pat in ["**/Wan2.2*/generate*.py", "**/Wan*/infer*.py"]:
        pass
# shallow
for p in [
    "/mnt/public/Wan2.2_models",
    "/mnt/group/lzdql",
    "/mnt/group/hjm",
]:
    bp = Path(p)
    if bp.is_dir():
        for child in list(bp.glob("*Wan*"))[:10]:
            cands.append(str(child))
info["nearby"] = cands[:20]
path = OUT / "result.json"
path.write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2)[:3000])
print("WROTE", path)
