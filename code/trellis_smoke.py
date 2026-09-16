#!/usr/bin/env python3
"""TRELLIS smoke via lzdql m3d-trellis env (has kaolin)."""
import os, sys, types, json, time
from pathlib import Path

os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("HF_HOME", "/mnt/group/mcp/hf_cache")
os.environ.setdefault("TORCH_HOME", "/mnt/group/mcp/hf_cache/torch")

# open3d stub (real import hangs headless)
o3d = types.ModuleType("open3d"); o3d.__version__ = "stub"
geom = types.ModuleType("open3d.geometry")
class TriangleMesh: pass
geom.TriangleMesh = TriangleMesh
o3d.geometry = geom
sys.modules["open3d"] = o3d
sys.modules["open3d.geometry"] = geom
rembg = types.ModuleType("rembg"); rembg.remove = lambda img, *a, **k: img
sys.modules["rembg"] = rembg

# Prefer lzdql TRELLIS tree (paired with m3d-trellis)
for root in ("/mnt/group/lzdql/TRELLIS", "/mnt/group/hjm/TRELLIS"):
    if Path(root, "trellis").is_dir():
        ROOT = root
        break
else:
    raise SystemExit("no TRELLIS code")

sys.path.insert(0, ROOT)
os.chdir(ROOT)
OUT = Path("/mnt/group/jxdong/wm_exp/runs/trellis_smoke")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

print("ROOT", ROOT, flush=True)
import trellis  # noqa
from trellis.pipelines import TrellisImageTo3DPipeline
from PIL import Image

path = "/mnt/group/lzdql/weights/TRELLIS-image-large"
print("LOADING", path, flush=True)
pipe = TrellisImageTo3DPipeline.from_pretrained(path)
pipe.cuda()
img_candidates = [
    f"{ROOT}/assets/example_image/T.png",
    "/mnt/group/hjm/TRELLIS/assets/example_image/T.png",
    "/mnt/group/mcp/out/prim12/boy.png",
]
img_path = next(p for p in img_candidates if Path(p).is_file())
img = Image.open(img_path).convert("RGBA")
print("RUN", img_path, img.size, flush=True)
outputs = pipe.run(img, seed=1)
print("KEYS", list(outputs.keys()), flush=True)
ply = OUT / "sample.ply"
outputs["gaussian"][0].save_ply(str(ply))
meta = {
    "ok": True,
    "root": ROOT,
    "path": path,
    "image": img_path,
    "ply": str(ply),
    "ply_bytes": ply.stat().st_size,
    "seconds": time.time() - t0,
}
(OUT / "result.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2), flush=True)
print("SUCCESS", flush=True)
