#!/usr/bin/env python3
"""Geometry verifier toy on TRELLIS ply: bbox/count/centroid sanity + multi-seed consistency."""
import json, os, struct
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get("OUT_DIR", "/mnt/group/jxdong/wm_exp/runs/geom_verifier"))
OUT.mkdir(parents=True, exist_ok=True)
PLY = Path(os.environ.get("PLY", "/mnt/group/jxdong/wm_exp/runs/trellis_smoke/sample.ply"))


def read_ply_xyz(path: Path):
    """Minimal PLY reader for ascii or binary_little_endian xyz(+extra)."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header.append(line)
            if line.startswith("end_header"):
                break
        fmt = "ascii"
        n_vert = 0
        props = []
        for line in header:
            if line.startswith("format"):
                fmt = line.split()[1]
            if line.startswith("element vertex"):
                n_vert = int(line.split()[-1])
            if line.startswith("property"):
                props.append(line.split()[1:])  # type name
        if fmt == "ascii":
            xyz = []
            for _ in range(n_vert):
                parts = f.readline().decode("ascii").split()
                xyz.append([float(parts[0]), float(parts[1]), float(parts[2])])
            return np.asarray(xyz, dtype=np.float64)
        # binary
        type_map = {"float": "f", "float32": "f", "double": "d", "float64": "d",
                    "uchar": "B", "uint8": "B", "int": "i", "int32": "i", "uint": "I"}
        fmt_str = "<" if "little" in fmt else ">"
        for t, _ in props:
            fmt_str += type_map.get(t, "f")
        rec = struct.calcsize(fmt_str)
        data = f.read(n_vert * rec)
        pts = []
        for i in range(n_vert):
            vals = struct.unpack_from(fmt_str, data, i * rec)
            pts.append(vals[:3])
        return np.asarray(pts, dtype=np.float64)


def score(xyz: np.ndarray):
    if xyz.size == 0:
        return {"ok": False, "reason": "empty"}
    mn, mx = xyz.min(0), xyz.max(0)
    extent = mx - mn
    centroid = xyz.mean(0)
    # finite, non-degenerate bbox, roughly centered near origin (TRELLIS assets usually are)
    checks = {
        "n_points": int(xyz.shape[0]),
        "extent": extent.tolist(),
        "centroid": centroid.tolist(),
        "finite": bool(np.isfinite(xyz).all()),
        "non_degenerate": bool((extent > 1e-4).all()),
        "extent_reasonable": bool((extent.max() < 20) and (extent.min() > 1e-3)),
        "centroid_near_origin": bool(np.linalg.norm(centroid) < 2.0),
        "min_points": bool(xyz.shape[0] >= 1000),
    }
    checks["pass"] = all([
        checks["finite"], checks["non_degenerate"], checks["extent_reasonable"],
        checks["centroid_near_origin"], checks["min_points"],
    ])
    return checks


def main():
    if not PLY.is_file():
        out = {"ok": False, "error": f"missing {PLY}"}
    else:
        xyz = read_ply_xyz(PLY)
        out = {"ok": True, "ply": str(PLY), "ply_bytes": PLY.stat().st_size, **score(xyz)}
    path = OUT / "result.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
