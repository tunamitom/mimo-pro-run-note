"""Full MXFP4 scale compression scan over MiMo-V2.6-Pro-MOPD (CPU, bounded).

For every routed-expert weight_scale plane: encode with the spike codec,
decode back, verify byte-equality, record sizes and exception stats.
Outputs scan_mimo_mxfp4.json + a digest to stdout.
"""

import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from safetensors import safe_open

from mxfp4_csf import decode_plane, encode_plane

ROOT = Path("/model")  # mounted read-only
INDEX = json.loads((ROOT / "model.safetensors.index.json").read_text())
SCALE_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight_scale$")


def scan_shard(filename):
    names = sorted(SHARDS[filename])
    out = []
    with safe_open(ROOT / filename, framework="np") as f:
        for name in names:
            plane = f.get_tensor(name)
            assert plane.dtype == np.uint8, (name, plane.dtype)
            rows, cols = plane.shape
            fixed, exc = encode_plane(plane)
            back = decode_plane(fixed, exc, rows, cols)
            if not (back == plane).all():
                raise RuntimeError(f"round-trip failed: {name}")
            m = SCALE_RE.match(name)
            out.append(
                {
                    "layer": int(m.group(1)),
                    "proj": m.group(3),
                    "rows": rows,
                    "cols": cols,
                    "orig": int(plane.nbytes),
                    "fixed": int(fixed.nbytes),
                    "exc_bytes": int(exc.nbytes),
                    "exc_count": int(exc.shape[0]),
                }
            )
    return out


def main():
    global SHARDS
    SHARDS = defaultdict(list)
    for name in INDEX["weight_map"]:
        m = SCALE_RE.match(name)
        if m:
            SHARDS[INDEX["weight_map"][name]].append(name)
    total_planes = sum(len(v) for v in SHARDS.values())
    print(f"shards with expert scales: {len(SHARDS)}, planes: {total_planes}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for i, part in enumerate(ex.map(scan_shard, sorted(SHARDS)), 1):
            results.extend(part)
            if i % 10 == 0:
                print(f"  shards done: {i}/{len(SHARDS)}", flush=True)

    orig = sum(r["orig"] for r in results)
    fixed = sum(r["fixed"] for r in results)
    exc_b = sum(r["exc_bytes"] for r in results)
    encoded = fixed + exc_b
    exc_counts = np.array([r["exc_count"] for r in results])
    origs = np.array([r["orig"] for r in results])
    ret = (np.array([r["fixed"] + r["exc_bytes"] for r in results]) / origs)
    by_proj = {}
    for proj in ("gate", "up", "down"):
        sel = [r for r in results if r["proj"] == proj]
        o = sum(r["orig"] for r in sel)
        e = sum(r["fixed"] + r["exc_bytes"] for r in sel)
        by_proj[proj] = {"planes": len(sel), "retained_pct": round(100 * e / o, 2)}
    summary = {
        "planes": len(results),
        "original_bytes": orig,
        "fixed_bytes": fixed,
        "exception_bytes": exc_b,
        "encoded_bytes": encoded,
        "retained_pct": round(100 * encoded / orig, 2),
        "saved_GiB": round((orig - encoded) / 2**30, 2),
        "per_plane_retained": {
            "min": round(100 * ret.min(), 2),
            "p50": round(100 * float(np.median(ret)), 2),
            "p95": round(100 * float(np.percentile(ret, 95)), 2),
            "max": round(100 * ret.max(), 2),
        },
        "exception_rate_pct": {
            "mean": round(100 * float(exc_counts.sum()) / int((origs / 1).sum() and sum(r["rows"] * r["cols"] for r in results)), 3),
            "planes_over_21p5pct": int((exc_counts / np.array([r["rows"] * r["cols"] for r in results]) > 0.215).sum()),
        },
        "by_projection": by_proj,
    }
    Path("/toolkit-out/scan_mimo_mxfp4.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()