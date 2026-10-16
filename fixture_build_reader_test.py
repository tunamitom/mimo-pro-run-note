#!/usr/bin/env python3
"""Build a tiny CSF fixture exercising the RUNTIME reader's naming contract.

Extracts 2 experts of layer 35 (gate/up/down packed weights + weight_scale
planes) from the source checkpoint and writes a mini-container with v2
tensor names: `<name>.mxfp4_csf_fixed` / `<name>.mxfp4_csf_exceptions`
(the exact keys vllm csf_utils.CsfTensorReader(root, names, "mxfp4")
looks up via safetensors get_tensor).

Host-side: stdlib + numpy only. Raw safetensors written by hand
(8-byte LE header length + JSON header + payload), same as the builder.
"""
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/tom/csf-toolkit")
from mxfp4_csf import encode_plane  # noqa: E402

import os

SRC = Path(os.environ.get("FIXTURE_SRC", "/opt/models/MiMo-V2.6-Pro-MOPD"))
DEST = Path(os.environ.get("FIXTURE_DEST", "/home/tom/.hermes/cache/scratch/csf-fixture"))
LAYER, EXPERTS = 35, [0, 1]
SUFFIX = "mxfp4"

SCALE_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight_scale$"
)

DEST.mkdir(parents=True, exist_ok=True)
(DEST / "tensors").mkdir(exist_ok=True)

index = json.loads((SRC / "model.safetensors.index.json").read_text())
wm = index["weight_map"]

entries = {}   # tensor name -> (dtype, shape, bytes)
order = []


def add(name, dtype, shape, buf):
    assert name not in entries
    entries[name] = (dtype, shape, buf)
    order.append(name)


def read_tensor(fname, tname):
    path = SRC / fname
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        meta = header[tname]
        a, b = meta["data_offsets"]
        f.seek(8 + hlen + a)
        buf = f.read(b - a)
        assert len(buf) == b - a, f"short read {tname}"
        return meta["dtype"], list(meta["shape"]), buf


source_names = {}
plan = []
for e in EXPERTS:
    for proj in ("gate", "up", "down"):
        wname = f"model.layers.{LAYER}.mlp.experts.{e}.{proj}_proj.weight"
        sname = f"model.layers.{LAYER}.mlp.experts.{e}.{proj}_proj.weight_scale"
        wf, sf = wm[wname], wm[sname]
        wdtype, wshape, wbuf = read_tensor(wf, wname)
        sdtype, sshape, sbuf = read_tensor(sf, sname)
        assert sdtype == "U8" and len(sbuf) == int(np.prod(sshape)), sname
        plane = np.frombuffer(sbuf, dtype=np.uint8).reshape(sshape)
        fixed, exc = encode_plane(plane)
        add(wname, wdtype, wshape, wbuf)
        add(f"{sname}.{SUFFIX}_csf_fixed", "U8", [int(fixed.nbytes)], fixed.tobytes())
        add(f"{sname}.{SUFFIX}_csf_exceptions", "U32", [int(exc.nbytes) // 4], exc.tobytes())
        source_names[wname] = "fixture.safetensors"
        source_names[sname] = "fixture.safetensors"
        plan.append((wname, sname, wshape, sshape, len(sbuf)))

header = {}
offset = 0
for name in order:
    dtype, shape, buf = entries[name]
    header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(buf)]}
    offset += len(buf)
hj = json.dumps(header, separators=(",", ":")).encode()
with open(DEST / "tensors" / "fixture.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(hj)))
    f.write(hj)
    for name in order:
        f.write(entries[name][2])

(DEST / "fixture-source-names.json").write_text(json.dumps(source_names, indent=1))
(DEST / "fixture-plan.json").write_text(json.dumps(
    [{"weight": w, "scale": s, "weight_shape": ws, "scale_shape": ss, "scale_bytes": sb}
     for w, s, ws, ss, sb in plan], indent=1))

total = sum(len(entries[n][2]) for n in order)
print(f"fixture written: {len(order)} tensors, {total/2**20:.1f} MiB payload")
print(f"experts: {EXPERTS}, layer: {LAYER}")
for w, s, ws, ss, sb in plan:
    print(f"  {w.split('mlp.')[-1]}: weight {ws}, scale {ss} ({sb} B)")
