#!/usr/bin/env python3
"""In-image fixture runner: the REAL runtime reader against v2 tensor names.

Runs inside ghcr.io/local-inference-lab/vllm:karmic-kraken-beta with the
fixture mounted at /fixture and the source model at /model. Uses vllm's
own CsfTensorReader (csf_utils) exactly as mxfp4_csf_loader does:

    with CsfTensorReader(root, contract["source_names"], "mxfp4") as reader:
        reader.matrix(weight, scale)

Asserts (fail-closed, raise on any mismatch):
 1. reader.matrix() finds `.mxfp4_csf_fixed` / `.mxfp4_csf_exceptions`
    under the v2 names (KeyError = FAIL).
 2. Weight views expose the right dtype/shape.
 3. Decoding the reader-returned fixed/exceptions (toolkit codec, the same
    codec the runtime's _slice_scale_plane consumes) reproduces the SOURCE
    scale planes byte-exactly, read independently from /model.
 4. vllm's own `_slice_scale_plane` accepts the reader-returned tensors and
    a full-width TP1 slice reproduces the same bytes (upstream-oracle path).
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/toolkit")
import mxfp4_csf as codec  # noqa: E402

from vllm.model_executor.model_loader.csf_utils import CsfTensorReader  # noqa: E402
from vllm.model_executor.model_loader.mxfp4_csf_loader import (  # noqa: E402
    _slice_scale_plane,
)

FIXTURE = Path("/fixture")
SRC = Path("/model")
reader_names = json.loads((FIXTURE / "fixture-source-names.json").read_text())
plan = json.loads((FIXTURE / "fixture-plan.json").read_text())


def source_scale_bytes(sname):
    index = json.loads((SRC / "model.safetensors.index.json").read_text())
    fname = index["weight_map"][sname]
    with open(SRC / fname, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        meta = header[sname]
        a, b = meta["data_offsets"]
        f.seek(8 + hlen + a)
        buf = f.read(b - a)
    return buf, list(meta["shape"])


checked = 0
with CsfTensorReader(FIXTURE, reader_names, "mxfp4") as reader:
    for item in plan:
        wname, sname = item["weight"], item["scale"]
        m = reader.matrix(wname, sname)

        # 2. weight view contract
        expected_w = item["weight_shape"]
        assert list(m.weight.get_shape()) == expected_w, (wname, m.weight.get_shape())
        assert m.weight.get_dtype() in ("I8", "U8"), (wname, m.weight.get_dtype())

        # 3. decode reader-returned scale components vs independent source read
        src_buf, src_shape = source_scale_bytes(sname)
        rows, cols = src_shape
        plane = codec.decode_plane(m.fixed, m.exceptions, rows, cols)
        assert plane.tobytes() == src_buf, f"{sname}: decode != source bytes"
        assert plane.shape == tuple(src_shape)

        # 4. upstream oracle: _slice_scale_plane full-width TP1 slice
        # geometry: gate/up rows=intermediate cols=hidden/32; down rows=hidden cols=intermediate/32
        if "gate" in wname or "up" in wname:
            r, c = rows, cols
            rsl, csl = (0, rows), (0, cols)
        else:
            r, c = rows, cols
            rsl, csl = (0, rows), (0, cols)
        sliced_fixed, sliced_exc = _slice_scale_plane(
            m.fixed, m.exceptions, r, c, rsl, csl
        )
        plane2 = codec.decode_plane(sliced_fixed, sliced_exc, r, c)
        assert plane2.tobytes() == src_buf, f"{sname}: oracle-sliced decode != source"

        checked += 1
        print(f"OK {wname.split('mlp.')[-1]}: reader+decode+oracle byte-exact "
              f"({rows}x{cols})", flush=True)

print(f"FIXTURE PASSED: {checked}/{len(plan)} projections — "
      "CsfTensorReader accepts v2 names; decode + upstream _slice_scale_plane "
      "reproduce source bytes exactly")
