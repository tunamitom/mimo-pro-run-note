#!/usr/bin/env python3
"""Container-level reader test: vllm's OWN contract+reader against the REAL
migrated 512 GiB container (not a fixture).

Runs in-image with the container at /container (ro) and the source at /model.
1. read_csf_contract(root, schema=..., codec=..., families={...}) — validates
   manifest/build-contract grammar; the MiMo family is injected locally
   because upstream FAMILIES does not list it yet (that is the known runtime
   gap). Everything else must pass exactly as production would.
2. CsfTensorReader(root, contract["source_names"], "mxfp4") — read real
   matrices the way mxfp4_csf_loader.read_mxfp4_csf_layer does.
3. Decode read planes with the toolkit codec and compare byte-exactly to the
   SOURCE checkpoint tensors (read independently from /model).
4. Also verify a TP-rank slice via vllm's _slice_scale_plane matches the
   source's rank-local slice bytes (the production TP8 read path).
Fail-closed: any mismatch raises.
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

from vllm.model_executor.model_loader.csf_utils import (  # noqa: E402
    CsfTensorReader,
    read_csf_contract,
    tp_extent,
)
from vllm.model_executor.model_loader.mxfp4_csf_loader import _slice_scale_plane  # noqa: E402

ROOT = Path("/container")
SRC = Path("/model")
SCHEMA = "lil-mxfp4-csf-checkpoint/1"
CODEC = "row-base-offset1-u24-exceptions/1"
FAMILIES = {"mimo_v26_pro_mxfp4": (384, 6144, 2048, range(1, 70))}

contract = read_csf_contract(str(ROOT), schema=SCHEMA, codec=CODEC, families=FAMILIES)
print("contract OK: family =", contract["family"],
      "; source tensors =", len(contract["source_names"]))
assert contract["family"] == "mimo_v26_pro_mxfp4"

index = json.loads((SRC / "model.safetensors.index.json").read_text())
wm = index["weight_map"]


def source_scale(sname):
    fname = wm[sname]
    with open(SRC / fname, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        meta = header[sname]
        a, b = meta["data_offsets"]
        f.seek(8 + hlen + a)
        buf = f.read(b - a)
    return buf, list(meta["shape"])


INTER, HIDDEN = 2048, 6144
checked = 0
with CsfTensorReader(ROOT, contract["source_names"], "mxfp4") as reader:
    for layer in (1, 35, 69):
        for e in (0, 191, 383):
            for proj, rows, cols in (
                ("gate_proj", INTER, HIDDEN // 32),
                ("up_proj", INTER, HIDDEN // 32),
                ("down_proj", HIDDEN, INTER // 32),
            ):
                p = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
                wname, sname = f"{p}.weight", f"{p}.weight_scale"
                m = reader.matrix(wname, sname)
                src_buf, src_shape = source_scale(sname)
                assert m.fixed.dtype == torch.uint8 and m.exceptions.dtype == torch.uint32
                # full-plane decode == source bytes
                plane = codec.decode_plane(m.fixed, m.exceptions, rows, cols)
                assert plane.tobytes() == src_buf, f"{sname} full decode mismatch"
                # production TP8 rank-3 slice == source rank-local slice bytes
                first, last = tp_extent(INTER, 3, 8, 32)
                if proj == "down_proj":
                    sf, se = _slice_scale_plane(
                        m.fixed, m.exceptions, rows, cols, (0, rows), (first // 32, last // 32))
                    sl_buf = np.frombuffer(src_buf, dtype=np.uint8).reshape(rows, cols)[
                        :, first // 32:last // 32].tobytes()
                else:
                    sf, se = _slice_scale_plane(
                        m.fixed, m.exceptions, rows, cols, (first, last), (0, cols))
                    sl_buf = np.frombuffer(src_buf, dtype=np.uint8).reshape(rows, cols)[
                        first:last, :].tobytes()
                sl = codec.decode_plane(sf, se,
                                        last - first if proj != "down_proj" else rows,
                                        cols if proj != "down_proj" else (last - first) // 32)
                assert sl.tobytes() == sl_buf, f"{sname} TP-rank-3 slice mismatch"
                checked += 1
            print(f"OK L{layer} e{e}: reader + full decode + TP8-rank3 slice byte-exact",
                  flush=True)

print(f"CONTAINER READER TEST PASSED: {checked} projections across layers 1/35/69, "
      "experts 0/191/383 — vllm contract+reader+TP-slicing all byte-exact on the "
      "real migrated container")
