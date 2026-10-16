#!/usr/bin/env python3
"""Offline end-to-end validation of the CSF overlay (no GPU, no service).

Mounts the 2 overlay files over the image's paths and exercises the exact
chain the serving stack will use:
 1. overlay modules import cleanly against the real image package
 2. checkpoint_contract accepts the MiMo container (family mimo_v26_pro_mxfp4)
 3. Mxfp4CsfConfig.from_config (dispatcher) -> MimoMxfp4CsfConfig (Fp8Config)
 4. override_quantization_method accepts the real source quant dict
 5. read_mxfp4_csf_layer (TP8 rank 3, layer 35) returns Mxfp4CsfWeights whose
    CPU-decoded expanded grids == source rank-local slices, byte-exact
 6. geometry gate: a draft-sized config passes through as None (not CSF)
Fail-closed: any mismatch raises.
"""
import json
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, "/toolkit")
import mxfp4_csf as codec  # noqa: E402

print("== 1. overlay imports ==")
from vllm.model_executor.model_loader.mxfp4_csf_loader import (  # noqa: E402
    FAMILIES,
    Mxfp4CsfModelLoader,
    checkpoint_contract,
    expert_tensor_names,
    read_mxfp4_csf_layer,
)
print("loader OK; families:", sorted(FAMILIES))
assert "mimo_v26_pro_mxfp4" in FAMILIES

from vllm.model_executor.layers.quantization.mxfp4_csf import (  # noqa: E402
    MimoMxfp4CsfConfig,
    Mxfp4CsfConfig,
)
print("quant OK:", MimoMxfp4CsfConfig.__mro__[1].__name__, "base")

print("== 2. container contract ==")
root = "/container"
contract = checkpoint_contract(root)
assert contract["family"] == "mimo_v26_pro_mxfp4", contract["family"]
print("family:", contract["family"], "| source tensors:", len(contract["source_names"]))

names = expert_tensor_names("mimo_v26_pro_mxfp4", 35, 0)
print("== expert names L35 e0:", names[0][0], "...")
assert names[0][0] == "model.layers.35.mlp.experts.0.gate_proj.weight"
assert names[2][0] == "model.layers.35.mlp.experts.0.down_proj.weight"

print("== 3. dispatcher -> MiMo config ==")
cfg = json.load(open("/model/config.json"))
q = cfg["quantization_config"]
qc = Mxfp4CsfConfig.from_config(q)
assert isinstance(qc, MimoMxfp4CsfConfig), type(qc)
assert qc.ignored_layers == q["ignored_layers"], "ignored_layers must carry over"
assert qc.is_checkpoint_fp8_serialized is True
assert qc.weight_block_size == [128, 128]
assert qc.activation_scheme == "dynamic"
print("MimoMxfp4CsfConfig:", type(qc).__name__,
      "| ignored:", len(qc.ignored_layers), "layers",
      "| root:", qc.checkpoint_root)

print("== 4. override acceptance (source quant dict) ==")
class HF:  # minimal stand-in with the attrs the protocol reads
    model_type = "mimo_v2"
got = MimoMxfp4CsfConfig.override_quantization_method(q, None, HF())
assert got == "mxfp4_csf", got
got2 = Mxfp4CsfConfig.override_quantization_method(q, None, HF())
print("override:", got, "| dispatcher-level:", got2)

print("== 5. read_mxfp4_csf_layer (TP8 rank 3, layer 35, CPU) ==")
LAYER, TP, RANK = 35, 8, 3
LOCAL = 2048 // TP
first, last = RANK * LOCAL, (RANK + 1) * LOCAL
s13 = torch.empty((384, 2 * LOCAL, 6144 // 32), dtype=torch.uint8)
s2 = torch.empty((384, 6144, LOCAL // 32), dtype=torch.uint8)
weights = read_mxfp4_csf_layer(
    root, LAYER,
    num_experts=384, hidden_size=6144, intermediate_size=2048,
    tp_rank=RANK, tp_size=TP, device="cpu",
    w13_scale_scratch=s13, w2_scale_scratch=s2,
)
# decode one expert's planes into the shared scratch and compare to source
impl = weights  # Mxfp4CsfWeights dataclass
e = 5
import dataclasses
flds = {f.name: getattr(impl, f.name) for f in dataclasses.fields(impl)}
print("Mxfp4CsfWeights fields:", {k: type(v).__name__ for k, v in flds.items()})
# CPU decode of the rank-local planes for expert e (w13 fused gate+up)
f13 = impl.w13_scales.fixed[e]
x13 = impl.w13_scales.exceptions[e]
dec13 = codec.decode_plane(f13, x13, 2 * LOCAL, 6144 // 32)
f2 = impl.w2_scales.fixed[e]
x2 = impl.w2_scales.exceptions[e]
dec2 = codec.decode_plane(f2, x2, 6144, LOCAL // 32)
# source rank-local slices, read independently from /model
def src_slice(sname, r0, r1, c0, c1):
    idx = json.load(open("/model/model.safetensors.index.json"))["weight_map"]
    with open("/model/" + idx[sname], "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        hdr = json.loads(fh.read(hlen))
        a, b = hdr[sname]["data_offsets"]
        fh.seek(8 + hlen + a)
        plane = np.frombuffer(fh.read(b - a), dtype=np.uint8).reshape(hdr[sname]["shape"])
    return plane[r0:r1, c0:c1] if plane.ndim == 2 else plane
g = src_slice(f"model.layers.{LAYER}.mlp.experts.{e}.gate_proj.weight_scale", first, last, 0, 6144 // 32)
u = src_slice(f"model.layers.{LAYER}.mlp.experts.{e}.up_proj.weight_scale", first, last, 0, 6144 // 32)
d = src_slice(f"model.layers.{LAYER}.mlp.experts.{e}.down_proj.weight_scale", 0, 6144, first // 32, last // 32)
assert np.array_equal(dec13[:LOCAL], g), "gate plane mismatch"
assert np.array_equal(dec13[LOCAL:], u), "up plane mismatch"
assert np.array_equal(dec2, d), "down plane mismatch"
print("expert 5 planes byte-exact vs source rank-3 slices (gate/up/down)")

print("== 6. geometry gate pass-through ==")
class DraftCfg:  # DFlash draft: no routed experts
    model_type = "mimo_v2"
    n_routed_experts = None
    hidden_size = None
    moe_intermediate_size = None
res = Mxfp4CsfModelLoader._resolve_csf_root(
    Mxfp4CsfModelLoader, {"quant_method": "fp8", "store_dtype": "mxfp4"}, DraftCfg())
assert res is None, "draft-shaped config must pass through"
class MiMoCfg:
    model_type = "mimo_v2"
    n_routed_experts = 384
    hidden_size = 6144
    moe_intermediate_size = 2048
res2 = Mxfp4CsfModelLoader._resolve_csf_root(
    Mxfp4CsfModelLoader, {"quant_method": "fp8", "store_dtype": "mxfp4"}, MiMoCfg())
assert res2 is not None, "MiMo-shaped config must resolve"
print("gate: draft -> None (pass-through); MiMo -> resolved")

print("\nOVERLAY VALIDATION PASSED — all 6 checks")
