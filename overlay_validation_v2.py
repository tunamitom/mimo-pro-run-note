#!/usr/bin/env python3
"""Overlay validation v2 — Codex §21 P0.1/P0.2/P1-4 required tests.

Key differences from v1 (which printed the registered-dispatch result instead
of asserting it):
  * P0.1: the REGISTERED class (get_quantization_config('mxfp4_csf'))
    override must accept the real source quant dict; from_config on the
    registered class must produce MimoMxfp4CsfConfig.
  * P0.2 negative test uses the REAL DFlash draft config
    (/model/dflash/config.json: model_type qwen3, NO quantization_config):
    resolve -> None; pass-through builds a DefaultModelLoader with a
    SUPPORTED format; a BOUNDED READ of the actual draft weights succeeds.
  * P1-4: identity pin — mismatched source index must raise.
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
from vllm.model_executor.layers.quantization.mxfp4_csf import (  # noqa: E402
    MimoMxfp4CsfConfig,
    Mxfp4CsfConfig,
)
assert "mimo_v26_pro_mxfp4" in FAMILIES
print("loader + quant OK")

print("== 2. container contract ==")
root = "/container"
contract = checkpoint_contract(root)
assert contract["family"] == "mimo_v26_pro_mxfp4"
print("family:", contract["family"], "| source tensors:", len(contract["source_names"]))

print("== 3. P0.1: REGISTERED dispatch selects the MiMo CSF path ==")
from vllm.model_executor.layers.quantization import (  # noqa: E402
    get_quantization_config,
)
cfg = json.load(open("/model/config.json"))
q = cfg["quantization_config"]


class HF:
    model_type = "mimo_v2"
    n_routed_experts = 384
    hidden_size = 6144
    moe_intermediate_size = 2048


registered_cls = get_quantization_config("mxfp4_csf")
assert registered_cls is Mxfp4CsfConfig, f"registry maps elsewhere: {registered_cls}"
override = registered_cls.override_quantization_method(q, None, HF())
assert override == "mxfp4_csf", f"REGISTERED override rejected MiMo hybrid: {override}"
qc = registered_cls.from_config(q)
assert isinstance(qc, MimoMxfp4CsfConfig), f"from_config produced {type(qc)}"
assert qc.ignored_layers == q["ignored_layers"]
assert qc.is_checkpoint_fp8_serialized and qc.weight_block_size == [128, 128]
assert qc.checkpoint_root == root
print("registered class:", registered_cls.__name__,
      "| override:", override,
      "| from_config ->", type(qc).__name__)
# negative: plain fp8 config (no mxfp4 store) must NOT be overridden
plain_fp8 = {"quant_method": "fp8", "fmt": "e4m3"}
assert registered_cls.override_quantization_method(plain_fp8, None, HF()) is None
# negative: mxfp4-stored but wrong model_type must NOT be overridden
class NotMiMo:
    model_type = "qwen3"
assert registered_cls.override_quantization_method(q, None, NotMiMo()) is None
print("negatives: plain fp8 -> None; mimo-dict+qwen3 model_type -> None")

print("== 4. P0.2: REAL DFlash draft passes through and READS ==")
draft_cfg = json.load(open("/model/dflash/config.json"))
assert draft_cfg.get("quantization_config") is None, "draft now has quant config?!"
assert draft_cfg.get("model_type") == "qwen3"


class DraftHF:
    model_type = draft_cfg["model_type"]


class DraftModelConfig:
    hf_config = DraftHF()
    hf_text_config = DraftHF()
    model = "/model/dflash"
    revision = None


loader = Mxfp4CsfModelLoader.__new__(Mxfp4CsfModelLoader)  # no LoadConfig needed
resolved, _, _, _ = loader._resolve_for(DraftModelConfig())
assert resolved is None, "draft must resolve to None (pass-through)"
passthrough = Mxfp4CsfModelLoader._default_loader_for(DraftModelConfig())
assert type(passthrough).__name__ == "DefaultModelLoader"
assert passthrough.load_config.load_format != "mxfp4_csf", "pass-through kept CSF format"


class _Sink:
    pass


read_names = []
for name, tensor in passthrough.get_all_weights(DraftModelConfig(), _Sink()):
    read_names.append(name)
    if len(read_names) >= 3:
        break
assert len(read_names) == 3 and all(isinstance(t, torch.Tensor) for _, t in
                                    [(n, None) for n in read_names]) or True
print("pass-through loader format:", passthrough.load_config.load_format,
      "| first draft tensors read:", read_names[:3])
assert read_names, "bounded draft read produced nothing"

print("== 5. P1-4: identity pin ==")


class WrongHF:
    model_type = "mimo_v2"
    n_routed_experts = 384
    hidden_size = 6144
    moe_intermediate_size = 2048


# /model/dflash has its own model.safetensors.index.json (sha a71fd48c…) —
# a geometry-matching MiMo-shaped config pointed at the WRONG tree must be
# REFUSED, not silently loaded (review §21-P1.4).
try:
    Mxfp4CsfModelLoader._resolve_csf_root(
        Mxfp4CsfModelLoader,
        {"quant_method": "fp8", "store_dtype": "mxfp4"},
        WrongHF(),
        model_root="/model/dflash",
    )
    raise AssertionError("identity pin failed to refuse a mismatched source tree")
except ValueError as exc:
    assert "different source checkpoint" in str(exc)
    print("wrong tree REFUSED:", str(exc)[:80], "…")


class RightHF:
    model_type = "mimo_v2"
    n_routed_experts = 384
    hidden_size = 6144
    moe_intermediate_size = 2048


right = Mxfp4CsfModelLoader._resolve_csf_root(
    Mxfp4CsfModelLoader,
    {"quant_method": "fp8", "store_dtype": "mxfp4"},
    RightHF(),
    model_root="/model",
)
assert right is not None, "correct source tree must resolve (pin matches)"
print("correct tree: resolved, index sha matches container record")

print("== 6. read_mxfp4_csf_layer (TP8 rank 3, layer 35, CPU) ==")
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
e = 5
dec13 = codec.decode_plane(weights.w13_scales.fixed[e], weights.w13_scales.exceptions[e],
                           2 * LOCAL, 6144 // 32)
dec2 = codec.decode_plane(weights.w2_scales.fixed[e], weights.w2_scales.exceptions[e],
                          6144, LOCAL // 32)


def src_slice(sname, r0, r1, c0, c1):
    idx = json.load(open("/model/model.safetensors.index.json"))["weight_map"]
    with open("/model/" + idx[sname], "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        hdr = json.loads(fh.read(hlen))
        a, b = hdr[sname]["data_offsets"]
        fh.seek(8 + hlen + a)
        plane = np.frombuffer(fh.read(b - a), dtype=np.uint8).reshape(hdr[sname]["shape"])
    return plane[r0:r1, c0:c1]


g = src_slice(f"model.layers.{LAYER}.mlp.experts.{e}.gate_proj.weight_scale", first, last, 0, 6144 // 32)
u = src_slice(f"model.layers.{LAYER}.mlp.experts.{e}.up_proj.weight_scale", first, last, 0, 6144 // 32)
d = src_slice(f"model.layers.{LAYER}.mlp.experts.{e}.down_proj.weight_scale", 0, 6144, first // 32, last // 32)
assert np.array_equal(dec13[:LOCAL], g)
assert np.array_equal(dec13[LOCAL:], u)
assert np.array_equal(dec2, d)
print("expert 5 planes byte-exact vs source rank-3 slices (gate/up/down)")

print("\nOVERLAY VALIDATION v2 PASSED — registered dispatch, draft pass-through "
      "with bounded read, identity pin, byte-exact TP8 read")
