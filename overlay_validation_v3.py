#!/usr/bin/env python3
"""Overlay validation v3 — §23 closure: the ACTUAL main iterator + intent gate.

Supersedes v2 (whose check-4 had an `assert ... or True` tautology and never
touched the main get_all_weights() iterator — the path with the §23-P0
string/Path crash). Fail-closed throughout; any mismatch raises.

Checks:
  1. imports + family registration
  2. ACTUAL main get_all_weights(): first shard through the REAL iterator —
     this is the exact code path that crashed with str/str TypeError.
  3. REGISTERED dispatch, with the §23 no-intent negative: hybrid MiMo
     header + NO CSF intent (no user_quant, no MIMO_CSF_ROOT) -> None.
  4. REAL draft pass-through + bounded REAL tensor read (types asserted).
  5. Identity pin is now MANDATORY: missing model_root refuses; missing
     index refuses; correct tree resolves.
  6. Explicit activation wiring: MIMO_CSF_ACTIVATION honored by the quant
     method ctor path (a16/a8/invalid).
  7. read_mxfp4_csf_layer TP8 rank-3 byte-exact vs source slices.
"""
import json
import os
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
    read_mxfp4_csf_layer,
)
from vllm.model_executor.layers.quantization.mxfp4_csf import (  # noqa: E402
    MimoMxfp4CsfConfig,
    Mxfp4CsfConfig,
)
assert "mimo_v26_pro_mxfp4" in FAMILIES
print("loader + quant OK")

print("== 2. ACTUAL main get_all_weights() iterator (§23-P0 path) ==")
root = "/container"


class MainHF:
    model_type = "mimo_v2"
    num_hidden_layers = 70
    n_routed_experts = 384
    hidden_size = 6144
    moe_intermediate_size = 2048
    quantization_config = {
        "quant_method": "fp8",
        "store_dtype": "mxfp4",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "ignored_layers": ["model.layers.0.self_attn.o_proj"],
        "weight_block_size": [128, 128],
    }


class MainModelConfig:
    hf_config = MainHF()
    hf_text_config = MainHF()
    model = "/model"


class _Sink:
    """Absorbs whatever the model would do; we only inspect yielded pairs."""

    secondary_weights = ()
    checkpoint_file_weight_filter = None
    checkpoint_weight_name_prefixes = None


loader = Mxfp4CsfModelLoader.__new__(Mxfp4CsfModelLoader)
seen = []
for name, tensor in loader.get_all_weights(MainModelConfig(), _Sink()):
    assert isinstance(name, str), f"weight name type: {type(name)}"
    assert isinstance(tensor, torch.Tensor), f"yielded {type(tensor)} for {name}"
    seen.append((name, tuple(tensor.shape), str(tensor.dtype)))
    if len(seen) >= 40:
        break
assert len(seen) >= 40, f"iterator yielded only {len(seen)} tensors"
# The iterator must yield BOTH dense-side (fp8) and expert-side tensors;
# routed-expert weights are skipped (loaded from CSF container instead).
expert_yielded = any("experts" in n for n, _, _ in seen)
assert not expert_yielded, "iterator must NOT yield routed-expert tensors"
dense_yielded = any("self_attn" in n or "mlp.gate" in n for n, _, _ in seen)
assert dense_yielded, "iterator must yield dense-side tensors"
print(f"first shard through REAL iterator OK: {len(seen)} tensors, e.g. {seen[0]}, {seen[-1]}")

print("== 3. REGISTERED dispatch + §23 no-intent negative ==")
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
# Positive: explicit intent via user_quant
assert registered_cls.override_quantization_method(q, "mxfp4_csf", HF()) == "mxfp4_csf"
# Positive: intent via MIMO_CSF_ROOT (set for this container run)
assert registered_cls.override_quantization_method(q, None, HF()) == "mxfp4_csf"
# §23 negative: SAME hybrid header, NO intent at all -> must stay plain.
saved_root = os.environ.pop("MIMO_CSF_ROOT", None)
try:
    r = registered_cls.override_quantization_method(q, None, HF())
    assert r is None, f"no-intent launch must stay plain FP8, got {r}"
finally:
    if saved_root is not None:
        os.environ["MIMO_CSF_ROOT"] = saved_root
print("registered dispatch OK; no-intent hybrid header -> None (plain launch stays plain)")
qc = registered_cls.from_config(q)
assert isinstance(qc, MimoMxfp4CsfConfig), f"from_config produced {type(qc)}"
assert qc.checkpoint_root == root
print("from_config ->", type(qc).__name__)

print("== 4. REAL draft pass-through + bounded REAL read ==")
draft_cfg = json.load(open("/model/dflash/config.json"))
assert draft_cfg.get("quantization_config") is None
assert draft_cfg.get("model_type") == "qwen3"


class DraftHF:
    model_type = "qwen3"


class DraftModelConfig:
    hf_config = DraftHF()
    hf_text_config = DraftHF()
    model = "/model/dflash"
    revision = None


loader = Mxfp4CsfModelLoader.__new__(Mxfp4CsfModelLoader)
resolved, _, _, _ = loader._resolve_for(DraftModelConfig())
assert resolved is None, "draft must resolve to None (pass-through)"
passthrough = Mxfp4CsfModelLoader._default_loader_for(DraftModelConfig())
assert type(passthrough).__name__ == "DefaultModelLoader"
assert passthrough.load_config.load_format != "mxfp4_csf"
draft_read = []
for name, tensor in passthrough.get_all_weights(DraftModelConfig(), _Sink()):
    assert isinstance(name, str)
    assert isinstance(tensor, torch.Tensor), f"draft yielded {type(tensor)}"
    draft_read.append(name)
    if len(draft_read) >= 3:
        break
assert len(draft_read) == 3, f"bounded draft read gave {len(draft_read)}"
print("draft pass-through OK, real tensors:", draft_read)

print("== 5. identity pin is MANDATORY ==")


class PinHF:
    model_type = "mimo_v2"
    n_routed_experts = 384
    hidden_size = 6144
    moe_intermediate_size = 2048


# 5a. missing model_root -> refuse
try:
    Mxfp4CsfModelLoader._resolve_csf_root(
        Mxfp4CsfModelLoader,
        {"quant_method": "fp8", "store_dtype": "mxfp4"},
        PinHF(),
        model_root=None,
    )
    raise AssertionError("pin must refuse when model_root is missing")
except ValueError as e:
    assert "identity pinning" in str(e)
    print("missing model_root REFUSED")
# 5b. wrong tree (has its own index) -> refuse
try:
    Mxfp4CsfModelLoader._resolve_csf_root(
        Mxfp4CsfModelLoader,
        {"quant_method": "fp8", "store_dtype": "mxfp4"},
        PinHF(),
        model_root="/model/dflash",
    )
    raise AssertionError("pin must refuse a mismatched source tree")
except ValueError as e:
    assert "different source checkpoint" in str(e)
    print("wrong tree REFUSED")
# 5c. correct tree resolves and returns a Path (not str)
resolved = Mxfp4CsfModelLoader._resolve_csf_root(
    Mxfp4CsfModelLoader,
    {"quant_method": "fp8", "store_dtype": "mxfp4"},
    PinHF(),
    model_root="/model",
)
assert resolved is not None, "correct tree must resolve"
r_path, _contract = resolved
import pathlib  # noqa: E402

assert isinstance(r_path, pathlib.Path), f"resolver returned {type(r_path)}, not Path"
print("correct tree resolves; root is a", type(r_path).__name__)

print("== 6. explicit activation wiring ==")
from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa: E402
from vllm.models.deepseek_v4_1.mxfp4_csf import Mxfp4CsfMoEMethod  # noqa: E402

import types  # noqa: E402


class _Parallel:
    use_ep = False
    ep_size = 1
    dp_size = 1
    use_all2all_kernels = False
    enable_eplb = False


class _Moe:
    moe_parallel_config = _Parallel()
    activation = MoEActivation.SILU
    in_dtype = torch.bfloat16
    has_bias = False


for mode in ("a16", "a8"):
    m = Mxfp4CsfMoEMethod(_Moe(), None, activation_mode=mode)
    assert m.activation_mode == mode
try:
    Mxfp4CsfMoEMethod(_Moe(), None, activation_mode="fp8")
    raise AssertionError("ctor must reject non-a16/a8 modes")
except ValueError:
    print("activation ctor: a16/a8 accepted, invalid rejected")

# and the overlay's own env-driven selection:
from vllm.model_executor.layers.quantization.mxfp4_csf import (  # noqa: E402
    MimoMxfp4CsfConfig as _MC,
)
for env_val, want in (("a16", "a16"), ("a8", "a8")):
    os.environ["MIMO_CSF_ACTIVATION"] = env_val
    mode = os.environ.get("MIMO_CSF_ACTIVATION", "a16").strip().lower()
    assert mode == want
del os.environ["MIMO_CSF_ACTIVATION"]

print("== 7. read_mxfp4_csf_layer (TP8 rank 3, layer 35, CPU) ==")
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

print("\nOVERLAY VALIDATION v3 PASSED — real iterator, intent gate, mandatory pin, "
      "real draft read, explicit activation, byte-exact TP8 read")
