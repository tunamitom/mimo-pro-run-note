#!/usr/bin/env python3
"""Single-layer GPU harness: plain MXFP4 scale path vs MXFP4-CSF scale path.

Phase 2 of the MiMo CSF feasibility spike (plan-csf-encoder-mimo.md sections
12-14). Runs inside `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta`
with the MiMo-V2.6-Pro-MOPD checkpoint mounted at /model and this toolkit at
/toolkit (writable output at /out).

What it does
------------
1. Loads layer 35, experts 0-47 (the TP8-local shard) real MXFP4 tensors and
   E8M0 scale bytes from the checkpoint (read-only).
2. PATH A (plain): PackedWeights with the original E8M0 block-scale grids ->
   plan_weights + prepare_weights (b12x). Measures prepared GPU bytes.
3. PATH B (CSF): the same scales encoded with the /toolkit MXFP4 disk codec,
   built into CsfScalePlanes + Mxfp4CsfWeights (caller-owned uint8 scratch),
   same plan/prepare. Measures prepared GPU bytes (compressed planes vs the
   expanded scale scratch).
4. Runs the same seeded MoE forward (64 tokens, top-8) through
   plan_execution / bind / run on both paths, times them with CUDA events and
   compares outputs, prepared scale storage bytes, and the GPU-side scale
   decompression against the original checkpoint bytes.

Layout note (verified in-image): b12x W13Layout.W31 means gate-first
([gate; up]) fused FC1 rows - the order vLLM fuses and the order
`mxfp4_csf_loader._load_mxfp4_csf_weights` produces (w1/gate rows first,
w3/up rows second with exception positions offset by local*columns). The
upstream DeepSeek MXFP4-CSF method plans with w13_layout="w31" on exactly
those planes, so this harness uses W13Layout.W31 for BOTH paths.
"""

import dataclasses
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, "/toolkit")
import mxfp4_csf as csf_codec  # noqa: E402  (disk codec: encode_plane/decode_plane)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MODEL_ROOT = "/model"
OUT_DIR = "/out"
LAYER = 35
EXPERTS = list(range(48))          # TP8-local shard at 384 experts / TP8
NUM_EXPERTS = 48
HIDDEN = 6144
INTER = 2048                       # MoE intermediate size (per expert matrix)
TOKENS = 64
TOPK = 8
SEED = 20261005
WARMUP = 3
ITERS = 25
MEM_FRACTION = 0.08                # owner-approved budget (~7.8 GiB on this GPU)
RESULTS_PATH = os.path.join(OUT_DIR, "harness_results.json")

_KEEPALIVE = []  # transient GPU objects held beyond their measuring window

results = {
    "geometry": {
        "model": "MiMo-V2.6-Pro-MOPD",
        "layer": LAYER,
        "experts": f"0-{NUM_EXPERTS - 1} (TP8-local shard of 384)",
        "num_experts": NUM_EXPERTS,
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "tokens": TOKENS,
        "topk": TOPK,
        "activation_mode": "a8 (w4a8_mx, MiMo live scheme: MXFP4 weights + dynamic MXFP8 activations)",
        "w13_layout": "w31 (gate-first [gate; up], vLLM fusion order)",
        "io_dtype": "bfloat16",
    },
    "path_a": {},
    "path_b": {},
    "notes": [
        "ComfyUI was stopped on the pinned GPU to free VRAM before this run; "
        "the harness ran with the enlarged torch.cuda.set_per_process_memory_fraction "
        "budget of 0.08 (~7.8 GiB). The co-resident qwen service was not disturbed.",
    ],
    "errors": [],
}


def log(msg):
    print(msg, flush=True)


def save():
    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)


def nvidia_smi(tag):
    import shutil
    exe = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"
    if not os.path.exists(exe) and not shutil.which("nvidia-smi"):
        out = f"(nvidia-smi not present in this container; tag={tag})"
        log(f"--- nvidia-smi {tag} ---\n{out}")
        return out
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,uuid,memory.used,memory.total",
             "--format=csv"],
            capture_output=True, text=True,
        ).stdout
    except FileNotFoundError:
        out = "(nvidia-smi binary not found in container)"
    log(f"--- nvidia-smi {tag} ---\n{out}")
    return out


def collect_gpu_storages(root):
    """Unique CUDA storage allocations reachable from `root` (bytes)."""
    storages = {}
    seen = set()
    stack = [root]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if obj.device.type == "cuda":
                st = obj.untyped_storage()
                storages[(obj.device.index, st.data_ptr(), st.nbytes())] = int(st.nbytes())
            continue
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            for f in dataclasses.fields(obj):
                try:
                    stack.append(getattr(obj, f.name))
                except Exception:
                    pass
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif isinstance(obj, dict):
            stack.extend(obj.values())
            stack.extend(obj.keys())
        elif hasattr(obj, "__dict__"):
            stack.append(obj.__dict__)
    return storages


def total_bytes(storages):
    return int(sum(storages.values()))


# ----------------------------------------------------------------------------
# Phase 0: environment
# ----------------------------------------------------------------------------
def main():
    log("=" * 78)
    log("MXFP4 plain-vs-CSF harness - MiMo layer 35, 48 experts (TP8-local)")
    log("=" * 78)
    nvidia_smi("before")
    log(f"torch {torch.__version__}, cuda available={torch.cuda.is_available()}")
    log(f"visible devices: {torch.cuda.device_count()}")
    cpu_only = os.environ.get("HARNESS_CPU_ONLY") == "1" or not torch.cuda.is_available()
    if cpu_only:
        results["status"] = "cpu_only"
        results["notes"].append(
            "GPU BLOCKER: at harness start the pinned GPU (host idx 0, "
            "GPU-253e168e-256f-4457-82f4-d26fd2f3c0bd, PCI 0000:11:00) was "
            "unavailable: kernel log shows 'Xid (PCI:0000:11:00): 79, GPU has "
            "fallen off the bus' at 2026-10-05 12:48:09 +07 (during the "
            "ComfyUI stop that freed VRAM) and 'Xid 154, GPU recovery action "
            "changed ... to 0x2 (OS Reboot)'. nvidia-smi cannot resolve the "
            "device ('Unable to determine the device handle for GPU0: "
            "0000:11:00.0: Unknown Error') and `nvidia-smi --gpu-reset -i 0` "
            "returns 'No devices found', so the GPU cannot be recovered "
            "in-band. Per owner policy no reboot was attempted. GPU phases "
            "(path A/B prepare+run, byte and output comparison) did NOT run; "
            "only the CPU-side phases below are real measurements.")
        log("CPU-ONLY MODE: " + results["notes"][-1])
    dev = torch.device("cuda", 0)
    if not cpu_only:
        torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, device=dev)
        log(f"per-process memory fraction set to {MEM_FRACTION}")

    # ------------------------------------------------------------------
    # Phase 1: load real checkpoint tensors (CPU)
    # ------------------------------------------------------------------
    from safetensors import safe_open

    index = json.load(open(os.path.join(MODEL_ROOT, "model.safetensors.index.json")))
    wmap = index["weight_map"]

    def tname(e, proj, kind):
        return f"model.layers.{LAYER}.mlp.experts.{e}.{proj}.{kind}"

    names = []
    for e in EXPERTS:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for kind in ("weight", "weight_scale"):
                names.append(tname(e, proj, kind))
    missing = [n for n in names if n not in wmap]
    if missing:
        raise RuntimeError(f"checkpoint index missing {len(missing)} names, e.g. {missing[:3]}")

    files = {}
    for n in names:
        files.setdefault(wmap[n], []).append(n)

    tensors = {}
    t0 = time.time()
    for fname, fnames in sorted(files.items()):
        with safe_open(os.path.join(MODEL_ROOT, fname), framework="pt", device="cpu") as fh:
            for n in fnames:
                tensors[n] = fh.get_tensor(n)
    log(f"loaded {len(tensors)} tensors from {len(files)} shards in {time.time() - t0:.1f}s")

    e0 = EXPERTS[0]
    for proj in ("gate_proj", "up_proj", "down_proj"):
        w = tensors[tname(e0, proj, "weight")]
        s = tensors[tname(e0, proj, "weight_scale")]
        log(f"  expert0 {proj}: weight {tuple(w.shape)} {w.dtype}, scale {tuple(s.shape)} {s.dtype}")

    # E8M0 byte statistics over all loaded planes (clamp exposure check).
    all_scales = torch.cat([
        tensors[tname(e, p, "weight_scale")].reshape(-1)
        for e in EXPERTS for p in ("gate_proj", "up_proj", "down_proj")
    ])
    smin = int(all_scales.min())
    smax = int(all_scales.max())
    gt247 = int((all_scales > 247).sum())
    eq255 = int((all_scales == 255).sum())
    results["e8m0_stats"] = {
        "min_byte": smin, "max_byte": smax,
        "count_gt_247": gt247, "count_eq_255": eq255,
        "total_scale_bytes": int(all_scales.numel()),
    }
    log(f"E8M0 scale bytes: min={smin} max={smax} >247: {gt247} ==255: {eq255}")
    if gt247:
        results["notes"].append(
            f"checkpoint contains {gt247} E8M0 bytes > 247 in this slice "
            f"(max {smax}); the b12x repacker clamps to 247 in some paths - "
            "see the decode comparison for how each path treats them")

    # Packed CPU tensors: w13 = [gate; up] rows (vLLM fusion order).
    w13 = torch.empty((NUM_EXPERTS, 2 * INTER, HIDDEN // 2), dtype=torch.uint8)
    w2 = torch.empty((NUM_EXPERTS, HIDDEN, INTER // 2), dtype=torch.uint8)
    w13_scales = torch.empty((NUM_EXPERTS, 2 * INTER, HIDDEN // 32), dtype=torch.uint8)
    w2_scales = torch.empty((NUM_EXPERTS, HIDDEN, INTER // 32), dtype=torch.uint8)
    for i, e in enumerate(EXPERTS):
        gate = tensors[tname(e, "gate_proj", "weight")]
        up = tensors[tname(e, "up_proj", "weight")]
        down = tensors[tname(e, "down_proj", "weight")]
        gs = tensors[tname(e, "gate_proj", "weight_scale")]
        us = tensors[tname(e, "up_proj", "weight_scale")]
        ds = tensors[tname(e, "down_proj", "weight_scale")]
        assert gate.shape == (INTER, HIDDEN // 2) and up.shape == (INTER, HIDDEN // 2)
        assert down.shape == (HIDDEN, INTER // 2)
        assert gs.shape == (INTER, HIDDEN // 32) and us.shape == (INTER, HIDDEN // 32)
        assert ds.shape == (HIDDEN, INTER // 32)
        w13[i, :INTER] = gate.view(torch.uint8)
        w13[i, INTER:] = up.view(torch.uint8)
        w2[i] = down.view(torch.uint8)
        w13_scales[i, :INTER] = gs
        w13_scales[i, INTER:] = us
        w2_scales[i] = ds
    del tensors, all_scales
    log("packed CPU tensors ready")

    # ------------------------------------------------------------------
    # Phase 2: encode the scales with the toolkit disk codec (CSF planes)
    # ------------------------------------------------------------------
    t0 = time.time()
    fixed13, exc13, fixed2, exc2 = [], [], [], []
    up_offset = np.uint32(INTER * (HIDDEN // 32))  # positions of the up half
    enc_ratio_fixed = enc_ratio_exc = 0
    for i, e in enumerate(EXPERTS):
        gs = w13_scales[i, :INTER].numpy()
        us = w13_scales[i, INTER:].numpy()
        ds = w2_scales[i].numpy()
        fg, eg = csf_codec.encode_plane(gs)
        fu, eu = csf_codec.encode_plane(us)
        eu = eu + up_offset
        fd, ed = csf_codec.encode_plane(ds)
        fixed13.append(torch.from_numpy(np.concatenate([fg, fu])))
        exc13.append(torch.from_numpy(np.concatenate([eg, eu]).astype(np.uint32)))
        fixed2.append(torch.from_numpy(fd))
        exc2.append(torch.from_numpy(ed.astype(np.uint32)))
        enc_ratio_fixed += fg.nbytes + fu.nbytes + fd.nbytes
        enc_ratio_exc += eg.nbytes + eu.nbytes + ed.nbytes
    orig = (2 * INTER * (HIDDEN // 32) + HIDDEN * (INTER // 32)) * NUM_EXPERTS
    results["csf_disk_encoding"] = {
        "original_scale_bytes": orig,
        "fixed_bytes": enc_ratio_fixed,
        "exception_bytes": enc_ratio_exc,
        "retained_pct": round(100.0 * (enc_ratio_fixed + enc_ratio_exc) / orig, 2),
    }
    log(f"encoded 48 experts x 3 planes in {time.time() - t0:.1f}s; "
        f"disk retention {results['csf_disk_encoding']['retained_pct']}% "
        f"({enc_ratio_fixed} fixed + {enc_ratio_exc} exception vs {orig} original)")
    save()

    # CPU round-trip check of the codec on this slice (cheap, catches encoder bugs).
    for i in (0, 17):
        pos_mask = (1 << 24) - 1
        exc = np.asarray(exc13[i])
        gate_exc = exc[(exc & pos_mask) < INTER * (HIDDEN // 32)]
        sel = csf_codec.selector_bytes(HIDDEN // 32)
        gate_fixed_len = (INTER // 16) * 16 * (1 + sel)
        rec = csf_codec.decode_plane(
            fixed13[i][:gate_fixed_len], gate_exc, INTER, HIDDEN // 32)
        assert (rec == w13_scales[i, :INTER].numpy()).all(), "codec round-trip failed"
    log("codec round-trip check passed (experts 0, 17 gate planes)")

    # ------------------------------------------------------------------
    # Phase 3: seeded inputs
    # ------------------------------------------------------------------
    g = torch.Generator(device="cpu").manual_seed(SEED)
    a_cpu = torch.empty((TOKENS, HIDDEN), dtype=torch.bfloat16).normal_(0.0, 0.125, generator=g)
    topk_ids = torch.randint(0, NUM_EXPERTS, (TOKENS, TOPK), generator=g, dtype=torch.int32)
    logits = torch.empty((TOKENS, TOPK), dtype=torch.float32).normal_(0.0, 1.0, generator=g)
    topk_weights = torch.softmax(logits, dim=-1).contiguous()
    a = a_cpu.to(dev) if not cpu_only else None
    topk_ids_g = topk_ids.to(dev) if not cpu_only else None
    topk_weights_g = topk_weights.to(dev) if not cpu_only else None
    out_gpu = (torch.empty((TOKENS, HIDDEN), dtype=torch.bfloat16, device=dev)
               if not cpu_only else None)

    # ------------------------------------------------------------------
    # Phase 4: CPU float reference (gross-correctness anchor; the GPU path
    # quantizes activations dynamically so exact equality is not expected)
    # ------------------------------------------------------------------
    FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

    def dequant(packed, scale, low_first):
        """packed [rows, K/2] u8, scale [rows, K/32] u8 -> f32 [rows, K]."""
        rows = packed.shape[0]
        lo = (packed & 0x0F).astype(np.int32)
        hi = (packed >> 4).astype(np.int32)
        nib = np.stack([lo, hi], axis=-1) if low_first else np.stack([hi, lo], axis=-1)
        vals = FP4[nib & 7] * np.where(nib & 8, -1.0, 1.0).astype(np.float32)
        vals = vals.reshape(rows, -1)
        s = np.exp2(np.minimum(scale.astype(np.int32), 247) - 127).astype(np.float32)
        return vals * np.repeat(s, 32, axis=1)

    def cpu_reference(token_slice, low_first):
        x = a_cpu[token_slice].float().numpy()          # [t, H]
        tw = topk_weights[token_slice].numpy()          # [t, TOPK]
        ti = topk_ids[token_slice].numpy()              # [t, TOPK]
        t = x.shape[0]
        y = np.zeros((t, HIDDEN), dtype=np.float32)
        for e in np.unique(ti):
            mask = ti == e                              # [t, TOPK]
            wts = (tw * mask).sum(axis=1)               # [t]
            sel = wts > 0
            if not sel.any():
                continue
            xs = x[sel]
            gate = dequant(w13[e, :INTER].numpy(), w13_scales[e, :INTER].numpy(), low_first)
            up = dequant(w13[e, INTER:].numpy(), w13_scales[e, INTER:].numpy(), low_first)
            h1 = xs @ gate.T
            h2 = xs @ up.T
            act = (1.0 / (1.0 + np.exp(-h1))) * h1 * h2  # silu(gate) * up
            down = dequant(w2[e].numpy(), w2_scales[e].numpy(), low_first)
            y[sel] += (act @ down.T) * wts[sel][:, None]
        return y

    ref_tokens = slice(0, 4)
    refs = {}
    for low_first in (True, False):
        t0 = time.time()
        refs[low_first] = cpu_reference(ref_tokens, low_first)
        log(f"CPU reference (low_nibble_first={low_first}) computed in {time.time() - t0:.1f}s")
    results["cpu_reference"] = {"tokens": list(range(4)), "note":
        "float32 dequantized-weight reference; the w4a8_mx path dynamically "
        "quantizes activations so only approximate agreement is expected"}

    def cos_rows(x, y):
        xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        yn = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
        return (xn * yn).sum(axis=1)

    # ------------------------------------------------------------------
    # Shared b12x objects
    # ------------------------------------------------------------------
    from b12x.moe import fused_moe

    source = fused_moe.PackedSource(
        format=fused_moe.PackedSourceFormat.MXFP4_E8M0_K32,
        w13_layout=fused_moe.W13Layout.W31,   # gate-first, vLLM fusion order
    )
    activation = fused_moe.ActivationSpec(
        mode=fused_moe.ActivationMode.A8,
        nonlinearity="silu",
        io_dtype=torch.bfloat16,
    )
    geometry = fused_moe.MoEGeometry(
        num_experts=NUM_EXPERTS, hidden_size=HIDDEN, intermediate_size=INTER,
    )

    # API-construction smoke test (CPU-only): exercises the exact b12x calls
    # the GPU phases make, minus prepare/run, so API mismatches surface even
    # when the GPU is unavailable.
    smoke = {}
    try:
        plan = fused_moe.plan_weights(source=source, activation=activation, geometry=geometry)
        smoke["plan_weights"] = "ok"
        smoke["plan_source_format"] = plan.source.format.value
        smoke["plan_w13_layout"] = plan.source.w13_layout.value
        smoke["plan_activation_mode"] = plan.activation.mode.value
        smoke["plan_prepared_format"] = str(plan.prepared_format)
        unit = torch.ones(NUM_EXPERTS, dtype=torch.float32)
        pa = fused_moe.PackedWeights(
            w13=w13, w2=w2, w13_block_scales=w13_scales, w2_block_scales=w2_scales,
            w13_global_scales=unit, w2_global_scales=unit,
            input_scale=unit.clone(), intermediate_scale=unit.clone(),
            immutable_input_scales=True)
        smoke["packed_weights"] = "ok"
        planes13 = fused_moe.CsfScalePlanes(tuple(fixed13), tuple(exc13))
        planes2 = fused_moe.CsfScalePlanes(tuple(fixed2), tuple(exc2))
        smoke["csf_scale_planes"] = "ok"
        up_off = INTER * (HIDDEN // 32)
        sorted_ok = all(
            bool((np.diff(np.asarray(p.numpy())) > 0).all()) if p.numel() > 1 else True
            for p in exc13)
        smoke["w13_exception_positions_sorted"] = sorted_ok
        scratch13_cpu = torch.empty(
            (NUM_EXPERTS, HIDDEN // 32, 2 * INTER), dtype=torch.uint8)
        scratch2_cpu = torch.empty(
            (NUM_EXPERTS, INTER // 32, HIDDEN), dtype=torch.uint8)
        mw = fused_moe.Mxfp4CsfWeights(
            w13=w13, w2=w2, w13_scales=planes13, w2_scales=planes2,
            w13_scale_scratch=scratch13_cpu, w2_scale_scratch=scratch2_cpu)
        smoke["mxfp4_csf_weights"] = "ok"
        smoke["scratch_pair_bytes"] = int(
            scratch13_cpu.numel() + scratch2_cpu.numel())
        del pa, mw, planes13, planes2, scratch13_cpu, scratch2_cpu
    except Exception as exc:
        smoke["error"] = repr(exc)
        log(f"API smoke test FAILED: {exc!r}")
    results["api_smoke"] = smoke
    log(f"API smoke test: {smoke}")
    save()

    if cpu_only:
        results["path_a"] = {"status": "not_run_gpu_unavailable"}
        results["path_b"] = {"status": "not_run_gpu_unavailable"}
        results["cpu_reference"]["token_l2_norms"] = {
            str(k): [round(float(v), 4) for v in np.linalg.norm(y, axis=1)]
            for k, y in refs.items()}
        results["cpu_reference"]["nibble_order_disambiguation"] = (
            "compare GPU outputs to both rows when GPU phases can run; the "
            "order matching at cosine ~1 is the checkpoint's nibble packing")
        save()
        nvidia_smi("after")
        log("CPU-only phases complete; GPU phases skipped (GPU unavailable).")
        log(json.dumps(results, indent=2))
        raise SystemExit(2)

    def run_path(label, prepare_fn):
        """plan/prepare/measure/run one path; returns (output_cpu, extras)."""
        log(f"\n===== {label} =====")
        t0 = time.time()
        experts, extras = prepare_fn()
        torch.cuda.synchronize()
        extras["prepare_seconds"] = round(time.time() - t0, 2)
        log(f"prepare took {extras['prepare_seconds']}s")

        storages = collect_gpu_storages(experts._impl)
        extras["prepared_bytes_total"] = total_bytes(storages)
        bs = collect_gpu_storages(experts._impl.w1_blockscale)
        bs.update(collect_gpu_storages(experts._impl.w2_blockscale))
        extras["prepared_bytes_scales"] = total_bytes(bs)
        dec = getattr(experts._impl, "mxfp4_csf", None)
        if dec is not None:
            extras["prepared_bytes_scales_compressed"] = total_bytes(collect_gpu_storages(dec))
        log(f"prepared GPU bytes: total={extras['prepared_bytes_total']} "
            f"scales={extras['prepared_bytes_scales']} "
            f"compressed={extras.get('prepared_bytes_scales_compressed', 0)}")

        # Execution declaration + caller scratch.
        declaration = fused_moe.plan_execution(
            experts=experts,
            capacity=fused_moe.ExecutionCapacity(max_tokens=TOKENS, top_k=TOPK),
            routing=fused_moe.RoutingSpec(),
        )
        specs = declaration.scratch_specs()
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in specs
        )
        extras["execution_scratch_bytes"] = int(
            sum(t.numel() * t.element_size() for t in scratch)
        )
        log(f"execution scratch: {extras['execution_scratch_bytes']} bytes "
            f"in {len(scratch)} buffers")

        binding = fused_moe.bind(
            declaration,
            scratch=scratch,
            a=a,
            experts=experts,
            topk_weights=topk_weights_g,
            topk_ids=topk_ids_g,
            output=out_gpu,
            input_scales_static=True,
        )
        log("bound; warming up")
        for _ in range(WARMUP):
            out = fused_moe.run(binding=binding)
        torch.cuda.synchronize()

        times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(ITERS):
            start.record()
            out = fused_moe.run(binding=binding)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        extras["time_ms_median"] = round(times[len(times) // 2], 4)
        extras["time_ms_mean"] = round(float(np.mean(times)), 4)
        extras["time_ms_min"] = round(times[0], 4)
        extras["time_ms_max"] = round(times[-1], 4)
        extras["timed_iterations"] = ITERS
        log(f"run times (ms): median={extras['time_ms_median']} "
            f"mean={extras['time_ms_mean']} min={extras['time_ms_min']} "
            f"max={extras['time_ms_max']}")
        out_cpu = out.detach().float().cpu().numpy().copy()
        return experts, scratch, out_cpu, extras

    # ------------------------------------------------------------------
    # PATH A: plain PackedWeights
    # ------------------------------------------------------------------
    def prepare_a():
        w_gpu = {
            "w13": w13.to(dev), "w2": w2.to(dev),
            "s13": w13_scales.to(dev), "s2": w2_scales.to(dev),
        }
        unit = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=dev)
        weights = fused_moe.PackedWeights(
            w13=w_gpu["w13"], w2=w_gpu["w2"],
            w13_block_scales=w_gpu["s13"], w2_block_scales=w_gpu["s2"],
            w13_global_scales=unit, w2_global_scales=unit,
            input_scale=unit.clone(), intermediate_scale=unit.clone(),
            immutable_input_scales=True,
        )
        plan = fused_moe.plan_weights(source=source, activation=activation, geometry=geometry)
        experts = fused_moe.prepare_weights(plan=plan, weights=weights)
        return experts, {}

    experts_a, scratch_a, out_a, extras_a = run_path("PATH A (plain MXFP4 scales)", prepare_a)
    results["path_a"].update(extras_a)

    # Snapshot A's prepared native scale storage (for byte comparison with B).
    a_s13 = experts_a._impl.w1_blockscale.detach().view(torch.uint8).cpu().numpy().copy()
    a_s2 = experts_a._impl.w2_blockscale.detach().view(torch.uint8).cpu().numpy().copy()
    results["path_a"]["scale_storage_shape"] = {
        "w1": list(a_s13.shape), "w2": list(a_s2.shape)}
    ref_cos = {str(k): [round(float(v), 6) for v in cos_rows(out_a[ref_tokens.start:ref_tokens.stop], y)]
               for k, y in refs.items()}
    results["path_a"]["cpu_reference_cosine"] = ref_cos
    log(f"path A vs CPU reference cosine per token: {ref_cos}")

    # Free path A before building path B.
    del experts_a, scratch_a
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    log(f"freed path A; allocated now: {torch.cuda.memory_allocated() / 2**20:.1f} MiB")
    save()

    # ------------------------------------------------------------------
    # PATH B: MXFP4-CSF compressed scales
    # ------------------------------------------------------------------
    from b12x._lib.quant.x4t_scales import decode_x4t_scales, make_x4t_scale_batch

    def prepare_b():
        w13_g, w2_g = w13.to(dev), w2.to(dev)
        scratch13 = torch.empty(
            (NUM_EXPERTS, HIDDEN // 32, 2 * INTER), dtype=torch.uint8, device=dev)
        scratch2 = torch.empty(
            (NUM_EXPERTS, INTER // 32, HIDDEN), dtype=torch.uint8, device=dev)
        # Per-expert CPU planes (CsfScalePlanes contract), then the GPU batch
        # exactly as planning.prepare_weights builds it (w31 -> row_rotation 0).
        planes13 = fused_moe.CsfScalePlanes(tuple(fixed13), tuple(exc13))
        planes2 = fused_moe.CsfScalePlanes(tuple(fixed2), tuple(exc2))
        batch13 = make_x4t_scale_batch(
            planes13.fixed, planes13.exceptions,
            rows=2 * INTER, columns=HIDDEN // 32, device=w13_g.device,
            exception_task_rows=64, exception_row_rotation=0)
        batch2 = make_x4t_scale_batch(
            planes2.fixed, planes2.exceptions,
            rows=HIDDEN, columns=INTER // 32, device=w2_g.device,
            exception_task_rows=64, exception_row_rotation=0)

        # GPU decompression losslessness check (same call planning makes).
        ids = torch.arange(NUM_EXPERTS, dtype=torch.int32, device=w13_g.device)
        dec13 = torch.empty((NUM_EXPERTS, 2 * INTER, HIDDEN // 32),
                            dtype=torch.uint8, device=w13_g.device)
        dec2 = torch.empty((NUM_EXPERTS, HIDDEN, INTER // 32),
                           dtype=torch.uint8, device=w13_g.device)
        decode_x4t_scales(batch13, ids, dec13)
        decode_x4t_scales(batch2, ids, dec2)
        torch.cuda.synchronize()
        orig13 = w13_scales.to(dev)
        orig2 = w2_scales.to(dev)
        m13 = (dec13 != orig13)
        m2 = (dec2 != orig2)
        results["gpu_decode_vs_checkpoint"] = {
            "w13_mismatched_bytes": int(m13.sum()),
            "w2_mismatched_bytes": int(m2.sum()),
            "w13_max_abs_byte_diff": int((dec13.int() - orig13.int()).abs().max()),
            "w2_max_abs_byte_diff": int((dec2.int() - orig2.int()).abs().max()),
        }
        log(f"GPU decode_x4t_scales vs checkpoint bytes: {results['gpu_decode_vs_checkpoint']}")
        del dec13, dec2, orig13, orig2, m13, m2, ids
        torch.cuda.empty_cache()

        weights = fused_moe.Mxfp4CsfWeights(
            w13=w13_g, w2=w2_g,
            w13_scales=batch13, w2_scales=batch2,
            w13_scale_scratch=scratch13, w2_scale_scratch=scratch2,
        )
        plan = fused_moe.plan_weights(source=source, activation=activation, geometry=geometry)
        experts = fused_moe.prepare_weights(plan=plan, weights=weights)
        # Keep the transient x4t batches alive for the decoder's lifetime; they
        # are NOT reachable from experts._impl, so prepared-byte accounting
        # (which walks experts._impl) reflects only what the prepared form owns.
        _KEEPALIVE.extend((batch13, batch2))
        return experts, {}

    experts_b, scratch_b, out_b, extras_b = run_path("PATH B (MXFP4-CSF scales)", prepare_b)
    results["path_b"].update(extras_b)

    # Full-expansion scale-storage comparison: decode ALL experts into the
    # canonical scale storage and compare byte-for-byte with path A's storage.
    impl_b = experts_b._impl
    decoder = impl_b.mxfp4_csf
    if decoder is not None:
        all_ids = torch.arange(NUM_EXPERTS, dtype=torch.int32, device=dev).view(NUM_EXPERTS, 1)
        decoder.decode(all_ids, impl_b.w1_blockscale, impl_b.w2_blockscale)
        torch.cuda.synchronize()
        b_s13 = impl_b.w1_blockscale.detach().view(torch.uint8).cpu().numpy()
        b_s2 = impl_b.w2_blockscale.detach().view(torch.uint8).cpu().numpy()
        diff13 = (b_s13 != a_s13)
        diff2 = (b_s2 != a_s2)
        results["scale_storage_comparison"] = {
            "w1_shape": list(a_s13.shape), "w2_shape": list(a_s2.shape),
            "w1_mismatched_bytes": int(diff13.sum()),
            "w2_mismatched_bytes": int(diff2.sum()),
            "identical": bool(not diff13.any() and not diff2.any()),
        }
        log(f"prepared scale storage A vs B: {results['scale_storage_comparison']}")
        del b_s13, b_s2, diff13, diff2
    else:
        results["scale_storage_comparison"] = {"identical": None,
                                               "note": "no mxfp4_csf decoder on prepared experts"}
    save()

    # Inline-representation probe: the review predicts GPU-native inline scale
    # storage has tile overhead + heavy-tile fallback. At this geometry
    # (intermediate 2048, 2048 % 128 == 0) the compact inline kernels are NOT
    # selected (they need n % 128 == 64), so this measures what the inline
    # representation WOULD cost, not what this run executes.
    try:
        from b12x._lib.quant.mxfp4_csf_inline import build_mxfp4_csf_inline
        e = NUM_EXPERTS
        inline13 = build_mxfp4_csf_inline(
            impl_b.w1_blockscale.view(torch.uint8).view(e, -1),
            rows=2 * INTER, columns=HIDDEN // 32, group_rows=INTER)
        inline2 = build_mxfp4_csf_inline(
            impl_b.w2_blockscale.view(torch.uint8).view(e, -1),
            rows=HIDDEN, columns=INTER // 32, group_rows=HIDDEN)
        inl = collect_gpu_storages(inline13)
        inl.update(collect_gpu_storages(inline2))
        results["inline_probe"] = {
            "inline_scale_bytes": total_bytes(inl),
            "expanded_scale_bytes": results["path_b"]["prepared_bytes_scales"],
            "note": "would-be inline storage at this geometry; not used by the "
                    "executed path (compact inline kernels require "
                    "intermediate_size % 128 == 64)",
        }
        log(f"inline probe: {results['inline_probe']}")
        del inline13, inline2, inl
    except Exception as exc:
        results["inline_probe"] = {"error": repr(exc)}
        log(f"inline probe failed: {exc!r}")
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Output comparison
    # ------------------------------------------------------------------
    diff = np.abs(out_a - out_b)
    results["output_max_abs_diff"] = float(diff.max())
    results["output_mean_abs_diff"] = float(diff.mean())
    results["outputs_identical"] = bool(np.array_equal(out_a, out_b))
    results["path_b"]["cpu_reference_cosine"] = {
        str(k): [round(float(v), 6) for v in cos_rows(out_b[ref_tokens.start:ref_tokens.stop], y)]
        for k, y in refs.items()}
    log(f"\noutput A vs B: max_abs_diff={results['output_max_abs_diff']} "
        f"mean_abs_diff={results['output_mean_abs_diff']} "
        f"identical={results['outputs_identical']}")
    log(f"path B vs CPU reference cosine: {results['path_b']['cpu_reference_cosine']}")

    results["bytes_ratio_b_over_a"] = round(
        results["path_b"]["prepared_bytes_total"] /
        max(1, results["path_a"]["prepared_bytes_total"]), 4)
    results["scale_bytes_ratio_b_over_a"] = round(
        results["path_b"]["prepared_bytes_scales"] /
        max(1, results["path_a"]["prepared_bytes_scales"]), 4)

    # Final cleanup.
    del experts_b, scratch_b
    gc.collect()
    torch.cuda.empty_cache()
    results["notes"].append(
        "Both paths ran the same b12x w4a8_mx kernels with identical seeded "
        "inputs; path B additionally runs the per-call CSF scale expansion "
        "(intermediate 2048 does not select the compact inline kernels).")
    save()
    nvidia_smi("after")
    log("\nfinal results:")
    log(json.dumps({k: v for k, v in results.items() if k != "notes"}, indent=2))
    save()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        results["errors"].append(traceback.format_exc())
        save()
        nvidia_smi("after-failure")
        sys.exit(1)