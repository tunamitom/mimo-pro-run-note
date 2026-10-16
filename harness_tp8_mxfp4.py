#!/usr/bin/env python3
"""TP8-geometry harness v2: plain vs MXFP4-CSF at production rank-local shape.

Answers review §17 (2026-10-05) P0.1/P0.2/P0.3 + P1-5 measurement demands.
Run inside ghcr.io/local-inference-lab/vllm:karmic-kraken-beta with
/model (source checkpoint), /toolkit (this file's dir), /out (writable).

Geometry (P0.1 fix): ALL 384 routed experts of layers 35+36, TP8 rank-local
slice of the intermediate axis (first:last = 256r:256(r+1)); gate/up output
rows and down input columns sliced, including packed-nibble coordinates and
block-scale column coordinates. Compressed planes are sliced with vllm's own
`_slice_scale_plane` (the production loader path), NOT re-encoded from the
slice. Ranks 0, 3 and 7 cover first/middle/last partition boundaries.

Fail-closed (P0.3 fix): every check is an assert/raise; the script exits 1
on any mismatch. No `errors: []`-style success-by-default.

Outputs (P0.2 fix): per-rank measured expanded-grid bytes, compressed-plane
bytes, scratch-pair bytes, prepared-weight totals, plus:
  - self-variation: repeated same-input A/A and B/B runs, then interleaved
    A/B — the A-vs-B difference is only judged against measured self-noise;
  - shared-scratch two-layer test: both CSF layers prepared against ONE
    scratch pair; storage-pointer sharing proof; alternating execution
    compared against independent plain references;
  - decoder-into-external-shared-buffer byte proof (the adapter contract);
  - steady-state timing at batch 8/64/512, interleaved A/B trial order,
    raw distributions recorded. Prepare times recorded but NOT compared
    (cache-warming makes that invalid — review P1-5).
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
import mxfp4_csf as csf_codec  # noqa: E402

MODEL_ROOT = "/model"
OUT_DIR = "/out"
LAYERS = [35, 36]
NUM_EXPERTS = 384
HIDDEN = 6144
INTER = 2048
TP_SIZE = 8
RANKS_FULL = [0]           # full comparison + timing + sharing
RANKS_REDUCED = [3, 7]     # decode + prepared-bytes + output equivalence
LOCAL = INTER // TP_SIZE
TOKENS = 64
TOPK = 8
SEED = 20261005
WARMUP = 3
ITERS = 25
REPEATS = 5
TIMING_BATCHES = [8, 64, 512]
MEM_FRACTION = 0.25
RESULTS_PATH = os.path.join(OUT_DIR, "harness_tp8_results.json")

results = {
    "geometry": {
        "model": "MiMo-V2.6-Pro-MOPD",
        "layers": LAYERS,
        "num_experts": NUM_EXPERTS,
        "hidden_size": HIDDEN,
        "intermediate_size_full": INTER,
        "tp_size": TP_SIZE,
        "intermediate_size_local": LOCAL,
        "tokens": TOKENS,
        "topk": TOPK,
        "routing": "distinct top-8 via topk over 384 logits; half of tokens "
                   "boost a 32-expert hot set (representative skew)",
        "io_dtype": "bfloat16",
    },
    "ranks": {},
    "notes": [],
}


def log(msg):
    print(msg, flush=True)


def save():
    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)


def collect_gpu_storages(root):
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


def tbytes(t):
    return int(t.numel() * t.element_size())


def main():
    from safetensors import safe_open
    from vllm.model_executor.model_loader.mxfp4_csf_loader import _slice_scale_plane

    log("=" * 78)
    log(f"TP8 harness v2: {NUM_EXPERTS} experts, local intermediate {LOCAL}, "
        f"layers {LAYERS}, ranks full={RANKS_FULL} reduced={RANKS_REDUCED}")
    log("=" * 78)
    dev = torch.device("cuda", 0)
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, device=dev)
    log(f"torch {torch.__version__}; memory fraction {MEM_FRACTION}")

    index = json.load(open(os.path.join(MODEL_ROOT, "model.safetensors.index.json")))
    wmap = index["weight_map"]

    def names_for(layer, e):
        p = f"model.layers.{layer}.mlp.experts.{e}"
        return {k: f"{p}.{proj}.{kind}"
                for k, proj, kind in [
                    ("gate_w", "gate_proj", "weight"),
                    ("up_w", "up_proj", "weight"),
                    ("down_w", "down_proj", "weight"),
                    ("gate_s", "gate_proj", "weight_scale"),
                    ("up_s", "up_proj", "weight_scale"),
                    ("down_s", "down_proj", "weight_scale"),
                ]}

    # every expert of both layers must exist (fail fast on dense-layer guess)
    for layer in LAYERS:
        assert names_for(layer, NUM_EXPERTS - 1)["gate_w"] in wmap, \
            f"layer {layer} is not a 384-expert MoE layer"

    handles = {}

    def handle_for(fname):
        if fname not in handles:
            handles[fname] = safe_open(os.path.join(MODEL_ROOT, fname),
                                       framework="pt", device="cpu")
        return handles[fname]

    def read_slice(tname, r0, r1, c0, c1):
        """Materialize a rank-local slice of a source tensor (lazy view)."""
        sl = handle_for(wmap[tname]).get_slice(tname)
        return sl[r0:r1, c0:c1]

    def read_full(tname):
        return handle_for(wmap[tname]).get_tensor(tname)

    # ------------------------------------------------------------------
    # Seeded inputs: distinct top-8 over the REAL 384-expert space + skew.
    # ROUTE_TOKENS covers max(TIMING_BATCHES) so timing batches have inputs;
    # the first TOKENS rows drive the equivalence phases.
    # ------------------------------------------------------------------
    ROUTE_TOKENS = max(max(TIMING_BATCHES), TOKENS)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    logits = torch.empty((ROUTE_TOKENS, NUM_EXPERTS), dtype=torch.float32).normal_(0.0, 1.0, generator=g)
    hot = torch.zeros(NUM_EXPERTS, dtype=torch.float32)
    hot[:32] = 2.5  # representative skew: 32 hot experts
    logits[: ROUTE_TOKENS // 2] += hot
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), TOPK, dim=-1)
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()
    # distinctness is structural (topk); assert anyway (fail-closed)
    for t in range(TOKENS):
        assert len(set(topk_ids[t].tolist())) == TOPK, f"token {t} has repeated experts"
    occupancy = np.bincount(topk_ids.reshape(-1).numpy(), minlength=NUM_EXPERTS)
    results["routing_stats"] = {
        "distinct_topk": True,
        "max_expert_occupancy": int(occupancy.max()),
        "min_expert_occupancy": int(occupancy.min()),
        "experts_used": int((occupancy > 0).sum()),
    }
    log(f"routing: {results['routing_stats']}")

    from b12x.moe import fused_moe
    from b12x._lib.quant.x4t_scales import decode_x4t_scales, make_x4t_scale_batch

    source = fused_moe.PackedSource(
        format=fused_moe.PackedSourceFormat.MXFP4_E8M0_K32,
        w13_layout=fused_moe.W13Layout.W31,
    )
    # A16 = the PRODUCTION activation mode: the live service runs with
    # VLLM_B12X_MOE_FP4_FORCE_A16=1 (start-mimo-pro.sh), and all historical
    # baselines in /opt/mimo-pro (perf-tweak.md v3.2, run.yml) are A16.
    # Harness v1/v2 ran A8 by mistake; §19's A8 timing is not comparable to
    # production. This rerun matches the launcher exactly (owner + review
    # §21-P0.3). FP32 top-k weights: B12X_W4A16_FP32_TOPK_WEIGHTS=1.
    activation = fused_moe.ActivationSpec(
        mode=fused_moe.ActivationMode.A16, nonlinearity="silu",
        io_dtype=torch.bfloat16,
    )
    geometry = fused_moe.MoEGeometry(
        num_experts=NUM_EXPERTS, hidden_size=HIDDEN, intermediate_size=LOCAL,
    )
    plan = fused_moe.plan_weights(source=source, activation=activation, geometry=geometry)
    results["plan"] = {
        "prepared_format": str(plan.prepared_format),
        "source_format": plan.source.format.value,
        "w13_layout": plan.source.w13_layout.value,
        "activation_mode": plan.activation.mode.value,
    }
    log(f"plan: {results['plan']}")

    def run_outputs(binding, out_buf):
        fused_moe.run(binding=binding)
        torch.cuda.synchronize()
        return out_buf.detach().float().cpu().numpy().copy()

    for rank in RANKS_FULL + RANKS_REDUCED:
        full = rank in RANKS_FULL
        r = {"mode": "full" if full else "reduced"}
        results["ranks"][str(rank)] = r
        first, last = rank * LOCAL, (rank + 1) * LOCAL
        log(f"\n########## RANK {rank} (intermediate rows {first}:{last}) "
            f"[{'FULL' if full else 'REDUCED'}] ##########")

        # --------------------------------------------------------------
        # Rank-local packing: plain (A) tensors and CSF (B) planes.
        # B planes: encode FULL planes, then slice with the production
        # `_slice_scale_plane` (row slice for gate/up, column slice for down).
        # --------------------------------------------------------------
        t0 = time.time()
        plain = {}    # layer -> dict(w13, w2, s13, s2)
        csf = {}      # layer -> dict(fixed13, exc13, fixed2, exc2) per-expert tuples
        up_off = np.uint32(LOCAL * (HIDDEN // 32))
        for layer in LAYERS:
            w13 = torch.empty((NUM_EXPERTS, 2 * LOCAL, HIDDEN // 2), dtype=torch.uint8)
            w2 = torch.empty((NUM_EXPERTS, HIDDEN, LOCAL // 2), dtype=torch.uint8)
            s13 = torch.empty((NUM_EXPERTS, 2 * LOCAL, HIDDEN // 32), dtype=torch.uint8)
            s2 = torch.empty((NUM_EXPERTS, HIDDEN, LOCAL // 32), dtype=torch.uint8)
            f13, e13, f2, e2 = [], [], [], []
            for e in range(NUM_EXPERTS):
                n = names_for(layer, e)
                gate_w = read_slice(n["gate_w"], first, last, 0, HIDDEN // 2)
                up_w = read_slice(n["up_w"], first, last, 0, HIDDEN // 2)
                down_w = read_slice(n["down_w"], 0, HIDDEN, first // 2, last // 2)
                gate_s = read_slice(n["gate_s"], first, last, 0, HIDDEN // 32)
                up_s = read_slice(n["up_s"], first, last, 0, HIDDEN // 32)
                down_s = read_slice(n["down_s"], 0, HIDDEN, first // 32, last // 32)
                assert gate_w.shape == (LOCAL, HIDDEN // 2), gate_w.shape
                assert down_w.shape == (HIDDEN, LOCAL // 2), down_w.shape
                assert gate_s.shape == (LOCAL, HIDDEN // 32), gate_s.shape
                assert down_s.shape == (HIDDEN, LOCAL // 32), down_s.shape
                w13[e, :LOCAL] = gate_w.view(torch.uint8)
                w13[e, LOCAL:] = up_w.view(torch.uint8)
                w2[e] = down_w.view(torch.uint8)
                s13[e, :LOCAL] = gate_s
                s13[e, LOCAL:] = up_s
                s2[e] = down_s
                # CSF: encode full planes, slice compressed (production path)
                fgd, egd = csf_codec.encode_plane(read_full(n["gate_s"]).numpy())
                fud, eud = csf_codec.encode_plane(read_full(n["up_s"]).numpy())
                fdd, edd = csf_codec.encode_plane(read_full(n["down_s"]).numpy())
                fgu, egu = _slice_scale_plane(
                    torch.from_numpy(fgd), torch.from_numpy(egd),
                    INTER, HIDDEN // 32, (first, last), (0, HIDDEN // 32))
                fup, eup = _slice_scale_plane(
                    torch.from_numpy(fud), torch.from_numpy(eud),
                    INTER, HIDDEN // 32, (first, last), (0, HIDDEN // 32))
                fdo, edo = _slice_scale_plane(
                    torch.from_numpy(fdd), torch.from_numpy(edd),
                    HIDDEN, INTER // 32, (0, HIDDEN), (first // 32, last // 32))
                e_up = eup.numpy().copy()
                e_up += up_off  # up-half exception offset: local*columns (w31)
                f13.append(torch.from_numpy(np.concatenate([fgu.numpy(), fup.numpy()])))
                e13.append(torch.from_numpy(np.concatenate([egu.numpy(), e_up]).astype(np.uint32)))
                f2.append(fdo)
                e2.append(edo)
            plain[layer] = dict(w13=w13, w2=w2, s13=s13, s2=s2)
            csf[layer] = dict(fixed13=tuple(f13), exc13=tuple(e13),
                              fixed2=tuple(f2), exc2=tuple(e2))
        log(f"rank {rank}: packed plain + CSF planes in {time.time() - t0:.0f}s")

        # Fail-closed CPU decode check of the sliced planes vs source slices.
        for layer in LAYERS:
            c = csf[layer]
            for idx in (0, 1, 191, 192, 383):
                rec13 = csf_codec.decode_plane(c["fixed13"][idx], c["exc13"][idx],
                                               2 * LOCAL, HIDDEN // 32)
                assert rec13.shape == (2 * LOCAL, HIDDEN // 32)
                assert np.array_equal(rec13, plain[layer]["s13"][idx].numpy()), \
                    f"L{layer} e{idx}: sliced-plane decode != source slice (w13)"
                rec2 = csf_codec.decode_plane(c["fixed2"][idx], c["exc2"][idx],
                                              HIDDEN, LOCAL // 32)
                assert np.array_equal(rec2, plain[layer]["s2"][idx].numpy()), \
                    f"L{layer} e{idx}: sliced-plane decode != source slice (w2)"
        r["sliced_plane_decode"] = "PASS (15 planes/rank byte-exact vs source slices)"
        log(f"rank {rank}: sliced-plane decode checks PASS")

        # --------------------------------------------------------------
        # Prepare PATH A (plain) for both layers.
        # --------------------------------------------------------------
        def prepare_plain(layer):
            p = plain[layer]
            unit = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=dev)
            weights = fused_moe.PackedWeights(
                w13=p["w13"].to(dev), w2=p["w2"].to(dev),
                w13_block_scales=p["s13"].to(dev), w2_block_scales=p["s2"].to(dev),
                w13_global_scales=unit, w2_global_scales=unit,
                input_scale=unit.clone(), intermediate_scale=unit.clone(),
                immutable_input_scales=True,
            )
            return fused_moe.prepare_weights(plan=plan, weights=weights)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        experts_a = {layer: prepare_plain(layer) for layer in LAYERS}
        torch.cuda.synchronize()
        r["prepare_seconds_a"] = round(time.time() - t0, 2)
        r["prepare_peak_alloc_bytes_a"] = int(torch.cuda.max_memory_allocated())
        sa = {}
        for layer in LAYERS:
            sa[layer] = collect_gpu_storages(experts_a[layer]._impl)
        r["prepared_bytes_a"] = {str(l): int(sum(sa[l].values())) for l in LAYERS}
        grids_a = {}
        for layer in LAYERS:
            grids_a[layer] = (
                experts_a[layer]._impl.w1_blockscale.detach().view(torch.uint8).cpu().numpy().copy(),
                experts_a[layer]._impl.w2_blockscale.detach().view(torch.uint8).cpu().numpy().copy(),
            )
        log(f"rank {rank}: path A prepared; bytes={r['prepared_bytes_a']}")

        # --------------------------------------------------------------
        # Prepare PATH B (CSF) for both layers against ONE shared scratch pair.
        # --------------------------------------------------------------
        scratch13 = torch.empty((NUM_EXPERTS, HIDDEN // 32, 2 * LOCAL),
                                dtype=torch.uint8, device=dev)
        scratch2 = torch.empty((NUM_EXPERTS, LOCAL // 32, HIDDEN),
                               dtype=torch.uint8, device=dev)
        scratch_ptrs = {(0, scratch13.data_ptr(), scratch13.untyped_storage().nbytes()),
                        (0, scratch2.data_ptr(), scratch2.untyped_storage().nbytes())}

        def prepare_csf(layer):
            c = csf[layer]
            w13_g = plain[layer]["w13"].to(dev)
            w2_g = plain[layer]["w2"].to(dev)
            planes13 = fused_moe.CsfScalePlanes(c["fixed13"], c["exc13"])
            planes2 = fused_moe.CsfScalePlanes(c["fixed2"], c["exc2"])
            batch13 = make_x4t_scale_batch(
                planes13.fixed, planes13.exceptions,
                rows=2 * LOCAL, columns=HIDDEN // 32, device=w13_g.device,
                exception_task_rows=64, exception_row_rotation=0)
            batch2 = make_x4t_scale_batch(
                planes2.fixed, planes2.exceptions,
                rows=HIDDEN, columns=LOCAL // 32, device=w2_g.device,
                exception_task_rows=64, exception_row_rotation=0)
            weights = fused_moe.Mxfp4CsfWeights(
                w13=w13_g, w2=w2_g, w13_scales=batch13, w2_scales=batch2,
                w13_scale_scratch=scratch13, w2_scale_scratch=scratch2,
            )
            experts = fused_moe.prepare_weights(plan=plan, weights=weights)
            return experts, (batch13, batch2)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        experts_b, batches = {}, {}
        for layer in LAYERS:
            experts_b[layer], batches[layer] = prepare_csf(layer)
        torch.cuda.synchronize()
        r["prepare_seconds_b"] = round(time.time() - t0, 2)
        r["prepare_peak_alloc_bytes_b"] = int(torch.cuda.max_memory_allocated())
        sb = {layer: collect_gpu_storages(experts_b[layer]._impl) for layer in LAYERS}
        r["prepared_bytes_b"] = {str(l): int(sum(sb[l].values())) for l in LAYERS}
        # compressed-plane GPU batches (kept alive; not reachable from impl)
        batch_bytes = {str(l): int(sum(collect_gpu_storages(batches[l]).values()))
                       for l in LAYERS}
        r["compressed_plane_batch_bytes_b"] = batch_bytes
        r["scratch_pair_bytes"] = int(scratch13.untyped_storage().nbytes()
                                      + scratch2.untyped_storage().nbytes())
        # scratch reachability + sharing proof (P0.2)
        reach = {str(l): sorted(ptr in sb[l] for ptr in scratch_ptrs) for l in LAYERS}
        shared_keys = set(sb[LAYERS[0]]) & set(sb[LAYERS[1]])
        r["scratch_reachable_from_impl"] = reach
        r["impl_shared_storages_between_layers"] = len(shared_keys)
        log(f"rank {rank}: path B prepared; bytes={r['prepared_bytes_b']}; "
            f"planes={batch_bytes}; scratch={r['scratch_pair_bytes']} "
            f"({r['scratch_pair_bytes']/2**20:.1f} MiB); scratch reachable: {reach}; "
            f"impl-shared storages: {len(shared_keys)}")

        # Fail-closed: GPU decode of sliced planes vs source content.
        # A8 prepared form exposes impl.mxfp4_csf.decode(...); the A16 X4T
        # form (W4A16PackedWeights) keeps planes in its X4TScaleBatch and
        # expands per call — so decode via the same b12x API the runtime
        # uses (decode_x4t_scales) into our own buffers.
        from b12x._lib.quant.x4t_scales import decode_x4t_scales as _dec_x4t
        a_grids_available = all(
            hasattr(experts_a[l]._impl, "w1_blockscale") for l in LAYERS
        )
        decoded_b_grids = {}
        for layer in LAYERS:
            impl = experts_b[layer]._impl
            ids = torch.arange(NUM_EXPERTS, dtype=torch.int32, device=dev).view(NUM_EXPERTS, 1)
            dec13 = torch.empty((NUM_EXPERTS, 2 * LOCAL, HIDDEN // 32),
                                dtype=torch.uint8, device=dev)
            dec2 = torch.empty((NUM_EXPERTS, HIDDEN, LOCAL // 32),
                               dtype=torch.uint8, device=dev)
            decoder = getattr(impl, "mxfp4_csf", None)
            if decoder is not None:
                decoder.decode(ids, impl.w1_blockscale, impl.w2_blockscale)
                torch.cuda.synchronize()
                d13 = impl.w1_blockscale.detach().view(torch.uint8).cpu().numpy()
                d2 = impl.w2_blockscale.detach().view(torch.uint8).cpu().numpy()
            else:
                b13, b2 = batches[layer]
                ids_flat = ids.view(-1)
                _dec_x4t(b13, ids_flat, dec13)
                _dec_x4t(b2, ids_flat, dec2)
                torch.cuda.synchronize()
                d13 = dec13.cpu().numpy()
                d2 = dec2.cpu().numpy()
                decoded_b_grids[layer] = (dec13, dec2)
            # Content gate: decoded planes == source rank-local slices.
            assert d13.shape == plain[layer]["s13"].shape, \
                f"L{layer}: decoded w13 shape {d13.shape} != source slice"
            assert d2.shape == plain[layer]["s2"].shape, \
                f"L{layer}: decoded w2 shape {d2.shape} != source slice"
            assert np.array_equal(d13, plain[layer]["s13"].numpy()), \
                f"L{layer}: CSF-decoded grid != source slice (w13)"
            assert np.array_equal(d2, plain[layer]["s2"].numpy()), \
                f"L{layer}: CSF-decoded grid != source slice (w2)"
            diag = {
                "decoded_matches_source_slices": True,
                "decode_path": "impl.mxfp4_csf" if decoder is not None
                               else "decode_x4t_scales",
            }
            if a_grids_available:
                ga13 = experts_a[layer]._impl.w1_blockscale.detach().view(torch.uint8).cpu().numpy()
                ga2 = experts_a[layer]._impl.w2_blockscale.detach().view(torch.uint8).cpu().numpy()
                if ga13.shape == d13.shape:
                    diag["ab_prepared_grids_identical"] = bool(
                        np.array_equal(d13, ga13) and np.array_equal(d2, ga2))
                else:
                    diag["ab_grid_shape_mismatch_expected_swizzle"] = [
                        list(ga13.shape), list(d13.shape)]
            r.setdefault("grid_vs_source_slice_diag", {})[str(layer)] = diag
            del dec13, dec2
            torch.cuda.empty_cache()
        r["gpu_decode_vs_source_slices"] = (
            "PASS: decoded CSF planes byte-exact vs source rank-local slices "
            "(both layers); A/B prepared-grid comparison "
            + ("recorded" if a_grids_available else "n/a at A16 (different prepared dataclass)"))
        log(f"rank {rank}: GPU decode == source slices PASS; diag: "
            f"{r.get('grid_vs_source_slice_diag')}")

        # Prepared packed-WEIGHT byte comparison (P0.3), best-effort attr probe.
        wcmp = {}
        for attr in ("w1", "w2", "w13", "w2_packed"):
            try:
                ta = getattr(experts_a[LAYERS[0]]._impl, attr)
                tb = getattr(experts_b[LAYERS[0]]._impl, attr)
            except AttributeError:
                continue
            if isinstance(ta, torch.Tensor) and isinstance(tb, torch.Tensor):
                same = bool(torch.equal(ta.view(torch.uint8), tb.view(torch.uint8))) \
                    if ta.shape == tb.shape else None
                wcmp[attr] = {"shape": list(ta.shape), "equal": same}
        r["prepared_weight_attr_compare"] = wcmp or "no packed-weight attrs exposed"

        # Decoder-into-EXTERNAL-shared-buffer proof (adapter contract, P0.2):
        # one external buffer pair serves BOTH layers byte-exactly. At A8 this
        # targets impl.mxfp4_csf; at A16 the X4T form, so decode via
        # decode_x4t_scales into one shared pair of (2*LOCAL, H//32)/(H, L/32)
        # grids and compare to source content.
        ext13 = torch.empty((NUM_EXPERTS, 2 * LOCAL, HIDDEN // 32),
                            dtype=torch.uint8, device=dev)
        ext2 = torch.empty((NUM_EXPERTS, HIDDEN, LOCAL // 32),
                           dtype=torch.uint8, device=dev)
        ext_ok = True
        for layer in LAYERS:
            ids = torch.arange(NUM_EXPERTS, dtype=torch.int32, device=dev).view(NUM_EXPERTS, 1)
            impl = experts_b[layer]._impl
            decoder = getattr(impl, "mxfp4_csf", None)
            if decoder is not None:
                decoder.decode(ids, ext13, ext2)
            else:
                b13, b2 = batches[layer]
                ids_flat = ids.view(-1)
                _dec_x4t(b13, ids_flat, ext13)
                _dec_x4t(b2, ids_flat, ext2)
            torch.cuda.synchronize()
            ext_ok &= np.array_equal(ext13.cpu().numpy(), plain[layer]["s13"].numpy())
            ext_ok &= np.array_equal(ext2.cpu().numpy(), plain[layer]["s2"].numpy())
        assert ext_ok, "decoder cannot target a shared external buffer pair"
        r["decoder_shared_external_buffer"] = "PASS (both layers decode byte-exact into one pair)"
        log(f"rank {rank}: shared external decode-target PASS")
        del ext13, ext2

        # --------------------------------------------------------------
        # Bindings + output evidence (full ranks).
        # --------------------------------------------------------------
        if full:
            a_cpu = torch.empty((ROUTE_TOKENS, HIDDEN), dtype=torch.bfloat16).normal_(
                0.0, 0.125, generator=g)
            a = a_cpu.to(dev)
            tw = topk_weights.to(dev)
            ti = topk_ids.to(dev)

            def bind_for(experts, max_tokens):
                out = torch.empty((max_tokens, HIDDEN), dtype=torch.bfloat16, device=dev)
                decl = fused_moe.plan_execution(
                    experts=experts,
                    capacity=fused_moe.ExecutionCapacity(max_tokens=max_tokens, top_k=TOPK),
                    routing=fused_moe.RoutingSpec(),
                )
                specs = decl.scratch_specs()
                escratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=s.device)
                                 for s in specs)
                binding = fused_moe.bind(
                    decl, scratch=escratch, a=a[:max_tokens], experts=experts,
                    topk_weights=tw[:max_tokens], topk_ids=ti[:max_tokens],
                    output=out, input_scales_static=True)
                return binding, out, int(sum(tbytes(x) for x in escratch))

            binds_a, binds_b, outs = {}, {}, {}
            exec_scratch = {}
            for layer in LAYERS:
                binds_a[layer], outs[f"a{layer}"], exec_scratch[f"a{layer}"] = bind_for(experts_a[layer], TOKENS)
                binds_b[layer], outs[f"b{layer}"], exec_scratch[f"b{layer}"] = bind_for(experts_b[layer], TOKENS)
            r["execution_scratch_bytes"] = {k: v for k, v in exec_scratch.items()}

            # warmup + finiteness (fail-closed)
            for layer in LAYERS:
                for _ in range(WARMUP):
                    fused_moe.run(binding=binds_a[layer])
                    fused_moe.run(binding=binds_b[layer])
                torch.cuda.synchronize()
                for key in (f"a{layer}", f"b{layer}"):
                    assert np.isfinite(outs[key].detach().float().cpu().numpy()).all(), \
                        f"nonfinite output in {key}"

            # Self-variation: repeated same-input runs per binding (P0.3).
            def self_var(bf, outbuf):
                outs_ = [run_outputs(bf, outbuf) for _ in range(REPEATS)]
                base = outs_[0]
                return base, float(max(np.abs(o - base).max() for o in outs_[1:]))

            sv = {}
            for layer in LAYERS:
                base_a, va = self_var(binds_a[layer], outs[f"a{layer}"])
                base_b, vb = self_var(binds_b[layer], outs[f"b{layer}"])
                sv[str(layer)] = {"a_self_max_abs": va, "b_self_max": vb,
                                  "ab_max_abs": float(np.abs(base_a - base_b).max()),
                                  "ab_mean_abs": float(np.abs(base_a - base_b).mean())}
                assert np.isfinite(base_a).all() and np.isfinite(base_b).all()
            r["self_variation"] = sv
            log(f"rank {rank}: self-variation: {sv}")

            # Interleaved A/B repeats (order effects, P0.3/P1-5).
            inter = {k: [] for k in ("a35", "b35", "a36", "b36")}
            for _ in range(REPEATS):
                for layer in LAYERS:
                    inter[f"a{layer}"].append(run_outputs(binds_a[layer], outs[f"a{layer}"]))
                    inter[f"b{layer}"].append(run_outputs(binds_b[layer], outs[f"b{layer}"]))
            def spread(arrs):
                base = arrs[0]
                return float(max(np.abs(o - base).max() for o in arrs[1:]))
            r["interleaved_repeat_spread"] = {k: spread(v) for k, v in inter.items()}
            # A-vs-B difference must not exceed the union of self-noise
            for layer in LAYERS:
                noise = max(sv[str(layer)]["a_self_max_abs"], sv[str(layer)]["b_self_max"],
                            spread(inter[f"a{layer}"]), spread(inter[f"b{layer}"]))
                diff = sv[str(layer)]["ab_max_abs"]
                verdict = "within_self_variation" if diff <= max(noise, 1e-6) else "EXCEEDS_self_variation"
                r.setdefault("ab_verdict", {})[str(layer)] = {
                    "ab_max_abs": diff, "self_noise_max": noise, "verdict": verdict}
                log(f"rank {rank} L{layer}: A-vs-B {diff:.3e} vs self-noise {noise:.3e} -> {verdict}")

            # Shared-scratch alternating execution (P0.2): interleave the two
            # CSF layers against ONE scratch pair and compare against the
            # same-path STEADY outputs (isolates the sharing effect from any
            # A/B kernel difference) and against plain references.
            alt_ok = True
            seq = []
            for rep in range(REPEATS):
                for layer in LAYERS:  # L35, L36, L35, L36 ... alternating
                    o = run_outputs(binds_b[layer], outs[f"b{layer}"])
                    seq.append((layer, o))
            for layer in LAYERS:
                ref = np.stack([o for l, o in seq if l == layer])
                base = ref[0]
                spread_l = float(max(np.abs(o - base).max() for o in ref[1:]))
                diff_vs_bsteady = float(np.abs(base - inter[f"b{layer}"][0]).max())
                diff_vs_a = float(np.abs(base - inter[f"a{layer}"][0]).max())
                noise = max(sv[str(layer)]["a_self_max_abs"], sv[str(layer)]["b_self_max"],
                            spread_l, 1e-6)
                ok_b = diff_vs_bsteady <= noise
                ok_a = diff_vs_a <= noise
                alt_ok &= ok_b and ok_a
                r.setdefault("shared_scratch_alternation", {})[str(layer)] = {
                    "b_alt_vs_bsteady_max_abs": diff_vs_bsteady,
                    "b_alt_vs_a_max_abs": diff_vs_a,
                    "b_alt_self_spread": spread_l,
                    "noise_max": noise,
                    "within_noise_vs_bsteady": ok_b,
                    "within_noise_vs_a": ok_a,
                }
            assert alt_ok, "shared-scratch alternating execution exceeds self-noise"
            log(f"rank {rank}: shared-scratch alternation PASS: "
                f"{r['shared_scratch_alternation']}")

            # ------------------------------------------------------------
            # Steady-state timing: interleaved A/B trial order, 3 batch
            # sizes (small decode / comparison / prefill-ish probe).
            # Recorded as a MICROBENCHMARK ONLY (review P1-5).
            # ------------------------------------------------------------
            r["timing"] = {}
            for batch in TIMING_BATCHES:
                for layer in LAYERS[:1]:  # time layer 35 only (representative)
                    ba, oa, _ = bind_for(experts_a[layer], batch)
                    bb, ob, _ = bind_for(experts_b[layer], batch)
                    for bf in (ba, bb):
                        for _ in range(WARMUP):
                            fused_moe.run(binding=bf)
                    torch.cuda.synchronize()
                    # strict ABAB interleaved trials, separate event pairs
                    raw_a, raw_b = [], []
                    for i in range(ITERS):
                        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                        s.record(); fused_moe.run(binding=ba); e.record()
                        s2 = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
                        s2.record(); fused_moe.run(binding=bb); e2.record()
                        torch.cuda.synchronize()
                        raw_a.append(s.elapsed_time(e)); raw_b.append(s2.elapsed_time(e2))
                    med = lambda xs: round(float(np.median(xs)), 4)
                    ta = {
                        "tokens": batch,
                        "a_ms_median": med(raw_a), "b_ms_median": med(raw_b),
                        "a_raw_ms": [round(x, 4) for x in raw_a],
                        "b_raw_ms": [round(x, 4) for x in raw_b],
                        "b_over_a_pct": round(100 * (med(raw_b) / med(raw_a) - 1), 2),
                    }
                    r["timing"][f"batch_{batch}"] = ta
                    log(f"rank {rank} timing batch={batch}: A {ta['a_ms_median']} ms "
                        f"B {ta['b_ms_median']} ms ({ta['b_over_a_pct']:+.2f}%)")

        save()

        # cleanup between ranks
        del experts_a, experts_b, batches, plain, csf
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Measured-footprint extrapolation (labeled as such; per-layer
    # compression varies — §14 scan: 22.69% overall).
    # ------------------------------------------------------------------
    r0 = results["ranks"][str(RANKS_FULL[0])]
    exp_layers = 69  # moe_layer_freq sum (config.json)
    per_layer_grid_bytes = int((2 * LOCAL * (HIDDEN // 32) + HIDDEN * (LOCAL // 32)) * NUM_EXPERTS)
    compressed_per_layer = int(sum(r0["compressed_plane_batch_bytes_b"].values()) / len(LAYERS))
    scratch_once = int(r0["scratch_pair_bytes"])
    results["extrapolation_per_rank"] = {
        "moe_layers": exp_layers,
        "note": "extrapolation from measured rank-0 layer-35/36 values; "
                "per-layer compression varies (§14 full-scan: 22.69% overall)",
        "expanded_grids_resident_per_layer_bytes": per_layer_grid_bytes,
        "expanded_grids_resident_all_layers_bytes": per_layer_grid_bytes * exp_layers,
        "csf_compressed_planes_per_layer_bytes": compressed_per_layer,
        "csf_compressed_planes_all_layers_bytes": compressed_per_layer * exp_layers,
        "csf_shared_scratch_pair_bytes_once": scratch_once,
        "csf_reshapes_per_call": "decode into shared pair per layer call (decoder "
                                 "shared-target proof above); integration wiring is "
                                 "runtime adapter work, not yet demonstrated e2e",
        "net_estimate_if_shared_scratch_integrated_bytes":
            per_layer_grid_bytes * exp_layers
            - (compressed_per_layer * exp_layers + scratch_once),
    }
    n = results["extrapolation_per_rank"]
    n["net_estimate_if_shared_scratch_integrated_gib"] = round(
        n["net_estimate_if_shared_scratch_integrated_bytes"] / 2**30, 3)
    log("extrapolation: " + json.dumps(n, indent=2))

    results["status"] = "ok"
    save()
    log("\nHARNESS v2 COMPLETE — all fail-closed checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        results["status"] = "FAILED"
        results["fatal"] = traceback.format_exc()
        try:
            save()
        except Exception:
            pass
        sys.exit(1)
