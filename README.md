# mimo-pro-run-note

> **Credits.** The CSF encoding/decoding was derived from the work of the folks
> at [Local Inference Lab](https://github.com/local-inference-lab) — this repo
> is just an application of their format. The serving image is built on top of
> their [vLLM](https://github.com/local-inference-lab/vllm) and
> [flashinfer](https://github.com/local-inference-lab/flashinfer) forks, with
> these additions: [vLLM #1022 — MXFP4-CSF support for MiMo-V2.6-Pro](https://github.com/local-inference-lab/vllm/pull/1022)
> and [flashinfer #13 — MiMo W4A16/MXFP4 MoE workstream](https://github.com/local-inference-lab/flashinfer/pull/13).
> This is just how I run the model — all credits go to the team at L-I-L.

Run **MiMo-V2.6-Pro** on 8 GPUs with vLLM, using compressed **MXFP4-CSF**
weights. This repo contains the run recipe and the toolkit that encodes the
model's MXFP4 expert scales into the CSF container format the vLLM
`mxfp4_csf` loader expects.

- Weights: [XiaomiMiMo/MiMo-V2.6-Pro-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD) (public, ~534 GiB, includes the `dflash/` draft model)
- Serving image: `ghcr.io/tunamitom/kk-mimo-pro` — vLLM with the MiMo MXFP4-CSF loader

## Requirements

- 8 GPUs with ≥ 90 GiB VRAM each (tested on 8× RTX PRO 6000)
- Docker + NVIDIA Container Toolkit
- ~550 GiB disk for the base weights, ~550 GiB more for the CSF container

## 1. Download the weights

```bash
pip install -U "huggingface_hub[cli]"
hf download XiaomiMiMo/MiMo-V2.6-Pro-MOPD \
  --revision adea8e2c5373181e5a973fa1ecb343cb31af214b \
  --local-dir MiMo-V2.6-Pro-MOPD
```

Use that pinned revision — the build script records it for byte-exact
rebuilds. The repo already contains `dflash/` (the speculative draft model),
so there is nothing else to fetch.

Also write the download manifest (per-file LFS SHA-256s) that the build step
uses for its fail-closed receipt checks:

```bash
python3 - <<'PY'
import json, urllib.request
repo, rev = "XiaomiMiMo/MiMo-V2.6-Pro-MOPD", "adea8e2c5373181e5a973fa1ecb343cb31af214b"
tree = json.load(urllib.request.urlopen(
    f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true", timeout=120))
manifest = {f["path"]: {"lfs_oid": (f.get("lfs") or {}).get("oid"), "size": f["size"]} for f in tree}
json.dump(manifest, open("source-manifest.json", "w"))
print("manifest entries:", len(manifest))
PY
```

(Generate it from the same revision you downloaded.)

## 2. Encode the CSF container

The runtime does not read the MXFP4 expert `weight_scale` tensors directly —
it expects them recoded as CSF planes: `<name>.mxfp4_csf_fixed` (U8) plus
`<name>.mxfp4_csf_exceptions` (U32). `build_mimo_csf.py` streams each source
shard once, byte-copies every non-scale tensor, recodes the whitelisted expert
scales, and verifies every receipt (source SHA-256 vs the download manifest's
LFS OIDs) — any mismatch raises.

```bash
git clone https://github.com/tunamitom/mimo-pro-run-note.git toolkit

docker run --rm \
  -v $PWD/MiMo-V2.6-Pro-MOPD:/model:ro \
  -v $PWD/source-manifest.json:/source-manifest.json:ro \
  -v $PWD/toolkit:/toolkit:ro \
  -v $PWD/csf-container:/out-container \
  ghcr.io/tunamitom/kk-mimo-pro \
  python /toolkit/build_mimo_csf.py build

docker run --rm \
  -v $PWD/MiMo-V2.6-Pro-MOPD:/model:ro \
  -v $PWD/source-manifest.json:/source-manifest.json:ro \
  -v $PWD/toolkit:/toolkit:ro \
  -v $PWD/csf-container:/out-container \
  ghcr.io/tunamitom/kk-mimo-pro \
  python /toolkit/build_mimo_csf.py verify
```

`verify` is fail-closed and writes `verification.json` into the container.
The build is memory-bounded (one tensor in flight per worker).

## 3. Serve on 8 GPUs

```bash
docker run -d --name mimo-pro \
  --gpus all --shm-size 32g -p 8000:8000 \
  -v $PWD/MiMo-V2.6-Pro-MOPD:/model:ro \
  -v $PWD/csf-container:/container:ro \
  ghcr.io/tunamitom/kk-mimo-pro \
  vllm serve /model \
    --served-model-name mimo-pro --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --load-format mxfp4_csf --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.982 \
    --max-model-len 1048576 --max-num-seqs 64 \
    --max-num-batched-tokens 8192 --max-num-scheduled-tokens 2048 \
    --prefill-compute-share 0.8 --kv-cache-dtype fp8 --block-size 64 \
    --attention-backend B12X --moe-backend b12x --linear-backend b12x \
    --max-cudagraph-capture-size 256 \
    --media-io-kwargs '{"video":{"num_frames":96}}' \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --speculative-config '{"method":"dflash","model":"/model/dflash","num_speculative_tokens":7,"draft_tensor_parallel_size":8,"max_model_len":1048576,"enable_adaptive_verification":true,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","attention_backend":"B12X","kv_cache_dtype":"fp8"}' \
    --enable-prefix-caching --enable-chunked-prefill --async-scheduling \
    --no-scheduler-reserve-full-isl \
    --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
    --generation-config auto
```

Notes:

- Everything runs on the 8 GPUs — the KV-cache pool is auto-sized from the
  VRAM left after the weights. No separate cache/offload GPU is involved.
- `--gpu-memory-utilization 0.982` assumes the 8 cards are dedicated to this
  model; use ~0.90 on shared machines.
- On a shared machine, replace `--gpus all` with
  `--gpus '"device=0,1,2,3,4,5,6,7"'` (or your own indices).
- `--load-format mxfp4_csf` reads the main weights from `/model` and the
  recoded expert scales from the `/container` CSF store.

## 4. Test

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mimo-pro",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

The first start loads and compiles for quite a while — the first successful
chat completion is the signal it's ready.

## Toolkit reference

- `mxfp4_csf.py` — the MXFP4-CSF codec (`row-base-offset1-u24-exceptions/1`).
- `build_mimo_csf.py` — build + fail-closed verify of the CSF container
  (step 2 above).
- `migrate_suffix_v2.py` — header-only container migration (writer v1 → v2
  lookup names).
- `scan_mimo_mxfp4.py` — source-checkpoint MXFP4 tensor scan.
- `test_mxfp4_csf.py`, `container_reader_test.py`,
  `fixture_build_reader_test.py`, `fixture_run_reader_test.py`,
  `overlay_validation*.py`, `harness_mxfp4_layer35.py`,
  `harness_tp8_mxfp4.py` — codec roundtrip tests, reader tests against the
  real migrated container, and TP8-geometry equivalence harnesses.

### CSF disk format (`mxfp4_csf.py`)

Codec `row-base-offset1-u24-exceptions/1`:

- **fixed stream**: `rows/16` slabs; each slab = 16 base bytes (one per row,
  0..254) followed by `16 * ceil(columns/8)` selector bytes (per row, little
  bitorder). Value = `base + selector bit`.
- **exceptions**: `uint32` words, sorted unique row-major positions,
  `word = position | (original_byte << 24)`.
- Deterministic encoder: per row, base = lowest `b` in `0..254` maximizing
  `count(v == b) + count(v == b + 1)`.