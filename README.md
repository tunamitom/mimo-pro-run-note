# mimo-pro-run-note

> **Credits.** The CSF encoding/decoding was derived from the work of the folks
> at [Local Inference Lab](https://github.com/local-inference-lab) — this repo
> is just an application of their format. The serving image is built on top of
> their [vLLM](https://github.com/local-inference-lab/vllm) and
> [flashinfer](https://github.com/local-inference-lab/flashinfer) forks, with
> these additions: [vLLM #1022 — MXFP4-CSF support for MiMo-V2.6-Pro](https://github.com/local-inference-lab/vllm/pull/1022)
> and [flashinfer #13 — MiMo W4A16/MXFP4 MoE workstream](https://github.com/local-inference-lab/flashinfer/pull/13).
> This is just how I run the model — all credits go to the team at L-I-L.
> The FLARE path below is NVIDIA's engine, served through our own fork
> ([`tunamitom/flare-local-inference`](https://github.com/tunamitom/flare-local-inference)).

Run **MiMo-V2.6-Pro** on 8 GPUs. There are two paths in this repo, and the
first one is the one to use:

1. **FLARE (recommended)** — NVIDIA's
   [flare-local-inference](https://github.com/NVIDIA/flare-local-inference)
   Rust engine serving a prepared runtime package: one metadata-only manifest
   plus immutable payload blobs, its own paged-KV scheduler, DFlash draft
   widths 1–8 with adaptive verification, prefix cache and HTTP server. No
   Python, no runtime compilation, no CSF container. Faster than the vLLM path
   on the same hardware — measured on 8× RTX PRO 6000 (30 s cells): **260 tok/s**
   single stream, **682 tok/s** at 8-way, **880 tok/s** at 16-way aggregate
   decode, prefill ~8.6k tok/s at 8k context.
2. **vLLM + MXFP4-CSF (alternative)** — the original recipe: encode the model's
   MXFP4 expert scales into the CSF container format and serve with the vLLM
   `mxfp4_csf` loader. Kept below for reproducibility and for the CSF toolkit.

- Weights: [XiaomiMiMo/MiMo-V2.6-Pro-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD) (public, ~534 GiB, includes the `dflash/` draft model)
- FLARE package: [mitomtuna/MiMo-V2.6-Pro-MOPD-FLARE](https://huggingface.co/mitomtuna/MiMo-V2.6-Pro-MOPD-FLARE)
  and image `ghcr.io/tunamitom/mimo-pro-flare` — see
  [FLARE runtime](#recommended-flare-runtime) below
- vLLM serving image: `ghcr.io/tunamitom/kk-mimo-pro` — vLLM with the MiMo
  MXFP4-CSF loader

## Requirements

- 8 GPUs with ≥ 90 GiB VRAM each (tested on 8× RTX PRO 6000)
- Docker + NVIDIA Container Toolkit
- Disk: **~537 GiB** for the FLARE package, or ~550 GiB of base weights plus
  ~550 GiB more for the CSF container on the vLLM path

## Recommended: FLARE runtime

[FLARE](https://github.com/NVIDIA/flare-local-inference) is NVIDIA's Rust
serving engine. It does not read the MXFP4-CSF container and does not use vLLM:
it serves a **prepared runtime package** — one metadata-only manifest plus
immutable payload blobs — with its own paged-KV scheduler, DFlash draft widths
1–8 with adaptive verification, prefix cache and HTTP server. The CSF steps in
the alternative path below do not apply here: nothing is encoded, and there is
no `--load-format`.

| | |
|---|---|
| Package | [mitomtuna/MiMo-V2.6-Pro-MOPD-FLARE](https://huggingface.co/mitomtuna/MiMo-V2.6-Pro-MOPD-FLARE) — 576.7 GB (537 GiB), 601 safetensors files (primary manifest `model.safetensors`, payloads under `weights/tp8/rank-0..7/` and `deployments/gb202-tp8/`) |
| Image | `ghcr.io/tunamitom/mimo-pro-flare:74eabe8f` — digest `sha256:6b0fedd4d5ef04e62ea8ab36af96b5448ba5beb79f435103480759a0eae5111c` |
| Engine fork | [`tunamitom/flare-local-inference`](https://github.com/tunamitom/flare-local-inference), branch `mimo-pro` (`74eabe8f`) — the export recipe lives in the branch under `packaging/mimo-pro/` |

### Option 1 — build from source (preferred)

Everything comes from one checkout of the `mimo-pro` branch: the export runs
this branch's `ignition` tree, and the image compiles this branch's `helios`
and `flare` trees. Hardware: eight RTX PRO 6000 Blackwell (sm_120, 96 GB) on
PCIe, tensor parallel 8, no NVLink.

```bash
# 1. the mimo-pro branch of the engine fork (-b selects it; nothing else does)
git clone -b mimo-pro https://github.com/tunamitom/flare-local-inference.git
cd flare-local-inference
git checkout 74eabe8f        # the commit the published package was exported from

# 2. the export environment (packaging/mimo-pro/ is a directory in that branch, not a branch)
packaging/mimo-pro/setup-venv.sh
```

Needs `uv` and Python 3.12 on `PATH` (the script creates `.venv` at the repo
root); branch `mimo-pro` already points at `74eabe8f` today — check it out
explicitly if you want a byte-identical rebuild.

`setup-venv.sh` runs `uv sync --frozen --extra prepare` (FlashInfer pinned at
`344e4837` in `uv.lock`) and then applies the five b12x patches in `patches/`
to the installed FlashInfer. The export fails, or silently prepares slower
graphs, without them. Export environment: Rust, Python 3.12, CMake, Ninja,
`protoc`, OpenSSL and Python headers, liburing, and the CUDA toolkit (`nvcc`,
CUPTI). The image build is a plain `docker build` — the Dockerfile compiles
`flare/` and `helios/` with `cargo --locked` inside the container, so only
Docker is needed for that step.

Download the source checkpoint (the same one the CSF path uses) and export:

```bash
hf download XiaomiMiMo/MiMo-V2.6-Pro-MOPD --local-dir MiMo-V2.6-Pro-MOPD

prlimit --memlock=unlimited:unlimited env \
  FLARE_MIMO_SWITCH_SPECULATION=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  packaging/mimo-pro/export.sh "$PWD/MiMo-V2.6-Pro-MOPD"
```

`export.sh` fixes every option: TP8, `--max-concurrency 64`,
`--image-concurrency 4`, `--prefill-chunk-size 8192`, DFlash MTP with
`--mtp-draft-tokens 7` and packed verification over 256 target rows
(`--speculative-token-capacity 256`), `--expert-scale-compression csf`, and
the recipe's A4/A16 prefill variants from 512 tokens.
`FLARE_MIMO_SWITCH_SPECULATION=1` makes one-request rounds choose their
verification width on device (the switch); the shipped package is exported
with it. The export needs unlocked memory (`ulimit -l` unlimited), runs about
an hour, and peaks at about 1.1 TB of disk (about 0.56 TB of package plus the
package cache). The package lands in the Flare model cache under
`models/<checkpoint-dir>/gb202-tp8-<date>/` beside the shared blobs — there is
no `--output` flag.

Build the serving image from the same checkout:

```bash
packaging/mimo-pro/build-image.sh            # local/flare:rt-mimo-pro-<commit>
```

A self-built package is signed with your own per-user development key; the
first run prints that fingerprint and refuses until you accept it (or pass the
two `--dangerously-*` flags with your own values).

### Option 2 — download the prepared package

```bash
pip install -U "huggingface_hub[cli]"
hf download mitomtuna/MiMo-V2.6-Pro-MOPD-FLARE \
  --local-dir MiMo-V2.6-Pro-MOPD-FLARE
```

576.7 GB (537 GiB) across 601 safetensors files (the download also fetches
`README.md`, `LICENSE`, `publisher.json`, `.gitattributes`). Keep the published
layout intact: `model.safetensors` is the manifest, payloads live under
`weights/tp8/rank-0..7/` and `deployments/gb202-tp8/`. Serve
`MiMo-V2.6-Pro-MOPD-FLARE/model.safetensors`. It is signed by the
**development** key in `publisher.json`, and it needs the matching engine image
(`:74eabe8f`) — the ABI changes between engine builds.

If you would rather keep everything in the Hub cache, FLARE can resolve the
same reference itself: `run mitomtuna/MiMo-V2.6-Pro-MOPD-FLARE@gb202
--filename model.safetensors --cache-dir /cache`. The trust flags below still
apply.

### Serve

`serve.sh` in the branch runs the shipped flags (throughput scheduler, adaptive
MTP verification with 7 draft tokens, checkpoint segment 256, KV cache sized
from memory, a 64 GB RAM prefix cache plus a 160 GB staged disk tier) and
passes `seccomp-io-uring.json`, which the disk tier needs because Docker's
default seccomp profile blocks `io_uring`:

```bash
GPUS=0,1,2,3,4,5,6,7 \
PUBLISHER_FINGERPRINT=<your key fingerprint> \
PUBLISHER_NAMESPACE=<your namespace> \
  packaging/mimo-pro/serve.sh IMAGE PACKAGE_DIR STATE_DIR PREFIX_CACHE_DIR 8000
```

`STATE_DIR` holds the publisher keys (`development-publisher.json` for a
self-built package, `publisher.json` for the downloaded one) and the
`trust.toml` that records acceptance. `healthcheck.sh` probes `/health`
without curl, which the image does not have.

Manual equivalent, if you would rather not use the script:

```bash
docker run -d --name mimo-pro \
  --gpus all --network host \
  --user "$(id -u):$(id -g)" \
  --security-opt seccomp=./flare-local-inference/packaging/mimo-pro/seccomp-io-uring.json \
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v $PWD/MiMo-V2.6-Pro-MOPD-FLARE:/model:ro \
  -v $PWD/flare-state:/state \
  -v $PWD/flare-prefix-cache:/prefix-cache \
  ghcr.io/tunamitom/mimo-pro-flare:74eabe8f \
  run /model/model.safetensors \
    --gpus 0,1,2,3,4,5,6,7 \
    --max-concurrency 64 \
    --mtp-draft-tokens 7 --mtp-adaptive-verification \
    --scheduler-policy throughput \
    --min-checkpoint-segment 256 \
    --kv-cache-limit memory \
    --prefix-cache-memory 64G \
    --prefix-cache-disk /prefix-cache \
    --prefix-cache-disk-limit 160G \
    --prefix-cache-disk-transport staged \
    --served-model-name mimo-pro \
    --publisher-keys /model/publisher.json \
    --trust-store /state/trust.toml \
    --dangerously-approve-key <key fingerprint> \
    --dangerously-accept-publisher <namespace> \
    --host 0.0.0.0 --port 8000
```

The `--dangerously-*` flags accept exactly one key, once, and record it in
`/state/trust.toml`. Check the fingerprint against `publisher.json` in the
package before you accept it.

### Check

```bash
curl -sf http://127.0.0.1:8000/health          # 200 = ready
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mimo-pro",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

### Notes

- The image runs as uid `10001` (`flare`). With bind mounts owned by your user
  you need `--user "$(id -u):$(id -g)"`; without it `flare` fails with
  `Permission denied` on `/state` or the package directory.
- `--ulimit memlock=-1:-1` matters: the engine locks host memory for its
  caches and would otherwise fail to raise the limit.
- `--prefix-cache-memory 64G` is the host-RAM tier and optional. The
  `--prefix-cache-disk` tier needs `seccomp-io-uring.json`; without it the
  server dies at startup when it cannot open `io_uring_setup`.
- First start takes several minutes (537 GiB of weights off disk); `/health`
  returning 200 is the readiness signal, not the port opening.
- `--max-concurrency 64` is what the published package was exported for; its
  graph catalog covers draft widths 1–8, so `--mtp-draft-tokens` can move in
  that range without a re-export. Changing concurrency needs a re-export.
- `--min-checkpoint-segment 256` lets short messages share the next prefill
  pass instead of stopping for a checkpoint at every message end.

## Alternative: vLLM with MXFP4-CSF

Run **MiMo-V2.6-Pro** on 8 GPUs with vLLM, using compressed **MXFP4-CSF**
weights. This repo contains the run recipe and the toolkit that encodes the
model's MXFP4 expert scales into the CSF container format the vLLM
`mxfp4_csf` loader expects.

- Weights: [XiaomiMiMo/MiMo-V2.6-Pro-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD) (public, ~534 GiB, includes the `dflash/` draft model)
- Serving image: `ghcr.io/tunamitom/kk-mimo-pro` — vLLM with the MiMo MXFP4-CSF loader
- FLARE runtime (no CSF container, no vLLM): prepared package
  [mitomtuna/MiMo-V2.6-Pro-MOPD-FLARE](https://huggingface.co/mitomtuna/MiMo-V2.6-Pro-MOPD-FLARE)
  and image `ghcr.io/tunamitom/mimo-pro-flare` — see
  [FLARE runtime](#flare-runtime) below

### 1. Download the weights

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

### 2. Encode the CSF container

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

### 3. Serve on 8 GPUs

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

### 4. Test

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