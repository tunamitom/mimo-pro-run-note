"""Build + verify MiMo-V2.6-Pro-MOPD-CSF container (MXFP4-CSF, private spike).

Replicates the lil-mxfp4-csf-checkpoint/1 grammar of the LIL GLM container
(manifest/build-contract/receipts/verification), per /opt/mimo-pro/
plan-csf-encoder-mimo.md sections 12-14.

build:  stream each source shard once; byte-copy non-scale tensors verbatim;
        replace whitelisted expert weight_scale tensors with
        `<name>.mxfp4_csf_fixed` (U8) + `<name>.mxfp4_csf_exceptions` (U32)
        (the runtime CsfTensorReader suffix; the full codec id stays in the
        contract/manifest, which read_csf_contract checks);
        atomic per-shard publish (.part + rename); content-addressed receipts
        (source sha256 vs download-manifest lfs_oid) with the raw source
        header preserved for exact rebuild.
verify: per shard (fail-closed — any mismatch raises): receipt source
        SHA == download-manifest LFS oid; reassembled original bytes must
        hash to the recorded source hash; recomputed compressed-file SHA-256
        must equal the receipt target hash. Writes verification.json with
        counts derived from completed checks.

Memory-bounded: one tensor in flight per worker. Run inside the karmic image.
"""

import base64
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "/toolkit")
from mxfp4_csf import decode_plane, encode_plane  # noqa: E402

SRC = Path("/model")
DEST = Path("/out-container")
SRC_MANIFEST = Path("/source-manifest.json")
SCHEMA = "lil-mxfp4-csf-checkpoint/1"
CODEC = "row-base-offset1-u24-exceptions/1"
TENSOR_SUFFIX = "mxfp4"  # vllm csf_utils.CsfTensorReader(root, names, "mxfp4")
WRITER_VERSION = 2
FAMILY = "mimo_v26_pro_mxfp4"
SOURCE_MODEL_ID = "XiaomiMiMo/MiMo-V2.6-Pro-MOPD"
SOURCE_REVISION = "adea8e2c5373181e5a973fa1ecb343cb31af214b"
SCALE_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight_scale$"
)
DTYPE_ITEMSIZE = {"U8": 1, "U32": 4, "F8_E4M3": 1, "F32": 4, "BF16": 2, "I8": 1}


def load_index():
    index = json.loads((SRC / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    shards = {}
    for name, file in wm.items():
        shards.setdefault(file, []).append(name)
    for names in shards.values():
        names.sort()
    scales = {n for n in wm if SCALE_RE.match(n)}
    assert len(scales) == 79488, f"scale whitelist mismatch: {len(scales)}"
    return wm, shards, scales


def load_source_manifest():
    if not SRC_MANIFEST.exists():
        return {}
    raw = json.loads(SRC_MANIFEST.read_text())
    return {k: v.get("lfs_oid") for k, v in raw.items() if isinstance(v, dict)}


def build_shard(args):
    file, names, manifest_oid = args
    src_path = SRC / file
    dst_path = DEST / "tensors" / file
    receipt_path = DEST / "receipts" / (file + ".json")
    if receipt_path.exists() and dst_path.exists():
        r = json.loads(receipt_path.read_text())
        if (
            r.get("writer_version") == WRITER_VERSION
            and r.get("source_content_address_verified") is True
            and r.get("target_file_bytes") == dst_path.stat().st_size
        ):
            h = hashlib.sha256()
            with open(dst_path, "rb") as vf:
                for chunk in iter(lambda: vf.read(1 << 26), b""):
                    h.update(chunk)
            if h.hexdigest() == r.get("target_sha256"):
                return f"skip {file} (already built; target sha re-verified)"
        # fall through: stale/incompatible/unverifiable receipt -> rebuild
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".part")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(src_path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header_raw = f.read(hlen)
        header = json.loads(header_raw)
        entries = [
            (name, meta) for name, meta in header.items() if name != "__metadata__"
        ]
        entries.sort(key=lambda kv: kv[1]["data_offsets"][0])
        cursor = 0
        for name, meta in entries:
            a, b = meta["data_offsets"]
            assert a == cursor, f"{file}:{name} offset gap"
            cursor = b
        data_bytes = cursor

        # PASS 1: encode whitelisted scales only (buffered compressed bytes
        # are tiny vs shard size; non-scale payloads are never materialized).
        encoded = {}
        for name, meta in entries:
            m = SCALE_RE.match(name)
            if m and name in names:
                nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
                f.seek(8 + hlen + meta["data_offsets"][0])
                buf = f.read(nbytes)
                if len(buf) != nbytes:
                    raise EOFError(f"{file}:{name} truncated scale read")
                plane = np.frombuffer(buf, dtype=np.uint8).reshape(meta["shape"])
                fixed, exc = encode_plane(plane)
                encoded[name] = (fixed.tobytes(), exc.tobytes(), {
                    "dtype": meta["dtype"],
                    "shape": list(meta["shape"]),
                    "source_bytes": nbytes,
                    "source_sha256": hashlib.sha256(buf).hexdigest(),
                    "fixed_bytes": int(fixed.nbytes),
                    "exception_bytes": int(exc.nbytes),
                })

        # destination header: scale entries replaced in place
        dst_entries = {}
        dst_order = []
        offset = 0
        for name, meta in entries:
            if name in encoded:
                fx, ex, _ = encoded[name]
                for sub, raw, dt, size in (
                    ("fixed", fx, "U8", len(fx)),
                    ("exceptions", ex, "U32", len(ex) // 4),
                ):
                    tname = f"{name}.{TENSOR_SUFFIX}_csf_{sub}"
                    dst_entries[tname] = {
                        "dtype": dt,
                        "shape": [size],
                        "data_offsets": [offset, offset + len(raw)],
                    }
                    dst_order.append(tname)
                    offset += len(raw)
            else:
                nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
                dst_entries[name] = {
                    "dtype": meta["dtype"],
                    "shape": meta["shape"],
                    "data_offsets": [offset, offset + nbytes],
                }
                dst_order.append(name)
                offset += nbytes

        # PASS 2: sequential source read for hashing + streamed target write.
        dst_header = {n: dst_entries[n] for n in dst_order}
        hj = json.dumps(dst_header, separators=(",", ":")).encode()
        hj += b" " * ((-len(hj)) % 8)
        src_hash = hashlib.sha256()
        tgt_hash = hashlib.sha256()
        head = struct.pack("<Q", hlen) + header_raw
        src_hash.update(head)
        head_dst = struct.pack("<Q", len(hj)) + hj
        tgt_hash.update(head_dst)
        f.seek(0)
        f.read(8 + hlen)  # reposition cursor at data start (already hashed)
        with open(tmp_path, "wb") as out:
            out.write(head_dst)
            for name, meta in entries:
                nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
                if name in encoded:
                    fx, ex, _ = encoded[name]
                    remaining = nbytes
                    while remaining:
                        chunk = f.read(min(1 << 26, remaining))
                        if not chunk:
                            raise EOFError(f"{file}:{name} truncated source read")
                        src_hash.update(chunk)
                        remaining -= len(chunk)
                    out.write(fx)
                    tgt_hash.update(fx)
                    out.write(ex)
                    tgt_hash.update(ex)
                else:
                    remaining = nbytes
                    while remaining:
                        chunk = f.read(min(1 << 26, remaining))
                        if not chunk:
                            raise EOFError(f"{file}:{name} truncated source read")
                        src_hash.update(chunk)
                        tgt_hash.update(chunk)
                        out.write(chunk)
                        remaining -= len(chunk)
            out.flush()
            os.fsync(out.fileno())
    source_sha = src_hash.hexdigest()
    target_sha = tgt_hash.hexdigest()
    os.rename(tmp_path, dst_path)
    scale_receipts = {n: encoded[n][2] for n in encoded}

    if manifest_oid is None:
        raise ValueError(
            f"{file}: no LFS oid for this shard in the source download "
            "manifest - refusing to build without content-addressed "
            "provenance (fail-closed)"
        )
    if source_sha != manifest_oid:
        raise ValueError(
            f"{file}: source SHA-256 {source_sha} != download-manifest "
            f"LFS oid {manifest_oid} - provenance mismatch (fail-closed)"
        )
    verified = True
    receipt = {
        "file": file,
        "scales": scale_receipts,
        "source_content_address_verified": bool(verified),
        "source_file_bytes": 8 + hlen + data_bytes,
        "source_header_base64": base64.b64encode(header_raw).decode(),
        "source_sha256": source_sha,
        "target_file_bytes": dst_path.stat().st_size,
        "target_sha256": target_sha,
        "tensor_suffix": TENSOR_SUFFIX,
        "writer_version": WRITER_VERSION,
    }
    tmp_receipt = receipt_path.with_suffix(".json.part")
    tmp_receipt.write_text(json.dumps(receipt, indent=1))
    os.rename(tmp_receipt, receipt_path)
    return f"built {file}: {len(scale_receipts)} scales"


def verify_shard(args):
    file, manifest_oid = args
    receipt = json.loads((DEST / "receipts" / (file + ".json")).read_text())
    if receipt.get("writer_version") != WRITER_VERSION:
        raise ValueError(f"{file}: receipt writer_version != {WRITER_VERSION}")
    if receipt.get("source_content_address_verified") is not True:
        raise ValueError(f"{file}: receipt provenance not verified - refusing")
    if manifest_oid is None:
        raise ValueError(f"{file}: missing download-manifest LFS oid at verify")
    if receipt["source_sha256"] != manifest_oid:
        raise ValueError(f"{file}: receipt source SHA != manifest LFS oid")
    header = json.loads(base64.b64decode(receipt["source_header_base64"]))
    entries = [
        (name, meta) for name, meta in header.items() if name != "__metadata__"
    ]
    entries.sort(key=lambda kv: kv[1]["data_offsets"][0])
    h = hashlib.sha256()
    h.update(struct.pack("<Q", len(base64.b64decode(receipt["source_header_base64"]))))
    h.update(base64.b64decode(receipt["source_header_base64"]))
    with open(DEST / "tensors" / file, "rb") as f:
        (dlen,) = struct.unpack("<Q", f.read(8))
        dheader = json.loads(f.read(dlen))
        for name, meta in entries:
            nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
            m = SCALE_RE.match(name)
            if not m or name not in receipt["scales"]:
                d = dheader[name]
                f.seek(8 + dlen + d["data_offsets"][0])
                buf = f.read(nbytes)
                if len(buf) != nbytes:
                    raise EOFError(f"{file}:{name} truncated tensor read")
                h.update(buf)
            else:
                rows, cols = receipt["scales"][name]["shape"]
                fd = dheader[f"{name}.{TENSOR_SUFFIX}_csf_fixed"]
                ed = dheader[f"{name}.{TENSOR_SUFFIX}_csf_exceptions"]
                f.seek(8 + dlen + fd["data_offsets"][0])
                fbuf = f.read(fd["data_offsets"][1] - fd["data_offsets"][0])
                if len(fbuf) != fd["data_offsets"][1] - fd["data_offsets"][0]:
                    raise EOFError(f"{file}:{name} truncated fixed read")
                fixed = np.frombuffer(fbuf, dtype=np.uint8)
                f.seek(8 + dlen + ed["data_offsets"][0])
                ebuf = f.read(ed["data_offsets"][1] - ed["data_offsets"][0])
                if len(ebuf) != ed["data_offsets"][1] - ed["data_offsets"][0]:
                    raise EOFError(f"{file}:{name} truncated exceptions read")
                exc = np.frombuffer(ebuf, dtype=np.uint32)
                plane = decode_plane(fixed, exc, rows, cols)
                h.update(plane.tobytes())
    restored = h.hexdigest()
    if restored != receipt["source_sha256"]:
        raise ValueError(
            f"{file}: reconstructed source SHA {restored} != receipt "
            f"{receipt['source_sha256']} - container cannot rebuild source"
        )
    t = hashlib.sha256()
    with open(DEST / "tensors" / file, "rb") as tf:
        for chunk in iter(lambda: tf.read(1 << 26), b""):
            t.update(chunk)
    if t.hexdigest() != receipt["target_sha256"]:
        raise ValueError(
            f"{file}: recomputed target SHA {t.hexdigest()} != receipt "
            f"{receipt['target_sha256']} - compressed file corrupted"
        )
    return {
        "file": file,
        "passed": True,
        "restored_source_sha256": restored,
        "target_sha256": t.hexdigest(),
        "scale_count": len(receipt["scales"]),
    }


def copy_metadata(shard_files):
    meta_sha = {}
    for path in sorted(SRC.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(SRC)
        if rel.parent == Path(".") and rel.name in shard_files:
            continue
        dst = DEST / "metadata" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)
        meta_sha[str(rel)] = hashlib.sha256(dst.read_bytes()).hexdigest()
    return meta_sha


def main():
    mode = sys.argv[1]
    wm, shards, scale_names = load_index()
    manifest = load_source_manifest()
    if mode == "build":
        if not manifest:
            raise ValueError(
                "source download manifest missing or empty - build requires "
                "content-addressed provenance for every shard (fail-closed)"
            )
        (DEST / "tensors").mkdir(parents=True, exist_ok=True)
        (DEST / "receipts").mkdir(exist_ok=True)
        tasks = [
            (file, set(names), manifest.get(file))
            for file, names in sorted(shards.items())
        ]
        with ProcessPoolExecutor(max_workers=6) as ex:
            for i, msg in enumerate(ex.map(build_shard, tasks), 1):
                print(f"[{i}/{len(tasks)}] {msg}", flush=True)
        print("BUILD PHASE DONE", flush=True)

        # metadata + contracts
        index_sha = hashlib.sha256(
            (SRC / "model.safetensors.index.json").read_bytes()
        ).hexdigest()
        meta_sha = copy_metadata(set(shards))
        source_payload = sum(
            (SRC / f).stat().st_size for f in shards
        )
        orig_scale_bytes = sum(v["source_bytes"] for f in shards
                               for v in (json.loads((DEST / "receipts" / (f + ".json")).read_text())["scales"]).values())
        fixed_bytes = sum(v["fixed_bytes"] for f in shards
                          for v in (json.loads((DEST / "receipts" / (f + ".json")).read_text())["scales"]).values())
        exc_bytes = sum(v["exception_bytes"] for f in shards
                        for v in (json.loads((DEST / "receipts" / (f + ".json")).read_text())["scales"]).values())
        shard_entries = []
        for f in sorted(shards):
            r = json.loads((DEST / "receipts" / (f + ".json")).read_text())
            shard_entries.append({
                "file": f,
                "source_file_bytes": r["source_file_bytes"],
                "source_sha256": r["source_sha256"],
                "target_file_bytes": r["target_file_bytes"],
                "target_sha256": r["target_sha256"],
            })
        contract = {
            "codec": CODEC,
            "compressed_scale_count": len(scale_names),
            "family": FAMILY,
            "metadata_sha256": meta_sha,
            "schema": SCHEMA,
            "source": str(SRC),
            "source_index_sha256": index_sha,
            "source_index_size_semantics": "safetensors_files",
            "source_names": wm,
            "source_payload_bytes": source_payload,
            "source_revision": SOURCE_REVISION,
            "tensor_suffix": TENSOR_SUFFIX,
            "writer_version": WRITER_VERSION,
        }
        manifest_out = dict(contract)
        manifest_out.pop("source_names")
        manifest_out.update({
            "compressed_scale_bytes": fixed_bytes + exc_bytes,
            "compressed_weight_file_bytes": sum(s["target_file_bytes"] for s in shard_entries),
            "original_scale_bytes": orig_scale_bytes,
            "saved_weight_file_bytes": sum(s["source_file_bytes"] for s in shard_entries) - sum(s["target_file_bytes"] for s in shard_entries),
            "shards": shard_entries,
            "source_model_id": SOURCE_MODEL_ID,
            "source_weight_file_bytes": sum(s["source_file_bytes"] for s in shard_entries),
            "status": "implemented; complete source reconstruction verification required",
        })
        (DEST / "build-contract.json").write_text(json.dumps(contract, indent=1))
        (DEST / "manifest.json").write_text(json.dumps(manifest_out, indent=1))
        print("CONTRACTS WRITTEN", flush=True)

    elif mode == "verify":
        results = []
        tasks = [(f, manifest.get(f)) for f in sorted(shards)]
        missing = [f for f, oid in tasks if oid is None]
        if missing:
            raise ValueError(f"missing download-manifest oids for: {missing[:5]}")
        with ProcessPoolExecutor(max_workers=4) as ex:
            for i, res in enumerate(ex.map(verify_shard, tasks), 1):
                results.append(res)
                print(f"[{i}/{len(shards)}] {res['file']} {'OK' if res['passed'] else 'FAIL'}", flush=True)
        passed = all(r["passed"] for r in results)
        verification = {
            "conditions": (
                "Per shard, fail-closed (any mismatch raises and aborts): "
                "receipt source SHA-256 == download-manifest LFS oid; the "
                "container reassembles the original shard bytes and their "
                "SHA-256 must equal the receipt source hash; the recomputed "
                "compressed-file SHA-256 must equal the receipt target "
                "hash. Counts below are derived from completed checks, not "
                "hardcoded."
            ),
            "passed": passed,
            "qualification": "storage and CPU reconstruction; native serving is not implied",
            "scale_matrices": sum(r["scale_count"] for r in results),
            "schema": SCHEMA,
            "shards": results,
            "shards_checked": len(results),
            "tensor_suffix": TENSOR_SUFFIX,
            "writer_version": WRITER_VERSION,
        }
        (DEST / "verification.json").write_text(json.dumps(verification, indent=1))
        print(f"VERIFY {'PASSED' if passed else 'FAILED'}", flush=True)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()