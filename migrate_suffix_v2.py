#!/usr/bin/env python3
"""Header-only suffix migration for the MiMo CSF container (writer v1 -> v2).

Review P0.4 (2026-10-05): v1 wrote tensor keys
`<name>.row-base-offset1-u24-exceptions/1_csf_{fixed,exceptions}`, but the
runtime reader (vllm model_executor/model_loader/csf_utils.CsfTensorReader,
instantiated with codec="mxfp4") looks up
`<name>.mxfp4_csf_{fixed,exceptions}`.

Migration strategy: rewrite ONLY the safetensors header of each shard.
Keys are renamed and the new JSON is space-padded to the ORIGINAL header
length, so every file keeps its exact byte size and every payload
byte/offset is untouched. Original headers are backed up under
headers-v1-backup/ before any write. Receipts get the new whole-file
SHA-256 (header bytes changed) plus writer_version=2; manifest.json shard
aggregates are regenerated from receipts after all shards migrate.

Host-side, stdlib only. Default is a dry run; --apply writes.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

ROOT = Path("/opt/models/MiMo-V2.6-Pro-MOPD-CSF")
TENSORS = ROOT / "tensors"
RECEIPTS = ROOT / "receipts"
BACKUP = ROOT / "headers-v1-backup"
OLD = ".row-base-offset1-u24-exceptions/1_csf_"
NEW = ".mxfp4_csf_"
WRITER_VERSION = 2


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 26), b""):
            h.update(chunk)
    return h.hexdigest()


def migrate(file, apply):
    rp = RECEIPTS / (file + ".json")
    receipt = json.loads(rp.read_text())
    if receipt.get("writer_version") == WRITER_VERSION and receipt.get("tensor_suffix") == "mxfp4":
        return "skip (already v2)"
    if receipt.get("source_content_address_verified") is not True:
        raise RuntimeError(f"{file}: provenance not verified - refusing")
    tp = TENSORS / file
    size_before = tp.stat().st_size
    if size_before != receipt["target_file_bytes"]:
        raise RuntimeError(f"{file}: size {size_before} != receipt target_file_bytes")
    with open(tp, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header_raw = f.read(hlen)
    header = json.loads(header_raw)
    n_old = sum(1 for k in header if OLD in k)
    if n_old == 0:
        if receipt.get("scales") == {}:
            # scale-free shard (mtp / dense-only): nothing to rename; stamp v2
            if apply:
                receipt["writer_version"] = WRITER_VERSION
                receipt["tensor_suffix"] = "mxfp4"
                receipt["suffix_migrated"] = {
                    "from_suffix": None,
                    "to_suffix": NEW,
                    "header_bytes_preserved": True,
                    "payload_untouched": True,
                    "note": "no compressed scales in this shard; header untouched",
                    "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                tmp = rp.with_suffix(".json.part")
                tmp.write_text(json.dumps(receipt, indent=1))
                os.replace(tmp, rp)
            return "no scales - stamped v2, header untouched"
        raise RuntimeError(f"{file}: no v1 suffix keys but receipt has scales - unexpected state")
    new_header = {}
    for k, v in header.items():
        if OLD in k:
            nk = k.replace(OLD, NEW)
            if nk in new_header:
                raise RuntimeError(f"{file}: renamed key collision {nk}")
            new_header[nk] = v
        else:
            if k in new_header:
                raise RuntimeError(f"{file}: key collision {k}")
            new_header[k] = v
    if len(new_header) != len(header):
        raise RuntimeError(f"{file}: header entry count changed")
    offs_old = sorted(tuple(m["data_offsets"]) for m in header.values())
    offs_new = sorted(tuple(m["data_offsets"]) for m in new_header.values())
    if offs_old != offs_new:
        raise RuntimeError(f"{file}: data offsets changed - aborting")
    new_json = json.dumps(new_header, separators=(",", ":")).encode()
    if len(new_json) > hlen:
        raise RuntimeError(f"{file}: new header longer than original - cannot pad")
    padded = new_json + b" " * (hlen - len(new_json))
    if not apply:
        return f"dry-run: {n_old} keys, header {hlen}B -> json {len(new_json)}B + {hlen - len(new_json)}B pad"
    BACKUP.mkdir(exist_ok=True)
    bak = BACKUP / (file + ".hdr")
    if not bak.exists():
        with open(bak, "wb") as bf:
            bf.write(struct.pack("<Q", hlen) + header_raw)
            bf.flush()
            os.fsync(bf.fileno())
    # write new header in place (size preserved)
    with open(tp, "r+b") as f:
        f.write(struct.pack("<Q", hlen))
        f.write(padded)
        f.flush()
        os.fsync(f.fileno())
    # validate: re-read, parse, size unchanged, key counts
    with open(tp, "rb") as f:
        (hlen2,) = struct.unpack("<Q", f.read(8))
        header2 = json.loads(f.read(hlen2))
    if hlen2 != hlen:
        raise RuntimeError(f"{file}: header length changed after write")
    if tp.stat().st_size != size_before:
        raise RuntimeError(f"{file}: file size changed after write")
    if sum(1 for k in header2 if NEW in k) != n_old:
        raise RuntimeError(f"{file}: renamed key count mismatch after write")
    if any(OLD in k for k in header2):
        raise RuntimeError(f"{file}: v1 keys remain after write")
    new_sha = sha256_file(tp)
    receipt["target_sha256"] = new_sha
    receipt["writer_version"] = WRITER_VERSION
    receipt["tensor_suffix"] = "mxfp4"
    receipt["suffix_migrated"] = {
        "from_suffix": OLD,
        "to_suffix": NEW,
        "header_bytes_preserved": True,
        "payload_untouched": True,
        "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    tmp = rp.with_suffix(".json.part")
    tmp.write_text(json.dumps(receipt, indent=1))
    os.replace(tmp, rp)
    return f"migrated {file}: {n_old} keys renamed, size unchanged, target sha {new_sha[:16]}..."


def regen_manifest():
    mp = ROOT / "manifest.json"
    m = json.loads(mp.read_text())
    shards = []
    for s in sorted(m["shards"], key=lambda x: x["file"]):
        r = json.loads((RECEIPTS / (s["file"] + ".json")).read_text())
        if r.get("writer_version") != WRITER_VERSION:
            raise RuntimeError(f"{s['file']}: receipt not v2 - manifest regen refused")
        shards.append({**s, "target_sha256": r["target_sha256"]})
    m["shards"] = shards
    m["tensor_suffix"] = "mxfp4"
    m["writer_version"] = WRITER_VERSION
    m["suffix_migration"] = {
        "from_suffix": OLD,
        "to_suffix": NEW,
        "strategy": "header-only rewrite, payloads and file sizes untouched",
        "header_backup": "headers-v1-backup/",
    }
    tmp = mp.with_suffix(".json.part")
    tmp.write_text(json.dumps(m, indent=1))
    os.replace(tmp, mp)
    return f"manifest.json regenerated from {len(shards)} v2 receipts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--shard", default=None, help="migrate a single shard file")
    ap.add_argument("--regen-manifest", action="store_true")
    args = ap.parse_args()
    if args.regen_manifest:
        print(regen_manifest())
        return
    files = sorted(p.name for p in RECEIPTS.glob("*.safetensors.json"))
    files = [f[:-len(".json")] for f in files]
    if args.shard:
        files = [f for f in files if f == args.shard]
        if not files:
            raise SystemExit(f"no receipt for shard {args.shard}")
    t0 = time.time()
    done = skipped = 0
    for i, f in enumerate(files, 1):
        msg = migrate(f, args.apply)
        if msg.startswith("skip"):
            skipped += 1
        else:
            done += 1
        print(f"[{i}/{len(files)}] {msg}", flush=True)
    print(f"{'APPLIED' if args.apply else 'DRY-RUN'}: {done} migrated, {skipped} skipped, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
