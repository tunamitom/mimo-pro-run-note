"""MXFP4-CSF disk codec (row-base-offset1-u24-exceptions/1) — private spike.

Format derived from the karmic image's
`vllm/model_executor/model_loader/mxfp4_csf_loader.py` (Apache-2.0, vLLM
project) and validated against its `_slice_scale_plane()` in tests.

Disk format (per scale plane, uint8 values, rows % 16 == 0):
- fixed stream: rows/16 slabs; each slab = 16 base bytes (one per row, 0..254)
  followed by 16 * ceil(columns/8) selector bytes (per row, little bitorder).
  Value prediction: base + selector bit (0 or 1).
- exceptions: uint32 words, sorted unique positions in row-major order,
  word = position | (original_byte << 24), position = row * columns + col.

The encoder is deterministic: per row, base = lowest b in 0..254 maximizing
count(v == b) + count(v == b + 1). Values outside {base, base+1} become
exceptions.
"""

import numpy as np

POSITION_MASK = (1 << 24) - 1


def selector_bytes(columns: int) -> int:
    return (columns + 7) // 8


def encode_plane(plane: np.ndarray):
    """Encode a uint8 [rows, columns] scale plane -> (fixed u8, exceptions u32)."""
    if plane.dtype != np.uint8 or plane.ndim != 2:
        raise ValueError("MXFP4-CSF plane must be a 2-D uint8 array")
    rows, columns = plane.shape
    if rows % 16:
        raise ValueError("MXFP4-CSF planes must have a multiple of 16 rows")
    sel = selector_bytes(columns)
    fixed = np.zeros((rows // 16, 16 * (1 + sel)), dtype=np.uint8)
    positions, values = [], []
    for s in range(rows // 16):
        block = plane[s * 16 : (s + 1) * 16]
        for r in range(16):
            row = block[r]
            hist = np.bincount(row, minlength=256)
            base = int(np.argmax(hist[:255] + hist[1:256]))  # lowest b on tie
            fixed[s, r] = base
            bit = row == base + 1
            fixed[s, 16 + r * sel : 16 + (r + 1) * sel] = np.packbits(
                bit, bitorder="little"
            )
            miss = np.nonzero((row != base) & ~bit)[0]
            positions.extend((s * 16 + r) * columns + int(c) for c in miss)
            values.extend(int(row[c]) for c in miss)
    words = np.array(
        [p | (v << 24) for p, v in zip(positions, values)], dtype=np.uint32
    )
    return fixed.reshape(-1), words


def decode_plane(fixed, exceptions, rows: int, columns: int) -> np.ndarray:
    """Independent decoder: (fixed u8, exceptions u32) -> uint8 [rows, columns]."""
    sel = selector_bytes(columns)
    stream = np.asarray(fixed, dtype=np.uint8).reshape(rows // 16, 16 * (1 + sel))
    bases = stream[:, :16].astype(np.int32)
    if (bases > 254).any():
        raise ValueError("MXFP4-CSF palette bases must be in 0..254")
    bits = np.unpackbits(
        stream[:, 16:].reshape(rows, sel), axis=1, bitorder="little"
    )
    if bits[:, columns:].any():
        raise ValueError("MXFP4-CSF unused selector bits must be zero")
    out = (bases.reshape(-1, 1) + bits[:, :columns]).astype(np.uint8)
    words = np.asarray(exceptions, dtype=np.uint32).reshape(-1)
    if len(words):
        positions = words & POSITION_MASK
        if positions[-1] >= rows * columns or (positions[1:] <= positions[:-1]).any():
            raise ValueError(
                "MXFP4-CSF exception positions must be unique, sorted and in range"
            )
        out.reshape(-1)[positions] = (
            words >> np.uint32(24)
        ).astype(np.uint8)
    return out