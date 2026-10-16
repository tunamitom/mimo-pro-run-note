"""Spike tests: MXFP4-CSF encoder/decoder round-trip + upstream-loader oracle.

The oracle is the karmic image's `_slice_scale_plane` extracted verbatim from
mxfp4_csf_loader.py (AST-extracted, so the file's vLLM imports never run).
"""

import ast
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from mxfp4_csf import decode_plane, encode_plane  # noqa: E402

LOADER = Path(
    "/home/tom/.hermes/cache/scratch/csf-study/mxfp4_csf_loader.py"
)


def load_oracle():
    tree = ast.parse(LOADER.read_text())
    keep = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name == "_slice_scale_plane")
        or (isinstance(n, ast.Assign) and n.targets[0].id == "POSITION_MASK")
    ]
    ns = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<oracle>", "exec"), ns)
    return ns["_slice_scale_plane"]


oracle = load_oracle()


def roundtrip(plane):
    fixed, exc = encode_plane(plane)
    out = decode_plane(fixed, exc, *plane.shape)
    assert (out == plane).all(), "round-trip mismatch"
    return fixed, exc


def test_shapes_and_edges():
    rng = np.random.default_rng(42)
    cases = [
        np.zeros((16, 192), np.uint8),  # all base 0
        np.full((16, 64), 255, np.uint8),  # all base 254 + bit
        np.full((32, 192), 127, np.uint8),  # single value interior
        rng.integers(0, 2, (48, 192), dtype=np.uint8),  # {0,1} -> base 0 or 1
        rng.integers(0, 256, (32, 64), dtype=np.uint8),  # incompressible
        rng.integers(0, 256, (16, 17), dtype=np.uint8),  # odd column count
    ]
    # realistic full-size planes
    for shape in ((2048, 192), (6144, 64)):
        p = rng.integers(60, 70, shape, dtype=np.uint8)  # E8M0-ish exponents
        flip = rng.random(shape) < 0.02
        p[flip] = rng.integers(0, 256, int(flip.sum()), dtype=np.uint8)
        cases.append(p)
    for p in cases:
        roundtrip(p)
    print(f"ok: round-trip on {len(cases)} planes incl. full-size")


def test_oracle_full_slice():
    rng = np.random.default_rng(7)
    for shape in ((32, 192), (2048, 192), (6144, 64)):
        p = rng.integers(55, 80, shape, dtype=np.uint8)
        fixed, exc = encode_plane(p)
        rows, cols = shape
        f2, e2 = oracle(
            torch.from_numpy(fixed.reshape(-1)),
            torch.from_numpy(exc),
            rows,
            cols,
            (0, rows),
            (0, cols),
        )
        assert np.array_equal(f2.numpy().reshape(-1), fixed), "oracle fixed mismatch"
        assert np.array_equal(e2.numpy(), exc), "oracle exceptions mismatch"
        out = decode_plane(f2.numpy().reshape(-1), e2.numpy(), rows, cols)
        assert (out == p).all(), "oracle-path reconstruction mismatch"
    print("ok: upstream _slice_scale_plane accepts our streams (full slice)")


def test_oracle_tp_slices():
    rng = np.random.default_rng(11)
    p = rng.integers(50, 90, (2048, 192), dtype=np.uint8)
    p[rng.random(p.shape) < 0.05] = rng.integers(0, 256, 1, dtype=np.uint8)
    fixed, exc = encode_plane(p)
    for (r0, r1, c0, c1) in ((0, 512, 0, 192), (0, 2048, 0, 48), (1024, 2048, 96, 192)):
        f2, e2 = oracle(
            torch.from_numpy(fixed.reshape(-1)),
            torch.from_numpy(exc),
            2048,
            192,
            (r0, r1),
            (c0, c1),
        )
        sub = decode_plane(
            f2.numpy().reshape(-1), e2.numpy(), r1 - r0, c1 - c0
        )
        assert (sub == p[r0:r1, c0:c1]).all(), f"slice {(r0, r1, c0, c1)} mismatch"
    print("ok: TP row/column slices reconstruct exactly via upstream oracle")


def test_malformed_rejected():
    fixed, exc = encode_plane(np.zeros((16, 64), np.uint8))
    bad = fixed.copy()
    bad[0] = 255  # base out of range
    try:
        decode_plane(bad, exc, 16, 64)
        raise AssertionError("base 255 accepted")
    except ValueError:
        pass
    bad = fixed.copy()
    bad[16 + 3] = 0x02  # unused selector bit (columns=64 -> bits 64..63? cols=64 uses all bits of 8 bytes) -> use 17 cols
    fixed17, exc17 = encode_plane(np.zeros((16, 17), np.uint8))
    bad17 = fixed17.copy()
    bad17[16 + 2] = 0x02  # bit 17 set (columns=17, valid bits 0..16)
    try:
        decode_plane(bad17, exc17, 16, 17)
        raise AssertionError("unused selector bit accepted")
    except ValueError:
        pass
    words = np.array([100 | (1 << 24), 50 | (2 << 24)], dtype=np.uint32)  # unsorted
    try:
        decode_plane(fixed, words, 16, 64)
        raise AssertionError("unsorted positions accepted")
    except ValueError:
        pass
    print("ok: malformed streams rejected")


if __name__ == "__main__":
    test_shapes_and_edges()
    test_oracle_full_slice()
    test_oracle_tp_slices()
    test_malformed_rejected()
    print("ALL TESTS PASSED")