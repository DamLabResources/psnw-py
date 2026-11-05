"""ms/alignment: psnw-py python vs numba on a synthetic subsidized deletion.

    python bench/bench.py [--lengths 150 300]

Synthetic: random 600 bp reference, subsidized window [300, 320), read = flank +
30 bp deletion + flank, target = anchor span, pse 5 on window bases.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import psnw_py
from psnw_py import align, msa


def case(read_len: int, seed: int = 1):
    rng = random.Random(seed)
    ref = "".join(rng.choice("ACGT") for _ in range(600 + read_len))
    ws, we = 300, 320
    half = read_len // 2
    left = ws - half + 10
    read = ref[left : ws + 10] + ref[ws + 40 : ws + 40 + (read_len - half)]
    target = ref[left : ws + 40 + (read_len - half)]
    pse = [5 if ws <= left + i < we else 0 for i in range(len(target))]
    return ref, target, read, pse


def timeit(fn, n):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[150, 300])
    args = ap.parse_args()

    t0 = time.perf_counter()
    align("ACGT", "ACT", backend="numba" if psnw_py.HAVE_NUMBA else "python")
    first = time.perf_counter() - t0

    print(f"numba available: {psnw_py.HAVE_NUMBA}; first numba call in this process: {first:.3f} s (cache=True)")
    print(f"{'read':>5} {'cells':>7} {'method':<28} {'ms/aln':>9} {'aln/s':>9}")
    for L in args.lengths:
        ref, target, read, pse = case(L)
        cells = len(target) * len(read)
        rows = [
            ("psnw-py python", timeit(lambda: align(target, read, pse=pse, backend="python"), 10)),
        ]
        if psnw_py.HAVE_NUMBA:
            rows.append(("psnw-py numba", timeit(lambda: align(target, read, pse=pse, backend="numba"), 3000)))
            rows.append(
                (
                    "psnw-py numba + shifts",
                    timeit(lambda: align(target, read, pse=pse, backend="numba").placements(), 3000),
                )
            )
        for label, dt in rows:
            print(f"{L:>5} {cells:>7} {label:<28} {dt * 1e3:9.3f} {1 / dt:9.0f}")

    if psnw_py.HAVE_NUMBA:
        rng = random.Random(7)
        ref, _, _, _ = case(150)
        ref = ref[150:450]
        reads = []
        for k in range(50):
            s, e = rng.randint(0, 40), rng.randint(260, 300)
            reads.append(
                (
                    f"r{k}",
                    ref[s:150] + ("GATTACA" if k % 2 else "") + ref[150 + (0 if k % 3 else 12) : e],
                )
            )
        dt = timeit(lambda: msa.profile(ref, reads), 3)
        print(
            f"profile MSA, 50 reads x ~250 bp vs 300 bp ref (numba): "
            f"{dt * 1e3:.1f} ms total, {dt * 1e3 / 50:.3f} ms/read"
        )


if __name__ == "__main__":
    sys.exit(main())
