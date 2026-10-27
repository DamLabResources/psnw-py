"""Pure-Python and numba kernels must agree exactly."""

import pytest

from gen import interior_subsidy, mutate, rand_seq, repeat_seq

import psnw_py
from psnw_py import align, msa

pytestmark = pytest.mark.skipif(not psnw_py.HAVE_NUMBA, reason="numba not installed")


@pytest.mark.parametrize("free", [False, True])
def test_pairwise_backends_agree(rng, free):
    for k in range(60):
        T = rng.randint(0, 120)
        t = rand_seq(rng, T) if k % 2 else repeat_seq(rng, T)
        q = mutate(rng, t, 0, T, rng.randint(0, 5))
        pse = interior_subsidy(rng, T, 0, 12)
        pso = interior_subsidy(rng, T, 0, 40)
        a = align(t, q, pso=pso, pse=pse, free_target_ends=free, backend="python")
        b = align(t, q, pso=pso, pse=pse, free_target_ends=free, backend="numba")
        assert (a.score, a.cigartuples, a.target_start, a.target_end) == (
            b.score, b.cigartuples, b.target_start, b.target_end,
        )


def test_profile_backends_agree(rng):
    ref = rand_seq(rng, 120)
    ins = "GATTACA"
    reads = []
    for k in range(8):
        s = rng.randint(0, 30)
        e = rng.randint(90, 120)
        seq = ref[s:60] + (ins if k % 2 else "") + ref[60:e]
        reads.append((f"r{k}", mutate(rng, seq, 5, len(seq) - 5, 1, 3)))
    a = msa.profile(ref, reads, backend="python")
    b = msa.profile(ref, reads, backend="numba")
    assert a.rows == b.rows
    assert a.placements == b.placements
