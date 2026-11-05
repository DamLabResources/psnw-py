"""psnw-py cost model and the psnw quirks it fixes."""

import numpy as np
import pytest

from gen import mutate, rand_seq

import psnw_py
from psnw_py import align, score_cigar

BACKENDS = ["python"] + (["numba"] if psnw_py.HAVE_NUMBA else [])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("free", [False, True])
def test_score_equals_rescored_cigar(rng, backend, free):
    """The DP score always equals the CIGAR rescored under the same model,
    including subsidies above the base cost and on boundary positions."""
    for _ in range(150 if backend == "numba" else 40):
        T = rng.randint(1, 90)
        t = rand_seq(rng, T, "ACGTN" if rng.random() < 0.2 else "ACGT")
        q = mutate(rng, t, 0, T, rng.randint(0, 4)) if rng.random() < 0.8 else rand_seq(rng, rng.randint(0, 50))
        pse = [rng.choice([0, 0, 3, 5, 9, 40]) for _ in range(T)]
        pso = [rng.choice([0, 0, 10, 30, 99]) for _ in range(T + 1)]
        aln = align(t, q, pso=pso, pse=pse, free_target_ends=free, backend=backend)
        assert aln.score == score_cigar(t, q, aln.cigartuples, target_start=aln.target_start, pso=pso, pse=pse)
        for s in aln.shifts(equal_score_only=False):
            assert s.score == aln.score_of(s.cigartuples)
            assert s.score <= aln.score


def test_leading_deletion_honours_pse():
    core = "ACGTTGCAAGGCTTACGATCGATTGCA"
    t = "CCCCCCCCCC" + core
    full = align(t, core)
    sub = align(t, core, pse=[5] * 10 + [0] * len(core))
    assert full.cigar == sub.cigar == f"10D{len(core)}M"
    assert full.score == 6 * len(core) - 30 - 5 * 10
    assert sub.score == 6 * len(core) - 30


def test_leading_insertion_honours_pso():
    core = "ACGTTGCAAGGCTTACGATCGATTGCA"
    q = "GGG" + core
    pso = [0] * len(core)
    pso[0] = 30
    aln = align(core, q, pso=pso)
    assert aln.cigar == f"3I{len(core)}M"
    assert aln.score == 6 * len(core) - 3 * 5


def test_trailing_insertion_uses_pso_tlen():
    core = "ACGTTGCAAGGCTTACGATCGATTGCA"
    q = core + "GGG"
    base = align(core, q)
    sub = align(core, q, pso=[0] * len(core) + [30])
    assert base.score == 6 * len(core) - 30 - 15
    assert sub.score == 6 * len(core) - 15


def test_subsidy_never_pays_a_bonus():
    """pse > gap_extend makes deletion extension free, never profitable."""
    a, w, b = "ACGTTGCAAGGCTTAC", "GATCGATTGCATGCAA", "TCAGGCATCGGATACC"
    t = a + w + b
    pse = [0] * len(a) + [9] * len(w) + [0] * len(b)
    aln = align(t, a + b, pse=pse)
    assert aln.cigar == f"{len(a)}M{len(w)}D{len(b)}M"
    assert aln.score == 6 * (len(a) + len(b)) - 30
    same = align(t, a + b, pse=[min(v, 5) for v in pse])
    assert same.score == aln.score


def test_subsidy_inputs():
    t = "ACGTACGTAC"
    a = align(t, "ACGTCGTAC", pse="0000500000", pso="0" * 10)
    b = align(t, "ACGTCGTAC", pse=np.array([0, 0, 0, 0, 5, 0, 0, 0, 0, 0]), pso=[0] * 11)
    assert (a.score, a.cigar) == (b.score, b.cigar)
    with pytest.raises(ValueError):
        align(t, "ACGT", pse=[0] * 9)
    with pytest.raises(ValueError):
        align(t, "ACGT", pso=[0] * 12)


def test_empty_sequences():
    assert align("", "").cigar == ""
    a = align("", "ACG")
    assert (a.cigar, a.score) == ("3I", -(30 + 15))
    d = align("ACG", "")
    assert (d.cigar, d.score) == ("3D", -(30 + 15))
    assert align("ACG", "", free_target_ends=True).score == 0


def test_md_nm():
    aln = align("ACGTACGTTTACGTACGT", "ACGAACGTACGTACGT")
    assert aln.cigar == "7M2D9M"
    assert aln.md == "3T3^TT9"
    assert aln.nm == 3
    assert aln.score == 15 * 6 - 18 - (30 + 2 * 5)
    assert [s.cigar for s in aln.shifts()] == ["8M2D8M"]


def test_free_target_ends():
    t = rand_seq(__import__("random").Random(3), 200)
    q = t[60:140]
    aln = align(t, q, free_target_ends=True)
    assert (aln.target_start, aln.target_end, aln.cigar, aln.score) == (60, 140, "80M", 480)


def test_shifts_every_gap():
    # two separate deletions, each inside a homopolymer: each can slide
    t = "ACGTCAGGT" + "AAAA" + "CGTGCATGC" + "TTTT" + "GACTGCATG"
    q = "ACGTCAGGT" + "AAA" + "CGTGCATGC" + "TTT" + "GACTGCATG"
    aln = align(t, q)
    sh = aln.shifts()
    assert {s.gap_index for s in sh} == {1, 3}
    assert all(s.score == aln.score for s in sh)
    assert len(aln.shifts(all_gaps=False)) == len([s for s in sh if s.gap_index == 1])


