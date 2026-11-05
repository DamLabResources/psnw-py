"""Error paths and rarely-hit API edges for 100% line coverage."""

from __future__ import annotations

import numpy as np
import pytest

import psnw_py
from psnw_py import align, deletions, encode, msa, score_cigar
from psnw_py import _jit, _kernel


def test_encode_variants():
    assert encode(b"ACGT").tolist() == [0, 1, 2, 3]
    assert encode(np.array(["A", "C", "G", "T"])).tolist() == [0, 1, 2, 3]
    assert encode([0, 1, 2, 3, 4]).tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="0..4"):
        encode([0, 5])
    with pytest.raises(ValueError, match="0..4"):
        encode([-1, 1])


def test_deletions_and_score_cigar_errors():
    assert deletions(10, "5M3D2M1I4M") == [(15, 3)]
    with pytest.raises(ValueError, match="query length"):
        score_cigar("ACGT", "ACG", "4M")
    with pytest.raises(ValueError, match="past the end"):
        score_cigar("ACGT", "ACGTAC", "4M2I", target_start=1)
    with pytest.raises(ValueError, match="unsupported CIGAR"):
        score_cigar("ACGT", "AC", [(4, 2)])  # soft clip


def test_alignment_accessors_and_placements():
    aln = align("ACGTACGTTTACGTACGT", "ACGAACGTACGTACGT")
    assert aln.target.startswith("ACGT")
    assert aln.query.startswith("ACGA")
    places = aln.placements()
    assert places[0] is aln
    assert places[0].score == aln.score


def test_kernel_rle_maps_skip_to_del():
    ops = np.array([0, 3, 3, 0, 1], dtype=np.uint8)
    assert _kernel.rle(ops, backend="python") == [(0, 1), (2, 2), (0, 1), (1, 1)]
    if psnw_py.HAVE_NUMBA:
        assert _kernel.rle(ops, backend="numba") == [(0, 1), (2, 2), (0, 1), (1, 1)]


def test_jit_parse_and_resolve():
    assert _jit.parse_backend_env("") == ""
    assert _jit.parse_backend_env(" Python ") == "python"
    with pytest.raises(ImportError, match="PSNW_PY_BACKEND"):
        _jit.parse_backend_env("fortran")
    with pytest.raises(ImportError, match="numba is not installed"):
        _jit.require_numba_if_requested("numba", False)
    _jit.require_numba_if_requested("python", False)
    with pytest.raises(ValueError, match="backend must"):
        _jit.resolve("fortran")
    if psnw_py.HAVE_NUMBA:
        assert _jit.resolve("numba") == "numba"
        saved = _jit.HAVE_NUMBA
        try:
            _jit.HAVE_NUMBA = False
            assert _jit.njit(lambda x: x) is None
            with pytest.raises(RuntimeError, match="numba"):
                _jit.resolve("numba")
        finally:
            _jit.HAVE_NUMBA = saved


def test_msa_edges(rng):
    ref = "ACGTACGTAC"
    with pytest.raises(KeyError):
        msa.star(ref, [("r", ref, 0, "10M")]).column_of(99)
    empty = msa.star(ref, [])
    assert empty.ncols == len(ref)  # half-open [0, len(ref)] builds len(ref) ref columns
    assert ">REF" in empty.to_fasta(width=4)
    with pytest.raises(ValueError, match="region"):
        msa.star(ref, [("r", ref, 0, "10M")], region=(-1, 5))
    # insertion column outside a read's span uses fill
    m = msa.star(
        ref,
        [("short", "ACGT", 0, "4M"), ("ins", "ACGTXXXACGT", 0, "4M3I4M")],
        fill=".",
    )
    assert m.to_fasta(width=5).count("\n") > 2
    # callable order and bad order
    m2 = msa.profile(ref, [("b", ref), ("a", ref[1:])], order=lambda r: r[0], free_ends=False)
    assert m2.order == ["a", "b"]
    with pytest.raises(ValueError, match="order"):
        msa.profile(ref, [("a", ref)], order="nope")


def test_profile_leading_deletion_sets_pos0():
    """With free_ends=False a leading D can be the first placed op."""
    ref = "AAAAACGTAAAA"
    # query matches the right half; leading target bases become a paid deletion
    m = msa.profile(ref, [("r", "CGTAAAA")], free_ends=False, order="input")
    pos0, cig = m.placements["r"]
    assert pos0 == 0
    assert any(op == 2 for op, _ in cig)


def test_expand_fill_at_edge():
    """New insertion columns next to fill get fill when fill != '-'."""
    from psnw_py.msa import _expand

    # left neighbour is fill → new insertion column becomes fill
    row = list("AC....GT")
    # new_cols inserts a column between index 1 and 2 (after 'C', before first '.')
    new_cols = [0, 1, -1, 2, 3, 4, 5, 6, 7]
    out = _expand(row, new_cols, fill=".")
    assert out[2] == "."
    # both neighbours real bases → stays gap
    row2 = list("ACGT")
    out2 = _expand(row2, [0, 1, -1, 2, 3], fill=".")
    assert out2[2] == "-"
