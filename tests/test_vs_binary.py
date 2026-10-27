"""Exact agreement with the lh3/psnw C binary where its quirks cannot apply.

Conditions: ``pse <= gap_extend`` and ``pso <= gap_open`` everywhere, and
subsidies are zero near both ends of the target (psnw ignores them in the
boundary row/column). Under those conditions the DP score, primary CIGAR,
MD and NM must be identical, including tie-breaking in repeats.
"""

import pytest

from gen import digits, interior_subsidy, mutate, rand_seq, repeat_seq
from conftest import run_binary

import psnw_py

pytestmark = pytest.mark.binary

PARAMS = [
    dict(match=6, mismatch=18, gap_open=30, gap_extend=5),
    dict(match=2, mismatch=4, gap_open=4, gap_extend=2),
    dict(match=1, mismatch=3, gap_open=5, gap_extend=1),
]


def _opts(p):
    return ["-A", str(p["match"]), "-B", str(p["mismatch"]), "-O", str(p["gap_open"]), "-E", str(p["gap_extend"])]


def _cases(rng, p, n):
    cases = []
    flank = 15
    ge, go = p["gap_extend"], p["gap_open"]
    for k in range(n):
        family = k % 4
        if family == 0:
            # unrelated random sequences, zero subsidy: boundary gaps and ties
            t = rand_seq(rng, rng.randint(1, 60))
            q = rand_seq(rng, rng.randint(1, 60))
            pso = [0] * len(t)
            pse = [0] * len(t)
        else:
            T = rng.randint(2 * flank + 10, 180)
            t = rand_seq(rng, T) if family != 2 else repeat_seq(rng, T)
            q = mutate(rng, t, flank, T - flank, rng.randint(1, 5))
            pse = interior_subsidy(rng, T, flank, min(ge, 9))
            pso = interior_subsidy(rng, T, flank, min(go, 9))
            if family == 3:
                pso = [0] * T
        cases.append((t, q, pso, pse))
    return cases


@pytest.mark.parametrize("pi", range(len(PARAMS)))
def test_primary_matches_binary(psnw_bin, rng, pi):
    p = PARAMS[pi]
    cases = _cases(rng, p, 400)
    res = run_binary(psnw_bin, [(t, q, digits(o), digits(e)) for t, q, o, e in cases], _opts(p))
    for (t, q, pso, pse), (c_score, cgs) in zip(cases, res):
        aln = psnw_py.align(t, q, pso=pso, pse=pse, **p)
        c_cigar, c_md, c_nm, _ = cgs[0]
        assert aln.score == c_score, (t, q, pso, pse)
        assert aln.cigar == c_cigar, (t, q, pso, pse)
        assert aln.md == c_md
        assert aln.nm == c_nm


def test_cg_scores_match_binary_recompute(psnw_bin, rng):
    """psnw's per-CIGAR AS clamps exactly like the psnw-py cost model, so every
    CG line (primary and shifts) must rescore identically, quirks or not."""
    p = PARAMS[0]
    cases = []
    for _ in range(300):
        T = rng.randint(20, 150)
        t = rand_seq(rng, T)
        q = mutate(rng, t, 3, T - 3, rng.randint(1, 4))
        pse = [rng.randint(0, 9) if rng.random() < 0.3 else 0 for _ in range(T)]
        pso = [rng.randint(0, 9) if rng.random() < 0.3 else 0 for _ in range(T)]
        cases.append((t, q, pso, pse))
    res = run_binary(psnw_bin, [(t, q, digits(o), digits(e)) for t, q, o, e in cases], _opts(p))
    n_checked = 0
    for (t, q, pso, pse), (_, cgs) in zip(cases, res):
        for cig, md, nm, c_as in cgs:
            ops = psnw_py.parse_cigar(cig)
            if any(n <= 0 or n > 10**6 for _, n in ops):
                continue  # psnw left-shift can underflow the flanking M
            assert psnw_py.score_cigar(t, q, cig, pso=pso, pse=pse, **p) == c_as
            n_checked += 1
    assert n_checked >= 300


def test_binary_shifts_are_subset(psnw_bin, rng):
    """psnw shifts the first longest gap; those placements must appear in
    ``shifts(all_gaps=False, equal_score_only=False)``."""
    p = PARAMS[0]
    cases = []
    for _ in range(300):
        T = rng.randint(40, 160)
        t = repeat_seq(rng, T)
        q = mutate(rng, t, 15, T - 15, rng.randint(1, 3))
        cases.append((t, q, [0] * T, interior_subsidy(rng, T, 15, 5)))
    res = run_binary(psnw_bin, [(t, q, digits(o), digits(e)) for t, q, o, e in cases], _opts(p))
    n_shift = 0
    for (t, q, pso, pse), (_, cgs) in zip(cases, res):
        aln = psnw_py.align(t, q, pso=pso, pse=pse, **p)
        mine = {s.cigar for s in aln.shifts(equal_score_only=False, all_gaps=False)}
        prim = list(aln.cigartuples)
        gk = max(range(len(prim)), key=lambda k: (prim[k][0] != 0, prim[k][1] if prim[k][0] else -1, -k)) if prim else 0
        ok_neighbours = 0 < gk < len(prim) - 1 and prim[gk - 1][0] == 0 and prim[gk + 1][0] == 0
        for cig, _, _, _ in cgs[1:]:
            ops = psnw_py.parse_cigar(cig)
            if not ok_neighbours or any(n <= 0 or n > 10**6 for _, n in ops):
                continue
            assert cig in mine, (t, q, cig, sorted(mine))
            n_shift += 1
    assert n_shift > 50
