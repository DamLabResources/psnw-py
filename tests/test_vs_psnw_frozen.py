"""Agreement with frozen lh3/psnw outputs (no binary required).

Cases were captured once from psnw under conditions where its quirks cannot
apply (``pse <= gap_extend``, ``pso <= gap_open``, subsidies zero near both
ends). Quirk rows document the upstream behaviour that psnw-py fixes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psnw_py import align

FIXTURE = Path(__file__).parent / "data" / "psnw_frozen.json"
ROWS = json.loads(FIXTURE.read_text())
AGREE = [r for r in ROWS if r["kind"] == "agreement"]
QUIRKS = {r["name"]: r for r in ROWS if r["kind"] == "quirk"}


@pytest.mark.parametrize("row", AGREE, ids=lambda r: f"A{r['match']}_O{r['gap_open']}_{r['psnw_cigar']}")
def test_primary_matches_frozen_psnw(row):
    aln = align(
        row["target"],
        row["query"],
        pso=row["pso"],
        pse=row["pse"],
        match=row["match"],
        mismatch=row["mismatch"],
        gap_open=row["gap_open"],
        gap_extend=row["gap_extend"],
    )
    assert aln.score == row["psnw_dp_score"] == row["psnw_as"]
    assert aln.cigar == row["psnw_cigar"]
    assert aln.md == row["psnw_md"]
    assert aln.nm == row["psnw_nm"]


def test_quirk_leading_pse():
    """psnw ignores pse on the boundary column; psnw-py honours it."""
    row = QUIRKS["leading_pse_ignored_by_psnw"]
    aln = align(row["target"], row["query"], pso=row["pso"], pse=row["pse"])
    assert (aln.cigar, aln.score) == (row["psnw_py_cigar"], row["psnw_py_score"])
    assert row["psnw_cigar"] != row["psnw_py_cigar"]
    assert row["psnw_as"] < row["psnw_py_score"]


def test_quirk_pse_above_gap_extend():
    """psnw DP can exceed the clamped CIGAR AS when pse > E; psnw-py clamps."""
    row = QUIRKS["pse_above_gap_extend"]
    aln = align(row["target"], row["query"], pso=row["pso"], pse=row["pse"])
    assert (aln.cigar, aln.score) == (row["psnw_py_cigar"], row["psnw_py_score"])
    assert row["psnw_dp_score"] > row["psnw_as"]
    assert aln.score == row["psnw_as"]
