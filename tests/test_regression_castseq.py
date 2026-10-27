"""45-read CAST-seq regression (chr9 cut window 9:136478803-136478822).

Reads the shared data set in place; nothing is copied into the repo. Skipped
when the directory is absent. Override with ``PSNW_REGRESSION_DIR``.
Expected (2026-10-08 coordinator handoff): del30 15/15 one 30 bp D at
136478785, del34 10/10 one 34 bp D at 136478782, intact/ctrl 122M kept or
skipped (parked elsewhere by Bowtie2).
"""

import os
from pathlib import Path

import pytest

from psnw_py import deletions, realign_read

pytestmark = pytest.mark.regression

DATA = Path(os.environ.get("PSNW_REGRESSION_DIR", "/mnt/local-tank/shared/cast-seq/2026-10-08-gap-open"))
SAM = DATA / "rfg_5_3.sam"
FASTA = DATA / "chr9.fa"
META = DATA / "gap_open_test.meta.tsv"
C_TSV = DATA / "psnw_realign.tsv"
CHROM, WIN_S, WIN_E = "9", 136478803, 136478822
PAD = 5000


def _fai_fetch(fasta: Path, chrom: str, start: int, end: int) -> str:
    with open(str(fasta) + ".fai") as fh:
        for line in fh:
            name, length, offset, lb, lw = line.split("\t")[:5]
            if name == chrom:
                length, offset, lb, lw = int(length), int(offset), int(lb), int(lw)
                break
        else:
            raise KeyError(chrom)
    end = min(end, length)
    b0 = offset + (start // lb) * lw + start % lb
    b1 = offset + (end // lb) * lw + end % lb
    with open(fasta, "rb") as fh:
        fh.seek(b0)
        raw = fh.read(b1 - b0)
    return raw.replace(b"\n", b"").replace(b"\r", b"").decode().upper()


def _tsv(path: Path):
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        return {r[0]: dict(zip(header, r)) for r in (l.rstrip("\n").split("\t") for l in fh if l.strip())}


@pytest.fixture(scope="module")
def results():
    if not (SAM.is_file() and FASTA.is_file() and Path(str(FASTA) + ".fai").is_file()):
        pytest.skip(f"regression data not found under {DATA}")
    off = WIN_S - PAD
    ref = _fai_fetch(FASTA, CHROM, off, WIN_E + PAD)
    meta = {k: v["class"] for k, v in _tsv(META).items()} if META.is_file() else {}
    out = {}
    with open(SAM) as fh:
        for line in fh:
            if line.startswith("@") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if int(f[1]) & 0x904 or f[2] != CHROM:
                continue
            status, placed = realign_read(f[9].upper(), ref, int(f[3]) - 1, f[5], WIN_S, WIN_E, ref_offset=off)
            out[f[0]] = (meta.get(f[0], f[0].split("_")[0]), int(f[3]) - 1, f[5], status, placed)
    return out


def test_classes(results):
    assert len(results) == 45
    by = {}
    for name, (cls, pos0, cig, status, placed) in results.items():
        by.setdefault(cls, []).append((name, pos0, cig, status, placed))
    assert {k: len(v) for k, v in by.items()} == {"del30": 15, "del34": 10, "intact3": 10, "ctrl5": 10}

    for cls, start, dl, lead in (("del30", 136478785, 30, "10M30D"), ("del34", 136478782, 34, "7M34D")):
        for name, _, _, status, placed in by[cls]:
            assert status == "realigned", name
            assert placed.pos0 == 136478775 and placed.cigar.startswith(lead), (name, placed)
            assert deletions(placed.pos0, placed.cigar) == [(start, dl)], name

    for cls in ("intact3", "ctrl5"):
        for name, pos0, cig, status, placed in by[cls]:
            assert cig == "122M"
            if status == "kept":
                assert (placed.pos0, placed.cigar) == (pos0, "122M")
            else:
                assert status == "skipped" and pos0 < WIN_S - 1000, name


def test_matches_c_binary_run(results):
    """Same status / pos0 / CIGAR / AS as the 2026-10-08 psnw binary run."""
    if not C_TSV.is_file():
        pytest.skip("psnw_realign.tsv not present")
    old = _tsv(C_TSV)
    assert set(old) == set(results)
    for name, (cls, pos0, cig, status, placed) in results.items():
        row = old[name]
        assert row["status"] == status, name
        if placed is None:
            assert int(row["pos0"]) == pos0 and row["cigar"] == cig
        else:
            assert (int(row["pos0"]), row["cigar"], int(row["as"])) == (placed.pos0, placed.cigar, placed.score), name
