"""Cut-site realignment helper on synthetic data, and parity with the
subprocess-based wrapper script when it and the C binary are available."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from gen import rand_seq

from psnw_py import deletions, realign_read

WRAPPER = Path(
    os.environ.get(
        "PSNW_WRAPPER_SCRIPT",
        "/home/jupyter-will/repos/damlab-wrappers/CRISPR/psnw-cut-realign/psnw_cut_realign.py",
    )
)


def _synthetic(rng, n_reads=30):
    ref = rand_seq(rng, 2000)
    ws, we = 1000, 1020
    reads = []
    for k in range(n_reads):
        dl = rng.choice([0, 12, 30, 34])
        ds = rng.randint(ws - 5, we - 5) if dl else ws
        left = min(ds, ws) - rng.randint(8, 20)
        right = ds + dl + rng.randint(80, 120)
        seq = ref[left:ds] + ref[ds + dl : right]
        # mapper-style bad placement: anchor downstream with a short indel
        reads.append((f"r{k}", seq, left + 3, f"{len(seq)}M", dl, left))
    return ref, ws, we, reads


def test_recovers_synthetic_deletions(rng):
    ref, ws, we, reads = _synthetic(rng)
    for name, seq, pos0, cig, dl, left in reads:
        status, placed = realign_read(seq, ref, pos0, cig, ws, we)
        assert status in ("realigned", "kept"), name
        dels = deletions(placed.pos0, placed.cigar)
        if dl:
            assert len(dels) == 1 and dels[0][1] == dl, (name, placed)
            assert dels[0][0] < we and dels[0][0] + dl > ws
        else:
            assert dels == [] and placed.pos0 == left


def test_ref_offset_slice(rng):
    ref, ws, we, reads = _synthetic(rng, 5)
    off = 700
    for name, seq, pos0, cig, dl, left in reads:
        full = realign_read(seq, ref, pos0, cig, ws, we)
        sl = realign_read(seq, ref[off:1400], pos0, cig, ws, we, ref_offset=off)
        assert full[0] == sl[0] and full[1].pos0 == sl[1].pos0 and full[1].cigar == sl[1].cigar


def test_far_read_skipped():
    ref = "A" * 3000
    assert realign_read("ACGT" * 30, ref, 10, "120M", 2000, 2020)[0] == "skipped"


@pytest.mark.binary
def test_parity_with_wrapper_script(rng, psnw_bin):
    if not WRAPPER.is_file():
        pytest.skip("damlab wrapper script not found")
    spec = importlib.util.spec_from_file_location("psnw_cut_realign_wrapper", WRAPPER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    ref, ws, we, reads = _synthetic(rng, 40)
    for name, seq, pos0, cig, dl, left in reads:
        st_c, pl_c = mod.realign_read(psnw_bin, name, seq, ref, pos0, cig, ws, we, 80, 6, 5)
        st_p, pl_p = realign_read(seq, ref, pos0, cig, ws, we)
        assert st_c == st_p
        assert (pl_c.pos0, pl_c.cigar, pl_c.score, pl_c.extra) == (pl_p.pos0, pl_p.cigar, pl_p.score, pl_p.extra)
