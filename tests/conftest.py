import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_PSNW_BIN = "/home/jupyter-will/repos/agent-lab/.cursor/skills/psnw-cut-realign/bin/psnw"


def psnw_bin_path():
    p = os.environ.get("PSNW_BIN", DEFAULT_PSNW_BIN)
    if p and os.path.isfile(p) and os.access(p, os.X_OK):
        return p
    return None


@pytest.fixture(scope="session")
def psnw_bin():
    p = psnw_bin_path()
    if p is None:
        pytest.skip("psnw binary not found (set PSNW_BIN)")
    return p


@pytest.fixture
def rng():
    return random.Random(20261027)


def run_binary(path, cases, opts=()):
    """Run psnw once over many cases.

    ``cases`` is a list of ``(target, query, pso_str, pse_str)``. Returns one
    ``(dp_score, [(cigar, md, nm, as), ...])`` per case, in order.
    """
    lines = "".join(f"c{k}\t{t}\t{q}\t{o}\t{e}\n" for k, (t, q, o, e) in enumerate(cases))
    proc = subprocess.run([path, *opts, "-"], input=lines, text=True, capture_output=True, check=True)
    out = []
    cur = None
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "SQ":
            cur = (int(parts[3].split(":")[-1]), [])
            out.append(cur)
        elif parts[0] == "CG":
            cur[1].append((parts[1], parts[2][5:], int(parts[3][5:]), int(parts[4][5:])))
    assert len(out) == len(cases)
    return out
