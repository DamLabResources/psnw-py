"""CLI smoke tests for ``psnw-py``."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from psnw_py.cli import main


def test_cli_file_and_flags(tmp_path):
    path = tmp_path / "in.tsv"
    path.write_text(
        "short\n"
        "r1\tACGTACGTTTACGTACGT\tACGAACGTACGTACGT\t000000000000000000\t000000005000000000\n"
        "r2\tAAAA\tAAA\n"
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main([str(path), "-a", "--backend", "python"]) == 0
    out = buf.getvalue()
    assert "SQ\tr1\t" in out
    assert "CG\t" in out and "MD:Z:" in out
    assert "O1\t" in out and "O2\t" in out and "O3\t" in out
    assert out.count("//") >= 2


def test_cli_stdin_all_shifts(monkeypatch):
    data = "n\tAAAA\tAA\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(data))
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(["-", "--all-shifts", "--longest-gap-only", "-A", "2", "-B", "4", "-O", "4", "-E", "2"]) == 0
    assert "SQ\tn\t" in buf.getvalue()


def test_cli_prints_insertion_alignment(tmp_path):
    path = tmp_path / "ins.tsv"
    # query longer than target forces an insertion in the printed alignment
    path.write_text("ins\tACGT\tACGGT\n")
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main([str(path), "-a", "--backend", "python"]) == 0
    out = buf.getvalue()
    assert "O1\t" in out and "-" in out.split("O1\t", 1)[1].split("\n", 1)[0]
