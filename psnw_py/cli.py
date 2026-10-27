"""``psnw-py``: psnw-compatible TSV in, SQ/CG report out.

Input lines: ``name  target  query  [pso  [pse]]`` (TAB or space separated);
``pso``/``pse`` are digit strings as in psnw. Output follows psnw (``SQ`` with
the placement count and DP score, one ``CG`` per placement with MD, NM and
AS), except that every gap is shifted and only equal-score shifts are listed
unless ``--all-shifts`` is given.
"""

from __future__ import annotations

import argparse
import sys

from .core import _md_nm, align, decode


def _report(out, name, aln, shifts, print_aln):
    places = [aln] + shifts
    out.write(f"SQ\t{name}\t{len(places)}\tAS:i:{aln.score}\n")
    for p in places:
        cig = p.cigartuples
        md, nm = _md_nm(aln._t, aln._q, cig, aln.target_start)
        out.write(f"CG\t{p.cigar}\tMD:Z:{md}\tNM:i:{nm}\tAS:i:{p.score}\n")
        if print_aln:
            o1, o2, o3 = _strings(aln, cig)
            out.write(f"O1\t{o1}\nO2\t{o2}\nO3\t{o3}\n")
    out.write("//\n")


def _strings(aln, cig):
    t, q = aln._t, aln._q
    x, y = aln.target_start, 0
    a, m, b = [], [], []
    for op, n in cig:
        for k in range(n):
            if op == 0:
                a.append(decode(t[x : x + 1]))
                b.append(decode(q[y : y + 1]))
                m.append("|" if t[x] == q[y] and t[x] < 4 else " ")
                x += 1
                y += 1
            elif op == 1:
                a.append("-")
                b.append(decode(q[y : y + 1]))
                m.append(" ")
                y += 1
            else:
                a.append(decode(t[x : x + 1]))
                b.append("-")
                m.append(" ")
                x += 1
    return "".join(a), "".join(m), "".join(b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="psnw-py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="TSV path or '-' for stdin")
    ap.add_argument("-A", type=int, default=6, help="match score [6]")
    ap.add_argument("-B", type=int, default=18, help="mismatch penalty [18]")
    ap.add_argument("-O", type=int, default=30, help="gap open penalty [30]")
    ap.add_argument("-E", type=int, default=5, help="gap extension penalty [5]")
    ap.add_argument("-a", action="store_true", help="print base alignment")
    ap.add_argument("--all-shifts", action="store_true", help="list shifts that lower the score too")
    ap.add_argument("--longest-gap-only", action="store_true", help="shift only the first longest gap (psnw)")
    ap.add_argument("--backend", choices=("python", "numba"), default=None)
    args = ap.parse_args(argv)

    fh = sys.stdin if args.input == "-" else open(args.input)
    out = sys.stdout
    out.write("CC\tCC     comment\nCC\tSQ     name    #records  AS\nCC\tCG     cigar   MD        NM  AS\n")
    out.write("CC\tO[123] alignment-strings\nCC\t//     end-of-report\nCC\n")
    try:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            name, target, query = parts[:3]
            pso = parts[3] if len(parts) > 3 else None
            pse = parts[4] if len(parts) > 4 else None
            aln = align(
                target, query, pso=pso, pse=pse, match=args.A, mismatch=args.B, gap_open=args.O,
                gap_extend=args.E, backend=args.backend,
            )
            shifts = aln.shifts(equal_score_only=not args.all_shifts, all_gaps=not args.longest_gap_only)
            _report(out, name, aln, shifts, args.a)
    finally:
        if fh is not sys.stdin:
            fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
