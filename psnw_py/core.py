"""Pairwise position-specific Needleman-Wunsch (psnw cost model, quirks fixed)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import _kernel

BAM_CMATCH = 0
BAM_CINS = 1
BAM_CDEL = 2
BAM_CREF_SKIP = 3
BAM_CSOFT_CLIP = 4
BAM_CHARD_CLIP = 5
BAM_CPAD = 6
BAM_CEQUAL = 7
BAM_CDIFF = 8

CIGAR_OPS = "MIDNSHP=X"
_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")
_REF_OPS = frozenset((BAM_CMATCH, BAM_CDEL, BAM_CREF_SKIP, BAM_CEQUAL, BAM_CDIFF))
_QUERY_OPS = frozenset((BAM_CMATCH, BAM_CINS, BAM_CSOFT_CLIP, BAM_CEQUAL, BAM_CDIFF))

CigarTuples = List[Tuple[int, int]]
SeqLike = Union[str, bytes, np.ndarray, Sequence[int]]
SubsidyLike = Union[None, str, np.ndarray, Sequence[int]]

ALPHABET = "ACGTN"
NT4 = np.full(256, 4, dtype=np.uint8)
for _i, _c in enumerate("ACGT"):
    NT4[ord(_c)] = _i
    NT4[ord(_c.lower())] = _i
_ALPHA_B = np.frombuffer(ALPHABET.encode(), dtype=np.uint8)


def encode(seq: SeqLike) -> np.ndarray:
    """Encode a sequence as uint8 codes A=0 C=1 G=2 T=3, anything else 4 (N)."""
    if isinstance(seq, str):
        seq = seq.encode("ascii")
    if isinstance(seq, (bytes, bytearray)):
        return NT4[np.frombuffer(bytes(seq), dtype=np.uint8)]
    arr = np.asarray(seq)
    if arr.dtype.kind in "US":
        return encode("".join(arr.tolist()))
    arr = arr.astype(np.int64)
    if arr.size and (arr.min() < 0 or arr.max() > 4):
        raise ValueError("integer sequences must use codes 0..4 (ACGTN)")
    return arr.astype(np.uint8)


def decode(codes: np.ndarray) -> str:
    return _ALPHA_B[np.asarray(codes, dtype=np.uint8)].tobytes().decode()


def as_subsidy(values: SubsidyLike, n: int, name: str, extra_ok: bool = False) -> np.ndarray:
    """Return an int64 subsidy array of length ``n`` (or ``n + 1`` if allowed).

    Strings are accepted for compatibility with the psnw TSV format: each
    character contributes ``ord(c) - ord('0')``.
    """
    if values is None:
        return np.zeros(n, dtype=np.int64)
    if isinstance(values, str):
        arr = np.frombuffer(values.encode("ascii"), dtype=np.uint8).astype(np.int64) - 48
    else:
        arr = np.asarray(values, dtype=np.int64).reshape(-1)
    ok = (n, n + 1) if extra_ok else (n,)
    if arr.shape[0] not in ok:
        want = " or ".join(str(x) for x in ok)
        raise ValueError(f"{name} has length {arr.shape[0]}, expected {want}")
    return arr


@lru_cache(maxsize=32)
def _substitution_matrix(match: int, mismatch: int) -> np.ndarray:
    mat = np.zeros((5, 5), dtype=np.int64)
    a, b = abs(match), -abs(mismatch)
    for i in range(4):
        for j in range(4):
            mat[i, j] = a if i == j else b
    mat.setflags(write=False)
    return mat


def substitution_matrix(match: int = 6, mismatch: int = 18) -> np.ndarray:
    """5x5 int64 matrix (read-only); N (code 4) scores 0 against everything, as in psnw."""
    return _substitution_matrix(int(match), int(mismatch))


def gap_costs(tlen: int, pso: SubsidyLike, pse: SubsidyLike, gap_open: int, gap_extend: int):
    """Clamped per-position gap costs ``(iop, dcost)``.

    * ``dcost[k] = max(0, gap_extend - pse[k])`` for deleting target base ``k``.
    * ``iop[x] = max(0, gap_open - pso[x])`` opens an insertion before target
      base ``x``; ``x == tlen`` (after the last base) uses ``pso[tlen]`` when
      ``pso`` has ``tlen + 1`` entries, otherwise no subsidy.
    """
    pse_a = as_subsidy(pse, tlen, "pse")
    pso_a = as_subsidy(pso, tlen, "pso", extra_ok=True)
    dcost = np.maximum(0, int(gap_extend) - pse_a)
    iop = np.full(tlen + 1, int(gap_open), dtype=np.int64)
    iop[: pso_a.shape[0]] = np.maximum(0, int(gap_open) - pso_a)
    return iop, dcost


def parse_cigar(cigar: Union[str, Iterable[Tuple[int, int]]]) -> CigarTuples:
    """CIGAR string or pysam-style tuples -> list of ``(op, length)``."""
    if isinstance(cigar, str):
        return [(CIGAR_OPS.index(op), int(n)) for n, op in _CIGAR_RE.findall(cigar)]
    return [(int(op), int(n)) for op, n in cigar]


def cigar_str(cigartuples: Iterable[Tuple[int, int]]) -> str:
    return "".join(f"{n}{CIGAR_OPS[op]}" for op, n in cigartuples)


def ref_length(cigartuples: Iterable[Tuple[int, int]]) -> int:
    return sum(n for op, n in cigartuples if op in _REF_OPS)


def query_length(cigartuples: Iterable[Tuple[int, int]]) -> int:
    return sum(n for op, n in cigartuples if op in _QUERY_OPS)


def deletions(pos0: int, cigar) -> List[Tuple[int, int]]:
    """``(start0, length)`` of each D, in reference coordinates."""
    out = []
    ref = pos0
    for op, n in parse_cigar(cigar):
        if op == BAM_CDEL:
            out.append((ref, n))
        if op in _REF_OPS:
            ref += n
    return out


def ops_to_cigartuples(ops: np.ndarray, backend: Optional[str] = None) -> CigarTuples:
    """Kernel op codes (0 M, 1 I, 2 D, 3 skip) to CIGAR tuples; skip -> D."""
    return _kernel.rle(ops, backend)


def _score_ops(t, q, cigartuples, tstart, iop, dcost, mat, gap_open, gap_extend) -> int:
    sc = 0
    x, y = tstart, 0
    for op, n in cigartuples:
        if op in (BAM_CMATCH, BAM_CEQUAL, BAM_CDIFF):
            sc += int(mat[t[x : x + n], q[y : y + n]].sum())
            x += n
            y += n
        elif op == BAM_CINS:
            sc -= int(iop[x]) + gap_extend * n
            y += n
        elif op == BAM_CDEL:
            sc -= gap_open + int(dcost[x : x + n].sum())
            x += n
        else:
            raise ValueError(f"unsupported CIGAR op {CIGAR_OPS[op]} in score_cigar")
    return sc


def score_cigar(
    target: SeqLike,
    query: SeqLike,
    cigar,
    *,
    target_start: int = 0,
    pso: SubsidyLike = None,
    pse: SubsidyLike = None,
    match: int = 6,
    mismatch: int = 18,
    gap_open: int = 30,
    gap_extend: int = 5,
) -> int:
    """Score a CIGAR under the psnw-py cost model (the same one the DP optimises)."""
    t, q = encode(target), encode(query)
    iop, dcost = gap_costs(len(t), pso, pse, gap_open, gap_extend)
    cig = parse_cigar(cigar)
    if query_length(cig) != len(q):
        raise ValueError("CIGAR query length does not match query")
    if target_start + ref_length(cig) > len(t):
        raise ValueError("CIGAR runs past the end of the target")
    return _score_ops(t, q, cig, target_start, iop, dcost, substitution_matrix(match, mismatch), gap_open, gap_extend)


@dataclass(frozen=True)
class Shift:
    """One alternative placement of a single gap."""

    cigartuples: Tuple[Tuple[int, int], ...]
    score: int
    gap_index: int
    offset: int
    target_start: int = 0

    @property
    def cigar(self) -> str:
        return cigar_str(self.cigartuples)


@dataclass(frozen=True)
class Alignment:
    """Result of :func:`align`.

    ``target_start`` / ``target_end`` delimit the consumed target span. In
    global mode they are ``0`` / ``len(target)`` and the CIGAR may begin or
    end with ``D`` like psnw. With ``free_target_ends`` leading and trailing
    target overhang is free and excluded from the CIGAR.
    """

    score: int
    cigartuples: Tuple[Tuple[int, int], ...]
    target_start: int
    target_end: int
    _t: np.ndarray = field(repr=False, compare=False)
    _q: np.ndarray = field(repr=False, compare=False)
    _iop: np.ndarray = field(repr=False, compare=False)
    _dcost: np.ndarray = field(repr=False, compare=False)
    _mat: np.ndarray = field(repr=False, compare=False)
    _gap_open: int = field(repr=False, compare=False, default=30)
    _gap_extend: int = field(repr=False, compare=False, default=5)

    @property
    def cigar(self) -> str:
        return cigar_str(self.cigartuples)

    @property
    def target(self) -> str:
        return decode(self._t)

    @property
    def query(self) -> str:
        return decode(self._q)

    def score_of(self, cigartuples, target_start: Optional[int] = None) -> int:
        """Score another CIGAR for the same target/query/costs."""
        ts = self.target_start if target_start is None else target_start
        return _score_ops(
            self._t, self._q, parse_cigar(cigartuples), ts, self._iop, self._dcost, self._mat,
            self._gap_open, self._gap_extend,
        )

    @property
    def md(self) -> str:
        return _md_nm(self._t, self._q, self.cigartuples, self.target_start)[0]

    @property
    def nm(self) -> int:
        return _md_nm(self._t, self._q, self.cigartuples, self.target_start)[1]

    def shifts(self, *, equal_score_only: bool = True, all_gaps: bool = True, min_flank: int = 1) -> List[Shift]:
        """Microhomology shifts of gaps in the primary CIGAR.

        A gap (I or D) flanked by M on both sides is slid one base at a time
        while the base pair it uncovers is identical (``t == q``), as in
        psnw's ``shift_cigar``. Each gap is shifted alone; the other gaps stay
        put. ``all_gaps=False`` restricts this to the first longest gap (the
        psnw behaviour). ``min_flank`` is the shortest M left on either side.
        Each shift is rescored under the position-specific costs; with
        ``equal_score_only`` only shifts tying the primary score are kept.
        The primary placement itself is not included.
        """
        cig = list(self.cigartuples)
        gaps = []
        x, y = self.target_start, 0
        longest = (-1, -1)
        for k, (op, n) in enumerate(cig):
            if op in (BAM_CINS, BAM_CDEL):
                gaps.append((k, x, y))
                if n > longest[1]:
                    longest = (k, n)
            if op in _REF_OPS:
                x += n
            if op in _QUERY_OPS:
                y += n
        if not all_gaps:
            gaps = [g for g in gaps if g[0] == longest[0]]

        t, q = self._t, self._q
        out: List[Shift] = []
        for k, gx, gy in gaps:
            if k == 0 or k == len(cig) - 1:
                continue
            if cig[k - 1][0] != BAM_CMATCH or cig[k + 1][0] != BAM_CMATCH:
                continue
            op, glen = cig[k]
            left_m, right_m = cig[k - 1][1], cig[k + 1][1]
            ex = glen if op == BAM_CDEL else 0
            ey = glen if op == BAM_CINS else 0
            o = 1
            while right_m - o >= min_flank and t[gx + o - 1] == q[gy + o - 1]:
                out.append(self._make_shift(cig, k, o))
                o += 1
            o = 1
            while left_m - o >= min_flank and t[gx + ex - o] == q[gy + ey - o]:
                out.append(self._make_shift(cig, k, -o))
                o += 1
        if equal_score_only:
            out = [s for s in out if s.score == self.score]
        return out

    def _make_shift(self, cig, k, o) -> Shift:
        new = list(cig)
        new[k - 1] = (BAM_CMATCH, cig[k - 1][1] + o)
        new[k + 1] = (BAM_CMATCH, cig[k + 1][1] - o)
        return Shift(tuple(new), self.score_of(new), k, o, self.target_start)

    def placements(self, **kw) -> list:
        """Primary alignment first, then :meth:`shifts` (same keywords)."""
        return [self] + self.shifts(**kw)


def _md_nm(t, q, cigartuples, tstart) -> Tuple[str, int]:
    parts = []
    nm = 0
    run = 0
    x, y = tstart, 0
    for op, n in cigartuples:
        if op in (BAM_CMATCH, BAM_CEQUAL, BAM_CDIFF):
            for k in range(n):
                a, b = t[x + k], q[y + k]
                if a == b and a < 4:
                    run += 1
                else:
                    parts.append(f"{run}{ALPHABET[a]}")
                    run = 0
                    nm += 1
            x += n
            y += n
        elif op == BAM_CINS:
            y += n
            nm += n
        elif op == BAM_CDEL:
            parts.append(f"{run}^{decode(t[x:x + n])}")
            run = 0
            x += n
            nm += n
    parts.append(str(run))
    return "".join(parts), nm


def align(
    target: SeqLike,
    query: SeqLike,
    *,
    pso: SubsidyLike = None,
    pse: SubsidyLike = None,
    match: int = 6,
    mismatch: int = 18,
    gap_open: int = 30,
    gap_extend: int = 5,
    free_target_ends: bool = False,
    backend: Optional[str] = None,
) -> Alignment:
    """Globally align ``query`` to ``target`` with position-specific gap costs.

    Parameters mirror psnw (``-A -B -O -E``). ``pso`` (length ``len(target)``
    or ``len(target) + 1``) discounts the open cost of an insertion placed
    before target base ``x``; ``pse`` (length ``len(target)``) discounts the
    cost of deleting target base ``k``. Both are clamped so a gap never costs
    less than zero per base (see README). ``free_target_ends`` makes target
    overhang at either end free (query still fully aligned).
    """
    t, q = encode(target), encode(query)
    T = len(t)
    iop, dcost = gap_costs(T, pso, pse, gap_open, gap_extend)
    mat = substitution_matrix(match, mismatch)
    qp = mat[:, q]
    skip = np.zeros(T, dtype=np.uint8)
    if free_target_ends:
        lead = np.zeros(T, dtype=np.int64)
    else:
        lead = -(int(gap_open) + np.cumsum(dcost))
    score, ops = _kernel.run(
        qp, t, len(q), iop, dcost, skip, lead, int(gap_open), int(gap_extend), free_target_ends, backend
    )
    cig = ops_to_cigartuples(ops, backend)
    tstart, tend = 0, T
    if free_target_ends:
        if cig and cig[0][0] == BAM_CDEL:
            tstart = cig.pop(0)[1]
        if cig and cig[-1][0] == BAM_CDEL:
            tend -= cig.pop()[1]
    return Alignment(
        score=score,
        cigartuples=tuple(cig),
        target_start=tstart,
        target_end=tend,
        _t=t,
        _q=q,
        _iop=iop,
        _dcost=dcost,
        _mat=mat,
        _gap_open=int(gap_open),
        _gap_extend=int(gap_extend),
    )
