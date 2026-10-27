"""Reference-anchored multiple sequence alignment.

Two builders share one :class:`MSA` result:

* :func:`star` merges existing per-read placements ``(pos0, cigar)`` against
  the reference. No DP; insertion columns at the same reference point are
  merged left-justified. Independent of read order.
* :func:`profile` aligns reads one at a time to a growing profile (reference
  plus reads added so far) with the psnw kernel. Reference-derived columns keep
  their position-specific gap costs; insertion columns created by earlier reads
  can be matched by later reads or skipped for free, so the same insertion in
  different reads lands in the same columns. Result depends on read order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import _kernel
from .core import (
    BAM_CDEL,
    BAM_CDIFF,
    BAM_CEQUAL,
    BAM_CINS,
    BAM_CMATCH,
    BAM_CREF_SKIP,
    BAM_CSOFT_CLIP,
    CigarTuples,
    SubsidyLike,
    encode,
    gap_costs,
    parse_cigar,
    substitution_matrix,
)

GAP = "-"


@dataclass
class MSA:
    """Aligned rows plus a per-column map back to the reference.

    ``ref_pos[c]`` is the 0-based reference coordinate of column ``c`` or
    ``-1`` for an insertion column. ``anchor[c]`` is the reference coordinate
    the column sits on (reference columns) or immediately before (insertion
    columns). ``placements`` maps each read to ``(pos0, cigartuples)`` against
    the reference; ``scores`` holds the profile DP score (profile mode only).
    """

    names: List[str]
    rows: List[str]
    ref_pos: np.ndarray
    anchor: np.ndarray
    placements: Dict[str, Tuple[int, CigarTuples]] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)

    @property
    def ncols(self) -> int:
        return int(self.ref_pos.shape[0])

    def row(self, name: str) -> str:
        return self.rows[self.names.index(name)]

    def column_of(self, ref_pos: int) -> int:
        hits = np.flatnonzero(self.ref_pos == ref_pos)
        if not hits.size:
            raise KeyError(ref_pos)
        return int(hits[0])

    def insertion_columns(self, anchor: Optional[int] = None) -> List[int]:
        mask = self.ref_pos < 0
        if anchor is not None:
            mask &= self.anchor == anchor
        return np.flatnonzero(mask).tolist()

    def to_fasta(self, width: int = 0) -> str:
        out = []
        for name, row in zip(self.names, self.rows):
            out.append(f">{name}")
            if width and width > 0:
                out.extend(row[i : i + width] for i in range(0, len(row), width))
            else:
                out.append(row)
        return "\n".join(out) + "\n"


# --------------------------------------------------------------------- star


def _walk(seq: str, pos0: int, cig: CigarTuples):
    """Per-read reference bases ``{ref: char}`` and insertions ``{anchor: str}``."""
    bases: Dict[int, str] = {}
    ins: Dict[int, str] = {}
    r, y = pos0, 0
    for op, n in cig:
        if op in (BAM_CMATCH, BAM_CEQUAL, BAM_CDIFF):
            for k in range(n):
                bases[r + k] = seq[y + k]
            r += n
            y += n
        elif op in (BAM_CDEL, BAM_CREF_SKIP):
            for k in range(n):
                bases[r + k] = GAP
            r += n
        elif op == BAM_CINS:
            ins[r] = ins.get(r, "") + seq[y : y + n]
            y += n
        elif op == BAM_CSOFT_CLIP:
            y += n
    return bases, ins, pos0, r


def star(
    ref: str,
    reads: Iterable[Tuple[str, str, int, Union[str, CigarTuples]]],
    *,
    ref_offset: int = 0,
    region: Optional[Tuple[int, int]] = None,
    ref_name: str = "REF",
    fill: str = GAP,
) -> MSA:
    """Star MSA from existing placements.

    ``reads`` yields ``(name, seq, pos0, cigar)``; ``pos0`` and ``region`` are
    reference coordinates (``ref[0]`` is at ``ref_offset``). Soft clips are
    dropped. Insertions anchored at the same reference point share columns,
    left-justified; the block width is the longest insertion there. Columns
    outside a read's span use ``fill``. Insertions at ``region`` end points
    are kept (anchor ``== start`` or ``== end``).
    """
    parsed = []
    for name, seq, pos0, cigar in reads:
        cig = parse_cigar(cigar)
        parsed.append((name, seq.upper(), int(pos0), cig) + _walk(seq.upper(), int(pos0), cig))
    if region is None:
        if parsed:
            start = min(p[6] for p in parsed)
            end = max(p[7] for p in parsed)
        else:
            start, end = ref_offset, ref_offset + len(ref)
    else:
        start, end = region
    if start < ref_offset or end > ref_offset + len(ref) or end < start:
        raise ValueError("region must lie inside ref")

    width: Dict[int, int] = {}
    for p in parsed:
        for a, s in p[5].items():
            if start <= a <= end:
                width[a] = max(width.get(a, 0), len(s))

    ref_pos: List[int] = []
    anchor: List[int] = []
    for r in range(start, end + 1):
        for _ in range(width.get(r, 0)):
            ref_pos.append(-1)
            anchor.append(r)
        if r < end:
            ref_pos.append(r)
            anchor.append(r)

    ref_row = "".join(GAP if rp < 0 else ref[rp - ref_offset] for rp in ref_pos)
    rows = [ref_row]
    names = [ref_name]
    placements = {}
    for name, seq, pos0, cig, bases, ins, rs, re_ in parsed:
        chars = []
        ins_k: Dict[int, int] = {}
        for rp, a in zip(ref_pos, anchor):
            if rp >= 0:
                chars.append(bases.get(rp, fill))
            else:
                k = ins_k.get(a, 0)
                ins_k[a] = k + 1
                s = ins.get(a, "")
                if k < len(s):
                    chars.append(s[k])
                elif rs < a < re_ or a in ins:
                    chars.append(GAP)
                else:
                    chars.append(fill)
        names.append(name)
        rows.append("".join(chars))
        placements[name] = (pos0, cig)
    return MSA(
        names=names,
        rows=rows,
        ref_pos=np.asarray(ref_pos, dtype=np.int64),
        anchor=np.asarray(anchor, dtype=np.int64),
        placements=placements,
        order=[p[0] for p in parsed],
    )


# ------------------------------------------------------------------ profile


def _ordered(reads: List[Tuple[str, str]], order) -> List[int]:
    idx = list(range(len(reads)))
    if order == "input":
        return idx
    if order == "length":
        return sorted(idx, key=lambda i: (-len(reads[i][1]), reads[i][0]))
    if callable(order):
        return sorted(idx, key=lambda i: order(reads[i]))
    raise ValueError("order must be 'input', 'length', or a key callable")


def profile(
    ref: str,
    reads: Iterable[Sequence],
    *,
    pso: SubsidyLike = None,
    pse: SubsidyLike = None,
    ref_offset: int = 0,
    ref_name: str = "REF",
    order: Union[str, Callable] = "input",
    free_ends: bool = True,
    match: int = 6,
    mismatch: int = 18,
    gap_open: int = 30,
    gap_extend: int = 5,
    ref_weight: float = 1.0,
    scale: int = 100,
    fill: str = GAP,
    backend: Optional[str] = None,
) -> MSA:
    """Progressive profile MSA anchored on ``ref``.

    ``reads`` yields tuples whose first two fields are ``(name, seq)`` (extra
    fields such as pos0/cigar are ignored). ``pso``/``pse`` are indexed by
    ``ref`` position exactly as in :func:`psnw_py.align`.

    Column scoring (integer DP on ``scale``-multiplied costs):

    * substitution: expected score of the read base against the column's
      residue counts (gaps excluded; REF counted with ``ref_weight`` on
      reference columns), rounded after scaling;
    * reference columns: deletion costs ``gap_open`` + clamped ``pse``
      extension, as in pairwise mode;
    * insertion columns: may be skipped for free (no gap opened);
    * new insertion before column ``x``: ``pso`` of the next reference column.

    ``free_ends`` makes reference overhang free so partial reads align
    semi-globally; reads themselves are always aligned end to end.
    """
    reads = [(str(r[0]), str(r[1]).upper()) for r in reads]
    T = len(ref)
    mat = substitution_matrix(match, mismatch)
    iop_ref, dcost_ref = gap_costs(T, pso, pse, gap_open, gap_extend)
    sc = int(scale)
    gapo, gape = int(gap_open) * sc, int(gap_extend) * sc
    matf = mat.astype(np.float64)

    col_ref: List[int] = list(range(T))
    counts = np.zeros((T, 5), dtype=np.float64)
    rcodes = encode(ref)
    counts[np.arange(T), rcodes] += ref_weight
    rows: List[List[str]] = [list(ref.upper())]

    placements = {}
    scores = {}
    order_idx = _ordered(reads, order)
    for ridx in order_idx:
        name, seq = reads[ridx]
        q = encode(seq)
        C = len(col_ref)
        is_ref = np.asarray([c >= 0 for c in col_ref], dtype=bool)
        cr = np.asarray(col_ref, dtype=np.int64)

        tot = counts.sum(axis=1)
        tot[tot == 0] = 1.0
        prof = np.rint(sc * (counts @ matf) / tot[:, None]).astype(np.int64)

        next_ref = np.empty(C + 1, dtype=np.int64)
        nxt = T
        next_ref[C] = T
        for c in range(C - 1, -1, -1):
            if is_ref[c]:
                nxt = col_ref[c]
            next_ref[c] = nxt
        iop = iop_ref[next_ref] * sc
        dcost = np.where(is_ref, dcost_ref[np.where(is_ref, cr, 0)] * sc, 0).astype(np.int64)
        skip = (~is_ref).astype(np.uint8)
        if free_ends:
            lead = np.zeros(C, dtype=np.int64)
        else:
            lead = np.empty(C, dtype=np.int64)
            acc, opened = 0, False
            for c in range(C):
                if is_ref[c]:
                    acc = acc - int(dcost[c]) if opened else -(gapo + int(dcost[c]))
                    opened = True
                lead[c] = acc

        score, ops = _kernel.run(prof[:, q], np.arange(C), len(q), iop, dcost, skip, lead, gapo, gape, free_ends, backend)
        scores[name] = score / sc
        ops = ops.tolist()

        qpos = [k for k, o in enumerate(ops) if o in (_kernel.OP_M, _kernel.OP_I)]
        first_q = qpos[0] if qpos else len(ops)
        last_q = qpos[-1] if qpos else -1

        new_cols: List[int] = []
        read_row: List[str] = []
        hits: List[Tuple[int, int]] = []
        cig: CigarTuples = []
        pos0 = None
        ci = qi = 0
        for k, o in enumerate(ops):
            inside = (first_q <= k <= last_q) or not free_ends
            if o == _kernel.OP_I or o == _kernel.OP_M:
                if o == _kernel.OP_I:
                    new_cols.append(-1)
                    op = BAM_CINS
                else:
                    new_cols.append(ci)
                    op = BAM_CMATCH if is_ref[ci] else BAM_CINS
                if pos0 is None:
                    pos0 = int(next_ref[ci])
                _push(cig, op)
                read_row.append(seq[qi])
                hits.append((len(new_cols) - 1, int(q[qi])))
                qi += 1
                if o == _kernel.OP_M:
                    ci += 1
                continue
            new_cols.append(ci)
            read_row.append(GAP if inside else fill)
            if inside and is_ref[ci]:
                if pos0 is None:
                    pos0 = int(next_ref[ci])
                _push(cig, BAM_CDEL)
            ci += 1
        placements[name] = (ref_offset + (pos0 if pos0 is not None else 0), cig)

        if len(new_cols) != C:
            rows = [_expand(r, new_cols, fill) for r in rows]
            new_ref: List[int] = []
            new_counts = np.zeros((len(new_cols), 5), dtype=np.float64)
            for k, c in enumerate(new_cols):
                if c >= 0:
                    new_ref.append(col_ref[c])
                    new_counts[k] = counts[c]
                else:
                    new_ref.append(-1)
            col_ref, counts = new_ref, new_counts
        for k, code in hits:
            counts[k, code] += 1.0
        rows.append(read_row)

    cr = np.asarray(col_ref, dtype=np.int64)
    anchor = np.empty(len(col_ref), dtype=np.int64)
    nxt = T
    for c in range(len(col_ref) - 1, -1, -1):
        if col_ref[c] >= 0:
            nxt = col_ref[c]
        anchor[c] = ref_offset + nxt
    ref_pos = np.where(cr >= 0, cr + ref_offset, -1)

    by_name = {reads[i][0]: rows[1 + n] for n, i in enumerate(order_idx)}
    names = [ref_name] + [r[0] for r in reads]
    out_rows = ["".join(rows[0])] + ["".join(by_name[r[0]]) for r in reads]
    return MSA(
        names=names,
        rows=out_rows,
        ref_pos=ref_pos,
        anchor=anchor,
        placements=placements,
        scores=scores,
        order=[reads[i][0] for i in order_idx],
    )


def _push(cig: CigarTuples, op: int) -> None:
    if cig and cig[-1][0] == op:
        cig[-1] = (op, cig[-1][1] + 1)
    else:
        cig.append((op, 1))


def _expand(row: List[str], new_cols: List[int], fill: str) -> List[str]:
    """Re-index an existing row onto ``new_cols`` (``-1`` = new insertion column)."""
    out = [GAP] * len(new_cols)
    for k, c in enumerate(new_cols):
        if c >= 0:
            out[k] = row[c]
    if fill != GAP:
        n = len(new_cols)
        left = [None] * n
        last = None
        for k in range(n):
            if new_cols[k] >= 0:
                last = out[k]
            left[k] = last
        last = None
        for k in range(n - 1, -1, -1):
            if new_cols[k] >= 0:
                last = out[k]
            elif left[k] in (None, fill) or last in (None, fill):
                out[k] = fill
    return out
