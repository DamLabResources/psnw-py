"""Cut-site realignment helpers (in-process port of ``psnw_cut_realign.py``).

Same anchor rule and ranking as the damlab wrapper: find the read's longest
prefix and suffix near the mapper alignment, require them on opposite sides
of the cut window, globally align the read to the reference span between
them with ``pse`` on window bases only, and keep the best placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .core import align, choose, cigar_str, parse_cigar, ref_length


@dataclass
class Placed:
    pos0: int
    cigar: str
    score: int
    extra: int
    n_alt: int
    cigartuples: Tuple[Tuple[int, int], ...] = ()


def _find_all(hay: str, needle: str, offset: int) -> List[int]:
    hits = []
    start = 0
    while True:
        i = hay.find(needle, start)
        if i < 0:
            break
        hits.append(offset + i)
        start = i + 1
    return hits


def anchor_pairs(
    read: str,
    ref: str,
    aln_start: int,
    aln_end: int,
    win_start: int,
    win_end: int,
    max_del: int = 80,
    min_k: int = 6,
    ref_offset: int = 0,
) -> List[Tuple[int, int, int]]:
    """Return ``(left0, right_end0, extra)``; ``extra = span - len(read)``.

    ``ref`` may be a slice of the chromosome starting at ``ref_offset``; all
    coordinates in and out are chromosome coordinates.
    """
    ref_end = ref_offset + len(ref)
    local_start = max(ref_offset, min(aln_start, win_start) - max_del)
    local_end = min(ref_end, max(aln_end, win_end) + max_del)
    local = ref[local_start - ref_offset : local_end - ref_offset]
    if len(read) < min_k * 2 or not local:
        return []

    left_k = None
    left_pos: List[int] = []
    for k in range(min(40, len(read) // 2), min_k - 1, -1):
        hits = [g for g in _find_all(local, read[:k], local_start) if aln_start - max_del <= g <= aln_start + 5]
        if hits:
            left_k = k
            left_pos = hits
            break
    if left_k is None:
        return []

    pairs: List[Tuple[int, int, int]] = []
    seen = set()
    for left in left_pos:
        for k in range(min(40, len(read) // 2), min_k - 1, -1):
            found = False
            for g in _find_all(local, read[-k:], local_start):
                end = g + k
                extra = end - left - len(read)
                if g < left + left_k:
                    continue
                if not (0 <= extra <= max_del):
                    continue
                if not (left < win_start and end > win_end):
                    continue
                key = (left, end)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((left, end, extra))
                found = True
            if found:
                break
    return pairs


def realign_read(
    seq: str,
    ref: str,
    aln_start: int,
    cigar,
    win_start: int,
    win_end: int,
    *,
    max_del: int = 80,
    min_k: int = 6,
    pse_on: int = 5,
    ref_offset: int = 0,
    match: int = 6,
    mismatch: int = 18,
    gap_open: int = 30,
    gap_extend: int = 5,
    backend: Optional[str] = None,
) -> Tuple[str, Optional[Placed]]:
    """Return ``(status, Placed | None)``; status is ``realigned``, ``kept``,
    ``no_anchor`` or ``skipped`` (same meanings as the wrapper TSV)."""
    cig = parse_cigar(cigar)
    aln_end = aln_start + ref_length(cig)
    if aln_end <= win_start - max_del or aln_start >= win_end + max_del:
        return "skipped", None
    pairs = anchor_pairs(seq, ref, aln_start, aln_end, win_start, win_end, max_del, min_k, ref_offset)
    if not pairs:
        return "no_anchor", None

    best = None
    for left, end, extra in pairs:
        target = ref[left - ref_offset : end - ref_offset]
        pse = [pse_on if win_start <= left + i < win_end else 0 for i in range(len(target))]
        aln = align(
            target, seq, pse=pse, match=match, mismatch=mismatch, gap_open=gap_open,
            gap_extend=gap_extend, backend=backend,
        )
        cands = aln.placements()
        pick = choose(cands, left, win_start, win_end)
        rank = (pick.score, int(extra == 0), -abs(extra))
        if best is None or rank > best[0]:
            best = (rank, pick, len(cands), left, extra)
    _, pick, n_alt, left, extra = best
    placed = Placed(
        pos0=left + pick.target_start,
        cigar=cigar_str(pick.cigartuples),
        score=pick.score,
        extra=extra,
        n_alt=n_alt,
        cigartuples=tuple(pick.cigartuples),
    )
    changed = placed.pos0 != aln_start or placed.cigar != cigar_str(cig)
    return ("realigned" if changed else "kept"), placed
