"""psnw-py: position-specific Needleman-Wunsch (after lh3/psnw) in Python + numba."""

from . import msa
from ._jit import DEFAULT_BACKEND, HAVE_NUMBA
from .core import (
    Alignment,
    Shift,
    align,
    choose,
    cigar_str,
    deletions,
    encode,
    decode,
    gap_costs,
    parse_cigar,
    score_cigar,
    substitution_matrix,
)
from .cut import Placed, anchor_pairs, realign_read

__version__ = "0.1.0"

__all__ = [
    "Alignment",
    "DEFAULT_BACKEND",
    "HAVE_NUMBA",
    "Placed",
    "Shift",
    "align",
    "anchor_pairs",
    "choose",
    "cigar_str",
    "decode",
    "deletions",
    "encode",
    "gap_costs",
    "msa",
    "parse_cigar",
    "realign_read",
    "score_cigar",
    "substitution_matrix",
]
