"""psnw-py: position-specific Needleman-Wunsch (after lh3/psnw) in Python + numba."""

from . import msa
from ._jit import DEFAULT_BACKEND, HAVE_NUMBA
from .core import (
    Alignment,
    Shift,
    align,
    cigar_str,
    deletions,
    encode,
    decode,
    gap_costs,
    parse_cigar,
    score_cigar,
    substitution_matrix,
)

__version__ = "0.1.0"

__all__ = [
    "Alignment",
    "DEFAULT_BACKEND",
    "HAVE_NUMBA",
    "Shift",
    "align",
    "cigar_str",
    "decode",
    "deletions",
    "encode",
    "gap_costs",
    "msa",
    "parse_cigar",
    "score_cigar",
    "substitution_matrix",
]
