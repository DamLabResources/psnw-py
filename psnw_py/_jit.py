"""Optional numba backend.

The kernels in :mod:`psnw_py._kernel` are plain Python that numba can compile
unchanged. ``PSNW_PY_BACKEND=python`` forces the interpreter path even when
numba is importable; ``PSNW_PY_BACKEND=numba`` makes a missing numba an error.
"""

from __future__ import annotations

import os
from typing import Optional


def parse_backend_env(value: Optional[str] = None) -> str:
    """Return a normalised ``PSNW_PY_BACKEND`` value (``''``, ``python``, or ``numba``)."""
    if value is None:
        value = os.environ.get("PSNW_PY_BACKEND", "")
    requested = value.strip().lower()
    if requested not in ("", "python", "numba"):
        raise ImportError(f"PSNW_PY_BACKEND must be 'python' or 'numba', got {requested!r}")
    return requested


def require_numba_if_requested(requested: str, have_numba: bool) -> None:
    """Raise if the env asked for numba but it is not installed."""
    if requested == "numba" and not have_numba:
        raise ImportError("PSNW_PY_BACKEND=numba but numba is not installed")


_REQUESTED = parse_backend_env()

try:
    import numba as _numba

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    _numba = None
    HAVE_NUMBA = False

require_numba_if_requested(_REQUESTED, HAVE_NUMBA)

DEFAULT_BACKEND = "numba" if HAVE_NUMBA and _REQUESTED != "python" else "python"


def njit(fn):
    """Compile ``fn`` with numba when available; otherwise return ``None``."""
    if not HAVE_NUMBA:
        return None
    return _numba.njit(cache=True, nogil=True)(fn)


def resolve(backend: str | None) -> str:
    if backend is None:
        return DEFAULT_BACKEND
    backend = backend.lower()
    if backend not in ("python", "numba"):
        raise ValueError(f"backend must be 'python' or 'numba', got {backend!r}")
    if backend == "numba" and not HAVE_NUMBA:
        raise RuntimeError("numba backend requested but numba is not installed")
    return backend
