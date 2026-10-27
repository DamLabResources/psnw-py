"""Gotoh DP and backtrack, ported from lh3/psnw ``ksw_ggd`` + ksw2 ``ksw_backtrack``.

The same source runs in CPython (lists / bytearray) and under ``numba.njit``
(int64 / uint8 arrays). Cell order, comparison operators, and backtrack bits
follow the C code so that ties resolve the same way.

Matrix convention: target in the outer loop (row ``i``), query in the inner
loop (column ``j``). Substitution scores come from a query profile as in
ksw2: ``qp[rowcls[i] * Q + j]`` is the score of query base ``j`` against
target row ``i``. Pairwise mode uses ``rowcls = target codes`` and a 5-row
profile; profile-MSA mode gives every column its own row.

Per-row costs, all already clamped to ``>= 0`` by the caller:

* ``dcost[i]``  cost of deleting target column ``i`` (extension, also charged
  on the first deleted base on top of ``gapo``).
* ``iop[x]``    open cost of an insertion placed before target column ``x``
  (``x`` in ``0..T``); every inserted base also pays ``gape``.
* ``skip[i]``   1 if target column ``i`` may be skipped for free (profile
  insertion columns). Direction code 3.
* ``lead[i]``   H(i, -1): score of a leading gap over target ``0..i``.

Direction byte per cell (as in ksw2): bits 0-2 source of H (0 diag, 1 E,
2 F, 3 free skip); 0x08 E(i+1, j) continues E(i, j); 0x10 F(i, j+1)
continues F(i, j).
"""

from __future__ import annotations

import numpy as np

from . import _jit

NEG_INF = -0x40000000

OP_M = 0
OP_I = 1
OP_D = 2
OP_SKIP = 3


def _fill(T, Q, qp, rowcls, iop, dcost, skip, lead, gapo, gape, H, E, z, lastcol):
    H[0] = 0
    io0 = iop[0]
    for j in range(1, Q + 1):
        H[j] = -(io0 + gape * j)
    d0 = gapo + dcost[0]
    for j in range(Q):
        E[j] = H[j + 1] - d0
    E[Q] = NEG_INF

    for i in range(T):
        dn = 0
        if i + 1 < T:
            dn = dcost[i + 1]
        del_oe = gapo + dn
        ins_oe = iop[i + 1] + gape
        h1 = lead[i]
        f = h1 - ins_oe
        qi = rowcls[i] * Q
        zi = i * Q
        if skip[i]:
            for j in range(Q):
                h = H[j] + qp[qi + j]
                e = E[j]
                v = H[j + 1]
                H[j] = h1
                d = 0
                if not (h >= e):
                    d = 1
                    h = e
                if not (h >= f):
                    d = 2
                    h = f
                if v > h:
                    d = 3
                    h = v
                h1 = h
                h_del = h - del_oe
                e -= dn
                if e > h_del:
                    d |= 0x08
                else:
                    e = h_del
                E[j] = e
                h_ins = h - ins_oe
                f -= gape
                if f > h_ins:
                    d |= 0x10
                else:
                    f = h_ins
                z[zi + j] = d
        else:
            for j in range(Q):
                h = H[j] + qp[qi + j]
                e = E[j]
                H[j] = h1
                d = 0
                if not (h >= e):
                    d = 1
                    h = e
                if not (h >= f):
                    d = 2
                    h = f
                h1 = h
                h_del = h - del_oe
                e -= dn
                if e > h_del:
                    d |= 0x08
                else:
                    e = h_del
                E[j] = e
                h_ins = h - ins_oe
                f -= gape
                if f > h_ins:
                    d |= 0x10
                else:
                    f = h_ins
                z[zi + j] = d
        H[Q] = h1
        lastcol[i] = h1
    return H[Q]


def _backtrack(Q, z, i0, j0, ops):
    n = 0
    i = i0
    j = j0
    state = 0
    while i >= 0 and j >= 0:
        tmp = z[i * Q + j]
        if state == 0:
            state = tmp & 7
        elif not ((tmp >> (state + 2)) & 1):
            state = 0
        if state == 0:
            state = tmp & 7
        if state == 0:
            ops[n] = 0
            i -= 1
            j -= 1
        elif state == 1:
            ops[n] = 2
            i -= 1
        elif state == 3:
            ops[n] = 3
            i -= 1
            state = 0
        else:
            ops[n] = 1
            j -= 1
        n += 1
    while i >= 0:
        ops[n] = 2
        n += 1
        i -= 1
    while j >= 0:
        ops[n] = 1
        n += 1
        j -= 1
    return n


def _rle(ops, rop, rlen):
    """Run-length encode ops into (op, len); skip (3) counts as D (2)."""
    k = -1
    for idx in range(len(ops)):
        o = ops[idx]
        if o == 3:
            o = 2
        if k >= 0 and rop[k] == o:
            rlen[k] += 1
        else:
            k += 1
            rop[k] = o
            rlen[k] = 1
    return k + 1


_fill_nb = _jit.njit(_fill)
_backtrack_nb = _jit.njit(_backtrack)
_rle_nb = _jit.njit(_rle)


def rle(ops, backend=None):
    """``[(op, len), ...]`` from an ops array."""
    n = len(ops)
    if _jit.resolve(backend) == "python":
        rop, rlen = [0] * n, [0] * n
        k = _rle(np.asarray(ops).tolist(), rop, rlen)
        return list(zip(rop[:k], rlen[:k]))
    rop = np.empty(n, dtype=np.int64)
    rlen = np.empty(n, dtype=np.int64)
    k = _rle_nb(np.ascontiguousarray(ops, dtype=np.uint8), rop, rlen)
    return list(zip(rop[:k].tolist(), rlen[:k].tolist()))


def run(qp, rowcls, q_len, iop, dcost, skip, lead, gapo, gape, free_trail, backend=None):
    """Fill + backtrack. Returns ``(score, ops)`` with ops in target order.

    ``qp`` is the ``(R, Q)`` query profile and ``rowcls`` maps each target row
    to a profile row. ``ops`` codes: 0 M, 1 I, 2 D, 3 free skip. With
    ``free_trail`` the end row is the first row maximising H(i, Q-1); later
    rows become trailing D (or skip).
    """
    backend = _jit.resolve(backend)
    T = len(lead)
    Q = int(q_len)
    if T == 0 and Q == 0:
        return 0, np.zeros(0, dtype=np.uint8)
    if T == 0:
        return -(int(iop[0]) + gape * Q), np.full(Q, OP_I, dtype=np.uint8)
    if Q == 0:
        ops = np.where(np.asarray(skip, dtype=bool), OP_SKIP, OP_D).astype(np.uint8)
        return int(lead[T - 1]), ops

    if backend == "python":
        qp_ = np.asarray(qp).reshape(-1).tolist()
        rc_ = np.asarray(rowcls).tolist()
        iop_ = np.asarray(iop).tolist()
        dcost_ = np.asarray(dcost).tolist()
        skip_ = np.asarray(skip).tolist()
        lead_ = np.asarray(lead).tolist()
        H = [0] * (Q + 1)
        E = [0] * (Q + 1)
        z = bytearray(T * Q)
        lastcol = [0] * T
        score = _fill(T, Q, qp_, rc_, iop_, dcost_, skip_, lead_, int(gapo), int(gape), H, E, z, lastcol)
        lastcol = np.asarray(lastcol, dtype=np.int64)
        bt = _backtrack
        opsbuf = [0] * (T + Q)
    else:
        qp_ = np.ascontiguousarray(qp, dtype=np.int64).reshape(-1)
        rc_ = np.ascontiguousarray(rowcls, dtype=np.int64)
        iop_ = np.ascontiguousarray(iop, dtype=np.int64)
        dcost_ = np.ascontiguousarray(dcost, dtype=np.int64)
        skip_ = np.ascontiguousarray(skip, dtype=np.uint8)
        lead_ = np.ascontiguousarray(lead, dtype=np.int64)
        H = np.empty(Q + 1, dtype=np.int64)
        E = np.empty(Q + 1, dtype=np.int64)
        z = np.empty(T * Q, dtype=np.uint8)
        lastcol = np.empty(T, dtype=np.int64)
        score = _fill_nb(T, Q, qp_, rc_, iop_, dcost_, skip_, lead_, int(gapo), int(gape), H, E, z, lastcol)
        bt = _backtrack_nb
        opsbuf = np.empty(T + Q, dtype=np.uint8)

    i_end = T - 1
    if free_trail:
        i_end = int(np.argmax(lastcol))
        score = lastcol[i_end]
    n = bt(Q, z, i_end, Q - 1, opsbuf)
    ops = np.asarray(opsbuf[:n], dtype=np.uint8)[::-1]
    if i_end < T - 1:
        tail = np.where(np.asarray(skip[i_end + 1:], dtype=bool), OP_SKIP, OP_D).astype(np.uint8)
        ops = np.concatenate([ops, tail])
    return int(score), np.ascontiguousarray(ops)
