"""Synthetic sequence generators for tests (no real reads)."""

import random

MOTIFS = ["A", "AT", "CAG", "GT", "ACGT", "TTA", "C"]


def rand_seq(rng: random.Random, n: int, alphabet: str = "ACGT") -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def repeat_seq(rng: random.Random, n: int) -> str:
    """Low-complexity sequence built from short tandem repeats plus noise."""
    out = []
    while len(out) < n:
        motif = rng.choice(MOTIFS)
        out.extend(motif * rng.randint(1, 8))
        if rng.random() < 0.3:
            out.append(rng.choice("ACGT"))
    return "".join(out[:n])


def mutate(rng: random.Random, seq: str, lo: int, hi: int, n_events: int, max_indel: int = 12) -> str:
    """Apply substitutions / insertions / deletions inside ``[lo, hi)`` only."""
    s = list(seq)
    for _ in range(n_events):
        hi_now = hi + (len(s) - len(seq))
        if hi_now - lo < 2:
            break
        p = rng.randrange(lo, hi_now)
        kind = rng.random()
        if kind < 0.4:
            s[p] = rng.choice("ACGT")
        elif kind < 0.7:
            L = rng.randint(1, max_indel)
            s[p:p] = rand_seq(rng, L) if rng.random() < 0.5 else s[max(lo, p - L):p]
        else:
            L = rng.randint(1, max_indel)
            del s[p : min(p + L, hi_now)]
    return "".join(s)


def digits(vals) -> str:
    return "".join(str(int(v)) for v in vals)


def interior_subsidy(rng: random.Random, n: int, flank: int, vmax: int, density: float = 0.5):
    """Random subsidies that are zero on the first/last ``flank`` positions."""
    out = [0] * n
    if n > 2 * flank:
        a = rng.randrange(flank, n - flank)
        b = rng.randrange(a, n - flank + 1)
        for i in range(a, b):
            if rng.random() < density:
                out[i] = rng.randint(0, vmax)
    return out
