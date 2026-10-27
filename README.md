# psnw-py

Position-specific Needleman–Wunsch in Python + numba, after Heng Li's
[lh3/psnw](https://github.com/lh3/psnw). It is a global aligner where every
target (reference) base carries its own gap-cost discount. psnw-py also
builds reference-anchored multiple sequence alignments.

The use case is CRISPR cut-site realignment (CAST-seq, GUIDE-seq). The read
mapper finds the locus but writes a 30 bp cut deletion as a short indel or a
soft clip. Aligning the read globally to the reference span between its
anchors, with free deletion extension on the cut window, places the real
deletion.

- numpy is the only hard dependency. numba is optional, and the same kernel
  source runs in pure Python without it.
- In-process, no subprocess per read. Returns score, CIGAR tuples (pysam op
  codes), MD, NM, and alternative microhomology placements.
- The psnw quirks are fixed. The score you get is the score the DP optimised
  (see [Cost model](#cost-model)).

## Install

```bash
pip install -e .            # numpy only
pip install -e '.[numba]'   # recommended
pip install -e '.[dev]'     # pytest + numba
```

## API

```python
import psnw_py

target = "ACGTTGCAAGGCTTAC" + "GATCGATTGCATGCAA" + "TCAGGCATCGGATACC"
query = "ACGTTGCAAGGCTTAC" + "TCAGGCATCGGATACC"
pse = [0] * 16 + [5] * 16 + [0] * 16          # free deletion extension on the window

aln = psnw_py.align(target, query, pse=pse)    # match=6 mismatch=18 gap_open=30 gap_extend=5
aln.score, aln.cigar, aln.md, aln.nm           # (162, '16M16D16M', '16^GATCGATTGCATGCAA16', 16)
aln.cigartuples                                # ((0, 16), (2, 16), (0, 16))
aln.shifts()                                   # equal-score microhomology shifts (Shift objects)
aln.placements()                               # [aln, *shifts]
psnw_py.score_cigar(target, query, "16M16D16M", pse=pse)   # 162

# tie-break used by the cut realigner: score, deletion overlap with the window, single D
best = psnw_py.choose(aln.placements(), pos0=1000, win_start=1016, win_end=1032)

# semi-global: target overhang is free; aln.target_start / target_end give the span
psnw_py.align(target, query[:20], free_target_ends=True)
```

`pso` / `pse` are integer arrays (lists or numpy). psnw-style digit strings
(`"0005550"`) are accepted too. Values are not capped at 9.

### Cut-site realignment

`psnw_py.realign_read` is an in-process port of `anchor_pairs` +
`realign_read` from `damlab-wrappers/CRISPR/psnw-cut-realign`. It uses the
same anchor rule and ranking, and the same `realigned` / `kept` /
`no_anchor` / `skipped` status values. `ref` can be a chromosome slice if you
pass `ref_offset`.

```python
status, placed = psnw_py.realign_read(seq, ref_slice, aln_start, "5M3I114M",
                                      win_start, win_end, ref_offset=slice_start)
placed.pos0, placed.cigar, placed.score
```

### MSA

```python
from psnw_py import msa

# (a) star: merge existing placements (name, seq, pos0, cigar); no DP, order independent
m = msa.star(ref, [("r1", seq1, 100, "50M3I47M"), ("r2", seq2, 98, list_of_cigartuples)])

# (b) profile: progressive alignment to REF + reads added so far
m = msa.profile(ref, [("r1", seq1), ("r2", seq2)], pse=pse, order="length")

print(m.to_fasta())          # REF first, then reads in input order
m.ref_pos                    # per column: 0-based reference coordinate, -1 for insertion columns
m.anchor                     # reference coordinate the column sits on / precedes
m.insertion_columns(anchor)  # insertion columns in front of one reference base
m.placements["r1"]           # (pos0, cigartuples) against the reference
```

### CLI

`psnw-py input.tsv` reads the psnw TSV (`name target query [pso [pse]]`) and
writes psnw's SQ/CG/`//` report. `-A -B -O -E -a` work as in psnw. By
default every gap is shifted and only equal-score shifts are listed;
`--all-shifts` and `--longest-gap-only` turn on the psnw behaviour.

### Backends

`backend="numba"` is the default when numba imports. The kernels use
`cache=True`: the first compile takes about 0.8 s, and loading from the cache
in a new process takes about 0.2 s. `backend="python"` is the pure-Python
fallback. The environment variable `PSNW_PY_BACKEND=python|numba` sets the
default; with `numba` it is an error if numba is missing.

## Algorithm

This is Gotoh affine-gap global alignment with the full band. The target is
in the outer loop, as in `ksw_ggd`. Cells are evaluated in the same order,
with the same comparison operators and the same ksw2 backtrack bits as the C
code. When none of the fixed quirks apply, primary CIGARs match psnw exactly,
including ties in repeats.

## Cost model

For target `t` (length `T`), query `q`, and scores `A` (match), `B`
(mismatch), `O` (gap open), `E` (gap extend):

| Event | Cost |
|---|---|
| match / mismatch | `+A` / `-B`. Any pair involving `N` scores 0 |
| delete target bases `k..k+L-1` | `O + Σ max(0, E - pse[k'])` |
| insert `L` query bases before target base `x` (`x = 0..T`) | `max(0, O - pso[x]) + L·E` |

`pso` has length `T` or `T+1`. `pso[T]` is for an insertion after the last
target base; if you omit it, that insertion gets no subsidy. The DP
optimises exactly this sum. `align(...).score`, `score_cigar(...)`, every
`Shift.score`, and the CLI `AS` are all the same number.

So `pse = E` makes deletion extension free. Larger `pse` values are clamped:
they never pay a bonus for deleting more bases. `pso = O` makes an insertion
cost only its extension.

## Fixes relative to upstream psnw (commit 88e25ab)

| psnw | psnw-py |
|---|---|
| First row/column ignore `pso` and `pse` (`FIXME` in `ksw_ggd.c`). A subsidised leading deletion can lose to a mismatch plus an interior deletion | Boundary H/E/F use the same costs as the interior |
| The DP adds `pse - E` as a per-base *bonus* when `pse > E` (and similarly `pso > O`). The per-CIGAR `AS` clamps at 0, so `SQ AS` ≠ `CG AS` | Costs are clamped inside the DP. The DP score equals the rescored CIGAR |
| Reads `pso[tlen]`, one past the end (in practice the field's NUL terminator) | Explicit `pso[T]`, default no subsidy |
| Subsidies are digit strings `0`–`9`. Lines capped at 64 KB | Integer arrays, no length cap |
| Shifts only the first longest gap. Neighbours are not checked to be `M`. The left shift can produce `0M` or underflow | Shifts every gap flanked by `M` (option: longest only), keeps at least `min_flank` M, and rescores each shift. `equal_score_only=True` by default |
| Band `-w` | Full band only (cut-site targets are a few hundred bp) |

Shifts slide one gap at a time while the uncovered base pair is identical,
which is psnw's microhomology rule. Combinations of shifts across several
gaps are not enumerated.

## MSA design

**Star** (`msa.star`) uses no DP. Each read's `(pos0, CIGAR)` is walked
against the reference: `M/=/X` put a base in the reference column, `D/N` put
`-`, and `S/H/P` are dropped. Insertions that sit in front of the same
reference base (the anchor) share one block of insertion columns. The block
is as wide as the longest insertion there, and each read's inserted bases
are left-justified in it. Columns outside a read's span get `fill`. The
result does not depend on read order.

**Profile** (`msa.profile`) adds reads one at a time with the psnw kernel,
aligning each to a profile made of REF plus every read added so far. Each
column keeps residue counts. REF is counted with `ref_weight` on reference
columns.

- **Substitution:** the expected score of the read base against the column's
  non-gap residue frequencies, `Σ_r p_r · S(r, base)`. Scores are scaled by
  `scale = 100` and rounded, so the DP stays in integers. The reported score
  is divided back.
- **Reference columns:** keep the position-specific deletion cost from `pse`
  at their reference coordinate.
- **Insertion columns** (made by earlier reads): a read with no base there
  skips them for free, without opening a gap. A read with an inserted base can
  match into them at the profile score. An identical insertion in a later
  read therefore lands in the same columns. A slightly different one (a
  mismatch, or a different length) shares columns instead of opening new
  ones, as long as that scores better.
- **New insertions** before column `x` pay `pso` of the next reference column.
- **`free_ends=True`** (default) makes reference overhang free, so partial
  reads align semi-globally. Reads are always aligned end to end; soft clips
  are not modelled.

Profile mode depends on read order. The default is input order;
`order="length"` (longest first, ties broken by name) or a key callable
gives a deterministic, input-independent order. Columns are never re-aligned
after they are created: this is progressive alignment, with no refinement
pass. `MSA.placements` gives each read's `(pos0, cigartuples)` against the
reference, derived from its profile path. Insertion columns map to `I`, and
skipped insertion columns are dropped.

## Tests

```bash
pytest -q                          # PSNW_BIN=/path/to/psnw enables binary agreement tests
PSNW_PY_BACKEND=python pytest -q   # pure-Python kernel
```

- `test_vs_binary.py`: 1200 random and repeat-rich cases under three scoring
  sets, compared with the psnw binary where quirks cannot apply. Score,
  CIGAR, MD and NM are identical. Every psnw `CG` line rescores to psnw's
  own `AS`, and psnw's shifts are a subset of psnw-py's. Skipped without the
  binary.
- `test_cost_model.py`: boundary subsidies are honoured, no bonus is paid,
  and score equals rescored CIGAR on both backends. It also documents the
  upstream quirks against the binary.
- `test_backends.py`: the Python and numba backends produce identical
  pairwise and profile results.
- `test_cut.py`: synthetic cut deletions, plus parity with the damlab wrapper
  script running the binary.
- `test_msa.py`: star and profile column maps and insertion merging.
- `test_regression_castseq.py`: the 45-read CAST-seq set, read in place
  (`PSNW_REGRESSION_DIR`). Skipped if absent. No sequencing data is stored in
  this repo.

## Benchmark

Run `python bench/bench.py`. It aligns a synthetic read with a 30 bp
deletion and `pse = 5` on a 20 bp window. Numbers are from one host
(Prime-Mover, 2026-10-27):

| read | cells | C psnw, 1 subprocess/aln | C psnw, batched | psnw-py python | psnw-py numba |
|---:|---:|---:|---:|---:|---:|
| 150 bp | 27 000 | 1.76 ms | 0.066 ms | 8.8 ms | 0.139 ms |
| 300 bp | 99 000 | 2.15 ms | 0.227 ms | 33.5 ms | 0.381 ms |

## Credits

The algorithm and the original C code are Heng Li's
[psnw](https://github.com/lh3/psnw). The DP and backtrack come from
[ksw2](https://github.com/lh3/ksw2) (MIT). See `LICENSE` for the third-party
notice. The psnw tree at the pinned commit has no license file.
