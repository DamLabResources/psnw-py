# Algorithm

psnw-py is a full-band Gotoh affine-gap aligner. The target (reference) sits in
the outer loop and every target column can carry its own gap costs. The DP is
a Python/numba port of Heng Li's `ksw_ggd` in
[lh3/psnw](https://github.com/lh3/psnw), with the quirks listed at the end of
this page fixed. The same kernel drives pairwise alignment and the progressive
profile MSA.

## Parameters you supply

### Sequences

| Argument | Meaning |
|---|---|
| `target` / `ref` | Reference string (or NT codes). Outer loop of the DP. |
| `query` / read seq | Query string. Always aligned end to end. |

Bases are encoded as A=0, C=1, G=2, T=3, anything else =4 (`N`). Pairs that
involve `N` score 0.

### Substitution and gap scores

These mirror psnw's `-A -B -O -E` and are the same defaults:

| Argument | Default | Role |
|---|---:|---|
| `match` (`A`) | 6 | Score for an identical A/C/G/T pair |
| `mismatch` (`B`) | 18 | Penalty for a differing A/C/G/T pair |
| `gap_open` (`O`) | 30 | Cost to open a gap |
| `gap_extend` (`E`) | 5 | Cost per base to extend a gap |

### Position-specific subsidies: `pse` and `pso`

These are the point of the algorithm. Both are optional; omit them (or pass
all zeros) for ordinary Needleman–Wunsch / Gotoh costs.

| Array | Length | Effect |
|---|---|---|
| **`pse`** (position-specific *extension* / deletion subsidy) | `len(target)` | Lowers the **per-base deletion** cost of target base `k` |
| **`pso`** (position-specific *open* / insertion subsidy) | `len(target)` or `len(target)+1` | Lowers the **gap-open** cost of an insertion placed **before** target base `x` |

They are turned into clamped costs before the DP runs:

```python
# psnw_py.core.gap_costs
dcost[k] = max(0, gap_extend - pse[k])   # delete target base k
iop[x]   = max(0, gap_open   - pso[x])   # open an insertion before base x
```

So:

- `pse[k] = gap_extend` (default 5) makes deleting base `k` free on extension.
  A run of such bases costs **one open** and nothing per base.
- `pse[k] > gap_extend` is clamped: it never pays a *bonus* for deleting more.
- `pso[x] = gap_open` (default 30) makes an insertion at `x` cost only
  `L · gap_extend` (no open).
- `pso` may be one longer than the target: `pso[T]` subsidises an insertion
  *after* the last base. If you omit that extra entry, the trailing insertion
  gets a full open.

Both accept integer lists/arrays or psnw-style digit strings (`"0005550"`).

Event costs the DP maximises:

| Event | Score / cost |
|---|---|
| match / mismatch | `+match` / `-mismatch` |
| delete bases `k..k+L-1` | `-(gap_open + Σ dcost[k'])` |
| insert `L` bases before `x` | `-(iop[x] + L · gap_extend)` |

`align(...).score`, `score_cigar(...)`, every `Shift.score`, and the CLI `AS`
are the same number: the DP optimises exactly this sum. See {doc}`cost_model`
for the compact table.

### Alignment mode flags

| Argument | Default | Role |
|---|---|---|
| `free_target_ends` (pairwise) / `free_ends` (profile MSA) | `False` / `True` | Target overhang is free (semi-global). The query is still fully consumed. |
| `backend` | `"numba"` if installed else `"python"` | Which kernel implementation to run. Override with `PSNW_PY_BACKEND`. |

### MSA-only parameters

| Argument | Default | Role |
|---|---|---|
| `order` | `"input"` | Read addition order: `"input"`, `"length"`, or a key callable |
| `ref_offset` | `0` | Chromosome coordinate of `ref[0]` (for reported `pos0`) |
| `ref_weight` | `1.0` | How heavily REF counts in each profile column |
| `scale` | `100` | Integer scale for profile substitution expectations |
| `fill` | `"-"` | Character for columns outside a read's placed span |

## Pairwise DP

`align` builds the substitution query-profile, converts subsidies to
`iop`/`dcost`, and calls the kernel:

```python
# psnw_py.core.align (simplified)
t, q = encode(target), encode(query)
iop, dcost = gap_costs(len(t), pso, pse, gap_open, gap_extend)
mat = substitution_matrix(match, mismatch)
qp = mat[:, q]                          # 5 × Q query profile
skip = zeros(len(t))                    # pairwise: no free-skip columns
lead = zeros(...) if free_target_ends else -(gap_open + cumsum(dcost))
score, ops = _kernel.run(qp, t, len(q), iop, dcost, skip, lead,
                         gap_open, gap_extend, free_target_ends, backend)
```

### Matrix layout

From the kernel docstring:

- Rows = target columns (`i`), columns = query bases (`j`).
- `H[j]` is the best score ending at the current target row with query
  prefix `j`.
- `E[j]` continues a deletion (gap in the query).
- `F` continues an insertion (gap in the target), carried across `j` inside
  each row.
- `lead[i]` is the score of a leading gap over target `0..i` (paid or free).

Initialisation seeds a leading insertion along the first row and a leading
deletion into `E`:

```python
# psnw_py._kernel._fill (opening)
H[0] = 0
for j in range(1, Q + 1):
    H[j] = -(iop[0] + gape * j)       # leading insertion before target[0]
for j in range(Q):
    E[j] = H[j + 1] - (gapo + dcost[0])
```

Each cell chooses among diagonal (match/mismatch), deletion (`E`), and
insertion (`F`), with the same comparison order as ksw2 so ties break like
psnw. Direction bits are stored for backtrack: bits 0–2 select the `H`
source; `0x08` / `0x10` mark continued `E` / `F`.

### Backtrack and CIGAR

Backtrack walks those direction bytes to an ops array (`M` / `I` / `D`, plus
`SKIP` in profile mode). Ops are run-length encoded into pysam-style CIGAR
tuples. With `free_target_ends`, a leading or trailing target deletion is
stripped from the CIGAR and recorded as `target_start` / `target_end`
instead.

### Microhomology shifts

After the primary path, `Alignment.shifts()` slides each gap (or only the
longest gap) while the uncovered base pair stays identical, then rescores:

```python
# slide while t[gx + o - 1] == q[gy + o - 1], keep min_flank matches
while right_m - o >= min_flank and t[gx + o - 1] == q[gy + o - 1]:
    out.append(self._make_shift(cig, k, o))
```

By default only equal-score shifts are kept (`equal_score_only=True`) and
every gap is considered (`all_gaps=True`). Upstream psnw shifts only the
first longest gap and does not always rescore.

## Reference-anchored MSA

### Star (`msa.star`)

No DP. Each read's `(pos0, CIGAR)` is walked onto the reference: matches fill
reference columns, deletions put `-`, and insertions that sit in front of the
same reference base share one block of columns (left-justified to the longest
insertion there). Order-independent.

### Profile (`msa.profile`)

Progressive alignment: start from REF, then align each read to a growing
profile with the same kernel.

1. **Columns.** Reference columns keep `pse`/`pso` indexed by reference
   coordinate. Insertion columns created by earlier reads are marked
   `skip=1` and may be passed for free (no gap open).
2. **Substitution.** Each column keeps residue counts. The score for a read
   base is the expected match/mismatch against the column's non-gap
   frequencies, scaled by `scale` and rounded so the DP stays in integers:

   ```python
   prof = rint(scale * (counts @ S) / tot)   # S = substitution matrix
   ```

3. **New insertions** before profile column `x` pay `pso` of the *next*
   reference column.
4. **`free_ends=True`** (default) zeroes `lead`, so reference overhang is
   free; partial reads align semi-globally. Reads are always consumed end to
   end (no soft-clip model).

```python
# psnw_py.msa.profile (per read, simplified)
iop = iop_ref[next_ref] * scale
dcost = where(is_ref, dcost_ref[ref_coord] * scale, 0)
skip = (~is_ref).astype(uint8)          # free skip on prior insertion columns
lead = zeros(C) if free_ends else paid_leading_deletions(...)
score, ops = _kernel.run(prof[:, q], arange(C), len(q),
                         iop, dcost, skip, lead, gapo, gape, free_ends, backend)
```

`order` controls which read is added first (`"length"` = longest first).
Columns are never refined after creation. `MSA.placements` maps each read
back to a reference `(pos0, cigartuples)`; skipped insertion columns become
ordinary `I` ops or are dropped when unused.

## Differences from lh3/psnw

| Upstream psnw | psnw-py |
|---|---|
| Boundary row/column ignore `pso` / `pse` (`FIXME` in `ksw_ggd.c`) | Boundary cells use the same costs as the interior |
| DP can add `pse - E` as a *bonus* when `pse > E`; per-CIGAR `AS` clamps at 0, so `SQ AS ≠ CG AS` | Costs clamped inside the DP; DP score equals rescored CIGAR |
| Reads `pso[tlen]` one past the array (NUL terminator in practice) | Explicit `pso[T]`, default no subsidy |
| Subsidies are digit characters `0`–`9`; lines capped at 64 KB | Integer arrays / digit strings, no length cap |
| Shifts only the first longest gap; neighbours not required to be `M`; left shift can yield `0M` | Shifts every gap flanked by `M` (or longest only); keeps `min_flank`; rescores; optional equal-score filter |
| Optional band `-w` | Full band only (targets are expected to be hundreds of bases) |
| C subprocess / TSV report | In-process library + optional CLI |

Under the conditions where those quirks cannot fire (`pse ≤ E`, `pso ≤ O`,
subsidies zero near both ends), primary CIGARs match psnw exactly, including
tie-breaking in repeats. That agreement is frozen in
`tests/data/psnw_frozen.json`.
