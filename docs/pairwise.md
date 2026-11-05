# Pairwise API and CLI

```python
import psnw_py

aln = psnw_py.align(target, query, pse=pse, pso=pso)
aln.score, aln.cigar, aln.md, aln.nm
aln.cigartuples          # pysam-style (op, length) tuples
aln.shifts()             # equal-score microhomology shifts
aln.placements()         # [aln, *shifts]

psnw_py.score_cigar(target, query, "16M16D16M", pse=pse)

# semi-global: target overhang is free
psnw_py.align(target, query[:20], free_target_ends=True)
```

`pso` / `pse` are integer arrays (lists or numpy). Digit strings (`"0005550"`)
are accepted for compatibility with the psnw TSV format.

## CLI

```bash
psnw-py input.tsv
```

Input lines: `name target query [pso [pse]]`. Output follows psnw (`SQ`,
`CG`, `//`). Flags `-A -B -O -E -a` work as in psnw. By default every gap is
shifted and only equal-score shifts are listed; `--all-shifts` and
`--longest-gap-only` turn on the psnw behaviour.
