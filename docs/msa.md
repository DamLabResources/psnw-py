# Reference-anchored MSA

```python
from psnw_py import msa

# (a) star: merge existing placements (name, seq, pos0, cigar); no DP
m = msa.star(ref, [("r1", seq1, 100, "50M3I47M"), ("r2", seq2, 98, cigartuples)])

# (b) profile: progressive alignment to REF + reads added so far
m = msa.profile(ref, [("r1", seq1), ("r2", seq2)], pse=pse, order="length")

print(m.to_fasta())
m.ref_pos                    # 0-based reference coordinate, -1 for insertions
m.insertion_columns(anchor)  # insertion columns in front of one reference base
m.placements["r1"]           # (pos0, cigartuples) against the reference
```

**Star** merges placements without DP. Insertions in front of the same
reference base share one block of columns. Result is order-independent.

**Profile** adds reads one at a time with the psnw kernel. Reference columns
keep `pse`. Insertion columns from earlier reads can be matched or skipped
for free. New insertions before column `x` pay `pso` of the next reference
column. `free_ends=True` (default) makes reference overhang free.
