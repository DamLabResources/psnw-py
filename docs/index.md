# psnw-py

Position-specific Needleman–Wunsch in Python (+ optional numba), after
[Heng Li's psnw](https://github.com/lh3/psnw). Every target base can carry its
own gap-cost discount. The package also builds reference-anchored multiple
sequence alignments.

```{toctree}
:maxdepth: 2

install
algorithm
cost_model
pairwise
msa
applications
api
```

## Quick example

```python
import psnw_py

target = "ACGTTGCAAGGCTTAC" + "GATCGATTGCATGCAA" + "TCAGGCATCGGATACC"
query = "ACGTTGCAAGGCTTAC" + "TCAGGCATCGGATACC"
pse = [0] * 16 + [5] * 16 + [0] * 16

aln = psnw_py.align(target, query, pse=pse)
aln.score, aln.cigar, aln.md
# (162, '16M16D16M', '16^GATCGATTGCATGCAA16')
```

Install with `pip install psnw-py` (add `[numba]` for the fast backend).
How the DP works, and what `pse` / `pso` mean, is in {doc}`algorithm`.
The compact cost table is in {doc}`cost_model`.
