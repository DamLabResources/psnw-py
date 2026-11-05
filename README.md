# psnw-py

Position-specific Needleman–Wunsch in Python (+ optional numba), after
[Heng Li's psnw](https://github.com/lh3/psnw). Every target (reference) base
can carry its own gap-cost discount. The package also builds
reference-anchored multiple sequence alignments.

```bash
pip install psnw-py
pip install 'psnw-py[numba]'   # recommended
```

```python
import psnw_py

target = "ACGTTGCAAGGCTTAC" + "GATCGATTGCATGCAA" + "TCAGGCATCGGATACC"
query = "ACGTTGCAAGGCTTAC" + "TCAGGCATCGGATACC"
pse = [0] * 16 + [5] * 16 + [0] * 16   # free deletion extension on the window

aln = psnw_py.align(target, query, pse=pse)
aln.score, aln.cigar   # (162, '16M16D16M')
```

Source: [github.com/DamLabResources/psnw-py](https://github.com/DamLabResources/psnw-py).
Docs: [psnw-py.readthedocs.io](https://psnw-py.readthedocs.io) (or `make docs`
from a checkout).

Separate tools use these gap costs for cut-site realignment after a mapper
has placed the read. This package is the aligner only.

```bash
make test      # 100% line coverage of psnw_py
make release   # test + docs + sdist/wheel (does not upload)
```
