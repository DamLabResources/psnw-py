# Applications

The library is an aligner. Other tools supply the biology.

Position-specific gap costs are useful whenever a mapper has already found a
locus and a short global realignment should prefer a known indel pattern:

- Set `pse = gap_extend` on a reference interval to make deleting that
  interval cost one gap open (for example a known cut-site deletion).
- Splice an expected insertion into the target and put the same subsidy on
  those bases so matching the insertion is preferred over a free-form gap.
- An interval between two sites is the same subsidy on a longer window.

SAM/BAM rewriting, cut catalogs, and long-read slicing belong in those
downstream tools, not in this package.
