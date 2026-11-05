# Cost model

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

Defaults match psnw: match 6, mismatch 18, gap open 30, gap extend 5.

For the DP walk-through, microhomology shifts, profile MSA, and the full
list of differences from upstream psnw, see {doc}`algorithm`.
