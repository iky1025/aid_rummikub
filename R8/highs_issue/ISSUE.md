# HiGHS 1.13.1: MIP presolve returns suboptimal solution declared optimal (gap 0.0)

> Suggested venue: comment on ERGO-Code/HiGHS **#2806** ("HiGHS 1.13 Terminates MIP
> Prematurely...") as an additional minimal reproduction, since the symptom matches
> and that issue is open. Fixed in 1.14.0 for this instance, so the main value is
> as a regression test case.

## Summary

On the attached 131-variable / 86-constraint maximization MIP with general
integer variables (bounds 0..2), HiGHS 1.13.1 with default presolve returns
objective **-58** with model status `kOptimal` and `mip_gap = 0.0`. The true
optimum is **-59** (this is a minimization objective after sense conversion;
-59 is better). With `presolve=off`, HiGHS 1.13.1 finds -59 correctly.

The returned suboptimal solution is declared optimal with zero gap, so the
error is silent — we only caught it because our application cross-checks the
solver against an independent exact dynamic-programming solver at runtime.

## Environment

- HiGHS 1.13.1 via `highspy` 1.13.1 (conda-forge, build `np2py311hb7ce6e1_0`)
- macOS 15 (Darwin 25.5.0), Apple M4 (arm64)
- Python 3.11

## Reproduction

Files: `repro.mps` (model, written by `Highs::writeModel` itself), `reproduce.py`.

```text
$ python reproduce.py
HiGHS version: 1.13.1
presolve=on : status=HighsModelStatus.kOptimal, objective=-58.0, mip_gap=0.0
presolve=off: status=HighsModelStatus.kOptimal, objective=-59.0, mip_gap=0.0
```

The same disagreement occurs whether the model is built through the C++ API
(`addCol`/`addRow`, via PuLP's `HiGHS` interface) or loaded from the attached
MPS/LP file, and with any thread count.

## Expected

Both settings report the optimum -59.

Cross-validation of -59:

- COIN-OR CBC on the same model: -59.
- An independent exact dynamic-programming solver for this problem class
  (Rummikub single-turn optimization) with a fully validated certificate
  (the DP's arrangement was verified feasible by direct constraint checking):
  agrees with -59.

## Version matrix

| Solver / setting | Objective | Status |
| --- | --- | --- |
| HiGHS 1.13.1, presolve=on | **-58 (wrong)** | kOptimal, gap 0.0 |
| HiGHS 1.13.1, presolve=off | -59 | kOptimal |
| HiGHS 1.14.0, presolve=on | -59 | kOptimal |
| HiGHS 1.14.0, presolve=off | -59 | kOptimal |
| CBC 2.10 | -59 | Optimal |

## Model background (optional)

The model is a set-partitioning-style MIP from a Rummikub (tile game) move
optimizer: integer variables x_i in [0, 2] select tile sets (a set can appear
twice because every tile exists in two copies); constraints enforce per-tile
availability upper bounds and lower bounds forcing all table tiles to be
covered. The wrong presolve-on solution under-covers the mandatory tiles'
aggregate by one unit relative to the optimum structure.

## Notes

- Not reproducible on 1.14.0 (tested via conda-forge highspy 1.14.0):
  presumably fixed by one of the MIP presolve changes in that release.
- We are staying on 1.13.1 with `presolve=off` as a workaround until 1.15.x
  reaches conda-forge (1.14.0 has a separate reported MIP issue, #2957).
