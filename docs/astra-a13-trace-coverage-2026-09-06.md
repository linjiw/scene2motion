# A13 archive diagnostic: early observation covers only part of the divergence

A new CPU analysis reconstructs the first differing state in all39 A12 pairs with
identical prepared target inputs. It confirms all prior exactness and onset labels.
Of29 differing executions,15 diverge inside the frozen eight-step A13 trace and14
diverge later. This keeps the current early stress probe useful, but makes it
insufficient for validating a fix across the full cohort.

This is post hoc analysis of existing absent-obstacle executions. No new
simulation, generation, independent unit, physical contact, or traversal result
is added. The mechanism remains unresolved.

| Archived subgroup | Pairs | Exact valid prefix | Diverged in first8 steps | Diverged later |
| --- | ---: | ---: | ---: | ---: |
| Group0 | 25 | 7 | 15 | 3 |
| Group1 | 14 | 3 | 0 | 11 |
| G1 encoder | 18 | 5 | 8 | 5 |
| Teleop encoder | 21 | 5 | 7 | 9 |
| All pairs | 39 | 10 | 15 | 14 |

Group and encoder rows are different partitions of the same39 pairs; do not add
their denominators. None is a held-out population or an independent simulation
repeat. Encoder modes and command/observation masks match across each entire
paired valid prefix.

![First execution differences across the archived target pairs](../outputs/astra_a13_trace_coverage_v1/onset_coverage.png)

Filled triangles mark the first differing state. Open circles mark the end of an
entirely exact valid prefix, including short terminated prefixes. The shaded
interval ends at0.16s, the current observer window. Array sample0 is post-step at
0.02s; sample1 is0.04s. Eight group1 pairs first differ at sample206, or4.14s,
including four G1 and four teleop rows. Synchronized state differences do not
identify an inference, reset, or physics cause.

All29 first differing states include joint-position differences. Fifteen also
have differing root positions and quaternion components at that sample; nine
have quaternion-component differences but identical root positions; five differ
only in joints initially. These are exported position components, not captured
actions, velocities, contact events, or proof of different physical orientation.

## Evidence and reproducibility

- Driver: `experiments/diagnose_astra_a13_trace_coverage.py`.
- Result: `outputs/astra_a13_trace_coverage_v1/summary.json`.
- Figure receipt, PNG and PDF are in that same directory.
- Validates previous result hashes, completed archive/process/verification hashes,
  clocks, row identities, endpoint equality and matched stable encoder ledgers.
- Tests cover one-ULP changes, sample7 versus sample8 coverage, sample-to-time
  conversion, short exact prefixes, nonfinite/padded input refusal and encoder drift.
- Focused observer/coverage tests:22 passed. Complete current CPU suite:1279
  passed in216.06s. The figure was visually inspected after rendering.
- Independent rebuild exactly matches the saved analysis. The first plotting
  attempt used ARDY Python without matplotlib; the unchanged completed analysis
  was rechecked and rendered with the documented Isaac figure interpreter.

Rebuild into a fresh output directory with the figure interpreter; this invokes
no simulator:

```bash
source env.sh
env LD_LIBRARY_PATH= OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/s2m_a13_mpl \
  /home/linjiw/isaaclab-install/env_isaaclab/bin/python \
  experiments/diagnose_astra_a13_trace_coverage.py \
  --source outputs/astra_a12_progress_correction_v1 \
  --diagnostic outputs/astra_a12_batch_context_v1/summary.json \
  --out outputs/astra_a13_trace_coverage_rebuild
```

## Subsequent validation cases, selected from existing outcomes

The following operational cases cover both encoder modes and early, late and
unchanged execution. They are explicitly outcome-selected development cases,
identified before any new apparatus fix. This table is a design input, not a
new simulator budget or a preregistered confirmation experiment.

| Purpose | Group/slot | Unit | Encoder | Archived first difference |
| --- | --- | --- | --- | --- |
| Early stress, existing A13 target | 0/19 | s04/u3 | G1 | 0.04s |
| Early, other encoder | 0/9 | s02/u1 | Teleop | 0.04s |
| Late | 1/2 | s08/u2 | G1 | 4.14s |
| Late, other encoder | 1/0 | s08/u0 | Teleop | 4.14s |
| Exact control | 0/1 | s00/u1 | G1 | Exact through8.26s |
| Exact control, other encoder | 0/12 | s03/u0 | Teleop | Exact through8.26s |

Resume the unchanged A13 campaign first. Require its original full-execution
equality before interpreting its early trace. Any resulting apparatus change
must then face late-window observation and full-prefix checks on both groups;
an early-only improvement cannot establish stable comparisons across the cohort.
The eventual validation must bind intervention, windows, exactness criterion,
budgets and same-context controls before launch. Keep all39 pairs in final
apparatus-level reporting rather than treating these six cases as sufficient.

## Resumption status

The second admission attempt again stopped before launch: available RAM7796MiB
versus9216 required, free VRAM3868MiB versus5500 required. Both refusal receipts
are preserved in `outputs/astra_a13_batch_observer_v1`. The frozen checkout and
64-execution budget remain unchanged; zero A13 seeds have been spent. No poller
was installed and no co-tenant workload was interrupted.
Continuation validation is recorded in `outputs/astra_a13_continuation_validation_v1/summary.json`.
