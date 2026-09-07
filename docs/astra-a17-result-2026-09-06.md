# A17: depth-only correction does not extend its pilot lead

September 6, 2026. **Closed at the preregistered discriminating gate.** Both new
candidate previews failed qualification on s01/u2. With the other35 selected
comparisons proven equivalent, the selected net gain over fixed shallow on the
remaining36 reused units is zero. The strict development lead gate fails.

This is a two-job decision, not a fully executed36-unit candidate benchmark.
Absolute edited-method pass counts and rates over36 remain unknown. A16's5/12
versus4/12 pilot result remains valid and separate; it is not a generalization result.

[Protocol](astra-a17-remaining-development-protocol.md) ·
[Result receipt](../outputs/astra_a17_remaining_development_v1/summary.json) ·
[Independent audit](../outputs/astra_a17_completion_v1/summary.json).

## The measured comparison

Same frozen ARDY-G1 references, SONIC runtime, nominal32-slot background, physics
seed0 and original group0/slot6. Scene s01, unit u2, generation seed64006.
Only native depth differs; onset lead1.08 m and recovery offset1.38 m are retained.
The obstacle is absent in these simulations.

| Control | Requested depth | Inflated clearance | Exit | Forward continuation,1 s | Qualified |
| --- | --- | --- | --- | --- | --- |
| Cached nominal |0.28 m|−29.51 mm|2.30 s|0.553 m|No: margin |
| New fixed shallow |0.26 m|−38.79 mm|2.22 s|0.613 m|No: margin |
| New depth only |0.30 m|−19.17 mm|2.62 s|0.370 m|No: margin and progress |

Both new runs finish tracking with the complete recovery suffix, upright and in
the corridor. The deeper candidate improves the clearance margin by19.62 mm
relative to shallow, but loses242.65 mm of continuation and misses the original
0.45 m requirement. Even its improved inflated clearance remains negative.
All three nominal geometric clearances are positive; the failed40 mm-inflated
margin must not be described as an observed physical collision. Contact is unknown.

## Why two jobs settle the incremental lead

The frozen input inventory covers scenes0–11, units1–3:36 previously used development
units and108 arm requests. Thirty depth-only payloads are exactly their same-unit
shallow payloads in the same runtime context. Five depth-only payloads equal their
nominal payloads, and independent raw-state reconstruction confirms all five
nominal previews qualify. Both selectors therefore preserve nominal on those five.

These35 pairs have equal selected qualification labels irrespective of the
unknown shared candidate outcomes. The sole distinct pair is s01/u2; both
selectors abstain there. Net selected difference is thus0 over the assigned36.
This is an input-equivalence argument plus one measured pair, not36 measured ties.
No population confidence interval or absolute edited-method rate is claimed.

The gate stopped the remaining35 jobs without spending their launches. Their
frozen job files and payloads remain preserved. No threshold, assignment, slot,
source motion or resource gate was relaxed. There is no pooled48-unit success rate.

## Verification and cost

The clean preparation checkout is `/tmp/s2m-a17-b1842f8`; simulator runtime remains
`/tmp/s2m-a15-88fcc45`. Protocol/implementation were committed before dispatch.
An earlier import-path preparation failure at73017ef occurred before output creation
or simulation; its receipt is preserved. The fix resolves preparation helpers from
their own checkout without changing the simulator environment.

The independent audit checks all32 reference fields in each of the37 prepared
jobs, reconstructs36 nominal measurements plus the two distinguishing candidates
from raw achieved states, checks original runtime arguments and archive hashes,
and independently recovers the35-pair equivalence and failed gate. Re-running
analysis must reproduce the saved summary exactly.

Two new32-environment jobs mean64 operational executions:2 useful scientific
previews and62 context-workload executions. There are no new generations,
independent units, obstacle-present trials or held-out scenes. Simulator-process
time sums to132.119 s; peak child RSS6068.148 MiB; minimum available RAM7332 MiB.
No resource refusal, timeout or abort occurred. Preparation/analysis time is excluded.
The full CPU suite passes1332 tests in204.89s; focused campaign/page checks pass46.
Analysis rebuild is exact, and the scientific figure and desktop/mobile pages were
visually inspected. [Completion validation](../outputs/astra_a17_validation_v1/summary.json).

## Research decision

Close this frozen depth-only correction rule on the reused development regime.
The positive A16 case motivates joint clearance/progress measurement, but the
remaining comparison does not support incremental superiority over fixed shallow.
Do not tune the same rule again on these units or call unexecuted candidates failures.

The next stage is controlled G1 contact sensitivity and a separately preregistered
obstacle-present comparison of the strongest supported native compiler. Any new
correction needs a new mechanism and fresh development protocol. Hold out scene
geometries only after method freeze; multi-beam routes and carrier-specific stepping
remain distinct extensions. See the [paper framing and evidence map](astra-icra-research-story-2026-09-06.md).
