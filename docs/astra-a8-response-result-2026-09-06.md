# A8 result and the next execution experiment

A8's baseline-anchored linear response **fails its preregistered readiness gate**.
The execution-phase model does not improve clearance or continuation prediction over
copying the initial achieved outcome. No optimizer expansion, new generation, present
method comparison or held-out confirmation was run. A7's negative result is unchanged.

The [protocol](astra-a8-paired-change-protocol.md) and implementation were committed
at `c850565`. The existing [result](../outputs/astra_a8_paired_change_v1/summary.json)
has SHA256 `ebc23bfd4de3b365a1390be8408ac624fdce8aab12113df0d241024377396d92`.
On September6 the complete evaluation was rebuilt in memory and matched every result
field; all bound source hashes matched. The separate midpoint diagnostic also rebuilt
exactly, and figure source/artifact hashes matched. These existing local artifacts
were inspected without rewriting them.

## Paired response prediction

All12 reused development scenes and48 units enter12 leave-scene-out folds.
Each omitted scene contributes four units and six edited assignments:288 total.
Only22/48 initial previews have all three numeric targets. Initial statuses are
22 observed,12 no-arrival,12 exit-censored and2 recovery-censored. Numeric targets
use independently available pairs; status scores retain all288 assignments.

| Predictor | Clearance change MAE (mm) | Exit change MAE (s) | Continuation change MAE (mm) | Status Brier |
|---|---:|---:|---:|---:|
| No change | 62.3 | 0.711 | 177.9 | 0.806 |
| Constant response | 61.7 | 0.744 | 179.0 | 0.640 |
| Scene/reference response | 67.0 | 0.755 | 197.0 | 0.602 |
| Execution-phase response | 66.8 | 0.732 | 187.7 | 0.554 |

MAE averages scene means over available scenes: clearance193/288 pairs in12 scenes;
exit110/288 in11; continuation97/288 in11. Missing scenes are not scored as zero.
Brier sums squared errors over four status categories, then averages over288.
The learned controls all use training execution labels and a measured initial anchor;
the scene/reference model removes execution context from response coefficients.

Execution context reduces status Brier and predicts the nonzero continuation direction
correctly on50/97 available pairs. These partial successes do not compensate for
failed numeric checks. Against no-change, its scene-mean error increases by4.48 mm
for clearance (scene-bootstrap95% interval −0.16 to8.41 mm) and9.82 mm for continuation
(−10.62 to32.08 mm). These describe reused development data, not fresh confirmation.

![A8 response comparison](../outputs/astra_a8_figures_v1/paired_response.png)

## Response shape and duplicate-reference variation

The [post hoc midpoint diagnostic](../outputs/astra_a8_linearity_v1/summary.json)
measures `abs((plus + minus)/2 - baseline)`. A linear response through baseline has
zero midpoint departure. Matched reference/achieved clearance medians are2.19/30.58 mm
for depth (27 triplets),7.35/41.74 mm for onset (27), and0.84/16.27 mm for recovery (33),
each out of48 assigned units. Observation status differs within28/48 depth,35/48 onset
and21/48 recovery triplets. Three observations do not identify the cause, prove
discontinuity, or establish behavior at a smaller command step.

The new [duplicate-reference analysis](astra-a6-padding-repeat-analysis-card.md)
and [receipt](../outputs/astra_a6_padding_repeats_v1/summary.json) inspect all128
scientific/padding pairs already executed in A6 group1, with zero new simulation.
They verify receipts, runtime modes and original scientific measurements before
comparing slots. Padding stays excluded from training and independent-unit counts.

Among112 native pairs,70 pairs from10 units across4 scenes share the realized encoder;
42 do not. In the70 matching-mode pairs,20 have different observation status,20 have
different termination labels, and4 disagree on the existing development descriptor.
No complete qpos pair is byte-identical. Available numeric differences are:

| Target | Both observed / assigned | Median absolute difference | Maximum |
|---|---:|---:|---:|
| Inflated clearance | 43/70 | 31.8 mm | 417.8 mm |
| Rear exit | 23/70 | 0.200 s | 2.880 s |
| One-second continuation | 22/70 | 72.3 mm | 337.2 mm |

Each numeric row covers3 of4 scenes; the receipt includes per-scene counts and means.
The10 matching-mode free-WALK pairs are separate: zero status/descriptor disagreements,
two termination disagreements; clearance differences have median4.89 mm on7/10 pairs.
These are correlated pair counts, not new success rates or independent trials.

Matching encoder masks is insufficient to make different slots interchangeable.
Slot origins, initial conditions, observation noise and other effects remain potential
explanations. These pairs cannot isolate same-slot nondeterminism or quantify a
same-slot noise floor, and do not invalidate A6's corrected same-slot command pairing.
Do not subtract these discrepancies from A8 error or use them as safety margins.

## Decision

The next bounded experiment is [A9 identical-slot replay](astra-a9-same-slot-replay-protocol.md):
repeat the original baseline and free-WALK jobs,32 slots each, with frozen inputs and
runtime. If exact replay holds, separately measure smaller native steps with no-edit
repeats. If it does not, characterize repeat variation before fitting a deterministic
inverse. Neither result alone licenses confirmation.

The [ICRA closure plan](icra-constructive-closure-2026-09-06.md) keeps duck correction
as the primary constructive hypothesis, positive G1 contact sensitivity as an open
measurement requirement, and stepping/downstream training as conditional extensions.
