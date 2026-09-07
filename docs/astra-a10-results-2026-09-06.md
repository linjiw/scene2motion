# Smaller native edits improve candidate supply, but response prediction still fails

A10 completed the preregistered half-step study at `fe44586`: 384 generated
references and 512 SONIC obstacle-absent executions across all 12 development scenes
and 48 existing motion units. A 2 cm shallower native crouch passes the development
clearance-and-progress descriptor on 14/48 units, versus 7/48 unchanged. The
execution-phase predictor nevertheless fails its frozen readiness gate. This is a
promising simple command baseline, not demonstrated execution-aware method gain,
obstacle-present traversal, or transfer to unseen scenes or other motion families.

Protocol: [A10 preregistration](astra-a10-half-step-protocol.md).
Evidence: [complete analysis](../outputs/astra_a10_analysis_v1/summary.json),
[operational audit](../outputs/astra_a10_half_step_v1/completion_audit.json),
[group 0 measurements](../outputs/astra_a10_half_step_v1/response_group0.json),
[group 1 measurements](../outputs/astra_a10_half_step_v1/response_group1.json).

![A10 prediction and candidate supply](../outputs/astra_a10_figures_v1/half_step_response.png)

[Standalone PDF](../outputs/astra_a10_figures_v1/half_step_response.pdf).
Every count below retains all assigned units unless an observed-target denominator
is explicitly given. The scene grid is position (1.2, 1.6, 2.0 m) × underside
(1.10, 1.20 m) × longitudinal length (0.24, 0.60 m), four existing seeds per scene.
The descriptor requires complete passage/recovery observation, the existing exit
deadline, at least 0.45 m continuation in one second, corridor/upright checks, and
no nominal or 40 mm inflated geometry violation through recovery. It is unchanged
from A6, and differs from the historical A5 local-passage endpoint.

## What improved in the executed candidate pool?

| Native command | A6 full edits: passes /48 | A10 half edits: passes /48 |
|---|---:|---:|
| Unchanged | 7 | 7 |
| Shallower depth | 8 | 14 |
| Deeper depth | 3 | 7 |
| Smaller onset lead | 5 | 10 |
| Larger onset lead | 7 | 8 |
| Earlier recovery | 6 | 7 |
| Later recovery | 7 | 8 |
| Free WALK | 0 | 0 |

The new edit sizes are depth ±0.02 m and onset/recovery ±0.09 m. Baseline native
parameters remain (0.28, 1.08, 1.38) m. Shallower depth therefore requests 0.26 m.
Its observed 29.2% versus 14.6% baseline rate is a 14.6 percentage-point difference.
Post hoc paired inspection finds eight gains and one regression, with six units
passing both commands. Six scene counts improve; six tie. Equal scene counts can
hide opposing unit-level outcomes. The smaller-onset-lead comparison gains four and
loses one. Highlighting these commands after observing all variants is exploratory;
the table does not establish a statistically confirmed selected policy.

| Scene | Position / underside / length (m) | Baseline /4 | Shallower /4 | Smaller onset lead /4 |
|---|---|---:|---:|---:|
| s00 | 1.2 / 1.1 / 0.24 | 0 | 1 | 1 |
| s01 | 1.2 / 1.1 / 0.60 | 0 | 1 | 1 |
| s02 | 1.2 / 1.2 / 0.24 | 2 | 3 | 3 |
| s03 | 1.2 / 1.2 / 0.60 | 1 | 1 | 0 |
| s04 | 1.6 / 1.1 / 0.24 | 0 | 0 | 0 |
| s05 | 1.6 / 1.1 / 0.60 | 0 | 0 | 0 |
| s06 | 1.6 / 1.2 / 0.24 | 1 | 2 | 2 |
| s07 | 1.6 / 1.2 / 0.60 | 1 | 1 | 1 |
| s08 | 2.0 / 1.1 / 0.24 | 1 | 1 | 1 |
| s09 | 2.0 / 1.1 / 0.60 | 0 | 0 | 0 |
| s10 | 2.0 / 1.2 / 0.24 | 1 | 2 | 1 |
| s11 | 2.0 / 1.2 / 0.60 | 0 | 2 | 0 |

The outcome-informed seven-command ceiling rises from 16/48 to 19/48. That ceiling
uses all seven outcomes; it is not a two-preview deployable method. Free-WALK's zero
is its result against the virtual beam task, not a free-space walking failure rate.

## Does the execution response become predictable?

Twelve scene-exclusion development folds use the frozen A8 feature map, ridge
penalty and gates. Physical half-step coordinates are explicitly normalized to
the same unit-step feature basis; records preserve the actual native commands.

| Predictor | Clearance MAE (mm) | Exit MAE (s) | Continuation MAE (mm) | Status Brier |
|---|---:|---:|---:|---:|
| No change | 56.47 | 0.6232 | 137.82 | 0.6875 |
| Constant response | 56.22 | 0.6361 | 134.46 | 0.6408 |
| Scene/reference | 61.91 | 0.6126 | 143.89 | 0.5720 |
| Execution phase | 65.01 | 0.6298 | 136.59 | 0.5102 |

Numeric errors are means of available scene MAEs. Clearance has 194/288 observed
initial/edit pairs in 12 scenes; exit 114/288 and continuation 103/288 in 11 scenes.
The unavailable counts are 94, 174 and 185 respectively. Status Brier retains all
288 edits and sums squared error across four categories. There are 22 complete
initial numeric previews. No missing numeric observation is replaced by zero.

Execution information improves status prediction and gets 54/103 continuation
effect signs right. Exit error stays within 5% of no change. But clearance loses
to every comparator, and continuation loses to constant response: four of ten
readiness checks fail. Execution-minus-no-change clearance MAE is +8.54 mm,
scene-bootstrap 95% interval [+0.41, +19.03] mm. Its continuation difference is
−1.23 mm [−26.16, +20.96] mm. These reused development folds cannot establish
untouched generalization. All scene errors and comparison intervals are archived.

## Do smaller commands reduce nonlinear response?

| Axis | Matched units /48; scenes /12 | Median achieved clearance midpoint departure, full → half (mm) | Scene-mean half−full difference, 95% interval (mm) |
|---|---|---|---|
| Depth | 25; 12 | 18.04 → 12.27 | −34.96 [−93.97, +8.08] |
| Onset | 22; 11 | 39.77 → 13.86 | −16.74 [−24.42, −9.26] |
| Recovery | 32; 12 | 16.26 → 11.97 | +1.22 [−10.87, +15.01] |

Midpoint departure is `abs((plus + minus)/2 − baseline)`, evaluated on the same
independently observed triplets at both scales in reference and execution. The
omitted unit counts are 23, 26 and 16. Median and scene-mean summaries need not
move together. The bootstrap uses 10,000 scene resamples and seed 69000.

Smaller onset edits reduce midpoint departure, but also reduce their median
achieved half-plus-minus clearance response from 12.80 to 8.63 mm. For depth that
response shrinks from 48.26 to 22.83 mm; recovery is 4.88 to 5.32 mm. Reduced
nonlinearity alone does not yield useful inverse control. Across all 48 units,
status-switch counts fall from 28→23 for depth, 35→30 for onset and 21→16 for
recovery. The complete continuation/exit comparisons and missingness remain in
the analysis; there is no complete-case-only success claim.

## Validation, costs and decision

All 96 unchanged generation controls match A6 exactly. All 128 unchanged execution
slots, including padding, match original qpos valid prefixes, valid lengths and
termination labels exactly. Apparatus, encoder modes and process checks pass for
all 16 jobs. The analysis rebuild equals the saved summary exactly. Figure inputs
and output hashes verify; the generated figure was visually inspected.

The 512 actual executions comprise 384 scientific previews and 128 padding
duplicates. The 384 generations comprise 288 edited candidates and 96 controls.
The sum of simulator-job elapsed times is 838.37 s; peak child RSS is 6105.92 MiB.
Generation batch decode time sums to 27.18 s, excluding model loads, geometry
measurement, queueing and resource refusals. It is not end-to-end compilation time.
Three generation resource refusals were preserved; unchanged resumes only consumed
untouched batches. All simulator jobs completed. Full CPU tests: 1245 passed in
206.67 s; focused response/collector tests: 35 passed in 2.42 s. There is no running
campaign or admission poller. Keep `/tmp/s2m-a10-fe44586` frozen.

**Decision:** close A10 with failed predictor readiness. Do not expand this inverse
model, tune its features on these results, or spend confirmation seeds. Carry the
0.26 m fixed-depth command forward as a stronger simple baseline. The next mechanism
design should use body/phase-resolved clearance and progress observations rather
than infer an inverse from a scalar minimum; whether body/time switches explain
the residual response is still a hypothesis. Its first deliverable is a bounded
actual-correction design with baseline preservation and its own final preview,
compared against the fixed shallower command and equal-budget search. Before
obstacle-contact claims, complete a controlled positive G1 contact sensitivity
test. Freeze any changed model and its separate development protocol before more
generation; retain A6/A10 source groups together in every split.

The [motion/scene expansion plan](motion-scene-expansion-2026-09-06.md) keeps
single-beam confirmation, multiple beams, stepping and free-locomotion regressions
as separate evaluation obligations. None is completed by this absent-preview study.
