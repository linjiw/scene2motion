# A12: actual bounded progress-first native correction

**Status: preregistered**. September 6, 2026. Commit collector, correction, analysis
and this protocol before generation. This separate development pilot is authorized
by the user's instruction to advance the next research stage. It does not reopen
the failed A8/A10 learned inverse or consume confirmation seeds.

## Mechanism and fixed decisions

A11 reconstructs all 672 native reference/execution traces. Body switching is rare
(1/30, 0/25 and 1/33 observed depth/onset/recovery triplets); fixing the primitive
and progress coordinate does not consistently reduce midpoint departure. Do not
fit a larger body-conditioned inverse on that premise. A10's fixed shallower depth
improves candidate supply, motivating a directly tested progress-first alternative.

Only the initial unchanged absent preview and its limiting inflated-geometry event
may inform each proposal. No edited trajectory/outcome, seed or slot determines
the proposed command. The baseline is native (depth, onset lead, recovery offset)
=(0.28, 1.08, 1.38) m. Keep route, prompt, ramps, speed and frozen ARDY/SONIC fixed.

1. Preserve a passing baseline during final selection. The operational paired
   campaign still generates and measures both proposed variants for every unit;
   this surplus work counts, including proposals subsequently unused.
2. If progress is unobserved, exit misses the existing deadline, continuation is
   below 0.45 m in one second, or the existing corridor/upright check fails, reduce
   depth by 0.02 m. Earlier recovery by 0.09 m is permitted only with a fully
   observed exit and slow continuation suffix. Do not impute exit for truncations.
3. Otherwise, for an observed inflated-margin violation whose limiting normal has
   z component at least 0.5, deepen by 0.02 m. If that event's root is at/before the
   beam centre, advance onset by increasing its lead 0.09 m; otherwise postpone
   recovery by 0.09 m. This coarse approach/departure cue is a hypothesis, not a
   calibrated phase-transfer estimate or dynamic-feasibility guarantee.
4. Remaining failures use the fixed 0.02 m shallower fallback.

The no-phase ablation uses identical rules and depths, omitting every onset/recovery
change. There is no response-model training, penalty selection, or automatic claim
that the proposal is qualified. Final selection preserves a qualified baseline,
otherwise requires the exact candidate's frozen A6 development descriptor; refuse
if neither passes. Present outcomes never enter selection.

## Fixed corpus, implementation controls and costs

Use all 12 A6/A10 scene geometries and 48 existing development seeds 64000–64047;
retain source groups `astra-a6/sXX/uY`. Zero new independent units. New generation
uses the same 8-row unit batches as A10, replacing its depth-minus slot with the
full correction and depth-plus slot with the no-phase ablation. The other six
rows are unchanged controls. Preserve keys' original slot suffixes in provenance;
the explicit `variant` and `proposal` fields name the actual A12 intervention.

48 batches ×8 =384 generations:96 proposal requests and288 no-edit controls.
All six unchanged qpos arrays per batch must exactly match A10. Preserve arrays,
start/complete receipts and any failure; a mismatch stops collection without retry.
Require identical generator/runtime/model identity and v2 noise. ARDY Horizon52,
208 frames,25 Hz,5 diffusion steps,CFG[2,2], WALK prompt and0.9 m/s route stay fixed.

Paired simulation uses the A6/A10 default encoder slots and32-env compact profile.
For each group, replay baseline and free-WALK first, then correction and no-phase.
Group0 has32 scientific slots; group1 has16 scientific and16 padding duplicates.
Eight launches =256 executions,192 scientific previews and64 padding. Of these,
96 scientific previews evaluate the two proposal arms;96 are unchanged controls.
All128 operational baseline/free control slots must exactly match A10 qpos/length/
termination and modes. Stop before new proposals on a control mismatch.

Each deployable arm has at most initial+one candidate and two absent previews;
batch-shape control overhead and surplus preview work are reported separately and
charged in total. The fixed shallower and search comparators reuse their complete
A10 development previews in matching slots. They are not new executions or new
data. New obstacle-present evaluations:0. New held-out confirmation evaluations:0.

## Comparators, endpoint and decision

Retain the exact A6 development geometry/progress descriptor, including all
non-arrivals, censored suffixes and refusals in the48-unit assigned denominator.
Compare unchanged baseline; baseline-preserving fixed shallower command; a
training-scene-selected one-edit search control from the six A10 half-step commands
(leave the evaluation scene out; ties follow original variant order); full A12;
and its no-phase ablation. The search has one initial and one selected edit preview,
not all seven at evaluation. No held-out generalization claim follows from these
reused development scenes, particularly as the mechanism was motivated by them.

Report raw candidate descriptor counts separately from selected/preserved results,
per-scene counts, paired wins/losses and10,000 scene-bootstrap draws with seed71000.
Keep branch/phase-action counts, unavailable progress observations, all simulator
terminations and actual costs. Timing contribution requires an improvement over
the ablation. Expansion requires full correction to exceed both fixed shallower
and equal-budget search selected counts, preserve baseline passes, and exceed its
no-phase ablation. A tie/failure closes this rule version: no threshold tuning or
confirmation inside A12. A pass motivates a separately preregistered physical
obstacle comparison with strong controls; it is not itself that closing result.

## Execution and resource contract

Fresh external `outputs/astra_a12_progress_correction_v1` from a clean pinned
checkout. Fresh analysis `outputs/astra_a12_analysis_v1`. Preserve A6–A11 archives.
Generation admission RAM8192/VRAM4096 MiB, rechecked between scenes. Simulation
admission RAM9216/VRAM5500 MiB, runtime floors2500/1200 MiB,900 s timeout, one ASTRA
lock. No co-tenant interruption or admission poller. Preserve resource refusals;
resume only untouched work or verified stages. Generator exits before simulation.
Respect output-local `STOP_AFTER_CURRENT_JOB`. Controlled positive G1 contact
sensitivity remains open; pose-distance traces are not physics contact forces.
