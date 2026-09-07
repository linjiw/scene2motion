# Scene2Motion: active research plan

**Updated 2026-09-07 following the user's new research guidance.** This plan supersedes
method closure in the September 1 plan and the two-backend/bank direction in the
[previous ASTRA plan](docs/astra-plan-through-2026-09-05.md). Historical endpoints and
receipts remain unchanged. Campaign-specific protocols govern already assigned work.

## Objective

Test a joint native height–forward-timing control space for single-beam ducking,
then build a bounded two-preview compiler only if candidate supply improves. Keep
ARDY-G1 and SONIC frozen in default mode. A17 closes the specific depth-only rule,
not native correction or depth adjustment in general.
The research question is whether measured execution response improves qualified local
passage with continued locomotion over strong scene-aware and budget-matched controls.
Call the system offline simulation-assisted compilation. Fitting a response model is
training; frozen generator/controller does not mean no training anywhere.

Fixed3/default has 17/32 historical A5 local passages; all17 exceed25 cm continuation
in the following second. The extra fixed2 pool outcome is nearly stationary. A4's
longer duck lost recovery successes. These observations motivate progress-constrained
command correction; they do not establish a new method gain or cross-scene transfer.

The user's September6 follow-up makes performance across motion families and scenes
the broader program objective. Follow the [motion/scene expansion sequence](docs/motion-scene-expansion-2026-09-06.md):
finish the native duck response/correction stage, retain free-locomotion controls,
then independently test held-out beam geometry, multiple beams and stepping. Avoid
controller-mode sweeps, fixed-pool selector training and new imitation dependencies
as substitutes for a constructive gain. Full7.2 m completion remains secondary to
approach–duck–clear–continue in the present duck study.

## Method and computational contract

The latest user guidance supersedes the old three-variable contract for new work.
Optimize only depth d and one forward-timing parameter beta; freeze spatial windows
from geometry and nominal achieved envelope events. Keep route endpoints, horizon,
prompt, checkpoints and controller mode fixed. Emit native root-position and
root-height constraints before generation; never time-rescale generated joints.
Historical A6–A17 protocols and their failed gates remain unchanged.

At deployment: initial generation+absent preview, one bounded correction+regeneration,
exact edited-candidate absent preview, then one present evaluation of the selected
output. Maximum two generated candidates and two previews; no present outcome may
influence command selection. Refusals count among all assigned attempts. A selected
output must have its own complete qualification measurement.

First measure joint-control candidate coverage on fresh development scenes. Do not
fit another response predictor until timing supplies useful new achieved outcomes.
Only then estimate a small scene-held-out execution response model.
Use achieved whole-body footprint overlap, clearance, rear-body exit and one-second
continuation. Preserve non-arrival and incomplete recovery. Require clearance, exit
and continuation separately; minimize edit size and duck burden within studied bounds.
Use grouped development residual allowances; these are empirical engineering estimates,
not safety certificates. Do not add the40 mm geometry inflation twice.

## Ordered work

**A18 now closed:** two of six jobs launched,64 instrument exposures. Contact
points on32/32 overlap placements, nonzero reported impulses0/32; exact observer
states. Force sensitivity unresolved; specificity/capacity doubling untested.
See [bounded result](docs/astra-a18-result-2026-09-07.md). No force-based clearance claim.

**A19 interrupted after verified prefix:** [protocol](docs/astra-a19-joint-control-protocol.md),
preparation `/tmp/s2m-a19-b7f9733`, outputs `outputs/astra_a19_joint_control_v1`.
Eight fresh geometries,16 carriers,176 scientific absent previews; maximum162
32-env jobs including a fresh exact nominal replay,5184 operational rollouts.
Nominal4/16,32 exact replay slots; two shallow targets verified. Fifth job aborted
at2495MiB RAM below2500MiB floor; no retry, joint previews0.
[Prefix result](docs/astra-a19-prefix-2026-09-07.md).
Nine native commands per unit; historical nominal/shallow remain exact comparators.
Do not fit a predictor before the repeated timing-supply gate passes.
[Updated claim–evidence plan](docs/astra-icra-joint-control-plan-2026-09-07.md).

**Current assignment from the new guidance:** close A18 as a six-job instrument
exercise, then A19 candidate coverage on8 fresh geometries×2 seeds×(9 joint controls
+2 exact historical command comparators). The new control module separates spatial
height from forward timing and preserves the old native endpoint arrays. A18 maps
three selected colliders among30 bodies/45 colliders; its detection claim will be
limited to tested surfaces. A19 may run independently of contact calibration because
its obstacle is absent. A19's9-candidate discovery search is not two-preview method
evidence. A later present capability bridge uses up to48 target executions; a frozen
practical rule and prospective scene pilot come only after useful supply is shown.
The best fixed joint command and matched-space search must remain strong comparators.
Preserving nominal never guarantees preserving unseen fixed-shallow successes.


**A10 complete; predictor readiness failed.** [Results](docs/astra-a10-results-2026-09-06.md):
384 generations and512 absent executions across12 scenes/48 units;96 generation
controls and128 simulation control slots exactly match A6. The0.26 m fixed-depth
command passes the development descriptor on14/48 versus baseline7/48, with eight
paired gains and one loss. It is a stronger simple baseline, not a selected-method
or present-traversal result. Execution-phase clearance MAE65.01 mm loses to no-change
56.47 mm. Preserve all three generation resource refusals and the frozen checkout.

**A11/A12 complete; fix batch-context comparability next.**
[Trace results](docs/astra-a11-trace-results-2026-09-06.md) reconstruct all672
reference/execution traces; body/progress anchoring does not consistently reduce
nonlinear response. [A12 results](docs/astra-a12-results-2026-09-06.md):384 generations,
256 absent executions; full correction13/48 versus fixed shallow/search/no-phase15/48.
The expansion gate fails. All288 generation controls and128 full-cohort execution
controls are exact, but only10/39 unchanged-target executions match across different
batch contexts despite39/39 exact prepared inputs. Root differences reach0.302 m.
Matching A10 inputs also show raw label changes. Do not retune A12 or fit another
inverse before resolving this comparability issue.

**A13 complete; onset localized downstream of policy output.** All64 observed
executions exactly match their original A12 archives;18 whole-input controls and
18 background substitutions preserve target actions. At the stress target,
observed state first differs at0.04s despite matching pre-state and applied action;
observations and actions differ later. A particular backend operation remains
unidentified. [Result](docs/astra-a13-result-2026-09-06.md), independent audit
`outputs/astra_a13_completion_v1`. Preserve both refusals and frozen `ad569b5`.

**A14 passes:** one candidate/original slot with31 immutable background references
gives39/39 byte invariants and64/64 exact full-prefix pairs from128 operational
executions, two edited targets under G1/teleop. [Result](docs/astra-a14-result-2026-09-06.md).
Freeze `/tmp/s2m-a14-a31207e`; preserve the failed prelaunch serialization attempt
and successful`outputs/astra_a14_canonical_context_v2`. CPU1294 pass, focused26;
independent compiled-value/archive audit passes. This controls incoming workload
composition at one useful candidate/job, not GPU island independence or motion gain.

**A15 complete; full correction ties fixed shallow.** In the common nominal
context both qualify4/12 versus nominal2/12, preserving both initial passes but
adding no full-correction gain over the simple comparator. The promotion gate
fails. [Result](docs/astra-a15-result-2026-09-06.md). Both nominal controls match
all64 original trajectories/metadata exactly.16 unique jobs/512 operational
executions supply36 assigned measurements with10 exact aliases;26 distinct
scientific previews,486 context-workload rows. No new generations or independent
units. Simulator807.123s; CPU1303 pass, independent audit and exact rebuild pass.
No resource refusal/abort. Preserve `/tmp/s2m-a15-88fcc45` and all A15 receipts.

**A16 passes its development lead gate:** the frozen depth-only ablation qualifies
5/12 against fixed shallow/full4/12 and nominal2/12, with no paired losses.
At s00/u0, restoring nominal onset retains+10.65mm inflated clearance and raises
continuation0.426→0.501m. [Result](docs/astra-a16-result-2026-09-06.md).
Two new32-env jobs/64 operational rollouts, ten full-context cache aliases;
all12 raw-state measurements independently reconstruct exactly. All36 inherited
comparator records remain unchanged. Simulator135.945s; CPU1315 pass, focused30;
no refusal/abort. Preserve `/tmp/s2m-a16-cc4d44e` and runtime `/tmp/s2m-a15-88fcc45`.
This is one additional reused-unit descriptor pass, not physical/held-out gain.

**A17 closes the depth-only lead.** [Result](docs/astra-a17-result-2026-09-06.md):
the sole distinguishing s01/u2 pair fails under both shallow and depth-only.
Thirty shared-candidate pairs and five verified nominal-preservation pairs prove
zero selected difference on the remaining36 reused units. Two jobs/64 operational
rollouts;35 conditional jobs remain unexecuted. Absolute edited-method rates over36
are unknown. Depth-only improves inflated clearance−38.79→−19.17mm but slows
continuation0.613→0.370m. Keep A16's pilot lead separate; no retuning this rule.
Independent audit checks37 prepared contexts and38 raw measurements, all pass.
Full CPU1332 pass; focused46; exact rebuild and figure inspection pass. Simulator132.119s, peak6068.148MiB, minimum RAM7332MiB; no resource refusal/abort.
Preserve preparation `/tmp/s2m-a17-b1842f8` and runtime `/tmp/s2m-a15-88fcc45`.

**Next design:** [A18 controlled G1 contact sensitivity](docs/astra-a18-g1-contact-design.md),
then separately preregister an obstacle-present comparison of the strongest supported
native compiler. No A18 jobs assigned. A replacement correction requires a new
mechanism and fresh development protocol. Scene confirmation follows method freeze;
multi-beam and stepping remain independent extensions. The updated
[ICRA framing and evidence map](docs/astra-icra-research-story-2026-09-06.md) makes
constructive superiority and prospective geometry transfer explicit open obligations.

**A13 coverage limitation:** the [archive onset diagnostic](docs/astra-a13-trace-coverage-2026-09-06.md)
reconstructs all39 unchanged-target pairs:15 of29 differing executions start in
the first eight steps;14 start later, including eight at4.14s across both encoder
modes. Keep the completed stress probe frozen. A later fix must pass full-prefix
checks and late-window diagnosis across both groups, with early/late/exact cases
under G1 and teleop identified in the diagnostic. No cause or performance gain
is inferred from synchronized onset.

1. **A6 response identification:** [preregistered protocol](docs/astra-a6-response-protocol.md),
   12 scenes×4 seeds×7 native commands=336 response candidates;48 free-WALK controls.
   Same-seed positive/negative perturbations;384 absent previews, no present search.
   Development units are reusable for versioned debugging; untouched confirmation
   scenes/seeds stay untouched. Assess whether regenerated edits transmit useful effects.
2. **Development correction comparison:** fixed2/fixed3, scene-relative geometry heuristic,
   long crouch, sensible two-candidate preview search, geometry-only bounded correction,
   full execution-aware correction. Equalize candidate/preview caps; freeze strongest
   simple and budget-matched controls before test outcomes. About10 percentage points
   is an engineering target, not an acceptance rule or promised result.
3. **Method freeze:** transfer predictor, allowances, final-candidate qualification,
   scene/group splits, baselines and budgets must be concrete and preregistered before
   held-out execution. No endpoint switching after looking at confirmation results.
4. **Confirmation target:**24 held-out scenes×8 fresh seeds×3 physics conditions=576
   assigned evaluations per arm, contingent on development evidence and resources.
   Distinguish interpolation from boundary extrapolation. Report per-scene paired
   effects and scene-level uncertainty; slots are not independent scenes.
5. **Mechanism:** remove execution-response information and progress constraints;
   inspect changed choices and achieved failure types. Compare signed clearance with
   negative values collapsed to zero using the same training records/model.

## Measurement and supporting artifact

Contact event attribution is complete; controlled positive G1 sensitivity remains
open alongside method development. Keep nominal penetration, inflated-margin violation and simulator
obstacle contact as distinct fields. Qualification must observe the continuation suffix.
Historical A5 remains immutable; freeze a separately named prospective endpoint on
commanded task speed or development evidence. The A6 development descriptor uses0.45 m
per second, half the0.9 m/s task speed, and is not a confirmation qualification rule.
Report supported-foot slip only with support evidence, joint dynamics, torso stability,
recovery progress; effort only with actual signals. Tracking error is diagnostic.

Inspect archived A5 full-route failures once for requested/achieved progress, valid
suffix length and failure timing. Do not turn long-route completion into a new dependency.
Release protocols, response records and permitted successful/failed achieved-state
videos. Separate reference geometry, executed records and qualification-passing segments.
Payload rights need their own review. A response corpus is a supporting artifact;
training utility requires the controlled label-information experiment.

## Completion and schedule

Core completion requires a held-out improvement over strong simple and budget-matched
controls, a mechanism ablation, and a reproducible permitted artifact. A development tie
justifies narrower claims, not silent endpoint changes or reopening old branches.

User-proposed milestones: September6–7 scope/contact/response collection;8–9 development
corrections/controls/video;10 method freeze;11–12 confirmation;13–14 figures/manuscript;
15 submission checks. These are targets, not runtime predictions. Verify official venue
requirements directly before submission/video planning; none are assumed by the harness.

## Work log

### 2026-09-06 — scope adopted and A6 implementation

The old method-closure plan is historical. Implemented scene-relative native requests,
explicit censored measurements, bounded response fitting/proposal, and resumable A6
collection harness. All pre-existing probe/report edits remain outside this work.
The frozen contact-event continuation was launched from its original clean checkout;
results and validation will be recorded in the next-stage result note.

Earlier work and publication receipts: [historical ASTRA work log](docs/astra-plan-through-2026-09-05.md#12-work-log-and-next-action).

### 2026-09-06 — response generation and default-encoder pairing correction

A6 generated384 references. Runtime default-mode readback in96 absent previews showed
that assigning variants to different slots confounds native-command response with
encoder realization. Stopped after the third completed batch; retain d0 as diagnostics.
No outcome fitting or generation retry. [Paired continuation](docs/astra-a6-paired-response-protocol.md)
keeps units in the same slots, requires matching realized modes and charges128 padding
rollouts alongside384 scientific previews. The two-candidate deployment budget is unchanged.
Contact excluded-event attribution and the one-time A5 suffix audit are complete;
[results and remaining contact sensitivity work](docs/astra-next-stage-result-2026-09-06.md).

### 2026-09-06 — A6 response dataset complete; confirmation remains gated

Completed all512 matched-slot previews:384 scientific observations plus128 padding,
covering12 scenes/48 units and336 native candidates/48 free controls. Preserve96 d0
confounded diagnostics separately. All runtime encoder pairings and apparatus checks pass.
[Results, figure and achieved-state videos](docs/astra-next-stage-result-2026-09-06.md).
Full CPU1202 passes. No present-method evaluation, held-out seed consumption or method gain.

Deeper native requests gain conditional median clearance but delay exit and increase
terminations; later recovery adds little clearance and reduces median continuation.
Only10/48 units in6/12 scenes have complete seven-command response observations.
Do not deploy complete-case local Jacobians as a transferred response model. Next:
fit a simple pooled response table/model retaining arrival/censoring and partial labels;
validate actual bounded regeneration plus the exact edited preview on development;
then compare strong geometry/simple and budget-matched search controls. Do not open
confirmation until that comparison justifies it. Contact event attribution is closed;
controlled positive G1 contact sensitivity remains open. Historical A5 is unchanged.

### 2026-09-06 — A7 transfer screen completed; generation expansion declined

[A7 result](docs/astra-a7-response-result-2026-09-06.md): twelve scene-grouped folds,
all48 units retained. Execution-aware and reference-only proposers both admit the
same12/48 units; training-selected two-preview search11/48, baseline7/48. Execution
information changes22 choices but zero labels. The seven-command outcome-informed
ceiling is16/48. No new generation, simulation or held-out data.

Partial-label modelling uses262 clearance,171 exit and166 continuation targets rather
than requiring ten complete units. Yet a post hoc matched comparison shows numeric
prediction worse than copying the initial achieved measurement for all three targets.
The next hypothesis is a paired-change model anchored exactly to the initial preview,
with explicit unavailable targets/status heads. This is a design to test, not a method
win. Preserve the A7 negative result and failed expansion gate; no endpoint changes.

### 2026-09-06 — A8 reconciled; identical-slot replay prepared for preregistration

[A8 result and new duplicate-reference analysis](docs/astra-a8-response-result-2026-09-06.md)
close the pending response screen. A8 fails readiness; its evaluation, midpoint
diagnostic and figure/source hashes verify exactly. New CPU analysis preserves all128
A6 scientific/padding pairs outside training. Matching encoder masks still gives20/70
native status disagreements across different slots; same-slot repeatability is unmeasured.

Implemented [A9](docs/astra-a9-same-slot-replay-protocol.md), a64-execution operational
probe retaining original inputs, slots, simulator sources and resource gates. Only
the destination changes; resume checks forbid changes to the simulation contract.
No new generation, present method evaluation or confirmation is assigned. Full CPU:
1236 passed in208.03 s; focused28 passed, including three guards added after full-suite
collection. Existing unrelated probe/report work and pending A8 source artifacts stay
untouched. The [ICRA closure plan](docs/icra-constructive-closure-2026-09-06.md) connects
the user's guidance to evidence-gated duck correction, contact, stepping and utility.

### 2026-09-06 — A9 completed; usable local response remains the next question

[A9 result](docs/astra-a9-replay-result-2026-09-06.md): baseline32/32 and free-WALK32/32
achieved arrays, valid lengths and termination labels match the original A6 jobs
exactly. Apparatus, runtime modes, status and descriptor checks also match.64 new
operational executions, zero generations, independent units or present evaluations.
Simulator wall86.687 s; peak child RSS6069.32 MiB. Preserve the initial sandbox
preflight refusal. Authorized host-level execution passed unchanged resource gates;
both jobs completed and no poller remains. An independent archive comparison confirms
all64 exact pairs. Keep the preparation checkout `/tmp/s2m-a9-95a0731` frozen.

Next: smaller-step native response with no-edit controls, then an actual bounded
correction if prediction/transmission supports it. A9 is operational repeatability,
not method gain; controlled positive G1 contact sensitivity and confirmation stay open.

### 2026-09-06 — A10 completed across all development geometries

[A10 result](docs/astra-a10-results-2026-09-06.md):384 generations,512 absent
executions,384 scientific previews and128 padding. All no-edit generation/execution
controls are exact; all16 jobs verify; independent valid-prefix archive checks and
an exact summary rebuild pass. Full CPU1245 passes, focused35. Simulator-job elapsed
sum838.37 s; generation batch decode27.18 s excludes loads and refusals. Three
resource refusals preserved; no spent batch retried, no poller remains.

Half-size edits increase candidate supply: fixed shallower depth14/48 versus7/48
baseline, outcome-informed seven-command ceiling19/48 versus original16/48. The
fixed shallower command gains eight units and loses one; six scene counts improve.
Smaller onset edits reduce matched clearance midpoint departure, but the execution
predictor still fails four readiness checks. Preserve the negative model result;
carry the simple command forward as a comparator. No present traversal, held-out
confirmation or motion-family transfer is demonstrated. The updated broader
[motion/scene plan](docs/motion-scene-expansion-2026-09-06.md) keeps those obligations
explicit and calls for a separately versioned mechanism before further generation.


### 2026-09-06 — A11 trace diagnosis and A12 actual correction completed

[A11](docs/astra-a11-trace-results-2026-09-06.md) reconstructs672 reference/execution
traces with no new simulation. Body switches are rare; fixed-body/progress anchoring
does not consistently reduce midpoint departure. [A12](docs/astra-a12-results-2026-09-06.md)
then implements and executes bounded progress-first native correction and timing
ablation:384 generations,256 absent executions,192 scientific previews and64 padding.
Full selected13/48 loses to fixed shallow/search/no-phase15/48. All seven initial
passes are preserved. The failed rule is closed without retuning.

All288 generation controls and128 whole-cohort execution controls match exactly.
Additional unchanged-target checks expose batch-context dependence:39 exact prepared
references across full/no-phase contexts yield only10 exact executed prefixes;
maximum root difference0.302 m, with matching labels in those39 pairs. Matching A10
commands also show raw label changes. This limits causal response interpretation
beyond the frozen gate failure. Next is the separately designed A13 early-step
observer/input replay, not more correction sampling. No present or held-out trials.
Full CPU1257 passes, focused18; A11/A12 analyses rebuild exactly. Simulator452.42 s,
no resource refusals, no active campaign or poller. Preserve the pinned checkout.
