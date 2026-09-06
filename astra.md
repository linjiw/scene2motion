# Scene2Motion Research and Work Plan

**Updated:** 2026-09-05. **Audit baseline:** local HEAD `4c679c6`, plus the explicitly uncommitted floating-beam probe/report changes present during this review.

**Status:** active research plan; proposed experiments below are not preregistered campaigns or reported results. Refer to this file when choosing the next task. Update the work log after each completed milestone, attaching the exact artifact and the decision it changes. Existing campaign protocols and historical receipts retain their own definitions.

**Newest follow-up (2026-09-05):** the [displacement-sensitivity receipt](outputs/astra_a5_progress_sensitivity_v1/summary.json)
checks the previous recovery conclusion without choosing a new quality threshold. All17
fixed3/default local passes advance more than25 cm in the following second;15 exceed50 cm.
The fixed2/fixed3 oracle reaches18 only when allowing1 mm backward tolerance; at positive
thresholds through25 cm it remains17. This is post-hoc development evidence, not fresh
method performance. Both website sources now share receipt-derived method/results tables
and a public JSON snapshot. Publication uses an allowlisted `gh-pages` branch, not a push
of the50 unpublished research commits or a motion-payload release. A second180 s contact
admission wait added10 refusals at4067–4072 MiB free VRAM; no simulator launch or seed spent.

**Newest decision milestone (2026-09-05):** [recovery and selection audit](docs/astra-a5-recovery-selection-result-2026-09-05.md)
rebuilt three CPU-only receipts. All17 default fixed3 local passes meet the descriptive
one-second kinematic recovery checks. An absent-execution bank selects18/32 local passes,
but its sole gain is essentially stationary afterward (−0.313 mm net displacement).
Adding positive one-second continuation leaves selection and both outcome-informed
pool ceilings at17/32, matching fixed3. This is post-hoc, threshold-sensitive diagnostic
evidence, not revised success or high-quality labels. Improve candidate supply rather
than train a selector over this pool. A versioned contact continuation at `acb6c21`
waited180 s and recorded10 admission refusals: free VRAM4306–4307 MiB<5500 MiB.
No simulator launched, no seed was spent and no poller remains. Full CPU1148 passed;
the final36 focused tests include14 later additions. Next: complete contact validation,
then freeze final-edited qualification and one fresh progress-aware correction.

**Previous measurement milestone (2026-09-05):** [contact validation](docs/astra-contact-validation-result-2026-09-05.md)
completed six corrected simple-body controls and64 exact matched G1 replays. Peak G1
process RSS6101.09 MiB; successful replay wall135.707 s. All32 present valid prefixes
report zero contacts across30 robot bodies, but the untrimmed reader saw up to6 points.
An excluded-event diagnostic hit the1200 MiB free-VRAM floor and stopped before output;
no automatic retry. Contact freedom outside the saved prefix and controlled G1 positive
sensitivity remain unverified. Original A5 labels are unchanged; no new method gain,
independent scientific unit or high-quality dataset claim. Full attempt accounting
preserves15 launches,6 unsuccessful. Next: gated, versioned contact diagnostic continuation,
then final-edited progress/quality qualification and one fresh bounded correction.

**Latest execution milestone (2026-09-05):** [A5 is complete](docs/astra-a5-results-2026-09-05.md):
24 fresh references, 576 verified executions, eight motion-seed groups, one reused beam.
Three-second duck present passes: default17/32, G1=17/32, teleop15/32. G1 gains two
matched passes over teleop, but gains one/loses one against default; no demonstrated
overall improvement over the established default comparator. No full-route completions.
All199,966 action samples have verified runtime modes. No retry or resource abort;
peak simulator RSS6104.58 MiB. Physical contact and high-quality motion remain unmeasured.
Next: final-edited-motion progress/quality qualification and contact instrumentation,
then one bounded progress-aware correction with fresh grouped evaluation. Do not expand
the losing A4 timing rule, launch another broad mode sweep, or retune spent A5 seeds.

**Latest diagnostic (2026-09-05):** [A5 progress/quality audit](docs/astra-a5-quality-audit-2026-09-05.md)
covers all24 references and576 executions using archived data only. Default fixed3's
17 present local passes have median root lag1.201 m at4 s, yet all advance over the
next post-dwell second (median0.592 m). Nine of15 failures lack a4 s observation;
their missing suffixes remain unknown. Joint RMSE alone cannot qualify demonstrations.
This CPU-only diagnostic preceded the live contact work above; its original geometry
labels remain unchanged. The [contact-validation plan](docs/astra-contact-validation-plan.md)
now links the versioned probe/replay results and the remaining contact-coverage gate.

## 1. Objective and decision

Build a method that turns a fixed scene, start state and goal into high-quality humanoid motion that completes obstacle traversal in physics simulation, then release the scene, reference, achieved motion and failure records needed to train and evaluate downstream systems.

The immediate method hypothesis is **execution-aware local compilation of a qualified motion carrier**. ARDY supplies coordinated motion; Scene2Motion chooses a scene-relative editing window, measures how the frozen controller transmits edits, constructs a bounded repair, and verifies or refuses it. Ducking and stepping share this interface but need different repair backends.

Three commitments:

1. Finish a bounded traversal method before claiming a navigation system. A short box or beam crossing is one skill; uninterrupted multi-obstacle navigation is a later, separate experiment.
2. Preserve the dataset objective. A useful dataset needs executed positive demonstrations, informative failures, scene coverage and a consumer experiment. A large reference-only archive does not complete it.
3. Research quality takes priority over a submission date. Do not promise a positive result, an acceptance probability, or a comprehensive dataset in ten days. If the frozen stack cannot transmit useful edits, record that boundary and make a deliberate architecture decision rather than repeatedly enlarging the same seed pool.

## 2. Current evidence: corrections to the pasted advice

Local receipts take precedence over old chat and the public website. The live project page retrieved on 2026-09-05 still asks to execute EXP-024 and EXP-031; that text is stale relative to the local results.

| Item | Verified state | Consequence |
| --- | --- | --- |
| EXP-021/022 | 44/64 elicit at least 3 cm somewhere; 12/64 clear the diagnostic fixed 5 cm box; no retained clearing trajectory in the studied pool | Motivation for modifying candidates. This finite-pool result is not impossibility for other priors, routes, budgets or controllers. |
| EXP-030 | 192 obstacle-absent/5 cm/20 cm rollouts; 0/64 completion in either present arm under evaluator v1 | Keep the original long-route definition; do not rename it as the newer local endpoint. |
| Evaluator v2 | Already implemented in `scene2motion/traversal_eval.py` and used prospectively by EXP-031 | Do not rebuild v2 or retroactively change EXP-031. Contact availability and whole-body passage still need attention for future studies. |
| EXP-031 | Completed: 2 candidates × 4 arms = 8 rollouts; repaired-present local success 0/2, source-pool success 0/64 | Both reached the root recovery condition upright without cutoff, but nominal geometry was penetrated. The immediate failure is achieved clearance, not simply support or long-route termination. |
| EXP-031 transmission | Intended peak local lift 21–47 mm; achieved peak paired change 4–18 mm in the aligned windows; minimum-clearance losses 75.5/80.1 mm | Diagnose phase/position error and weak edit transmission. Two carriers cannot set an execution margin or justify inverse-gain correction. |
| EXP-024 | Completed: 128/128 prospective rollouts. AUC 0.8549; flagged cutoff 92/95; passing cutoff 20/33; `contract_confirmed=false` | The 0.20 s rule is a risk feature. Passing it is not carrier qualification or an execution admission guarantee. Root pinning did not produce the valid conjunction. |
| Compiler v2 | Quintic lift, stance report, absolute dynamics report and finite-sample margin helper implemented | Achieved-phase prediction, response identification, IK integration and calibrated edited-motion execution remain unfinished. |
| Floating beam | Operational paired probe complete but inconclusive: 0/2 no-beam controls reached x=1.82 m | A trajectory difference alone does not validate the traversal apparatus. Qualify controls before a comparative ducking campaign. |
| Duck correction | Historical v2 reference result: 72.9% → 99.3%, versus 77.4% best-of-three, 36 scenes × 8 seeds | A promising reference mechanism, not a guaranteed physical positive result. Old scenes are development/translation data, not fresh geometry holdout. |
| EXP-028 | Driver/protocol present; no completed campaign artifacts found in this working tree | Outstanding supporting experiment; do not assume either falls or recovery after disabling cutoff. |
| DB preview | Validator passes: 300 records; 274 reference-geometry, 26 scene-feasibility; execution not measured for all 300; motion payloads omitted | The larger repository has execution archives, but the preview has not integrated them. Its scene-ID split is not Geometry-OOD. |

Primary evidence: [EXP-024 result](docs/ramp-exp024-reference-contract-result-2026-09-04.md), [summary](outputs/exp024_reference_contract/summary.json); [EXP-031 result](docs/ramp-exp031-constructive-step-repair-result-2026-09-04.md), [summary](outputs/exp031_constructive_step_repair_execution/summary.json); [anchoring development map](outputs/analysis_execution_anchoring_development/summary.json); [beam probe](outputs/probe_floating_beam_20260904/report_h0.25_z1.175.json); [research ledger](docs/REPORT.md); [preview receipt](outputs/scene2motion_db_preview_v1/receipt.json).

The beam probe and the latest ledger/protocol edits were uncommitted at audit time. Their source state must be frozen before reuse as campaign infrastructure. The earlier [compiler-v2 plan](docs/execution-aware-step-compiler-v2-plan-2026-09-04.md) remains the detailed stepping design reference.

## 3. Scientific questions and prospective contributions

**H1 — execution-aware editing:** on qualified carriers and fixed scenes, anchoring edits to predicted achieved overlap improves local traversal over reference-position anchoring. A changed window alone must be compared with all other components held fixed.

**H2 — measured correction:** a bounded update from a signed scene deficit improves executed clearance over both a tuned fixed deformation and additional generator sampling. Failure to transmit the edit, or success only after excessive motion distortion, weakens this claim.

**H3 — useful records:** execution-labelled scene/motion groups improve held-out execution-risk prediction or skill selection at a fixed data/compute budget. Geometry-only classification accuracy does not establish this.

Provisional contributions, conditional on evidence:

1. A scene-relative compiler with **two explicit backends**, native-control duck correction and contact-window step repair, retaining the motion prior and controller.
2. Paired obstacle-present evaluation separating qualification, admission, local crossing, motion quality and full-route completion, with strong fixed-heuristic and resampling controls.
3. A reproducible scene–reference–execution corpus with grouped splits and measured downstream utility. Until utility and coverage are demonstrated, call it a released record corpus rather than a comprehensive navigation benchmark.

Neither spline interpolation, IK, filtering nor the word “compiler” establishes novelty. The proposed contribution is the tested relationship between scene-relative deficits, controller response, bounded edits and executed task completion.

## 4. Method specification to implement

### 4.1 Qualified carrier bank and information contract

Start with archived neutral WALK controls for **apparatus development**; they are candidates, not already qualified carriers. Preserve all 16 `all_walk` controls from EXP-023 and EXP-023b in a preparation manifest. EXP-023b's refused SQUEEZE control does not erase its archived WALK rows, but its source campaign status must travel with them. Reference floor penetration is already nontrivial in some WALK clips; prompt identity alone proves nothing about quality.

For a fresh bank, freeze a bounded generation pool, qualification criteria, route horizon and physics repeats before generation. Record every generated, screened, executed and qualified candidate. Qualify on obstacle-absent local progress/recovery, uprightness, corridor containment, acceptable floor behavior and measured tracking; record the support proxy without treating it as sufficient.

Choose an explicit deployment mode:

- **Bank mode, first implementation:** each reusable carrier has paid obstacle-absent execution measurements made before target-scene evaluation. Its map can be reused on new scenes. Report bank construction cost and amortized cost separately; share access to the bank with applicable baselines.
- **Novel-carrier mode, later:** predict execution for an unseen carrier from development data. A new obstacle-absent rollout at test time is extra computation, not zero-shot prediction. Count it if allowed; never access repaired-present test states when choosing the edit.

Hold out carrier identities as well as scene families when claiming novel-carrier transfer.

### 4.2 Predict achieved overlap; measure whether edits transmit

Use explicit timestamps and the archive convention: achieved sample i is after step i+1; at 50 Hz, `qpos[1::2][k]` aligns with reference frame k+1. Trim padding/reset frames using the existing loader. No dynamic time warping to the test outcome.

For each foot, find predicted **achieved swept-envelope overlap** with the fixed obstacle and pair it with an existing swing/support window. Preserve disconnected overlap intervals; do not force an oscillating foot trajectory into a monotone world-x map. Refuse unavailable horizon, ambiguous phase, stance-only overlap or missing lead/trail passage instead of moving the obstacle.

Probe small, bounded task-space edits on development carriers with matched raw/edited obstacle-absent rollouts. Estimate sign, amplitude and uncertainty of the achieved response. Avoid negative lifts that violate floor clearance; an admissible zero/positive dose pair is preferable to blindly using symmetric perturbations. Weak or sign-changing response is a reason to stop inversion. Do not divide by a near-zero fitted gain.

### 4.3 Two repair backends under one budget

**Ducking:** retain generator-native height controls and measured overhead correction. Tune a fixed extra-crouch baseline on development data, then freeze it. Preserve a no-op on scenes already satisfying the required clearance.

**Stepping:** use the existing endpoint-smooth quintic task-space offsets, warm-start bounded IK and explicit verification of whole-body geometry, stance drift, floor penetration, joint limits, absolute speed/acceleration and boundary continuity. Both feet must cross. Recheck after any smoothing or retiming. A penalty weight is not a hard limit; violating a checked limit means refusal.

Keep the initial parameterization small. Do not add pelvis motion, global time scaling, contact rescheduling and a full trajectory optimizer simultaneously. First test achieved anchoring and transmission on the current bounded-edit interface. Add one degree of freedom only when a recorded failure identifies its need.

A static CoM-in-support-polygon test is not a universal condition for dynamic walking or jumping. Dynamic/contact constraints need an appropriate model; do not use an unsupported static rule to remove legitimate gaits.

### 4.4 Execution margin and finite-sample limits

Separate nominal geometry, the current 0.04 m geometry-coverage inflation, and execution loss. If the collision API already applies `BODY_MARGIN`, do not add it twice.

Calibrate loss on the **frozen edited-motion policy's distribution**, including rollout failures. A raw-WALK error quantile does not automatically transfer to repaired motion, a new scene family or obstacle contact. Qualification and response fitting use development data; margin calibration and final evaluation use separate grouped data.

For aligned per-body signed clearance, one candidate score is `max(0, max_{t,b}(d_ref(t,b) - d_ach(t,b)))`. A difference of two trajectory minima is only a diagnostic and need not bound loss at every time/body. An unobserved suffix after cutoff cannot silently count as zero loss. Define refusal/censoring before calibration.

Use the one-sided order statistic at rank `ceil((n+1)*(1-alpha))`; if rank exceeds n, refuse a finite margin. At alpha=0.05 at least 19 independent calibration units are needed merely for a finite rank; this is not evidence of a tight or transferable bound. Physics repeats are not independent carriers. Phase-specific strata need enough independent units or a prespecified pooled fallback. Report held-out coverage and abstention; avoid certificate language.

## 5. Ordered work packages and stop rules

### A0 — consolidate evidence and prepare controls (completed)

Deliver this plan, verify the current regression paths and DB package, and create a deterministic reference-only WALK preparation manifest. Do not rerun completed EXP-024/031 or rewrite their receipts. Keep existing uncommitted files intact.

### A1 — qualify the traversal apparatus before method comparisons

**Update 2026-09-05: operational gate completed.** A1's first cohort passed 6/8, the declared
reserve passed 7/8; all live readback checks passed. A1b lowered the beam to 1.05 m and blocked
all seven previously qualified controls. See [results](docs/astra-a1-results-2026-09-05.md).
This does not complete motion-quality qualification or corrected-duck execution.

**Question:** can the matched no-obstacle carrier traverse the interaction region, and is the floating beam actually at its commanded pose throughout the run?

1. Start with eight EXP-023 WALK controls in their archived order; use the other eight as a declared second development batch if needed, preserving both batches' outcomes. Do not select controls by beam-present success.
2. Run paired no-beam and raised-beam controls with sufficient horizon and identical tracker configuration. Evaluate local recovery, not just maximum root x. Preserve refusals and failure reasons.
3. Complete first/last-step pose, extent and neighboring-environment read-back. A CLI cuboid specification or nonzero trajectory difference is not pose verification.
4. Use a deliberately blocking beam only after the no-beam controls are qualified. Report actual contact only if a sensor path was validated with positive/negative controls; otherwise report nominal and inflated replay geometry, with contact unknown.

The [EXP-029 protocol](docs/ramp-exp029-selection-vs-coverage-protocol.md) already fixes V1 pose tolerances and V2/V3 thresholds of at least 7/8 controls. Follow it if executing EXP-029. A smaller, local ASTRA pilot requires its own versioned protocol; it must not silently replace EXP-029's scene grid, seeds, goal or statistical design. Its historical “will not run before deadline” statement is a forecast, not permission to reinterpret an unfinished campaign as complete.

**Stop:** if reachable controls cannot be obtained in the fixed development budget, pause beam comparisons. Inspect input/controller compatibility or choose a separately declared controller-native positive control. Do not generate hundreds of obstacle failures on an unqualified route.

### A2 — smallest constructive ducking test

**Prerequisite:** A1 passes. **Development budget proposal:** 8 scene–carrier units × 4 arms × 2 physics seeds = at most 64 obstacle-present rollouts, plus separately counted absent qualification runs. This is engineering development, not the final comparative estimate.

Arms: uncorrected, tuned fixed crouch, measured correction with at most two updates, best-of-three resampling. All use fixed scene geometry, the same observation contract and the same final verifier. Record generator calls, CPU verification/IK time, simulator calls and failed attempts; “three candidates” is not equal total compute if qualification differs.

**Progression criterion:** a repeatable margin-preserving local crossing on at least two distinct development scenes and an interpretable paired comparison. This is an engineering gate, not a significance or acceptance threshold. If only fixed crouch works, keep it as the winner and revise the method hypothesis. If all fail before the beam even without the beam, return to A1.

### A3 — staged stepping compiler

Follow D1–D3 in the existing compiler-v2 plan: fresh neutral-carrier qualification → small-signal transmission → achieved-window anchoring and bounded candidate construction. The archived eight-carrier map is development evidence only.

Start with at most 8 development carriers and two admissible edit doses per carrier; archive the raw matched control. Freeze dose bounds before their executions. Stop the foot-edit branch if no useful, stable transmission exists across the declared development set. Extra height and lower support thresholds are not justified substitutes for that result.

After offline construction succeeds, use a small preregistered absent/present paired stepping pilot with new units. s4408/s4434 remain the closed EXP-031 pilot. A weak-transmission result motivates a separately scoped alternative representation or controller compatibility study, not an endless buffer sweep.

### A4 — fresh method comparison, only after constructive development

Select one primary behavior family based on development outcomes, then freeze the entire comparison. Suggested initial design: 24 independent scene–carrier units × 4 arms × 3 physics seeds = at most 288 target-obstacle rollouts, plus qualification/calibration cost. Eight distinct units or 24 repeats of one scene do not establish 24-scene generalization. Final sample size should follow a paired effect/uncertainty calculation on development data; the numbers here are planning budgets, not a power guarantee.

Ducking retains A2's four arms. Stepping compares raw carrier, a development-tuned fixed spline, best-of-three native STEP, and the full compiler. Use a separately budgeted ablation grid to compare reference versus achieved anchoring and no-margin versus calibrated margin; geometry-only IK remains the historical baseline. Freeze the exact tested arms before the campaign, rather than dropping a losing baseline afterward.

Primary endpoint: all-assigned local completion under a stated geometric-margin criterion. Report raw per-seed rates and the prespecified “at least 2 of 3 repeats” endpoint; call the latter repeat success, not broad robustness. Report qualification yield, admission coverage and conditional completion separately. Rejected scenes stay in the denominator. Full-route completion and physically sensed contact-free completion are separate columns.

Use paired unit-level comparisons and intervals clustered at the scene level; reuse of a carrier across scenes also induces correlation and requires grouped/multiway treatment or an independently paired design. Prespecify the primary contrast and adjustment for multiple comparisons. Do not interpret a bootstrap over 50 Hz frames as population uncertainty.

### A5 — supporting mechanisms, not another prerequisite loop

EXP-024 and EXP-031 are closed. EXP-028 can answer what removing the evaluator cutoff actually changes, with its own frozen settings and paired existing cohort. Inspect its completed artifacts if they become available in another worktree before launching anything. Distinguish fall, recovery, stall, collision and unmeasured contact. Do not preregister the expected conclusion that cutoff necessarily hides a fall.

Kimodo execution transfer, squeeze, jumps, ramps, stairs and hardware follow a successful primary comparison; none should block it. A named second prior is useful only with matched conversion, rate, controller and evaluation contracts.

## 6. Evaluation and motion quality

Keep evaluator v2 immutable for its completed campaigns. It uses a **root-only** recovery line; it does not establish that every body part has cleared. Before a new general traversal claim, add a versioned whole-body rear-envelope/feet passage field and require a fixed upright corridor dwell after crossing. Pin recovery distance, dwell, timeout, allowable contacts and termination configuration before seeing new results. Existing 0.50 m/0.20 s/4.0 s values belong to EXP-031, not every task.

Report independent events with their times as well as the exclusive outcome. Unknown contact remains null/not assessed. Distinguish a later route cutoff from an event before local completion. Track geometry sampling rate and check near-boundary candidates with denser/interpolated collision queries; millimeter numeric output does not establish millimeter physical accuracy.

“High-quality motion” needs more than passage. For raw, repaired and achieved motion report:

- minimum nominal/inflated clearance and floor penetration;
- stance slip and contact/support consistency under the available measurement;
- joint speed, acceleration and jerk, with sampling/filter conventions;
- boundary pose/velocity jumps, edit norm and upper-body distortion;
- time/progress and full-route continuation, with failures retained;
- control effort/torque limits when exported, otherwise not measured.

Numerical limits should use robot specifications and declared development controls, not the chat's arbitrary 10 rad/s² or 15 mm. Aesthetics or human-likeness require their own comparison; physical constraints and smoothness alone do not establish them.

## 7. Scene2Motion-DB release and downstream experiments

### 7.1 Build paired execution records first

Extend the existing exporter as a new schema/version. Preserve the preview unchanged. Import EXP-022/024 tracking records and EXP-030/031 obstacle-present records with their actual evaluator, obstacle, timestamp and provenance contracts. An EXP-024 row is not a physics-obstacle label. Retain failed/rejected records and unavailable payloads explicitly.

Use a lineage graph, not just a pair ID. Group raw/repair/resamples, physics repeats, obstacle-absent/present variants and motion-conditioned scene ladders by shared source motion and scene ancestry. Splits must keep connected groups together; otherwise one source clip can appear with different boxes in train and test despite distinct pair IDs.

Evidence fields are independently available: geometry, reference screens, obstacle-absent tracking, obstacle-present execution. Store `assessed` and null values for missing quantities. Include scene primitives/meshes, robot geometry hash, start/goal, coordinate frames, joint order, units, reference/achieved timestamps, source hashes, controller/simulator settings, method parameters and per-stage costs.

Keep two usable products: (i) executed positive demonstrations with explicit quality criteria; (ii) diagnostic/intervention records including failures. Do not feed failed rollouts to imitation learning as positive expert actions. Export segmentation and termination masks so downstream researchers can choose their supervision.

### 7.2 Coverage and splits

Freeze development, calibration, fresh IID and Geometry-OOD groups before learner evaluation. Add motion/carrier holdout when claiming carrier transfer. Predefine OOD geometry intervals or compositions with a gap from development ranges; a new ID alone is not OOD.

Publish a coverage matrix over task family, obstacle height/depth/position, approach speed, route curvature, passage width and obstacle composition. Show counts of scenes, independent motions, variants and successful executions separately. Missing classes remain empty cells. The first validated release may cover only short beam/low-box traversal; it is not yet comprehensive humanoid navigation data.

Provide loading, replay and evaluator examples plus a dataset card describing intended uses, exclusions, biases and third-party redistribution terms. Keep generated motion, robot assets, weights and code licensing distinct. Validate a portable package outside its original absolute paths before a public release.

### 7.3 Smallest useful learner experiment

Use identical pre-execution inputs and architecture for two rankers: binary executed-outcome supervision versus binary plus signed **execution** deficit supervision. No achieved trajectory, cutoff time or physical contact field may be a test-time input. If the target is reference collision and exact clearance is already an input, the result merely relearns the geometric verifier.

Prefer fully labelled small candidate groups for evaluation; otherwise selective launch labels hide missed successes. Include the existing exact geometry/support rule as a nonlearned baseline. Prespecify a task with enough positive and negative training/test examples; if target-obstacle success remains zero, success-ranking utility is not identifiable from that corpus.

Report execution-success AUPRC, Brier score, risk–coverage, failure recall, false-admission rates with explicit denominator, and expected launch cost on held-out groups. An offline replay of a complete label matrix is an offline selection experiment. A navigation or online launch-efficiency claim needs actual prospective execution.

### 7.4 Motion → critical scene as the subsequent data engine

Retain this direction, without presenting it as a cure for untrackable motion. Start from execution-qualified motions; construct primitive scene ladders where the adapted motion passes reference geometry, a declared nominal comparator fails, and removing the critical obstacle restores the comparator. Call this criticality **relative to the tested alternative set**, not necessity or optimality over all possible robot motions.

Keep inverse-generated scenes separate from exogenous fixed-scene tests. Reexecute a stratified subset with the obstacles present; obstacle-absent feasibility does not transfer automatically. Compare random procedural scenes, motion-compatible scenes and critical counterfactual scenes using the same learner and data budget. Test on independently generated or externally sourced layouts, never on another deformation of the same source motion's scene.

## 8. Completing navigation beyond the first paper

After local traversal works, learn a small scene-to-skill/event interface before training a new whole-body generator. Inputs include geometry/perception, current state and goal; outputs include skill, local path and event parameters. The frozen prior or bank supplies the carrier, the compiler adapts it, and execution feedback updates the next decision.

Navigation milestones: straight local crossing → turns and approach-state variation → two sequential obstacles without resetting → mixed clutter with a real route choice → held-out layouts and perception noise. Count detours as legitimate navigation actions when allowed; the current corridor traversal task intentionally forbids them. Report uninterrupted mission completion, progress, collisions, recovery and quality under one fixed budget.

If no useful edit transmission emerges from the bounded development study, test a controller-native or scene-conditioned alternative as a new comparison. Freezing both prior and controller is a scientific constraint for the first study, not a permanent restriction on the user's goal of producing good motion and useful data.

## 9. Resources, schedule and paper decision

Run one Scene2Motion campaign owner at a time and preserve other users' processes. Record exact project/tracker/checkpoint identities, source manifests, prediction and array hashes, batch plans, seeds and attempts. Use clean isolated campaign worktrees when needed. Do not commit another session's changes to manufacture a clean launch.

The current measured `SONIC_LAUNCH_GATE` is 5,500 MiB free VRAM / 9,500 MiB available RAM with concurrent Isaac processes recorded. Some frozen legacy protocols carry stricter gates. Apply the actual campaign's frozen gate; do not replace it silently with either the newer preset or old chat's 12/18 GiB numbers. The operational beam probe separately defaults to a 10,000 MiB RAM floor.

| Window | Deliverable | Decision |
| --- | --- | --- |
| Sep 5–6 | A0 complete; A1 protocol, control qualification and pose instrumentation | Apparatus works, or record why methods cannot yet be compared. |
| Sep 6–8 | Bounded A2/A3 development | Choose the mechanism supported by execution; no guaranteed winner. |
| Sep 8 | Evidence/readiness review | Start fresh comparison only if the method and infrastructure are ready. Otherwise continue the research beyond this deadline. |
| Sep 8–11, conditional | A4 frozen comparison and crucial ablations | Report uncertainty and failure boundaries; no outcome-based extension of test seeds. |
| Sep 11–14 | Evidence-derived figures, release validation, bounded manuscript | Every result sentence maps to a receipt and a measurement tier. |
| After local closure | DB execution release, learner utility, navigation composition | Dataset and system completeness gates below. |

The [official ICRA 2027 call](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/) checked Sep 5 lists September 15, 2026 at “11:59 PST”, eight pages including references, and video windows Aug 5–Sep 9 and Sep 17–22 (closed Sep 10–16). Confirm the submission portal's timezone/window interpretation; do not plan a first video upload on Sep 14. Venue timing is a constraint on a submission, not evidence that the research is complete.

If fresh comparative evidence is absent, the available manuscript remains an evaluation/failed-repair study. Do not claim “safe”, “first”, universal applicability or completed navigation. Do not default to abandoning the method and dataset objective merely to submit that manuscript.

## 10. Definition of done

**Method milestone:** documented input/output and online-information contract; repeated obstacle-present success on a fixed-scene holdout; paired comparison against tuned heuristic and resampling; qualification/refusal/cost denominators; achieved-motion quality and uncertainty; reproducible code and archives. A single success establishes existence only.

**Dataset milestone:** reusable scene/reference/achieved payloads with valid provenance and redistribution scope; explicit evidence/missingness; leakage-resistant lineage splits; coverage and quality tables; successful demonstrations plus failures; portable replay; one useful downstream comparison or an explicitly bounded release without utility claims.

**Navigation milestone:** mixed held-out scenes, sequential skills without resets, scene-to-skill decisions using declared observations, task-level completion/collision/quality/cost, and a comparison with a route-aware baseline. This is additional work beyond the local traversal paper.

## 11. Literature boundaries checked for this plan

This is a focused primary-source check, not a complete novelty review. The comparison below motivates questions; it does not establish priority.

- [CLoSD](https://openreview.net/pdf?id=pZISppZSTv) already couples a diffusion planner with a physics tracking controller. “Closing the loop” alone is not the contribution here.
- [ReactiveBFM](https://arxiv.org/abs/2606.30362) explicitly addresses imperfect physical states with planner training and asynchronous replanning. The frozen-stack setting must disclose the capability it gives up.
- [Moving Through Clutter](https://arxiv.org/abs/2603.05993) builds scene-aware human/humanoid locomotion supervision through VR and retargeting. Scene2Motion needs a concrete execution/repair/negative-label contribution rather than claiming that paired scene–motion data does not exist.
- [LfH](https://arxiv.org/abs/2007.14479), [LfLH](https://arxiv.org/abs/2108.09793), and [LfH-CP](https://arxiv.org/abs/2509.26513) motivate inverse supervision and critical-constraint factorization. Their navigation evidence does not establish humanoid contact-dynamic feasibility.
- [MIME](https://openaccess.thecvf.com/content/CVPR2023/html/Yi_MIME_Human-Aware_3D_Scene_Generation_CVPR_2023_paper.html) generates human-motion-compatible indoor layouts. Our proposed critical-scene tier adds a tested comparator and execution labels, but that difference still requires a broader novelty audit before a paper claim.

## 12. Work log and next action

### 2026-09-05 — repository review and plan

- Read the supplied discussion, current methods, evaluator, protocols, result notes, receipts and public project page. Corrected obsolete EXP-024/031/v2 tasks.
- Verified 44 focused regression tests across projection, evaluator measurement, execution anchoring/gap, dataset release and the current beam probe: all passed.
- Revalidated `outputs/scene2motion_db_preview_v1` using its existing portable validator: passed, 300/300 records, execution not measured for all 300.
- Completed A0's archived WALK control preparation: [preparation script](experiments/prepare_astra_walk_controls.py), [manifest](outputs/astra_walk_controls_v1/manifest.json), and [regression tests](tests/test_astra_walk_controls.py). All 16 controls are preserved; the manifest binds source receipts, row/archive hashes and per-motion array identities. Seven new tests pass, and independent in-memory reconstruction matches the written manifest. No motion was regenerated and no simulator was launched.
- Added a README link to this plan. Preserved the pre-existing uncommitted beam/protocol/ledger edits and left `AGENTS.md` unchanged.
- Final repository CPU regression run: `source env.sh && env LD_LIBRARY_PATH= "$S2M_PY" -m pytest tests -q` — **939 passed in 212.62 s**. `git diff --check` also passed. These software checks do not establish carrier qualification or traversal success.

Reproduce A0's new artifact:

```bash
source env.sh
env LD_LIBRARY_PATH= "$S2M_PY" experiments/prepare_astra_walk_controls.py --check
env LD_LIBRARY_PATH= "$S2M_PY" -m pytest -q tests/test_astra_walk_controls.py
```

To create a new preparation output, omit `--check` and supply a fresh `--out` path. The command refuses to overwrite an existing manifest. A preparation manifest is not a launch protocol or a carrier qualification result.

### 2026-09-05 — A1 implementation, freeze and resource refusals

- Implemented the independent [A1 protocol](docs/astra-a1-protocol.md), [driver](experiments/astra_a1_apparatus.py), [read-only apparatus callback](scene2motion/astra_apparatus_export.py) and [whole-body local qualification](scene2motion/astra_qualification.py). Fixed cohorts/arms/budget, source and artifact hashes, resource gates, no-overwrite receipts and completed-arm-only resume are in place. Old evaluator and campaign definitions are untouched.
- Frozen in `401acc1`, with a pre-execution process-identification correction in `2b28dd4`. Full CPU suite: **956 passed in 205.76 s**; after two additional focused regressions, the ASTRA tests alone total **26 passed**. Runtime USD/Isaac instrumentation is still unvalidated; unit tests do not count as apparatus evidence.
- Actual launch attempts stopped **before any SONIC subprocess**: available RAM was 17744 MiB initially, then 17884 and 18234 MiB under the corrected harness, below the unchanged 18432 MiB gate. The original process detector also misidentified two Isaac documentation servers; the corrected detector records/exempts only those entrypoints and still blocks simulator processes.
- Preserved [initial preparation/identity](outputs/astra_a1_apparatus_v1/identity.json) and [corrected attempt](outputs/astra_a1_apparatus_v1_1/identity.json), including timestamped resource refusal reports. **New generator calls: 0; new SONIC rollouts: 0; A1 qualification result: not measured.** No co-tenant was stopped and no campaign is running in the background.
- See [implementation/preflight note](docs/astra-a1-preflight-2026-09-05.md) for the exact clean checkout, source commit and resume command. The other session's uncommitted probe/protocol/ledger files remain untouched.

### 2026-09-05 — user-authorized resource amendment; A1 and A1b completed

- The user explicitly authorized using less than 18 GiB RAM and adapting parallelism. Before
  any rollout, [protocol amendment 2](docs/astra-a1-protocol.md) adopted the measured 5500 MiB
  free-VRAM / 9500 MiB available-RAM launch preset and retained active 1200/2500 MiB abort
  floors. Co-tenants are recorded and not interrupted. Earlier refusals remain historical.
- Executed A1 from `8c07e00`: **32 rollouts**, two cohorts × absent/raised × eight environments.
  Local paired pass counts: **6/8**, then **7/8**. Every apparatus readback passed; within each
  cohort absent and raised achieved arrays are elementwise identical. The declared progression
  gate passed on the reserve, without dropping any input or changing the endpoint.
- Preregistered and executed [A1b](docs/astra-a1b-target-beam-protocol.md) from `f2d60e3`:
  **8 more rollouts** with the same reserve WALK references and beam underside lowered to
  1.05 m. Local completion **0/8**; previously qualified controls blocked **7/7**. All have
  nominal/margin intrusion and all pose checks pass. Six cutoffs, two uncut stalls; physical
  contact remains unmeasured. This validates the target-beam apparatus, not a repair method.
- Total: **40 new executed rollouts; zero new generation/training**. SONIC subprocess wall
  time **273.95 s**, peak launch VRAM 3645 MiB, minimum available RAM across launches
  6024 MiB. No resource abort. Eight environments were the useful cohort size; future larger
  comparisons can use the measured 32-environment configuration with a frozen batch plan.
- Created a [rebuild-checked execution index](outputs/astra_execution_index_v1/index.json):
  40 records / 16 carrier groups, 16 tracking and 24 obstacle-present records, all development.
  Its 26 local-pass records are WALK controls, not corrected obstacle-traversal demonstrations.
  Unknown contact and full quality remain unknown. The old DB preview is unchanged.
- [Result note](docs/astra-a1-results-2026-09-05.md) links the exact summaries and reproduction
  command. Final validation: **969 CPU tests passed in 214.24 s**, execution-index rebuild
  matched, and `git diff --check` passed. No campaign is left running in the background.

**Next executable research milestone:** A2 native-control ducking correction, with fresh
generation seeds, matched absent qualification and target-beam execution, and the four proposed
arms. Freeze its exact generation budget, quality gates and batch layout before sampling. A
32-environment eight-unit/four-arm matrix is now operationally reasonable under the measured
resource budget; report qualification cost separately. Training is justified only by a defined
consumer/comparison, not GPU utilization alone. The first missing method endpoint is a corrected
duck that actually crosses; A1/A1b no longer block that test.

### 2026-09-05 — A2 method-development protocol frozen

- Implemented [native duck d0](scene2motion/astra_duck.py): smooth low-dimensional native
  root-height commands, bounded measured-headroom updates and a shared reference-only
  selector. This is an uncalibrated engineering hypothesis, not the final compiler.
- [A2 protocol](docs/astra-a2-native-duck-protocol.md) reserves 60000–60007,
  60100–60107 and 60200–60207: 56 actual generations, four method arms, two beam heights,
  two physics seeds, 128 paired method rollouts plus eight free-height controls.
- Main SONIC batches use 32 environments with the measured host budget. Added optional
  environment/seed parameters without changing A1's default commands or historical results.
- Focused validation before sampling: 43 ASTRA tests passed. Next: generate the frozen
  candidates, release ARDY, execute absent/present comparisons, then record the result and
  choose the next change from observed failures. Existing other-session files stay untouched.

### 2026-09-05 — A2 harness failure preserved; isolated A2r successor frozen

- A2 completed 56 references and 136 rollouts. All initial references meet the reference
  target, so measured correction is an exact no-op; it cannot demonstrate feedback gain.
- Seven present readbacks found neighboring beams in the route: the rough-terrain importer
  can reuse origins despite `env_spacing=12`. Enforcement was incorrectly delayed until
  final analysis. The entire method comparison is quarantined, including two individual
  local-pass flags. [Result and cause](docs/astra-a2-harness-result-2026-09-05.md).
- Added a 136-record grouped diagnostic index with explicit campaign invalidity; unknown
  contact/quality stay unknown. No invalid records are promoted into demonstration data.
- Frozen [A2r isolation repair](docs/astra-a2r-isolated-protocol.md): explicit unique terrain
  patches, 32 matched slots reused across methods, immediate apparatus checks, unchanged
  references/selection/endpoints. Seven launches / 224 actual rollouts; measured aliases
  initial instead of spending duplicate execution budget. Still development, no training.
- Full CPU regression on A2: 975 passed. Subsequent focused isolation/record tests: 35
  passed. Next: execute A2r, verify every batch, then diagnose valid motion outcomes.

### 2026-09-05 — A2r complete: constructive native ducking, weak coverage

- Executed all seven isolated, slot-matched batches: **224/224 rollouts**, all apparatus
  checks pass. Fixed-extra duck: **2/32 local passes**, both at 1.15 m (`u4`, two of four
  repeats), 25.0/22.6 mm achieved conservative clearance. Both paired absent runs pass;
  full-route completion remains false. Initial and resampling: 0/32. Measured is an exact
  no-op/alias, not an independently improved method. No unit reaches 3/4-slot repeatability.
- Free-height local progress is only 12/32; initial 6/32, deeper 3/32, resampling 16/32.
  Carrier quality still limits coverage. Actual root arrival lags reference by a median
  1.14 s among 69 observed non-free absent crossings; do not impute the 27 nonarrivals.
- Archived [224 valid execution records](outputs/astra_a2r_execution_records_v1/index.json)
  separately from 136 quarantined harness records. Eight shared development groups prevent
  raw/repair/absence/presence/replica leakage. Contact and full quality remain unknown.
- [Result note](docs/astra-a2r-results-2026-09-05.md) states the existence-only result and
  posture-proxy limitation. Full regression: **983 passed in 210.17 s**. A2r peak VRAM
  3791 MiB; minimum available RAM 4679 MiB; SONIC wall 390.58 s. No training.

**Next action — A3 design, not yet a launch protocol:** derive per-carrier achieved-body
overlap and clearance response from matched absent development runs. Implement a smooth
**dual-window hold** covering reference and predicted achieved overlap; compare with a fixed
long-hold and fixed-depth baseline at equal information/cost. Do not simply delay the dip
and sacrifice reference clearance, or use one median lag for every carrier. Qualify the
edited carrier's progress and task-appropriate posture first, then preregister fresh units.
The 11 deep-dip upright/low-pelvis flags motivate validating a duck-specific posture rule,
not retroactively changing A2r's endpoints. No campaign remains running after this milestone.

### 2026-09-05 — A3 narrowed to a timing mechanism test before adaptation

- [Absent-only whole-body overlap audit](outputs/astra_overlap_windows_v1/index.json):
  73/128 observed forward exits, 55 unobserved; preserve disconnected overlap intervals
  and censoring. Only 59/128 satisfy the proposed <=4 s dual-window hold budget. These are
  unexecuted development proposals, not a calibrated execution bound or admission proof.
- Freeze [A3 hold protocol](docs/astra-a3-hold-protocol.md): eight fresh seeds 61000–61007,
  fixed 0.28 m depth, holds 2/3/4 nominal seconds, plus free-height absent/present control.
  32 references, 256 executions in eight isolated 32-slot launches, one reused geometry.
- This intentionally precedes the adaptive dual-window method: vary timing alone to test
  early recovery, establish a strong fixed-long-hold baseline, and measure progress cost.
  Do not add a second depth sweep, change spent posture gates, or train on sparse positives.
- Candidate deliverables remain conditional: censored overlap descriptor, bounded native
  hold intervention, and paired execution records. Next: frozen clean-clone generation,
  release ARDY, execute with immediate apparatus/resource checks, then decide from evidence.

### 2026-09-05 — A3 generated; 64/256 executions complete, resource-paused

- Source `3e92cd3`, [progress and continuation](docs/astra-a3-progress-2026-09-05.md).
  All 32 references generated once. Hold2/3/4: each 8/8 reference clear; free nominal 0/8.
  This is not an execution improvement. The two free-control batches completed 64 executions
  with valid apparatus; all six hold-arm batches (192 executions) remain unlaunched.
- [256-entry ledger](outputs/astra_a3_progress_v1/index.json) retains pending assignments
  with unknown outcomes. Two preflight RAM refusals are preserved; no scientific retries,
  resource-gate changes, or co-tenant interruption. Last observed available RAM remained
  below 9500 MiB. Completed execution cost 121.66 s; peak own VRAM 3791 MiB, min RAM 3316 MiB.
- Full regression: 993 passed in 210.57 s; subsequent progress test also passes. Added the
  complete-campaign grouped execution/overlap index builder, which refuses partial input.
- **Immediate next action:** resume `--stage sonic` from `/tmp/astra-a3-hvxxHR/repo` at the
  original commit when the measured gate passes; never regenerate. Then analyze all arms,
  rebuild-check records, and choose between timing refinement and carrier redesign based
  on paired clearance/progress. No ASTRA job is left running at this handoff.

### 2026-09-05 — Resource optimization requested, equivalence probe frozen

- Add a standard-library-only resident supervisor; preparation and verification exit before
  simulation. The old A3 import/loaded-input worker measured 704692 KiB peak RSS (~688 MiB).
- [Resource probe](docs/astra-resource-probe-protocol.md) compares default allocation with
  two glibc allocator settings at the **same 32 slots**, using archived free controls.
  128 operational replays, no generations or extra independent research examples.
- Keep 5500/9500 MiB launch and 1200/2500 MiB running floors while measuring. No precision,
  terrain, timestep, checkpoint or batch changes. Only exact achieved-state equivalence can
  justify an unchanged-result claim; otherwise keep A3 unchanged and report the difference.
- Focused validation: 12 passed. Next: benchmark RSS/wall time and exact output equality;
  derive a versioned workload-specific execution profile only if the measurements support it.
- Initial resource-probe preparation `9841b66` resolved the external release config as if
  it lived in the tracker worktree. The hash check refused before any Isaac launch. Preserve
  `astra_resource_probe_v1` (zero executions); use the manifest's authoritative resolved
  source paths in a fresh preparation. Scientific workload/protocol remain unchanged.

### 2026-09-05 — Low-memory path reproduces all operational controls

- `astra_resource_probe_v2`: 128/128 operational replays complete. Both baseline/compact
  allocator profiles reproduce original absent/present achieved archives and apparatus
  **exactly**. These replays add no independent scientific samples.
- Resident supervisor sampled ~16 MiB vs old importer peak ~688 MiB. Isaac child peaks:
  baseline absent/present 6144/6213 MiB; compact 6042/6052 MiB. Compact wall time rises
  ~1–2.5%; report this tradeoff, not a compute-speedup claim.
- [Measured profile](outputs/astra_resource_profiles_v1/index.json) sets compact admission
  to 9216 MiB available RAM from worst child+supervisor peaks +2500 MiB reserve +512 MiB
  uncertainty, rounded up. VRAM/floors unchanged. It is workload-specific, not universal.
- Freeze [A3 lean continuation](docs/astra-a3-lean-continuation-protocol.md): only the 192
  unlaunched hold executions, same 32 slots/arrays/physics/endpoints; reuse 64 original controls.
  Fresh provenance-linked output, worker-exit phase separation, bounded waits, immediate
  apparatus validation. Never resize batch silently, regenerate, or retry spent failures.

### 2026-09-05 — Optimized continuation advances A3 to 96/256

- [Resource optimization report](docs/astra-resource-optimization-results-2026-09-05.md):
  128 operational replays exactly reproduce controls. Compact child RSS saves 102–161 MiB;
  phase separation removes the heavy waiting importer. No scientific batch/precision change.
- Continuation source `d6f18f7`, `/tmp/astra-a3-lean-zbs5Cx/repo`:
  `hold2_absent` completes 32/32 with valid apparatus, peak child RSS6037 MiB, no abort.
  Scientific total **96/256** including original controls; **160 pending**, all unknown.
- RAM then fell below 9216 MiB; the 180 s bounded wait expired without launching hold2-present.
  No ASTRA process remains running. Resume the existing lean queue, not old A3 generation or
  preparation. [Joined ledger](outputs/astra_a3_lean_progress_v1/index.json) keeps planned rows;
  operational replays are excluded from scientific counts.
- Full hold comparison still awaits completion. Next technical option, if more savings are
  needed, is a separately tested inference-only loader; unused optimizer/trainer state was
  found in the release path, but no such loader change has yet been made.
- Verification: full CPU suite **1004 passed in 211.33 s**, plus the later joined-progress
  test; resource profile and joined ledger rebuild exactly. No remaining ASTRA worker.

### 2026-09-05 — A3 complete: freeze 3 s as the next strong fixed baseline

- Completed the remaining **160** executions using the frozen lean queue: **256/256**
  scientific executions, all apparatus checks pass, no resource abort or regeneration.
  The 128 operational allocator replays remain excluded. [Full result](docs/astra-a3-results-2026-09-05.md).
- Present local passage, with all 32 slots retained per arm: free **0**, hold2 **13**,
  hold3 **19**, hold4 **11**. Hold3 gains six matched passes and loses none against hold2;
  >=3/4-repeat units are 0/8, 3/8, 4/8, 2/8 respectively. One reused geometry, eight
  motion seeds, one physics seed: not a geometry holdout or significance result.
- Hold3 also improves absent local progress to 19/32. Its absent/present pass labels agree
  in all slots. Do not attribute all gains to clearance alone; carrier progress/recovery
  remains part of the bottleneck. Hold4 is worse despite a longer crouch.
- [Execution records](outputs/astra_a3_execution_records_v1/index.json) retain 256 paired
  records in eight development groups, original/continuation provenance, signed clearance,
  censored body overlap, cutoff/posture flags. No present arm completes the full route;
  actual contact and full demonstration quality remain unmeasured.
- The six lean batches use 352.60 s SONIC wall time, peak child RSS6069 MiB. Keep the
  validated 9216 MiB admission profile; unused capacity is not permission to change the
  scientific batch, precision, or endpoints.
- Validation: **1006 CPU tests passed in 245.44 s**; joined endpoint and record-index
  rebuilds are exact. All assigned A3 work is finished; no ASTRA process is left running.

**Next work, in order (design plan, not yet an execution preregistration):**

1. **Preserve the fixed baseline and qualify the edited carrier.** Hold3 is a fixed
   heuristic, not our final feedback method. Use A2r/A3 as development only. Specify the
   absent-only qualification rule and count every source seed, calibration rollout and
   rejection. Do not select carriers using their obstacle-present successes.
2. **Build the smallest adaptive scheduling test.** Use the reference body-overlap interval
   plus an absent-only achieved-overlap estimate to propose a bounded hold window. Keep
   the current reference window covered; censor missing exits rather than inventing a lag.
   Retain fixed3 as fallback/comparator, and explicitly refuse if progress, reference
   clearance or the finite timing budget fails. A changed generation must be reverified;
   the original trajectory's timing is not a guarantee for its edited version. Freeze
   update count and cost before any fresh present execution. Do not train a large model yet.
3. **Prospective comparison with equal information.** After implementation tests, freeze
   new disjoint motion seeds and beam configurations, including a geometry-disjoint test
   axis. Compare fixed2, fixed3 and adaptive at equal generator/calibration-launch budgets;
   include raw controls. Match slots/physics seeds, but analyze at motion/scene-group level.
   Report all-assigned passage, qualification coverage, conditional passage, deformation,
   latency and launch cost. No retrospective A3 selector may be described as a blind test.
4. **Quality gate before positive-data release.** Independently inspect achieved-state
   velocity/acceleration, stance drift, self/floor intersection, recovery and task completion.
   Calibrate any new limits on separate carriers; keep A3's endpoint unchanged. Validate
   actual obstacle-contact logging with known-contact/absent probes before asserting
   contact-free execution. Until then, release these as operational execution records,
   not high-quality humanoid demonstrations.
5. **Dataset utility only after label quality.** Keep each source carrier and all its
   variants in one split, including reused carriers across scenes. Freeze scene-family
   holdouts before training the lightweight risk ranker; compare identical inputs with
   binary versus additional signed execution supervision. No execution-derived input
   leakage, no claimed navigation gain from reference geometry alone.

**Go/no-go:** adaptive scheduling must beat the now-strong fixed3 baseline on fresh
all-assigned passage without hiding calibration launches or sacrificing motion quality.
If it does not, retain the honest fixed-control result and improve the carrier/interface;
do not expand hold sweeps or rename selection as constructive repair.

### 2026-09-05 — A4 one-step adaptation protocol frozen

- [A4 protocol](docs/astra-a4-adaptive-hold-protocol.md): fresh seeds62000–62007, same beam
  for a mechanism test before geometry transfer. Fixed3 is the strong comparator; fixed2
  and free nominal remain controls. 32 actual generations,256 executions including64
  initial absent controls; decisions frozen before any present launch.
- New bounded scheduler uses four fixed3 absent repeats: >=3/4 progress passes, all exits
  observed, reference/achieved dual-window maximum plus.10 s within[2,4]. Otherwise explicit
  fixed3 fallback, charged as a generation; no post-hoc deletion or adaptive retry.
- Retain the qualified fixed3 carrier only as calibration information, not as a guarantee
  for the changed generation. Execute all finite candidates in this simulation probe and
  report both all-assigned results and qualification coverage. No deployment safety claim.
- Eight focused tests pass, covering fresh assignments, matching constraints, source-hash
  checks, immutable decisions, absent-only inputs, censoring, bounds and fallback semantics.
  Next: clean-clone initial generation, lean absent queue, freeze decisions, final generation
  and paired execution. Keep CPU tests separate from memory-constrained Isaac launches.

### 2026-09-05 — A4 attempted; preflight protects unspent seeds

- [A4 implementation and continuation](docs/astra-a4-progress-2026-09-05.md): driver/scheduler
  frozen84e649e in `/tmp/astra-a4-6VBizH/repo`. Initial generation stopped at its RAM gate
  before loading the model:7327<8192 MiB. No sample was generated or executed.
- A later check after CPU testing still fails (RAM5934 MiB; free VRAM4141 MiB). Preserve
  this host-pressure boundary, not a new method failure. No co-tenant or gate was changed.
- [Progress ledger](outputs/astra_a4_progress_v1/index.json):256 pending assignments with
  unknown outcomes,0 generated references. Resume `generate_initial` only once its unchanged
  gate passes; then initial queue → calibration → final generation/queue. No active worker.
- [Absent-only development check](outputs/astra_adaptive_development_v1/index.json):4/8 A3
  carriers qualify for timing edits; proposals2.34–3.48 s. Other carriers retain fixed3.
  These are unexecuted proposals, not improved traversal. The artifact rebuilds exactly.
- Complete A4 analysis now prices shared calibration and standalone fixed3 separately,
  retains losses/fallbacks and groups all variants by source seed. Full CPU suite1016 passed;
  two later tests also pass in the12-test focused suite. Progress and dry-run rebuilds exact.

### 2026-09-05 — A4 complete: the overlap-only timing rule does not advance

- Completed all **32 generations and 256 executions** from the frozen84e649e clone.
  All apparatus checks pass, no retries or resource aborts. The historical resource refusal
  and zero-progress snapshot stay unchanged. [Result](docs/astra-a4-results-2026-09-05.md).
- Present local passage /32: free0, fixed3=19, fixed2=18, adaptive16. Units with >=3/4
  present passes /8:0,4,5,3. Adaptive gains0 and loses3 matched passes against fixed3;
  fixed2 gains8 and loses9. Keep both directions and group repeats by motion seed.
- Four units qualify for absent-only timing edits. All losses are a4u2, whose hold changes
  from3.00 to3.06 s: absent progress3/4→0/4; present passage4/4→1/4. The three lost repeats
  preserve archived geometry/posture but miss the4 s recovery deadline. Calibration of the
  original generation did not qualify the edited one. Do not relabel late recovery as success.
- Four fallback references are array-identical to fixed3; all16 fallback present local
  labels agree. One fallback cutoff flag differs, so do not assert full execution identity
  or attribute every batch difference to adaptation. Contact and quality remain unknown.
- All256 paired records are in [the complete receipt](outputs/astra_a4_adaptive_d0/summary.json),
  with eight source-motion split groups, timestamps, reference/achieved payload paths,
  signed clearance, independent flags and full costs. No high-quality positive bank yet.
- Simulator wall465.18 s, child peak6101 MiB, supervisor16.86 MiB. All generation/physics
  settings and resource gates retained. Full CPU regression **1018 passed in224.90 s**.
  Complete endpoint/calibration receipt rebuilds exactly; no ASTRA worker remains running.

**Next work, in order (not a new campaign preregistration):**

1. **Keep the result and stop overlap-only expansion.** Fixed3 stays the primary comparator;
   retain fixed2 because it succeeds on different carriers. No larger hold sweep, spent-seed
   retuning, deadline relaxation or training before a better-defined interface exists.
2. **Audit the archived edited carriers before spending more GPU launches.** Compare aligned
   root progress, body-exit/recovery lag, joint velocity/acceleration, floor behavior and
   stance drift for fixed/adaptive successes and failures. Keep root-only timing distinct
   from whole-body completion and distinguish posture proxies from physical falls. The
   immediate question is why preserving beam overlap did not preserve timely recovery.
3. **Revise only the measured bottleneck.** Any successor must account for recovery/progress
   and validate the final edited reference; qualification cannot be inherited from fixed3.
   Test a bounded revision against both fixed heuristics on fresh source groups under a
   preregistered calibration/generation/launch budget. Choosing an archived winner with
   present labels is a development oracle, never the deployed method.
4. **Keep dataset quality and geometry transfer as gates.** Validate actual contact logging
   and achieved-motion quality before positive demonstrations; then freeze a geometry-disjoint
   test. All A3/A4 variants remain development. Delay the risk-ranker utility experiment
   until execution labels and grouped holdouts support it.

### 2026-09-05 — Archived progress/quality audit completed without GPU work

- [Audit](docs/astra-a4-quality-audit-2026-09-05.md) covers32 references/256 executions;
  zero new generations or simulations. Added exact shared-clock alignment, per-joint
  dynamics, censored root-error measurements and explicitly non-contact foot-plane proxies.
  Dataset groups and historical outcomes remain unchanged; no new admission thresholds.
- a4u2 reference root x at4 s increases only8 mm, but achieved positions decrease54–286 mm
  across four paired repeats. Median signed root error -1.628→-1.810 m. Executed peak joint
  speed/acceleration both decrease in all four pairs; smoothing-only is not an established fix.
- [Static interface inventory](outputs/astra_sonic_interface_v1/index.json) verifies the
  launch config/source: G1's configured inputs have no explicit horizontal root-position
  error, while release mode sampling includes multiple encoders. A4 lacks per-slot active
  encoder logging; do not infer G1-only execution or a mode-specific cause from the config.
- **Immediate next task:** instrument resolved encoder/command-mode readback and progress
  before freezing any explicit-mode experiment; keep contact logging/quality qualification
  separate. Mode changes require versioned controls, never a silent A4 alteration. Then
  evaluate a bounded progress-aware revision on fresh groups, qualifying its final output.
- Quality/interface artifacts rebuild exactly. Full CPU1030 passed; four later inventory
  tests pass within16 focused tests. No GPU or background worker is left running by ASTRA.

### 2026-09-05 — Runtime encoder logging validated in64 exact operational replays

- [Result](docs/astra-controller-readback-result-2026-09-05.md),
  [receipt](outputs/astra_controller_readback_v1/summary.json), preregistration `c7c864f`.
  Two32-slot fixed3 replays preserve every valid achieved state, termination flag,
  timestep and apparatus readback exactly. No generation, training, new scientific samples,
  historical relabels, retries or resource aborts.
- Each replay uses14 G1/18 teleop encoder slots; matched absent/present assignments,
  zero valid-prefix transitions. All26,432 logged action masks agree with the pre-step
  command. Mode weights alone did not reveal this executed mixture. These are runtime
  labels of the replay, not retroactive measurements of uninstrumented A4 arms.
- The logger reads the policy's input before physics/reset, never changes the mode, and
  preserves valid-prefix alignment. The explicit mode field must travel with future
  execution records; replay variants retain the source split group and add no independence.
- **Decision:** freeze a prospective mode-controlled comparison before another method
  revision. Compare explicit G1, explicit teleop and the default sampler on matched fresh
  carriers; validate runtime mode and final edited progress/recovery. Do not assume G1 is
  better, or attribute A4's loss to mode sampling without that intervention. Keep the
  contact/quality gates and fixed2/fixed3 comparators. No new campaign is preregistered here.
- Total simulator wall114.70 s; peak child6067.31 MiB, supervisor16.63 MiB. The validated
  compact profile and9216/5500 MiB launch gates remain unchanged; both jobs have exited.
- Full CPU regression1051 passed in220.16 s; two later summary tests pass within19
  focused tests. The complete operational summary rebuilds exactly; no worker remains.

### 2026-09-05 — A5 explicit-mode comparison completed, no win over default

- [Protocol](docs/astra-a5-encoder-protocol.md), committed `df69290` before generation;
  [result](docs/astra-a5-results-2026-09-05.md),
  [576-row receipt](outputs/astra_a5_encoder_d0/summary.json). Seeds63000–63007 are spent.
  Three native-reference arms × three modes × absent/present × eight units × four slots.
  All24 references frozen before execution; no selection, resampling, training or retuning.
- Every mode passed the frozen minimal free-walk reachability gate: two units with4/4
  absent passes. All units continued to present execution, including unqualified ones.
  Present fixed3 default/G1/teleop:17/17/15 out of32; units passing>=3/4:4/4/3 out of8.
  Fixed2:14/15/13; free:0/0/0. All full-route counts remain zero.
- Primary G1-vs-teleop: two paired wins, zero losses; per-unit differences
  `[.25,0,0,0,.25,0,0,0]`. G1-vs-default: one win/one loss, zero net gain in every unit.
  No significance or geometry-generalization claim. A5 does not support a new mode-based
  repair contribution or retrospectively explain the earlier adaptive loss.
- All199,966 valid action samples have matching observation/command modes and contiguous
  clocks. Explicit requested branches remained active; default measured G1/teleop.
  All variants stay in `astra-a5/a5uN` development groups; contact/quality remain null.
- Simulator wall1112.65 s, maximum child6104.58 MiB and supervisor16.88 MiB RSS;
  minimum observed available RAM/free VRAM8825/3232 MiB. Same compact profile,32 slots,
  unchanged9216/5500 MiB admission and2500/1200 running floors. No aborts or retries.
  CPU1071 passed in227.05 s. Full receipt rebuild matches exactly; all workers have exited.

**Updated next work (not preregistered):**

1. Keep default fixed3 and fixed2 as strong controls. Explicit G1 can be a fixed,
   mode-transparent development condition, not a claimed performance upgrade; retain
   default in any future matched comparison. Do not train a present-label mode selector.
2. Audit A5 final edited references and achieved states for progress, recovery and
   motion quality. Use aligned clocks and explicit censoring. In a post hoc check, all
   fixed3 present failures miss whole-body dwell by4 s, including early cutoffs; this is
   not proof of a single physical cause. Do not inherit qualification from free WALK.
3. Validate actual-contact logging and quality measurements on preregistered operational
   probes with positive/negative controls and read-only replay equivalence. Current local
   successes are not yet a high-quality imitation bank.
4. Only then freeze one bounded correction that must preserve final-output progress as
   well as clearance, fit from permitted absent data, and test fresh carrier–scene groups.
   No overlap-only extension, relaxed deadline, broad sweep or large-model training.
5. Integrate qualified positives and failures into grouped records, freeze a geometry
   holdout and measure pre-execution consumer utility before a comprehensive dataset claim.

### 2026-09-05 — A5 archived quality audit and contact-buffer groundwork

- [Audit](outputs/astra_a5_quality_audit_v1/index.json) covers24 references and576
  executions with source/analysis hashes and exact timestamp alignment. It rebuilds
  exactly; no original receipt, endpoint or split group changed. One achieved archive
  is loaded at a time; no generator or simulator launch was needed.
- Default fixed3 present passes:17/32, median root error at4 s−1.201 m; all17 have
  positive one-second post-dwell displacement (median0.592 m). Failures:15 assigned,
  only6 observed at4 s (median error−1.898 m),9 censored. Smaller valid-prefix joint
  RMSE among failures does not imply better quality; prefixes differ. z=0 foot-depth
  measurements remain geometric proxies, not terrain-relative penetration/contact.
- Implemented CPU contact identity/buffer validation and control-step aggregation.
  Opposing normal vectors cannot cancel scalar impulse sums; zero-force points,
  transient substep events, invalid/saturated buffers and missing clocks are explicit.
  Live PhysX callback/export integration and simulator validation remain unfinished.
- [Draft next gate](docs/astra-contact-validation-plan.md): six scripted positive/negative
  probe executions across logger/capacity settings, then64 matched G1 operational
  replays if probes pass. Zero new scientific units; freeze source, resource budget
  and checks before launch. Do not infer contacts in original A5 from replay labels.
- **Decision:** finish contact instrumentation and final-edited progress/recovery
  qualification before fresh bounded correction. No mode sweep, spent-seed tuning,
  arbitrary quality threshold or large-model training. Physical contact/high-quality
  qualification remain unknown; the research objective is not yet complete.
- Validation:1093 CPU tests passed in224.47 s;30 focused quality/contact tests passed.
  The complete audit rebuild matches exactly. No ASTRA simulator worker was launched.

### 2026-09-05 — Contact reader and exact G1 replay validated; excluded events unresolved

- [Result](docs/astra-contact-validation-result-2026-09-05.md): six corrected positive/
  negative/logger/capacity probes pass. Signed PhysX force scalars are preserved; impulse
  magnitude cannot cancel. All prior timeout/error buffers and receipts are retained.
-64 G1 operational replays match A5 states, valid lengths, termination/progress,
  apparatus and modes exactly. All30 body paths are checked. Present valid prefixes
  have zero observations, including17 local passes and15 failures;10 terminated.
  These are linked replay measurements, not retroactive A5 contact labels.
- The reader's untrimmed peak6 points exposes a prefix-coverage limitation. Sparse
  all-action event logging is implemented and tested, but its32-slot diagnostic
  stopped on free VRAM1158 MiB<1200 MiB before producing output. Do not auto-resume,
  reduce the frozen batch or infer where the contacts occurred. No co-tenant stopped.
- [Accounting](outputs/astra_contact_validation_v2/summary.json):15 process launches,
  six unsuccessful,655.703 s total; successful G1 replay135.707 s, peak6101.09 MiB RSS.
  Full CPU1126 passed in242.15 s;52 focused tests include later event-scope regressions. No worker
  remains and no training or independent scientific unit was added.
- **Decision:** version the excluded-event continuation only when resource gates pass;
  resolve G1 positive-contact sensitivity. Keep prospective final-edited quality
  qualification and one fresh progress-aware correction as the next method gate.
  A5 default fixed3 remains17/32 operational local passage, not a new performance gain.

### 2026-09-05 — Recovery-aware pool ceiling and resource-only continuation

- [Result](docs/astra-a5-recovery-selection-result-2026-09-05.md): all288 A5 present
  assignments retained. All91 local passes have complete one-second recovery windows;
  89 meet continued kinematic bounds and strictly positive root displacement.
  The fixed2/default exception is essentially stationary (−0.313 mm), not a fall;
  the fixed2/teleop exception backtracks43.449 mm and recrosses the root recovery line.
- An absent-only, paid-execution bank selects18/32 versus fixed3's17/32, one gain and
  no losses. The single extra local pass is the near-stationary case. Under the
  post-hoc recovery conjunction, selection, fixed3, unitwise oracle and slotwise union
  all reach17/32. Eight development groups, one geometry; no prospective gain claim.
- Three CPU receipts rebuild exactly; one archive loaded at a time and at most51
  frames per recovery FK segment. Full CPU1148 passed in213.60 s; final focused36
  passed, including14 tests added after the full-suite collection.
- Contact continuation protocol committed at `acb6c21`;180 s wait yielded10 preflight
  refusals with4306–4307 MiB free VRAM below5500. RAM passed. No process launch,
  new reference, seed, training or background poller. Preserve the pinned clean clone
  and all preflights for later gated execution; original event abort unchanged.
- **Decision:** keep fixed3/default as strong baseline; no selector training or
  broader timing sweep on this spent pool. Complete contact coverage, then define
  prospective final-edited qualification and test one bounded progress-aware candidate
  modification on fresh groups. Contact freedom and high-quality dataset remain open.

### 2026-09-05 — Recovery sensitivity and public progress presentation

- Added a CPU-only strict displacement sweep at−1,0,10,50,100,250,500 mm on the
  frozen default fixed2/fixed3 development pool. Historical local labels stay unchanged;
  incomplete recovery and observed bound violations cannot pass via a relaxed displacement.
  No threshold is recommended as a motion-quality standard.
- Fixed3 conjunction counts:17 at every threshold through250 mm,15 at500 mm. Slotwise
  oracle:18 with−1 mm tolerance,17 from0 through250 mm,15 at500 mm. Unit-fixed-hold oracle
  agrees. Thus the extra selection opportunity is near-stationary, while all17 fixed3
  local passes show appreciable continuation in this one scene. Do not tune future
  methods or claim held-out generalization from this spent pool.
- Built shared current-progress prose, method stages, all nine A5 rows, losing A4
  comparison, recovery sensitivity, contact limits and dataset gates. Public snapshot
  validates row membership/counts and receipt hashes; the old stepping section remains
  explicitly historical. Videos remain historical stepping evidence, not A5 footage.
- Dedicated static publication exports selected committed docs/JSON/assets only; no
  pickle, qpos payload, checkpoint, process log or private draft. Preserve research
  `master` locally; GitHub Pages can use `gh-pages:/docs` without publishing that history.
- The resumed resource-only continuation waited180 s;10 new preflight refusals,
  RAM9421–10341 MiB and VRAM4067–4072 MiB. RAM passed but VRAM did not. No simulator
  process, generator, training or autonomous poller remains. Contact sensitivity and
  excluded-event attribution remain the next execution gate.
- Validation:1185 CPU tests passed in211.46 s;23 focused publication/sensitivity
  checks passed. Chromium desktop/light and mobile/dark checks found no script errors;
  both pages fit390 px after fixing historical code-path wrapping. Sensitivity rebuild exact.
