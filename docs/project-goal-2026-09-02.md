# Scene2Motion — project goal, title, abstract and vocabulary (2026-09-02, rev. 3)

Adopted from the advisor's three reviews (`docs/pi-advice-2026-09-02.md`,
`-b.md`, `-c.md`). This is the single statement of what the project is trying to show and how it
is worded. It supersedes the three-pillar framing in
`docs/framing-2026-09-02-contract-repair-engine.md` and the "Contract Before Control" title of
the Codex draft. Receipts, protocols and the ledger are unchanged.

## The question

> **Which generated motions can this controller use to get through this obstacle, from this
> starting state?**

## Title

**Scene2Motion: Evaluating Generated Humanoid Motions for Obstacle Traversal**

## Project goal

> Determine when generated humanoid motions can complete obstacle traversal, and test whether
> scene-aware selection and measured correction improve completion under a fixed controller and
> computation budget.

## Central message

> A motion can resemble a step or a duck without clearing the intended obstacle. Even a
> reference that clears the obstacle may lose clearance when tracked by a humanoid controller.
> Scene2Motion separates these failures by measuring reference geometry, controller outcomes,
> and achieved clearance. Current experiments reveal limitations in generated stepping motions
> and show that measured correction can improve ducking-reference clearance. Obstacle-present
> traversal remains to be validated.

The strongest current contribution is an evaluation of where generated motions fail plus a
promising correction method. It is not a demonstrated obstacle-traversal system, and the paper is
coherent only if that boundary is kept.

## Two honest papers; the experiments decide which one this is

| if the evidence lands as | the paper is | promised now? |
|---|---|---|
| where generation, selection and tracking fail for obstacle-relative tasks | an **evaluation contribution** | yes, this is what we have |
| a specific selection or correction mechanism improves actual traversal under stated conditions | a **method contribution** | no, not until EXP-029 measures it |

Do not promise the second before measuring it. The title works for either.

## Local traversal is not navigation

- **Local traversal:** pass through a specified opening or corridor. Walking around it does not
  satisfy the task. This is what the project measures.
- **Navigation:** reach a destination through a scene. Walking around an obstacle may be an
  excellent solution; ducking or stepping is required only when the task or route demands it.
  Navigation adds route choice, transitions between motions, replanning and recovery.

Establish local traversal first. Successful isolated clips do not establish navigation, and the
two must never share a success label.

## Scientific identity and writing rules

Start with a concrete robot problem, define what success means, and let each experiment support
one specific claim. Three principles from *Moving Through Clutter*:

1. Start with the obstacle, not the software pipeline.
2. Explain the missing capability: producing a stepping motion does not ensure it happens at the
   box.
3. Separate evidence levels: a collision-free reference, a completed tracking run and a
   successful traversal are different results.

Two more, from the TEXEDO reading:

4. **Name the conflict.** Selecting a motion the controller can execute is not the same as
   selecting a motion that solves the scene task. Where two desirable properties conflict, show
   the conflict and say how the decision rule handles it.
5. **Make each design choice necessary.** Not "our framework has a scene module, a support module
   and a repair module", but "we measure clearance at the specified obstacle because a motion can
   clear the same obstacle elsewhere along its route" — followed by the experiment testing
   whether that measurement improves decisions.

**Shorten explanations, not definitions.** Plain words are welcome; the measurement behind each
number stays exact. Experiment identifiers belong in reproducibility records, not in the
reader's main explanation of the paper.

## Measurement definitions (exact; do not simplify)

- **Produced (stepping).** In a whole-body geometry check, the reference clears a 3 cm box at
  some tested position along the route. A box-clearance statement, not a foot height.
  exp021: 44 of 64.
- **Event timing.** Among the 49 references with positive clearance somewhere along the route,
  the reference root reaches the location of maximum box clearance after a median of 1.4 s
  (frame 35; 40 of 49 within the first 50 frames). Not a detected foot-crossing event.
- **Placed.** The reference clears the obstacle at the position the scene specifies, not at a
  position chosen to fit the clip. exp021: 12 of 64 at 5 cm, 11 of 64 at 8 cm, at x = 1.2 m; the
  position was selected after inspecting the clips and is fixed for the replication.
- **Support test.** A foot meets it at a frame when its sole is within 4.65 cm of the ground and
  its planar speed is below 1.18 m/s (calibrated on tracker-successful references before any
  stepping experiment). The **reference screen** rejects a reference whose longest period with
  neither foot meeting the test exceeds 0.20 s. It predicts the controller evaluator's stopping
  rule. It does not establish zero contact force, a fall, or physical impossibility.
- **Valid step passing the support test** (EXP-024 column). A locally valid step by the 13-gate
  evaluator *and* no period over 0.20 s outside the support test. 0 of 32 in every arm, while
  33 of 128 pass the screen alone.
- **Completed tracking run.** The controller followed the reference to the end without the
  evaluator's tracking-error cutoff (pelvis height error > 0.25 m, orientation > 1.0 rad,
  ankle/wrist height error > 0.25 m, ankle position error > 0.2 m; release configuration).
- **Clearance preserved after tracking.** A reference that clears the obstacle yields a tracked
  trajectory that passes through the obstacle corridor, finishes beyond the obstacle, remains
  collision-free at the graded height, and satisfies the stated termination rule. A geometry
  query alone is not this endpoint: a robot that stops before the box looks collision-free.
  exp021/EXP-022A: 0 of 64, and still zero with the termination condition removed.
- **Local traversal completion** (the endpoint EXP-029 will use, with the obstacle present in
  the physics scene): passes through the specified corridor, finishes beyond the obstacle, no
  prohibited contact, no fall, within the time limit. Never yet measured.
- **Upright at the last archived state.** Pelvis height 0.56–0.95 m at the final frame, none
  below 0.5 m. Say exactly this; do not say "did not fall".

## Evidence levels (report each separately; never let one stand in for the next)

| level | stepping (exp021/EXP-022A, 64 refs) | ducking |
|---|---|---|
| produced | 44/64 | — |
| placed (clears at the specified position, 5 cm) | 12/64 | reference geometry: see repair table |
| collision-free reference | 12/64 | TCN 72.9 % → 99.3 % after two corrections |
| completed tracking run | 11/64 | not evaluated under a usable protocol |
| clearance preserved after tracking | 0/64 | 0/859 under a protocol confounded by a 14 s clip cap |
| local traversal completion, obstacle present | not tested | not tested (EXP-029) |

## Metrics discipline

Prioritise, and report separately:

- **Traversal completion** as defined above; **navigation completion** only under stated
  route-choice rules, and not before it is measured.
- **World-space progress and clearance**, not only pose similarity after subtracting the root
  position.
- **Outcome breakdown**: success, collision, fall, evaluator cutoff, timeout or stall before the
  obstacle, rejected before execution.
- **Cost**: generation calls, scoring latency, executed duration, interventions. "Same maximum
  number of generations" is not equal computation.

**Report success over all assigned trials.** A rule that rejects a candidate without executing it
scores that trial as a non-completion; if the rejection avoided an unsafe attempt, report that
separately as an outcome class. Rejecting everything must never look like perfect task
performance. Keep a learned or calibrated score distinct from a physical guarantee: where no
candidate passes a screen, the honest options are rejection, resampling or replanning, not
quietly executing the best-scoring candidate anyway.

## Is it selection, or a missing candidate?

Report **pool coverage** (does any candidate in the pool complete the task) separately from
**selected success** (does the deployed rule pick one that does). High coverage with low selected
success is a selector problem; low coverage at every budget is a candidate problem that no
ranking function can fix; clearing references that fail in execution are a controller-
compatibility problem.

For the stepping family this is already answered retrospectively
(`experiments/analyze_pool_coverage.py` → `outputs/analysis_pool_coverage/summary.json`), at the
specified obstacle, 5 cm, n = 64, under the replay endpoint:

| | count |
|---|---|
| reference clears at the specified position | 12 |
| completes tracking without a cutoff | 11 |
| **both** | **0** |
| passes the corridor and finishes beyond | 14 (11 without a cutoff) |
| never reaches the obstacle | 50 |
| satisfies the full endpoint | 0 |

The two sets are disjoint, so no selector over this pool could have succeeded whatever its
ranking function. Reference clearance reaches 90 % coverage at about 12 fresh draws (5 cm; 13 at
8 cm, the ledger's N90 convention), while joint coverage stays at zero for every budget up to the
whole pool. For stepping, the limitation is the candidate pool, not the selection rule. Scope:
obstacle absent from the physics scene, one route, one scene, physics seed 0, one rollout per
reference.

## Abstract (adopted; unchanged from rev. 2)

Generating a humanoid motion is not enough to traverse an obstacle: the motion must occur at the
right place and remain collision-free when tracked. We present Scene2Motion, a framework for
evaluating and repairing generated motions while keeping the motion model and controller fixed.
Using ARDY-G1 and SONIC in simulation, we separate motion generation, obstacle-relative
clearance, and clearance after tracking. Of 64 step-prompted references, 12 cleared a 5 cm box at
a position selected after inspecting the motions; none retained clearance after tracking.
Tracking ran without the obstacle, with recorded robot states subsequently checked against its
geometry. A foot-support screen flagged all 53 references whose rollouts triggered the configured
tracking-error cutoffs, while also rejecting three of the eleven runs that did not terminate
early. This analysis predicts evaluator behavior, not whether the robot would fall. For ducking
motions, we use measured clearance errors to adjust the generation controls. Across 36 beam
scenes and eight seeds, up to two repairs increased collision-free references from 72.9 % to
99.3 % for a learned proposal model, compared with 77.4 % for resampling under the same
three-attempt limit. These improvements concern reference geometry; successful obstacle traversal
has not yet been demonstrated under the tested tracking protocols. The results identify limits of
the current generator–controller combination and show where measured feedback improves clearance.
Prospective screen validation and obstacle-present execution remain future work.

No dataset throughput in the abstract.

## The four questions the results answer

1. Do generated pools contain motions suitable for the specified obstacle?
2. Can a selector identify them?
3. Does correction improve the available motions?
4. Does the benefit survive controller execution?

## Claims and the experiment that supports each

| claim | supported by | status |
|---|---|---|
| The step-prompted motion is produced but rarely at the specified position | exp021 whole-body clearance at the fixed centre; exp019/EXP-1C/EXP-023/023b for the other channels | done |
| Clearance is not preserved after tracking for any of the 64 references | EXP-022A guarded endpoint | done |
| For this stepping pool the reference-clearing and tracking-completing sets are disjoint, so selection cannot help | `analyze_pool_coverage.py` over the EXP-022A rows | done, descriptive |
| Long periods outside the support test are associated with the evaluator's cutoff | trackability analysis on exp021 (post hoc, AUC 0.997; 53/53, 8/11) | done, post hoc |
| The screen predicts cutoffs on fresh references | EXP-024 tracking of all 128 | pending |
| A cutoff is not a fall | EXP-028 rollouts without the cutoff rule; upright-at-last-frame is the current evidence | pending |
| The screen does not merely penalise dynamic motion | known-trackable walking and jumping controls | pending |
| Measured correction improves reference clearance for the learned proposer but not for every proposer | phase4e 36 × 8 × 15 arms; `analyze_repair_paired_bootstrap.py` | done, descriptive |
| Correction helps because it fixes a systematic error | fixed extra-crouch baseline under the same budget (EXP-029 C) | not run |
| Scene-aware selection changes and improves the choice | shared-pool obstacle-position study (EXP-029 A) | not run |
| Correcting the reference improves achieved clearance and completes traversal | obstacle-present ducking experiment (EXP-029 C) | not run |
| Records with rejection reasons help a navigation policy | separate future study | future work |

## Repair result, stated exactly

Two corrections minus best-of-three resampling, per-scene paired differences, 36-scene cluster
bootstrap (30,000 resamples), descriptive
(`outputs/analysis_repair_paired_bootstrap/summary.json`):

| proposer, endpoint | difference (pp) | 95 % interval |
|---|---|---|
| learned (TCN), collision-free | +21.9 | +9.4 to +35.4 |
| learned (TCN), 18 cm margin | +36.1 | +25.0 to +47.6 |
| QP teacher, 18 cm margin | +38.5 | +28.1 to +48.6 |
| heuristic, 18 cm margin | −8.0 | −14.2 to −1.7 |

Supported conclusion: *measured correction improves reference clearance for the learned
proposer, but does not outperform resampling for every proposer.* The learned proposer's 99.3 %
collision-free rate is not its margin rate (37.5 %). Mean generations actually used: 2.72 under
correction, 2.99 under resampling.

## Vocabulary

| old wording | wording to use |
|---|---|
| addressability audit | can the motion be placed at the obstacle? |
| trackability contract | reference screen for predicted tracking cutoffs |
| measured-deficit repair | correcting measured clearance errors |
| bilateral no-support phase / float | a period when neither foot meets the support test |
| "the step is not a supported step" / "not a hop" | long periods outside the support test are associated with tracking-error cutoffs; a flight-time comparison motivates physical validation |
| "0 of 53 fell" | all 53 stopped rollouts were upright at the last archived state |
| refusal-annotated corpus | motion records with rejection reasons |
| retained clearance | clearance preserved after tracking (full definition above) |
| contact-consistent (EXP-024 column) | valid step passing the support test |
| traversal (unqualified) | local traversal completion, or navigation completion — never both |
| certificate / certified | never; say screened, checked, audited |

## Related-work positioning

> Scene2Motion studies how obstacle-relative reference clearance changes under a fixed humanoid
> tracker, and evaluates interpretable screening and correction without retraining the motion
> generator.

Already in prior work: object-relative placement and scene-bound interaction keyframes
(MotionBricks); a learned dynamics verifier informed by tracking rollouts, with candidate
selection under a frozen generator and controller (TEXEDO); closed-loop diffusion planning with
a controller (CLoSD); scene-aware motion data and benchmarks through clutter (Moving Through
Clutter, HumanoidPF). What this study adds: obstacle-relative clearance at a fixed scene position
for a frozen prior; the same clearance measured after a fixed tracker under a traversal endpoint;
an interpretable reference-only screen whose prediction target is named and tested by tracking
flagged and passed references alike; coverage reported separately from selected success; and a
measured correction compared against equal-budget resampling with its losses reported.

TEXEDO's lesson and its boundary: it shows that filtering candidates by predicted execution
quality raises tracking success by 11.1 points over a single sample while trading against
instruction match, with neither generator nor controller retrained. Its verifiers take motion and
language, not scene geometry, and its hardware results establish execution of selected motions
rather than obstacle-relative traversal. That is a scope distinction, not a defect.

## Next work, in the advisor's order

1. **Complete the prospective screen evaluation.** Rule frozen (done); obtain outcomes for all
   128 references, flagged and passed; run the cutoff-free rollouts; include known-trackable
   walking and jumping controls; distinguish evaluator cutoffs from falls. The jump control asset
   ships with the tracker checkout as
   `gear_sonic_deploy/reference/example/tired_one_leg_jumping_R_001__A359` (500 steps, 29 joints)
   but is in the deployment CSV format, so it needs a duration- and ordering-preserving
   conversion to the motion pkl our eval bridge consumes (REPORT §44).
2. **Run the shared-pool obstacle-position ducking study with the beam present**
   (`docs/ramp-exp029-selection-vs-coverage-protocol.md`), after validating the traversal
   evaluator, the rollout horizon and the beam spawn. Report coverage separately from selected
   success, and compare extra sampling, a fixed extra-crouch adjustment and the measured
   correction under one budget.
3. **Downstream navigation-policy improvement is future work.** A separate research question;
   this paper does not promise it.
