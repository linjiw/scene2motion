# Scene2Motion — project goal, title, abstract and vocabulary (2026-09-02)

Adopted from the advisor's review (`docs/pi-advice-2026-09-02.md`). This file is the single
statement of what the project is trying to show and how it is worded. It supersedes the
three-pillar wording in `docs/framing-2026-09-02-contract-repair-engine.md` and the
"Contract Before Control" title of the Codex draft; the receipts, protocols and ledger are
unchanged.

## The question

> **Can a humanoid produce the right motion, at the right place, and still clear the obstacle
> when a controller follows it?**

## Title

**Scene2Motion: Evaluating Generated Humanoid Motions for Obstacle Traversal**

Question-led alternative for talks: *Can Generated Humanoid Motions Clear Obstacles?*
"Contract Before Control" may survive as a presentation subtitle only; *contract* is not the
paper's scientific concept, because the screen predicts one evaluator's stopping rule, not a
guarantee.

## Plain-language goal

> Turn generated humanoid motions into useful candidates for moving through clutter: check
> whether they fit the scene, improve their clearance when possible, and test whether a
> controller can reproduce them.

## Research goal

> Determine when a pretrained humanoid motion model can provide useful references for obstacle
> traversal without retraining the model or controller. Measure whether scene placement and
> clearance survive tracking, and evaluate whether feedback-based correction and reference
> screening improve outcomes under a fixed sampling budget.

Choosing whether to duck, step over or take another route is the longer-term navigation goal.
It stays in the roadmap until that complete behaviour has been demonstrated.

## Scientific identity

Testing the gap between generated motion and usable robot motion, with targeted repairs where
measurement helps. Three principles borrowed from *Moving Through Clutter*:

1. Start with the obstacle, not the software pipeline. The robot must duck under a beam or
   step over a box.
2. Explain the missing capability. Producing a stepping motion does not ensure it happens at
   the box.
3. Separate evidence levels. A collision-free reference, a completed tracking run and a
   successful obstacle traversal are different results.

## Evidence levels (report each separately; never let one stand in for the next)

| level | meaning | current evidence |
|---|---|---|
| generated | the model produced a motion under the prompt | 5,393 references under the corrected sampler |
| placed | the motion occurs at the obstacle position the scene specifies | 12/64 step references clear a 5 cm box at x = 1.2 m; the position was selected after inspecting the motions |
| collision-free reference | the reference clears the obstacle geometry with the stated margin | ducking: 72.9 % → 99.3 % (learned proposer, two repairs); stepping: 12/64 |
| completed tracking run | the controller followed the reference to the end without an evaluator cutoff | stepping: 11/64; the cutoff is a tracking-error rule, not a fall (every cut-off robot upright) |
| clearance preserved after tracking | the recorded robot states clear the obstacle geometry | stepping: 0/64; ducking: 0/859 under a protocol confounded by a 14 s clip cap |
| obstacle traversed in simulation with the obstacle present | not yet tested in any protocol | — |

## Abstract (adopted)

Generating a humanoid motion is not enough to traverse an obstacle: the motion must occur at
the right place and remain collision-free when tracked. We present Scene2Motion, a framework
for evaluating and repairing generated motions while keeping the motion model and controller
fixed. Using ARDY-G1 and SONIC in simulation, we separate motion generation, obstacle-relative
clearance, and clearance after tracking. Of 64 step-prompted references, 12 cleared a 5 cm box
at a position selected after inspecting the motions; none retained clearance after tracking.
Tracking ran without the obstacle, with recorded robot states subsequently checked against its
geometry. A foot-support screen flagged all 53 references whose rollouts triggered the
configured tracking-error cutoffs, while also rejecting three of the eleven runs that did not
terminate early. This analysis predicts evaluator behavior, not whether the robot would fall.
For ducking motions, we use measured clearance errors to adjust the generation controls. Across
36 beam scenes and eight seeds, up to two repairs increased collision-free references from
72.9 % to 99.3 % for a learned proposal model, compared with 77.4 % for resampling under the
same three-attempt limit. These improvements concern reference geometry; successful obstacle
traversal has not yet been demonstrated under the tested tracking protocols. The results
identify limits of the current generator–controller combination and show where measured
feedback improves clearance. Prospective screen validation and obstacle-present execution
remain future work.

Dataset throughput is deliberately absent from the abstract: generation speed is not yet the
rate of producing successfully executed traversal examples.

## Vocabulary (use the right-hand column everywhere a reader may be new to the project)

| old wording | wording to use |
|---|---|
| addressability audit | can the motion be placed at the obstacle? |
| trackability contract | reference screen for predicted tracking cutoffs |
| measured-deficit repair | correcting measured clearance errors |
| bilateral no-support phase / float | a period when neither foot meets the support test |
| refusal-annotated corpus | motion records with rejection reasons |
| retained clearance | clearance preserved after tracking |
| traversal substrate | motion generator used for obstacle traversal |
| posture-conditional traversability | whether the robot can pass using a different posture |
| contact-consistent (EXP-024 column) | valid step passing the support test |
| certificate / certified | (never) — say audited, checked, screened |

The EXP-024 column correction matters: "valid step passing the support test" is a local-step
validity check *and* the support test, so its 0/32 counts do not contradict the 33 references
that pass the screen alone.

## Numbers that must be kept side by side

- Learned proposer (TCN), ducking, 36 scenes × 8 seeds: collision-free 72.9 % → 99.3 % after
  two repairs vs 77.4 % for three-sample resampling; **the 18 cm margin rate after two repairs
  is 37.5 %**, not 99.3 %.
- Heuristic proposer: repair 98.3 % → 100 % collision-free, margin 53.5 % → 64.6 %;
  **three-sample resampling reaches 72.6 % margin, higher than repair**. Keep this losing
  comparison in the paper.
- QP proposer: 87.2 % → 100 % collision-free, margin 0.7 % → 41.7 %; resampling 97.2 % / 3.1 %.
- Stepping: 44/64 lift ≥ 3 cm somewhere; 12/64 clear 5 cm at the fixed position; 8/64 pass the
  support screen; 0/64 preserve clearance after tracking; 53/64 rollouts cut off; every cut-off
  robot upright.
- Screen: flags 53/53 cut-off references and 8/11 completed runs pass (three rejected); AUC
  0.997 on the same 64 references, post hoc; prospective test pending.

## Next work, in the advisor's order

1. **Establish what the screen predicts.** Finish the fresh-reference test (EXP-024 tracking
   stage) and the runs without tracking-error cutoffs (EXP-028). Three separate questions: does
   the screen predict early termination on new references; without the termination rule does the
   robot fall, stall or continue; does the achieved motion clear the obstacle geometry. Include
   known-trackable controls, including a jumping example, before reading an airborne phase as
   physical impossibility.
2. **One convincing ducking experiment.** A walking control that reaches the beam; uncorrected
   vs repaired vs resampled references across predetermined beam heights; rollouts long enough to
   reach the beam; the beam present in simulation. Question: does correcting the reference's
   clearance also improve the robot's achieved clearance? Keep collisions, failure to reach the
   beam and evaluator cutoffs separate.
3. **Rejection labels that are useful without being ground truth.** Record per candidate:
   reference intersects the obstacle; screen predicts an evaluator cutoff; tracking stops before
   reaching the obstacle; achieved motion clears the obstacle. A rejected candidate does not mean
   the scene is untraversable. Whether the records improve a navigation policy is a separate
   experiment.

## Page and paper organisation

Lead with the same-motion, different-obstacle comparison. Then **placement, tracking, repair**.
Experiment identifiers, implementation history and schedules live in the research log
(`docs/REPORT.md`), not on the page. Make the reader understand the robot's problem before
asking them to understand the terminology.
