# PI advice, second review (received 2026-09-02, evening)

Verbatim record. Responds to the reframed page, manuscript v2 and project goal. Adopted in
`docs/project-goal-2026-09-02.md`, `docs/paper-draft-v2-2026-09-02.md` and `docs/index.html`;
the paired reanalysis in §4 is reproduced by `experiments/analyze_repair_paired_bootstrap.py`.

---

The story is much clearer now. I would keep the title:
**Scene2Motion: Evaluating Generated Humanoid Motions for Obstacle Traversal**

The most useful lesson from the Xiao-inspired writing and review skills here is: start with a concrete robot problem, define what success means, and let each experiment support one specific claim.

Your strongest current contribution is an evaluation of where generated motions fail, plus a promising correction method—not yet a demonstrated obstacle-traversal system. That is a coherent paper if you maintain that boundary.

I compared the updated webpage, manuscript, project goal, relevant research-log sections, and measurement code. Five revisions matter most.

## 1. Keep the simple language—but restore the exact measurement definitions

Two simplifications accidentally changed what the experiments measured.

| Current wording | What the code actually measures |
|---|---|
| "44/64 lift a foot ≥3 cm" | 44/64 references can clear a 3 cm box somewhere along the route, according to the whole-body geometry check. |
| Leading foot crosses the route's midline after a median of 1.4 seconds | The reference root reaches the location of maximum box clearance after a median of 1.4 seconds. |

These distinctions matter: foot height is not whole-body clearance, and the timing statistic is not a detected foot-crossing event.

Copy-ready replacements:

> In a whole-body geometry check, 44 of 64 references cleared a 3 cm box at some tested position along the route.

> Among the 49 references with positive clearance somewhere along the route, the reference root reached the location of maximum box clearance after a median of 1.4 seconds.

Also restore the full definition of clearance preserved after tracking. Merely avoiding the obstacle is insufficient: a robot that stops before the box can appear collision-free. The endpoint needs the passage, corridor, finish-beyond-obstacle, and termination conditions—not just a geometry query.

> Clearance is preserved when a reference that clears the obstacle yields a tracked trajectory that passes through the obstacle corridor, finishes beyond the obstacle, remains collision-free, and satisfies the stated termination rule.

General writing lesson: shorten explanations, not definitions.

## 2. Separate the proposed system from the experiments already completed

The method section describes an integrated flow: measure, reject or correct, and send passing motions to the controller. But the completed evidence comes from separate component studies. In the stepping audit, tracking rejected references is essential because it reveals whether the screen predicts the outcome correctly.

> We evaluate the support screen and ducking correction as separate components. To evaluate the screen, we track both flagged and passed references.

There is also a concrete status inconsistency: the records section says the 128 fresh references already carry tracking outcomes and achieved clearance. Their receipt lists prediction as complete but tracking and outcome analysis as planned. Mark those fields pending, consistently across the manuscript, webpage, and project goal. For that prospective study, obtain outcomes for all 128 references, not only those passing the screen.

## 3. Treat the support test as a predictor—not a physical verdict

The webpage still says "Why: the step is not a supported step." and "That is not a jump either." Those conclusions exceed the measurement. The support test uses foot height and speed; failing it does not directly establish zero contact force. The flight-time comparison uses pelvis motion and an approximate interval, not a complete dynamics analysis.

> Long periods outside the foot-support test are associated with tracking-error cutoffs. A simple flight-time comparison motivates further physical validation.

Similarly, replace "0 of 53 fell" with:

> All 53 stopped rollouts were upright at the last archived state.

Your planned known-trackable walking and jumping controls are important here. They test whether the screen distinguishes problematic references from legitimate dynamic motions, rather than simply penalizing motion without conventional foot support.

## 4. The repair result is promising—but its benefit depends on the starting planner

Keep the comparison that shows both the TCN improvement and the heuristic case where resampling wins. Paired differences from the committed experiment rows, resampling the 36 scenes (30,000 scene-cluster bootstrap samples):

| Two corrections minus three-sample resampling | Difference, pp | Descriptive 95 % bootstrap interval |
|---|---|---|
| TCN: collision-free reference | +21.9 | +9.4 to +35.4 |
| TCN: 18 cm margin met | +36.1 | +25.0 to +47.6 |
| Heuristic: 18 cm margin met | −8.0 | −14.2 to −1.7 |

This is a descriptive reanalysis, not a new robot experiment or evidence of generalization beyond this benchmark. The supported conclusion:

> Measured correction improves reference clearance for the learned proposer, but does not outperform resampling for every proposer.

The stronger explanation—"measurement helps because it corrects systematic error"—still needs a mechanism test. A useful additional baseline is a fixed extra-crouch adjustment, evaluated under the same maximum generation budget. Also distinguish "same maximum number of generations" from equal actual computation time.

## 5. Narrow the related-work contrast

The statement that none of the listed methods considers scene placement or where a behavior occurs is too broad. MotionBricks explicitly uses object-relative placement and scene-bound interaction keyframes. TEXEDO already uses a learned dynamics verifier informed by tracking rollouts. A stronger positioning statement:

> Scene2Motion studies how obstacle-relative reference clearance changes under a fixed humanoid tracker, and evaluates interpretable screening and correction without retraining the motion generator.

Then state explicitly which capabilities belong to prior work and which measurements your study adds.

## The next research steps

1. Complete the prospective screen evaluation. Freeze the rule, obtain outcomes for every reference, include valid walking/jumping controls, and distinguish evaluator cutoffs from falls.
2. Run one decisive obstacle-present ducking experiment. Compare uncorrected generation, resampling, measured correction, and a simple fixed correction. Report passage, clearance, tracking error, falls, and cost separately.
3. Leave downstream navigation-policy improvement as future work.

Project goal:

> Determine when generated humanoid motions clear an obstacle at its specified location, whether that clearance survives controller tracking, and when measured correction improves the outcome.

Central message:

> A motion can resemble a step or a duck without clearing the intended obstacle. Even a reference that clears the obstacle may lose clearance when tracked by a humanoid controller. Scene2Motion separates these failures by measuring reference geometry, controller outcomes, and achieved clearance. Current experiments reveal limitations in generated stepping motions and show that measured correction can improve ducking-reference clearance. Obstacle-present traversal remains to be validated.
