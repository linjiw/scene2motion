# Page draft (2026-09-03) — the results narrative

Draft copy for the project page's results section. **Task C only: this file does not touch the
page.** It obeys `docs/project-goal-2026-09-02.md` — the title, the question, the measurement
definitions, the evidence levels and the vocabulary rules — and is organised as that document's
four questions. Experiment identifiers are kept out of the narrative and collected in the
reproducibility table at the end.

Every number below carries its committed artifact and the JSON key it comes from. Nothing here is
sourced from prose in another document.

---

## The headline, stated so it cannot be misread

This is an **evaluation contribution plus a promising correction method**. It is **not** a working
traversal system.

With a real obstacle in the physics scene, **0 of 64 references completed local traversal** — 0 of
64 against a 5 cm box and 0 of 64 against a 20 cm box, both at the position the scene specifies
(`outputs/exp030_obstacle_present/summary.json` → `arms.present_05.local_traversal_completion.completed`,
`arms.present_20.local_traversal_completion.completed`; Wilson 95 % 0–0.057 at
`…local_traversal_completion.wilson95`).

Local traversal here means passing the obstacle **inside the corridor** and finishing at the goal.
Walking around it is a failure, not a success, and it is not navigation
(`outputs/exp030_obstacle_present/summary.json` → `arms.present_05.local_traversal_completion.definition`).

Four questions follow. Each is answered at one evidence level and no answer is allowed to stand in
for the next.

---

## 1. Does the model produce the motion at all?

**Yes, from the text prompt, and only from the text prompt.**

"Produced" is a whole-body geometry statement, not a foot height: the reference clears a **3 cm
box somewhere along the route** in the collision model.

| way of asking | produced (3 cm box, somewhere on the route) | where the count lives |
|---|---|---|
| text prompt from the first frame, 64 seeds | **44 / 64** (0.688; Wilson 0.566–0.788) | `docs/figures/fig2_channel_funnel_numbers.json` → `channels[key=text_step_frame0].stages.elicits.{k,n,wilson95}`; cross-checked at `outputs/exp021_elicited_lift_distribution_v2/receipt.json` → `summary.n_clearing["0.03"]` |
| free walking, no step asked for, 8 seeds | **0 / 8** at 3 cm; 3 of 8 show any lift at all, the tallest 5 mm | `docs/figures/fig3_channel_response_numbers.json` → `panel_a_text["WALK exp019 v7 nominal (8 seeds)"].{ge_3cm, gt_0, max_m}` |
| free walking, matched windows, 16 seeds | **0 / 16** | `docs/figures/fig2_channel_funnel_numbers.json` → `channels[key=free_walk_control].stages.elicits.{k,n}` |
| position channel (a commanded foot lift), 144 clips | 81 / 144 at the clip's own swing peak | `docs/figures/fig2_channel_funnel_numbers.json` → `channels[key=position_lift_exp1c].stages.elicits.{k,n}` |
| rotation packets, absolute / residual, 8 seeds each | 7 / 8 and 5 / 8 | `…fig2_channel_funnel_numbers.json` → `channels[key=rotation_packet_absolute].stages.elicits.k`, `…rotation_packet_residual…` |
| the prompt switched mid-rollout (walk, then step) at frame 52, 16 seeds | **3 / 16** (0.188; Wilson 0.066–0.430) | `…fig2_channel_funnel_numbers.json` → `channels[key=delayed_prompt_walk_to_step_52].stages.elicits.{k,n,wilson95}` |
| the same prompt from the first frame, 8 seeds | 6 / 8 | `outputs/exp023_prompt_handoff/receipt.json` → `summary.event_rates_missing_retained.step_0.present` (of `.planned` = 8) |

The floor matters: free walking on the same route produces the behaviour **0 of 8** times at the
3 cm definition and **0 of 16** times in the matched-window control, so the 44 of 64 is the prompt
doing work, not the route.

A later prompt still transmits, but it produces the behaviour **less often, later, and unplaced**.
The three post-switch cases arrive 21, 45 and 75 frames after the switch — 0.84 s, 1.8 s and 3.0 s
at 25 fps (`outputs/exp023b_prompt_switch_control/receipt.json` →
`summary.prompt_relative_latency.step_52.step_event_latency_frames.values`).

**When it happens.** Among the 49 references with any positive whole-body clearance along the
route, the reference root reaches the location of **maximum** clearance after a median of
**1.4 s** (frame 35; q10–q90 20.8–55.4 frames), with **40 of 49 inside the first 50 frames**
(Wilson 0.686–0.900) and 4 of 49 after frame 60
(`outputs/analysis_event_frames/receipt.json` →
`summary.exp021.lifting_all.A_root_crossing.{seconds.median, frames.median, frames.q10, frames.q90, inside_first_50_frames.count, inside_first_50_frames.wilson95_of_present, after_frame_60.count}`).
Under the route-nominal speed conversion the count is 42 of 49
(`…summary.exp021.provisional_numbers_in_plan.computed_B_nominal_real.inside_first_50_frames`).
This is the root arriving at a location, **not a detected foot-crossing event**.

**A second released model produces it too, from text alone**: 23 of 64 step-prompted clips at the
3 cm definition, 41 of 64 with any positive clearance, against **0 of 64** in its own walk control
(`outputs/exp025_kimodo_cross_prior/summary.json` → `arms.step.elicitation.k`,
`arms.step.any_lift.k`, `arms.walk.elicitation.k`, all over `n = 64`).

---

## 2. Does it put it where the scene says?

**Mostly not — and no released model tested here does.**

"Placed" is clearance at the position the scene specifies (a 0.20 m-deep box centred at
x = 1.2 m), not at a position chosen to fit the clip.

| box height | references clearing at the specified position, of 64 |
|---|---|
| 3 cm | 13 |
| 5 cm | **12** (Wilson 0.111–0.300) |
| 8 cm | 11 |
| 12 cm | 7 |
| 20 cm | 6 |
| 30 cm | 2 |

`outputs/analysis_exact_centre_cost_curve/receipt.json` →
`at_exp022a_centres.staged.reference_exact_hits` (3/5/8 cm) and
`at_exp022a_centres.staged.paired_guarded_retention.<height>.reference_clear` (all six heights);
Wilson interval at `outputs/analysis_trackability_contract/receipt.json` →
`summary["exact_clear_5cm_at_x1.2"].wilson95_of_64`.

So 44 of 64 produce the motion and 12 of 64 put it at the box. **The gap between those two numbers
is the paper's first result.** The reason is spread, not weakness: across the 64 clips the location
of maximum clearance has median 1.24 m but a standard deviation of 1.04 m, with q10–q90 spanning
0.73–1.97 m (`outputs/exp021_elicited_lift_distribution_v2/receipt.json` →
`summary.lift_position_m.{quantiles["0.5"], sd, quantiles["0.1"], quantiles["0.9"]}`).

**The position was chosen after inspecting the clips.** On a 0.05 m grid, x = 1.2 m is the unique
maximum at 5 cm (12 of 64) and at 8 cm (11 of 64); at 3 cm the maximum is 15 of 64 at 1.15 m
(`outputs/analysis_exact_centre_cost_curve/receipt.json` → `extrema_post_hoc["h=0.05"]`,
`["h=0.08"]`, `["h=0.03"].{max_successes, tied_centres_m}`). Every number above is therefore an
upper bound for a prospectively fixed obstacle, and the centre is frozen for replication.

**The cost of resampling at the fixed position.** Best-of-N at x = 1.2 m and 5 cm rises
0.188 → 0.340 → 0.564 → 0.810 → 0.964 → 0.999 for N = 1, 2, 4, 8, 16, 32; 90 % coverage needs
**about 12 fresh draws** (13 at 8 cm) (`docs/figures/fig4_numbers.json` →
`best_of_n_0.05.{curve, ns, n90}`, `best_of_n_0.08.n90`).

**The tolerant window is addressability, never success.** If the box is allowed to move to
whichever scanned centre each clip happens to clear, **24 of 64** clear at 5 cm within ±0.25 m of
1.2 m and **17 of 64** within ±0.10 m
(`outputs/analysis_exact_centre_cost_curve/receipt.json` →
`tolerant_union.windows["r=0.25"].by_height["h=0.05"].union_successes`,
`…["r=0.10"]…union_successes`). The receipt labels these itself: *"Addressability/capability
analysis, never a fixed-obstacle success probability… the obstacle is allowed to move to that
clip's own lift after the clip has been seen; the scene does not grant that tolerance"*
(`…tolerant_union.label`). The fixed-obstacle number is 12 of 64, and only that number may be
quoted as a rate of clearing an obstacle where it actually is. The window buys 12 extra references
at ±0.25 m and 5 at ±0.10 m (`…windows["r=0.25"].by_height["h=0.05"].extra_references_bought_by_the_window`).

**Neither released model places it.** The second, offline, non-autoregressive prior clears
**0 of 64 at every tested height** (3, 5, 8, 12, 20 and 30 cm) at the specified x = 1.2 m
(`outputs/exp025_kimodo_cross_prior/summary.json` → `arms.step.exact_clearance.staged.<height>.k`).
This is not a coverage artefact: every one of the 64 clips swept the scan point with its whole-body
envelope (`…arms.step.coverage.obstacles.staged.body_swept.{k,n}` = 64 of 64), and a clip that never
swept a centre is recorded as *not reached*, never as a pass (`…coverage.rule`). At an unstaged
centre 3.6 m along the same route it clears 4 of 64 at 3 cm and 3 of 64 at 5 and 8 cm
(`…arms.step.exact_clearance.unstaged`).

The other conditioning channels place it no better: 0 of 8 and 0 of 8 for the two rotation-packet
arms at their predeclared per-seed obstacle, and 0 of 16 for the switched prompt at its predicted
centre x = 3.11 m (`docs/figures/fig2_channel_funnel_numbers.json` →
`channels[key=rotation_packet_absolute].stages.clears.k`, `…rotation_packet_residual…`,
`channels[key=delayed_prompt_walk_to_step_52].stages.clears.{k,n}`).

---

## 3. Can the controller use it?

Three findings, in order: a screen that predicts the controller evaluator's stopping rule; a
coverage-versus-selection decomposition that says selection could not have rescued this pool; and
the first campaign with the obstacle actually in the physics scene.

### 3a. A reference-only screen predicts the evaluator's cutoff

The support test: a foot counts as supporting at a frame when its sole is within 4.65 cm of the
ground and its planar speed is below 1.18 m/s (`outputs/analysis_trackability_contract/receipt.json`
→ `inputs.threshold_receipt.{support_height_m, support_speed_mps, path, sha256}`; frozen before
these references were generated, `summary.definitions.gate`). The **screen** rejects a reference
whose **longest period with neither foot meeting the test exceeds 0.20 s**. Its calibration corpus
is the 144 control references of the earlier position-channel study together with the rollouts
taken from them, so it carries no predictive claim of its own and is excluded from the numbers
below (`summary.definitions.families_note`, `summary.exp1c_ctrl_calibration_corpus.n` = 144).

On the 64 step-prompted references, 53 rollouts hit the evaluator's tracking-error cutoff. The
longest such period separates cut-off from surviving rollouts with **AUC 0.997** (bootstrap
0.987–1.00) (`…summary.exp021_step.single_feature_auc.max_unsupported_run_s.{auc, ci95_bootstrap}`).
At 0.20 s the screen **flags 53 of 53 cut-off rollouts** (sensitivity 1.00, Wilson 0.932–1.00) and
**passes 8 of 11 that completed their run** (specificity 0.727, Wilson 0.434–0.903)
(`…summary.exp021_step.gate_0p2s_primary.{terminated_above, terminated_n, survived_above, survived_n, sensitivity, specificity}`).
A post hoc threshold of 0.28 s gives 51 of 53 and 11 of 11; it is reported as one row of a sweep,
never as the rule (`…summary.exp021_step.sweep_max_unsupported_run_s`, entry `threshold_s = 0.28`).

It is not merely "did the reference lift". Lift height alone gives AUC 0.924
(`…summary.exp021_lift_height_only.auc`), and among the 20 references that never lift — 10 of them
cut off — the same period still separates outcomes with AUC 0.98
(`…summary.exp021_non_lifting_stratum.{n, terminated, run_auc}`). All 44 produced references are
flagged (`…summary.exp021_lift_ge_3cm.n_within_calibrated_gate` = 0 of 44) and all 8 references
that pass the screen completed their run (`…summary.exp021_within_calibrated_gate.{n, survived}`).

**What the screen predicts is a rule, not physics.** The cutoff is the release evaluator ending
the episode on a tracking error. It is not a fall. At the last archived state the pelvis of every
cut-off rollout sits between 0.56 m and 0.95 m, with **0 of 53 below 0.5 m** — say *upright at the
last archived state*, and nothing more
(`…summary.fall_vs_cutoff_at_last_sample.{achieved_pelvis_z_min_median_max, n_pelvis_below_0p5m}`).
The cutoff lands within 0.2 s of the reference's first ≥ 0.2 s unsupported onset in **47 of 53**
cases, median +0.04 s, and before the onset in 10
(`…summary.termination_vs_first_nosupport_onset.{within_0p2s, median_s, before_onset, n_terminated}`).

### 3b. Coverage versus selection: no selection rule could have succeeded

At the specified obstacle, 5 cm, over the same 64 references and the same replay endpoint
(`outputs/analysis_pool_coverage/summary.json` → `results[obstacle_label="staged"]`):

| | count | key |
|---|---|---|
| reference clears at the specified position | **12** | `…by_height["0.05"].n_reference_clears` |
| rollout completes without the evaluator cutoff | **11** | `…n_completes_tracking` |
| **both** | **0** | `…by_height["0.05"].n_reference_clears_and_completes_tracking` |
| passes the corridor and finishes beyond the obstacle | 14 (11 of them without a cutoff) | `…n_passed_corridor_and_finished_beyond`, `…n_passed_and_finished_and_completed` |
| never reaches the obstacle | 50 | `…n_never_reached_obstacle` |
| satisfies the full endpoint | 0 | `…by_height["0.05"].n_traversal_endpoint` |

The two sets are **disjoint**. Reference clearance alone reaches 90 % coverage at about 12 fresh
draws (`…by_height["0.05"].coverage_curve_reference.n90_budget_fresh_draws`), while joint coverage
is **0 at every budget up to the whole pool**
(`…coverage_curve_joint.coverage_at_full_pool` = 0.0). For this stepping family the limitation is
the candidate pool, not the ranking function — and that is the conflict worth naming: *selecting a
motion the controller can execute is not the same as selecting a motion that solves the scene
task, and here the two requirements have no common candidate.*

### 3c. With the obstacle actually in the scene

Every executed result before this one tracked references with the obstacle **absent** from the
physics scene and replayed the recorded states against the collision model. The first
obstacle-present campaign ran the same 64 archived references three ways — no box, a 5 cm box and a
20 cm box at x = 1.2 m — under the release evaluator, physics seed 0, one rollout each.

**Outcomes over all 64 assigned trials per arm** (`outputs/exp030_obstacle_present/summary.json` →
`arms.<arm>.outcomes`, `arms.<arm>.n_assigned_trials`):

| outcome class | no box | 5 cm box | 20 cm box |
|---|---|---|---|
| completed local traversal | **1** | **0** | **0** |
| evaluator cutoff | 54 | 44 | 34 |
| hit the box | 0 (no box present) | **20** | **30** |
| stalled | 9 | 0 | 0 |
| fell | 0 | 0 | 0 |
| hit a corridor wall | 0 | 0 | 0 |
| rejected before execution | 0 | 0 | 0 |
| timeout | not assessed | not assessed | not assessed |

Completion rates over all assigned trials: 1/64 = 0.016 (Wilson 0.003–0.083) with no box, **0/64
(Wilson 0–0.057)** with either box
(`arms.absent.local_traversal_completion.{completed, rate, wilson95}` and the two present arms).
The single completion in the no-box arm is reference `s4434`
(`arms.absent.local_traversal_completion.completing_motion_keys`) — and it is a **different
evidence level**: there was no obstacle to pass. Collision rates over assigned trials are 0.313 and
0.469 (`arms.present_05.collision_rate`, `arms.present_20.collision_rate`). The timeout count of
zero means *not assessed*: no wall-clock deadline was preregistered
(`scene.time_limit_s` = null, `scene.timeout_note`).

**"Hit the box" here means the robot actually contacted it.** The receipt states the distinction
itself: *"The obstacle was present in the Isaac scene, so 'collided_obstacle' here means the robot
actually contacted the box — unlike every earlier campaign, where it meant the recorded motion
intersected the box's volume in replay"* (`interpretation_guard`). This is the only campaign in the
project where that word is literal.

**All three preregistered predictions held** (`outputs/exp030_obstacle_present/summary.json` →
`predictions`):

- **P1, the control.** The no-box arm reproduces the earlier campaign on **63 of 64** termination
  flags against a preregistered threshold of 58 (`predictions.P1.{n_agreeing, threshold,
  prediction_held}`; 53 versus 54 cut off in total, `p1_absent_vs_exp022a.terminated_counts`). Valid
  rollout length agrees on 54 of 64 and is reported alongside, not as the rule
  (`p1_absent_vs_exp022a.valid_length.{n_agreeing, role}`).
- **P2, completion.** 0 of 64 in both present arms, as predicted
  (`predictions.P2.{completions, prediction_held}`). The prediction was written as a prediction of
  failure so that a completion would be a finding, not a surprise.
- **P3, the proxy.** The obstacle-absent replay predicts the obstacle-present class on **63 of 64**
  references: agreement **0.984**, Cohen's **κ = 0.964**, percentile bootstrap 95 % **0.882–1.0**
  over 2,000 resamples (`predictions.P3.{agreement_fraction, kappa, kappa_ci95}`;
  `q1_proxy_check.kappa_bootstrap.{method, n_resamples, n_finite}`). The confusion is
  {inferred hit → measured hit 20, inferred hit → measured cutoff 1, inferred cutoff → measured hit 0,
  inferred cutoff → measured cutoff 43} (`q1_proxy_check.confusion.matrix`), the single disagreement
  being reference `s4410` (`q1_proxy_check.disagreeing_references`).

That last result is what lets the project's earlier obstacle-absent execution numbers keep their
meaning: on this pool, in this scene, the replay was an accurate proxy for the outcome class. It is
one scene and one obstacle position.

**How much the box changed the rollout.** Paired per reference, the maximum forward progress falls
by a median of 0.000 m with an interquartile range of 0.000–0.001 m — most references never reach
the box — while **8 references lose more than 0.05 m**, up to 3.93 m
(`paired_progress_change.{median_m, iqr_m, n_falling_more_than_threshold, threshold_m, max_m,
references_falling_more_than_threshold}`).

---

## 4. Does measured correction help?

**Yes for the learned proposer, on reference geometry, and not against every baseline.**

The setting is the ducking family: 36 beam scenes × 8 seeds = **288 scene–seed trials per arm**, a
0.18 m clearance target, three proposers (a heuristic, a QP teacher, a learned temporal
convolutional model), each run uncorrected, with one and two measured corrections, and against
best-of-two and best-of-three resampling (`outputs/phase4e_architecture_v2_s8/experiment.json` →
`summary.<arm>.n`, `target_m`, `n_beams`, `heights`, `gaps`, `seeds`, `method_specs`).

**The win.** For the learned proposer, up to two corrections raise collision-free references from
**72.9 % to 99.3 %**, against **77.4 %** for resampling under the same three-generation cap
(`summary.tcn.collision_free_rate` = 0.7292, `summary["tcn+2"].collision_free_rate` = 0.9931,
`summary["tcn-resample3"].collision_free_rate` = 0.7743).

Paired per-scene differences, 36-scene cluster bootstrap, 30,000 resamples, **descriptive**
(`outputs/analysis_repair_paired_bootstrap/summary.json` → `results[*].{paired_difference_pp,
bootstrap_95_pp, rate_a, rate_b}`; `n_scenes`, `n_seeds`, `n_boot`):

| proposer, endpoint | correction − resampling | 95 % interval |
|---|---|---|
| learned, collision-free reference | **+21.9 pp** | +9.4 to +35.4 |
| learned, 18 cm margin | +36.1 pp | +25.0 to +47.6 |
| QP teacher, 18 cm margin | +38.5 pp | +28.1 to +48.6 |
| QP teacher, collision-free reference | +2.8 pp | 0.0 to +6.9 |
| heuristic, collision-free reference | 0.0 pp (tie, 1.000 vs 1.000) | 0.0 to 0.0 |
| **heuristic, 18 cm margin** | **−8.0 pp** | −14.2 to −1.7 |

**The losing comparison is part of the result.** For the already-good heuristic proposer, plain
resampling beats correction on margin: **72.6 % versus 64.6 %**
(`results[proposer="heuristic", endpoint="meets_target"].{rate_b, rate_a}`), with 41 scene–seed
trials won by resampling alone against 18 won by correction alone
(`…{discordant_b_only, discordant_a_only}`). The supported conclusion is therefore narrow:
*measured correction improves reference clearance for the learned proposer, but does not outperform
resampling for every proposer.*

**Two other things must be said in the same breath.** The learned proposer's 99.3 % is a
collision-free rate, **not** its margin rate, which is 37.5 %
(`outputs/phase4e_architecture_v2_s8/experiment.json` → `summary["tcn+2"].margin_satisfaction_rate`).
And equal caps are not equal computation: both compared arms are capped at three generations, but
the mean actually used is 2.72 under correction against 2.99 under resampling for the learned
proposer, 2.67 against 2.98 for the QP teacher, and 1.84 against 1.81 for the heuristic
(`docs/figures/fig7_numbers.json` → `budget_parity.{mean_generation_calls_used,
all_compared_arms_capped_at_three_generations, note}`). Correction is cheaper where it wins and
slightly dearer where it loses. Across the campaign 1,925 correction steps were applied, mean
magnitude 0.108 m, maximum 0.352 m (`…experiment.json` → `repair_stats.{n_repair_steps,
mean_magnitude_m, max_magnitude_m}`).

**Scope, stated with the result: this is reference geometry only.** No corrected ducking reference
has been shown to survive a controller. The one tracked ducking campaign selected only the
heuristic arms, ran 859 rollouts, and **0 passed the last obstacle** and 0 executed successfully,
with 554 evaluator cutoffs and 200 rollouts reaching the first obstacle
(`outputs/exp1b_execution_clearance_v2/receipt.json` → `selection.{methods, n_selected, n_scenes,
n_unique_clips}`, `outcomes.{passed_last_obstacle, executed_success, terminated,
reached_first_obstacle}`).

**The record engine, for cost.** Running the correction loop over 300 fresh scenes took **301.9 s**
and produced **268 collision-free records** (192 accepted at target, 76 accepted but short of the
0.18 m margin), **6 rejected** after two corrections, and **26 refused with a named deficit** — 13
for overhead clearance and 13 for lateral clearance
(`outputs/corpus_pilot_v2/receipt.json` → `n_scenes`, `wall_clock_s`,
`counts.{accepted, accepted_margin, rejected, refused}`; the refusal split from
`outputs/corpus_pilot_v2/manifest.jsonl` → `refusal.functional`). A refusal is an outcome class
with a reason attached, not a success; rejecting a scene scores that trial as a non-completion.

---

## What generalises

Two claims were tested for generality. **One crossed both boundaries; one did not cross either.**

### The screen crosses a behaviour boundary — weakly, and in one direction only

On a second behaviour family (ducking, 526 references over 36 scenes, 344 evaluator cutoffs), the
contact feature separates cut-off from surviving rollouts with **AUC 0.674 pooled** (cluster
bootstrap 0.625–0.721) and **0.694 within scene**
(`outputs/analysis_duck_contract/receipt.json` → `summary.pooled_auc_by_group.contact.{auc,
cluster_bootstrap.ci95}`, `summary.within_scene_auc_by_group.contact.weighted_mean_auc`,
`summary.{n_clips, n_scenes, n_terminated}`).

The speed confound is refuted rather than assumed: the speed feature scores **0.441** pooled — below
chance — and the contact-minus-speed difference is **+0.233** with a pooled interval of
+0.120 to +0.362 (`summary.pooled_auc_by_group.speed.auc`,
`summary.decision.pooled.contact_minus_speed`, `summary.contact_minus_speed_pooled.ci95`). The
contact feature is above chance in **all seven strata** — three crouch-depth bins (0.651, 0.578,
0.693) and four route classes by beam count (0.664, 0.663, 0.670, 0.707)
(`summary.strata.dip_bins.strata[*].auc.contact`, `summary.strata.route_classes.strata[*].auc.contact`).

**Both weaknesses, stated plainly.** The transfer is much weaker than on the stepping family
(0.674 against 0.997) and it is **directional**: at 0.20 s the screen keeps sensitivity 0.910
(Wilson 0.875–0.936) but specificity collapses to **0.297** (Wilson 0.235–0.367)
(`summary.screen.calibrated_0p20s.{sensitivity, sensitivity_wilson95, specificity,
specificity_wilson95}`) — it catches cutoffs and rejects most of the good references with them.
And the population is saturated: **441 of 526** duck references exceed 0.20 s
(`summary.float_fraction.{n_with_run_over_0p20s, n}`). The within-scene contact-minus-speed
interval, +0.092 with 95 % −0.018 to +0.210, **straddles zero**
(`summary.decision.within_scene.contact_minus_speed`, `summary.contact_minus_speed_within_scene.ci95`).

### The screen crosses a model boundary — by a thin margin whose interval straddles the threshold

On the second, offline, non-autoregressive released prior, **19 of 23 elicited clips exceed the
0.20 s screen: 0.826**, against a preregistered threshold of 0.80. The Wilson 95 % interval is
**0.629–0.930**, which straddles that threshold
(`outputs/exp025_kimodo_cross_prior/summary.json` → `arms.step.screen.float_primary_0p20s.{k, n,
rate, wilson95}`; rule at `decisions.screen_rule.threshold_fraction`). The secondary 0.28 s
threshold gives the same 19 of 23 (`arms.step.screen.float_secondary_0p28s`). Over **all 64
assigned trials** the rate is 19/64 = 0.297 for elicited clips above the screen and 31/64 = 0.484
for every clip above it, elicited or not
(`arms.step.screen.{float_primary_elicited_over_all_assigned, float_primary_over_all_assigned}`).
Median longest unsupported period is 0.183 s over all 64 and 0.667 s over the 23 elicited
(`arms.step.screen.max_unsupported_run_s.q0.5`, `…max_unsupported_run_s_elicited.q0.5`).

Nothing in that campaign was executed by a tracker, so none of it is a tracking outcome
(`tiers.{generated, kinematically_scored, sonic_executed}` = 128 / 128 / 0; `decisions.scope`).

### Early-lift timing does not generalise

The autoregressive model's early window does **not** reproduce on the offline prior. Of the 41
second-model clips with a lift position, **1 places it inside the first 2.0 s** (0.024; Wilson
0.004–0.126); of the 23 elicited clips, **0 do** (Wilson 0–0.143). The median lift time is 3.87 s by
root crossing and 3.77 s by nominal speed
(`outputs/exp025_kimodo_cross_prior/summary.json` →
`arms.step.lift_time_s.nominal_speed.{within_first_2s, within_first_2s_elicited}` and the medians at
`decisions.timing_rule.definitions.{root_crossing, nominal_speed}.median_lift_time_s`), against
40 of 49 and 42 of 49 for the autoregressive
model (`decisions.timing_rule.ardy_reference.{root_crossing, nominal_speed, denominator}`).

The preregistered decision rule (≥ 0.7 generalises, ≤ 0.4 attributes the window to autoregressive
rollout context) resolves at the lower end: outcome
`ardy_window_attributed_to_autoregressive_rollout_context`
(`decisions.timing_rule.definitions.nominal_speed.outcome`; rule at `decision_rules.timing` in
`outputs/exp025_kimodo_cross_prior/receipt.json` → `campaign_design.decision_rules.timing`).

**So: the support signature travels; the timing signature does not.** The two preregistered rules
split, and that split is the finding.

---

## What is not shown

- **No obstacle traversal has been completed.** 0 of 64 at 5 cm and 0 of 64 at 20 cm with the
  obstacle in the physics scene. The 1 of 64 in the obstacle-absent control arm is a lower evidence
  level — there was no obstacle to pass — and must never be quoted as a traversal
  (`outputs/exp030_obstacle_present/summary.json` → `arms.*.local_traversal_completion`).
- **Clearance preserved after tracking is 0 of 64** at every height, under the guarded endpoint
  (non-terminated, clears the exact box, passes inside the corridor, finishes beyond it), and stays
  0 at every graded height (`outputs/exp022_exact_tracking_bridge/summary.json` →
  `paired_reference_to_achieved_retention.staged.<height>.{reference_clear, achieved_guarded_clear,
  endpoint_guard}`).
- **The tracked stages of two campaigns are still pending.** The prospective screen test has its
  128 references generated and its predictions frozen — 95 flagged, 33 passing the screen — but no
  rollouts (`outputs/exp024_reference_contract/receipt.json` → `stage` = "predicted",
  `stages.predict.{n, primary_flagged, secondary_flagged, written_before_sonic}`,
  `stages.analyze.status` = "planned"). Its kinematic column is already a zero: **0 of 32 in every
  one of the four conditioning arms** is a locally valid step that also passes the support test
  (`outputs/exp024_reference_contract/rows.jsonl` → `reference.local_step.local_step_success`
  and `reference.gate_predictions.primary_flag`, per `arm`). The cross-model campaign executed no
  rollouts at all (`outputs/exp025_kimodo_cross_prior/summary.json` → `tiers.sonic_executed` = 0).
- **The stepping screen result is post hoc**, and its source campaigns tracked references with the
  obstacle absent from the physics scene
  (`outputs/analysis_trackability_contract/receipt.json` → `status` = "post_hoc_exploratory",
  `scope`).
- **One route and one scene for every stepping rate**, one obstacle position, **one physics seed**,
  one rollout per reference. Seeds are the resampling budget inside that single scene, not
  independent scenes (`outputs/exp030_obstacle_present/summary.json` → `interpretation_guard`;
  `outputs/analysis_pool_coverage/summary.json` → `scope`). Only the ducking family has 36 scenes
  and a cluster bootstrap across them.
- **The timeout outcome class was not measured** in the obstacle-present campaign: no deadline was
  configured, so its count of zero is the absence of a measurement
  (`…summary.json` → `scene.{time_limit_s, timeout_note}`). The `fell` class is scored over archived
  samples only, and an episode stops archiving at the evaluator's cutoff — so a zero there is not a
  claim about what would have happened afterwards. For rollouts that were cut off, the reportable
  fact is that they were **upright at the last archived state**.
- **The tracked ducking negative never touched the learned proposer.** Its selection was the
  heuristic arms only, and its 14 s clip cap is a property of that campaign, not of ducking
  (`outputs/exp1b_execution_clearance_v2/receipt.json` → `selection.methods`,
  `outcomes.passed_last_obstacle` = 0). So the correction result stands at the reference-geometry
  level and no further.
- **The obstacle position was chosen after inspecting the clips**
  (`outputs/analysis_exact_centre_cost_curve/receipt.json` → `extrema_post_hoc.*.label`), and the
  correction and coverage analyses are marked descriptive by their own receipts
  (`outputs/analysis_repair_paired_bootstrap/summary.json` → `descriptive_only`;
  `outputs/analysis_pool_coverage/summary.json` → `descriptive_only`).
- **Navigation is not measured at all.** Nothing here is a navigation result, and no navigation
  result may share a success label with local traversal.

---

## Reproducibility record

Experiment identifiers live here, not in the narrative above.

| identifier | what it measured | artifact | n |
|---|---|---|---|
| EXP-021 | step-prompted reference pool; lift height and position on a fixed route | `outputs/exp021_elicited_lift_distribution_v2/receipt.json`, `rows.jsonl`, `qpos.npz` | 64 references |
| EXP-022A | those 64 references tracked, obstacle absent; guarded clearance-after-tracking endpoint | `outputs/exp022_exact_tracking_bridge/summary.json`, `reference_rows.jsonl`, `achieved_rows.jsonl` | 64 rollouts |
| EXP-030 | the same 64 references with the obstacle **present** in physics, plus an obstacle-absent control | `outputs/exp030_obstacle_present/{receipt.json, summary.json, rows.jsonl}` | 3 arms × 64 = 192 rollouts |
| EXP-1C | position-channel lift arms and the screen's calibration corpus | `outputs/exp1c_stepover/{receipt.json, rows.jsonl}` | 288 clips (144 lift, 144 control) |
| EXP-019 v7 / EXP-020 | rotation-packet arms, free nominal walk arm, text-only control | `outputs/exp019_gait_matched_stepover_v7/{arm_rows.jsonl, placement_analysis.json}`, `outputs/exp020_text_only_control/` | 8 seeds per arm |
| EXP-023 / EXP-023b | prompt switched mid-rollout, with walk and sidestep controls | `outputs/exp023_prompt_handoff/`, `outputs/exp023b_prompt_switch_control/` | 8 + 8 seeds per arm |
| EXP-024 | conditioning-arm ablation; screen predictions frozen before tracking | `outputs/exp024_reference_contract/{receipt.json, rows.jsonl, predictions.jsonl}` | 128 references, 0 tracked |
| EXP-025 | cross-prior replication on a second released G1 model | `outputs/exp025_kimodo_cross_prior/{receipt.json, summary.json, rows.jsonl}` | 128 clips (64 step, 64 walk), 0 tracked |
| EXP-1B | tracked ducking campaign (heuristic arms only) | `outputs/exp1b_execution_clearance_v2/receipt.json` | 859 rollouts |
| phase4e | proposer × correction × resampling benchmark on 36 beam scenes | `outputs/phase4e_architecture_v2_s8/experiment.json` | 15 arms × 288 trials |
| corpus pilot | the record engine with rejection reasons | `outputs/corpus_pilot_v2/{receipt.json, manifest.jsonl}` | 300 scenes |
| screen analysis (stepping) | support-test features versus evaluator cutoffs | `outputs/analysis_trackability_contract/{receipt.json, rows.jsonl}` | 384 rows |
| screen analysis (ducking) | the same features on the duck family | `outputs/analysis_duck_contract/{receipt.json, rows.jsonl}` | 526 references, 36 scenes |
| coverage vs selection | pool coverage against selected success | `outputs/analysis_pool_coverage/summary.json` | 64 references |
| cost curve | exact-centre clearance, best-of-N, tolerant windows | `outputs/analysis_exact_centre_cost_curve/{receipt.json, curve.jsonl, exact_hits.npz}` | 64 × 121 centres × 3 heights |
| event timing | when the reference root reaches maximum clearance | `outputs/analysis_event_frames/{receipt.json, rows.jsonl}` | 96 rows |
| correction bootstrap | paired scene-cluster bootstrap over phase4e | `outputs/analysis_repair_paired_bootstrap/summary.json` | 36 scenes, 30,000 resamples |
| traversal outcomes (replay) | outcome classes from the obstacle-**absent** replay | `outputs/analysis_traversal_outcomes/summary.json` | 64 references × 6 heights |

**Figures already built.** Read the companion `*_numbers.json` before citing any of them: `fig2`
(channel funnel), `fig3` (channel response), `fig4` (cost curve), `fig5` (the screen), `fig6` (the
screen across families), `fig7` (correction versus resampling), `fig8` (outcome classes), all under
`docs/figures/`. Two caveats the page must respect: **fig6 covers three ARDY corpora plus the duck
family and does not include the cross-prior campaign** (`docs/figures/fig6_numbers.json` →
`corpora[*].key` = `exp021_step`, `exp1c_lift`, `duck`), and **fig8 is the obstacle-ABSENT replay**,
where "collided_obstacle" means the recorded trajectory intersects the box's volume, not that the
robot felt contact (`docs/figures/fig8_numbers.json` → `scope`). Neither figure carries the
obstacle-present campaign. A figure showing the obstacle-present outcome table does not exist yet.
