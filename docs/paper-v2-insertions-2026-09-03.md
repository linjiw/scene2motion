# Paper v2.2 — ready-to-paste insertions for the evidence on the shelf (2026-09-03)

Work order: §2 of `docs/audit-2026-09-03-paper-readiness.md` ("Evidence on the shelf"), the ten
committed results the manuscript does not use, plus the one built-and-unused figure that carries
the most important of them.

Scope of this file: **prose only**. It does not edit
`docs/paper-draft-v2-2026-09-02.md`; each block says where it goes and what sentence it follows,
so the manuscript edit is a paste. Numbers were re-read from the named receipts by the author of
this file, not copied from the audit; where the audit and the receipt disagreed, the receipt won
and the disagreement is recorded at the end.

Vocabulary follows `docs/project-goal-2026-09-02.md`: *reference screen*, not contract; *a period
in which neither foot meets the support test*, not float; *evaluator cutoff* and *upright at the
last archived state*, never "fell"; *local traversal*, never "traversal" unqualified; no
"certified". Evidence levels stay separate: produced / placed / collision-free reference /
completed tracking run / clearance preserved after tracking.

Line numbers below are at commit `620bafa`, as in the audit.

---

## B1 — §VI-C. Does the screen only describe stepping? (EXP-026)

**Target:** §VI-C, after the retrospective-screen paragraph.
**Instruction:** insert as two new paragraphs immediately after the sentence ending
"...reported as a sweep, not as the screen." (L346–347), before the "**Prospective test, in
progress.**" paragraph.

> **Does the screen only describe stepping?** The same feature was measured on a second behaviour
> family, produced through a different conditioning pipeline: 526 ducking references, whose
> crouch comes from a proposer's root-height schedule and its measured correction rather than
> from a step prompt, across 36 multi-beam scenes, with one tracking rollout scored per
> reference. There the longest period in which neither foot meets the support test still ranks
> the evaluator's cutoffs above chance: AUC 0.674 pooled (36-scene cluster bootstrap 0.625–0.721)
> and 0.694 within scene, above 0.5 in all seven preregistered strata (0.578–0.707). The confound
> we expected does not account for it. Root speed ranks the same outcomes at 0.441 (0.328–0.552,
> spanning chance; the references that completed tracking are the faster ones), the pooled
> difference between the support feature and the speed feature is +0.233 (0.120–0.362), and the
> two correlate at r = −0.18. Crouch depth separates the outcomes pooled (0.707) but falls to
> 0.565 within scene, so most of what it carries is between-scene. The screen is therefore not an
> artefact of the stepping family or of one way of asking for a motion.
>
> It is also much weaker there, and it is not a decision rule for ducking. At its calibrated
> 0.20 s threshold it flags 313 of the 344 references whose rollouts ended at the evaluator's
> cutoff (sensitivity 0.910, 0.875–0.936) but also 128 of the 182 whose rollouts completed
> (specificity 0.297, 0.235–0.367), because 441 of the 526 ducking references contain a period
> longer than 0.20 s in which neither foot meets the support test. Used as an accept/reject rule
> on this family it would reject most of what works; what transfers is the ranking, not the
> operating point. Two scopes bound the result. The analysis is post hoc: the feature groups,
> their primaries and the direction of the test were fixed before any feature was computed, but
> the outcomes were already known. And in that tracking campaign only 200 of 859 rollouts reached
> the first beam, so most of the cutoffs the feature ranks occurred before the obstacle.

**Sources.** `outputs/analysis_duck_contract/receipt.json`:
`summary.decision.pooled.contact_auc` 0.6740, `.speed_auc` 0.4411, `.contact_minus_speed` 0.2330,
`.verdict` "contract_transfers_to_the_duck_family";
`summary.decision.within_scene.contact_auc` 0.6936;
`summary.pooled_auc_by_group.contact.cluster_bootstrap.ci95` [0.6245, 0.7211];
`summary.pooled_auc_by_group.speed.cluster_bootstrap.ci95` [0.3275, 0.5520];
`summary.contact_minus_speed_pooled.ci95` [0.1196, 0.3618];
`summary.pooled_auc_by_group.crouch.auc` 0.7072 and
`summary.within_scene_auc_by_group.crouch.weighted_mean_auc` 0.5647;
`summary.primary_correlation.pearson` (contact × speed) −0.1811;
`summary.strata.dip_bins.strata[*].auc.contact` 0.6513 / 0.5777 / 0.6934 and
`summary.strata.route_classes.strata[*].auc.contact` 0.6643 / 0.6632 / 0.6701 / 0.7070 (seven
strata, minimum 0.578);
`summary.screen.calibrated_0p20s` {flagged_terminated 313, terminated 344, flagged_survivors 128,
survivors 182, sensitivity 0.9099 (Wilson 0.8749–0.9358), specificity 0.2967 (0.2351–0.3667)};
`summary.float_fraction` {n 526, n_with_run_over_0p20s 441};
`summary.n_clips` 526, `summary.n_scenes` 36; `post_hoc` true and `interpretation`.
Distinguishing the survivors as faster: `summary.distribution_by_outcome.max_root_planar_speed`
(median 1.868 survived vs 1.749 terminated). Reaching the beam:
`outputs/exp1b_execution_clearance_v2/receipt.json` → `outcomes.reached_first_obstacle` 200 of
`selection.n_selected` 859.

**Notes for the writer.** The within-scene difference between the support feature and the speed
feature is +0.092 with interval −0.018 to +0.210 (`summary.contact_minus_speed_within_scene.ci95`),
i.e. it spans zero; the pooled difference is the one with an interval clear of zero. If a reviewer
presses on the within-scene comparison, say so rather than quoting only the pooled number. The
within-scene AUC is computed over the 13 of 36 scenes with at least five rollouts of each outcome
(`summary.within_scene_auc_by_group.contact.n_evaluable_scenes` 13, `total_pairs` 1144).

---

## B2 — §VI-C. Cross-corpus ranking, one sentence

**Target:** §VI-C.
**Instruction:** insert as the last sentence of B1's *first* paragraph, immediately after
"...not an artefact of the stepping family or of one way of asking for a motion."

> Between those two corpora the same feature ranks cutoffs at 0.974 across all 288 references of
> the position-channel ladder (139 cut off) and at 0.921 on its 144 lift arms (135 cut off, 9
> completed), and an 18-feature logistic fitted on one stepping corpus and applied to the other
> ranks at 0.90–0.97 in both directions; on the 144 control references that calibrated the support
> thresholds, of which only 4 were cut off, it ranks at 0.513, which is why that corpus is
> reported as calibration and carries no predictive claim.

**Sources.** `outputs/analysis_trackability_contract/receipt.json`:
`summary.exp1c_all` {n 288, terminated 139,
`single_feature_auc.max_unsupported_run_s.auc` 0.9741};
`summary.exp1c_lift` {n 144, terminated 135, auc 0.9206};
`summary.transfer_auc` {exp1c_all_to_exp021 0.9160, exp021_to_exp1c_all 0.9712,
exp1c_lift_to_exp021 0.9160, exp021_to_exp1c_lift 0.9037};
`summary.exp1c_ctrl_calibration_corpus` {n 144, terminated 4, auc 0.5134};
`summary.definitions.families_note`.

**Correction to the audit.** These keys are in
`outputs/analysis_trackability_contract/receipt.json`, not in the duck receipt the audit's second
§2 bullet points at.

---

## B3 — §VI-C. The screen rejects the whole producing set

**Target:** §VI-C, the retrospective-screen paragraph.
**Instruction:** insert immediately after the sentence ending "...passes 8 of the 11 completed
ones, rejecting three." (L342–343).

> The two rules disagree completely on this pool. Of the 44 references that clear a 3 cm box
> somewhere along the route — the whole producing set — not one passes the screen, and 43 of
> them were stopped by the evaluator; of the eight references the screen does pass, all eight
> completed their tracking runs. So on these 64 the screen is an almost perfect predictor of the
> controller's evaluator and an exact rejector of every candidate the obstacle needs. That is the
> conflict this paper reports rather than resolves: a rule that selects what the controller can
> execute is not a rule that selects what solves the scene task, and here the two sets do not
> intersect.

**Sources.** `outputs/analysis_trackability_contract/receipt.json`:
`summary.exp021_lift_ge_3cm` {n 44, terminated 43, n_within_calibrated_gate 0};
`summary.exp021_within_calibrated_gate` {n 8, survived 8}.
The "43 of them" leaves exactly one produced reference that completed tracking; it is one of the
three completed runs the screen flags, so both statements in the paragraph hold.

---

## B4 — §VI-A. The post hoc position choice, as a number

**Target:** §VI-A.
**Instruction:** insert immediately after the sentence ending "...takes about 12 fresh draws at
the observed rate (13 at 8 cm)." (L279–280), as the end of that paragraph.

> That budget belongs to a position chosen after inspecting the clips. Scored at an unchosen
> position on the same route — 3.6 m, the fixed probe of our earlier ladder — the same 64
> references clear the 5 cm box twice, 90 % coverage would take about 73 fresh draws instead of
> 12, and at 20 cm and above no reference in the pool clears at all. The placement numbers in this
> section are therefore a best case for this generator on this route, not a typical one, and the
> fresh-seed replication fixes the same position in advance so that the next measurement is not.

**Sources.** `outputs/analysis_pool_coverage/summary.json`:
`results[1]` (obstacle_label "unstaged", obstacle_x_m 3.6) →
`by_height["0.05"].n_reference_clears` 2 and
`by_height["0.05"].coverage_curve_reference.n90_budget_fresh_draws` 73;
`by_height["0.2"].n_reference_clears` 0 and `by_height["0.3"].n_reference_clears` 0
(`coverage_curve_reference.coverage_at_full_pool` 0.0 at both).
Staged comparison: `results[0].by_height["0.05"]` → 12 clears, N90 12 fresh draws (13 at 0.08).

**Correction to the audit.** The audit's fourth §2 bullet says the unstaged position gives "2 of
64 at every height". The receipt gives 2 at 3, 5 and 8 cm, **1** at 12 cm (N90 147 draws) and
**0** at 20 and 30 cm. Write "twice at 5 cm", not "at every height".

---

## B5 — §VI-C. Everything that got past the box went through it

**Target:** §VI-C, the outcome-matrix discussion.
**Instruction:** insert immediately after the sentence ending "...Rates are over all assigned
trials." (L333).

> The fourth row of Table I and the last column of this table are consistent, and it is worth
> saying why. At every box height, all 14 trajectories that pass the corridor and finish beyond
> the obstacle also intersect its volume in replay, and exactly one rollout of the 64 reaches the
> goal region — that one intersects the box as well. Reaching the far side of the obstacle's
> position is not clearing it, which is what separates the corridor-passage row from the endpoint.

**Sources.** `outputs/analysis_traversal_outcomes/summary.json`:
`by_height["0.05"].per_clip` — 14 rows with `passed_obstacle` true, all 14 with
`collided_obstacle` true; exactly one row with `reached_goal` true (`motion_key` "s4434",
`outcome` "collided_obstacle"). Identical at `by_height` 0.03, 0.08, 0.12, 0.2 and 0.3 (14 / 14 / 1
in each). Scene: `scene` {obstacle_x_m 1.2, corridor_half_width_m 1.4, goal_m [7.2, 0.0],
goal_tolerance_m 0.5}.

**Caveat for the writer.** This summary is evaluator v1 output; the working tree carries an
uncommitted `scene2motion/traversal_eval.py` with `EVALUATOR_VERSION = 2` and an explicit timeout
class (§4 of the audit). If the outcome table is regenerated under v2, regenerate these three
counts with it before pasting.

---

## B6 — §VI-A. The position channel has an executed outcome, not just a kinematic one

**Target:** §VI-A, the "other ways of asking" paragraph.
**Instruction:** insert immediately after the sentence ending "...both feet fail the support test
for most of the lift." (L272–273), as a continuation of that paragraph.

> That channel was also tracked, and its ladder is the clearest measurement we have of the trade
> between the two properties. Over six requested amplitudes, 24 fresh seeds each and one rollout
> per reference, the reference geometry improves with the request — at the fixed probe position
> the median tallest box the body clears rises from 0 m at the smallest request to 0.32 m at the
> largest — while completed tracking runs fall from 6 of 24 to 2 of 24, 1 of 24 and then 0 of 24
> at every amplitude of 0.16 m and above. Matched control references on the same seeds complete 22
> to 24 of 24 at every rung, so this is the request and not the route. The campaign's
> preregistered kill condition, an amplitude with tracked success at least 0.5 and median
> clearance at least 0.05 m, fired at no rung. Asking harder for the clearance buys reference
> geometry and spends execution.

**Sources.** `outputs/exp1c_stepover/receipt.json`: `per_amplitude.{0.05,0.08,0.12,0.16,0.2,0.28}`
→ `sonic_lift.success` 0.250 (k 6 / n 24), 0.0833 (2/24), 0.0417 (1/24), 0.000, 0.000, 0.000;
`sonic_ctrl.success` 1.000, 1.000, 0.9583 (23/24), 0.9167 (22/24), 0.9583, 1.000;
`median_clearance_at_fixed_m` 0.000, 0.0143, 0.0025, 0.0933, 0.1561, 0.3234;
`overshoot_x` 1.677–2.323; `n_seeds` 24; `fixed_probe_x_m` 3.6; `kill_condition`;
`verdict` "claim KILLED on this ladder"; `survived_amplitudes` [].
SONIC's reported success rate is `1 − terminated` over the launch's environments
(`gear_sonic/trl/callbacks/im_eval_callback.py:747`), i.e. exactly the *completed tracking run*
evidence level — not clearance after tracking, which this campaign never scored.

**Notes for the writer.** The median clearance rises across the ladder but is not monotone (the
0.12 m rung dips to 0.0025 m); write "rises from 0 m to 0.32 m across the ladder", not
"monotonically". A stronger sentence is available but needs a committed scalar first: joining
`rows.jsonl` (`clearance_at_fixed_m ≥ 0.05` on the `kin` rows) to the per-clip `sonic_clip` rows
in the same file gives 6, 8, 9, 13, 14 and 15 references of 24 clearing a 5 cm box at the fixed
position against 6, 2, 1, 0, 0 and 0 completed runs, with **0 references doing both at any of the
six amplitudes** — the §VI-B disjointness replicating in a second channel over 144 references.
That is a derivation from committed rows, not a receipt key; add the scalar to the receipt (as
§4 of the audit asks for the ±r union) before it goes in the paper.

---

## B7 — §VI-C. The prospective arms trade placement against the screen

**Target:** §VI-C, the "**Prospective test, in progress.**" paragraph.
**Instruction:** replace the sentence beginning "No setting yields a *valid step passing the
support test*..." and ending "...(median 0.48 → 0.36 / 0.22 s). The screen passes 33 and rejects
95." (L352–355) with the text below; keep the surrounding sentences.

> No setting yields a *valid step passing the support test* — a locally valid step with no period
> over 0.20 s outside the test — in any of its 32 references, and the four settings trade the two
> properties against each other rather than improving both. Pinning root height does shorten the
> longest period outside the support test, from a median of 0.48 s in the free setting to 0.36 s,
> and to 0.22 s when heading is pinned as well; it also costs the placement the screen was meant
> to protect, with references clearing the 5 cm box at the specified position falling from 7 of 32
> to 1 of 32, and to 2 of 32 with both pinned, while the step is produced in 19 and 11 of 32
> rather than 23. Pinning heading alone moves the other way on both counts: 8 of 32 placed, 27 of
> 32 produced, and the longest period outside the support test lengthens to a median of 0.74 s.
> The screen passes 33 of the 128 and rejects 95, and the settings whose references pass it most
> often are the settings that place the motion least often. Which side of that trade is right is
> exactly what the tracking outcomes decide.

**Sources.** `outputs/exp024_reference_contract/receipt.json` →
`stages.score.reference_summary_per_arm`:
free {median_max_unsupported_run_s 0.48, exact_clear_5cm_staged.k 7, elicitation.k 23,
primary_flagged.k 26, contact_consistent.k 0, n 32};
pin_y (root height pinned at 0.78 m) {0.36, 1, 19, 24, 0, 32};
pin_h (heading pinned at 0.0) {0.74, 8, 27, 29, 0, 32};
pin_yh {0.22, 2, 11, 16, 0, 32}. Arm definitions: `campaign_design.arm_contracts`.
Screen totals: 26 + 29 + 24 + 16 = 95 flagged of 128, 33 passed. Elicitation is the exp021
definition, a 3 cm whole-body box cleared somewhere on the route
(`experiments/exp024_reference_contract.py:128`, `ELICITATION_MIN_M = 0.03`).
Status of the tracked stage: `status` "running", `stage` "predicted", `launches` {} — the
outcomes remain pending and the paragraph's closing sentence must stay.

---

## B8 — §VIII. First obstacle-present evidence in the project

**Target:** §VIII, Conclusion and Next Steps.
**Instruction:** insert immediately after the sentence ending "...and comparing extra sampling, a
fixed extra-crouch adjustment and the measured correction under one budget." (L461–462), before
"If correction improves traversal completion...".

> The second of those is now buildable. In a two-motion operational probe, a collidable box of
> 0.2 × 2.0 × 0.30 m spawned at 0.5 m along the route, with its pose carried per motion inside the
> reference file, changed the achieved motion: a reference that advances to 6.06 m with the box
> absent stops at 0.29 m with it present, and its rollout ends at the evaluator's cutoff where the
> obstacle-free rollout ran to the end. That is an instrument check and not a result — two
> motions, physics seed 0, no local-traversal endpoint scored, and the probe's own record marks it
> as not campaign evidence — but it removes the reason every execution measurement in this paper had
> to be a replay against the obstacle's geometry rather than a rollout against the obstacle.

**Sources.** `outputs/probe_obstacle_present/report_h0.3.json`:
`comparison.per_motion[0]` (motion_key "s4434") → `max_root_x_m` {no_box 6.0567, box 0.2929},
`terminated` {no_box false, box true}, `differs` true;
`comparison.obstacle_has_physical_effect` true, `comparison.n_motions` 2;
`obstacle` {size_xyz [0.2, 2.0, 0.3], pos [0.5, 0.0, 0.15]};
`campaign_evidence` false; `table_metadata_in_pickle` true; `status` "complete".
The second motion (s4459) moves the same way, 0.985 m → 0.383 m, and was already cut off in both
arms — use the first only, as above.

---

## B9 — Repair paired-difference table. The two collision-free rows the table drops

**Target:** the paired per-scene difference table in §VI-D (labelled "Table III" at L386–391;
**Table IV** after the renumbering of §1 item 11).
**Instruction:** add the two rows below to the table — QP after the existing QP row, heuristic
after the existing heuristic row — and append the sentence to the paragraph that follows it.

> | QP teacher, collision-free | +2.8 | 0.0 to +6.9 | 8 : 0 |
> | heuristic, collision-free | 0.0 | 0.0 to 0.0 | 0 : 0 |

> The analysis has six rows and the two omitted ones are ceiling effects worth stating: the QP
> teacher's references are collision-free in 87.2 % of pairs uncorrected and reach 100 % under
> correction against 97.2 % under resampling, a +2.8 pp difference whose interval touches zero,
> and the heuristic's reach 100 % either way. The correction's collision-free advantage is
> therefore a property of the proposer that starts furthest from the constraint.

**Sources.** `outputs/analysis_repair_paired_bootstrap/summary.json`:
`results[2]` {proposer qp, endpoint collision_free, rate_a 1.0, rate_b 0.9722,
paired_difference_pp 2.778, bootstrap_95_pp [0.0, 6.944], discordant_a_only 8, discordant_b_only 0};
`results[4]` {proposer heuristic, endpoint collision_free, rate_a 1.0, rate_b 1.0,
paired_difference_pp 0.0, bootstrap_95_pp [0.0, 0.0], 0 : 0}.
Uncorrected rates: `outputs/phase4e_architecture_v2_s8/experiment.json` →
`summary.qp.collision_free_rate` 0.8715, `summary["qp-resample3"].collision_free_rate` 0.9722,
`summary["qp+2"].collision_free_rate` 1.0.
Design of the interval: `n_scenes` 36, `n_boot` 30000, `descriptive_only` true.

**Correction to the audit.** The audit's ninth §2 bullet names only `results[2]` but asks for
"completeness of the six-row analysis"; completeness needs `results[4]` as well, which is why both
rows are given here. It also calls the destination "Table III", which is the current label; under
the renumbering the audit itself prescribes (§1 item 11) this table becomes Table IV.

---

## B10 — §VI-E. The tracked ducking negative never touched the learned proposer

**Target:** §VI-E, "Does the benefit survive execution?".
**Instruction:** insert immediately after the ducking-tracking sentence that §1 item 3 installs,
ending "...the ~14 s clip length bounds the 305 rollouts that were not cut off."

> That study also never touched the proposer whose repair result this paper reports. Its 859
> rollouts came from the heuristic proposer only — uncorrected, one correction and two corrections
> — so the learned proposer, whose collision-free rate rises from 72.9 % to 99.3 % under two
> corrections, has never been tracked at all: it has evidence at the reference-geometry level and
> at no level beyond it. The negative execution result bounds the heuristic pipeline; the headline
> repair gain is untested in execution in either direction.

**Sources.** `outputs/exp1b_execution_clearance_v2/receipt.json`:
`selection.methods` ["heuristic", "heuristic+1", "heuristic+2"], `selection.n_selected` 859,
`selection.n_unique_clips` 526, `selection.n_scenes` 36, `status` "completed_negative";
`outcomes` {terminated 554, reached_first_obstacle 200, passed_last_obstacle 0,
executed_success 0}. TCN rates: `outputs/phase4e_architecture_v2_s8/experiment.json` →
`summary.tcn.collision_free_rate` 0.7292, `summary["tcn+2"].collision_free_rate` 0.9931.

---

## B11 — Figure. The screen across families (built, unused)

**Target:** §VI-C, alongside B1.
**Instruction:** place `docs/figures/fig6_screen_across_families.pdf` as Fig. 6 and cite it in
B1's first paragraph — change "The same feature was measured on a second behaviour family" to
"Fig. 6 **[FIG-6-FAMILIES]** measures the same feature on a second behaviour family". Caption
below.

> **Fig. 6. The reference screen across three corpora and two behaviour families.** (a) How well
> the longest period in which neither foot meets the support test ranks the controller's evaluator
> cutoffs in each corpus, with bootstrap intervals: 0.997 (0.987–1.00) on the 64 step-prompted
> references, 0.921 (0.820–0.983) on the 144 position-channel references, and 0.674 (0.625–0.721,
> cluster bootstrap over 36 scenes) on the 526 ducking references. (b) Within the ducking family,
> the same ranking for each preregistered feature group in every stratum: the support feature is
> the only one that stays above chance throughout. (c) The screen used as a filter at its
> calibrated 0.20 s threshold — sensitivity 1.00 / 0.94 / 0.91 against specificity 0.73 / 0.56 /
> 0.30. The ranking transfers across behaviour families and generation pipelines; the operating
> point does not, and on ducking references the screen is not usable as an accept/reject rule.

**Sources.** `docs/figures/fig6_numbers.json` → `corpora[*]` {auc, ci95, sensitivity,
specificity, n, terminated, survivors} for keys `exp021_step`, `exp1c_lift`, `duck`;
`duck_strata` (seven strata); `screen_threshold_s` 0.20.
Provenance: `inputs.analysis_trackability_contract.sha256`
`6c0c73ce0d29578064714e8d42451ee2c7f7c32cfd2ea8a7fa4519186443612b` and
`inputs.analysis_duck_contract.sha256`
`9d05fa427d6f03a3d981a08ff15c78028f37ed97ff677d6c284b3847dab62fb4`, both equal to the current
receipts on disk (checked with `sha256sum`), so this figure — unlike Fig. 5 before §1 item 16 —
is already bound to the receipts it draws.
Regeneration: `experiments/fig_screen_across_families.py` under
`/home/linjiw/isaaclab-install/env_isaaclab/bin/python`; it reads committed outputs only.

---

## Where the audit and the receipts disagreed

1. **Source path, §2 bullet 2.** `summary.exp1c_all` (0.974), `summary.exp1c_lift` (0.921),
   `summary.transfer_auc` (0.916 / 0.971 / 0.916 / 0.904) and
   `summary.exp1c_ctrl_calibration_corpus` (0.513) are keys of
   `outputs/analysis_trackability_contract/receipt.json`, not of
   `outputs/analysis_duck_contract/receipt.json` ("same receipt"). Every value checks out.
2. **Number, §2 bullet 4.** The unstaged position (x = 3.6 m) does **not** give "2 of 64 at every
   height": `results[1].by_height` gives 2 at 0.03 / 0.05 / 0.08 m, 1 at 0.12 m and 0 at 0.2 and
   0.3 m; N90 is 73 fresh draws at 5 cm and 147 at 12 cm, and undefined above.
3. **Scope, §2 bullet 9.** "Completeness of the six-row analysis" needs `results[4]` (heuristic,
   collision-free, 0.0 pp, 0 : 0) as well as `results[2]`; and the destination table is Table IV
   once §1 item 11's renumbering is applied.
4. **Half a result, §2 bullet 1.** The EXP-026 bullet quotes the pooled contact-minus-speed
   difference and its interval but not the within-scene one, which is +0.092 with interval
   −0.018 to +0.210 and spans zero (`summary.contact_minus_speed_within_scene.ci95`). B1 states
   both.
5. **Freshness, §2 bullet 5.** `outputs/analysis_traversal_outcomes/summary.json` is evaluator v1
   output while the working tree holds an uncommitted evaluator v2; the 14 / 14 / 1 counts must be
   regenerated if the outcome table is.

Everything else in §2 reproduced exactly: 0.674 vs 0.441 and +0.233 (0.120–0.362) with verdict
`contract_transfers_to_the_duck_family` over 526 clips and 36 scenes; 44 / 43 / 0 and 8 / 8;
14 of 14 `passed_obstacle` rows also `collided_obstacle` with exactly one `reached_goal`;
0.250 / 0.083 / 0.042 / 0.000 / 0.000 / 0.000 against 0.917–1.000 controls with overshoot
1.68–2.32× and the killed verdict; the EXP-024 medians 0.48 / 0.74 / 0.36 / 0.22 s with placement
7 / 8 / 1 / 2 of 32; s4434 at 6.06 m and 0.29 m with the cutoff flipping false → true, marked
`campaign_evidence: false`; +2.8 pp (0.0–6.9) with 8 : 0 discordant pairs; and
`selection.methods` carrying the three heuristic arms only.
