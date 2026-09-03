# EXP-026 protocol — does the reference screen predict duck-family cutoffs, or is it speed?

**Status:** preregistered (2026-09-03, before the first feature was computed). The feature
groups, the primary comparison and the decision rule are copied unchanged from the plan of
record's EXP-026 row, fixed in commit `0379d47` (2026-09-02) before this analysis existed:

> does a contact or root feature predict the 554 exp1b terminations beyond speed? · 526 unique
> clips (first rollout per clip); three preregistered feature groups (planar speed, crouch depth,
> contact); leave-one-scene-out AUC within dip bins and route classes · counts as contract
> transfer only if contact-feature AUC > speed-feature AUC; else reported as speed-limited and
> confounded by the 14 s cap

This file makes those words exact and adds nothing to them. No new samples are generated and no
GPU is used: it re-scores committed archives on CPU.

## 1. Why this is worth running

The reference screen — reject a reference whose longest period with neither foot meeting the
support test exceeds 0.20 s — currently predicts the controller's stopping rule on **one
behaviour family**, the step-over, through two different actuation channels: the text-prompted
pool (exp021, AUC 0.997) and the position-channel ladder (EXP-1C lift arms, AUC 0.92), with
cross-family logistic transfer 0.92 / 0.90. Both are stepping motions that leave the ground.
If the screen is a property of *this controller's evaluator* rather than of *stepping*, it
should also separate cutoffs in the **duck family**, where the motion is a crouch and the
references were never expected to leave the ground.

The duck campaign (EXP-1B) has a known confound the plan named in advance: a 14 s clip cap forced
reference speeds up to ~1.8 m/s, and speed alone may explain the cutoffs. EXP-026 is the test
that separates the two readings, and its decision rule was written to make the negative outcome
reportable rather than quiet.

## 2. Data (all committed; nothing is generated)

- **References:** the 526 unique duck clips of EXP-1B, keyed by `clip_key` in
  `outputs/exp1b_execution_clearance_v2/rows.jsonl` and stored as `(T, 36)` qpos at 25 fps in the
  content-addressed cache `scene2motion/demo_outputs/clips` (cache version 2, noise stream v2,
  model `ARDY-G1-RP-25FPS-Horizon52`; each `.npy` has a sidecar `.json` carrying `scene_id`,
  `seed`, `peak_dip_m`, `repair_iteration`, `n_frames`, `fps`). All 526 keys were confirmed
  present before this protocol was written; the analyser **refuses** if any key is missing, if a
  sidecar disagrees with the row on `scene_id`, or if `fps ≠ 25` or `cache_version ≠ 2`.
- **Outcomes:** the **first rollout of each clip** in file order from the same `rows.jsonl`
  (physics seed 0, SONIC release evaluator, schema-v1 achieved archives): `terminated` is the
  outcome label. 344 of the 526 first rollouts are terminated. The 859-rollout campaign total
  (554 terminated) counts repeated rollouts of the same clip under different selection methods
  and is **not** the denominator here.
- **Support thresholds:** the calibrated pair from the hash-locked exp016 receipt
  (`outputs/exp016_threshold_calibration/receipt.json`, sha `f6dba8be…`; sole ≤ 4.65 cm,
  planar speed ≤ 1.18 m/s), loaded through
  `analyze_trackability_contract.load_support_thresholds`. The screen threshold stays
  `max_unsupported_run_s > 0.20 s`.
- **Features:** exactly `analyze_trackability_contract.features(body, qpos, sup_h, sup_v, fps=25)`,
  the same function the step-family analysis used, unmodified.

## 3. The three preregistered feature groups

One primary scalar per group is named now; the other members are reported but cannot replace the
primary in the decision rule.

| group | primary scalar | also reported | what it represents |
|---|---|---|---|
| **G1 speed** (the confound) | `max_root_planar_speed` | `mean_root_planar_speed` | how fast the 14 s cap forced the reference to travel |
| **G2 crouch depth** | `peak_dip_m` (committed row) | `ref_min_overhead_m` (row), `root_z_min` | how deep the requested adaptation was |
| **G3 contact** (the screen) | `max_unsupported_run_s` | `bilateral_flight_frac`, `mean_support_feet` | how long the reference asks for no support |

Higher values of every primary are hypothesised to predict a cutoff, so each AUC is computed in
that fixed direction and **not** flipped to whichever side scores higher.

## 4. Endpoints

1. **Primary — leave-one-scene-out AUC per group.** For each of the 36 scenes, hold out its
   clips, and score the held-out clips by the group's primary scalar (a single feature needs no
   fitting, so LOSO is a stratified evaluation, not a training loop; the identical procedure is
   applied to all three groups so they are comparable). Pool the held-out scores and report one
   AUC per group with a **cluster bootstrap over scenes** (2,000 resamples, scenes drawn with
   replacement, seed fixed in the analyser).
2. **Primary comparison and the decision rule.** Report `AUC(G3) − AUC(G1)` with its cluster
   bootstrap interval. Per the plan: **contract transfer is claimed only if the contact AUC
   exceeds the speed AUC.** If it does not, the result is reported as *speed-limited and
   confounded by the 14 s clip cap*, and the screen's cross-family evidence stays the two
   step-family channels.
3. **Stratified replication.** The same three AUCs within **dip bins** (`peak_dip_m` in
   [0.25, 0.35), [0.35, 0.45), [0.45, 0.50]) and within **route classes** (beam count parsed from
   `scene_id`), each with its n and its terminated count. A stratum with fewer than 20 clips or
   fewer than 5 of either outcome is reported as "not evaluable", never as an AUC.
4. **The frozen screen as a classifier on this family.** Sensitivity and specificity of
   `max_unsupported_run_s > 0.20 s` with Wilson intervals, and the same for the post hoc
   step-family cut `> 0.28 s`, reported as a transfer check of the *threshold*, separate from the
   ranking question in (1)–(2).
5. **Descriptive context.** The distribution of each primary by outcome, the fraction of duck
   references that are floats at all, and the correlation between the three primaries (a contact
   feature that is merely a proxy for speed must be visible as such).

## 5. What this campaign cannot establish

- The outcome labels were public before this analysis (REPORT §25), so this is a **post hoc
  analysis of a completed campaign**, exactly like the step-family contract analysis. The
  protection is that the groups, the primary scalars, the direction and the decision rule were
  fixed in the plan before the features existed, and this file is sha-bound into the receipt.
- One physics seed, one rollout per clip, schema-v1 archives, one tracker checkpoint.
- EXP-1B's own limitation stands: 0/859 rollouts traversed, and the 14 s cap is a property of
  that campaign, not of ducking. A negative result here therefore does **not** show that the
  screen fails for ducking in general; it shows that on this corpus the cutoffs are explained at
  least as well by speed.
- Nothing here is evidence about obstacle-present traversal.

## 6. Gates and artifacts

Clean worktree; `outputs/analysis_duck_contract/` empty before the run; every input file hashed
into the receipt (rows, the 526 clip `.npy`/`.json` pairs by content hash, the exp016 threshold
receipt, `analyze_trackability_contract.py`, this protocol); planned denominator 526 with a
refusal if any clip is missing; rows written before the summary. Budget: CPU only, minutes.
Driver `experiments/analyze_duck_contract.py`, tests `tests/test_analyze_duck_contract.py`.
