# exp1b frame red-flag diagnosis — 2026-08-31

Question: are the 216 suspect rows (all `passed_last_obstacle=false`, `exec_goal_error_m`
5.4–24.6 m at progress 1.0, clearance pinned at the 0.6 m cap in 154/216, mpjpe_g ~5194 mm vs
mpjpe_l ~68.7 mm) explained by an un-subtracted Isaac env-origin / motion-lib normalization
offset, or by genuine tracker drift?

**Verdict: genuine tracker drift (forward-progress stall), plus one archive artifact — the
post-reset teleport frame of review defect #1. There is no frame-offset bug to subtract.**

All numbers below are pure-numpy overlays of `achieved_qpos.npz` root XY against the reference
clip root translation (clip cache `scene2motion/demo_outputs/clips`, keys via `run/join.jsonl`)
over the 5 intact launches `heuristic_00..04_seed0` — 180 envs, 137 terminated / 43 not.
Mapping used: achieved[j] = post-step state at t=(j+1)·0.02 s ↔ 25 fps reference frame (j+1)/2
(no frame-0 sample; odd j land exactly on reference frames).

## 1. Env-origin / normalization offset: refuted

- Frame-0 XY offset |achieved[0] − ref[0]|: **mean 0.026 m, median 0.025 m, max 0.041 m**
  across all 180 envs. An un-subtracted 36-env Isaac grid would put env origins at multiples of
  the env spacing (metres); nothing of the kind is present. The callback's
  `scene.env_origins` subtraction (`sonic_state_export.py`, `_snapshot_achieved_qpos`) works.
  The residual ~2.5 cm is one 50 Hz control step of forward walking (0.9 m/s × 0.02 s ≈ 0.02 m):
  the first archived sample is the state *after* the first policy step (defect #2's convention).
- SONIC motion-lib `root_trans_offset` normalization is a no-op here: every reference clip
  already starts at the origin (ref[0] ≈ (0,0)), and achieved[0] lands on it.
- The offset is **not constant within a rollout**: XY error quartile means across envs are
  **Q1 0.48 → Q2 1.88 → Q3 3.61 → Q4 5.43 m**, growing in 173/174 envs with ≥8 comparable
  frames. Subtracting the frame-0 offset leaves metres of residual; subtracting any constant
  cannot make achieved ≈ reference. This is cumulative lag, not a frame mismatch.

## 2. What the robot actually does: it stalls, upright, before the first beam

- Median furthest forward progress: **max x = 2.07 m** (mean 2.77 m) against routes of
  10.5–25 m. The first beam of every demo scene is at **x = 4.0 m** (`demo/scene_builder.py`,
  `BEAM_X`); most rollouts never reach it. Playback speed is correct (non-terminated envs time
  out at exactly `get_motion_num_steps` = 2× clip frames, i.e. 1× real time), so the lag is the
  policy's, not a resampling error.
- Dose structure (launches 00–04, all method=heuristic):
  - beam_h 0.95 m, peak_dip 0.50 m (ref pelvis min ≈ 0.31 m): n=91, **87 % terminated**,
    median max x **1.77 m**;
  - beam_h 1.05 m, peak_dip 0.35 m (ref pelvis min ≈ 0.46 m): n=89, **65 % terminated**,
    median max x **2.51 m**.
  Deeper requested crouch ⇒ earlier stall and more terminations: a genuine capability limit of
  the tracker for deep-crouch locomotion (EXP-011's straight-walk duck-deep already sat at
  0.625 success), not an artifact.
- mpjpe_l 68.7 mm vs mpjpe_g 5193.7 mm is exactly this picture: posture tracks, global root
  does not. The launch-00 "success" bucket is the two non-terminated stalled envs whose
  mean root-XY error I measure at 4.91 and 5.42 m — averaging ≈ 5.2 m = SONIC's 5193.7 mm.

## 3. The one real archive artifact: the post-reset teleport frame (defect #1)

- For **all 43** non-terminated envs, the final archived frame satisfies
  |achieved[−1] − ref[0]| = **0.000 m** (reset pose at the env origin) after a last-step XY jump
  of **0.17–15.95 m** (mean 5.38 m). `motion_time_out` fires during iteration ref_len−1 and the
  in-step reset is what gets archived; `valid_lengths` = ref_len keeps it.
- The lost join had already worked around this at read time: all 62 non-terminated rows in
  `run/rows.jsonl` carry `dropped_teleport_frame: true`, and their `exec_goal_error_m` matches
  distance from achieved[−2] (the real last pose) to the goal, not from the teleport pose.
  The archive itself remains poisoned for any other consumer; fixed at the source in
  `sonic_state_export.py` (schema v2), with the v1 handling kept in the new join.

## 4. "Early termination" (154/216 = 71.3 %): real terminations, not falls, not timeouts

- The 0.6 m clearance cap rows are exactly the 154 terminated rows: stalled at median x ≈ 2 m,
  they never came within `CLEARANCE_MARGIN = 0.6 m` (`robot.py:41`) of a beam at x ≥ 3.88 m, so
  min clearance reads the search cap.
- At the last archived frame of terminated envs: pelvis z median **0.451 m** (only 3/137 below
  0.35 m), height error vs reference median **0.084 m**, projected-gravity tilt difference
  median 0.074. These robots are upright and crouched, not on the floor.
- What fires is SONIC's *local* eval termination set (`terminations/tracking/eval.yaml`):
  `exceeded_anchor_height` (z-only, 0.25 m), `anchor_ori_full` (1.0), `ee_body_pos` (0.25 m on
  `body_pos_relative_w`, i.e. re-anchored to the robot). **None of these watch XY lag**, which
  is why 43 envs stalled 2–16 m behind the reference yet ran to `motion_time_out` with
  progress 1.0. XY lag at the moment of termination: median **2.91 m** (>1 m in 125/137). The
  violating step itself is excluded by the progress-based trim, so the archived last frame
  looks benign; the terminations are still genuine tracking-error events — most plausibly the
  anchor-height term as the reference rises out of a dip the robot is still descending into,
  or a re-anchored end-effector error — occurring long after forward tracking was already lost.
- Per `docs/exec-gate-audit.md`, termination before clearing the obstacle is an **execution
  failure**, not a censored clearance observation; the rebuilt join keeps that convention.

## 5. Consequences for the recovered campaign

1. `exec_goal_error_m` 5.4–24.6 m = route_len (10.5–25 m) minus stall x — arithmetically
   consistent with the archives; no join subtraction step was missing.
2. The archive-level fix needed is defect #1 (+ the #2 convention metadata), now in
   `sonic_state_export.py` schema v2; old v1 archives are joined with the teleport frame
   dropped for non-terminated rollouts (recorded per launch as `archive_schema_version`).
3. `run/rows.v1-suspect.jsonl` (the 216 rows) is numerically coherent but is superseded by the
   from-scratch rebuild; its `valid_frames` count still includes the teleport frame it dropped.
4. The scientific content stands: at these dips the tracker mostly cannot traverse the route,
   so the execution gate will be dominated by termination-as-failure rows, and clearance-loss
   observations come from the minority that reach an obstacle.
