# The verified data engine: record schema, tiers, and downstream use

**Written:** 2026-09-02 under the PI mandate of the same day (Scene2Motion-G1 as a closed-loop
generator of verified, execution-audited traversal motions and datasets). This is the
consumer-facing specification the paper's C3 promises. Every field names the committed ledger
that already carries it, or is marked *planned* with the campaign that will supply it. Nothing
here is a training claim; downstream use is stated as what the labels make possible.

## 1. Pipeline stages and what each stage emits

```
Route layer  →  Actuation layer  →  Directional MuJoCo tracing  →  Measured-deficit repair  →  Trackability gate  →  Tracker execution (SONIC)
(scene, start/goal,  (prompt + native   (overhead vs lateral       (Δq = e/|g′|, 3τv lead,      (support-envelope     (achieved-state archive,
 preference, A*)     constraint spec)   clearance traces)          bounded, refusal)             contract features)    evaluator/physical outcome)
```

Each stage appends fields to one record; a record that stops early carries the stage at which
it stopped and the reason. The engine never drops a record: refusals and failures are rows with
their deficit, not absences (corpus pilot v2: 300/300 scenes accounted for, 26 refused with a
named deficit).

## 2. Record schema

| group | field | meaning | where it exists today |
|---|---|---|---|
| identity | `scene_id`, `seed`, `physics_seed`, `noise_stream_version`, `cache_version`, `motion_key` | inference units and sampler provenance; v1 and v2 never pooled | `outputs/corpus_pilot_v2/manifest.jsonl`, `outputs/exp1b_execution_clearance_v2/rows.jsonl`, every EXP-02x `rows.jsonl` |
| scene | `scene.{n_beams, beam_height, beam_width, gap}` or `obstacle.{x_m, depth_m, height_m}`, `route_len_m`, start/goal, `preference`, `target_m` | full geometry, endpoints and the requested margin | corpus pilot manifest (duck family); EXP-022A/EXP-024 rows (step family: `obstacle_x_m`, `obstacle_depth_m`, graded heights) |
| request | `prompt`, `proposer`, `max_repairs`, constraint-spec hash, channels written (root_2d / root_y / heading / joints) | what the prior was asked, through which native channel | exp021/exp023/EXP-024 rows (`prompt`, spec sha256, channel usage audit) |
| response | reference `qpos` (200 frames, 36-D, archived for every clip incl. gate failures), `progress_ratio`, route MAE | the full-body reference trajectory and route fidelity | `qpos.npz` beside every campaign ledger; `supporting_motion` in exp023/EXP-024 rows |
| clearance traces | overhead vs lateral clearance along the route; `ref_min_overhead_m`, `peak_dip_m` (duck); `lift_x_m`, `lift_height_m`, `n_lift_regions`, exact clears per graded height at fixed centres (step) | the measured deficit the repair acts on; exact obstacle-centred outcome | `verify/trace.py` (duck; corpus pilot, exp1b rows); exp021 rows, `outputs/analysis_exact_centre_cost_curve/`, EXP-022A reference rows |
| repair | iterations used, Δq per iteration, anticipation lead, saturation flag, `outcome ∈ {accepted, accepted_margin, rejected, refused}` | the closed-loop correction and its bound | corpus pilot manifest (`outcome`, `refusal.*`); Table 2 arms (`phase4e_architecture_v2_s8`) |
| refusal | `refusal.{functional, deficit_m, scene_value_m, best_available_m, obstacle, obstacle_x}` | quantified, located reason an instance is unsolvable | corpus pilot manifest (26/300) |
| contract | `max_unsupported_run_s`, `bilateral_flight_frac`, `mean_support_feet`, `ballistic_ratio`, `root_z_max`, … (18 features); gate flags `run > 0.20 s` (calibrated) and `run > 0.28 s` (post hoc, ≥ 8 frames) | support-envelope compliance of the *reference*; the launch-budget filter | `outputs/analysis_trackability_contract/rows.jsonl`; EXP-024 `predictions.jsonl` (prospective, committed before SONIC) |
| execution | `tracker_terminated`, `valid_frames`, `tracker_reported_progress`, achieved `qpos` archive (50 Hz, schema v2), guarded retention per height, `stalled`, `passed_within_lateral_corridor`, `finished_beyond_obstacle`; duck: `exec_min_overhead_m`, `exec_collision_{overhead,lateral}`, `executed_success` | evaluator outcome and achieved-state geometry replay (obstacle absent from Isaac) | EXP-022A `achieved_rows.jsonl`, exp1b rows, achieved-state archives |
| divergence | first-firing evaluator term and its timestamp (offline recomputation), termination − first no-support onset, pelvis height and tilt at cutoff, physical outcome class `fell / stalled / walked_through / cleared` with terminations disabled | "evaluator cutoff" vs "dynamic fall", per record | `termination_snapshot` in the contract rows; **planned:** EXP-028 Part A rows (`part_a_rows.jsonl`, `part_a_evaluator_terms.npz`) |
| test–retest | terminated flag per physics seed, agreement | how much a single-seed label can be trusted | exp1b (51/179 disagree); **planned:** EXP-028 Part B |
| provenance | HF revision + denoiser sha, ARDY runtime commit + source manifest, `g1.xml` sha, threshold receipt sha, tracker commit + core manifest + checkpoint sha, resolved termination config, protocol sha, host-gate report | reproducibility binding | every receipt from exp021 on; resolved termination dump from EXP-028/EXP-024 |

Torque and joint-velocity traces are **not** in the achieved-state archive (schema v2 stores
qpos only); a schema-v3 export is the first extension once EXP-028 lands.

## 3. Tiers (house rule 10) and the numbers that exist

| tier | duck family | step family |
|---|---|---|
| generated | 300 scenes / 301.9 s (corpus pilot v2, heuristic proposer, ≤ 2 repairs, RTX 5080) | 5,393 v2 references; ≈ 0.2 s/clip batched |
| kinematically verified | 268 (192 accepted at 0.18 m + 76 collision-free below margin); 6 rejected; 26 refused with reason | exp021 64 scored at 7.2 s/clip; 49/64 lift; exact 5 cm at 1.2 m 12/64 |
| gate-accepted (contract) | EXP-026 (CPU, appendix) | 8/64 pass the calibrated gate; 0/44 lifting clips |
| tracker-executed | 859 rollouts / 526 references; 0 traversals under the 14 s-cap protocol | 352 references; 0 retained at the box; 53/64 evaluator cutoffs, 0 falls |

The per-GPU-day figure is quoted only as the extrapolation 7.7 × 10⁴ kinematically verified duck
records per GPU-day from the 302 s pilot, never as "10⁵ verified clips".

## 4. Downstream use, stated without a learned model

- **Launch-budget filter.** At the calibrated gate the engine would have skipped every SONIC
  launch that the evaluator cut off (53/53) at the cost of 3/11 survivors; the post hoc ≥ 8-frame
  cut skips 51/53 at no survivor cost (both on exp021; prospective test = EXP-024).
- **Supervised pre-training of scene-conditioned whole-body navigation policies.** The records
  pair scene geometry, request, and a full-body reference with a measured clearance trace and a
  contract label; the negatives (refusals, gate failures, evaluator cutoffs) come with signed
  deficits and locations, which a classifier or a cost model can consume directly.
- **Privileged-to-proprioceptive distillation under real tracking dynamics.** The achieved-state
  archives give, for every executed reference, what the tracker actually did at 50 Hz beside what
  it was asked to do; EXP-028 adds the physical outcome class and the first-firing term so a
  student can be trained on references a tracker can follow rather than on kinematic mirages.
- **Not** "a dataset for training" in the paper's claims until a downstream result exists; the
  paper releases the schema, the tiers and the records.

## 5. Modularity contract

A future step-over channel (a fine-tuned prior, a contact-consistent guidance term, a
re-prompting policy) enters the engine at the actuation layer and must emit the same record:
the clearance trace, the contract features and gate flags, and, if launched, the evaluator and
physical outcomes. The repair harness (`scene2motion/verify/loop.py`) and the gate
(`analyze_trackability_contract.gate_predictions`) are unchanged by the channel; the audit
protocol (elicitation → exact placement → event timing → contract → paired retention) is what
decides whether the channel is "physically grounded".
