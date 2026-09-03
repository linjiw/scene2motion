# EXP-030 protocol — the stepping pool with the obstacle actually in the scene

**Status: preregistered** (2026-09-03, before the first launch). Arms, pool, endpoints,
predictions and kill conditions below are fixed; the file's sha256 is bound into the receipt.

## 1. Why this campaign exists

Every executed result in this project — EXP-1B, EXP-1C, EXP-011/012/014, EXP-022A, and the
pending EXP-024/EXP-028 — tracked references with the **obstacle absent from the physics scene**
and then replayed the achieved states against our MuJoCo collision model. Every sentence about
execution therefore rests on one untested assumption: *that replaying a trajectory against a box
the robot never touched predicts what would have happened if the box had been there.*

On 2026-09-03 that stopped being untestable. An obstacle can now be put in the tracker's scene and
the robot feels it (REPORT §47): a 0.30 m box stopped a reference that otherwise walked 6.06 m.
EXP-030 asks the two questions that follow:

- **Q1 (the proxy question).** Does the obstacle-absent replay predict the obstacle-present
  outcome, per reference? If it does, the project's executed results keep their meaning cheaply.
  If it does not, every one of them inherits a stated caveat.
- **Q2 (the endpoint the project has never measured).** With the obstacle present, does any
  prompt-elicited reference **complete local traversal** — pass through the corridor, past the
  obstacle, to the goal, upright and inside the time limit?

Neither question needs a new generator sample: the pool is archived.

## 2. Pool, controller and scene

- **Pool:** the 64 archived EXP-021 references
  (`outputs/exp021_elicited_lift_distribution_v2/qpos.npz`), the same pool EXP-022A tracked, so
  the two are directly comparable. No new ARDY samples; no seeds spent.
- **Controller:** SONIC release checkpoint `sonic_release/last.pt` (sha `e6bdab3f…`), release
  evaluation terminations (`+manager_env/terminations=tracking/eval`), physics seed 0, one
  rollout per reference, 32-environment launches in EXP-022A's chunk order.
- **Tracker source:** this campaign **requires** the fork commit that lets `add_table` work
  without `add_object` (`7c63c53`, reverted in `350cae1` on the legacy checkout so the
  obstacle-absent campaigns can keep EXP-022A's pinned manifest). **Amendment, 2026-09-03, before
  the first launch:** per the owner's two-checkout ruling recorded in `CLAUDE.md`, this campaign
  runs against the dedicated patched worktree **`/home/linjiw/lucid/GR00T-WBC-exp029`** (branch
  `exp029-obstacle-present`, pinned at `7c63c53`), never against the legacy checkout, and all
  three arms — including `absent` — run there so the comparison stays internally consistent.
  EXP-030 declares **its own** tracker baseline: the
  receipt records the checkout commit and the full core-source manifest, and the campaign
  **refuses to launch** unless the `table_to_robot_contact_sensor` fix is present. It does not
  assert equality with EXP-022A's `44e98c45…`; the two manifests differ by that fix alone, and
  the `absent` arm below measures whether the difference is inert.
- **Obstacle:** a corridor-spanning box at the scene-specified centre **x = 1.2 m**, depth
  0.20 m, width taken from `stepover_eval.step_scene` so the physics box matches the geometry our
  collision model has always scored. Carried as per-motion `table_pos` / `table_quat` **inside the
  motion pickle** (REPORT §47: the command-line position does not survive a reset), with
  `table_size` as the cuboid's full x/y/z extents and the position its centre.
- **Scene for scoring:** start (0, 0), goal (7.2, 0), the same obstacle, scored by
  `scene2motion.traversal_eval.evaluate_traversal`, which requires passing the obstacle **inside
  the corridor** — walking around it is a failure, because this is local traversal, not
  navigation.

## 3. Arms (paired; identical references, identical seed)

| arm | obstacle in physics | why |
|---|---|---|
| `absent` | none | replicates EXP-022A's configuration on this tracker build; the control for Q1 and the check that the fork fix is inert |
| `present_05` | 5 cm box at x = 1.2 m | the project's staged endpoint height; the primary arm |
| `present_20` | 20 cm box at x = 1.2 m | a graded, harder obstacle; separates "the box is too small to matter" from "the robot cannot pass any box" |

Six launches of 32 (two chunks × three arms). Budget ≈ 6–10 GPU-minutes plus CPU scoring; the
measured need is ≈ 3.8 GiB VRAM and ≈ 6.8 GiB host RAM per launch
(`scene2motion.host_gate.SONIC_LAUNCH_GATE`).

## 4. Endpoints (planned denominators: every rate is over all 64 assigned trials per arm)

1. **Outcome class** per reference from `evaluate_traversal`, in its preregistered precedence
   `fell > collided_obstacle > collided_wall > cutoff > timeout > stalled > completed`, reported
   as the full breakdown, never collapsed to a single success rate.
2. **Local traversal completion** per arm, with a Wilson 95 % interval. This is the endpoint the
   project has never measured.
3. **Q1, the proxy check.** For `present_05`, compare the physics-measured class against the
   **replay-inferred** class obtained by scoring the `absent` arm's achieved states against the
   same 5 cm box (exactly what §45 did to EXP-022A). Report the confusion matrix over all 64,
   Cohen's κ, and the per-class agreement fraction. The comparison is per reference and paired.
4. **How much the obstacle changes the rollout:** paired difference in maximum achieved root x
   (`absent` − `present_05`), its median and interquartile range, and the count of references
   whose maximum root x falls by more than 0.05 m. A reference that never reaches the box must
   show ≈ 0.
5. **Agreement of the `absent` arm with EXP-022A** (termination flag and valid length per
   reference), which measures whether the fork fix plus co-tenancy changed anything.

## 5. Predictions, fixed now

- **P1 (control).** The `absent` arm reproduces EXP-022A: ≥ 58/64 references agree on the
  termination flag. Below that, the fork fix or the run conditions are not inert and every
  cross-campaign comparison in the paper is re-scoped before anything else is claimed.
- **P2 (completion).** Local traversal completion is **0/64** in both present arms. This is a
  prediction of failure and it is stated so that a non-zero result is a finding rather than a
  surprise: any completion is the project's first measured local traversal and is reported
  prominently, with its reference identified.
- **P3 (the proxy).** The replay-inferred and physics-measured classes for `present_05` agree on
  ≥ 80 % of references (κ ≥ 0.6). Below that, the paper states that obstacle-absent replay is a
  biased proxy and names the direction of the bias.
- No prediction is made about the `present_20` breakdown; it is descriptive.

## 6. Kill and refusal conditions

- Refuse to launch unless the tracker carries the `add_table`-without-`add_object` fix, and record
  the checkout commit and core-source manifest in the receipt.
- Refuse a non-empty output directory for a fresh campaign; preserve any refused attempt beside
  the ledger and resume on the same directory only through the documented resume path.
- Refuse unless the host gate passes (≥ 5,500 MiB free VRAM, ≥ 9,500 MiB available RAM); record
  concurrent Isaac processes in every launch record.
- If any launch returns non-zero or archives fewer than 32 rollouts, stop and preserve; do not
  re-run a completed arm with different settings.
- If P1 fails, the campaign still completes and is reported, but Q1's answer is scoped to this
  tracker build and the disagreement is the headline rather than the proxy result.

## 7. Statistics and scope

One route, one scene, one obstacle position, physics seed 0, one rollout per reference. Rates are
Wilson intervals over the 64 references within this single scene and are not generalised. Paired
per-reference comparisons are descriptive at n = 64; κ carries a bootstrap interval over
references. The obstacle is now present in physics, so `collided_obstacle` here means the robot
actually contacted the box — unlike every earlier campaign, where it meant the recorded motion
intersected the box's volume in replay. That distinction is stated wherever the two are compared.

Driver `experiments/exp030_obstacle_present.py`; tests `tests/test_exp030_obstacle_present.py`;
ledger `outputs/exp030_obstacle_present/`.
