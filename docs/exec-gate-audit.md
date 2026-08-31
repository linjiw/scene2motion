# Execution-aware acceptance-gate audit — 2026-08-30

## Verdict

No depth-dependent execution-clearance threshold can be fit legitimately from the artifacts
currently on disk. There are **zero** `achieved_qpos*.npz` archives. Existing SONIC runs pair a
generated reference with per-motion scalar tracking summaries, termination, and progress, but do
not retain the achieved trajectory needed to recompute signed scene clearance.

The new `scene2motion.sonic_state_export` callback supplies the missing achieved state, but it has
not yet been run. Its archive stores environment-local MuJoCo-compatible qpos, valid length,
termination, progress, motion key/id, joint order, quaternion convention, and sampling period.

## What exists

- EXP-011/012/014/014-small/015 contain 19 per-arm `metrics_eval.json` files, 8 motions each:
  **152 reference-to-scalar tracker records**. Corresponding motion-library pickles retain the
  generated root pose and 29 DOFs.
- The scalar bundles include per-motion `mpjpe_l`, `mpjpe_g`, body-subset variants,
  `terminated`, `progress`, and `motion_keys`.
- Phase 4D has 180 selected rows and 171 unique selected qpos cache entries. Every one is cache
  version 1. The complete clip cache contains 582 entries, all version 1.
- There are no generated-vs-achieved qpos pairs in any output directory.

All of the above reference clips are long `seeds=` generations from legacy noise-stream v1, so
they can be used to smoke-test state export but not for a confirmatory gate fit.

## Why the published 85.6 mm proxy is not an execution certificate

1. SONIC's `vr_3points` subset is explicitly `torso_link`, `left_wrist_yaw_link`, and
   `right_wrist_yaw_link`; it is not head plus hands. The reported 45.6 mm duck value is the
   time-and-joint mean over those three bodies.
2. `mpjpe_l` subtracts the root position independently at every frame. It omits global root
   translation, including root-height loss that directly changes overhead clearance.
3. MPJPE is an unsigned Euclidean mean. Clearance needs the worst signed error along the active
   contact normal at the obstacle interaction.
4. Current `G1Body` clearance already expands every scene box by `BODY_MARGIN = 0.04 m` before
   MuJoCo computes distance. Adding `BODY_MARGIN` again to a threshold on
   `min_clearance_m`/`min_overhead_m` double-counts it. On the legacy 5,479-clip ledger, 23.5%
   fall below 85.6 mm, but 9.7% fall below 45.6 mm. Neither percentage is an observed execution
   failure rate because the 45.6 mm quantity is still the wrong residual.
5. Existing SONIC success is tracker survival in an obstacle-free tracking environment, not
   whole-body collision-free traversal.

The exploratory EXP-011 means do show dose structure (33.5, 29.7, 45.6, 71.9 mm at requested
duck depths 0, 0.20, 0.35, 0.50 m), but the shallow arm is non-monotone and the metric has all
of the mismatches above. It cannot identify a defensible slope.

## Correct gate and calibration target

Let `c_ref^M` and `c_exec^M` be exact minimum clearance against the same scene geometry, with
the existing mesh-coverage margin `M` already applied once by `G1Body`. For each candidate or,
preferably, each obstacle interaction, measure the realized clearance loss

```text
loss = c_ref^M - c_exec^M
```

and treat termination before clearing the obstacle as an execution failure, not a censored
clearance observation. Fit a one-sided upper loss bound, initially

```text
loss_hat(d) = beta_0 + beta_1 * realized_local_dip
tau_alpha(d) = max(0, loss_hat(d) + conformal_quantile(calibration_residual))
```

with `beta_1 >= 0` only if the data support that monotonic restriction. Then report two gates:

```text
executed collision-free:  c_ref^M >= tau_alpha(d)
retains 18 cm margin:      c_ref^M >= 0.18 + tau_alpha(d)
```

Do **not** add another 0.04 m while using the current margin-inflated clearance. If a future
metric uses raw primitive-to-obstacle distance instead, add the mesh correction exactly once.

## Minimum next executable experiment

1. First run a 2-arm state-export smoke test (neutral and duck, a few existing legacy clips)
   into a fresh output directory. Validate motion-key joins, 25 Hz reference versus 50 Hz
   achieved timing, root-frame alignment, qpos forward kinematics, archive lengths, and
   agreement of archive `terminated`/`progress` with `metrics_eval.json`. This validates the
   instrument only; do not fit from it.
2. After the Phase 4 proposer-by-feedback matrix is regenerated under noise-stream/cache v2,
   export every **unique selected candidate** (and ideally each repair attempt) to SONIC. Run
   explicit simulator seeds and save achieved qpos with the callback.
3. Replay both reference and achieved qpos through `G1Body(the_exact_scene)`. Record exact
   whole-body and overhead clearance, culprit robot/scene geom, goal/progress, termination,
   floor penetration, and whether the achieved root passed every obstacle.
4. Split by scene family/scene, not by duplicated candidate or physics rollout. Fit the loss
   bound on calibration scenes, freeze it, and report false-safe risk versus coverage on held-out
   scenes. Re-score collision-free and 18 cm margin as separate outcomes.

The state archive should be accompanied by a row ledger containing at least:

- `scene_id`, serialized scene/hash, route hash, obstacle id and interaction interval;
- proposer, feedback mode, attempt, selected flag, clip key, schedule hash;
- ARDY model/hash, noise-stream/cache version, ARDY seed;
- SONIC checkpoint/hash, resolved config hash, simulator seed;
- requested peak dip, realized local/peak dip, dip rate and duration;
- reference and achieved archive keys, FPS/sample period, valid lengths;
- reference/achieved whole-body and overhead minimum clearance, culprit geoms;
- `clearance_loss_m`, terminated, progress/goal, collision, and `false_safe`.

Because several arms can select the same qpos, tracking should be deduplicated by content hash;
statistical splitting and uncertainty should use independent scenes/motions, not duplicated
method rows.
