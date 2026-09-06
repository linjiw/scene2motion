# EXP-031 result — the reference repair survives open-space tracking but does not close traversal

**Date:** 2026-09-04

**Protocol:** `docs/ramp-exp031-constructive-step-repair-protocol.md`, preregistered before launch

**Evaluator:** `traversal_eval` v2, commit `a651b16`

**Execution artifact:** `outputs/exp031_constructive_step_repair_execution/`

**Status:** complete; the preregistered constructive endpoint did not hold

## Result in one sentence

The frozen per-frame foot-envelope IK repair produced **0/2 repaired-present local traversal
completions** (**0/64 over the assigned source pool**): both achieved trajectories reached the
local recovery location upright and without a tracker cutoff, but both entered the nominal box
geometry.  Physical contact was not instrumented and is therefore **not assessed**, not false.

This is a failed two-carrier engineering pilot, not a method-rate estimate.  It does not license a
claim that repair enables traversal.

## Frozen endpoint

The primary endpoint required the root to remain at least 0.50 m beyond the box far edge for
0.20 s, upright and inside the corridor, within 4.0 s, while remaining outside the nominal box
and the conservative 4 cm body-margin envelope.  The 7.2 m route endpoint was secondary.

| arm | s4408 | s4434 | primary interpretation |
|---|---|---|---|
| raw, obstacle absent | stalled at 4.658 m | completed 7.2 m route | raw tracking control |
| repaired, obstacle absent | stalled at 4.446 m | completed 7.2 m route | the edit did not destroy the only raw route completer |
| raw, 5 cm box present | nominal penetration; local kinematics at 3.52 s | nominal penetration; local kinematics at 1.98 s | paired unmodified baseline |
| repaired, 5 cm box present | nominal penetration; local kinematics at 3.98 s | nominal penetration; local kinematics at 1.98 s | **0/2 local traversal completions** |

No arm fell and no arm triggered the tracker cutoff.  The repaired-present arm lost 0.087 m and
0.012 m of maximum forward progress relative to repaired-absent for s4408 and s4434 respectively
(median −0.050 m; descriptive, `n=2`).  P2 therefore held for the one eligible carrier, but P1
did not.

The exact completion denominators are both retained:

* admitted-candidate conditional endpoint: **0/2**, Wilson 95% **0–0.658**;
* all-assigned source-pool endpoint: **0/64**, Wilson 95% **0–0.057**, with 62 references refused
  or rejected before execution under the frozen gates.

## What the v2 evaluator separates

For both present arms, the robot passed the box and reached the kinematic recovery location.  The
failed endpoint was clearance, not progress, fall, cutoff, corridor departure, or the 4.0 s
deadline.  The three contact-like quantities remain separate:

| carrier | raw nominal penetration | repaired nominal penetration | raw 4 cm-envelope penetration | repaired 4 cm-envelope penetration |
|---|---:|---:|---:|---:|
| s4408 | 1.233 mm | 2.506 mm | 40.104 mm | 40.496 mm |
| s4434 | 0.225 mm | 0.112 mm | 41.045 mm | 40.112 mm |

The repair slightly improved s4434's achieved geometric depth and worsened s4408's.  Neither
change crossed the endpoint.  The simulator box was present and acted on the rollout, but the
tracker configuration did not export a valid table-to-robot contact report.  Consequently:

1. **nominal geometry penetration** is measured;
2. **4 cm conservative margin violation** is measured;
3. **PhysX contact/impulse** is not measured.

The result must not replace item 3 with either item 1 or item 2.

## Post-hoc mechanism diagnosis

`experiments/analyze_exp031_execution_gap.py` reproduces the following development-set diagnosis
from the frozen achieved-state archives.  Its output is
`outputs/exp031_constructive_step_repair_execution/execution_gap_diagnostic.json`.  This analysis
was written after seeing P1 fail.  It does not change the endpoint and is not a calibrated
execution margin.

### 1. The box materially changed the achieved trajectory

When the repaired **obstacle-absent** achieved trajectories are replayed against the same 5 cm
box, they penetrate the nominal primitive by 34.919 mm (s4408) and 34.760 mm (s4434).  With the
box in the physics scene, the archived achieved trajectories enter it by only 2.506 mm and
0.112 mm.  This boundary-piling is evidence that the obstacle altered the closed-loop motion; it
is not a substitute for a contact sensor.

### 2. Reference clearance did not survive tracking

The admitted repaired references had conservative minimum clearances of +5.193 mm (s4408) and
+1.015 mm (s4434).  Their obstacle-absent achieved trajectories, replayed against the box, had
minimum conservative clearances of −74.919 mm and −74.533 mm.  The trajectory-level gap from
the repaired reference minimum to the achieved replay minimum is therefore 80.113 mm and
75.547 mm.  These are two diagnostic values from a development pool, not a margin to freeze for
a fresh experiment.

### 3. The failure is spatial and phase-dependent, not only a missing vertical buffer

Using the achieved archive's declared alignment
(`achieved_qpos[1::2][k]` ↔ 25 Hz reference frame `k+1`), during the frames where the repaired
reference foot overlaps the obstacle in x:

* s4408's achieved left/right foot is on average 0.461/0.396 m **behind** the reference and
  91/45 mm lower;
* s4434's achieved left/right foot is on average 0.060/0.152 m **ahead** of the reference and
  153/99 mm lower.

The two carriers err in opposite longitudinal directions.  A global fixed phase offset or a
single extra-height constant cannot explain both.

### 4. SONIC transmits only part of the local reference edit

In obstacle-absent paired rollouts, the largest intended reference lift versus the raw motion was
37/47 mm for the left/right foot of s4408 and 43/21 mm for s4434.  The largest achieved repaired
minus raw lift over those same aligned windows was only 9/18 mm and 11/4 mm respectively.  The
window-mean achieved lift was 0.7/8.3 mm for s4408 and 0.7/−3.1 mm for s4434.  The unchanged
reference root also led to a changed achieved root path, so reference-space surgery cannot be
treated as an independently transmitted foot command.

## Decision and next method version

The frozen two-carrier protocol is closed.  Its thresholds will not be loosened and the same two
rollouts will not be relabelled or rerun as a new primary result.  A successor receives a new
protocol, method version, output directory, and fresh test units.

The failure ordering rules out the simplest next stories:

* **not primarily support-screen rejection:** both repaired-present runs avoid the cutoff and
  fall;
* **not simply repair-induced loss of global trackability:** s4434 retains full-route completion
  without the box;
* **not fixed by a claimed 15 mm buffer:** the diagnostic clearance loss is 76–80 mm and includes
  carrier-specific longitudinal/phase error and attenuated foot lift;
* **not evidence for a monolithic trajectory QP yet:** the present data identify the error term
  that the next, simpler task-space/windowed solver must model first.

The next stepping operator should therefore be a new
**execution-aware, scene-relative contact-window compiler**:

1. choose an obstacle-absent controller-compatible carrier;
2. estimate a carrier-specific mapping from reference phase/root position to achieved phase/root
   position on a development set;
3. anchor lead- and trail-foot task-space splines to the predicted achieved obstacle location;
4. solve bounded local IK with endpoint continuity and absolute joint velocity/acceleration
   limits;
5. require a held-out, phase-conditioned execution-clearance quantile rather than the two
   diagnostic values above; and
6. verify or refuse before an obstacle-present launch.

The faster statistical risk line remains beam-present ducking with uncorrected, fixed-crouch,
measured-correction, and equal-budget resampling arms.  That experiment, not a second look at
s4408/s4434, is the shortest path to a multi-scene constructive result.

## Reproduction

```bash
source env.sh
env LD_LIBRARY_PATH= "$S2M_PY" experiments/analyze_exp031_execution_gap.py --check
env LD_LIBRARY_PATH= "$S2M_PY" "$S2M_PY" -m pytest -q \
  tests/test_exp031_execution_gap.py \
  tests/test_exp031_constructive_step_repair.py \
  tests/test_exp031_prepare_step_repair.py \
  tests/test_traversal_eval_measurement.py
```

The completed campaign validator passes independently, and the four achieved-state archive hashes
are bound in the campaign receipt.  The post-hoc diagnostic artifact has SHA-256
`9eb921dc62f0cc89caaa94993ec24e9ce5c9bae860b4c685f7c10e90bcf05d4c`.
