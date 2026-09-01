# RAMP route-phase calibration protocol

**Status: preregistered; no campaign result is reported here.** This document locks the
fresh neutral-WALK calibration needed before the fixed-obstacle, route-warped RAMP pilot.
The calibration estimates a target swing-observability threshold and route-timing bounds.
It does not test absolute versus residual packets, traversal success, collision clearance,
SONIC execution, or response repair.

## Question and dependency boundary

The campaign asks two narrow questions before any packet-arm outcome is observed:

1. When is a complete neutral-gait swing sufficiently observable from its own physical-foot
   clearance trace to serve as a target phase event?
2. Which deterministic route-progress warps keep that event at a fixed spatial obstacle while
   remaining inside empirically calibrated timing bounds?

The existing physical support-threshold receipt at
`outputs/exp016_threshold_calibration/receipt.json` is an immutable common-support
**dependency**. Its hash and normalized threshold identity must be bound into the new receipt.
It supplies support height, support speed, support dwell, stance-support, and floor-penetration
settings used to identify physical cycles. Reusing it is not new confirmatory evidence for
those thresholds, and validation in this campaign may not revise them.

The adapted-donor quality gate, `min_relative_lift_m = 0.04`, remains a separately identified
future packet-source requirement. It is not used to discover neutral swings, calibrate target
prominence, or accept validation targets.

## Frozen generation design

Every generated clip uses:

* prompt `A person walks forward.`;
* frozen ARDY-G1 and its recorded checkpoint identity;
* a straight root-XZ-only route, with no root-height or heading constraint and
  `first_heading = 0`;
* 200 frames at 25 Hz, so duration is `(200 - 1) / 25 = 7.96 s`;
* five DDIM steps, CFG weight `[2.0, 2.0]`, and noise-stream version 2.

The three requested route-progress speeds are:

| Label | Speed | Route endpoint after 7.96 s |
|---|---:|---:|
| slow | `0.6 m/s` | `4.776 m` |
| reference | `7.2 / 7.96 = 0.904522613... m/s` | `7.2 m` |
| fast | `1.2 m/s` | `9.552 m` |

Calibration seeds are 3200--3215 and validation seeds are 3300--3307. The splits are
disjoint. The same seed is paired across all three speeds; speed is not assigned to different
seed groups.

The complete campaign contains exactly **72 generated samples**, not 72 generator calls:

* 16 calibration seeds x 3 speeds = 48 samples;
* 8 validation seeds x 3 speeds = 24 samples.

Generation occurs in exactly **nine batch invocations of eight samples each**. In order, the
invocations are seeds 3200--3207 at slow/reference/fast, seeds 3208--3215 at
slow/reference/fast, and seeds 3300--3307 at slow/reference/fast. Thus a completed campaign
has 72 returned samples and nine Python `generate(...)` invocations. A failed invocation is
not replaced or retried under another seed: the failure receipt records launched and returned
samples separately and the campaign stops.

## Own-foot phase evidence

Cycle discovery uses exact qpos-derived foot kinematics and the bound physical support
thresholds, but disables the donor/source minimum-lift gate. A complete swing has physical
takeoff, apex, and landing plus the required contralateral stance support. Four stable samples
immediately before takeoff and four beginning at landing form its baseline windows. The
packet-support window is the apex plus or minus two frames and must remain inside the complete
cycle.

For swing-foot bottom clearance above the world floor, `c_swing(t)`, target prominence is

$$
P = c_{\mathrm{swing}}(t_{\mathrm{apex}})
    - \max\left(
        \operatorname{median}(c_{\mathrm{pre}}),
        \operatorname{median}(c_{\mathrm{post}})
      \right).
$$

The signal and both baselines use the **same swing foot**. Contralateral-foot motion cannot
increase the prominence. Each complete cycle produces exactly two preregistered null
contrasts:

$$
B_{\mathrm{pre}} = \max(c_{\mathrm{pre}})-\operatorname{median}(c_{\mathrm{pre}}),
\qquad
B_{\mathrm{post}} = \max(c_{\mathrm{post}})-\operatorname{median}(c_{\mathrm{post}}).
$$

Nulls are deduplicated within each generated clip by the physical identity
`(baseline support foot, window start frame, window end frame, measurement method)`. A cycle
or prominence-receipt digest is deliberately not part of that identity: two cycle records
that reuse one physical support window contribute only one null. Conflicting values for one
physical identity invalidate the campaign.

## Calibration quantities

### Target prominence gate

For each of the 16 calibration seeds, take the maximum deduplicated null contrast over every
measured cycle and all three speeds. Every calibration seed must supply background evidence.
Let `Q95_NR` denote the nearest-rank empirical 95th percentile, using one-indexed rank
`ceil(0.95 n)`. The frozen target gate is

$$
P_{\min}
=
\operatorname{ceil}_{0.001\,\mathrm m}
\left[
1.25\,Q95_{\mathrm{NR}}
\left(
\left\{\max B_i\right\}_{i=1}^{16}
\right)
\right].
$$

Rounding is outward with decimal arithmetic. If the exact rounded value is zero, calibration
is degenerate and stops. No unregistered positive floor may be added after inspecting the
data.

### Event-aligned route programs

The calibration program uses a fixed 7.2 m straight route and a fixed obstacle progress of
3.6 m. For every eligible cycle, the physical swing-foot offset at the apex is

$$
o_f = x_{\mathrm{foot}}(t_{\mathrm{apex}})
      - x_{\mathrm{root}}(t_{\mathrm{apex}}),
$$

and the root event anchor is fixed as

$$
s_{\mathrm{event}} = 3.6\,\mathrm m - o_f.
$$

Route geometry and obstacle position remain fixed. Only scalar route progress is
reparameterized. The schedule is the deterministic, solver-free three-anchor C1 cubic Hermite
program through start, event, and end: endpoint slopes are the adjacent segment means, and the
event slope is their arithmetic mean. Candidate progress speed must remain in
`[0.6, 1.2] m/s` in both continuous and sampled checks.

When more than one eligible cycle exists for a `(seed, speed, swing side)`, selection is
outcome-free. Choose the minimum lexicographic route-deformation key:

1. normalized progress deformation;
2. RMS route-progress speed deviation;
3. maximum absolute discrete route-progress acceleration;
4. phase-evidence receipt digest as the deterministic tie-break.

No generated packet response, collision result, tracker result, or traversal outcome may enter
cycle selection.

### Route-timing bounds

For each calibration seed, take maxima over its selected speed/side programs. The three
per-seed quantities are:

* acceleration: the maximum of absolute analytic-continuous and 25 Hz discrete scalar
  route-progress acceleration;
* jerk: maximum absolute 25 Hz discrete scalar route-progress jerk, defined as the first
  difference of discrete acceleration times FPS, equivalently the second difference of speed
  times FPS squared;
* endpoint deviation: maximum absolute endpoint route-progress speed deviation from
  `7.2 / 7.96 m/s`.

Each frozen cap is `1.25 x nearest-rank Q95` over the 16 per-seed maxima, rounded outward.
Acceleration is rounded to `0.01 m/s^2`, jerk to `0.1 m/s^3`, and endpoint-speed deviation to
`0.01 m/s`. Acceleration and jerk must retain a positive one-quantum floor if their rounded
calibration value is zero; endpoint deviation receives no added floor. These caps and
`P_min` are frozen before validation is evaluated.

## Held-out validation and kill rules

Validation is rejection-only: it cannot refit, widen, floor, smooth, or otherwise modify the
calibration quantities. It cannot replace seeds, add samples, select a different cycle using
an outcome, or reinterpret the support dependency. The following are simultaneous kill rules.

At **each** of slow, reference, and fast speed:

* at most 1/8 validation seeds may have missing background evidence or a per-seed maximum
  deduplicated null strictly above frozen `P_min`;
* at least 6/8 seeds must contain a complete, packet-window-valid cycle with prominence at
  least `P_min`;
* for each swing side separately, at least 4/8 seeds must contain such a complete cycle;
* for each side, the nearest-rank Q25 of the outcome-free selected per-seed prominence must be
  at least

$$
\max\left(
P_{\min},
1.25\,Q95_{\mathrm{NR}}(B_{\mathrm{same\ side}})
\right),
$$

  where the background term uses the same-side, per-seed maximum deduplicated physical-window
  nulls. Unselected high-prominence cycles cannot satisfy this separation rule.

The dynamics coverage rules are deliberately split by speed:

* at the **reference** speed, at least 7/8 seeds must admit an outcome-free selected program
  under the frozen route-progress speed, acceleration, and jerk bounds;
* at **each endpoint speed** (`0.6` and `1.2 m/s`), at least 6/8 seeds must admit a selected
  program under the **full** frozen bounds, including endpoint-speed deviation.

All counts use the eight predeclared validation seeds as their denominator. Missing evidence,
measurement errors, or rejected route programs remain failures in that denominator.

## Provenance, termination, and interpretation

The harness must start from a clean Git revision, bind the repository and checkpoint
identities, bind the normalized old support-threshold dependency, archive sample/qpos hashes
and incremental evidence, and preserve the exact generation accounting. Calibration and
validation outputs are written before a terminal decision wherever possible.

Passing this protocol only authorizes the later route-warped packet pilot under the frozen
identities and bounds. It is not evidence that a packet representation improves traversal.
If any generation invariant, calibration requirement, or validation kill rule fails, the run
ends as a hash-anchored **refusal**. That refusal is preserved as research evidence; thresholds,
coverage rules, seeds, or speed strata are not relaxed after seeing the failure.
