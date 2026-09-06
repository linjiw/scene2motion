# Execution-aware stepping compiler v2 — development plan after EXP-031

**Date:** 2026-09-04

**Status:** development plan, not preregistered and not a method result

**Inputs:** completed EXP-022 obstacle-absent archives, completed EXP-031 pilot, frozen evaluator v2

## Decision

Do not enlarge the EXP-031 foot-lift buffer and rerun s4408/s4434. The v1 repair failed because
the reference-space edit did not remain aligned with the obstacle after tracking, not because the
0.20 s support rule alone rejected the trajectories. The successor should be a
**controller-compatible carrier + predicted-achieved phase anchor + task-space window repair +
execution-margin verification** pipeline.

The full-paper identity remains conditional:

> Scene2Motion compiles a frozen motion prior into a scene-relative reference, then either verifies
> it under a calibrated execution margin or refuses it with a signed reason.

Until a fresh obstacle-present test succeeds, this is the proposed method direction rather than a
paper claim.

## Implementation status (2026-09-04)

`scene2motion/contact_phase_projection.py` now implements and tests four v2 primitives:

1. a two-segment quintic task-space lift with zero position, velocity and acceleration offsets
   at window entry/exit and zero velocity/acceleration at the apex;
2. a stance-lock report over absolute foot speed, carrier-pose drift and floor penetration;
3. absolute joint-speed and joint-acceleration gates; and
4. a one-sided split-conformal execution-loss margin that refuses insufficient calibration
   populations. In particular, the two 75.5/80.1 mm EXP-031 losses cannot produce a margin.

`tests/test_contact_phase_projection.py` pins those contracts, including the insufficient-`n`
refusal. This is an implemented solver building block, not completion of D1–D3: carrier-specific
achieved-phase anchoring, integration into the bounded IK projection, development calibration and
an obstacle-present result are still missing.

## Evidence that selects this design

### EXP-031, two edited development carriers

The preregistered per-frame foot-envelope IK pilot completed 0/2 repaired-present local
traversals. Both trajectories reached the local recovery position upright and without a tracker
cutoff but nominally penetrated the box. s4434 retained obstacle-absent full-route completion after
repair, so global trackability was not the first loss.

The post-hoc paired readout then showed:

* repaired-reference conservative clearance: +5.2 mm (s4408), +1.0 mm (s4434);
* repaired obstacle-absent achieved replay: −74.9 mm, −74.5 mm;
* trajectory-level minimum-clearance gap: 80.1 mm, 75.5 mm;
* intended maximum local foot lifts: 21–47 mm;
* achieved repaired-minus-raw maximum lift in the same windows: 4–18 mm.

Those two values diagnose a problem but are too few to calibrate an execution margin.

### Eight support-compatible development carriers

`experiments/analyze_execution_anchoring_development.py` reuses the eight EXP-021 references that
pass the frozen support screen and their already-completed EXP-022 obstacle-absent rollouts. It
does not select a final test set or measure traversal.

| quantity | development readout |
|---|---:|
| carrier-specific reference-root anchor minus box x | −0.218 to +0.669 m; median +0.089 m |
| signs of that anchor | 3 negative, 5 positive |
| achieved foot-at-box window centre minus v1 repair-window centre | −10 to +18 frames; median +3 |
| temporal IoU, v1 repair window versus achieved foot-at-box frames | 0.00 to 0.90; median 0.55 |
| mean achieved-minus-reference foot-bottom error while achieved foot overlaps box | −151 to +33 mm; median −46 mm |

One carrier has no v1 edit at all for a foot that later sweeps the box. For s4408 the left-foot
windows have zero overlap and the right-foot IoU is 0.03; for s4434 the left-foot centre is ten
frames early and its IoU is 0.05. The error changes sign across carriers, so neither a universal
world-x shift nor a universal frame delay is admissible.

## Proposed compiler

Let the frozen prior supply a carrier (x_0), and let an obstacle-absent development rollout of a
matched carrier supply a prediction of achieved state under the frozen controller. The compiler
has four explicit stages.

### 1. Carrier admission

Admit a neutral carrier only if it passes preregistered reference screens and an obstacle-absent
controller test:

* support-screen operating point;
* whole-route/corridor progress appropriate to the local endpoint;
* absolute joint velocity and acceleration limits calibrated from known-trackable controls;
* foot-floor penetration bound;
* no controller cutoff or fall in the development rollout.

The 0.20 s screen remains a launch-budget feature. It is neither a contact constraint nor a
physical guarantee.

### 2. Predicted-achieved scene anchoring

Estimate a monotone carrier-specific map

\[
\hat x^{\mathrm{ach}}_{b}(t)
= f_i\!\left(x^{\mathrm{ref}}_{0:t}, q^{\mathrm{ref}}_{0:t}, b\right)
\]

for the pelvis and the two foot envelopes. The first implementation need not learn a large model:
piecewise-linear interpolation of an obstacle-absent achieved rollout is the diagnostic oracle.
The repair window for foot (b) is selected where its *predicted achieved* swept envelope overlaps
the obstacle:

\[
W_{i,b}(S) =
\{t:\ [\hat x^{\mathrm{ach}}_{b,\min}(t),
          \hat x^{\mathrm{ach}}_{b,\max}(t)]
       \cap [x_o-d_o/2, x_o+d_o/2] \neq \varnothing\}.
\]

This is “phase triggering” in the defensible sense: choosing a carrier window, not switching the
generative model during rollout.

### 3. Smooth task-space repair and local IK

Within the predicted window, construct lead- and trail-foot Hermite or minimum-jerk offsets whose
position and velocity are zero at the boundary. The target height includes geometry and calibrated
execution margins:

\[
z^{\mathrm{target}}_{b}(t)
\ge h_o + m_{\mathrm{geometry}} + m_{\mathrm{execution},b,\phi}.
\]

Keep stance-foot world pose and velocity near the carrier values; do not push the stance foot
downward. Solve bounded damped least-squares IK with temporal warm starts, followed by a small
joint-window projection that penalizes first and second differences. Freeze absolute limits
against a known-trackable development distribution:

\[
|\dot q_t| \le \dot q_{\max}, \qquad
|\ddot q_t| \le \ddot q_{\max}.
\]

This staged solver is intentionally smaller than a full 12-DOF × time non-convex whole-body
trajectory optimization. Whole-body geometry is verified after IK, so shin, ankle, foot, pelvis
and upper-body collisions cannot be hidden by a foot-centre target.

### 4. Calibrated verify/refuse

On development carriers only, measure phase- and body-conditioned reference-to-achieved signed
clearance residuals. Freeze a held-out quantile or split-conformal margin before final testing.
The verified reference must clear

\[
m_{\mathrm{geometry}} + m_{\mathrm{execution}}
\]

and pass the support, floor, joint, velocity, acceleration, boundary-continuity and task-residual
checks. Otherwise the compiler refuses with the signed failing quantity. “Calibrated margin” is
the strongest allowed phrase; no certificate or guarantee is claimed.

## Minimum development sequence

### D0 — archived anchoring map (complete)

The eight-carrier map above is frozen as exploratory evidence. It rejects a shared trigger but is
not used to report a final success rate.

### D1 — neutral-carrier calibration

Generate a new development pool of neutral WALK carriers on the frozen route, screen every
reference, and track every admitted carrier without an obstacle. Predeclare the development seeds
and keep them out of the final test. The goal is to estimate achieved phase/root mapping and
absolute dynamic bounds, not traversal.

### D2 — small-signal repair-transmission probe

For a fixed subset of admitted development carriers, apply preregistered small positive/negative
task-space foot offsets at the predicted window and track them obstacle-absent. Estimate a local,
regularized response from reference edit to achieved foot-envelope edit. If the response is weak,
non-monotone, or changes sign, refuse that carrier rather than invert it.

### D3 — offline v2 candidate construction

Run the task-space spline and bounded local IK on the development carriers. Report all-source-pool
admission coverage, reason counts, whole-body clearance, support, floor, absolute dynamics,
boundary continuity and predicted execution margin. No obstacle-present rollout is tuned until
these checks are frozen.

### D4 — fresh paired test

Only after D1–D3 freeze the method, create fresh scene–reference units. The minimum paper test
compares:

1. oracle geometry/support best-of-three from the native STEP prior;
2. raw admitted carrier;
3. v1 geometry-only per-frame IK;
4. v2 predicted-achieved window repair without execution margin;
5. full v2 with calibrated execution margin.

Report robust local traversal over all assigned scene–reference units, admission coverage and
conditional completion side by side. Physics seeds are repeated measurements within a unit, not
independent scenes. The original 64-reference pool and s4408/s4434 remain development data.

## Parallel risk line

The lower-risk constructive experiment is still beam-present ducking: uncorrected, fixed extra
crouch, measured two-step correction and equal-budget best-of-three resampling. Its primary unit
is the scene, its primary endpoint is evaluator-v2 local beam crossing, and its obstacle-present
contact signal must be either genuinely instrumented or marked not assessed. This line should run
before betting the paper on the more difficult hybrid-contact stepping problem.

## Claim gate

The paper may use “feasibility-aware motion compiler” as its central method only after a fresh,
obstacle-present, all-assigned test shows a constructive gain. Before that gate, the correct
framing is:

> We measure the scene–execution gap, show why selection cannot close an empty candidate
> intersection, and report a failed reference-repair pilot that identifies the missing
> execution-aware anchoring term.
