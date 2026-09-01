# EXP-018 protocol: route-warped placement persistence and paired packet arms

**Status: preregistered; no exp018 result is reported here.** Successor to the closed
exp017 fixed-frame pool (`docs/ramp-e1-protocol.md`, `REPORT.md` §31) on the calibrated
route-phase foundation (`REPORT.md` §32). This document is frozen before any exp018 sample
is generated. It stages two separable questions so that neither can silently answer the
other:

* **Stage A — placement persistence.** When a nominal WALK seed is re-generated under its
  calibrated, outcome-free route-progress warp, does a complete same-side swing still
  occur at the fixed obstacle — i.e., does the ±8-frame placement gate that exhausted the
  fixed-frame pool (1/8) now pass by design?
* **Stage B — representation.** Among persisting seeds only, does the residual packet
  improve the placed step-over endpoint vector over the same-support absolute packet?

A Stage-A failure is a publishable negative result about route-timing reparameterization
of a frozen prior; it is measured before any packet arm exists and cannot be confounded
with packet effects.

## Frozen dependencies (hash-locked; a mismatch aborts before generation)

| Dependency | Identity |
|---|---|
| Physical support thresholds | `outputs/exp016_threshold_calibration/receipt.json`, file SHA-256 `f6dba8be…e9bf` (calibration loader, unchanged) |
| Route-phase calibration | `outputs/calibrate_ramp_route_phase_v3/receipt.json`, file SHA-256 `745c8ad3c7784c686ba03434a84c980b5f1a6be65b5b48fc16fa6973e31c2b58`; supplies frozen Pmin = 0.042 m, timing caps, placement set {3.0, 3.3, 3.6, 3.9, 4.2} m, and the selection-key semantics |
| Donor bundle | exp017's archived selection: donor bank seeds 2600–2603, selected seed 2603, left swing, `packet_pair.json` file SHA-256 `3bdaace6d9d1161cb579fb469fa14e733808cc78bc4033f254c17e836d7da773`, pair hash `5f056ec098213d7885aad0af307811fb88ad073b2550ad62bd5b0d297a675303` |

The donor bank is **regenerated deterministically** with exp017's exact settings (prompts,
seeds, 200 frames at 25 Hz, the 7.2 m straight route — byte-identical to the calibration
reference route — five DDIM steps, CFG (2.0, 2.0), noise-stream v2) and the freshly
extracted `CoherentPacketPair` must reproduce the archived pair hash, packet hashes, and
selected seed exactly. Any deviation is a fail-closed determinism refusal, not a retry.
Donor selection machinery, gates (0.04 m adapted lift, 0.75 progress), and the
selection key are exp017's, unchanged.

## Design and exact budget

* Nominal pool: **K = 16** fresh seeds 3800–3815, disjoint from every prior campaign,
  each generated at the three calibrated speeds (paired seed across strata; six batch
  invocations of eight) — 48 samples.
* Pool selection (outcome-free, frozen): the v3 machinery verbatim — Pmin, packet-window,
  placement enumeration under the frozen caps, strata pooled per (seed, side), the frozen
  minimum-deformation key with reference-stratum tie preference — restricted to the
  packet's swing side (**left**). A seed is eligible iff it has a left-side full-frozen
  program. **N = 6** evaluation seeds are the first six eligible in predeclared seed
  order. Fewer than six eligible ⇒ fail-closed pool-exhaustion stop (rates from §32
  predict ≈ 0.66–0.75 eligibility per seed).
* Stage A: for each selected seed, generate one WALK clip with the same seed under the
  **warped route** `root_xz[t] = (0, s(t))` from its selected program — N samples. A seed
  **persists** iff (1) realized progress ratio vs the warped route ≥ 0.75; (2) it contains
  a complete left-swing cycle under the locked physical thresholds with
  `min_relative_lift_m = 0.0` — target discovery deliberately does *not* inherit the
  0.04 m donor-quality gate, per the calibration's donor/target separation; and (3)
  exp017's unchanged target assignment succeeds at the seed's single placement within the
  unchanged **±8-frame** center-shift bound and packet window. The **persistence rate
  (per-seed, out of N)** is the primary Stage-A endpoint; the required center shift,
  realized apex progress error, and warped-route progress ratio are its supporting
  measurements. Zero persisting seeds ⇒ stop before arms.
* Stage B: for each persisting seed, exactly one absolute and one residual arm (STEP
  prompt for both, exp017's locks: identical support hashes, channel usage
  {root_2d, root_y_pos, global_joints_rots}, no position channel, free body heading,
  strength 1, duration scale 1, shared target phase receipt), generated under the same
  warped route — 2·M samples, M = persisting count.

A completed campaign spends exactly `2D + 3K + N + 2M = 8 + 48 + 6 + 2M ≤ 74`
frozen-prior samples with exact accounting; every planned arm stays in its denominator.

## Endpoints and analysis

Stage-B endpoints are exp017's E1a vector, unchanged: obstacle-centred whole-body
box-height lower bound and collision-free flag against the simulation collision model
with the measured body margin (never foot peak), crossing position/frame error, swing
side vs packet side, stance/contact measures, progress ratio (vs the warped route), path
error, integrated program deformation, and generator calls / wall-clock. Analysis is
descriptive and paired: per-seed residual-minus-absolute differences and their raw
counts; **no confidence intervals, p-values, or population claims** — one donor bundle,
one box geometry, per-seed placements are strata, seeds are the only replication unit.
Placement, stratum, center shift, and warp deformation are reported per seed.

Interpretation boundaries: "SONIC execution success" is out of scope (kinematic stage
only, per exp017 §Endpoints); Stage-B conclusions are conditional on the donor bundle and
persisting subset; Stage A cannot be rescued by moving obstacles, widening the shift
bound, extending K, replacing seeds, or re-selecting after any outcome. The decision rule
from exp017 stands: a residual advantage on the joint endpoint vector (or a valid
kinematic step where absolute has none) licenses the local response optimizer next; a
clean negative across both arms with persisting placement redirects to output-space
residual adaptation. A Stage-A persistence failure redirects design toward measuring
events on warped substrates directly instead of transporting linear-route events.

Provenance: clean worktree, full identity binding (repository, checkpoint, text cache,
runtime) with post-generation revalidation, atomic incremental evidence, hash-anchored
refusals. Harness-defect aborts (not preregistered gates) may be fixed and rerun into a
fresh output directory with every attempt preserved, per exp017 practice; gate outcomes
are never rerun.
