# Route-phase calibration v1: hash-anchored refusal (2026-09-01)

The preregistered campaign in `docs/ramp-route-phase-calibration-protocol.md` ran to
completion on generation — 9/9 batch invocations, 72/72 samples returned, converted, and
analyzed, exact accounting — and then **failed closed at the calibration stage**:

> `target-prominence calibration lacks background evidence for seeds [3206, 3209]`

Evidence is preserved at `outputs/calibrate_ramp_route_phase/` (failure schema
`ramp-route-phase-calibration-failure-v1`, stage `calibration`). Per the protocol, no
threshold, coverage rule, seed, or speed stratum is relaxed for that campaign. This note
records the diagnosis that sizes the v2 preregistration
(`docs/ramp-route-phase-calibration-protocol-v2.md`). Reproduce every number with
`experiments/analyze_calibrate_ramp_route_phase_v1.py`.

## Diagnosis: three distinct failure sources

**1. Neutral-substrate attrition (the blocking failure).** 17/72 clips (24%) contain zero
physically-complete unilateral swing cycles under the frozen support-threshold discovery
(bilateral flight, boundary-truncated swings, insufficient phase samples). Because the
design pairs one seed across all three speeds, the paired structure exposes that this is a
*seed-level gait property*: calibration seeds 3206 and 3209 are washouts at **all three
speeds**, so they supply no background nulls, and v1's requirement that *every* calibration
seed supply background evidence is unsatisfiable. Per-speed zero-cycle rates are 7/24
(slow), 5/24 (reference), 5/24 (fast).

This is a real finding, not merely a protocol defect: **roughly one in eight ARDY WALK
seeds produces a gait with no support-consistent complete swing at any tested speed, and
roughly one in five clips fails at a given speed.** Any event-aligned motion program built
on a nominal substrate must budget for this attrition; it is the same phenomenon that
exhausted the exp017 fixed-event nominal pool (1 eligible of K=8).

**2. Timing-envelope washout at a single fixed event progress.** With the obstacle fixed at
3.6 m of a 7.2 m route and the scalar route-speed envelope frozen at [0.6, 1.2] m/s, both
segment-mean-speed constraints bind, so only apexes in a ~2 s window of the 7.96 s clip can
be warped to the event. Of 326 prominence-passing packet-valid cycles, only 83 (25%) admit
a feasible warp to the fixed 3.6 m event. A five-point placement set
{3.0, 3.3, 3.9, 3.6, 4.2} m recovers 131 (40%), and — because a seed needs only one
warpable cycle — recovers most per-seed coverage.

**3. Endpoint-stratum brittleness.** Replaying the downstream v1 kill rules with an
attrition-tolerant gate shows validation would still have failed at the endpoint speeds
(slow: background 2/8 missing, endpoint coverage 4/8; fast: completeness 5/8, endpoint
coverage 4/8) while every reference-stratum rule and **every separation rule passed**
(selected-signal Q25 0.043–0.058 m against a 0.041 m floor). The endpoint coverage rules
were sized above the measured per-clip washout rate; they gate speed-generalization that
the E1 pilot does not consume.

## What survives intact

- Phase observability is real and well separated from its own-foot background at every
  speed and on both sides — the core measurement premise holds.
- The informative target gate over the 14 evidenced seeds is Pmin = 0.041 m
  (nearest-rank Q95 0.0327 m × 1.25, ceil 1 mm), versus a 0.055 m median valid-cycle
  prominence.
- The frozen physical support-threshold dependency, the batch/accounting machinery, and
  the identity binding all worked exactly as designed.

## Consequences for v2 (preregistered before any v2 sample is generated)

1. Fresh disjoint seeds (calibration 3400–3415, validation 3500–3507); v1 seeds are
   never reused.
2. Attrition-tolerant calibration: ≥12/16 seeds with background evidence; timing caps over
   all-strata selected programs with ≥10/16 contributing seeds (v1 evidence: 14/16 and
   12/16 respectively).
3. Event placement becomes a frozen five-point set with outcome-free minimum-deformation
   selection per seed/speed/side.
4. Validation kill rules gate only the reference stratum the E1 pilot consumes; the
   endpoint strata are generated, measured, and reported descriptively.
5. Substrate attrition becomes a first-class reported quantity with its own gate
   (≤3/8 validation reference clips attrited), feeding the E1 nominal-pool budget.
