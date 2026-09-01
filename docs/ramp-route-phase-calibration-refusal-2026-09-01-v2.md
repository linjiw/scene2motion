# Route-phase calibration v2: held-out validation refusal (2026-09-01)

The v2 campaign (`docs/ramp-route-phase-calibration-protocol-v2.md`, fresh seeds
3400–3415 / 3500–3507) completed generation and calibration cleanly and then **failed
closed at held-out validation**:

> `reference: full-frozen feasible coverage 3/7 < 5; reference: locked robust quantile
> separation failed`

Evidence: `outputs/calibrate_ramp_route_phase_v2/` (failure schema
`ramp-route-phase-calibration-failure-v2`). As preregistered, nothing was relaxed after
seeing the failure. This note records what replicated, what was falsified, and how the v3
preregistration is sized from the pooled v1+v2 evidence.

## What replicated on fresh seeds (the measurement core is stable)

- **Target gate:** Pmin froze at **0.042 m** on the fresh calibration seeds versus the
  0.041 m informative value from the v1 evidence — the own-foot observability calibration
  is reproducible.
- **Timing caps:** acceleration 0.3 m/s², jerk 2.1 m/s³, endpoint deviation 0.26 m/s
  (v1-informed replay: 0.2 / 1.0 / 0.19) from 15/16 contributing seeds.
- **Calibration attrition tolerances:** 15/16 seeds evidenced (gate ≥12) — the v1 refusal
  cause is handled.
- **Separation:** the side with enough evidence passed with margin (right: selected Q25
  0.0605 m vs floor 0.0503 m). Where separation "failed" it was a *side-count* failure
  (left: 2 selected signals < 3), not a signal-versus-background failure.
- Reference-stratum substrate attrition 1/8; all reference backgrounds below Pmin.

## What was falsified

The v2 gate assumed per-seed reference-stratum warp feasibility ≈ 0.85 (the v1-replay
point estimate, 6/7). Fresh seeds measured **3/7**; pooled across both campaigns the rate
is **9/14 ≈ 0.64**, below the 5/7 gate. The mis-sizing is the gate's, not the machinery's:
the descriptive strata show the same fresh seeds are frequently feasible at the *other*
speeds (fast: 4 seeds; slow: 2, including the seed attrited at reference). The union over
strata is **6/8 seeds feasible**, and no fresh validation seed is a washout at all three
speeds.

## Consequence for v3

The remaining error in the protocol family is demanding one global operating point. The
fix is the same move that the placement set made for event position: **the speed stratum
becomes an outcome-free program parameter**, selected per seed/side by the frozen
minimum-deformation key (preferring reference on ties), under the same frozen caps.
Gates become rate-honest and pooled:

1. substrate washout (zero cycles at *all three* speeds) ≤ 2/8 — pooled measured rate
   ≈ 6% of seeds;
2. background exceedance among non-washout seeds ≤ 1 (per-seed max deduplicated null
   across strata vs Pmin);
3. seeds with ≥1 full-frozen program in any stratum ≥ 4/8 — pooled measured rate ≈ 0.75;
4. pooled separation: ≥ 4 selected (seed, side) signals, Q25 ≥ max(Pmin, 1.25 × Q95 of
   per-seed max background) — replay on v2 evidence: 5 signals, 0.0605 vs 0.0503.

Per-stratum and per-side coverage remain reported, descriptively. This is the third
protocol in the family; all three, including both refusals, are reported together — the
two refusals measured real substrate properties (seed-level gait attrition; ~60%
single-stratum warp feasibility) that size the E1 pilot's nominal-pool budget.
