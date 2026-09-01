# RAMP route-phase calibration protocol v3

**Status: preregistered; no v3 campaign result is reported here.** Third protocol in the
family, superseding v2 after its held-out validation refusal
(`docs/ramp-route-phase-calibration-refusal-2026-09-01-v2.md`). Both prior campaigns are
preserved as hash-anchored evidence and all three are reported together; the two refusals
measured real substrate properties, not tuning noise. Every change below is sized on the
pooled v1+v2 evidence and frozen **before any v3 sample is generated**. v3 runs on fresh
seeds; no prior seed or outcome is reused as evidence.

## Changes from v2, with their evidential basis

| # | Change | Basis in pooled v1+v2 evidence |
|---|---|---|
| 1 | Fresh seeds: calibration 3600–3615, validation 3700–3707 | v2 outcomes were observed |
| 2 | The speed stratum becomes an outcome-free program parameter: candidates pool all three strata per (seed, side); the frozen key prefers reference on cost ties | single-stratum warp feasibility measured 9/14 ≈ 0.64 (v2 gated 5/7); strata union feasible for 6/8 fresh seeds, incl. the seed attrited at reference |
| 3 | Washout redefined as zero cycles at **all three** speeds; gate ≤ 2/8 | pooled all-strata washout ≈ 6% of seeds; 0/16 among fresh+v1 validation seeds |
| 4 | Program-coverage gate becomes ≥ 4/8 seeds with a full-frozen program in some stratum | pooled any-stratum rate ≈ 0.75; gate matches the minimum pool the E1 pilot needs |
| 5 | Separation pools (seed, side) signals across strata and sides: ≥ 4 signals, Q25 ≥ max(Pmin, 1.25 × Q95 of per-seed max background across strata) | v2's only separation "failure" was a side with 2 signals; the evidenced side passed 0.0605 vs 0.0503; pooled replay on v2 evidence: 5 signals, passes |
| 6 | Per-stratum and per-side coverage are descriptive only | twice-demonstrated rate/gate mismatch at stratum level |

Unchanged from v2: the frozen generation design (72 samples, nine batches of eight, three
paired speeds), the immutable support-threshold dependency, Pmin construction with
calibration attrition tolerance (≥12/16 evidenced seeds), the five-point placement set
{3.0, 3.3, 3.6, 3.9, 4.2} m, all-strata timing-cap calibration (≥10/16 contributing
seeds, per-stratum selection), quantile/headroom/rounding machinery, the C1 route
program, the [0.6, 1.2] m/s envelope, identity binding, and fail-closed refusal
semantics. Replication note: Pmin froze at 0.041 m (v1 informative) and 0.042 m (v2) —
v3's fresh value is expected in this range and a large departure would itself be
informative.

## Selection

Candidates are every (complete cycle × placement) pair with prominence ≥ P_min and a
valid packet window, under the timing bounds in force. The frozen lexicographic key:

1. normalized progress deformation;
2. RMS route-progress speed deviation;
3. maximum absolute discrete route-progress acceleration;
4. |placement − 3.6 m|;
5. placement;
6. stratum preference (reference, then slow, then fast);
7. phase-evidence receipt digest.

Timing-cap calibration selects per (seed, speed, side) as in v2. Validation selects per
(seed, side) **pooling the strata**. The selected (stratum, placement, program) triple is
bound per seed into the later pilot: every packet arm of one seed uses the same triple,
fixed before any arm outcome exists. No outcome may enter selection.

## Held-out validation kill rules (all simultaneous, rejection-only)

Over the eight validation seeds, with frozen P_min and caps:

1. **Substrate washout:** at most 2/8 seeds with zero measured complete cycles at all
   three speeds.
2. **Background:** among non-washout seeds, at most 1 with its per-seed maximum
   deduplicated null (across strata and sides) missing or strictly above P_min.
3. **Program coverage:** at least 4 seeds with a full-frozen-feasible program in some
   stratum.
4. **Pooled separation:** at least 4 selected (seed, side) signals, and nearest-rank Q25
   of their prominences ≥ max(P_min, 1.25 × Q95 of the per-seed maximum background).

Validation cannot refit, widen, floor, or reinterpret any frozen quantity. Per-stratum
attrition, per-side signal counts, placement and stratum usage are reported
descriptively.

## Interpretation

Passing v3 authorizes the route-warped packet pilot under the frozen identities, with
each nominal seed's stratum/placement/program selected by the frozen key, and with the
measured washout and feasibility rates sizing the pilot's nominal-pool budget K. It is
not evidence that any packet representation improves traversal. A third refusal would be
strong evidence that the neutral-WALK substrate is less reliable than pooled estimates
suggest, and would redirect design toward substrate repair rather than another gate
resize. Any invariant or kill-rule failure ends the campaign as a hash-anchored refusal
with no post-hoc relaxation.
