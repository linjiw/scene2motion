# RAMP route-phase calibration protocol v2

**Status: preregistered; no v2 campaign result is reported here.** This protocol
supersedes `docs/ramp-route-phase-calibration-protocol.md` after that campaign's
hash-anchored refusal (`docs/ramp-route-phase-calibration-refusal-2026-09-01.md`).
Every design change below is sized on the v1 refusal evidence
(`experiments/analyze_calibrate_ramp_route_phase_v1.py`) and is frozen here **before any
v2 sample is generated**. v2 runs on fresh seeds; the v1 seeds and outcomes are never
reused as evidence.

## Changes from v1, with their evidential basis

| # | Change | Basis in v1 refusal evidence |
|---|---|---|
| 1 | Fresh seeds: calibration 3400–3415, validation 3500–3507 | v1 outcomes were observed; reuse would be post-hoc selection |
| 2 | Calibration tolerates missing seeds: ≥12/16 with background evidence; ≥10/16 contributing timing maxima | measured seed-level washout 2/16 (all-speed); per-clip zero-cycle rate 24% |
| 3 | Frozen five-point event placement set {3.0, 3.3, 3.6, 3.9, 4.2} m, outcome-free minimum-deformation selection | fixed 3.6 m event admits a warp for only 25% of valid cycles; the set recovers 40% and near-full per-seed coverage |
| 4 | Timing caps calibrated over selected programs from **all three speed strata** | reference-only selections contribute just 6/16 seeds; all-strata contribute 12/16 |
| 5 | Validation kill rules gate **only the reference stratum**; slow/fast are generated, measured, and reported descriptively | endpoint coverage rules were sized above the measured washout rate and gate generality the E1 pilot does not consume |
| 6 | Substrate attrition is a first-class gated quantity (≤3/8 reference validation clips attrited) | attrition, not observability, was the blocking phenomenon; E1's nominal-pool budget consumes this rate |

Unchanged from v1: the frozen generation design (prompt, checkpoint, DDIM steps, CFG,
noise-stream v2, 200 frames at 25 Hz, straight root-XZ-only routes, three paired speeds,
nine batches of eight, 72 samples total), the immutable physical support-threshold
dependency (`outputs/exp016_threshold_calibration/receipt.json`, file SHA-256
`f6dba8be…e9bf`), the own-foot prominence and deduplicated-null definitions, the
prominence/timing quantile machinery (nearest-rank Q95, ×1.25 headroom, outward decimal
rounding, same quanta), the C1 three-anchor route program, the [0.6, 1.2] m/s scalar
route-speed envelope, the separation rule form, and the fail-closed refusal semantics.
The donor-step quality gate (`min_relative_lift_m = 0.04`) remains a separate identity,
never used for target discovery or calibration.

## Frozen design

Generation is identical to v1 except seeds. The nine batch invocations, in order: seeds
3400–3407 at slow/reference/fast, seeds 3408–3415 at slow/reference/fast, seeds
3500–3507 at slow/reference/fast. Speeds: slow 0.6 m/s, reference 7.2/7.96 m/s,
fast 1.2 m/s; the same seed is paired across speeds. A failed invocation is not replaced
or retried; the failure receipt records launched and returned samples separately and the
campaign stops.

### Event placements

The calibration route remains the fixed 7.2 m straight line. The frozen placement set is

```
E = {3.0, 3.3, 3.6, 3.9, 4.2} m of route progress,
```

with 3.6 m remaining the reference placement. For a complete cycle with apex-frame
swing-foot forward offset `o_f`, each placement `e` in E defines one candidate root event
anchor `s_event = e − o_f` and one candidate three-anchor C1 route program (v1
construction, unchanged). A candidate is feasible when the program satisfies the timing
bounds in force. Selection among feasible candidates for one (seed, speed, swing side) is
outcome-free and minimizes the lexicographic key

1. normalized progress deformation;
2. RMS route-progress speed deviation;
3. maximum absolute discrete route-progress acceleration;
4. |placement − 3.6 m|;
5. placement;
6. phase-evidence receipt digest.

No generated packet response, collision result, tracker result, or traversal outcome may
enter selection. The selected placement is bound per seed into the later pilot: within a
seed, every packet arm uses the same placement, so placement is a nuisance parameter fixed
before any arm outcome exists.

## Calibration quantities

### Target prominence gate

Per-seed background maxima are computed exactly as in v1 (per-seed maximum across all
three speeds of the within-clip physical-window-deduplicated own-foot nulls). Let `S_bg`
be the calibration seeds with at least one deduplicated null. The campaign **fails closed
if |S_bg| < 12**. Otherwise

```
P_min = ceil_1mm( 1.25 × Q95_NR({max B_i : i ∈ S_bg}) ),
```

with nearest-rank one-indexed `ceil(0.95 n)` indexing and outward decimal rounding. A
zero rounded value is degenerate and stops the campaign. Seeds outside `S_bg` are recorded
as calibration attrition; no imputation is applied.

### Route-timing caps

Broad-bounds candidates (speed envelope only) are built for every complete,
packet-window-valid cycle with prominence ≥ P_min, across **all three speed strata** of
the calibration split and all five placements; one candidate is selected per
(seed, speed, side) by the key above. Each contributing seed's maxima over its selected
programs give the three per-seed quantities of v1 (max continuous/discrete acceleration;
max discrete jerk; max endpoint speed deviation). Let `S_t` be the contributing seeds. The
campaign **fails closed if |S_t| < 10**. Caps are `1.25 × Q95_NR` over `{per-seed max :
S_t}`, rounded outward (acceleration to 0.01 m/s², jerk to 0.1 m/s³, endpoint deviation to
0.01 m/s; acceleration and jerk keep a positive one-quantum floor). P_min and the caps are
frozen before validation is evaluated.

## Held-out validation

Validation is rejection-only: it cannot refit, widen, floor, smooth, or otherwise modify
the frozen quantities, replace seeds, or select any cycle using an outcome. All kill rules
below apply to the **reference stratum** of the eight validation seeds. Definitions:

* a validation seed is **attrited** at a speed when its clip has zero measured complete
  cycles; `n_ok` = number of non-attrited reference seeds;
* a non-attrited seed is **full-feasible** when at least one (cycle, placement) candidate
  satisfies P_min, the packet window, and **all** frozen timing caps;
* the **selected** program per (seed, side) is the minimum-key candidate under the frozen
  caps.

Kill rules (all simultaneous):

1. **Attrition:** at most 3/8 reference validation seeds attrited.
2. **Background:** among non-attrited seeds, at most 1 has its reference-stratum
   deduplicated-null maximum missing or strictly above P_min.
3. **Feasibility:** full-feasible seeds ≥ max(4, ceil(0.6 × n_ok)).
4. **Side evidence:** each swing side has ≥3 selected programs (over non-attrited seeds).
5. **Separation, per side:** nearest-rank Q25 of the selected per-seed prominence ≥
   max(P_min, 1.25 × Q95_NR of the same-side per-seed maximum deduplicated reference
   background over non-attrited seeds). Unselected high-prominence cycles cannot satisfy
   this rule.

The slow and fast strata are measured and reported in full (attrition, broad-bounds
coverage, placement usage, separation quantities) but are **descriptive**: they gate
nothing in v2 and are not evidence of endpoint-speed generalization.

## Provenance, termination, and interpretation

Identical to v1: clean-worktree launch, repository/checkpoint/text-cache/runtime identity
binding with revalidation after generation and after analysis, immutable support-threshold
dependency, atomic incremental evidence, exact query accounting, and hash-anchored refusal
on any invariant or kill-rule failure — with no post-hoc relaxation. Passing v2 authorizes
only the route-warped packet pilot at the reference speed under the frozen identities,
placement set, P_min, and caps, with the measured attrition rate sizing that pilot's
nominal-pool budget. It is not evidence that any packet representation improves traversal,
and it licenses no endpoint-speed claim.
