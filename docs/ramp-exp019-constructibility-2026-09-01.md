# EXP-019: the packet transports onto only ~a third of the prior's own walk swings

Two attempts of the gait-matched pilot ran on 2026-09-01
(`docs/ramp-e1-gait-matched-protocol.md`), both fail-closed, both preserved.

**Attempt 1** (`outputs/exp019_gait_matched_stepover/`, commit `2bdfee8`) reached 12/16
placement eligibility and then aborted at `program_render`. Cause: placement eligibility
was measured with the phase-observability enumerator while the packet transport consumes
`step_phase` cycles and a rendered ConstraintSpec — different gates. Seed 3904 had an
observability cycle at apex 139 with no corresponding `step_phase` cycle. Harness defect,
fixed by the constructibility probe (commit `6f6d999`).

**Attempt 2** (`outputs/exp019_gait_matched_stepover_v2/`, commit `6f6d999`) regenerated
the donor bundle byte-identically for the third time, generated all 48 pool clips, and
stopped at the preregistered pool-exhaustion rule:

> `pool has 5 placeable left-side seeds; requires N=10 from frozen K=16`

62 samples launched and returned, exact accounting, no arm generated.

## What the probe measures

Of 51 placeable candidates (obstacle in-route at `route_progress(apex) + foot_offset`,
left side, prominence ≥ frozen Pmin, valid packet window):

| outcome | n |
|---|---:|
| constructible | 5 |
| render window collapses (`duration_scale` check) | 24 |
| zero-shift assignment fails (phase alignment) | 15 |
| `step_phase` cycle exists but not at that apex | 6 |
| no valid `step_phase` left cycle in the clip | 1 |

**Per seed: 12/16 placeable, 5/16 constructible.**

## Mechanism: a swing-duration mismatch between donor and prior

The donor's step-over swing is **11 frames** (takeoff 50 → landing 61). Across the 123
left swings in the pool, the median walk swing is **8 frames** and **97 % are shorter than
the donor's**. Constructibility is confined to a narrow band:

| target swing (frames) | 6 | 7 | 8 | 9 | 10 | 11 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| constructible | 0 | 0 | 2 | 3 | 0 | 0 | 0 | 0 |
| window collapse | 1 | 10 | 13 | 0 | 0 | 0 | 0 | 0 |
| assignment failure | 0 | 0 | 0 | 11 | 1 | 1 | 1 | 1 |

Short target swings compress the packet's ±2-frame source window to under one integer
frame on one side of the apex, so the render refuses; long target swings fail phase
alignment instead. Widening the source half-window is not available: at half-window 3 and
4 **no donor pair in the bank is phase-alignable at all**, because the adapted and neutral
cycles lose common phase support. The ±2 window is simultaneously the widest the source
supports and too wide for most targets.

This is a substantive limitation of the coherent-packet representation as implemented, not
a tuning accident: **a step-over donor's swing is intrinsically longer than a walk swing,
so transporting it onto the prior's own nominal gait is constructible for only about a
third of seeds.** It is worth reporting alongside whatever the representation comparison
eventually shows, and it bounds how often any absolute-or-residual packet can be applied
at all.

## Consequence

The N=10 gate was sized from a replay that predated the constructibility probe, so it
assumed placement eligibility (12/16) was the binding rate; the true rate is 5/16 ≈ 0.31.
Nothing is relaxed on this pool: exp019's seeds are closed. The rerun preregistered at
K = 32 fresh seeds (4000–4031) with N = 8 keeps a real requirement (expected yield ≈ 10)
while sizing the pool to the measured rate — the same rate-honest resizing the calibration
family used at v2 → v3. The donor bundle, Pmin, placement rule, selection key,
constructibility probe, endpoints, and decision rule are unchanged.
