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

## Addendum (v3/v4 attempts): placement validity and constructibility are anti-correlated

The K=32 rerun (seeds 4000–4031) cleared its gate at **17/32** and produced the E1
family's first complete set of arms — 8 nominal, 8 absolute, 8 residual, 120 samples
exact (`outputs/exp019_gait_matched_stepover_v3/`). It made no representation claim: the
obstacle was placed at the swing apex without requiring a footfall-free footprint, so in
**all 8 seeds** a support-phase footfall sat 0.007–0.115 m from the obstacle centre inside
the 0.140 m half-extent. Every arm — including the unmodified nominal — collided and
scored zero whole-body clearance, and residual − absolute was noise on every endpoint. The
free nominal reference arm is what made this legible; it is now permanent.

Adding the footfall requirement with the obstacle still pinned to the apex
(`outputs/exp019_gait_matched_stepover_v4/`) fails closed the other way: **6/32**. The two
requirements are anti-correlated, because constructibility wants the 8–9-frame target
swings above while footfall-free footprints want longer strides:

| requirement (K=32 pool) | seeds |
|---|---:|
| packet constructible | 17/32 |
| footfall-free footprint at the apex | 21/32 |
| **both, obstacle pinned to the apex** | **6/32** |
| both, obstacle at the footfall-free frame nearest the apex, ±1 | 11/32 |
| **both, same within ±2 (the packet half-window)** | **13/32** |
| both, same within ±4 | 14/32 |

The protocol therefore lets the obstacle sit at any footfall-free route frame within ±2
frames of the apex — the packet's own temporal footprint, a quarter of exp017's ±8 gate —
probing every such frame, with |shift| in the outcome-free key so an exactly-at-apex
placement still wins when one exists.

## Retraction: the 13/32 replay estimate was not reproducible

The ±2 row above (13/32) came from an offline replay and **is retracted**. Three
independent fresh K=32 pools run through the real harness measured **6/32, 6/32 and
5/32** — the last with every footfall-free frame probed, which was the one implementation
difference the replay had exploited. The replay was permissive in some way I could not
reproduce, and three real pools outweigh it.

**The standing finding, independent of what the comparison eventually shows:** for this
frozen prior and this donor bundle, a scene where the step-over packet is both
*applicable* and *winnable* exists for roughly **one walk seed in six** (≈ 17 %), and only
because the obstacle is allowed to follow the gait rather than the reverse. That rate, not
any residual-minus-absolute delta, is the honest headline about how far a coherent packet
extends the prior's usable capability. Pools 4000–4031, 4100–4131 and 4200–4231 are
closed; K = 64 (seeds 4300–4363) yields ~11 expected against N = 8.
