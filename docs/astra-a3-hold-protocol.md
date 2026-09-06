# ASTRA A3: fixed-depth native hold-time mechanism probe

**Status: preregistered** — 2026-09-05, version `hold-d0`, before generation or execution.

## Question and evidence boundary

A2r produced two local crossings with fixed-extra ducking, but poor progress and delayed
achieved overlap. The absent-only body-envelope audit preserves censored exits instead of
imputing zero lag. It does not calibrate a controller error bound. Before designing an
adaptive hold controller, test whether extending the native height hold improves executed
clearance without sacrificing progress. This is a development mechanism probe, not a new
geometry holdout, learned method, safety guarantee, or final paper comparison.

Provisional deliverables are: a censored scene-relative overlap descriptor; a bounded
native hold intervention; and paired execution records. Any performance contribution is
conditional on the measured result. Fixed long holds are necessary strong controls for a
later adaptive method, not evidence that adaptation already works.

## Frozen assignment and intervention

Eight new units `a3u0`–`a3u7`, ARDY seeds 61000–61007. One reused geometry: beam underside
1.15 m, center (1.2, 0, 1.275) m, size (0.24, 2.25, 0.25) m. Straight route 7.488 m,
208 frames, 25 Hz, nominal speed 0.9 m/s. Cached “A person walks forward.” text, pinned
ARDY feature-noise v2, five DDIM steps, CFG [2,2]; no encoder loading or training.

Four arms in order: `free_nominal`, `hold2`, `hold3`, `hold4`. Each gets one generation per
unit, same seed and root path/heading. Free has no height constraint. Other arms share a
0.28 m dip, quintic ramp over 0.45 m, and recovery over 0.9 m. Only recovery onset varies:
nominal route-time 2, 3, or 4 s (1.8, 2.7, or 3.6 m). The route-time grid spans 0–8.32 s;
reference samples are timestamped i/25, ending 8.28 s. Do not confuse these clocks.
Hold2 exactly reproduces A2's original dip shape at this amplitude. No candidate selection,
extra sampling, adaptive update, or early removal of finite failed references.

## Budgets, apparatus and provenance

Four batches of eight = **32 actual references**. Each arm runs absent then present in
32 environments: eight units × four slot repeats, unit-major keys `a3uN_rR`. Total eight
launches = **256 actual executions**. Same slots, terrain assignment and physics seed 0
across all arms. Repeats are not independent scenes; there is one geometry and eight
motion seeds. Include free-nominal present, not merely an absent reference control.

Use the A2r explicit isolated terrain callback; require >=15.9 m origin separation and
unchanged origins during rollout. Enforce apparatus pose/extent/neighbor checks immediately
after each launch, before continuing. Keep all failures and refuse on invalid apparatus.
Preregistered code runs from a clean immutable clone with external outputs. Bind source,
protocol, model, runtime, cache, reference, pickle, launch, log and output hashes. No
regeneration after stochastic sampling starts. Resume only hash-verified completed stages;
preserve partial attempts. One ASTRA lock, no co-tenant termination.

Generation resource gate: free VRAM 4096 MiB / available RAM 8192 MiB. SONIC: 5500/9500 MiB;
running abort floors 1200/2500 MiB, 900 s per launch. Separate generation and execution
processes to release ARDY. Retain measured resource costs; no 18 GB minimum.

## Endpoints, reporting and next decision

Keep A1's frozen whole-body recovery past the beam, root >=1.82 m, corridor, 0.20 s dwell,
4.0 s deadline, pelvis >=0.5 m and up-z >=0.7 proxies. Report low-pelvis flags without
calling them instrumented falls. Separate target-beam geometric replay, absent progress
with a virtual raised beam, obstacle-present local passage and full-route completion.
Maintain the 40 mm primitive-coverage margin. Physical contact and full motion quality
remain unmeasured; local success does not certify a high-quality demonstration.

Report per-slot successes, per-unit 3/4 repeatability, matched wins/losses against hold2,
and all-assigned denominators (8 units, 32 executions per arm/condition). This small probe
does not establish significance. After execution, inspect achieved overlap and clearance
loss using valid state prefixes; nonarrival/unfinished overlap remains censored.

If longer holds improve clearance but reduce progress, do not scale the method: improve
carrier compatibility. If no timing arm closes reliably, question the carrier/native-control
interface instead of extending hold indefinitely. If improvement preserves progress,
freeze that fixed baseline and design an adaptive dual-window method on development data,
then evaluate fresh units/geometries. Do not relax spent endpoints or claim superiority
from the best post-hoc arm alone.
