# A4: one-step absent-only hold adaptation

**Status: preregistered** — 2026-09-05, before generation/execution.

## Question and provisional contributions

A3's fixed3 achieved 19/32 local passes versus fixed2's 13/32; fixed4 lost progress.
Does per-carrier absent-only overlap timing improve on the strong fixed3 baseline?
Provisional deliverables: a four-repeat conservative timing rule; a 256-execution paired
development test; and records that price calibration and preserve fallback cases.
An adaptive performance contribution remains conditional on the result.

## Assignments and retained stack

Eight fresh units a4u0–a4u7, ARDY seeds 62000–62007; same beam underside 1.15 m,
center (1.2,0,1.275), size (.24,2.25,.25). This is fresh motion-seed evaluation, **not**
geometry generalization. Keep A3's 208 frames, 25 Hz, route 7.488 m, speed .9 m/s,
WALK cache, feature-noise v2, five DDIM steps, CFG [2,2], .28 m dip, quintic ramp/recovery,
frozen ARDY/SONIC and isolated 32-slot apparatus. Physics seed0, unit-major four slot repeats.

Arms: free_nominal, fixed3, fixed2, adaptive. Generate free_nominal/fixed3 first (16 actual
references), run their absent batches (64 executions), then freeze adaptation decisions
**before any present execution**. Generate fixed2/adaptive (16 references). Final queue:
free present, fixed3 present, fixed2 absent/present, adaptive absent/present (192 executions).
Total **32 actual generations, 256 actual executions**, including all calibration trials.
No new text encoding, model training, resampling, retiming, or post-generation IK.

## Deterministic adaptation and information budget

For each fixed3 carrier, use its reference whole-body overlap and four matched absent
overlap descriptors. Qualification requires >=3/4 absent local-progress passes and all four
forward exits observed. Let H=max(2, reference_exit+.10, max(absent_exits)+.10) seconds.
If qualified and H<=4, generate adaptive with hold H; otherwise regenerate the fixed3
specification as an explicit unchanged-control fallback. Preserve reasons and count the call.
No imputation of censored exits. One proposal, one new generation, zero feedback iterations.
The clocks are the same as A3: reference i/25, achieved (i+1)*dt; hold is nominal route-time.
This mapping is a hypothesis, not a physical invariant.

All comparators share the same calibration information/budget; the fixed rules ignore its
timing content by design. Report standalone fixed3 cost as well, so shared calibration cannot
hide deployment overhead. All finite generated references are executed in this **simulation
mechanism probe**, including unqualified fallbacks and reference-margin failures. This is
not a deployment admission rule or a screened high-quality dataset. Invalid arrays abort.
Record whether fallback arrays reproduce fixed3; do not assume batch determinism.

## Integrity, resources and endpoints

Clean immutable clone, externally stored fresh output, bound identities/input hashes,
one ASTRA lock, no retries of spent attempts, immediate apparatus validation per batch.
Generation and simulation run in separate processes. Reuse the validated compact allocator
and 32-env/208-frame release-tracker workload, same primitive apparatus; available-RAM
admission9216 MiB, free-VRAM5500 MiB, running floors2500/1200 MiB, timeout900 s.
Generation keeps8192/4096 MiB gates. Bounded waits180 s; never stop co-tenants.

Keep A3's whole-body local recovery/dwell/deadline, posture/corridor proxies and40 mm coverage
margin unchanged; full route separate. Report all-assigned /32 per arm, per-unit >=3/4,
matched gains/losses versus fixed3, qualification/fallback coverage, absent progress,
reference clearance and actual costs. Repeats are not independent scenes; no significance
claim from slot counts. Physical contact and full motion quality remain not measured.

## Decision

If adaptive loses to fixed3, preserve the loss and inspect whether the edited generation
invalidated its calibration; do not tweak the spent seeds or report only qualified cases.
If passage is preserved with shorter holds, measure deformation/quality before claiming
efficiency. A favorable result advances to new scene geometries and quality instrumentation,
not directly to a general traversal claim or a training-data release.
