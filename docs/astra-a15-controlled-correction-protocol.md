# A15: frozen corrections in a common nominal execution context

**Status: preregistered** before preparation or simulator execution. September 6, 2026.

This implements the [A15 design](astra-a15-controlled-correction-design.md).
A14's canonical adapter is fixed. The question is whether the existing A12 full
correction improves the original absent-preview geometry/progress descriptor
over the baseline-preserving fixed0.26 m shallow command. No response refit,
controller change, new generation, present test or confirmation is assigned.

## Sources and assignments

Use `outputs/astra_a12_progress_correction_v1/g{0,1}_baseline.pkl` as immutable
backgrounds. The nominal controls compile that entire context without changing
any reference value. Candidate sources are A12 `g{0,1}_correction.pkl` and A10
`outputs/astra_a10_half_step_v1/g{0,1}_depth_minus.pkl`. Bind their original input
receipts and validated source jobs. A12's generated proposals and thresholds stay
frozen; its historical failed comparison remains unchanged.

Assign `u0` from scenes s00–s11 by index, one reused development seed per scene:
seed64000+4*scene index (the executable manifest also binds the original row).
Scenes s00–s07 retain group0 slots0,4,8,12,16,20,24,28; s08–s11 retain group1
slots0,4,8,12. Preserve the original realized encoder mask in each slot, all
32 ordered identities, physics seed0, callbacks, simulator/model/checkpoint and
reference values. The preparation manifest stores all three source rows and
encoder masks before launch. No padding duplicate is a scientific unit.

Each candidate changes only its assigned target relative to the nominal
background. Use the A14 canonical serializer, with exact non-target value checks.
Compare nominal, fixed shallow and frozen full correction for all12 units.
The free-nominal controls and constructibility checks remain the verified A12/A10
source controls; this context pilot assigns no new free-WALK performance arm.

## Execution order, equality and reuse

First run group0 nominal and group1 nominal, requiring exact original A12 full
valid qpos, identities, sample clocks, lengths, termination and progress on all
64 slots. Save a failed comparison and stop before candidates if either fails.
These controls establish that the nominal previews used to compute A12 proposals
remain valid under the chosen canonical background.

Then visit scenes in increasing index: fixed shallow followed by full correction.
Before outcomes, assign duplicate-input aliases by the pair (source group,
SHA256 of the complete canonical32-slot payload). Within a group all remaining
execution arguments, seed, modes, background and pinned sources are identical;
only output/input filenames differ. The first request owns the execution. A
later exact alias uses that owner's verified target row and receives an explicit
reuse receipt. No cross-group alias or target-only cache is allowed. An alias
does not add independent evidence. Candidate contexts need not yield unchanged
background trajectories; only the declared input backgrounds are fixed.

Maximum26 jobs/832 operational rollouts; two nominal controls plus at most24
one-target jobs. There are36 assigned scientific preview measurements for12
reused units, including nominal rows; report alias counts and unique jobs
separately. At most two previews per method/unit, counting the shared nominal.
No launched job is automatically retried. All failures and resource refusals
are preserved; outputs are fresh and outside a clean committed source checkout.

Use the existing32-env compact supervisor and ASTRA lock. Admission remains
RAM9216MiB/VRAM5500MiB; runtime floors2500/1200MiB;900s/job, maximum23400s
simulator-job wall budget. Preparation and numerical verification workers exit
before simulator launches. Preserve all original provenance and check apparatus,
clock and original-slot encoder signatures after each job. No background poller,
allocator override, gate relaxation or co-tenant interruption.

## Endpoints and promotion

Use unchanged `scene2motion.scene_relative_duck.measure_motion`: whole-body
geometry with40mm inflation, witnessed rear-body exit by rear_m/0.9+2.5s,
at least0.45m continuation over the complete following second, and upright
corridor recovery. Preserve non-arrivals and censored/incomplete recovery.
Report raw references and achieved descriptors, margin/nominal penetration,
rear exit, continuation, termination and every assigned denominator separately.
These are obstacle-absent measurements, not physical collision or traversal.

Use unchanged `progress_duck_correction.select`: retain a qualified nominal;
otherwise choose the method's own qualified candidate; otherwise refuse. Report
both raw and selected passes, per-scene paired wins/losses, and descriptive
Wilson95 intervals. No statistical significance or held-out claim is authorized.

Promotion to the remaining reused development seeds requires full correction's
selected pass count to strictly exceed fixed shallow's and zero losses to
nominal. Equality fails promotion. Passing only authorizes designing the next
development validation, not method freeze or confirmation. If this gate fails,
preserve the frozen method's negative result and design a successor from the
controlled response evidence; do not change endpoints on these spent units.
