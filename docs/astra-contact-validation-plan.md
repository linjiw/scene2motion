# ASTRA contact telemetry: staged validation plan

**Status: historical staged design, superseded by versioned preregistrations and
[2026-09-05 results](astra-contact-validation-result-2026-09-05.md).** Six corrected
simple-body controls and64 exact G1 replays are complete. An excluded-event diagnostic
was resource-aborted; contact coverage/controlled G1 positive sensitivity remain open.
The proposed sections below record the original plan, not the current completion state.

## Question and current implementation

Can we attribute solver-reported contacts and normal impulses to the floating beam
without changing the existing rollout? The existing net robot contact sensor cannot
separate beam from terrain. A5 geometry labels must remain unchanged and its physical
contact field must stay unknown until a separately validated measurement is available.

`scene2motion/contact_telemetry.py` now validates explicit view identities, referenced
contact-buffer slices and complete control-step substep coverage on CPU. Tests include
opposing normals, zero-force points, a transient contact absent from the final substep,
invalid buffers, reordered paths and missing substeps. **The live callback and exporter
integration were pending at this plan's creation.** See the result above for live tests;
CPU tests alone do not establish simulator correctness.

## Measurement contract

1. Read the pinned local PhysX `RigidContactView.get_contact_data(dt=physics_dt)` API.
   Construct explicit per-environment Table sensor paths and robot-body filter paths;
   check returned `sensor_paths`/`filter_paths` and ordering. No unresolved wildcard,
   cross-environment filter or unverified all-body coverage is admissible.
2. Collect every physics substep **before reset**, through the existing post-physics
   decimation callback. Preserve the original callback exactly once. Do not change
   actors, collision settings, controller observations, RNG, dt or decimation.
3. Store point counts separately from the sum of scalar normal impulses. Net vectors
   may cancel; reported points may have zero force. Keep minimum reported separation,
   buffer occupancy and clocks. Normal impulse is not total frictional impulse, and
   reported solver contact is not interchangeable with nominal geometric intersection.
4. Associate substeps with their policy action and apply the same valid-prefix mask as
   achieved-state exports. If retaining terminal-action contacts, mark them separately;
   never pair them with reset states or silently extend the scored state prefix.
5. Reject invalid indices, overlapping slices, exhausted capacity, missing substeps or
   identity mismatch. Missing or unreliable data means **unknown**, not contact-free.
   Buffer headroom alone does not prove absence of backend truncation: inspect simulator
   warnings and repeat a positive probe at doubled capacity.
6. Reduce buffers each substep; keep compact per-body/per-action summaries rather than
   retaining every padded GPU buffer. Record peak RAM/VRAM and logger cost separately.

## Proposed staged experiment (freeze before launch)

**Stage I: six scripted operational probes.** Use a simple rigid body contacting a
table as the positive case and a separated matched body as the negative case. Each
runs with logging off, base capacity and doubled capacity. Freeze trajectories,
duration, initialization and seeds before execution. Require positive-case reported
points and positive integrated normal impulse; negative-case zero reported points.
Require identical states across logging settings and capacity-invariant contact
summaries under a preregistered tolerance. Any mismatch stops progression; preserve
the failed receipts. This validates the API, not G1 traversal performance.

**Stage II: 64 matched operational G1 replays**, only after Stage I passes. Reuse A5's
default fixed3 absent/present arrays with the original 32-slot order, configuration,
physics seed, action modes and terrain. Compare against archived achieved states,
valid lengths, termination, live apparatus and mode records. Require exact equality
for those already deterministic readbacks; freeze any contact-only floating-point
tolerance before seeing results. Independently verify all intended robot body filters.

Proposed total: **70 operational executions, zero generated references, zero new
independent scientific units**. Replays stay in their source development groups.
Do not relabel archived A5 runs from replay contact labels; store linked replay records.
Any additional validation attempt requires a versioned amendment and full accounting.

Use the validated compact supervisor/profile, one ASTRA lock, and unchanged simulator
admission gates (9216 MiB available RAM / 5500 MiB free VRAM). Additional logging must
pass resource monitoring; stop rather than silently change batch size or physics.
Do not kill co-tenants. Freeze actual capacity and live-hook source hashes before launch.

## Decision after validation

If instrumentation preserves execution and passes controls, retain contact, geometry,
world progress and recovery as separate evidence fields. Only then define a prospective
final-edited-motion qualification rule on development data and test one bounded
progress-aware correction on fresh grouped units. Neither a zero contact count nor
low joint RMSE alone establishes high-quality motion or a safety guarantee.
