# A5 matched contact-instrument replay

**Status: preregistered**. Commit before preparation and execution.

## Gate and budget

The six corrected simple-body probes in `outputs/astra_contact_probe_v4`, source7ece77f,
must all verify, including exact state and capacity equivalence. Preserve the five
earlier operational attempts; these are instrumentation development, not scientific data.

Replay A5 fixed3/default absent then present,32 original slots each, exactly once:
**64 operational executions**, zero generated references, training or new independent
units. All variants retain `astra-a5/a5uN` development groups. Stop on any failed launch,
identity, coverage, contact-buffer or replay check; no automatic retry or seed tuning.

## Single change

Replace only the callback with `AstraContactExportCallback` and output destination.
Retain original reference pickle and order, explicit default encoder weights, checkpoint,
tracker7c63c53, isolated terrain assignment, physics seed0,32 slots, clocks, termination
rules, horizon and beam geometry. Read every physics substep through a scoped wrapper of
`recorder_manager.record_post_physics_decimation_step`; call its original implementation
exactly once and restore the descriptor on success or exception. No actor/RNG/physics edits.

Resolve Table and **all live articulation body paths** in every environment; verify
exact per-environment order. Fixed contact capacity8192 across the32-slot view. Invalid,
missing, saturated or overlapping data stops the launch; archive headroom and inspect
contact-capacity warnings. The simple-body capacity-doubling result is not a proof that
all possible G1 contact loads fit; untested loads remain outside this validation.

Preserve signed PhysX scalars/vectors, non-cancelling normal-impulse magnitudes and point
counts. Accumulate four0.005 s substeps per0.02 s action, then trim with the **same valid
prefix as achieved qpos**. Terminal/reset-frame contact is excluded, not zero-imputed.
No-beam records are explicitly not applicable, not fabricated contact measurements.
Store compact numeric arrays in a compressed archive; do not retain padded raw buffers
through the full rollout. Do not modify the original apparatus or mode records.

## Pass and interpretation

Require exact equality to A5 for achieved arrays, valid lengths, termination/progress,
apparatus and complete controller-mode records. Contact arrays must have matching motion
membership, body count and valid-prefix clocks, finite values and consistent count/impulse
fields. Retain all32 records per arm, including cutoffs and noncrossings.

A pass establishes read-only contact logging under this configuration. It does not
retroactively label A5 contacts, certify high-quality motion, establish safety or add
independent scenes. Publish contacts on linked **replay** records. Compare reported points,
nonzero normal impulse and replay geometry separately, including inconvenient disagreements.
Any new motion-quality admission threshold requires a later prospective protocol.

## Resources

Use the validated compact allocator and lean supervisor, one ASTRA lock, clean committed
checkout and fresh external output `outputs/astra_contact_replay_v1`. Keep9216/5500 MiB
RAM/VRAM admission,2500/1200 MiB runtime floors,900 s timeout and180 s bounded admission
wait. Log memory and elapsed cost; do not change batch size or kill co-tenants.
