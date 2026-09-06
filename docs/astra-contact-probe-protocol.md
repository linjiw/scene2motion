# ASTRA contact instrument: simple-body validation v1

**Status: preregistered**. Commit this protocol and implementation before running.
Implements Stage I of the draft contact plan only; G1 replay is not authorized by a
passing simple-body result alone and needs its own integration protocol.

## Frozen question and comparison

Does read-only, obstacle-specific PhysX contact logging detect a deliberately contacting
body, report zero points for a separated body, and preserve simulated states? Six
operational executions, in order: positive off/base/double, negative off/base/double.
No motion generation, controller, training, fresh scientific units or success-rate claim.
Each execution starts a separate Isaac process at physics seed0 on CUDA:0.

Table: kinematic cuboid, size(1,1,0.2) m, centre(0,0,0.4) m. Body: dynamic cuboid,
size(0.2,0.2,0.2) m, mass1 kg, identity orientation, zero velocity, centre(0,0,1) m
for positive or(2,0,1) m for negative. No ground plane. Both bodies have contact
reporting activated in **all** arms, including logger-off. Gravity/default physics
settings come from the installed Isaac Lab; source state and runtime API hash are bound.
Initialize poses/velocities explicitly after simulator reset. Run200 steps of0.005 s,
grouped into50 intervals of0.02 s; no rendering or subsequent resets.

## Instrument and pass rules

- Explicit sensor `/World/Table`, filter `/World/Body`; exact view readback required.
  Off creates no contact view. Base capacity64 contact points; double128.
- Read every physics substep immediately after stepping. Preserve counts, scalar normal
  impulses, net normal impulses and reported separation; distinguish zero-force points.
  Aggregate all four substeps; reject malformed, saturated or incomplete buffers.
- Each positive body's final20 samples must lie within0.02 m of z=0.6 m. Each negative
  body must end below z=0. The checks validate the scripted apparatus, not humanoid safety.
- Positive logged arms must have positive point count and integrated normal impulse.
  Negative logged arms must have exactly zero points and normal impulse.
- Entire200×2×13 body/table state tensors must be **exactly equal** to logger-off.
  Base/double per-substep contact summaries, except buffer capacity, must be **exactly
  equal**. No tolerance relaxation after results. Preserve contact-related overflow,
  truncation, exceeded-capacity or insufficient-buffer warnings as a failed gate.
- Stop on the first launch or verification failure; retain all artifacts. Resume verifies
  completed jobs, never repeats a spent launch. A harness correction requires a versioned
  amendment and fresh output; it does not erase the original attempt.

## Resources and execution

Use one `/tmp/scene2motion-astra-a1.lock`, clean committed execution checkout, external
fresh output, compact allocator and standard-library supervisor. Keep9216 MiB RAM and
5500 MiB VRAM admission,2500/1200 MiB running floors;300 s timeout per launch. Never
kill co-tenants or change dt, body count, capacities or scene to fit current resources.
This is a separate small workload, **not** a new lower-memory SONIC equivalence claim.

Run `python -m experiments.astra_contact_probe --out <external-fresh-directory>` from
the clean checkout with the repository on PYTHONPATH. Store launch identities, commands,
logs, resource samples, states, contact summaries and per-job verification. The primary
result is six operational pass/fail rows, not a traversal percentage.

## Next decision

If every gate passes, integrate the scoped pre-reset hook into G1's exporter and freeze
the64 matched absent/present replay checks separately. If not, diagnose the measurement
failure before any G1 launch. Original A5 physical-contact labels remain unknown.
