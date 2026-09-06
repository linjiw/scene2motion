# ASTRA A2r: isolated, slot-matched execution of frozen duck references

**Status: preregistered** — 2026-09-05; version `isolated-d0-r1`, before any A2r launch.

## Why a successor is necessary

A2 generated all 56 references and executed 136 rollouts, but apparatus checks failed:
rough-terrain initialization reused some world origins. `env_spacing=12` does not override
the terrain importer's random patch selection. A2's automatic driver also incorrectly waited
until final analysis to enforce apparatus checks. Preserve A2 unchanged as an invalid harness
attempt, not a valid negative or positive method result. Initial and measured references
were byte-identical; their different-slot outcomes cannot identify a correction effect.

This successor changes **physics initialization and batching**, not references, selection,
scene heights, endpoints, controller checkpoint, or method parameters. No regeneration or
training; no claim of a blind new test. It is a documented harness repair on spent references.

## Isolation and pairing

Use the same pinned rough terrain (20x20 patches, 8 m patch width). A new, explicitly active
callback assigns unique `(row, col)` indices in row-major `[0,2,...,18]² order, rather than
sampling patch levels. Require finite terrain origins and >=15.9 m pairwise horizontal
Chebyshev separation. Keep terrain level/type bookkeeping consistent. Verify origins stay
unchanged at every sample, and enforce live table pose/extent/neighbor checks after **every
launch before launching the next**. Stop on the first invalid apparatus receipt.

Each launch uses 32 environments: eight original units × four slot replicates, unit-major.
Every method now occupies **the same 32 environment indices**, same motion key ordering,
physics seed 0 and initialization path in separate launches. Four slots are repeated
initialization/terrain measurements, not four independent scenes or physics seeds. All are
development; there are still only two geometries.

Initial and measured selected arrays are identical in all eight units. Enforce that equality
before any launch and reuse initial execution for measured as an explicit alias. It is not
an independent rollout and cannot demonstrate measured-repair benefit. Each original arm
still has the same declared logical generation budget; no-op reuse is reported separately.

## Budget and frozen order

Seven launches of 32: free nominal absent; initial absent/present; fixed-extra absent/present;
best-of-three absent/present = **224 actual executions** (256 method-arm records including
64 measured aliases, plus 32 free controls = 288 logical records). One physics seed.
All source qpos arrays and the shared reference-only selections come from A2's archived
generation receipts and are hash-verified. All finite selected references execute; do not
remove failed controls from the denominator. No scientific retry or gate relaxation.

Same SONIC measured resource gate 5500/9500 MiB, running abort floors 1200/2500 MiB and 900 s
per-launch timeout; one lock, clean clone, outputs outside, no co-tenant interruptions.
Source, tracker, model, input pickle, protocol and output hashes bind each launch. Resume
only completed hash-verified launches, with identical source identity. The original eight-
environment and mixed-slot protocols remain unchanged and their outputs are not pooled.

## Endpoints and decision

Keep A1 whole-body recovery / 0.20 s dwell / 4.0 s deadline and all geometric definitions.
Absence progress uses the virtual raised beam; target replay and real target presence are
separate. Contact and full motion quality remain unmeasured. Report all eight units and
four repeats, local successes per slot, independent event flags, and paired differences
under matched slots. Do not claim significance or measured-feedback superiority.

If qualified local crossings exist, inspect motion quality and execution clearance loss
before promoting demonstrations. If uncut trajectories are simply slow, report that
diagnostic without changing this endpoint. If all corrected references fail in matched
absence, the next method must preserve a better carrier/contact pattern; no more blind
depth sweeps. A2r is not the final scene-disjoint method study.
