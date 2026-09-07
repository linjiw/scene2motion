# A14: one-target compilation into a fixed execution context

**Status: preregistered** before A14 preparation or execution. September 6, 2026.

A13 completes64 exact observer replays and54 pure policy calls. At its stress
target, the first observed difference appears after step1 despite equal target
pre-state and applied actions; all18 fixed-target policy interventions are exact.
This localizes the observed onset downstream of the policy output, without
identifying a particular PhysX operation or hidden state.

## Constructive intervention

Compile one candidate into its original slot in a fixed32-slot reference context.
All31 other input references come from an immutable background, irrespective of
what other rows accompanied the incoming candidate. Keep the original ARDY/SONIC
weights, reference values, seed, environment slots, physics, isolated origins,
callback and encoder weights. This changes evaluation composition, not robot
dynamics or policy arithmetic. No GPU determinism flag is enabled: NVIDIA's
[PhysX5.6.1 flag documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/structPxSceneFlag.html)
states that enhanced determinism is unsupported on GPU.

`scene2motion.canonical_batch.compile_target` retains ordered identities and
copies numeric reference fields without changing dtype or values. Incoming rows
other than the target never reach the simulation. Canonical array layout and
serialization make the equality test byte-exact. Invalid identities, nonfinite
values and changed field schemas refuse compilation.

The cost is one useful candidate per32-env job. Background trajectories are
operational workload, not independent candidate successes. This pilot is not an
efficient final deployment claim. A different background requires separate
evaluation; canonicalization does not make GPU physics independent of scenes.

## Fixed assignments and incoming contexts

Use the original A12 `correction` input as the immutable background per group.
Candidates come unchanged from the original A12 `no_phase` input:

| Group | Target slot | Unit | Recorded encoder |
| --- | ---: | --- | --- |
| 0 | 7 | s01/u3 | G1 |
| 1 | 3 | s08/u3 | Teleop |

These are the lowest changed scientific slots for the indicated encoder in each
group. They are reused development cases selected for apparatus validation,
not new independent units or evidence of method improvement.

Incoming context A is the complete original no-phase batch. Incoming context B
is the original correction batch with that same no-phase target inserted. Compile
both through the identical fixed background. Before simulation, require equal
compiled bytes and exactly one changed reference relative to that background.
Also compile both original incoming contexts for every one of the39 previously
unchanged targets; require byte equality in all39 cases. These CPU checks are
construction invariants, not simulator results.

## Simulation, gates and budget

Four32-env absent jobs: group0/A, group0/B, then group1/A, group1/B. Budget128
operational executions, two distinct pilot candidates, zero new generations or
independent units. No present evaluation, held-out test, extra policy forwards,
or policy training. Source jobs inherit original A12 arguments except motion
file, output destination and the clean committed Scene2Motion Python path.

Require source/hash validation, original apparatus and slot-specific encoder
checks for every job. After each A/B pair, require exact equality of every full
valid qpos prefix, identities, sample clock, lengths, termination and progress.
Persist a failed equality receipt and stop before the next group. Do not loosen
the criterion, overwrite artifacts or automatically retry a launched job.

Use the compact supervisor with32 environments and ASTRA lock; unchanged admission
RAM9216MiB/VRAM5500MiB, runtime floors2500/1200MiB, timeout900s/job. Preparation
and numerical verification workers exit before each simulator launch. Preserve
preflight refusals; no background poller or co-tenant interruption. All outputs
are outside the clean committed preparation checkout.

## Interpretation and next gate

Passing means the adapter excludes unrelated incoming references and the two
resulting compiled candidates reproduce their execution under both tested
encoder realizations and groups. Report64 paired trajectories from128 executions,
including background rows; do not call this128 independent successes.

This does not identify the GPU backend's numerical mechanism, eliminate target
response sensitivity, or improve traversal. It supplies a controlled comparison
apparatus. Before renewed correction claims, candidate previews must use the
same declared background, slot and seed; cached previews require the entire
execution context identity. Preserve A12's failed comparison. Late and exact
cases remain in full-prefix apparatus checks; the six selected A13 cases do not
replace reporting over all39 unchanged-target construction checks.

## Prelaunch implementation correction

The initial `e2c4453` preparation failed its byte-invariance gate before writing
an identity or launching a simulator. Equal source arrays retained different
dtype and field-name object sharing after separate pickle loads, producing
different pickle memoization. Preserve that attempt in
`outputs/astra_a14_canonical_context_v1/preparation_failure.json`.
The corrected compiler canonicalizes metadata sharing as well as array layout;
motion values/dtypes, target assignments, thresholds and128-execution budget are
unchanged. A regression uses independent pickle loads, and all39 real source
invariants now pass. Use a fresh committed checkout and `astra_a14_canonical_context_v2`.
