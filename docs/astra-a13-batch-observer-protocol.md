# A13: locate the first batch-context divergence

**Status: preregistered** before any A13 simulator launch. September 6, 2026.

This operational diagnostic follows A12's failed expansion and exact whole-cohort
controls. It does not create new independent units or change a scientific gate.
Source: `outputs/astra_a12_progress_correction_v1`, group0 `correction` and
`no_phase`, in that order. Outcome-selected stress target: slot19, s04/u3.
No new generation, present obstacle, held-out evaluation, or policy training.

## Frozen workload and observation

Two32-env seed0 absent replays, maximum64 executions, no automatic retry after a
launch. Preserve both original32-slot input PKLs, checkpoint, encoder weights,
environment origins, spacing, physics, terminations, resource profile, and all
other original command arguments. The only command changes are fresh output,
observer subclass and its original/preceding archive arguments. Replace the A12
checkout entry in PYTHONPATH with the new clean committed observer checkout;
retain original source pins and add the observer's provenance.

`AstraPolicyObserver` inherits the exact A12 controller/apparatus/state exporters.
For warm-up (-1, no physics) and steps0–7 it copies all observations, full actor
history inputs including strides/dtypes, encoder commands/times, existing cached
encoder latents/tokens, raw actions, and pre-state. Physics steps also capture
applied actions from the wrapper's existing `extras.env_actions`, post-state and
done flags. Root state is world-frame; environment origins and joint names are
explicit. The original valid-prefix archive and encoder ledger remain authoritative.
No extra inference, hook, history reset, normalizer, or state mutation occurs
during simulation. Post-reset states in the finite trace retain done flags and
must not be described as valid achieved continuation.

The generic wrapper has an optional action-transform module; the pinned release
disables it. Record/assert that it remains inactive and all policy modules are in
evaluation mode. The actor's running normalization is disabled in this release.
No normalization or backend mechanism is assumed from these source observations.

## Gates and inference intervention

Compare each observed32-slot archive to its own original A12 archive with exact
array equality, identities, lengths, termination, progress and sample clock.
Require original apparatus and encoder checks too. Persist failed comparison and
stop before the next launch if any observer execution differs. Never substitute
a fitted tolerance. Require all64 exact before interpreting the cross-context trace.

After the second simulation completes and its exact archive comparison passes,
use that process's loaded policy for at most54 pure-forward calls. The first
context's external verification and traces must already be pinned in the second
job. Restore captured full inputs with their original dtype, shape and stride.
For each of nine calls from each context, forward the entire input twice:36 calls.
Require both an exact live-action replay and exact repeat across every batch row.
If any fails, preserve the control failure and execute no hybrids.

Only if all18 whole-input controls pass, use the other context's full batch,
replace slot19 with the current context's entire input row, and forward once per
call/direction:18 hybrids. Assert target input equality and unchanged batch
shape/stride. Report exact target-action equality and maximum absolute differences.
All extra forwards happen after physics and archive export. They cannot alter
the completed execution; the external verifier still checks the final receipt.

These probes locate input versus policy versus downstream divergence. They do
not establish a general fix. If the first difference is already in inputs,
trace its producer next. If controlled policy actions differ, isolate the relevant
operation before implementing a correction. If actions agree, inspect applied
actions/state transition. No conditional branch authorizes a further simulation.

## Resources, provenance and reporting

Use the standard-library compact supervisor and ASTRA lock. Admission requires
available RAM9216MiB/VRAM5500MiB; runtime floors2500/1200MiB; timeout900s/job.
Preserve refusals, launch/process/resource receipts and all failed artifacts.
No background poller and no co-tenant interruption. The preparation worker exits
before the supervisor launches Isaac. Source identity includes both original jobs,
verified artifacts, original input hashes, checkpoint/config/runtime checkout,
policy/wrapper modules, and the committed observer, harness and protocol.

Report all assigned operational executions separately from scientific trials,
observer equality counts, first differing target fields, whole-input controls and
hybrid outcomes. A subsequent apparatus fix needs a separately frozen validation
covering additional slots and encoder realizations, exact same-context repeats,
and target stability under declared context substitutions. Keep A12's failed
method gate, broader motion/scene goals and positive G1 contact obligation intact.
