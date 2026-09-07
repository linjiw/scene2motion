# A9: identical-input, identical-slot execution replay

**Status: preregistered**. September 6, 2026. Commit protocol and driver before
preparation. Operational development probe motivated by A8's failed numeric response
gate and the separately labelled A6 different-slot padding diagnostic. A8 is closed.

Replay exactly two original A6 paired jobs, in order: `g0_baseline_absent`, then
`g0_free_nominal_absent`. Each has 32 scientific slots spanning development scenes
a6s00–a6s07, seeds64000–64031. Use the original serialized input, slot order, physics
seed0, default encoder weights, checkpoint, frozen simulator/source checkouts and
environment delta. Only the evaluation output directory changes. No regeneration,
new command, generated-trajectory editing, encoder sweep, physics randomization or
present obstacle. Bind original jobs, launches, process/artifact receipts and inputs.

Budget: at most two new32-slot launches =64 operational replays, zero new independent
units or confirmation seeds. Pair each slot with its original A6 run. Keep baseline
and free-WALK results separate. All32 assigned pairs per arm stay in accounting.
These are one repeat of each job, not a variance estimate or population guarantee.

Before interpreting each completed job require original apparatus validity, matching
runtime encoder signature in every slot, original clock and membership. Report exact
valid-prefix qpos equality, length/termination equality, common-prefix root-position
RMS/max in metres, observation-status and existing A6 descriptor disagreements.
Numeric clearance, exit and continuation differences require both targets observed;
record unavailable counts and per-scene observations. No frame tolerances, altered
thresholds or post-reset frames. Compare the original scientific descriptor with its
unchanged reconstruction; source A6 measurements remain immutable.

The exact replay check passes only if all64 paired qpos arrays, valid lengths and
termination labels match exactly. A state difference is a scientific diagnostic,
so complete both arms unless provenance, apparatus, encoder, resources or process
integrity fail. If exact replay fails, do not interpret finite differences as a
deterministic transfer function without a separately designed repeat/noise study.
If it passes, prepare a separate bounded half-step native-generation protocol with
duplicate no-edit controls; passing this check alone does not reopen A8 or authorize
an optimizer/confirmation run. Controlled positive G1 contact sensitivity remains
a separate open measurement task.

Use fresh `outputs/astra_a9_same_slot_replay_v1` outside a new clean preparation
checkout. Keep original `/tmp/s2m-a6-paired-d9ac674` unchanged. Inherit compact32
profile, RAM/VRAM admission9216/5500 MiB, runtime floors2500/1200 MiB,900 s timeout
and single ASTRA lock. One immediate admission check per invocation, no poller.
Preserve refusals; an unlaunched prepared job can later be admitted unchanged.
Never retry a launched incomplete/failed job. Existing completed jobs are verified
before skipping. No co-tenant process is interrupted.

Prepare with `experiments/astra_a9_same_slot_replay.py --stage prepare --out PATH`
from its committed clean checkout. Run with the same driver and `--stage execute`;
each lightweight supervisor process finishes before CPU geometry verification.
`--stage analyze` rebuilds the completed result without new simulation.
