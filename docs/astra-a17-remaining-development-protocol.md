# A17: remaining-development depth-only comparison

**Status: preregistered**. September 6, 2026. This executable protocol adopts
[the outcome-free design](astra-a17-remaining-development-design.md) without changing
commands, endpoints, assignments or budgets. Preparation must run from its clean
committed checkout; the simulator continues using `/tmp/s2m-a15-88fcc45` unchanged.

## Assignments and comparison

All 36 previously used development units: scenes 0–11, units 1–3, seed
64000 + 4*scene + unit, group=scene//8, slot=(scene%8)*4+unit. Three arms:
original nominal, fixed 0.26 m depth, and frozen A12 no-phase correction.
Original generated arrays, encoders, 32-slot nominal backgrounds, physics seed 0,
controller and checkpoints are immutable. No generation or training is assigned.
The inventory has 108 arm requests, 41 prior-context aliases, 30 same-unit
candidate aliases and 37 unique possible new jobs. Preserve every assigned row.

Use original absent-execution qualification: whole-body scene-relative clearance
with 40 mm inflation, the original exit deadline, corridor/upright requirement,
and at least 0.45 m forward continuation. Preserve qualified nominal previews;
otherwise select the exact qualified candidate, or abstain. Never use an
obstacle-present outcome to select. This is development qualification, not
physical collision ground truth or confirmation on unseen scenes.

## Proof and dispatch order

Before dispatch, revalidate all original source hashes and complete cache runtime
contracts, reconstruct all 36 nominal measurements from A15 raw states, and prove:
30 depth-only payloads equal their same-unit shallow payloads; five no-edit
payloads equal nominal AND their cached nominal measurements qualify. Under the
common nominal-preserving selection these 35 pairs have identical selected
labels. Shared candidate absolute outcomes remain unknown.

Only s01/u2 (seed 64006, group 0/slot 6) differs. Dispatch its fixed-shallow job
first, then its no-phase job. Freeze the remaining 35 shallow jobs at preparation,
in scene/unit inventory order, but execute them only if no-phase qualifies and
shallow does not under the common selection. Tie or loss closes the strict lead
gate, saves the partial summary and stops after two jobs (64 operational rollouts).
Do not report absolute edited-method rates over 36 in this branch. Report the one
measured pair, 35 construction-equivalent pairs, and net selected difference only.

If the pair wins, execute the other 35 jobs, reconstruct all 108 arm measurements,
and report all 36 units. Require strict selected gain over shallow and no lost
nominal passes. Report per-scene paired wins/losses, descriptive Wilson intervals,
and a 10,000-resample percentile bootstrap over the 12 scene clusters, NumPy
PCG64 seed 170006. This interval is descriptive reused-development uncertainty.
Keep the A16 12-unit pilot separate; any pooled 48-unit table is secondary.

## Resource and integrity contract

Maximum 37 jobs × 32 environments = 1,184 new operational executions, including
37 useful target previews and 1,147 context-workload executions. The initial gate
uses two useful previews and 62 context rows. Maximum 900 s/job, 33,300 s declared
simulator timeout budget. Admission: RAM 9,216 MiB / VRAM 5,500 MiB; runtime floors
2,500 / 1,200 MiB. Use the existing exclusive supervisor lock and timeout rules.
Numerical preparation/analysis workers exit before a new simulation starts.
Do not interrupt co-tenants or install a poller. Resource refusal spends no launch;
preserve it. A launched failure is never retried. Resume validates all sources,
payloads, runtime contracts and completed receipts. Honor STOP_AFTER_CURRENT_JOB.

Run focused boundary/proof/provenance tests before preparation, and the full CPU
suite before reporting completion. Rebuild analysis exactly; independently verify
the decisive pair's raw states, all nominal records and the equivalence proof.
A failed lead gate closes this frozen correction rule on these development units.
A positive gate supports a separate contact-calibrated obstacle-present protocol;
it does not authorize a new success label or a generalization claim.
