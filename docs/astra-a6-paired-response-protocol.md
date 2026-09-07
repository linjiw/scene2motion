# A6 paired default-encoder continuation

**Status: preregistered**. 2026-09-06; commit before preparation/launch. This is an
execution-layout correction to A6, not a gate relaxation or regeneration.

## Preserved design defect

A6 d0 atd4607e4 generated all384 references. Three32-slot absent batches completed
before orchestration was stopped at a completed-process boundary. The default SONIC
weights[1,1,1] select different internal encoder masks across slots. Putting the eight
native variants of one unit into eight slots therefore confounds command response with
encoder realization. The first unit's baseline is teleop but its depth−/+ are G1.
This was detected from runtime mode readback before response fitting or geometry/progress
analysis. Preserve all96 d0 previews as confounded development diagnostics; do not fit
them as matched local command derivatives or silently pool them with this continuation.
No simulator was killed. The interrupted post-process verification is completed separately.

## Corrected assignments and exact budget

Reuse the exact384 archived A6 qpos arrays, commands, seeds and split groups from
`outputs/astra_a6_response_d0`; bind every generation receipt and archive hash.
No new generation, qpos deformation, seed draw, controller setting, physics setting,
scene endpoint or geometry margin. Keep default weights[1,1,1], physics seed0 and
32 environments with the previously validated compact profile.

For each variant in the original order (baseline, depth−/+, onset−/+, recovery−/+,
free_nominal), execute the same unit in the same simulator slot across all variants.
Group0 contains the first32 units (scene indices0–7), one per slot. Group1 contains
remaining16 units (scenes8–11) in slots0–15 and the same16 as explicitly labelled
padding duplicates in slots16–31. Padding provides the unchanged32-env workload;
its rollouts are recorded and charged but are not extra independent samples.

New cap:16 launches×32 = **512 actual absent previews:384 scientific candidate previews
and128 padding duplicates**. Together with96 d0 diagnostics, maximum608 operational
executions for384 generated candidates. The original336 response candidates/48 free
controls and48 development unit groups are unchanged. No obstacle-present or held-out
execution. This budget amendment is required to match default encoder realizations
without changing the controller or the validated simulator batch size. It does not
change the eventual two-generation/two-preview deployment budget.

Require exact slot membership and constant runtime encoder masks per valid preview.
For every nonbaseline variant, the32-slot mask vector must equal its group's baseline
mask vector exactly. Refuse and preserve any mismatch before the next launch or local
response fitting. This checks realized encoder identity; it does not assert bitwise
identity of all physics perturbations. All scene/command differences remain unchanged.

## Provenance, gates and stop behavior

Fresh output `outputs/astra_a6_response_d1` outside a new clean pinned checkout.
Original checkout and d0 archives stay immutable. Fix resource-evidence path resolution
by explicitly binding the original resource archive, including its local logs; never
skip a hash or weaken the gate. All A6 resource floors, timeout, source/checkpoint checks,
apparatus isolation and all-assigned/censoring rules remain fixed.

Check before each launch. Resume only complete validated jobs; do not repeat failed or
incomplete launches. Stop on admission failure with unlaunched work pending. The driver
supports `--max-jobs` and an output-local `STOP_AFTER_CURRENT_JOB` sentinel, checked
between launches, so stopping never requires interrupting a simulator.

All native command/outcome measurement and development-only descriptor definitions
are identical to the original A6 protocol. Fit only scientific slots with complete
response data; report missingness and refusals for every assigned unit. Report padding
and diagnostic overhead separately. Training residuals are not held-out allowances,
within-unit fitting is not transfer, and a corrected preview is not present passage.
