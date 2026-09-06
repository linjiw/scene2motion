# ASTRA A1b: Target-beam intervention on reachable WALK controls

**Status: preregistered**. 2026-09-05; version 1. Engineering development, not method evaluation.

## Entry evidence and assignment

A1 (`outputs/astra_a1_apparatus_v1_2`, execution commit `8c07e00`) completed 32 rollouts.
The two cohorts passed 6/8 and 7/8 paired local checks; all apparatus readback checks passed.
Take **all eight** members of the first passing cohort, EXP-023b s4640–s4647, in original order.
Do not remove s4645, which failed qualification. The seven qualified members are a separately
reported conditional diagnostic, never a replacement for the all-eight denominator.

One new eight-environment SONIC launch, physics seed 0. Reuse the archived references, patched
tracker/checkpoint, spacing, episode duration, callback and local endpoint. Change only the beam
underside from 1.60 to **1.05 m** (center z=1.175; x=1.2, y=0; full extents 0.24×2.25×0.25 m).
The 1.05 m target is the previously specified floating-beam development condition, now tested on
reachable controls rather than the earlier two non-arriving STEP references. Bind A1 summaries,
process receipts, achieved/apparatus archives and the new per-motion table metadata before launch.

## Question and progression

Does lowering the beam into the nominal WALK body envelope prevent or perturb local passage
that was possible with the beam raised? Read back actual target-beam poses/extents, retain all
achieved states, and rescore the absent trajectories against the target beam as a **virtual**
counterfactual. Compare achieved states at identical timestamps and valid shared prefixes;
do not align away the intervention by dynamic time warping.

Report all-eight local completion, nominal/margin intrusion, fall, cutoff, maximum progress,
root/whole-body completion and paired state change. The operational blocking check passes if
all apparatus checks pass, at least 6/7 previously qualified controls fail target-beam local
completion, and at least one of those controls has >0.1 mm maximum paired root-position change
over the shared valid prefix. Report these ingredients separately. This is an engineering
sanity check, not a causal failure taxonomy or evidence of physical-contact-free traversal.
No force/impulse sensor is installed: physical contact remains **not measured**.

If the condition does not block WALK, stop and inspect the geometry/instrumentation rather than
moving the beam or changing thresholds on these outcomes. If it does, prepare a separately
frozen ducking correction pilot with matched no-beam qualification and meaningful repair arms.
Do not claim that raw-WALK blocking proves a corrected duck can pass.

## Resources, budget and records

One launch / eight assigned rollouts / zero new generations, plus the already paid A1 controls.
Use the user-authorized 5500 MiB VRAM / 9500 MiB RAM launch gates and 1200/2500 MiB running
abort floors; log co-tenants, never terminate them. Same single-owner lock and 900 s timeout.
Execute from clean committed source into fresh `outputs/astra_a1b_target_beam_v1`, outside that
checkout. Preserve every receipt and failed attempt; resume only verified completed execution.
No changes to earlier A1 results or endpoint meanings.
