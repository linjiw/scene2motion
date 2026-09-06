# A5: controlled encoder-interface comparison

**Status: preregistered** — 2026-09-05, before generation or execution.

## Question and contribution boundary

Does choosing G1 versus teleop change local passage and recovery for identical references
under the frozen release SONIC checkpoint? The default sampler remains a comparator.
The preceding readback study measured mixed masks, not a causal failure mechanism.
This simulation study supplies (i) a runtime-checked mode intervention, (ii) a paired
comparison across eight fresh motion seeds, and (iii) grouped execution records. It does
not introduce a new repair algorithm, prove quality, or estimate geometry generalization.

## Assignments, controls and budget

Use a5u0–a5u7, fresh ARDY seeds63000–63007. Generate three B=8 batches, in order:
free_nominal, fixed3, fixed2. Keep A4's cached WALK prompt, v2 noise,208 frames at25 Hz,
five diffusion steps, CFG[2,2], route7.488 m, speed0.9 m/s and native0.28 m dip with
quintic ramps. Fixed3/fixed2 hold ends are3/2 nominal seconds. Free nominal removes
root-height control. An outcome-free constructibility check requires finite, correctly
sized schedules with fixed2/fixed3 distinct and their root-XZ/heading identical.
Freeze all **24 reference arrays** before any controller execution; no selection or repair.

Every reference has four unit-major slot repeats at physics seed0 in32 isolated environments.
Execute each reference arm under default, g1 and teleop modes, absent then present:
24 references ×4 slots ×3 modes ×2 apparatus conditions = **at most576 executions**.
These are eight motion-seed groups, not96 independent motions or32 independent scenes.
All finite references stay assigned, even reference-margin failures. No resampling,
training, retiming, new text encoding, or extension of the budget after observing outcomes.

Beam remains center(1.2,0,1.275), size(.24,2.25,.25), underside1.15 m. This is one reused
geometry. Free walking is a negative control for unnecessary crouching and a reachability
control, not an already qualified high-quality carrier bank.

## Mode intervention and validity

Keep all encoder keys and pretrained modules. Override only command sampling weights,
ordered[g1,teleop,smpl]: default[1,1,1], g1[1,0,0], teleop[0,1,0]. Preserve checkpoint,
controller architecture, reference pickle/order, physics, initial-state rules and evaluator.
Use the readback callback validated at c7c864f. Every valid action must have the requested
one-hot mask in explicit arms, matching live command; resolved names/weights and contiguous
command steps must match the protocol. Default masks are measured, not assumed14/18.
Mode-conditioned GPU tensor shapes may differ; this tests the deployed mode configuration,
not an isolated neural representation independent of numerical batch effects.

Run nine absent batches first. Mode order for free_nominal: default,g1,teleop; fixed3:
g1,teleop,default; fixed2: teleop,default,g1. Freeze a reachability receipt before present
launches: each mode must have at least one free_nominal unit with >=3/4 absent local
progress passes. If any mode fails, stop the entire present stage, report absent results
and pending—not failed—present assignments. This minimal apparatus/interface gate is not
population carrier qualification. If it passes, execute all nine present batches, same
reference-arm order, reversing the within-arm mode order. No outcome-based per-unit removal.

## Endpoints and interpretation

Retain the A3/A4 whole-body local endpoint: rear coverage envelope>=1.32 m, root>=1.82 m,
upright pelvis>=0.5 m and up-z>=0.70 inside corridor |y|<=1.2,0.20 s dwell by4.0 s,
no nominal or40 mm inflated geometric hazard in the completed prefix. Score absent progress
with the existing virtual raised beam; target-beam replay on absent states remains separate.
Full-route7.2 m completion, cutoff, collision flags and times remain separate. Actual contact,
torque and high-quality demonstration qualification remain null, not inferred from geometry.

Primary contrast: fixed3 present g1 versus teleop, per-unit fraction of four successes
and number of units passing>=3/4. Report paired wins/losses and all eight unit differences;
no significance claim or interval on the paired difference at n=8. Report all-assigned /32
rates with descriptive Wilson intervals, explicitly not independent-slot population intervals;
also Wilson intervals over the8 unit repeat-success indicators. Secondary: each explicit
mode against default, fixed2/free controls, absent progress, full route, and censored root-x
at4 s (no last-value imputation after cutoff). No post hoc mode selector or “best of modes”
headline. A losing or null contrast is retained. A mode-specific result motivates fresh
final-output qualification; it does not explain every earlier A4 failure retrospectively.

## Integrity and resources

Clean committed clone, fresh external output, source/checkpoint/array hashes and single
ASTRA lock. Separate generation, preparation, simulation and verification workers.
Generation admission8192 MiB RAM/4096 MiB VRAM. Simulation uses compact malloc settings,
32 slots, admission9216/5500 MiB, running floors2500/1200 MiB,900 s timeout,180 s bounded
admission wait. Mode variants retain these conservative gates; savings are measured, not
assumed. Stop on malformed data, provenance drift, wrong runtime mode, invalid apparatus,
resource abort or failed process. Preserve failed attempts; no automatic retries or gate
relaxation. A resource-unlaunched batch stays pending and does not spend a scientific seed.

Record generation time, simulator time, peak process memory, all576 planned assignments,
mode traces and reference/achieved paths. All mode/hold/apparatus/slot variants of a5uN
share split_group_id astra-a5/a5uN and remain development data. No public redistribution
or positive imitation-dataset claim is made by this campaign.
