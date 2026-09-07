# A10: smaller native edits across the complete development scene grid

**Status: preregistered**. September6,2026. Commit protocol, collection and analysis
before generation. Authorized by the user's request to advance the next research
stage and validate performance in simulation. A8 remains failed; A9's64 exact same-slot
replays motivate this response-scale experiment, not confirmation or an optimizer gain.

## Question and fixed scope

Do half-size native edits reduce nonlinear response while retaining useful clearance
and progress transmission, across all12 A6 development scenes and48 motion units?
The complete grid is beam x=(1.2,1.6,2.0) m × underside=(1.10,1.20) m × longitudinal
length=(0.24,0.60) m. Width2.25 m, thickness0.25 m. Every scene has four existing
development seeds64000–64047 in the A6 ordering. Reusing these units is explicit;
descendants retain `astra-a6/sXX/uY` split groups. No untouched confirmation unit is
used, and this study cannot establish transfer to all motions, scenes or multi-beams.

Keep frozen ARDY-G1 Horizon52, v2 noise, WALK prompt,208 frames,25 Hz,5 diffusion steps,
CFG[2,2],0.9 m/s straight route, original batch8 shape and unit/variant ordering.
Only native root-height constraints change; no post-generation qpos deformation.
Native baseline theta=(0.28,1.08,1.38) m and quintic ramps0.45/0.90 m remain fixed.
Six half-step edits are ±(0.02,0.09,0.09) m on one coordinate at a time. Include unchanged
baseline and free-WALK as the first/last members of every generation batch, as in A6.
All original-scale A6 observations are immutable controls, not newly generated samples.

## Budget, pairing and controls

12 scenes×4 units×8 variants =384 generations:288 new half-step candidates plus96
no-edit controls. After each8-row batch, preserve the arrays/receipt and require both
no-edit qpos arrays to equal their original A6 arrays exactly. A mismatch stops further
generation without a tolerance, retry or deleting the failed batch. Checkpoints,
generator/runtime/physical identities must equal A6; source code/receipts bind A10.

Use the A6 corrected32-slot layout. Group0 has32 scientific units from scenes0–7;
group1 has16 scientific units from scenes8–11 and16 labelled padding duplicates.
Same unit occupies the same slot across all variants, matching original A6 default
encoder masks. Per group run baseline and free-WALK controls first, then depth−/+,
onset−/+ and recovery−/+. All128 control execution slots, including padding, must
match original qpos and termination exactly; mode/apparatus/clock checks stay fixed.
Stop on any control mismatch before the group's edited requests. This is measurement
comparability, not a new traversal endpoint.

Total16 launches×32 =512 absent previews:384 scientific and128 padding. All costs
count; padding is never independent data or a training target. Freeze SONIC release,
patched tracker identity, physics seed0, default weights[1,1,1], compact32 resource
profile and apparatus. Zero present evaluation, compound command, new physics seed,
controller-mode sweep or downstream training in A10.

## Measurements and prospective analysis

Use unchanged A6 whole-body inflated clearance, rear exit and one-second continuation,
with original observation/censoring statuses and development descriptor. Refusals,
timeouts and incomplete suffixes stay in all-assigned accounting. No physical-contact
label or prospective quality qualification is inferred from absent geometry replay.

Compare full/half scale on the same unit groups. For each axis/target, retain only
triplets where baseline, minus and plus targets are independently observed at both
scales and in both reference/achieved measurements. Report every omitted count.
Report absolute midpoint departure `abs((plus+minus)/2-baseline)` and response magnitude
`abs((plus-minus)/2)` separately; reduced effects alone are not an improvement. Report
all48-unit status transitions for each scale. Bootstrap half-minus-full scene-mean
midpoint differences with10000 resamples, seed69000; never fill unobserved scenes
with zero. These reused development data do not constitute untouched confirmation.

Repeat the frozen A8 four-model comparison, feature map, ridge10 and12 scene-exclusion
folds on half-step data. Normalize physical changes by(0.02,0.09,0.09), so each axis
edit has magnitude1 just as in A8. Implementation may explicitly map deep-copied
model-input commands into A8's canonical feature coordinates; output receipts must
retain actual physical commands and normalized step units. Do not change observations,
generator requests or historical A8 code/results. Predict no-edit exactly from the
initial measurement. Missing initial targets remain unavailable. No predictor sees
edited references/previews, scene identifiers, seeds or slot identifiers.

Keep A8's readiness checks: execution-phase model beats no-change, constant response
and scene/reference response on scene-mean clearance and continuation MAE; exit MAE
no more than5% above no-change; continuation sign correct on more than half observed
pairs; status Brier better than no-change; at least12 complete initial previews.
Publish all288 edited-assignment statuses, per-target matched counts and per-scene
errors/uncertainty. No feature/penalty search or endpoint switching inside A10.
Per-variant descriptor counts and the seven-command outcome-informed ceiling describe
candidate supply, not a deployable selection method or obstacle-present traversal.

If readiness fails, preserve the failure and do not expand the optimizer under A10.
If it passes, prepare a separate bounded actual-regeneration/present-comparison protocol
with strong geometry, fixed-duration and budget-matched search controls, nested error
allowances and progress ablation. A10 is response identification, not that final study.

## Provenance, resources and stopping

Fresh external `outputs/astra_a10_half_step_v1`, from a new clean pinned checkout.
Keep all previous archives/checkouts unchanged. CPU summary uses fresh
`outputs/astra_a10_analysis_v1`. Generation admission RAM8192/VRAM4096 MiB, rechecked
between scenes. Simulator admission RAM9216/VRAM5500 MiB, runtime floors2500/1200 MiB,
900 s timeout and single ASTRA lock, with co-tenants recorded and never interrupted.
Generation runs in an exited subprocess before simulator preparation/launch. Immediate
admission checks only, no poller. Preserve refusals and every spent attempt; resume
only untouched work or verified completed stages. The output-local
`STOP_AFTER_CURRENT_JOB` sentinel stops between simulator jobs.
