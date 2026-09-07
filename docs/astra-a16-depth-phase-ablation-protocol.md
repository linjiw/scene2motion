# A16: frozen depth-only ablation in the A15 nominal context

**Status: preregistered** before preparation or simulation. September 6, 2026.

This implements the [A16 design](astra-a16-depth-phase-ablation-design.md).
A15 full correction ties fixed shallow at 4/12 versus nominal 2/12. That failed
promotion remains closed. This is a diagnostic follow-up on the same development
units, motivated by a clearance/continuation tradeoff; it is not confirmation.

## Fixed inputs and assignments

Use original A12 `outputs/astra_a12_progress_correction_v1/g{0,1}_no_phase.pkl`
targets, compiled into that campaign's original `baseline` background through
the unchanged canonical adapter. Reuse all 12 A15 scene/u0 assignments: seeds
64000+4*scene index; group0 slots 0,4,8,12,16,20,24,28 for scenes s00–s07;
group1 slots 0,4,8,12 for s08–s11. Keep original encoder masks and physics seed0.
The manifest binds all source rows, reference hashes and cache owners before launch.

No new generation, refitting, command tuning, controller change, present test,
held-out scene or new free-WALK arm. Verified A12/A10 free-nominal and no-edit
constructibility controls remain inherited source controls. The method's original
depth/onset/recovery bounds and baseline-preserving selection rule stay fixed.

For s00/u0 and s01/u0, the full command used depth0.30 m, onset lead1.17 m,
recovery offset1.38 m. The no-phase command preserves depth and recovery and
restores onset lead1.08 m. This comparison isolates the incremental onset edit
at the deeper depth; it does not establish a complete factorial interaction.

## Complete-context cache contract

Before preparation, run the frozen A15 analysis worker and require exact rebuild
of its preserved results and all apparatus/mode checks, including the64 original
nominal replay controls. Validate all source hashes and clean frozen git states.

The simulator keeps **`/tmp/s2m-a15-88fcc45` in its PYTHONPATH**, together with the
identical SONIC checkout, checkpoint, callbacks and remaining environment. New
jobs copy the A15 nominal job of their group. Only motion-file and output paths
change. A16 preparation provenance adds pins without removing or altering old
pins; its new checkout is not inserted into the simulator's import path.
Run the A16 script by its absolute frozen path from the A15 runtime directory,
after sourcing that directory's `env.sh`, with `LD_LIBRARY_PATH=` and
`OPENBLAS_NUM_THREADS=1`, matching the A15 launch. This also retains A15's
`S2M_ROOT` and other environment setup; do not source A16's environment instead.

Cache matching requires the complete canonical32-slot payload SHA256, the same
group, normalized execution arguments (only motion/output locations normalized),
all other job fields, and exact original source/hash and git-state dictionaries.
Validate cached jobs before reuse. No target-only cache, cross-group alias,
changed environment variable, model source, physics argument or mode is allowed.
An identity failure stops the experiment; it does not authorize extra controls
or an enlarged budget on these units.

Prelaunch input inspection yields exactly ten full-payload aliases: s02–s06 and
s09–s11 reuse their A15 fixed-shallow jobs; s07 and s08 reuse their group nominal
jobs. Only s00 and s01 require new jobs. Preparation must reproduce this exact
two-new/ten-cached split or refuse before writing a launchable manifest.

## Budget, order and stops

Two new32-env jobs, s00 then s01: **64 new operational executions**, two distinct
new scientific previews and62 context-workload rows. All12 no-phase measurements
are assigned, including ten cached measurements. The three comparator arms reuse
36 A15 measurements. Report all48 arm/unit measurements while distinguishing
the64 new executions from A15's previously reported512; no independent units are
added. Cache aliases are not independent repeats.

Use fresh outputs outside the clean committed preparation checkout. Keep the
compact supervisor and ASTRA lock: admission RAM9216MiB/VRAM5500MiB, runtime
floors2500/1200MiB,900s/job and1800s total simulator-job budget. Preparation and
numerical verification workers exit before launches; the orchestrator imports
only the standard library. Preserve every refusal and launched failure, never
retry a spent job, stop on a failed apparatus/mode/identity check, and do not
interrupt co-tenants or create a resource poller. Unlaunched resource refusals may
resume under the same committed contract when host resources become available.

## Endpoint and decision

Use unchanged `scene_relative_duck.measure_motion` and
`progress_duck_correction.select`: keep a qualified nominal, otherwise choose
the arm's own qualified candidate, otherwise refuse. Qualification retains40mm
inflation, witnessed rear-body exit by rear_m/0.9+2.5s, a complete following
second with continuation at least0.45 m, and upright corridor recovery. Preserve
all non-arrivals, censored exits, terminations and failures in the denominator.
The A15 0.426 m continuation remains a failure.

Report raw and selected passes over all12, per-unit paired gains/losses,
descriptive Wilson95 intervals, reference and achieved clearance, rear exit and
continuation separately. A development lead requires no-phase selected passes
to strictly exceed both full correction and fixed shallow, with zero losses of
nominal passes. A tie/loss fails this gate. Do not iterate depth/timing offsets
on these units after a failure. A pass only motivates a separately preregistered
remaining-development-seed validation; it cannot retroactively promote A15,
establish significance or count as physical traversal or unseen-scene utility.
