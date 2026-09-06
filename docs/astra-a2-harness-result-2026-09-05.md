# A2 development attempt: references complete, mixed-slot execution invalid

Source: `bf22d54`, [protocol](astra-a2-native-duck-protocol.md),
[raw summary](../outputs/astra_a2_native_duck_d0/summary.json),
[quarantined grouped records](../outputs/astra_a2_development_records_v1/index.json).

## What ran

56/56 reference generations, 136/136 physical rollouts, no training. Generation/decode
wall time 4.414 s (excludes loading/scoring); SONIC subprocess wall time 290.415 s.
One RAM preflight refusal occurred before the first 32-environment launch; resumption
did not repeat any spent launch. CPU regression: 975 passed in 212.25 s.

All initial/fixed/repair/resampling references clear their assigned beam in the reference
geometry; none of the eight free-height controls does. Initial overhead slack is already
above the 0.02 m target: **zero active correction updates**. All 16 generated repair
no-op arrays exactly reproduce initial arrays. This cannot establish feedback benefit.

## Why the execution comparison is not valid

The rough-terrain importer assigns random terrain levels; several environments reuse
world origins. `env_spacing=12` does not override that assignment. Seven present-arm
readbacks across the two physics seeds fail `neighbor_beam_in_route_envelope`; own beam
poses and extents match their requests. The original driver enforced this check too late,
at final analysis rather than between launches. Both defects are harness errors.

The initial/measured arrays are identical but all 32 cross-slot achieved comparisons
differ, showing why same seed alone does not make unequal environment slots a matched
method intervention. Two individual present records meet local conditions with passing
own readbacks (`u4_fixed_extra`, `u6_measured`, physics seed 1), but **the campaign is
not promoted to a valid method comparison or robust closure**. No arm passes both seeds
for the same unit. Keep all successes, failures and invalid records; do not cherry-pick
the passing slots. Contact and high-quality-motion qualification remain unmeasured.

## Repair, not silent rerun

[A2r](astra-a2r-isolated-protocol.md) explicitly changes physics initialization: unique
terrain patches separated by 16 m, fixed slot mapping reused across methods, immediate
apparatus gate after each launch. It reuses frozen references and selectors without new
generation or threshold changes. Initial/measured reuse is a declared alias, not another
independent trial. Results remain development data from two geometries. Old receipts and
the original simulator baseline stay unchanged.
