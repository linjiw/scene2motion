# A18 controlled G1 contact sensitivity

**Status: preregistered**. September 6, 2026 (local time). This is a bounded
instrument exercise, not a traversal experiment. No ARDY or SONIC policy call.

## Actual asset and fixed placements

Consume `outputs/astra_a18_collider_inventory_v2`: read-only USD inspection found
30 rigid bodies and45 colliders. The converted source is
`/tmp/IsaacLab/usd_20260906_223418_6311/main.usd`, created during A17's no-phase
launch window. Pin its used layers and the flattened USD; the asset is not edited
in place. All descendant geometry comes from actual converted USD, not MuJoCo.
The inventory alone has zero physics steps. A fresh probe instantiates this full
G1 articulation32 times on a4 m grid, root z=1 m, zero joint targets, zero actuator
commands, gravity disabled and no ground. This explicitly differs from traversal.

Select the greatest +x support point (path breaks ties) among capsule/sphere
colliders on left_ankle_roll_link, left_knee_link and torso_link. The resulting
colliders are mesh_3/capsule, mesh_0/capsule and mesh_3/capsule respectively. A25 mm
radius kinematic sphere is the test obstacle, with body-pair sensor identity
`Table` as in the existing reader. It is an instrument stimulus, not a beam task.
For slot i, target index=i%3 and level=(i//3)%2. Positive nominal surface gaps are
−2 mm and−10 mm; separated gaps are+50 mm and+100 mm. Center equals the frozen
support point plus(radius+gap,0,1) in the local environment. Validate live
collider transforms to1e-5 m before stepping. Existing collision/contact offsets
are retained; authored offsets (including unavailable defaults) are recorded.

## Six-job ceiling and checks

Execute positive/off, positive/base, positive/double, then negative/off,
negative/base, negative/double. Each job has32 environments,200 physics steps
at0.005 s; four-step aggregation at0.02 s. Physics seed0. Contact reader capacity
4096/base and8192/double; off skips the reader but retains the same contact-enabled
asset. Save full30-body states and joint positions every physics step.

Positive: every assigned target body must have a reported point and positive
integrated normal-impulse magnitude in the200-step observation. Negative: no
reported points on any of the30 filtered robot–probe pairs. Point presence and
impulse magnitude are distinct. Report each of32 assignments and all detected
body pairs, preserving event times through raw substep records. Base/double
contact records must agree exactly after removing only declared buffer capacity.
No active buffer saturation is permitted. Off/base/double full state and joint
arrays must match exactly; failures are measured observer effects, not relaxed
tolerances. Aggregates must rebuild exactly from every physics step.

Stop on the first failed launch or verification gate; preserve it and close the
exercise with the observed coverage and outstanding measurement gap. Do not
retry spent launches or turn this into a capacity sweep. A corrected harness
requires a separately versioned protocol beside the original failure.

Maximum6 jobs/192 operational instrument exposures; zero generated motions,
independent traversal units or passage outcomes. Use the ASTRA exclusive lock,
RAM9216 MiB/VRAM5500 MiB admission, runtime floors2500/1200 MiB,900 s/job timeout,
no admission polling. Numerical validation exits before the next simulator.
Do not interrupt co-tenants. Commit clean preparation sources before launch.
Preserve resource refusals; they spend no exposure budget.

## Claim boundary and next artifact

A positive result supplies a detection table and a recording contract for the
three tested collider placements. It does not validate every overhead-contact
surface, even when all30 body filters are present. Say “no detected contact on
the instrumented robot–obstacle pairs” wherever broader coverage is incomplete.
Keep nominal geometric penetration, inflated-margin violation, detected points,
impulse and obstructed traversal separate. Initial-overlap impulse is not a
calibration of collision severity during normal passage.

A19 obstacle-absent candidate coverage may proceed independently. Contact-sensitive
present claims depend on this instrument's demonstrated scope. Independent audit
must reconstruct body mapping, per-slot detection and logging-state comparisons;
focused tests precede launch and the full CPU suite precedes completion reporting.
