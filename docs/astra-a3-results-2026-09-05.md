# A3: a moderate crouch hold improves local passage in this development cohort

## Completed experiment

All **256 assigned executions** are complete: 64 original free controls and 192 explicitly
joined lean-continuation executions. Eight motion seeds, four matched environment slots per
seed, one physics seed (0), one reused beam geometry (underside 1.15 m). The 32 references
were generated once; the earlier 128 allocator-control replays are excluded from scientific
counts. Every batch passes the isolated-apparatus checks. No execution was retried.

| Native height control | Absent local progress /32 | Present local passage /32 | Units with >=3/4 present passes /8 |
| --- | ---: | ---: | ---: |
| Free nominal | 8 | 0 | 0 |
| 0.28 m dip, 2 s hold | 13 | 13 | 3 |
| Same dip, 3 s hold | 19 | 19 | 4 |
| Same dip, 4 s hold | 11 | 11 | 2 |

The 3 s arm gains **six matched passes and loses zero** against 2 s; 4 s gains zero and
loses two. These slots are repeated measurements, not 32 independent scenes. Three-second
successes cover six of eight motion seeds; per-unit counts are 0,3,4,0,4,4,2,2.
This is a development comparison, not statistical significance or geometry generalization.

The frozen endpoint requires the whole body beyond the beam, root >=1.82 m, corridor and
posture proxies, 0.20 s recovery dwell within 4 s, and preserved 40 mm coverage-inflated
geometry through that prefix. All arms have **zero full-route completions** with the beam
present. Physical contact forces and full demonstration quality remain **not measured**.

## What changed, and what did not

For successful local prefixes, the 3 s arm has 41.7–117.4 mm additional geometric clearance
(median 55.9 mm), after the 40 mm coverage inflation. Dwell ends at 2.84–3.96 s. This is
not a calibrated execution-error bound; it is conditional on the 19 successful prefixes.

Longer is not uniformly better. Four-second holds have 26/32 observed whole-body forward
exits in absent runs, but only 11/32 meet the full local endpoint. Three-second holds have
27/32 exits and 19/32 local passes. Preserve unobserved exits as censored/nonarrival.
No A3 valid prefix has the upright-below-0.5 m flag seen in the earlier A2r cohort.

For 3 s and 4 s, absent-progress and present-local pass labels agree in every matched slot;
for 2 s they agree in 28/32. Thus the result supports investigating **edited-carrier progress
and recovery timing**, not claiming that extra clearance alone caused all six gains. Native
conditioning changes the generated trajectory, not only a deterministic achieved-state timer.
Do not compare A2r's 2/32 directly with A3's 19/32 as a method gain: seeds and geometry
assignment differ. The current winner is still a fixed heuristic, not measured-feedback repair.

## Records and resource cost

- [Joined endpoint receipt](../outputs/astra_a3_lean_d0/summary.json): all 256 outcomes,
  paired contrasts, immutable source hashes and launch costs.
- [Grouped execution index](../outputs/astra_a3_execution_records_v1/index.json): all arms,
  absence/presence and slot repeats stay together in eight development groups. Contains
  reference/achieved clearance, valid prefixes, censored overlap, cutoff and posture flags;
  contact and quality labels remain unknown. No held-out split is claimed.
- Continuation source: `d6f18f7`, original source: `3e92cd3`. Six continuation launches take
  352.60 s total, peak child RSS 6069 MiB; total scientific SONIC wall time is 474.26 s,
  excluding operational replays, generation and analysis. No resource abort occurred.
- This turn resumed only the remaining 160 assignments. The lightweight queue retained
  the validated 9216 MiB RAM /5500 MiB free-VRAM admission gates and 32-slot workload.

The full CPU test run was initially stopped to free memory when a co-tenant began training;
it was restarted after our simulation completed. No co-tenant was interrupted.
Validation: **1006 CPU tests passed in 245.44 s**; the joined endpoint receipt and grouped
execution index both rebuild exactly. No ASTRA simulation or analysis process remains running.

## Research decision

Freeze **3 s as the strong fixed baseline** for the next study; keep all three A3 timing
arms and failures in the record. Do not extend holds beyond the tested budget or retune A3.
See the next-stage work plan in [astra.md](../astra.md): qualify edited carriers using
absent-only evidence, develop bounded per-carrier scheduling, and compare it with fixed 3 s
on fresh assignments under equal calibration/generation budgets. Full quality and contact
instrumentation are prerequisites to publishing these passes as high-quality demonstrations.

Rebuild with `experiments.analyze_astra_a3_lean --out outputs/astra_a3_lean_d0 --check` and
`experiments.analyze_astra_a3_results --continuation outputs/astra_a3_lean_d0 --check`
using `$S2M_PY` after sourcing `env.sh`.
