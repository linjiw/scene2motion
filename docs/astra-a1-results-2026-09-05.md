# ASTRA A1/A1b: reachable controls and a verified target beam

## Result and decision

**The operational apparatus gate passed.** Neutral WALK can reach the local recovery region;
the raised beam is inert on the archived achieved trajectories; lowering it to the target
height prevents local completion on all seven previously qualified controls. Proceed to a
separately frozen native-control ducking correction pilot. This is not yet a repair success.

| Development condition | Local completion | 95% Wilson interval | Operational decision |
| --- | --- | --- | --- |
| EXP-023 WALK, absent and raised separately | 6/8 in each arm | 0.409–0.929 | Below 7/8; use declared reserve |
| EXP-023b WALK, absent and raised separately | 7/8 in each arm | 0.529–0.978 | Qualifies apparatus progression |
| EXP-023b WALK, target beam underside 1.05 m | 0/8 | 0–0.324 | Blocking check passes |

These are descriptive development proportions over fixed historical controls, not independent
scene-generalization estimates. The seven previously qualified controls all fail at target height
(7/7; Wilson 0.646–1.000); s4645 remains in the all-eight target denominator.

## What was actually measured

- Five SONIC launches, eight environments each, physics seed 0: **40 new executed rollouts**,
  zero new generator calls and zero training runs.
- A1 used source commit `8c07e00`; A1b used `f2d60e3`. Every arm used the same patched tracker
  at `7c63c539a17008f5efb1e768034c0fb434ae1f65`, released checkpoint, evaluator and robot XML.
- All readback checks passed: live per-motion pose, first/end spawned USD extents and neighboring
  environment separation. Physical contact forces/impulses remain **not instrumented**.
- Within each A1 cohort the absent/raised achieved qpos arrays are elementwise identical. This
  supports raised-beam inertness for these runs, not general simulator equivalence.
- s4502 and s4645 failed their absent controls by tracker cutoff. s4503 eventually reached the
  root recovery criterion at 5.18 s but failed the frozen 4.0 s deadline; its failure was retained.
- With the target beam, all eight achieved trajectories enter nominal and 4 cm inflated geometry.
  Six have tracker cutoffs, two remain uncut but stalled; no fall flag occurs in the archived
  valid prefixes. The qualified controls' shared-prefix maximum paired root differences range
  from about 0.287 to 6.285 m. Large values include accumulated stalling, not instantaneous impact.
- Local completion requires all robot collision primitives, including feet with coverage
  inflation, beyond the beam far edge plus the fixed upright recovery dwell. Full-route and
  root-only v2 fields remain separate; this does not certify floor/slip/dynamic motion quality.

## Resource decision, with actual costs

The user authorized replacing the conservative 18 GiB preflight requirement with the existing
measured SONIC preset. The versioned protocol records 5500 MiB free VRAM / 9500 MiB available
RAM at launch, co-tenant logging, and 1200/2500 MiB VRAM/RAM running abort floors. Older
preflight refusals are preserved. No co-tenant was stopped and no running abort was triggered.

All five launches peaked at **3645 MiB launch VRAM**. Their minimum available host RAM was
6024–10094 MiB; total SONIC subprocess wall time was **273.95 s**, excluding CPU preparation,
analysis and source freezing. A shared-GPU observation reached 96% utilization, but that includes
co-tenants and is not a per-method throughput measurement. Eight environments matched each
assigned cohort; duplicated padding was not used to manufacture more evidence.

## Records and reproduction

- [A1 first cohort](../outputs/astra_a1_apparatus_v1_2/cohort0/summary.json)
- [A1 qualified reserve](../outputs/astra_a1_apparatus_v1_2/cohort1/summary.json)
- [Target-beam intervention](../outputs/astra_a1b_target_beam_v1/summary.json)
- [Grouped execution index](../outputs/astra_execution_index_v1/index.json)

The rebuild-checked index has 40 executions in 16 carrier groups: 16 obstacle-absent tracking
records and 24 obstacle-present records. Its 26 local-pass records are **easy WALK controls**, not
26 ducking demonstrations. Every group stays in development; contact and full motion-quality
labels are unknown. This index supplements, but does not silently alter, the older 300-record
DB preview or establish learner utility.

Validation: **969 repository CPU tests passed in 214.24 s**; the execution index independently
rebuilds exactly and `git diff --check` passes. No experiment process remains running.

```bash
source env.sh
env LD_LIBRARY_PATH= "$S2M_PY" experiments/build_astra_execution_index.py --check
env LD_LIBRARY_PATH= "$S2M_PY" -m pytest tests/test_astra_apparatus.py tests/test_astra_execution_index.py -q
```

## Next experiment

Implement A2 with fresh generation seeds and paired absent/target-beam execution for uncorrected,
fixed extra crouch, measured correction and equal-budget resampling. Test native height-control
transmission and motion quality before claiming correction benefit. Use up to 32 useful parallel
environments for an eight-unit/four-arm matrix if the measured resource gate permits; bind batch
layout before either paired arm. Do not enlarge the training program merely to occupy the GPU:
the next missing evidence is an executed corrected duck, followed by a comparative result.
