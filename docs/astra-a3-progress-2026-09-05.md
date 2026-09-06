# A3 hold-time probe: generation complete, execution paused by host resources

Frozen execution source: `3e92cd3`. [Protocol](astra-a3-hold-protocol.md),
[completion ledger](../outputs/astra_a3_progress_v1/index.json),
[generation receipt](../outputs/astra_a3_hold_d0/generation_complete.json).

## Completed work, not a method result yet

Generated all **32 assigned references** with eight fresh seeds, one reused 1.15 m beam
geometry and four arms. Fixed dip depth is 0.28 m; hold2/3/4 differ only in nominal hold
time. Each constrained arm has **8/8 reference-geometry passes**; free nominal has **0/8**.
These are reference labels, not executed crossing rates. Generation/decode totals **2.689 s**,
excluding model loading, specification preparation and CPU verification.

Completed `free_nominal_absent` and `free_nominal_present`: **64 actual executions**, all
apparatus checks pass. The six constrained-motion batches (**192 executions**) have not
launched. No hold-arm execution result or timing-performance comparison exists yet.
The 256-entry ledger retains all planned assignments; unlaunched outcomes remain unknown,
not failures. All variants share eight development groups; no test or demonstration labels.

Two preflight refusals were preserved: available RAM 9434 MiB before free-present (later
resumed successfully), and 8698 MiB before hold2-absent. Subsequent host observation remained
below the frozen 9500 MiB launch gate. No threshold was lowered; no generated seed or
completed execution was repeated. Other campaigns were not interrupted.

The two completed SONIC subprocesses took **121.66 s**, peak launch VRAM **3791 MiB**,
minimum available RAM **3316 MiB**. No running resource abort or timeout occurred. Whole-card
utilization includes co-tenants and is not this campaign's throughput measurement.

## Why this experiment precedes adaptive repair

The [A2r absent-only overlap audit](../outputs/astra_overlap_windows_v1/index.json) finds
73/128 observed forward exits; 55 do not have an observed complete exit. Only 59/128 meet
the proposed <=4 s dual-window hold budget. Censoring and disconnected overlap intervals
are explicit. These are post-hoc, unexecuted proposals, not calibrated execution bounds.

Varying hold time alone establishes whether early recovery is a useful intervention and
provides a fixed-long-hold baseline for future adaptive scheduling. Do not simultaneously
tune depth, move the beam, relax posture/deadline gates, or train a model from two old
positive crossings. An improvement must preserve absent progress as well as present
clearance. Sparse positive geometry labels do not establish high-quality motion.

## Exact continuation

Use the existing clean immutable clone `/tmp/astra-a3-hvxxHR/repo` at `3e92cd3`. If it is
lost, recreate a clean clone at that exact commit; do not resume using the latest source
identity. Output remains external to the execution clone. **Do not run generation again.**

```bash
cd /tmp/astra-a3-hvxxHR/repo
source env.sh
env LD_LIBRARY_PATH= "$S2M_PY" -m experiments.astra_a3_hold \
  --out /home/linjiw/scene2motion/outputs/astra_a3_hold_d0 --stage sonic
env LD_LIBRARY_PATH= "$S2M_PY" -m experiments.astra_a3_hold \
  --out /home/linjiw/scene2motion/outputs/astra_a3_hold_d0 --stage analyze
```

The driver rechecks completed batches and resumes only missing launches under the same
identity. It will refuse again if RAM/VRAM are insufficient. After all eight batches finish,
build the grouped, signed execution index from the main repository:

```bash
source env.sh
env LD_LIBRARY_PATH= "$S2M_PY" -m experiments.analyze_astra_a3_results
env LD_LIBRARY_PATH= "$S2M_PY" -m experiments.analyze_astra_a3_results --check
```

The full index builder deliberately refuses partial campaigns. Progress snapshots are
separate, immutable bookkeeping artifacts. Physical contact and full motion quality stay
unmeasured; preserve A1's endpoint and low-pelvis proxy semantics. Report paired hold3/4
wins/losses against hold2 and all eight units, not just the best post-hoc arm.

Validation: **993 CPU tests passed in 210.57 s**, plus the subsequently added progress-ledger
test passed. New tests cover unchanged root paths, matched slots, timing bounds, censoring,
receipt tampering, immediate apparatus-stop behavior, grouped records and unknown labels.
The research-design skill informed the single-variable hypothesis and evidence limits;
no positive execution result is inferred from that guidance. No ASTRA process remains running
at this resource-paused handoff; the next action is to resume unlaunched batches, not redesign
the method before seeing their results.
