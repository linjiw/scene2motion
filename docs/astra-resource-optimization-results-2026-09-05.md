# Lower-memory execution with exact control reproduction

**Later completion update:** A3 has now finished all 256 scientific executions; the paused
96/256 state below is historical. See [A3 results](astra-a3-results-2026-09-05.md). The six
lean batches complete without resource abort; peak child RSS6069 MiB. No further gate change.

## Measured result

The [operational probe](astra-resource-probe-protocol.md) completed **128 replays**, comparing
default and compact allocation in absent/present conditions. All four 32-slot batches
reproduce archived A3 **qpos archives and apparatus JSON byte-for-byte**; ordered valid
states, lengths, termination flags and timestep also match. These are operational controls,
not 128 new independent research samples. No precision, terrain, policy, timestep, source
reference, environment count, slot ordering or endpoint was changed.

| Condition | Default child peak RSS | Compact child peak RSS | Default wall | Compact wall |
| --- | ---: | ---: | ---: | ---: |
| Absent | 6144 MiB | 6042 MiB | 61.19 s | 61.81 s |
| Present | 6213 MiB | 6052 MiB | 60.78 s | 62.27 s |

The old A3 importer/loaded-input worker peaks at **704692 KiB (~688 MiB)**. The new resident
supervisor samples **15.7–16.2 MiB RSS** during simulation; its separately reported process
high-watermark is conservatively **113–115 MiB** in the probe receipts. Do not confuse a
sampled resident value with an exact peak. Phase separation removes substantial idle parent
overhead; compact allocation additionally saves **102–161 MiB** inside Isaac. Its measured
wall time increases ~1–2.5%, so this is a memory/throughput tradeoff, not a speedup claim.

The two process-local settings are `MALLOC_ARENA_MAX=2` and
`MALLOC_TRIM_THRESHOLD_=131072`, as defined by the
[glibc allocator documentation](https://sourceware.org/glibc/manual/latest/html_node/Memory-Allocation-Tunables.html).
No system-wide setting or other process was changed. The first preparation had a resolved-
path defect caught before launch; `astra_resource_probe_v1` is preserved with zero executions.
The successful probe source is `3ffd1c9`; its outputs and exact comparisons are in
`outputs/astra_resource_probe_v2/`. The measured profile receipt is
[`outputs/astra_resource_profiles_v1/index.json`](../outputs/astra_resource_profiles_v1/index.json).

## Engineering changes

- `lean_supervisor.py`: standard-library-only resident process, source/input hash checks,
  bounded resource waits, RAM/VRAM monitoring, fail-closed telemetry, own-process-only abort,
  timeout escalation, immutable receipts. Preparation and validation workers exit before
  the next simulator launch.
- `lean_queue.py`: serial pending jobs, validate each batch immediately, resume completed
  jobs by verification rather than replay. A failed or interrupted launch is never retried.
- `resource_profiles.py`: select only an exactly validated profile for the matching workload
  and environment count. Unknown resources or no fitting profile means wait/refuse; it
  does not silently change batch sizes or omit assigned trials.

The compact workload-specific RAM admission is **9216 MiB (9 GiB)**: worst measured child
peak + worst reported supervisor peak +2500 MiB reserve +512 MiB uncertainty, rounded up to
256 MiB. Keep free VRAM admission 5500 MiB, running floors 1200/2500 MiB and a 900 s timeout.
This does not modify historical A3's 9500 MiB rule; a separate
[continuation protocol](astra-a3-lean-continuation-protocol.md) binds the new profile.
It is not a guarantee under arbitrary co-tenant pressure. Isaac's fixed memory cost remains;
earlier 2/16/32-env measurements do not support proportional RAM savings from smaller batches.

## Applied to the experiment

The optimized continuation completed **hold2-absent: 32 executions**, all apparatus checks
pass, child peak RSS 6037 MiB, supervisor reported peak 16.1 MiB, minimum available RAM
4376 MiB, wall time 61.56 s. No resource abort occurred.

Shared-host memory then fell below the new gate. After the frozen 180 s wait,
hold2-present remained unlaunched and the queue exited. No gate was lowered further and
no other job was stopped. A3 totals now are **96/256 scientific executions** (64 original
free controls +32 new), with **160 pending**. The
[joined progress ledger](../outputs/astra_a3_lean_progress_v1/index.json) retains every assignment
with unknown outcomes where unlaunched. There is still no complete hold-time performance
comparison. The full joined analyzer deliberately refuses an incomplete campaign.

## Resume and verify

Use the frozen continuation clone at `d6f18f7`, not a newly changed source identity:

```bash
cd /tmp/astra-a3-lean-zbs5Cx/repo
source env.sh
env LD_LIBRARY_PATH= /usr/bin/python3 -m scene2motion.lean_queue \
  --manifest /home/linjiw/scene2motion/outputs/astra_a3_lean_d0/manifest.json \
  --wait-seconds 180
```

No preparation or generation is needed again. After all six continuation batches complete,
run from the main repository:

```bash
source env.sh
env LD_LIBRARY_PATH= "$S2M_PY" -m experiments.analyze_astra_a3_lean \
  --out outputs/astra_a3_lean_d0
env LD_LIBRARY_PATH= "$S2M_PY" -m experiments.analyze_astra_a3_lean \
  --out outputs/astra_a3_lean_d0 --check
```

Further large savings need another measured equivalence study, not a smaller arbitrary gate.
One audited lead is that the release checkpoint includes optimizer/value/training state and
the evaluation entrypoint imports the PPO trainer. An inference-only loader could avoid some
unused work, but checkpoint compatibility and exact state/RNG behavior must be tested before
deployment. That change is **not implemented or claimed here**.

Validation: **1004 CPU tests passed in 211.33 s**, plus the subsequently added joined-progress
test passed. Tests cover lightweight imports, exact comparison sensitivity, hash tampering,
fail-closed telemetry, pending-job preservation, queue resume without replay, and profile
workload/batch matching. The resource profile and joined progress ledger both rebuild exactly.
No ASTRA process remains running at handoff; unrelated worktree changes are preserved.
