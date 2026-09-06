# ASTRA low-memory execution equivalence probe

**Status: preregistered** — 2026-09-05, before operational replays.

User requests lower memory/compute without changing the experiment. Do not lower A3's
frozen gate or alter its reference, timestep, terrain, precision, environment count, order,
seed, endpoint or controller. Smaller batches can change random draws and GPU arithmetic;
they are not an automatically equivalent continuation.

## Intervention and budget

Preparation/validation run in short-lived workers. The resident supervisor uses only the
Python standard library and the lightweight host gate, avoiding the measured ~688 MiB
research-stack importer. Isaac commands and 32 environment slots remain unchanged.

Compare default allocation with process-local `MALLOC_ARENA_MAX=2` and
`MALLOC_TRIM_THRESHOLD_=131072`. No global environment/system changes, cache dropping,
precision changes, thread-count changes, co-tenant termination, or terrain simplification.
These [glibc allocation controls](https://sourceware.org/glibc/manual/latest/html_node/Memory-Allocation-Tunables.html)
limit arenas and change when releasable heap memory is returned; savings and runtime must
be measured, not assumed. Record exact environment differences and pinned sources.

Replay archived A3 free-nominal pickle, same keys/slots/physics seed 0: baseline absent,
compact absent, baseline present, compact present. Four launches ×32 = **128 operational
replays**, zero generated references. They are equivalence/resource checks, never additional
independent scientific samples. Fresh external output directory, clean clone, single ASTRA
lock. Each launch uses the original 5500/9500 MiB admission gate; running floors 1200/2500
MiB, 900 s timeout, fail closed on missing telemetry. No failed-launch retry.

## Verification and deployment gate

Immediately verify all apparatus poses, extents, origin separation and archived-state hashes
after each launch. Compare ordered valid qpos arrays, lengths, termination flags, timestep
and apparatus readbacks against archived A3 controls and between allocation profiles.
Require exact state equality for an unchanged-result claim; otherwise report differences,
do not merge into A3, and investigate nondeterminism separately. No endpoint tolerance tuning.

Measure child kernel peak RSS, sampled RSS, supervisor RSS, host minimum available memory
and wall time. Host memory delta is confounded by co-tenants; prefer attributable process
measurements for the comparison. Reject allocator deployment if either condition differs or
memory/runtime tradeoffs do not help. A later, explicitly versioned resource profile may be
calibrated from successful measurements plus reserve; no automatic global gate relaxation.

Resource-aware scheduling must choose only validated profiles for the exact workload,
wait when none fit, retain pending jobs, and never silently reduce assigned trials. A3
continuation under a new profile requires a separate provenance-linked continuation record.
