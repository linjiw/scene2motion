# Contact logging preserves A5 execution; terminal-contact attribution remains open

## Result and research decision

Six corrected simple-body controls passed, followed by64 matched G1 operational
replays with **exact achieved-state, termination/progress, apparatus and controller-mode
equality** to A5. This validates the read-only integration for the tested configuration;
it does not improve traversal performance or qualify high-quality demonstrations.

The32 present valid prefixes contain zero reported Table contacts across all30 live
robot-body filters. This includes17 operational local passes and15 failures. A5's
nominal/margin geometry also has no violations in this subset. Ten trials terminated;
absence of contact in their retained prefixes does not establish contact-free episodes.
All original A5 physical-contact labels remain unknown. New measurements belong to
linked replay records in the same eight development groups.

Importantly, the untrimmed reader reached6 used points despite zero retained-prefix
contacts. A follow-up event diagnostic was resource-aborted before output; those
points cannot yet be located in terminal versus later reset actions. A controlled G1
positive-collision probe has not been completed. Do not promote these zeros into a
general contact-free or safety claim.

## What was implemented and tested

`ContactReader` resolves explicit per-environment Table and articulation-body paths,
checks buffer dimensions/indices/capacity, and reads every0.005 s physics substep.
A scoped recorder wrapper preserves the original callback exactly once. Four substeps
are reduced per0.02 s action; compact arrays share the achieved-state valid-prefix mask.
Absent-obstacle records are not applicable, not fabricated zero measurements.

The installed PhysX API supplies **signed scalar normal forces**. Positive-probe
diagnostics exposed negative scalars with unit normals, invalidating the initial
nonnegative-force assumption. The reader now retains signed scalars/vectors and sums
absolute scalar force times dt for non-cancelling normal-impulse magnitude. This is
not frictional impulse or a unique-contact count.

The scripted positive body settles on the table; the separated negative body falls.
Logger-off/base64/doubled128 controls have exact state equality. Positive base/double
each report572 point observations and9.809936 N·s; negatives report zero. Capacity
summaries match exactly. Positive logger-off arrays also match across v1/v2/v3/v4,
despite earlier teardown failures. This does not establish G1 sensitivity to every
contact or validate all possible contact-buffer loads.

## Preserved attempt accounting

| Stage | Process launches | Outcome |
| --- | ---: | --- |
| Simple-body v1 | 1 | Saved states, then300 s teardown timeout; verifier interpreter also lacked NumPy |
| Simple-body v2 | 1 | Same states, teardown still hung; only owned worker manually stopped |
| Simple-body v3 | 2 | Off passed; logged probe failed before completed output |
| Signed-force diagnostic | 1 | Preserved offending buffers and exception; no qualification claim |
| Corrected simple-body v4 | 6 | All passed |
| G1 replay v1 | 1 | Output-directory startup defect; no rollout archive, present not launched |
| G1 replay v2 | 2 |32 absent +32 present, all exact |
| Excluded-event diagnostic | 1 | VRAM running-floor abort; no completed outcome |

All15 launches, including6 unsuccessful processes, cost655.703 s simulator-process
wall time. Earlier receipts/logs remain immutable. The first accounting snapshot is
retained; the v2 accounting additionally reruns the G1 verifiers rather than merely
summarizing their recorded status.

Successful simple-body controls cost42.708 s with2809.48 MiB sampled peak child RSS.
The64 G1 replays cost135.707 s with6101.09 MiB peak child RSS (5.96 GiB); minimum
available RAM9397 MiB, minimum free VRAM4584 MiB. These are shared-host observations,
not an isolated throughput or VRAM-allocation comparison.

The event diagnostic was stopped after27.765 s when sampled free VRAM reached1158 MiB,
below the1200 MiB running floor. Admission was unchanged, no batch-size adjustment or
automatic retry was made, and no co-tenant was stopped. No ASTRA simulator remains.

## Reproduce and next gate

Receipts: [paired replay](../outputs/astra_contact_replay_v2/summary.json),
[complete attempt accounting](../outputs/astra_contact_validation_v2/summary.json).
Sources were frozen at7ece77f (corrected probes), ac86eeb (G1 replay), and f95fa1c
(event diagnostic). Protocols and identities are bound in their output manifests.

```bash
source env.sh
LD_LIBRARY_PATH= "$S2M_PY" -m experiments.analyze_astra_contact_replay --out "$PWD/outputs/astra_contact_replay_v2"
LD_LIBRARY_PATH= "$S2M_PY" -m experiments.analyze_astra_contact_validation --outputs "$PWD/outputs" --out "$PWD/outputs/astra_contact_validation_v2/summary.json"
```

Both receipts rebuild exactly. Full CPU suite:1126 passed in242.15 s;52 focused tests
passed in1.16 s, including four event-scope tests added after full-suite collection. The research-design
review kept instrument validation, operational traversal and motion-quality claims
separate throughout.

Next: version a continuation for the aborted event diagnostic when shared-host gates
permit; explicitly test G1/Table positive-contact sensitivity before stronger contact
claims. Then freeze final-edited-motion qualification combining local passage,
post-dwell progress/recovery, contact coverage and motion dynamics. Only after that
test one progress-aware correction on fresh grouped carriers against fixed2/fixed3
and equal-budget controls. No additional mode sweep, spent-seed tuning, training,
or comprehensive-dataset claim is justified by this instrumentation milestone.
