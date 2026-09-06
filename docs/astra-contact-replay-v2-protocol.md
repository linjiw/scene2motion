# A5 contact replay v2: startup directory fix

**Status: preregistered**. Commit before preparation and execution.

Inherit all identities, gates, comparisons, budgets and interpretation limits from
[v1](astra-contact-replay-protocol.md). Preserve `outputs/astra_contact_replay_v1`:
its absent batch failed before rollout because the callback wrote topology before
the parent's end-of-evaluation output-directory creation. No achieved records were
produced; the present arm was never launched. The failed startup consumed18.234 s
and sampled5522.03 MiB child RSS, with no resource abort.

The sole fix creates the callback output directory before the exclusive topology
write. Existing topology still refuses overwrite; add a CPU startup regression.
No motion, physics, threshold, seed, sensor or recording scope changes.

Use fresh external `outputs/astra_contact_replay_v2`, absent then present,32 slots
each: at most64 operational replay executions. Keep the v1 failed startup in total
cost accounting, and retain all original development groups. Stop on any failure;
no automatic retries. No new independent scientific samples or training.
