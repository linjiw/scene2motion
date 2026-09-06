# Contact prefix-exclusion diagnostic

**Status: preregistered**. Commit before launch.

The corrected A5 replay at ac86eeb exactly reproduced64 states/mode records. All32
present valid prefixes have zero contact observations, but the live reader recorded
peak occupancy6. Do not interpret prefix zeros as entire-episode contact freedom.

Run one further present-only32-slot operational replay in fresh external
`outputs/astra_contact_event_v1`, using the identical fixed3/default source references,
seed0, checkpoint, tracker, terrain and all physics settings. No new generated motion,
scientific units or method tuning. Use the same lean supervisor, compact allocator,
RAM/VRAM admission9216/5500 MiB, running floors2500/1200 MiB,900 s timeout and one lock.

The only change is a subclass retaining sparse nonzero contact summaries from the
already-read per-action buffer. Do not read physics twice or change valid-prefix files.
Record action/env indices, body names, counts and impulse. Label events before the
valid length, at the first excluded action, or later; later events may follow reset
and must not be attributed to the original motion's achieved-state prefix.

Require exact A5 states, apparatus, modes and the existing contact schema checks.
Require nonempty event evidence, matched membership and no valid-prefix event (the
previous observation). Stop on disagreement and preserve it; no automatic retry.
This locates previously dropped contacts. It is not a controlled G1 positive-collision
probe, new traversal result, quality certification or retroactive A5 label change.
