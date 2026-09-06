# A3: resource-equivalent continuation of unlaunched hold arms

**Status: preregistered** — 2026-09-05, before any hold-arm execution. Conditional on all
four operational resource-probe controls reproducing archived qpos, lengths, cutoffs,
timestamps and apparatus exactly. If this condition fails, do not launch.

The user requested resource optimization, not a scientific protocol change. Preserve A3's
original references, eight units, four slots, seed, terrain, controller, constraints and
endpoints. Reuse the 64 original completed free controls. Execute only the six still-unlaunched
hold2/3/4 absent/present batches: **192 new executions**, zero generations. Preserve the
original paused directory and link it by hashes; use a fresh external continuation directory.

Use the measured compact allocator and lightweight supervisor, with workers exiting between
preparation, simulation and validation. No dynamic batch resizing or precision changes.
Profile RAM admission is the rounded-up (256 MiB) sum of maximum measured child peak RSS,
maximum reported supervisor peak RSS, 2500 MiB running reserve, and 512 MiB uncertainty buffer.
Use the worse of absent/present resource probes. This is a conservative workload-specific
measurement, not a universal bound. Keep free VRAM admission 5500 MiB, running floors
1200/2500 MiB, 900 s timeout and fail-closed telemetry. Old A3 receipts/gates are unchanged;
the continuation explicitly binds the new measured profile. If no fit, wait up to 180 s per
job and leave it unlaunched. Do not stop co-tenants or claim uninterrupted availability.

Preparation requires all resource-probe artifacts hash-match their calibration receipt,
all control comparisons are exact, original free executions verify, and no hold launch has
already started. Freeze every job before the queue runs. Validate each completed batch's
apparatus before the next launch. Resume verifies completed jobs and never retries failures.

Final analysis joins 64 original controls with 192 continuation executions under an explicit
versioned receipt, not silent replacement. Operational replays are excluded from the scientific
denominator. Preserve unknown physical contact/quality and all-assigned counts. Resource
equivalence on controls supports this execution path; it does not promise determinism for
arbitrary hardware, environments, or future workloads.
