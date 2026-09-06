# ASTRA A1 implementation and preflight record

## Outcome

The local apparatus pilot is implemented and frozen, but **no SONIC subprocess or rollout was
started**. Apparatus qualification and WALK execution qualification remain unmeasured. The next
action is the same paired absent/raised-beam test, not a ducking comparison.

## Delivered and checked

- `401acc1`: research plan, 16-control preparation manifest, preregistered A1 protocol,
  apparatus callback, qualification scorer, bounded/resumable driver and tests.
- `2b28dd4`: pre-execution process-classification fix. The two installed Isaac documentation
  MCP servers are recorded but no longer mistaken for simulation engines. Actual simulator
  processes and unknown Isaac entrypoints remain blocking.
- First/last valid live table pose, complete valid pose trace, USD extents and neighboring
  environment positions are now exportable. This instrumentation has **CPU tests only**;
  successful Isaac runtime readback is still an A1 acceptance requirement.
- The new endpoint requires all robot collision primitives (including feet, with 4 cm coverage
  inflation) beyond the beam far edge and a full elapsed 0.20 s upright recovery dwell.
  Hazards through that entire prefix invalidate completion, even if root-only v2 completed
  earlier. Historical evaluator code and receipts were not changed.
- Full CPU regression at `401acc1`: **956 passed in 205.76 s**. Subsequent focused tests
  including documentation-service discrimination and reset-prefix trimming: **26 passed**.
  The reference-only scorer smoke check is not an execution result. `git diff --check` passed.

## Preserved attempts

| Output | Execution source | Observed preflight | Spent rollouts |
| --- | --- | --- | --- |
| `outputs/astra_a1_apparatus_v1` | `401acc1` | 17744 MiB available RAM; two documentation services falsely identified as Isaac | 0 |
| `outputs/astra_a1_apparatus_v1_1` | `2b28dd4` | first 17884, then 18234 MiB available RAM; no simulator process | 0 |

The first refusal survives as its input/identity files and the protocol's amendment record;
the amended harness also writes timestamped machine-readable refusal reports. The frozen RAM
gate is **18432 MiB (18 GiB)**, VRAM gate **12288 MiB (12 GiB)**. A separate read-only check
briefly observed 18435 MiB RAM, but the actual launch preflight dropped below threshold again.
No threshold was reduced; no co-tenant was stopped. There is no background campaign poller.

## Resume exactly this pilot

The clean execution checkout is currently:
`/tmp/scene2motion-astra-a1-20260905-hNQwO3/repo-v1-1`, pinned at `2b28dd4`.
It contains only committed sources, not the other session's pending probe/report edits.
Outputs are stored persistently in the primary repository, outside the execution checkout.

```bash
cd /tmp/scene2motion-astra-a1-20260905-hNQwO3/repo-v1-1
source env.sh
env LD_LIBRARY_PATH= "$S2M_PY" experiments/astra_a1_apparatus.py \
  --out /home/linjiw/scene2motion/outputs/astra_a1_apparatus_v1_1 --run --resume
```

Run only when the host satisfies the frozen gates. If the temporary checkout is unavailable,
recreate a clean checkout of `2b28dd4`; do not substitute a later HEAD for this frozen attempt.
The driver verifies source, archive, checkpoint, XML and protocol identities before execution.
Preflight refusals may resume; spent failed/partial launches require diagnosis, not blind retries.

Success would qualify only the operational apparatus. A blocking-beam control, motion-quality
qualification and constructive ducking/stepping tests still need their own prospective evidence.
