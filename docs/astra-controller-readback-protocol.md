# ASTRA controller-mode readback validation

**Status: preregistered** — 2026-09-05, before instrumented operational replay.

## Question and scope

Which encoder mask actually reaches SONIC's policy in each environment? The pinned
configuration exposes multiple encoders but historical A4 exports do not record active
masks. Configuration weights alone cannot resolve the executed interface. This is an
instrumentation test, not a new method comparison, calibration cohort, or quality claim.

## Fixed inputs and bounded budget

Replay archived A4 `fixed3_absent`, then `fixed3_present`: two launches of 32 slots,
**64 operational executions; zero generated references or independent scientific samples**.
Reuse `outputs/astra_a4_adaptive_d0/fixed3.pkl`, all eight original carriers, four slots
each, order, physics seed 0, 208 reference frames at 25 Hz, frozen checkpoint, isolated
origins, 1.15 m beam, physics and termination settings. Only callback target and output
location change. Historical arrays, labels, thresholds and receipts remain immutable.

Use a clean committed clone, fresh external output and the existing exclusive ASTRA lock.
Preparation/verification workers exit before Isaac. Preserve the validated compact malloc
profile, 9216 MiB available-RAM/5500 MiB free-VRAM admission, 2500/1200 MiB running
floors, 900 s launch timeout and at most 180 s admission wait. No failed-launch retry,
batch-size reduction, precision change, co-tenant termination or concurrent training.

## Readback and alignment

The new subclass reads the flattened tokenizer observation already supplied to the policy
immediately before `env_step`, slices `encoder_index` using the runtime observation
manager's ordered dimension metadata, and archives its mask plus the live command mask
and command frame index. Record encoder names/weights and observation dimensions.
Allow legitimate multi-hot masks; reject malformed, nonfinite or empty masks.

Mask sample i precedes physics step i+1; achieved qpos sample i follows that step. Trim
both to the identical valid prefix. Do not read a reset-selected encoder after the step
and label it as the action-producing mask. No random calls, setters or mode overrides.

## Verification and stop rule

After each launch, bind controller readback, process and source receipts by hash. Validate
32 unique ordered motion keys, environment indices, mask dimensions, valid lengths and
sample timestep; report observed mode counts/transitions and observation/command agreement.
Require bit-exact valid achieved qpos, lengths and termination flags, identical apparatus
readback and timestep against the original same-condition A4 archive. Save any discrepancy
before failing verification, and stop the queue before the next launch. No tolerance tuning.

Only exact replay permits an instrumentation noninterference claim for these two batches.
Readback belongs to the new replay, not retroactively measured historical A4 execution.
Mode/outcome associations are descriptive, not a causal test. A mode override, new
controller or training intervention requires a separate prospective protocol and cohort.
