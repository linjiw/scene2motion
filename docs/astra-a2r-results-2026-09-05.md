# A2r: isolated duck execution produces two local crossings, not robust repair

Execution source `5733fd3`; [preregistered protocol](astra-a2r-isolated-protocol.md),
[full summary](../outputs/astra_a2r_isolated_d0_r1/summary.json),
[grouped execution records](../outputs/astra_a2r_execution_records_v1/index.json).

## Outcome

All **224 assigned executions** completed and all seven batches passed live apparatus
checks. Unique terrain origins and matched environment slots repair the original
[A2 harness failure](astra-a2-harness-result-2026-09-05.md). The old 136-rollout attempt
remains quarantined; do not pool it with this study.

| Method | Absent local progress | Present local crossing | Units passing >=3/4 slots |
| --- | ---: | ---: | ---: |
| Initial native dip | 6/32 | 0/32 | 0/8 |
| Fixed extra 0.10 m dip | 3/32 | **2/32** | 0/8 |
| Best-of-three | 16/32 | 0/32 | 0/8 |
| Measured correction | alias of initial | alias of initial | 0/8 |

Free-height controls complete absent local progress in **12/32**. Each method has eight
development seed–scene units and four initialization/terrain slots per unit; one physics
seed, two geometries. Slots are repeated measurements, not 32 independent scenes.

The two present passes are `u4_r0` and `u4_r2`, beam underside **1.15 m**, fixed-extra
command dip 0.28 m. Whole-body recovery+dwell finishes at **3.44/3.50 s**. Their achieved
whole-valid-prefix conservative clearances are **25.0/22.6 mm**, after the existing 40 mm
geometry-coverage inflation. Both matched absent runs also pass. Neither present run
cuts off, but neither completes the full route. Physical contact was not instrumented;
do not call these contact-free, safe or fully quality-qualified demonstrations.

This is a constructive **native-control existence result**, achieved by a fixed heuristic,
not evidence that feedback repair outperforms a baseline. All initial references already
meet the reference target, so the measured operator makes zero active updates and its
execution is explicitly shared with initial. There is no independent measured-arm success
rate or correction-gain estimate. No unit achieves the descriptive 3/4-slot repeatability
level; robust performance remains unfinished.

## What the next method must address

1. **Carrier quality:** new free-height generation only yields 12/32 local progress. Native
   depth changes can further damage progress; qualifying archived WALK does not qualify
   newly generated constrained motion. Count the bank construction cost and refusals.
2. **Execution timing:** among 69/96 observed non-free, absent root crossings, actual root
   arrival at x=1.2 m is a median **1.14 s later** than reference arrival. Medians by arm:
   initial 1.20 s (26 observed), fixed extra 1.70 s (19), resampling 0.62 s (24). Nonarrivals
   are censored, not zero lag. This motivates a timing experiment, not a shared delay constant
   or causal proof. Root arrival is also not the complete body-overlap window.
3. **Execution clearance:** selecting the geometrically clearest reference does not provide
   traversal. A reference-only update is blind when reference slack passes but achieved slack
   fails. Use paid, matched absent execution to identify per-carrier timing and clearance
   response, on development data, before freezing a held-out method.
4. **Ducking posture semantics:** 11/32 fixed-extra absent runs trip the frozen pelvis-height
   proxy while their minimum torso-up component remains >=0.7. Do not interpret that flag as
   an instrumented fall. Preserve this experiment's endpoint; a future duck-specific quality
   protocol needs validated contact/posture criteria before fresh samples, not a relaxed
   threshold applied to these seeds.

## Next bounded work (not yet preregistered)

Develop a **dual-window crouch hold**: cover both reference-body overlap and a predicted
achieved-body overlap, rather than merely shifting the event and breaking reference clearance.
Compare against a development-tuned fixed long hold and fixed deeper dip under identical
carrier information, generation calls, absence calibration cost and simulator slots.
Only extend or reschedule within explicit smoothness/posture limits; refuse if absent
execution loses progress. Use the current records for development, then freeze fresh
carrier/scene units for the method comparison. Do not add training without a defined
consumer and sufficiently many quality-qualified positive records.

## Cost, data and checks

- A2r: **390.58 s** SONIC subprocess wall time, peak launch VRAM **3791 MiB**, minimum
  available host RAM **4679 MiB**. No resource abort. No new generation or training.
- This work session: **56 generated references; 360 actual executions** = 136 quarantined
  +224 isolated. Generation/decode 4.414 s excludes model loading and CPU scoring; combined
  SONIC wall time 680.99 s excludes preparation/analysis. Do not label all 360 valid evidence.
- New valid index: 224 records, eight grouped development units; 128 absent and 96 present.
  It retains failures, signed geometry, timing, provenance, unknown contact and quality.
  The 64 measured aliases are not additional executions. Old DB preview remains unchanged.
- Full CPU regression: **983 passed in 210.17 s**. Record validators reject missing groups,
  alias inflation and promotion of unknown contact/quality. Raw logs retain original whitespace
  because their hashes are part of the receipts.

Rebuild-check the two separate indices from the repository root:

```bash
source env.sh
env LD_LIBRARY_PATH= "$S2M_PY" -m experiments.analyze_astra_a2_development --check
env LD_LIBRARY_PATH= "$S2M_PY" -m experiments.analyze_astra_a2r_results --check
```

The research-design skill informed the bounded hypothesis, explicit aliases/costs, matched
controls and restricted claims; the outcomes above come only from archived execution records.
