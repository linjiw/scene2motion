# A4: absent-only timing adaptation does not improve the fixed baseline

## Completed comparison

All **32 references and 256 assigned executions** are complete under the frozen
[A4 protocol](astra-a4-adaptive-hold-protocol.md), execution source `84e649e`.
Eight fresh motion seeds (62000–62007), four matched environment-slot repeats per seed,
one physics seed (0), and one reused beam geometry (underside 1.15 m). Calibration
decisions were frozen after the initial absent runs and before any present launch.
Every apparatus check passes; no generation or execution was retried.

| Arm | Absent local progress /32 | Present local passage /32 | Units with >=3/4 present passes /8 |
| --- | ---: | ---: | ---: |
| Free nominal | 26 | 0 | 0 |
| Fixed 3 s | 17 | 19 | 4 |
| Fixed 2 s | 21 | 18 | 5 |
| Adaptive | 14 | 16 | 3 |

Adaptive has **zero matched gains and three losses** against fixed3: 50.0% versus
59.4%, a 9.4-percentage-point decrease. Fixed2 has eight gains and nine losses;
its higher >=3/4-unit count must remain visible beside its lower slot-level count.
The slots are repeated measurements, not 32 independent scenes. This is a fresh-motion
development test, not geometry generalization or statistical significance.

The unchanged endpoint requires whole-body passage, root >=1.82 m, posture/corridor
proxies, and 0.20 s dwell within 4 s, with 40 mm coverage-inflated clearance preserved
through the local prefix. **All present arms have zero full-route completions.** Actual
contact forces and full demonstration quality remain unmeasured.

## Where the adaptive rule lost

Only fixed3 absent outcomes informed the proposals. Four of eight units qualified;
the other four retained fixed3. Counts below are present local passes out of four.

| Unit | Adaptive hold (s) | Timing edit qualified | Fixed3 | Adaptive |
| --- | ---: | --- | ---: | ---: |
| a4u0 | 2.44 | yes | 4 | 4 |
| a4u1 | 3.06 | yes | 4 | 4 |
| a4u2 | 3.06 | yes | 4 | 1 |
| a4u3 | 3.00 | no | 2 | 2 |
| a4u4 | 3.00 | no | 1 | 1 |
| a4u5 | 3.00 | no | 0 | 0 |
| a4u6 | 2.32 | yes | 4 | 4 |
| a4u7 | 3.00 | no | 0 | 0 |

All three losses are a4u2 repeats 1–3. They have no archived nominal penetration,
coverage-margin violation, posture-failure flag, or tracker cutoff. Their root-only
kinematic dwell diagnostic occurs at 4.24, 4.06, and 4.30 s; none meets the frozen
whole-body deadline. These later diagnostics do **not** replace the primary failures.
Absent local progress for this unit also falls from 3/4 to 0/4 after regeneration.
Its original calibration therefore did not preserve the edited motion's progress.
The observed problem is recovery timing, not insufficient geometric clearance.

The two shorter-hold units retain 4/4 local passes, but shorter control duration is
not evidence of better motion quality or lower end-to-end cost. The four fallback
reference arrays reproduce fixed3 exactly, and all 16 fallback present local labels
agree. Execution identity is not guaranteed: a4u4 repeat2 cuts off in adaptive but
not fixed3, despite identical reference arrays; both fail local passage. Do not assign
that cutoff difference to a timing edit. Absent/present local labels agree in 30/32
slots for both fixed3 and adaptive, not universally.

## Cost, records, and reproducibility

- [Complete receipt and grouped records](../outputs/astra_a4_adaptive_d0/summary.json):
  256 outcomes, reference/achieved archive links and timestamps, signed clearance,
  independent event flags, all decisions, source hashes, and costs. All variants of a
  source motion share `astra-a4/a4uN`; this is a development corpus, not an IID/OOD split.
- Total: 32 actual generations and 256 executions, including 64 initial absent controls.
  The adaptive rule specifically consumes 32 fixed3 calibration executions. Standalone
  fixed3 costs eight generations plus 32 present executions; adaptive costs 16 generations
  and 96 executions including calibration and its final absent/present checks. Shared
  experiment information does not erase this deployment overhead.
- Eight simulator batches total **465.18 s**, excluding generation and CPU analysis.
  Maximum child RSS **6101 MiB**, supervisor **16.86 MiB**; minimum available host RAM
  **8893 MiB**. No resource abort. Keep the validated 32-slot compact allocation profile
  and its workload-specific 9216/5500 MiB RAM/VRAM admission gates unchanged.
- The earlier resource refusal and zero-execution progress snapshot remain historical
  records. They do not describe the completed campaign.
- CPU regression suite: **1018 passed in 224.90 s**. The endpoint receipt rebuilds
  exactly, including calibration, apparatus checks, and all outcomes from the archives.
  No ASTRA simulation or analysis process remains running.

```bash
source env.sh
env LD_LIBRARY_PATH= "$S2M_PY" -m experiments.analyze_astra_a4 \
  --out outputs/astra_a4_adaptive_d0 --check
```

## Research decision

The one-step overlap rule fails its advancement criterion. Keep fixed3 as the primary
fixed comparator and fixed2's complementary successes; do not retune A4, extend its
deadline, or market adaptive as an improvement. No training run is justified by this result.

The next task in [astra.md](../astra.md) is an archive-based progress/recovery and motion-quality
audit, followed by a bounded method revision only if it addresses the measured failure.
It must assess the **edited** carrier, not inherit qualification from a different
generation. Contact instrumentation and geometry-disjoint evaluation remain necessary
before a high-quality demonstration release. These records support that work; they do
not yet constitute a complete humanoid-navigation dataset.
