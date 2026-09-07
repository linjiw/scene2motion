# A16: restoring nominal onset recovers one qualified development passage

The frozen depth-only ablation qualifies **5/12** assigned development units,
versus **4/12** for fixed shallow and full correction and **2/12** nominal.
It gains s00/u0 without losing any prior pass. The preregistered development
lead gate passes. This is one additional obstacle-absent descriptor pass on reused
development data; it does not establish physical traversal or unseen-scene gain.

| Method | Raw passes | Final-choice passes | Descriptive final-choice Wilson95 |
| --- | ---: | ---: | --- |
| Nominal | 2/12 | 2/12 | 4.7–44.8% |
| Fixed shallow, preserve nominal | 4/12 | 4/12 | 13.8–60.9% |
| Full correction, preserve nominal | 4/12 | 4/12 | 13.8–60.9% |
| Depth only, preserve nominal | 5/12 | 5/12 | 19.3–68.0% |

Depth only gains three units over nominal, and one over each comparator, with
zero paired losses. The broad intervals and reuse of the A15 development units
preclude a significance or confirmation claim. A15's full-correction promotion
failure remains unchanged.

![A16 comparison](../outputs/astra_a16_figures_v1/depth_phase.png)

## The measured clearance–progress tradeoff

At s00/u0, both full and depth-only commands use depth 0.30 m and recovery
offset 1.38 m. Restoring onset lead from 1.17 to 1.08 m changes the executed result:

| Achieved measurement | Full correction | Depth only |
| --- | ---: | ---: |
| Inflated whole-body clearance | +14.94 mm | +10.65 mm |
| Rear-body exit | 2.76 s | 2.44 s |
| Following-second continuation | 0.426 m | 0.501 m |
| Original geometry/progress descriptor | Fail | Pass |

The ablation gives up 4.29 mm of inflated clearance while recovering 75.23 mm of
continuation and exiting 0.32 s earlier. It retains positive clearance and exceeds
the original 0.45 m continuation threshold. Nominal and fixed shallow still fail
their margin checks. No threshold or safety margin changes.

At s01/u0, depth-only clearance remains −16.50 mm versus −16.81 mm full correction;
continuation is 0.680 versus 0.687 m. Both fail. The result supports dropping the
onset advance in one controlled case, not a universal timing rule or a complete
depth-by-onset interaction model. No new command was generated or fitted.

Across all 12, depth only has eight clearance-qualified measurements, nine with
at least 0.45 m continuation, ten complete one-second suffixes, two censored exits
and one tracker termination. These overlapping components do not replace the
five conjunction passes. All failed and censored units remain assigned.

## Execution context, cost and validation

The simulator retains A15's frozen import path, checkpoint, callbacks, mode masks,
physics seed and all other declared execution fields. Each target replaces only
its original slot in the original nominal 32-reference background. Input and
output filenames are the only changed simulator arguments. Preparation revalidates
all A15 results and its 64 exact nominal replay controls.

- Two new jobs with 32 environments: 64 new operational rollouts, two new scientific
  previews, and 62 context-workload rows. No new generations or independent units.
- Ten of 12 no-phase measurements reuse byte-identical complete contexts; the
  three comparator arms reuse 36 A15 measurements. These are 48 arm/unit
  measurements, not 48 new executions. A15's 512 rollouts are not counted again.
- Simulator-job elapsed sum 135.945 s; peak child RSS 6072.875 MiB; minimum available
  RAM 6861 MiB. No resource refusal, timeout, runtime abort or spent-job retry.
- Full CPU suite: 1315 passed in 206.23 s; focused harness/cache/compiler: 30 passed.
- Independent audit checks all fields of 32 references in each of 12 compiled contexts,
  original runtime/source pins, slot modes and receipt hashes. All 12 achieved
  measurements reconstruct exactly from raw valid state prefixes. All 36 inherited
  comparator records remain unchanged. Frozen analysis rebuild is exact.
- PNG/PDF figures were generated from audited receipts and visually inspected.

Frozen preparation checkout: `/tmp/s2m-a16-cc4d44e`. Simulator runtime remains
`/tmp/s2m-a15-88fcc45`; preserve both. [Protocol](astra-a16-depth-phase-ablation-protocol.md),
[manifest](../outputs/astra_a16_depth_phase_v1/identity.json),
[results](../outputs/astra_a16_depth_phase_v1/summary.json),
[independent audit](../outputs/astra_a16_completion_v1/summary.json),
[figure receipt](../outputs/astra_a16_figures_v1/receipt.json),
[final validation](../outputs/astra_a16_validation_v1/summary.json).

## Next validation

The [A17 design](astra-a17-remaining-development-design.md) retains u1–u3 from
each existing scene. Its input-only inventory identifies 37 possible new jobs,
but only one remaining unit has a depth-only input distinct from nominal and
fixed shallow. A proposed two-job discriminating gate can avoid a larger campaign
when strict improvement is already impossible; complete absolute performance
still requires the remaining executions. This design assigns no simulation.

The remaining units are reused development seeds, not untouched confirmation.
Controlled positive G1 contact sensitivity and paired obstacle-present passage
remain necessary, followed by independent beam geometries, multi-beam corridors,
stepping and downstream utility. A16 is a bounded positive lead toward that work.
