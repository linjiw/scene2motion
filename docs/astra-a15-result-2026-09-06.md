# A15: controlled correction ties the fixed shallow comparator

The common nominal-context comparison completes on all 12 preassigned scene/u0
development units. Frozen A12 full correction and fixed 0.26 m shallow both qualify
4/12, versus 2/12 nominal. Each gains s10/u0 and s11/u0 and preserves the nominal
s07/u0 and s08/u0 passes. Full correction has zero wins and zero losses against
fixed shallow: **the preregistered promotion gate fails**. A12 is not retuned.

| Method | Raw candidate passes | Final-choice passes | Descriptive final-choice Wilson95 |
| --- | ---: | ---: | --- |
| Nominal | 2/12 | 2/12 | 4.7–44.8% |
| Fixed shallow, preserve nominal | 4/12 | 4/12 | 13.8–60.9% |
| Frozen full correction, preserve nominal | 4/12 | 4/12 | 13.8–60.9% |

These are obstacle-absent geometry/progress descriptors on reused development
units, not physical traversal or held-out improvement. The 12 scenes cover the
existing geometry grid; no new beam geometry, multi-beam corridor or stepping
trial is evaluated. The 512 simulator rows are not 512 independent trials.

![Controlled A15 comparison](../outputs/astra_a15_figures_v2/controlled_correction.png)

## Clearance improvement can still lose forward continuation

At s00/u0, the frozen full correction increases depth 0.28→0.30 m and advances
onset lead 1.08→1.17 m together. Inflated clearance changes from −3.49 to +14.94 mm,
but next-second continuation falls from 0.597 to 0.426 m, below the unchanged
0.45 m requirement. Exit remains on time (2.76 s). The fixed shallow candidate
continues 0.633 m but has −14.33 mm inflated clearance. None qualifies.

At s01/u0, full correction improves inflated clearance from −26.37 to −16.81 mm;
the margin violation persists. Continuation is 0.687 m. Neither distinctive
full-correction branch adds a pass. These comparisons change depth and onset
together and do not identify either component as the cause.

| Component, all 12 assigned | Nominal | Fixed shallow | Full correction |
| --- | ---: | ---: | ---: |
| Reference descriptor passes | 6 | 10 | 9 |
| Witnessed rear-body exit | 8 | 10 | 10 |
| Exit within frozen deadline | 5 | 7 | 7 |
| Complete one-second continuation suffix | 8 | 10 | 10 |
| Continuation at least 0.45 m | 8 | 9 | 8 |
| Upright corridor recovery | 7 | 10 | 10 |
| Nominal and inflated clearance qualified | 7 | 7 | 8 |
| Exit censored | 4 | 2 | 2 |
| Tracker terminated | 4 | 1 | 1 |

Component counts overlap; only their original conjunction supplies qualification.
All failed/censored units remain in the denominator. No contact-force conclusion
is drawn from the geometry columns. The reference row is a separate kinematic
measurement and must not be read as executed performance.

## Control, reuse and cost

Both nominal controls reproduce every original A12 valid trajectory and
all archive metadata exactly: 64/64 slots. Every candidate retains its original
slot, encoder realization, frozen controller and 31 nominal background references.
Independent payload readback checks every field of all 32 references for all 26
compiled requests, including aliases. All 16 jobs pass apparatus/mode verification.

Before outcomes, complete-payload hashes identify 10 aliases: eight corrections
equal their shallow candidate and two preserve the nominal input. Only s00/u0
and s01/u0 have full corrections distinct from both comparators. This branch
coverage limits what this pilot says about the full rule.

- 16 unique jobs with 32 environments: 512 operational executions; 36 assigned
  scientific previews, 26 distinct execution/target measurements, 10 aliases.
- 486 rows supply context workload rather than new candidate measurements.
- Zero new generations, independent units, present evaluations or held-out tests.
- Simulator-job elapsed sum 807.123 s; peak child RSS 6096.01 MiB; minimum available
  RAM 4605 MiB. No resource refusal, timeout, runtime abort or spent-job retry.
- Full CPU suite: 1303 passed in 206.84 s; focused harness/compiler tests: 24 passed.
- Independent audit verifies all 64 nominal prefixes/metadata, all compiled payload
  fields, selection/counts, slot modes, source hashes and operational accounting.
  Reanalysis from the frozen checkout reproduces the preserved summary exactly.

## Artifacts and next decision

Frozen preparation/execution checkout: `/tmp/s2m-a15-88fcc45`.
[Protocol](astra-a15-controlled-correction-protocol.md),
[campaign manifest](../outputs/astra_a15_controlled_correction_v1/identity.json),
[results](../outputs/astra_a15_controlled_correction_v1/summary.json),
[independent audit](../outputs/astra_a15_completion_v1/summary.json),
[figure receipt](../outputs/astra_a15_figures_v2/receipt.json),
[final validation](../outputs/astra_a15_validation_v1/summary.json).

Do not promote the full correction to remaining-seed validation on this tie.
The [A16 design](astra-a16-depth-phase-ablation-design.md) instead tests the
already frozen depth-only ablation in the same nominal context, preserving all 12
assignments and the current failure. It is diagnostic development, not a new
confirmation split. Positive G1 contact sensitivity, present passage, independent
beam geometries, multi-beam traversal, stepping and downstream utility remain open.
