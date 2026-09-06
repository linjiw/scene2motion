# A5: does local passage preserve useful progress?

## Result and decision

The CPU audit covers **all 24 references and 576 archived executions** without new
generation, simulation, selection or changes to A5 labels. Default fixed3's 17 present
local successes remain a median **1.201 m behind their references at 4 s**. Nevertheless,
all 17 advance during the one-second window after their recorded whole-body dwell,
by a median **0.592 m** (range 0.272–0.736 m). Thus neither “local passage implies good
tracking” nor “every local success immediately stalls” describes these measurements.

Keep default fixed3/fixed2 controls. Qualify the **final edited motion**, including
world-frame progress and recovery, before expanding a correction policy. Do not extend
the losing A4 hold rule, relax the spent cohorts' deadlines, or train from a proxy quality
label. Actual obstacle contact and high-quality demonstration qualification remain unknown.

## Evidence and measurement

- [Immutable audit](../outputs/astra_a5_quality_audit_v1/index.json), including source and
  analysis hashes, per-reference diagnostics and all 576 per-execution records.
- [Original campaign](astra-a5-results-2026-09-05.md) and
  [source receipt](../outputs/astra_a5_encoder_d0/summary.json) retain their original endpoints.
- [Analyzer](../experiments/analyze_astra_a5_quality.py) reuses the tested A4 quality helpers.

Reference and achieved poses are compared on their exact shared timestamps, accounting
for achieved samples occurring after each control step. Missing suffixes after cutoff
are not extrapolated. Root error is achieved minus reference world-x at 4 s. The
post-dwell window starts at the stored dwell endpoint, not at a retrospectively chosen
favorable frame. Joint derivatives retain their native sampling intervals; first-four-second
and full-valid-prefix diagnostics are distinct. Foot-pad depth relative to z=0 is a
geometric proxy, **not** terrain-relative penetration, measured contact or stance slip.

## Default fixed3, obstacle present

| Diagnostic | Local pass | Local fail |
| --- | --- | --- |
| Assigned executions | 17 | 15 |
| Observed / missing root error at 4 s | 17 / 0 | 6 / 9 |
| Median root error at 4 s | −1.201 m | −1.898 m |
| Median root speed, 3–4 s | 0.618 m/s | 0.198 m/s (6 observed) |
| Observed positive post-dwell one-second progress | 17 / 17 | No recorded dwell |
| Median joint RMSE, valid common prefix | 0.272 rad | 0.176 rad |

The smaller joint error among failures is not a causal result or a quality ranking:
the compared prefixes differ, and early termination censors difficult suffixes. It
shows why joint RMSE alone must not become a demonstration-admission label.
Among default fixed3 local passes, maximum first-four-second foot-pad depth below
the zero plane reaches 20.1 mm; this calls for terrain/contact measurement, not a
retroactive physical-penetration claim or an arbitrary new threshold.

The corresponding G1 and teleop local-pass groups also all advance in the following
second: medians 0.600 m (17 executions) and 0.598 m (15). These descriptive checks do
not change A5's no-overall-gain-over-default finding. All full-route counts remain zero.
There are eight motion-seed groups and one reused beam, **not 576 independent scenes**.

## Reproduction and next gate

```bash
source env.sh
LD_LIBRARY_PATH= "$S2M_PY" -m experiments.analyze_astra_a5_quality \
  --out outputs/astra_a5_quality_audit_v1/index.json --check
```

The analyzer loads one achieved archive at a time and caches only the 24 references.
Reuse of archived executions avoids another GPU campaign; no equivalence claim for
a changed simulator batch size is needed. The next instrumentation step is specified
in the [draft contact-validation plan](astra-contact-validation-plan.md). Its CPU
buffer tests are implemented; live simulator validation is not yet complete.

Validation: **1093 CPU tests passed in 224.47 s**; the focused quality/contact set
passed30 tests. The complete24-reference/576-execution audit rebuilt exactly.
