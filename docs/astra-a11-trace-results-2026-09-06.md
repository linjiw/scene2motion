# Body switching does not explain the failed local inverse

A11 reconstructs all 672 body-resolved reference/execution traces for the 336
native A10 candidates, retaining all 48 development units and 12 scenes. Every
scalar clearance and passage measurement matches the frozen A10 record. This is
post hoc analysis of existing simulation states: zero generation, new simulation,
independent units, or held-out evaluation.

[Design](astra-a11-trace-diagnostic-design.md) ·
[Source-bound results](../outputs/astra_a11_trace_diagnostic_v1/summary.json) ·
[Trace measurement](../scene2motion/duck_clearance_trace.py)

The initial reference minimum is on the head in 48/48 units. Of the 36 initial
executions with an observed minimum, 35 are head-limited with an overhead normal;
one is torso-limited. Twelve non-arrivals have no minimum. These 36 observed
minima do not mean 36 completed passages: the baseline has only 22 complete
passage/recovery observations.

| Half-step axis | Observed minimum triplets /48 | Body switches | Median root-position span of the three minima (m) | Fixed-body/progress matched units /48 |
|---|---:|---:|---:|---:|
| Depth | 30 | 1/30 | 0.230 | 24 |
| Onset | 25 | 0/25 | 0.233 | 19 |
| Recovery | 33 | 1/33 | 0.189 | 26 |

The reference event-position spans are much smaller: 0.015, 0.028 and 0.001 m.
Execution changes where the minimum occurs even when the same primitive limits
clearance. To distinguish that effect, the diagnostic samples the baseline's
limiting primitive at its limiting root position in both edited executions.
Only unique forward crossings with observed distances and a bracket no wider
than 0.05 m are usable. Ambiguous or unobserved crossings are not filled in.

| Axis | Median scalar → anchored midpoint departure on paired units (mm) | Scene-mean anchored−scalar difference; scene-bootstrap 95% interval (mm) | Available scenes /12 |
|---|---|---|---:|
| Depth | 12.05 → 7.69 | −14.82 [−32.90, +0.14] | 11 |
| Onset | 6.18 → 20.58 | +4.23 [−36.31, +37.16] | 9 |
| Recovery | 9.83 → 11.25 | +1.68 [−19.58, +21.41] | 12 |

All three intervals include zero. The anchored statistic does not consistently
reduce midpoint departure, and body switching is rare. This weakens the proposed
explanation; it does not prove that timing is irrelevant or identify the controller's
causal dynamics. These half-scale-only counts differ from A10's comparison requiring
observations at both full and half scales. The fixed-anchor metric is descriptive
and never replaces whole-body clearance in qualification.

The next contribution hypothesis is therefore narrower: an explicit progress-first
native intervention may improve development passage supply without relying on the
failed learned inverse. [A12](astra-a12-progress-correction-protocol.md) freezes a
small correction rule, regenerates its actual output, measures its exact preview,
and compares it against a stronger fixed shallower command, equal-budget search,
and a timing ablation. Initial-preview proposals assign 38 units to reduced crouch
burden, preserve seven passing baselines, and assign three units with observed
progress and overhead deficit to bounded deepening. Timing changes apply to six
slow recovery suffixes and those three overhead cases. These are pre-execution
branch counts, not successes.

Focused trace tests pass, including scalar reconstruction, achieved timestamp
offset, missing/ambiguous spatial anchors and a synthetic moving-minimum example.
No original scorer or archived A6–A10 result was changed. Physics contact-force
sensitivity on G1 remains a separate open task.
