# Actual progress-first correction loses to the simple baseline

A12 completed all 384 generations and 256 obstacle-absent SONIC executions at
`f00b4a5`, across the same 12 development scenes and 48 source units. The full
correction passes the development descriptor on 13/48 selected outputs, versus
15/48 for the baseline-preserving shallower command, one-edit search, and timing
ablation. Its preregistered expansion gate fails. An additional unchanged-input
diagnostic exposes batch-context dependence, further limiting mechanism attribution.

[Protocol](astra-a12-progress-correction-protocol.md) ·
[Complete analysis](../outputs/astra_a12_analysis_v1/summary.json) ·
[Independent completion audit](../outputs/astra_a12_completion_v1/summary.json) ·
[Batch-context diagnostic](../outputs/astra_a12_batch_context_v1/summary.json) ·
[Matching A10 commands](../outputs/astra_a12_reused_commands_v1/summary.json)

![A12 development results](../outputs/astra_a12_figures_v1/progress_correction.png)

[Standalone PDF](../outputs/astra_a12_figures_v1/progress_correction.pdf).

## Implemented and executed intervention

[A11](astra-a11-trace-results-2026-09-06.md) reconstructed 672 archived body traces
but did not support expanding the failed local inverse around body switching.
A12 instead implements a deterministic progress-first rule from the initial
preview: reduce crouch burden when progress is missing/insufficient; deepen slightly
for an observed overhead deficit with adequate progress; adjust onset or recovery
using the observed event or suffix. It changes only native depth/onset/recovery,
regenerates with frozen ARDY, and measures the exact new output with frozen SONIC.
It does not edit qpos after generation or predict that its own output is qualified.

Initial-preview decisions: 38 reduced-burden units, seven preserved baseline passes,
three overhead-deficit units. Timing changes are assigned to six slow suffixes and
three overhead events. The no-phase arm makes identical depth decisions without
timing changes. The campaign generates both arms even when the commands coincide;
96 proposal requests are not 96 novel commands. Only nine full-arm commands differ
from their no-phase counterpart. All other operational work is charged explicitly.

Final selection preserves a passing initial preview, otherwise requires the exact
candidate's unchanged A6 development descriptor. If neither passes, it refuses.
The endpoint requires observed passage/recovery, the existing exit deadline, at
least 0.45 m continuation in one second, corridor/upright checks, and no nominal or
40 mm inflated geometry violation through recovery. It is not actual obstacle
contact or a confirmation endpoint. Fixed shallow/search reuse matching A10 previews;
the search selects its one edit using training scenes excluding the evaluation scene.

## All-assigned results

| Arm | Selected descriptor passes /48 | Raw new candidate passes /48 |
|---|---:|---:|
| Unchanged baseline | 7 | — |
| Fixed shallower + preserve baseline | 15 | — |
| One-edit search + preserve baseline | 15 | — |
| No timing + preserve baseline | 15 | 14 |
| Full correction + preserve baseline | 13 | 12 |

The raw versus selected difference matters: preserving the original qualified
output protects it even if a newly generated nominally unchanged command fails in
a different batch context. The three 15/48 controls need not pass identical units.

Full correction has six paired gains and no losses versus the unchanged baseline,
but zero gains and two losses versus each stronger comparator. Full-minus-fixed
shallow is −4.17 percentage points, scene-bootstrap 95% interval [−10.42, 0.00]
points. The same aggregate difference/interval holds against search and no-phase,
though the lost units differ. These are descriptive scene bootstrap intervals on
reused development data, not evidence of a general causal method effect.

| Scene | Baseline /4 | Fixed shallow /4 | Search /4 | No timing /4 | Full /4 |
|---|---:|---:|---:|---:|---:|
| s00 | 0 | 1 | 1 | 2 | 1 |
| s01 | 0 | 1 | 1 | 1 | 1 |
| s02 | 2 | 3 | 3 | 3 | 3 |
| s03 | 1 | 2 | 2 | 2 | 1 |
| s04 | 0 | 0 | 0 | 0 | 0 |
| s05 | 0 | 0 | 0 | 0 | 0 |
| s06 | 1 | 2 | 2 | 2 | 2 |
| s07 | 1 | 1 | 1 | 1 | 1 |
| s08 | 1 | 1 | 1 | 1 | 1 |
| s09 | 0 | 0 | 0 | 0 | 0 |
| s10 | 1 | 2 | 2 | 2 | 2 |
| s11 | 0 | 2 | 2 | 1 | 1 |

Both new arms have 28 fully observed passage/recovery records, ten censored exits,
nine non-arrivals and one censored recovery. Baseline has 22/12/12/2 respectively.
All 48 units stay in each denominator. The full arm's two losses versus fixed
shallow are s03/u3 and s11/u3, both assigned earlier recovery. This association
does not isolate a timing cause given the additional control failure below.

## Why exact whole-batch replays are insufficient

All 288 unchanged generation controls and all 128 baseline/free execution slots
match A10 exactly. All eight jobs pass the preregistered apparatus, mode and process
checks. Those controls establish reproducibility when the full input cohort repeats.

A post hoc comparison holds the target command, reference, prepared SONIC fields,
slot and encoder realization fixed while other motions in the batch differ:

- Full versus no-phase: 39/39 references and prepared inputs match exactly, but
  only 10/39 achieved prefixes do. Valid lengths, termination, observation status
  and descriptor labels remain equal for all 39 pairs.
- The median across these pairs of maximum root-position difference is 34.82 mm;
  the largest is 301.81 mm at group0/slot19 (s04/u3). Its first state difference
  appears at sample1; root difference exceeds 1 mm at sample36. This is not merely
  an array-format or reference-conversion discrepancy.
- Versus matching A10 commands, all 39 full-arm and all 48 no-phase prepared
  references also match exactly. Yet one full-arm and two no-phase raw descriptor
  labels change from pass to fail; four observation statuses change in each arm.
  These are overlapping diagnostic comparisons, not 87 independent units.

The repeated s03/u2 baseline request loses its raw pass in both new arms, and the
s11/u3 no-phase shallow request loses its A10 shallow pass. Baseline preservation
recovers the former for selection. The backend cause is unresolved: differences
could arise in policy inputs, inference, or simulation. Do not call this a proven
physics bug, nondeterministic controller, or quantified causal timing effect.
The prepared inputs include exact root translations/rotations, poses, joint data,
FPS and table transforms; the mismatch survives that additional check.

## Validation and accounting

The 384 generations comprise 96 proposal requests and 288 unchanged controls. The
256 executions comprise 192 scientific previews and 64 padding executions; 96
scientific previews are for the two proposals and 96 are baseline/free controls.
The simulator-job elapsed sum is 452.42 s; peak child RSS is 6099.77 MiB. Generation
batch decode sums to 31.22 s, excluding model loading, geometry measurement and
other overhead. No resource refusal or simulator failure occurred.

The complete A11 and A12 analyses rebuild exactly. Independent array checks validate
generation controls, full valid-prefix execution controls and the additional
unchanged-command contrasts. Figure sources/output hashes verify and the figure
was visually inspected. Full CPU suite: 1257 passed in 205.82 s; focused tests:18
passed in 2.44 s. The post hoc audit/diagnostic scripts were additionally run against
the complete source-bound archives. Logs remain preserved locally. The campaign is
finished, no admission poller remains, and `/tmp/s2m-a12-f00b4a5` stays frozen.

**Decision:** close A12 without expansion or retuning. It fails all three comparator
superiority checks while preserving baseline passes. Keep the fixed shallower
command as a development comparator, not a confirmed universal solution. Before
more native correction or confirmation, implement the
[batch-isolation design](astra-a13-batch-isolation-design.md). G1 positive-contact
sensitivity and obstacle-present testing remain open. This study establishes no
new actual-obstacle traversal, held-out transfer, multi-beam or stepping success.
