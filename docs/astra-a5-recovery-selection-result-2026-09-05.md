# A5 recovery and selection: improve the candidate pool, not its ranker

## Decision

Keep default fixed3 as the strong comparator. In the existing default fixed2/fixed3
pool, selection has only one additional operational local pass available. That trial
is essentially stationary during the following second. Under the descriptive
conjunction of local passage and positive one-second kinematic continuation, both
selection and the outcome-informed pool ceiling remain **17/32**, matching fixed3.
This is a post-hoc development finding, not a prospective performance improvement,
a physical safety verdict, or evidence that other candidate pools cannot improve.

## Recovery measurement

The [recovery receipt](../outputs/astra_a5_recovery_audit_v1/summary.json) retains all
288 A5 present assignments. For each available whole-body dwell endpoint, inspect
the next second on the native 50 Hz clock: 51 samples including both endpoints.
Require an observed complete window, positive net root displacement, and continued
compliance with the existing pelvis, upright, corridor, root recovery-line and
whole-body far-edge bounds. Missing suffixes remain censored, not successful.
These are post-hoc kinematic checks; positive displacement is not a calibrated
motion-quality threshold. Contact and high-quality qualification remain unknown.

| Hold / mode | Original local passes / 32 | Passes also meeting recovery checks |
| --- | ---: | ---: |
| fixed2 / default | 14 | 13 |
| fixed2 / G1 | 15 | 15 |
| fixed2 / teleop | 13 | 12 |
| fixed3 / default | 17 | 17 |
| fixed3 / G1 | 17 | 17 |
| fixed3 / teleop | 15 | 15 |

Free nominal has zero local passes in all three modes. All 91 original local passes
have complete recovery windows; 89 meet the additional checks. Neither exception
violates pelvis, upright, corridor or whole-body far-edge bounds:

- `a5u3_r1`, fixed2/default: net displacement −0.313 mm; maximum backtracking
  15.591 mm. It misses only the strict positive-displacement condition. Describe it
  as essentially stationary, not falling or unsafe.
- `a5u3_r1`, fixed2/teleop: net displacement −43.449 mm; maximum backtracking
  57.339 mm. The root recrosses the recovery line at archived sample 212.

Original pass labels remain unchanged. One geometry, eight motion groups and four
matched slot repeats do not constitute 91 independent successful demonstrations.

## Selection ceiling and information cost

The [analysis card](astra-a5-bank-selection-analysis-card.md) fixes an absent-only
bank rule: choose one hold per carrier for all four repeats using virtual target-beam
pass counts, observed four-second horizons, complete-horizon median progress, then
fixed3 as tie-break. Present outcomes are read only after selection. Nevertheless,
A5 outcomes were already known during research design: this is not a blind test.
The bank consumes 16 existing references and 64 existing absent executions.

The [selection receipt](../outputs/astra_a5_bank_selection_v1/summary.json) chooses
fixed2 for `a5u1` and `a5u3`, fixed3 otherwise. It obtains 18/32 local passes versus
17/32 for fixed3: one gain, no losses. Both the best fixed hold per unit and the
slotwise outcome-informed union also reach only 18/32. These ceilings are not
deployable policies. The sole gain is the near-stationary default trial above.

The independently checked [joined receipt](../outputs/astra_a5_recovery_ceiling_v1/summary.json)
therefore gives 17/32 for all four alternatives when adding the recovery condition.
The tiny negative displacement makes this particular conjunction threshold-sensitive;
the result justifies avoiding a selector-training campaign, not declaring a new
physical impossibility or relabelling historical outcomes.

## Resource-gated contact continuation

The [continuation protocol](astra-contact-event-continuation-protocol.md), committed
at `acb6c21`, preserves the frozen 32-slot batch, references, physics seed and gates.
The 180-second admission wait produced ten preserved
[preflight refusals](../outputs/astra_contact_event_v2/fixed3_default_present/).
Free VRAM was 4306–4307 MiB, below 5500 MiB; available RAM was at least 11458 MiB.
**No simulator launched**, no new seed was spent and no co-tenant was stopped.
No poller remains. The earlier 15 process attempts remain the process-launch total.

The prepared job remains bound to the clean execution clone
`/tmp/astra-contact-6vd0OZ/repo` at `acb6c21`; do not update that clone in place.
Any later invocation must revalidate its source identity and resource admission.
The original aborted event run is preserved separately. Excluded-event coverage
and controlled G1 positive-contact sensitivity remain unresolved.

## Validation and next work

All three CPU receipts rebuilt exactly. The full CPU suite passed 1148 tests in
213.60 seconds before the final 14 selector/join tests were added; the final focused
suite passed 36 tests, including those additions. Recovery scoring loads one achieved
archive at a time and computes FK only for the observed one-second segment. No model
generation, training or GPU execution was needed for these analyses.

1. Complete the resource-gated contact diagnostic and controlled G1 contact probe;
   do not promote prefix observations to episode-level contact-free labels.
2. Specify prospective final-edited-motion qualification, including progress and
   recovery, with calibration separate from fresh evaluation.
3. Develop one bounded candidate modification addressing achieved clearance **and**
   progress. Keep fixed3/default and fixed2 controls; do not revive the losing
   overlap-only hold extension or tune against spent A5 present outcomes.
4. Evaluate frozen edits on fresh grouped units. Retain raw, edited, absent, present,
   refusal and unknown-quality records together. Dataset utility and scene coverage
   remain outstanding; this diagnostic does not complete the dataset objective.
