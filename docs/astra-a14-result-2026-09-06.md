# A14 passes: fixed-background compilation gives repeatable pilot comparisons

The adapter makes a candidate's simulator input independent of unrelated incoming
reference rows by compiling that candidate into its original slot in an immutable
32-reference context. In the operational pilot, both incoming contexts produce
the same compiled bytes and the same complete valid execution prefixes.

| Group | Edited target | Encoder | Repeated trajectories exact | Target valid samples |
| --- | --- | --- | ---: | ---: |
| 0 | s01/u3, slot7 | G1 | 32/32 | 413 in each run |
| 1 | s08/u3, slot3 | Teleop | 32/32 | 413 in each run |
| Total | Two distinct edited candidates | Both | 64/64 pairs | 128 operational executions |

Each compiled input differs from its declared background only at the pilot
target, whose fields equal its original A12 no-phase reference. All39 previously
unchanged target pairs also compile to byte-identical batches across incoming
contexts. These39 checks establish a construction invariant; they are not39 new
simulated candidate comparisons. Both pilot targets finish their tracking
references, but no new geometric clearance or obstacle-present traversal rate is
claimed.

## What this fixes and what it costs

Unrelated incoming candidates can no longer change the background used to execute
a target. The complete context is now explicit and must be included in preview
and cache identity. A14 retains SONIC/ARDY weights, reference values, original
slots, modes, physics and source apparatus. It changes comparison composition.

This is not a GPU solver fix, a guarantee of independent physics islands, or
evidence of robustness to disturbances. Changing the declared background may
still change results. The cost is one useful candidate per32-env job; other rows
are operational workload. Full-prefix equality includes the late/exact cases
present in both groups, but does not identify their hidden numerical mechanism.

## Validation and preserved attempts

- Frozen successful checkout: `/tmp/s2m-a14-a31207e`.
- Protocol: [A14](astra-a14-canonical-context-protocol.md).
- Campaign: `outputs/astra_a14_canonical_context_v2/summary.json`.
- Independent readback checks every compiled reference field against the original
  target/background and independently compares all64 paired valid trajectories:
  `outputs/astra_a14_completion_v1/summary.json`.
- Four successful jobs,249.4907s total simulator job time; peak child RSS6084.84MiB;
  minimum available RAM7417MiB. No A14 resource refusal, timeout or runtime abort.
- Latest full CPU suite:1294 passed in206.25s; focused compiler/driver/observer26
  passed. This follows an earlier1293-test pass before the serialization correction.
- Preserve the initial prelaunch failure at`e2c4453`: equal reference values had
  different pickle memoization from independently loaded dtype and field-name
  objects. The revised compiler normalizes metadata sharing. No gate, target,
  reference value, or budget changed; the failed preparation spent no seeds.
- A13 separately completed64 original-context replays and54 pure policy calls.
  Together A13 and A14 add192 operational executions, zero generations and zero
  independent scientific units. A13's two earlier resource refusals remain intact.

## Return to the performance question

The next [controlled correction comparison](astra-a15-controlled-correction-design.md)
uses a common nominal background for unchanged, fixed-shallow and frozen A12
correction candidates. Its baseline control must reproduce the original nominal
previews before those previews can be reused. This makes the feedback underlying
the frozen correction proposals comparable to the subsequent one-target edits.
Keep the historical losing A12 result separate from that new context.

No new A15 budget is assigned by this result. The broader requirements remain:
constructive motion gain, controlled positive G1 contact sensitivity, held-out
and multi-beam testing, stepping, and downstream utility. Reproducible comparison
is an enabling result, not a substitute for those performance experiments.
