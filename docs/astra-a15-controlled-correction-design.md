# Next stage: compare frozen corrections in a common nominal context

**Status: design only; not preregistered, prepared or launched.** September 6, 2026.

Historical design status above. This design was subsequently implemented under
the separate [A15 protocol](astra-a15-controlled-correction-protocol.md) and
[completed](astra-a15-result-2026-09-06.md): full correction ties fixed shallow
4/12 against nominal2/12; promotion fails. The original design is retained below.

A13 localizes early divergence downstream of equal policy outputs, and A14
validates a one-target/fixed-background adapter. The next experiment should
return to motion performance under that controlled comparison contract.

## Question and fixed comparison

Does the existing frozen A12 execution-aware correction improve the original
duck clearance/progress descriptor over the baseline-preserving fixed-shallow
method when both execute against the same nominal background?

Use the original baseline reference batch in each group as the immutable context.
First compile/replay the complete nominal context and require exact agreement
with its original A12 nominal execution. This gate is necessary: A12's proposals
were computed from those nominal previews, which must still be valid. Changing
the background requires these new controls even though the adapter passed A14.

For each assigned unit, compare unchanged nominal, fixed0.26m depth, and the
original frozen A12 full correction. Reuse the existing generated references;
keep prompts, weights, modes, slots and seeds. Do not refit A12 or alter its
thresholds based on this comparison. Each method has nominal plus at most one
candidate preview. Its final choice must use its own candidate's qualification
and preserve the nominal candidate under the original fallback rule.

## Proposed bounded development coverage

Use unit`u0` from each of the12 existing development scenes, selected by index
rather than performance. This spans both source groups and both realized encoder
modes without a mode sweep. It is one reused seed per scene, not confirmation or
generalization to new geometries.

Two nominal32-env control jobs can supply all nominal rows in their respective
backgrounds. Up to24 one-target jobs supply fixed-shallow and full-correction
previews: maximum26 jobs/832 operational rollouts,36 assigned scientific preview
measurements before any documented alias reuse. Exact duplicate compiled inputs
may reuse a verified execution only when the entire context identity matches;
count aliases separately. No new generation, present-obstacle test or new policy
training is needed for this comparison.

Retain every failure, non-arrival and insufficient continuation. Report raw
candidate and final-choice outcomes over all12 units, per-scene results, paired
gains/losses, and clearance, rear-body exit and one-second continuation separately.
Preserve the40mm inflation and original metric/termination conventions. This
small development comparison is not a statistical significance claim.

## Before launch

Turn this design into a separate committed protocol and executable harness.
Bind the exact nominal/candidate sources, original fallback rule, encoder rows,
reuse ledger, promotion criterion and resource budget before outcomes. Use the
existing compact resource gates and stop after any control/equality failure.
The proposal above assigns no simulation by itself.

If the frozen correction again loses, preserve that result and use the controlled
response evidence to design its successor; do not reinterpret the result as a
win through another endpoint. If it gains, validate across the remaining reused
development seeds before method freeze and untouched confirmation. Explicit
perturbation robustness, positive G1 contact sensitivity, multi-beam/stepping
expansion and downstream utility remain separate obligations.
