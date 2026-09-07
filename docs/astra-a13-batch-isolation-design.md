# Next stage: isolate batch-dependent execution before further correction

**Status: design, not preregistered or launched.** September 6, 2026.

Implementation update: this design now has a [preregistered observer protocol](astra-a13-batch-observer-protocol.md)
and [prepared-campaign status](astra-a13-status-2026-09-06.md). The original design
below is retained; the protocol governs its assigned diagnostic workload.

A12's frozen correction fails its development comparator gate. More importantly,
identical prepared target inputs can execute differently when the other batch rows
change, despite exact full-cohort replays. A11's body traces and A12's bounded rules
do not justify another inverse-model or threshold expansion. This problem directly
affects the reliability of motion editing comparisons across all scene families.

## Smallest useful contribution and test

Develop an execution comparison that keeps target context controlled, with a
reproducible regression demonstrating that the target's observations/actions and
execution do not change solely because unrelated motion rows change. Retain frozen
ARDY, SONIC weights, original slots, physics configuration and source artifacts.
An instrumentation test is not a new motion-generation or traversal result.

Use the already observed worst unchanged-input diagnostic, s04/u3 in group0/slot19,
as an explicitly outcome-selected operational stress case. Its prepared reference
is identical between A12's full and no-phase batches; achieved root separation
reaches0.302 m. It is not an independent scientific or held-out unit.

1. Inspect the pinned policy-input and action path and the batch-dependent
   preprocessing. Add a narrow observer capturing the first eight policy steps'
   full-batch actor inputs, encoder masks, raw actions, applied actions and pre/post
   root states, with exact timestamps and row identities. Keep contact and physics
   settings unchanged. This finite trace should distinguish an input change from
   an inference change or downstream physics change at the first divergence.
2. Before launch, preregister the observer, both exact A12 group0 contexts, source
   pins, bytes expected unchanged, resource budget and stopping rule. The proposed
   first budget is two32-env observed replays (64 operational executions), zero
   new generations. Compare every achieved prefix with its own uninstrumented A12
   archive. If observation changes execution, stop; do not diagnose the mechanism
   from a perturbed run or silently loosen equality.
3. Once instrumentation preserves both contexts, replay captured policy inputs:
   hold the target row fixed, substitute only other rows, and first check an exact
   repeat of the entire input. Keep batch shape and weights fixed. This is an
   operational inference probe; it must distinguish target-input differences from
   target-action differences and numerical changes from downstream amplification.
4. Implement the smallest fix justified by the first divergence. Possibilities
   such as context-dependent preprocessing, inference arithmetic or physics are
   hypotheses, not diagnoses. If no local fix is established, design a controlled
   fixed-background or isolated-environment evaluation and validate its resource,
   encoder and execution contract before transferring scientific experiments.

## Boundaries and acceptance

Keep the A12 failure and every context-dependent outcome. Do not rerun its spent
seeds under a looser success gate, count padding as independent evidence, or use a
single operational stress case as broad scene confirmation. Only a separately
versioned corrected apparatus may support later renewed motion comparisons.

Acceptance for the operational fix must cover exact same-context repeatability,
unchanged-target stability under the declared context changes, and preserved
identity/timestamps/valid-prefix handling. Freeze the numerical criterion before
the validation run; never derive a passing tolerance from the observed discrepancy.
Include additional preassigned target slots and both encoder realizations in the
subsequent validation design rather than concluding from slot19 alone.

The original resource gates, ASTRA lock, compact supervisor, refusal preservation
and no-co-tenant-interruption rules still apply. No poller, held-out confirmation,
present-obstacle campaign or new policy training is assigned by this design.
Controlled positive G1 contact sensitivity remains the next physical-measurement
obligation alongside a validated execution apparatus. Then resume actual bounded
correction versus the fixed shallower comparator, before multi-beam and stepping
extensions. Do not make those families inherit a duck-only success claim.
