# A17 design: validate the depth-only lead on remaining development units

**Status: design only; not preregistered, prepared or launched.** September 6, 2026.

[A16](astra-a16-result-2026-09-06.md) improves the reused u0 pilot by one unit:
depth only5/12 versus fixed shallow/full4/12. Retain the frozen A12 no-phase
rule and evaluate its incremental value on u1–u3 from each of the12 existing
scenes. These36 units were outside A16's pilot but have appeared in earlier
development campaigns; they are not untouched seeds or new scene geometries.

## Fixed comparison and input inventory

Compare nominal, fixed0.26 m shallow and frozen depth-only correction, with the
same nominal-preserving selection,40mm geometry inflation and0.45 m continuation
gate. Reuse original generated references. Keep original slots, encoders, seed0,
nominal backgrounds and the frozen A15 simulator runtime. Nominal controls already
contain all36 remaining scientific rows, but their cached states and complete
execution identities must be revalidated before reuse.

Assign scene indices0–11 and unit indices1–3 by index: seed64000+4*scene+unit,
group=scene//8, slot=(scene%8)*4+unit. No geometry, seed, command or slot is selected
by its new outcome. All three methods remain bounded by the nominal preview and
at most one candidate preview. No generation, policy training or present outcome
is used to select commands.

The [input inventory](../outputs/astra_a17_input_inventory_v1/summary.json) derives
budget and aliases from complete payload hashes and validated runtime contracts,
without measuring new outcomes. It contains108 arm/unit requests:

- 41 prior cache requests:36 nominal and five depth-only nominal-preservation cases.
- 30 depth-only requests equal their same-unit fixed-shallow complete payload.
- 37 possible new jobs:36 fixed-shallow contexts and one distinct depth-only context.
- Maximum1184 new operational rollouts if the complete comparison is dispatched.

The sole additional depth-only context is **s01/u2, seed64006, group0/slot6**.
The inventory itself assigns zero jobs and spends zero simulation. Recompute and
bind every source/alias before preregistering execution; the inventory is not a
substitute for the full cache validation contract.

## Proposed discriminating gate before full dispatch

For the30 shared candidate contexts, both methods consume exactly the same
preview under the same fallback rule. For the five no-edit cases, verify that
the cached nominal states qualify and that both methods preserve them. These
checks establish equal final decisions in35 pairs, independently of the shared
candidate's unknown absolute outcome. Failure of this proof must stop preparation.

Preregister s01/u2's fixed-shallow and depth-only jobs first: two jobs/64 rollouts.
Use the common cached nominal and the original selection rule. If depth only
does not win this pair, strict improvement on the remaining36 is impossible
under the verified35-pair equivalence. Save that failed lead gate and stop without
claiming absolute performance on unexecuted units. Do not label those35 unexecuted
shared candidates as passes or failures, and do not loosen the gate.

If depth only wins, dispatch the remaining35 distinct shallow contexts to
complete all36 units and reconstruct the aliased depth-only measurements.
Report the full108 measurements,37 new jobs/1184 rollouts, all terminations and
censored attempts. Require a strict selected-pass gain over fixed shallow and
zero lost nominal passes; report per-scene wins/losses and scene-clustered
uncertainty alongside descriptive Wilson intervals. Keep A16 pilot results
separate from this36-unit validation; pooled48-unit counts are secondary.

This gate order is proposed before measuring the distinct pair. It limits
unnecessary computation; it does not create additional independent evidence.
Only one pair can change the selected-method comparison, which sharply limits
the breadth of any positive claim even if absolute cohort performance improves.

## Before launch and after the decision

Commit a separate executable protocol with the equivalence proof, source hashes,
all36 assignments, original mode masks, dispatch/stop order, alias ledger,
resource budget and analysis rules. Keep the compact32-env supervisor, frozen
resource floors and no-retry policy. The proposed maximum is37 jobs with900s/job;
admission remains RAM9216MiB/VRAM5500MiB and runtime floors2500/1200MiB.

A positive result supports evaluating the strongest qualified compiler with
controlled G1 contact sensitivity and paired obstacle-present passage. A failed
remaining-unit gate closes the incremental depth-only lead at this stage; retain
the successful pilot and stronger simple baseline without more tuning on these
units. Method freeze, untouched scene confirmation, multi-beam/stepping transfer
and downstream policy utility each require their own prospective evidence.
