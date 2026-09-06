# EXP-024 result — the prospective screen ranks cutoffs but fails its operating contract

**Completed:** 2026-09-04. **Evidence tier:** obstacle-absent, closed-loop SONIC simulation;
128 references, one rollout per reference, physics seed 0. This is a prediction of the release
evaluator's tracking-error cutoff, not a fall, contact, or obstacle-traversal result.

The generation, scoring, and prediction stages are documented in
`docs/ramp-exp024-kinematic-stage-2026-09-02.md`. The 128 per-reference predictions were committed
before any SONIC rollout (`predictions.jsonl`, sha256 `18a2fb14…`). The completed ledger is
`outputs/exp024_reference_contract/`; `summary.json` is the source of every number below.

## Primary result: P1 fails

The preregistered P1 contract required all three conditions over all 128 references:

1. at least 90% of references flagged by `r_max > 0.20 s` terminate;
2. at most 30% of references passing the rule terminate; and
3. prospective single-feature AUC at least 0.90.

| frozen 0.20 s rule | evaluator terminated | evaluator survived | total |
|---|---:|---:|---:|
| flagged | 92 | 3 | 95 |
| passed | 20 | 13 | 33 |

- Flagged termination rate: **92/95 = 0.968** (Wilson 95% 0.911–0.989), passing its 0.90 bar.
- Passed termination rate: **20/33 = 0.606** (0.437–0.753), failing its 0.30 bar.
- Sensitivity: **92/112 = 0.821** (0.740–0.881); specificity: **13/16 = 0.812**
  (0.570–0.934).
- Prospective AUC: **0.855**, bootstrap 95% **0.742–0.943** (2,000 resamples, seed 20260902),
  failing the preregistered 0.90 bar and the 0.95 strong bar.

The post-hoc secondary rule (`r_max > 0.28 s`, equivalently at least eight 25 fps frames or
0.32 s) also fails: 79/81 flagged references terminate, but 33/47 passing references terminate.
It is reported beside the primary rule and never substituted for it.

**Decision:** `contract_confirmed = false`. The feature has prospective ranking information in
this campaign, but the frozen operating point omits too many failures to be an execution-admission
rule. It may be used as a diagnostic or launch-priority feature with the full confusion table
shown; it may not be called a validated feasibility invariant, safety filter, or trackability
guarantee.

## Root-contract ablation: kinematic improvement does not close the execution chain

All four arms were constructible under their preregistered criteria. Pinning root height reduced
the reference's unsupported duration, but did not produce a valid supported step and did not make
the 0.20 s pass set reliable under tracking.

| arm (32 paired seeds each) | produced | fixed-position 5 cm clear | median `r_max` | terminated | valid step passing support test |
|---|---:|---:|---:|---:|---:|
| free | 23/32 | 7/32 | 0.48 s | 29/32 | 0/32 |
| pinned heading | 27/32 | 8/32 | 0.74 s | 30/32 | 0/32 |
| pinned root height | 19/32 | 1/32 | 0.36 s | 32/32 | 0/32 |
| both pinned | 11/32 | 2/32 | 0.22 s | 21/32 | 0/32 |

P2 passes: the free arm reproduces the original reference-side elicitation and placement rates
inside their frozen intervals (23/32 produced and 7/32 fixed-position 5 cm clear). P3 also passes
its kinematic constructibility test: `pin_y` shortens `r_max` in 20/32 paired seeds and `pin_yh`
in 27/32, with median root peaks 0.776 and 0.770 m. Those are reference measurements, not dynamic
success claims. P4 is a prescriptive **NO-GO**: every arm has 0/32 references satisfying the
registered valid-step-plus-support condition, hence 0/32 satisfying the full retained-clearance
conjunction.

The `pin_yh` arm is informative but not a constructive success. Relative to free, it reduces
terminations from 29/32 to 21/32 while reducing elicitation from 23/32 to 11/32 and fixed-position
5 cm clearance from 7/32 to 2/32. The paired termination comparison is descriptive
(`p = 0.0574`, exact two-sided McNemar); no superiority claim is made.

## What this changes

1. **Screening remains useful for diagnosis and launch ordering, not admission.** The high
   flagged termination rate can save launches only if omitted failures remain explicitly charged;
   passing the rule cannot be equated with controller compatibility.
2. **Native root pinning is not the missing compiler.** It trades away the requested step and
   placement while leaving the valid conjunction empty.
3. **The empty-intersection result is strengthened.** Candidate ranking cannot create a reference
   that is simultaneously placed, support-valid, and retained. The next method must rewrite an
   admitted carrier and verify the achieved obstacle-relative trajectory.
4. **EXP-031's 76–80 mm loss remains diagnostic, not a fixed margin.** Two development carriers
   cannot calibrate `m_exec`. Compiler v2 must estimate carrier-specific phase/position
   transmission on development carriers and freeze a held-out execution-clearance quantile before
   its fresh obstacle-present comparison.

## Provenance and interruption accounting

All four launches completed and all 128 achieved trajectories were archived. A launch-03 timeout
that produced no process result, log, or evaluation artifact is preserved as `attempt-000`; an
exactly scoped resume rule created `attempt-001`, which completed. Earlier fail-closed receipts
for the source-amendment and campaign-output provenance checks are retained beside the ledger.
The tracker was a clean detached worktree at the frozen SONIC commit/core manifest and released
checkpoint; the unrelated modified shared tracker checkout was not changed.

The next deciding experiments are (i) the beam-present ducking comparison, which is the lowest-risk
path to a positive physical local-traversal result, and (ii) the staged Compiler-v2 calibration and
fresh stepping test in `docs/execution-aware-step-compiler-v2-plan-2026-09-04.md`.
