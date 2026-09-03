# Scene2Motion ICRA claim–evidence map

**Date:** 2026-09-03

**Purpose:** bind each proposed contribution to the experiment that can support it, including the
result that would weaken or remove the claim. This is a planning artifact, not a result.

## Operational bottleneck

A frozen motion prior and a frozen whole-body controller must jointly put whole-body clearance at
the obstacle, retain a controller-compatible support pattern, and complete the obstacle-present
route. The present pool fails at the intersection: 12/64 references clear the fixed 5 cm box and
11/64 complete obstacle-absent tracking, with 0/64 in both sets; obstacle-present completion is
0/64. A selector cannot repair an empty intersection.

The smallest constructive contribution is therefore not a new scorer. It is an
**obstacle-relative reference repair on a controller-compatible carrier**, followed by the same
geometry and support measurements and then an obstacle-present rollout. The frozen generator,
controller, scene, evaluator, and source reference are retained contracts.

## Claim ledger

| ID | Candidate paper claim | Highest evidence available now | Missing discriminating evidence | Claim if the test loses |
|---|---|---|---|---|
| C1 | Open-loop humanoid priors can produce clearance without placing it at a specified obstacle, and placement and tracking can be disjoint. | Measured on 64 ARDY references at one fixed scene; placement failure also observed kinematically on 64 Kimodo step prompts. | A prospectively fixed multi-position or multi-scene study. | Keep as a one-scene diagnostic result; remove language implying a population-wide property. |
| C2 | A frozen reference-only unsupported-duration feature predicts SONIC evaluator cutoffs. | Post-hoc AUC 0.997 on the 64-reference stepping pool; frozen 0.20 s rule flags 53/53 cutoffs and passes 8/11 completed runs. | EXP-024: outcomes for all 128 references whose predictions were committed before tracking; known-trackable controls. | Report the original result as retrospective and retain the prospective failure as the transfer boundary. |
| C3 | Bounded obstacle-relative repair can close the reference-to-traversal chain without retraining ARDY or SONIC. | Two of 64 archived references are admitted pre-execution after support screening and foot-envelope IK; no repaired rollout exists. | EXP-031: paired raw/repaired, obstacle absent/present rollouts for the two frozen candidates under evaluator v2. | If 0/2, report a failed engineering pilot and use its event ordering to choose windowed dynamics or achieved-state feedback; make no traversal-method claim. |
| C4 | Measured closed-loop headroom correction improves ducking references relative to equal-budget resampling when the proposer is weak. | Reference geometry over 36 scenes × 8 seeds: learned proposer +2 repairs beats best-of-3 by 21.9 points collision-free and 36.1 points on the 18 cm margin; heuristic loses by 8.0 points on margin. | Beam-present paired execution under the same candidate pool, controller, budget, and clip duration; fixed extra-crouch control. | Retain a proposer-dependent reference-geometry claim only. |
| C5 | Scene2Motion-DB is a feasibility-annotated record corpus, including failures rather than silently dropping them. | Validated preview: 300/300 pilot scenes, 268 motion payload hashes, 26 scene-feasibility refusals, 6 residual-collision rejections, separate signed overhead and collision labels. | Redistribution terms; execution-labelled tier from EXP-024/028/031; preregistered geometry-OOD split; downstream utility ablation. | Release the schema and preview as infrastructure, not as a training benchmark or generalization dataset. |

## Experiment cards, in decision order

### E1 — prospective support contract (EXP-024)

- **Question:** does the committed 0.20 s reference rule rank and classify later SONIC cutoffs?
- **Unit:** reference (`n = 128`); four root-control settings are reported separately and pooled
  only where the protocol permits.
- **Arms:** every flagged and passed reference is executed; no reference is removed after seeing
  an outcome.
- **Primary:** prospective AUC with interval. **Operating-point accounting:** sensitivity and
  specificity with raw numerators/denominators at 0.20 s.
- **Failure accounting:** evaluator cutoff, completion, and every rejected/failed launch over all
  assigned references.
- **Decision:** success upgrades C2 from post-hoc to prospective evidence. Failure does not change
  C3, because the screen is not a physical guarantee.

### E2 — cutoff-free outcome semantics (EXP-028)

- **Question:** when the evaluator rule is disabled, do flagged references physically fall,
  contact, stall, or continue?
- **Changed factor:** termination rule only; source references, controller, scene, and physics
  seed remain fixed.
- **Primary:** independent physical event flags and first-event time, not an exclusive label alone.
- **Decision:** this calibrates what the screen predicts. It cannot retroactively turn a cutoff
  into an observed fall.

### E3 — first constructive closure (EXP-031)

- **Question:** can the frozen repair convert either of two pre-execution-admitted carriers into
  an obstacle-present local traversal?
- **Arms:** raw absent, repaired absent, raw present, repaired present; the same two identities in
  every arm.
- **Primary:** at least one repaired-present completion under the predeclared whole-body,
  corridor, upright, and goal definition.
- **Denominators:** print both `k/2` admitted candidates and `k/64` source-pool references.
- **Meaning:** `k >= 1` establishes bounded engineering existence, not a fresh-pool success rate.
  A paper-level method comparison still requires fresh references and component ablations.
- **Historical-outcome disclosure:** the two candidates were admitted by pre-execution reference
  gates after development on the eight support-passing references.  In EXP-030, `s4434` completed
  the obstacle-absent route while `s4408` stalled; both avoided the evaluator cutoff and both
  contacted the present 5 cm obstacle.  The pilot must not describe both as historically
  route-completing.

### E4 — beam-present correction (EXP-029 successor)

- **Question:** does measured two-step correction retain its advantage after controller execution?
- **Necessary controls:** uncorrected, fixed extra crouch, two measured corrections, and
  equal-budget best-of-three resampling.
- **Unit:** scene (`n = 36`); seeds and physics seeds are repeated measurements within scene.
- **Primary:** obstacle-present local traversal over all assigned trials. Reference geometry and
  tracker survival remain separate rungs.
- **Required losing subgroup:** the heuristic proposer, for which resampling already wins on the
  reference-margin endpoint.

### E5 — dataset utility

- **Question:** do signed negative records add information beyond collision-free references?
- **Single-factor comparison:** one discriminator architecture, optimizer, compute budget, and
  scene split; train once on collision-free references only and once on the complete labelled
  records.
- **Test:** a scene-family geometry-OOD set fixed before training.
- **Metrics:** per-failure-mode recall, macro average, calibration, and task-level launch savings
  with the rejected trials counted as non-completions.
- **Boundary:** a lower training loss or an IID preview split is not utility evidence.

## Paper contribution shape after the tests

Use no more than three contribution bullets:

1. **Formulation and audit:** the six-rung reference-to-traversal evidence chain and the measured
   placement/trackability disjunction, fenced to the tested priors, controller, and scenes.
2. **Method:** obstacle-relative repair plus a calibrated pre-execution support contract, with the
   exact closed-loop endpoint and the losing conditions reported.
3. **Artifact:** the rebuild-grade Scene2Motion-DB schema and records, with dataset utility claimed
   only if E5 moves a held-out metric.

Do not use “guarantees,” “robust,” “general,” “first,” or “outperforms” unless the corresponding
proof, perturbation sweep, population, literature search, or same-contract comparison exists.
Attach the evidence tier and denominator to every result sentence.

## Current execution state

- The isolated EXP-028 → EXP-024 GPU chain passed its code/provenance preflight and is armed. Its
  resource gate remains closed while another Isaac workload owns the GPU; the chain waits and
  rechecks rather than interrupting it.
- EXP-031 remains unlaunched. Evaluator-v2 and exact-clearance hash re-locking must be committed by
  their current owner; then the protocol is changed from `draft` to `preregistered`, candidates
  are prepared and hashed, and the eight paired rollouts may start.
- Exploratory pelvis and leg-IK support projections were not admitted: variants either destroyed
  box clearance or required discontinuous joint changes. They do not enter a launch pool or a
  paper claim.
