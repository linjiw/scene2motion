# Xiao-inspired review of Scene2Motion's current evidence

**Date:** 2026-09-03
**Scope:** the committed Scene2Motion evidence, constructive EXP-031 pilot, prospective
screen campaign, and Scene2Motion-DB preview. This is an evidence-focused review using the local
Xiao paper-review and paper-writing rubrics; it is not an impersonation or a prediction of any
individual reviewer's score.

## Bottom line

Scene2Motion has a credible diagnostic result and the beginnings of a constructive method, but it
is not yet supported as a traversal system. The highest-leverage next result is one
obstacle-present completion from the frozen EXP-031 pilot. That would upgrade the paper from a
negative audit to a bounded engineering-existence result. It would not yet support a success-rate,
generality, or component-attribution claim.

The current work is strongest when it keeps three contributions distinct:

1. **Measurement:** the generated behavior, fixed-obstacle placement, reference collision,
   completed tracking, achieved clearance, and obstacle-present traversal rungs.
2. **Method:** a reference-only support screen followed by obstacle-relative foot-envelope repair
   on a controller-compatible carrier, with every rejected candidate retained in the denominator.
3. **Artifact:** a rebuild-checked record schema and a 300-scene metadata preview, pending
   redistribution review, execution-labelled records, an OOD split, and a downstream utility test.

## Charitable reconstruction of the claim

The operational problem is set-theoretic, not cosmetic. In one frozen 64-reference ARDY pool,
12 references clear the fixed 5 cm box and 11 finish the controller run without its evaluator
cutoff, but the intersection is empty. With the box present in physics, local traversal is 0/64.
A better ranker cannot select an element from an empty intersection.

The proposed intervention therefore preserves the frozen generator, controller, scene, and source
reference while changing one object: the controller's reference. It first refuses references that
violate the frozen 0.20 s unsupported-duration rule, then moves only leg joints to place a
support-preserving foot envelope over the fixed obstacle. The repaired reference is re-measured
before any rollout. This is the mechanism the present code implements. It is not yet evidence for
a generic phase dispatcher, a QP guarantee, or learned scene conditioning.

## Claim audit

| Claim | Evidence rung reached | Reviewer-safe wording now | Evidence needed to upgrade it |
|---|---|---|---|
| Placement and tracking can conflict | 64 ARDY references in one route/scene; Kimodo supplies a kinematic placement boundary | The useful sets were disjoint in the tested ARDY pool | Prospectively fixed obstacle positions across multiple scenes |
| Unsupported duration predicts SONIC cutoffs | Retrospective AUC 0.997; frozen 0.20 s operating point on the original pool | A reference feature retrospectively ranked one evaluator's cutoffs | EXP-024 outcomes for all 128 pre-labelled references |
| Repair closes the traversal chain | Two reference-level candidates; zero repaired rollouts | Two archived references passed pre-execution repair gates | At least one EXP-031 repaired-present completion |
| Ducking correction improves execution | Reference geometry only; proposer-dependent and losing for the heuristic proposer | Measured correction improved learned-proposer reference geometry under an equal generation cap | Beam-present paired rollouts plus fixed-extra-crouch control |
| Scene2Motion-DB is a benchmark | 300-scene validated metadata preview; no execution labels or downstream consumer | A feasibility-annotated record-corpus preview | Redistribution clearance, payload release, execution tier, OOD split, and utility ablation |

## Highest-priority concern

There is still no closed-loop constructive endpoint. The repair was developed on the same eight
support-passing references from which its two admitted candidates were obtained. Running more
offline geometry on those references cannot answer whether the edit survives SONIC or the
physical obstacle. The smallest discriminating test is therefore EXP-031 exactly as scoped:
raw versus repaired, obstacle absent versus present, on the same two frozen identities.

The two identities do not provide interchangeable substrate evidence. In EXP-030, both avoided
the tracker cutoff and both contacted the 5 cm obstacle, but only `s4434` completed the
obstacle-absent route; `s4408` stalled short of the goal. The pilot must print this asymmetry and
must not call both candidates historically route-completing.

If either repaired-present rollout completes, the licensed sentence is:

> In one preregistered scene, obstacle-relative repair converted at least one of two
> pre-execution-admitted archived references into an obstacle-present local traversal under the
> frozen SONIC controller.

That is an existence claim. The denominator also remains 1-or-2 of the 64-reference source pool.
If both fail, the event ordering—not a relaxed rerun—selects the next mechanism.

## Other concerns and bounded repairs

### The screen's target is narrower than dynamic feasibility

The 0.20 s rule predicts the release evaluator's cutoff. A cutoff is not a fall, and the last
archived state of each original cutoff remained upright. EXP-024 prospectively tests prediction;
EXP-028, with the cutoff removed and independent event flags retained, tests what physically
happens afterward. Neither result should be substituted for the other.

### The mechanism is not yet isolated

EXP-031 changes a combined longitudinal-and-vertical foot-envelope edit. It can establish that
the combined surgery works on a selected carrier, not whether placement, lift, smoothing, or the
support screen caused the outcome. After an existence success, the cheapest causal study uses
fresh references and zeroes one component at a time: raw, longitudinal-only, height-only, full
repair, and equal-budget resampling.

### A 20 cm result is not the next cheapest test

The frozen operator admits no 20 cm candidate from the eight development substrates. Launching a
known-rejected motion would not test traversal. First establish the 5 cm chain; then change the
workspace or solve a windowed whole-leg trajectory problem under a new protocol.

### The artifact is not yet a public benchmark

The preview accounts for all 300 scenes and can validate its own metadata, but it does not include
the 268 motion payloads by default, has no execution-labelled tier, uses an IID development split,
and has no downstream consumer. Until those gaps close, call it a record-corpus preview rather
than a released training benchmark. The discriminating utility experiment holds the learner and
budget fixed and changes only whether signed negative records are included, evaluated on a
predeclared scene-family OOD split.

## Prioritized decision sequence

1. **Finish EXP-028 and EXP-024 without relaxing the existing resource/provenance gates.** These
   campaigns are already armed and answer the screen-semantics and prospective-prediction claims.
2. **Preregister and run EXP-031 after evaluator v2 and its exact-clearance hash relock land.**
   This is the conclusion-changing constructive test.
3. **If EXP-031 succeeds, freeze the operator and run the fresh-pool component comparison.** If it
   fails, use first collision, first cutoff, and achieved-state error ordering to choose one new
   repair mechanism; do not tune and repeat the same two trials.
4. **Only then expand the execution tier:** 20 cm stepping and beam-present ducking.
5. **Promote the dataset preview last:** merge execution records, resolve redistribution, freeze an
   OOD split, and run the negative-record utility ablation.

## Provisional three-contribution shape

1. A six-rung evaluation that identifies where scene placement and controller execution separate,
   scoped to the tested priors, controller, route, and scenes.
2. An obstacle-relative, support-screened reference repair, with any traversal claim stated at the
   exact closed-loop denominator and with losing conditions retained.
3. A rebuild-checked feasibility record schema and corpus, upgraded to a benchmark contribution
   only if the release and held-out utility obligations are met.

This order makes the paper constructive without promoting a proposal into a result.
