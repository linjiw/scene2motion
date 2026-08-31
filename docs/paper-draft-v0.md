# A Frozen Motion Prior Is Not a Planner: Auditing, Verifying, and Repairing Scene-Conditioned Humanoid Motion Generation

**Draft v0 — 2026-08-31.** Target: CoRL / RSS systems track. Anonymous for review; artifact
release under the project name **Scene2Motion-G1**. Numbers marked ⟨v2⟩ come from the fixed
sampler and are final pending only seed-count notes; the Table 2 cells marked ⟨4E⟩ are being
produced by the 8-seed run now in flight and slot in without any other text change.

> **Framing decision:** this is a **methods + systems paper** whose released artifact is a
> **stable baseline and verified-dataset generator** for scene-conditioned humanoid
> navigation/traversal — the scene grammar, the closed-loop pipeline, the provenance-frozen
> ledgers, and the audit toolkit. The pipeline's value to industry and academia is that it
> turns a frozen off-the-shelf prior into a reliable producer of *labeled, verified*
> traversal data (§9); the method is what makes that data trustworthy at all.

---

## Abstract

Large frozen motion priors are increasingly proposed as drop-in "planners" for humanoid
robots: give the prior a path and some constraints, hand the result to a whole-body tracker,
and traversal should follow. We test that proposition end-to-end on a released
prior–tracker stack (ARDY-G1 → SONIC) and find it fails in a specific, measurable, and
repairable way. First, we introduce a **calibrated capability audit** for frozen priors used
as actuators: paired-seed generation, null-calibrated distinguishability thresholds, a
stability gate, and a physics stage. Under this audit, the prior's nominally 43-dimensional
constraint interface collapses to roughly **one reliably commandable, physically executable
behavior axis** — a naive single-seed count overstates the repertoire **6× kinematically and
~10× after tracking**, and we show the same counting mistakes arise from choices a careful
evaluator could make without noticing; the overcount replicates on a second released prior
(Kimodo-G1: 4.5× under the identical protocol), and so does the channel asymmetry that
predicts it. Second, we close the loop the audit shows is missing:
a black-box **generate → verify → repair → refuse** planner that measures what the generated
motion *actually* cleared against the robot's simulation collision model (with a separately
measured coverage margin), corrects the command locally from the measured deficit, and
refuses with a quantified reason when no body mode fits. On
out-of-distribution multi-beam scenes (8 paired seeds, 4 320 runs) this loop lifts every
proposer to ≥ 0.99 collision-free with zero regressions, while the equal-budget control the
field usually omits — independent resampling of the unchanged proposal — recovers only a
fraction of the gap, and a retrained imitation proposer none of it: measurement supplies
information that neither more samples nor more teacher data can. Third, we release the harness that
makes every number in this paper re-derivable: provenance-frozen ledgers for ~34 000
generations, a defect catalogue of ~30 measurement errors our own controls caught (most
biased toward our hypotheses), and an audit toolkit whose acceptance test is reproducing our
own earlier wrong answers. The system plans, generates, verifies, and repairs a scene →
whole-body motion in ~1–3 s per scene on one consumer GPU — in a measured pilot, 300
randomized scenes yielded 268 kinematically verified traversal records in 302 s. (Throughput
is tiered: raw generation, kinematic verification, accepted records, and physics-executed
records are separate rates; only the first three are covered by the pilot figure.)

---

## 1. Introduction

A humanoid robot that must cross a cluttered room needs whole-body answers: duck under this,
step over that, go around, or admit it cannot. Recent releases make an appealing recipe
available off the shelf: a **frozen kinematic motion prior** with text and constraint
conditioning (ARDY, Kimodo) feeding a **motion-tracking controller** trained at scale
(SONIC). The recipe's implicit claim is that the prior's conditioning interface *is* a
planning interface — that asking is the same as getting.

This paper measures that claim and then engineers around its failure. Our central finding is
best stated as a design principle:

> **A frozen motion prior is a stochastic, partially addressable motion actuator — not a
> planner.** Using one safely requires discovering what it can reliably be asked for,
> verifying what it actually produced, repairing locally from measurement, and refusing
> when neither works.

**Scope, stated up front.** This paper's contribution is the *mechanism and the protocol*,
not a breadth claim. Ducking is the archetype on purpose: it is the hardest kind of
adaptation to certify — continuous in amplitude, safety-critical in both directions (too
shallow collides, too deep destabilizes tracking), and coupled to the route. Everything the
protocol establishes for it — addressability, verification, repair, refusal, execution-aware
acceptance — is behavior-agnostic machinery; each new behavior the interface (or a future
prior) exposes drops into the same loop. We therefore write "scene-conditioned clearance
adaptation," and we show that even one certified axis, industrialized, is worth having (§9).

**Why the obvious evaluation overstates capability.** One generated clip is not a repeatable
capability; visible displacement is not command response; kinematic validity is not
executability. We formalize these distinctions into an audit (§4) and show they are not
pedantry: on the same 36 requests and the same clips, four defensible counting choices give
**10.0, 9.0, 6.7, and 1.67** addressable behavior modes per scene — a 6× spread — and the
physics stage removes most of what remains. Along the way, *five* successive
capability-absent claims of our own fell to better-formed requests, which yields the audit's
sharpest lesson: **a capability-absent claim is only as strong as the best request anyone has
tried.**

**Why closing the loop beats better proposing.** The prior's response to a command drifts out
of distribution exactly where planning gets interesting (more obstacles, deeper crouches). A
convex-optimal teacher and its distilled student both certify against a fitted response model
that is optimistic there; adding teacher data does not help because the teacher is the
ceiling. Verification against measured geometry sidesteps the ceiling: bounded local repairs, driven
by the measured clearance deficit, lift every proposer to ≥ 0.99 collision-free on 8-seed
out-of-distribution scenes (TCN 0.729 → 0.993; QP 0.872 → 1.000) with zero regressions —
while equal-budget *resampling* of the unchanged proposal, the control the field usually
omits, recovers only a fraction of the gap (TCN best-of-3: 0.774). Because the loop never
trusts the request, it also knows when to stop: refusal is an output with a quantified
reason, not a failure mode.

**Contributions.**
1. **A calibrated capability-audit protocol** for frozen generative priors used as robot
   actuators, with the 6×/10× overcount result and the elicitation principle, packaged as a
   toolkit whose acceptance test reproduces our own naive-vs-calibrated answers (§4, §8).
2. **A black-box generate–verify–repair–refuse planner** over a frozen prior: directional
   clearance tracing (overhead vs lateral), a measured-deficit repair operator with
   anticipation derived from the prior's own lag, quantified refusal, and a proposer ×
   feedback × resampling ablation that isolates *where the information comes from* (§3, §5).
3. **A mechanistic account of the interface**: which conditioning channels are addressable
   and why — absolute-height signals reach the body decoder directly while ground-plane
   signals pass only as velocity; rotations, not positions, are the cleanly addressable leg
   channel; text selects gaits but does not place them (§6).
4. **An execution-aware acceptance criterion**: tracking error grows with adaptation depth,
   so the margin a plan must certify is depth-dependent; we specify the signed-loss
   calibration τ(d) and release the achieved-state instrumentation for it (§7).
5. **The artifact**: scene grammar and hard set, provenance-frozen ledgers (~34 k
   generations), the ~30-defect catalogue with the controls that caught each, the demo, and
   the full harness (§8).

## 2. Related work

**Frozen prior → tracker pipelines.** Kimodo clips are shipped directly into the GEAR-SONIC
demo with no scene verification; ARDY positions itself as a planner for SONIC. We keep the
stack frozen and add the missing measurement layer. **Post-training the generator.** RLPF and
PhysMoDPO fine-tune motion generators from physical feedback; 2604.17335 and the perceptive
behavior-foundation line fine-tune the tracker or wrap terrain-aware reference synthesis.
Our setting is deliberately weight-frozen and per-instance: correction happens at test time,
from measurement, on a black box. **Test-time guidance and gating.** BRIC steers a frozen
diffusion planner with signal-space guidance in simulation; SafeFlow gates unsafe references
before a G1 tracker without scene geometry; ConstrainedMimic enforces constraints inside the
controller. None measures realized scene clearance and repairs the *generator command*; to
our knowledge the generate–verify–repair–refuse loop over a released frozen prior on scene
traversal is new. **Perceptive whole-body RL.** HumanoidPF and Perceptive Humanoid Parkour
train task policies that crouch, squeeze, hurdle, and climb on real G1 — broader repertoires
than ours. Our contribution is orthogonal: we characterize and safely exploit an *arbitrary
frozen* prior rather than train for the environment, and our audit applies to their evaluation
problem too. **Text-to-motion control.** LangWBC, Harmon, UniAct, MaskedMimic, CLoSD form the
comparison set for the text-channel analysis; CLoSD is the closest architecture (autoregressive
text-diffusion planner + tracker) but is jointly trained and scene-free. **Benchmarks.**
Moving Through Clutter's VR-collected clutter suite is the natural external validity target
once released.

## 3. System

**Problem.** Input: scene geometry (boxes), start, goal, and a stated preference. Output: a
whole-body qpos trajectory that reaches the goal collision-free with a stated margin — or a
refusal naming the binding constraint and its deficit.

**Route layer.** A* over (x, y, body-mode) with mode envelopes that are *measured, not
assumed*: each mode's clearance envelope is a split-conformal bound over paired-seed
generations (α = 0.1), so the planner refuses rather than guesses (~0.01 s per plan). Three
preferences (shortest / stay-upright / maximum-clearance) are argument sets to the same
planner, and produce genuinely different routes for the same scene and goal (Fig. 3).

**Proposal layer.** A duck-depth schedule along the route from one of three proposers: a
calibrated mode-lattice heuristic; a convex QP against a fitted response model; a TCN
distilled from the QP. The ablation in §5 treats all three identically.

**Actuation layer.** The frozen prior (ARDY-G1, 25 fps, horizon 52) conditioned through its
native interface: root path, heading, and absolute pelvis height; deterministic DDIM with
per-sample seed streams (one advancing stream per sample per generation — we found and fixed
a subtle sampler defect here, §8), ~0.4 s per clip at 5 denoising steps.

**Verification layer.** Exact MuJoCo collision geometry — the robot's shipped primitives
inflated by a *measured* 4 cm mesh-coverage margin — evaluated per route position and split
by contact normal into **overhead** (the duck schedule's responsibility) and **lateral** (the
route's responsibility). The split matters: an undifferentiated minimum tells a repair loop
to crouch at walls.

**Repair layer — a general recipe for black-box repair of generative actuators.** The
operator assumes only that the realized quantity responds monotonically to a scalar command
along a schedule, and it is built from three components, each earned by a measurement and
each carrying a guarantee:

1. **Directional gain inversion.** dq(s) = e(s) / |g′(q(s))| with e(s) = max(0, target −
   measured(s)), where |g′| is a *forward* secant over the direction the correction moves —
   a centred secant borrows steep gain *behind* the current command that a deepening
   correction cannot use (measured: within 4 mm of the exact inverse where the centred form
   undershoots by up to 27 %). A slope floor keeps the step bounded near saturation, making
   the repair deliberately partial there; the next verification decides whether partial was
   enough.
2. **Lag-derived anticipation.** The correction is shifted early by 3τ·v, with τ the
   generator's measured first-order response lag — not a hand-tuned lead. The sweep that
   could have refuted this (1.5τ–8τ) confirmed the derived value and rejected both extremes.
3. **Strict monotonicity.** dq ≥ 0 and q is clipped, so a repair can only deepen or extend
   the adaptation — it cannot trade one clearance violation for another, which is what makes
   "zero regressions among at-risk scenes" a structural property plus an empirical check
   rather than luck. Iterations are bounded (two, then refuse): an unbounded loop against a
   stochastic generator converges to whatever the seed happened to do; a bounded one fixes a
   response-model error and cannot launder an infeasible route.

Nothing in (1)–(3) is duck-specific or ARDY-specific: the recipe applies to any frozen
generator commanded through a scalar schedule whose output can be measured against a target
profile.

**Refusal.** When no calibrated mode fits and no route avoids the obstacle, the system
returns the binding functional, the scene value, the best available body value, and the
deficit — e.g. *overhead_clearance: scene 0.60 m, best mode 0.879 m, deficit 0.279 m* (§ Fig. 3d).

**End-to-end cost ⟨v2⟩:** 1–3 s per scene on one RTX 5080 (route ~10 ms; one generation
0.4 s; verification ~0.5 s; ≤2 repairs).

## 4. The capability audit

**Protocol.** (i) *Paired deltas*: every request is scored against a neutral-body control on
the same route and the same seed (deterministic sampler ⇒ controlled comparison). (ii) *Null
calibration*: distinguishability thresholds are the q99 of the prior's own seed scatter,
measured per descriptor channel — not round numbers. (iii) *Stability gate*: a behavior
counts only if the same request produces the same modal active set on ≥ 80 % of seeds.
(iv) *Deduplication* in whitened descriptor space, so dense sampling cannot inflate coverage.
(v) *Physics stage*: SONIC tracking as a reality filter.

**Result (Table 1).** Same 36 requests, same clips, four counting rules:

| counting rule | modes/scene |
|---|---|
| 1 seed, any change > 1 mm | 9.00 |
| 1 seed, round 1 cm threshold | **10.00** |
| 1 seed, 1 cm, dropping never-validating clips | 6.67 |
| **6 paired seeds, q99-calibrated, stability ≥ 0.8** | **1.67** |
| … and after the physics tracker | **~1** |

Each naive row is a choice a careful evaluator could make without doing anything obviously
wrong. That is the point: the failure modes are *procedural*, not exotic. (⟨v2⟩ re-check of
the calibrated row is part of the standing revalidation; the counting spread is a property of
the rules, not of the sampler.) The direct test that this is evaluation practice rather than
one checkpoint's quirk is cross-model replication: the same audit runs unchanged on
Kimodo-G1 (same skeleton, same harness, different architecture) — and it replicates.
**Reduced audit on Kimodo-G1** (12-program battery, 6 paired seeds + 6-null calibration,
84 clips, 173 s): naive counting rules yield 8.0 / 9.0 / 9.0 modes; the calibrated rule
yields **2.0** — a **4.5× overstatement** on a non-autoregressive prior, against ARDY's 6×.
The channel asymmetry replicates with it: the duck ladder is monotone with stability 1.00 at
every rung (Δtop 0.010 → 0.067 m for requests 0.10 → 0.40 m), lateral tuck sits at the null
q99 with stability 0.33–0.67, and the position lift over-responds ~1.6× on the requested
foot. (Caveats, stated in the receipt: vendored joint-point descriptors rather than MuJoCo
geometry, scene-free validity rule, one route — paired deltas are comparable, absolute
extents are not.)

**The elicitation principle.** Five times in this project a capability was declared absent —
tuck, yaw, composed programs, position-commanded lift, ungated 0.35 m step-over — and five
times the cause was how the request was formed, not the model. A frozen prior gives no signal
that a better request exists. Capability-absent claims must therefore be scoped to the
request family tried; ours are.

## 5. Closing the loop: proposer × feedback × resampling

**Design.** On a 36-scene out-of-distribution hard set (3–6 beams; training stopped at 2),
every proposer runs at fixed generation budgets under three feedback regimes: none (one-shot),
verification-guided repair (≤1, ≤2), and — the control the field usually omits — *independent
resampling with the proposal held byte-identical* at the same budget, so "feedback helps"
cannot be confused with "more samples help". One candidate generation per attempt; per-scene
paired seeds; provenance (checkpoint hashes, code commit, tracked-diff hash) frozen into every
row.

**Table 2 ⟨v2, 8 paired seeds, 36 OOD scenes, 4 320 runs, sampler v2⟩** — cells are
collision-free / meets-0.18-m / mean generations:

| proposer | one-shot | +1 repair | +2 repairs | best-of-2 seeds | best-of-3 seeds |
|---|---|---|---|---|---|
| heuristic | .983 / .535 / 1.0 | **1.000 / .622 / 1.5** | 1.000 / .646 / 1.8 | .993 / .656 / 1.5 | 1.000 / .726 / 1.8 |
| QP | .872 / .007 / 1.0 | 1.000 / .326 / 2.0 | 1.000 / .417 / 2.7 | .944 / .017 / 2.0 | .972 / .031 / 3.0 |
| TCN | .729 / .003 / 1.0 | .913 / .278 / 2.0 | .993 / .375 / 2.7 | .764 / .007 / 2.0 | .774 / .014 / 3.0 |

Executable-criterion rates (tracking-error-only threshold, depth-interpolated): heuristic+1
0.986, heuristic+2 0.993, QP+2 0.997, TCN+2 0.906. Wilson 95 % lower bounds accompany every
1.000 (0.987 at n = 288). Every paired comparison in the table has **zero regressions**, with
the at-risk denominators stated (283 / 251 / 210 collision-free opportunities for
heuristic / QP / TCN). *Statistics note:* seeds are nested within scenes, so the scene
(n = 36) is the primary inference unit; scene-level cluster-bootstrap CIs accompany the
seed-pooled McNemar values (auxiliary) in the analysis receipt.

**Three findings, each pre-registered as a possibility and now measured.**

*Feedback is the information source, not sampling.* For the two proposers that fail out of
distribution, equal-budget resampling of the byte-identical proposal recovers only a fraction
of what repair recovers: TCN +2 repairs reaches 0.993 collision-free where best-of-3 seeds
reaches 0.774 (paired on identical scene × seed: 63 scenes repair-only-fixes vs 0 the
reverse, exact p = 2.2 × 10⁻¹⁹); QP shows the same shape (1.000 vs 0.972). The repair signal
is *measured geometry*, and no amount of re-rolling the dice substitutes for it. The one
symmetric case is the already-calibrated heuristic near its ceiling, where best-of-3 seeds
buys margin (0.726 vs 0.646) — seed selection is a margin optimizer for a good proposer, and
no rescue at all for a bad one.

*The learned scheduler exits the system.* The pre-registered kill condition fired:
heuristic + 1 repair (1.000 / 0.622 / 1.47 generations) Pareto-dominates the repaired TCN
(0.993 / 0.375 / 2.72) on every axis at half the cost — paired: heuristic+1-only-fails 0 vs
tcn+2-only-fails 2. Phases 2–3 accordingly become the paper's instructive ablation: a TCN
*retrained with held-out beam counts in distribution* still does not close the gap, because
its QP teacher meets the target on ~1 % of these scenes. Scoped precisely: *this fitted
response model stays optimistic under this scene shift, and a student distilled from it
cannot recover information absent from that teacher* — **for open-loop imitation the teacher
is the ceiling, and measurement is not an imitator**. (This does not preclude a
response-conditioned learner; that is the successor method's hypothesis.) The deployed
system is: calibrated heuristic proposes, measurement certifies, repair corrects, refusal
bounds.

*Zero regressions is structural plus empirical, and it held at scale.* Across all 4 320 runs,
no repair or resample arm broke a single collision-free or margin-satisfying outcome its
one-shot achieved (monotonicity guarantees the direction; the 0 / 283-at-risk observation
confirms the cross-coupling it cannot guarantee is absent). Margin gains under repair are
large and exact (heuristic +25/0, p = 6 × 10⁻⁸; QP +118/0, p = 6 × 10⁻³⁶; TCN +107/0,
p = 1.2 × 10⁻³²). The anticipation term is visibly doing its work: mean repair onset shift
−3.1 cm (earlier) with +3.1 cm added duration over 1 925 repair steps, and the slope floor
never bound.

**Saturation is reported, never laundered.** Residual margin misses that end at the command
limit are labeled `accepted_margin` with the limit named; 12/21 of the v1 TCN's residuals
were command-range limits, not repair failures.

## 6. What the interface can and cannot say

**A mechanism, not a mystery.** ARDY's two-stage decoder overwrites *root* constraints into
the sample at every denoising step, but passes *body* constraints only as conditioning
features to a latent decoder whose local-root context is [heading rate, planar velocity,
**absolute root height**]. Height is therefore a first-class command; ground-plane body
targets are reconciled against a root the model is simultaneously generating. This predicts
the audit: duck (absolute height, root) is addressable and executable with a graceful
dose-response (tracked success 1.000 / 0.750 / 0.625 at 0.20 / 0.35 / 0.50 m); lateral
narrowing through positions is sub-noise (≤1.6 σ, sign-flipping) *and* structurally capped by
the thighs (~32 mm vs a 55 mm noise floor); leg **rotations** are the cleanest channel we
measured (+271.8 mm foot rise at sd 6.6 mm — ~41 σ — with the pelvis still) yet are absent
from the official constraint family, which exposes only chain-base (ankle/wrist/pelvis)
rotations.

**Text selects behaviors; it does not place them.** Changing seven words ("steps over an
obstacle") raises the swing foot +56.9 mm (3.6 σ, contacts preserved, tracks 0.625) where the
position channel, asked for the same thing, over-delivers 1.8–2.5× and goes ballistic. But
under the whole-body box-clearance metric the text gait clears **0.0 m in 8/8 seeds**, with
swing placement uncorrelated with the obstacle (peak-x sd ≈ 0.70 m, same as walking). The
prior *contains* more behaviors than any single channel exposes — and no channel we tested
composes "which behavior" with "where". That composition (text × transplanted keyframes ×
window-level prompt switching) is the pre-registered follow-on experiment, with the honest
prior that our one existing composition datum is null.

**The step-over door, closed for this request family ⟨1C, 24 seeds⟩.** A per-side-gated
position-channel lift, re-measured with each seed's own gait phase (fixing a gating defect in
the earlier experiment), fails tracking at every amplitude on a pre-registered ladder:
success 0.250 → 0.000 across requests 0.05–0.28 m while matched controls track 0.92–1.00,
with bilateral flight 0.91–0.98 *despite* gating — the channel's ~2× over-response makes the
swing ballistic at any dose. The earlier "weakly executable" 3/8 was a small-sample,
mis-gated artifact, and the conformal envelope's original 0/20 refusal of this family stands
re-validated. Per the elicitation principle we scope the kill to the request family tried:
position-channel step-over is closed; rotation-scaffold and text × keyframe requests are the
pre-registered open door.

**Refusal as a first-class output.** The conformal envelope refuses scenes with a quantified
deficit; on the demo's 0.60 m beam: deficit 0.279 m against the deepest calibrated mode
(Fig. 3d). Refusals are auditable — a refusal once thought wrong (via the text channel's
foot-peak delta) turned out right under the whole-body metric, and the audit trail is what
settled it both times.

## 7. From kinematic certificates to executable ones

A collision-free *generated* clip is not a collision-free *executed* one: tracking error at
the clearance-critical bodies grows with adaptation depth (torso+wrist proxy: 33.5 → 45.6 →
71.9 mm at duck 0 / 0.35 / 0.50 m), so the certificate a plan needs is depth-dependent. We
specify the calibration: signed clearance loss τ(d) = one-sided conformal bound on
c_ref − c_exec over the *same* margin-inflated geometry, with early termination counted as
failure — and we release the achieved-state export instrumentation for SONIC that makes
c_exec measurable at all (with the reviewed off-by-one at rollout end corrected). Two gates
follow: c_ref ≥ τ(d) (executable) and c_ref ≥ 0.18 + τ(d) (retains margin). We report the
risk–coverage curve of the acceptance rule rather than a single rate, with a pre-declared
acceptance target: **≥ 95 % executed success among gate-accepted trajectories** at the
declared coverage — the gate is only a certificate if what it accepts survives dynamics.
The demo makes the same point visually: a naive accepted trajectory whose tracking drift
clips the beam, side by side with the τ(d)-gated one passing cleanly (Fig. 6 / video).

**Instrumentation validated ⟨smoke, 2 clips⟩.** The achieved-state export runs end-to-end
(58 s per SONIC launch; per-motion joins and 50 Hz sampling verified), and it empirically
confirmed the defect our pre-run review predicted: the final snapshot of a successful rollout
is a post-reset teleport (6.875 m root jump on the last interval vs 6.5 mm the interval
before), removed by the consumer-side truncation. The first two executed-clearance readings
preview §7's stakes: on an upright detour, tracking drift consumed 0.251 m of a 0.270 m
certified margin (executed minimum 18.6 mm, with a lateral contact the kinematic certificate
never sees); a deep-duck reference terminated at 50 % progress. Kinematic acceptance without
τ(d) is not a safety statement — which is the point of this section. ⟨Full 36-scene × 8-seed
1B table and τ(d) fit slot here.⟩

## 8. The artifact: harness, ledgers, defects, demo

**Reproducibility as a measured quantity.** Every experiment writes a per-candidate ledger
(raw evidence, not summaries); every Table-2 row carries checkpoint SHA-256s, the code
commit, and the tracked-diff hash; the clip cache is keyed by schedule hash, seed, sampler
version, and repair iteration, so a pre-repair clip cannot be served for a post-repair claim.
The audit toolkit's acceptance test is that it **reproduces our own wrong answers** (10.00
naive, 1.67 calibrated) from the shipped ledgers.

**The defect catalogue.** ~30 measurement, control, and harness defects, most favoring the
hypothesis under test, each with the control that caught it — including a sampler defect
(per-window latent replay) found by an independent implementation agent and verified against
the model's autoregressive loop, after which every affected number was relabeled and is being
regenerated under the fixed sampler. We believe this catalogue is a contribution in itself:
it is what "evaluating a generative robot prior carefully" actually costs.

**Demo (Fig. 3, video).** Four scenes through the full loop under v2: (a) proposal 15 mm
short → one repair → accepted at 187 mm; (b) five OOD beams, proposal at 40 mm → two repairs
→ 180 mm; (c) stay-upright preference routes *around* the same beam family, 270 mm clearance,
no repair; (d) a 0.60 m beam → quantified refusal. Each with the attempt ladder, per-position
clearance trace, and rendered MuJoCo video of the exact trajectory the verifier scored.

## 9. The pipeline as a verified-dataset generator

The closed loop converts a frozen off-the-shelf prior into a **production line for verified
traversal data**. Each accepted record carries: the scene (procedural parameters + geometry),
start/goal and preference, the planned route with its alternatives, the command schedule and
every repair step (with hashes), the full-body 25 fps trajectory, the *measured* per-position
overhead/lateral clearance of that exact trajectory, goal error, sampler/checkpoint/code
provenance, and — where the physics stage has run — tracked outcome. Refusals are records
too, with the binding constraint and its deficit: a navigation dataset that knows what it
*cannot* contain is rarer than one that only contains successes. Breadth is not the premise
here either: even with duck as the single robust axis, a pipeline that emits physically
verified, refusal-labeled traversals of arbitrarily hard multi-obstacle scenes at this rate
is directly useful training and benchmarking substrate for end-to-end RL and distilled
policies — the label quality, not the verb count, is what such consumers lack today.

Three properties make this corpus useful beyond this paper. **Scale:** at ~0.5 s per verified
candidate, one consumer GPU yields on the order of 10⁵ verified, labeled traversals per day
across procedurally sampled scenes. **Trust:** the audit protocol (§4) guarantees the
generator only emits behaviors it can reliably produce, the verification layer guarantees
every clearance label is measured rather than requested, and the defect catalogue (§8)
documents exactly which measurement mistakes the harness defends against. **Coverage
control:** the scene grammar is parametric, so difficulty (beam count, height, aperture,
route topology) is a sampling knob, and the counterfactual-ladder construction supports
causal slices, not just aggregate statistics. A pilot corpus generated with Table 2's winning
configuration (heuristic + ≤2 repairs) accompanies the release: **300 fully randomized
scenes in 302 s on a GPU shared with a training job — 268 verified traversals (192 meeting
the full 18 cm margin, 76 collision-free below it and labeled as such), 6 rejections after
repair, 26 refusals each carrying its binding constraint and deficit** — 13 MB compressed
including per-position clearance traces, i.e. ~86 000 scenes per GPU-day even under
contention. Intended uses include
benchmarking scene-aware motion generation, training scene-conditioned navigation and
whole-body planning models, and — as future work we deliberately do not claim here —
privileged-to-proprioceptive distillation of a student policy from SONIC-tracked rollouts of
the verified references.

## 10. Limitations

The counting and channel-asymmetry results now replicate on Kimodo-G1 (reduced audit);
executability, repair, and the certificate remain single-prior, single-robot,
single-tracker. Scenes are procedural corridor ladders,
not scans; topology-level hold-out and non-beam geometry are staged. The certificate is
kinematic until §7's calibration lands. The executable repertoire is duck: the
position-channel step-over was killed on a pre-registered 24-seed ladder (§6), and richer
request families (rotation scaffolds, text × keyframes) remain open but unproven. We say
"scene-conditioned clearance adaptation", not "general obstacle traversal". Historical v1 numbers are labeled and quarantined; headline claims ride
only on v2 runs. Single-scene demo clips are one seed by design; every scientific table is
multi-seed.

---

### Figures & tables to produce
- **Fig. 1** System diagram: scene+start/goal → route ×3 preferences → proposal → ARDY →
  directional verify → repair ≤2 → accept/refuse (assets exist in the demo page).
- **Fig. 2** The counting funnel + audit protocol schematic (Table 1 inline).
- **Table 2** phase4e 8-seed matrix ⟨run in flight⟩ + McNemar/regression/executable columns.
- **Fig. 3** Demo quartet: stills + attempt ladders + clearance traces (rendered, in page).
- **Fig. 4** Channel signal-to-noise chart — the contrast IS the figure: root height clean
  and executable, position targets drowned in the seed-noise floor (≤1.6 σ, sign-flipping),
  rotation at ~41 σ; local-root mechanism inset.
- **Fig. 5** τ(d) risk–coverage with the ≥95 % executed-success acceptance point marked
  ⟨after 1B⟩.
- **Fig. 6** Side-by-side physics: naive accepted trajectory drifting into the beam vs
  τ(d)-gated trajectory passing ⟨after 1B; rendered from achieved states⟩.
- **Appendix** defect catalogue **as a table: phenomenon → the false positive/negative it
  would have produced → the control that caught it**; claim ledger; prompt/constraint
  programs; release manifest.

### Writing debts (punch list, from guidance 2026-08-31)
- [x] **Table 2 landed** (`outputs/phase4e_architecture_v2_s8`, 4 320 runs, 2 242 s):
      zero regressions in every paired arm at stated at-risk n; McNemar p down to 10⁻³⁶;
      kill condition fired — learned scheduler removed from the system.
- [ ] §7 numbers after the state-export last-frame fix and the 36-scene replay; hold the
      gate to the ≥95 % executed-success target or say plainly that it missed and why.
- [x] **Kimodo-G1 replication (1D) landed**: 4.5× naive-vs-calibrated overcount, duck
      stable / lateral unstable — in §4 main text with its caveats.
- [ ] Fig. 3 four-panel exactly as the demo shows it: slight-deficit repair · OOD two-round
      repair · topology detour · quantified refusal. Fig. 4 as the SNR contrast. Fig. 6
      side-by-side physics.
- [ ] Appendix defect table (phenomenon → biased conclusion it would have produced → the
      control that caught it) generated from the ledger, not retyped.
- [x] **Step-over 1C landed — claim killed** (24 seeds, per-seed gating, pre-registered bar:
      no amplitude reaches 0.5 tracked success; 0.91–0.98 bilateral flight at every dose).
      §6 and §10 updated; text × constraint composition (2A) remains the open door.
