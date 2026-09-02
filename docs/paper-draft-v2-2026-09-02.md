<!--
Scene2Motion paper draft v2.1 (2026-09-02), after the advisor's second review
(docs/pi-advice-2026-09-02-b.md). Evidence cutoff: repository receipts at HEAD on 2026-09-02.
Bold bracketed tokens are slots for results that have not landed; fill only from the named
receipt. Layout budget (ICRA 2027, 8 pages incl. references):
I 1.0 · II 0.6 · III 1.0 · IV 0.9 · V 0.6 · VI 2.3 · VII 0.6 · VIII + refs 1.0.
Definitions and vocabulary: docs/project-goal-2026-09-02.md. Every number has a receipt path
in docs/REPORT.md or the ledger cited inline.
-->

# Scene2Motion: Evaluating Generated Humanoid Motions for Obstacle Traversal

**Anonymous Authors** · ICRA 2027 submission draft v2.1

## Abstract

Generating a humanoid motion is not enough to traverse an obstacle: the motion must occur at
the right place and remain collision-free when tracked. We present Scene2Motion, a framework
for evaluating and repairing generated motions while keeping the motion model and controller
fixed. Using ARDY-G1 and SONIC in simulation, we separate motion generation, obstacle-relative
clearance, and clearance after tracking. Of 64 step-prompted references, 12 cleared a 5 cm box
at a position selected after inspecting the motions; none retained clearance after tracking.
Tracking ran without the obstacle, with recorded robot states subsequently checked against its
geometry. A foot-support screen flagged all 53 references whose rollouts triggered the
configured tracking-error cutoffs, while also rejecting three of the eleven runs that did not
terminate early. This analysis predicts evaluator behavior, not whether the robot would fall.
For ducking motions, we use measured clearance errors to adjust the generation controls. Across
36 beam scenes and eight seeds, up to two repairs increased collision-free references from
72.9% to 99.3% for a learned proposal model, compared with 77.4% for resampling under the same
three-attempt limit. These improvements concern reference geometry; successful obstacle
traversal has not yet been demonstrated under the tested tracking protocols. The results
identify limits of the current generator–controller combination and show where measured
feedback improves clearance. Prospective screen validation and obstacle-present execution
remain future work.

## I. Introduction

A humanoid robot walking down a corridor meets a beam at chest height and, a few metres later,
a box on the floor. To pass, it must duck under the first and step over the second, and it must
do each where the obstacle is. Humanoid controllers now track a wide range of generated motions
in open space, and text-conditioned motion models produce a convincing duck or a convincing step
on request. It is tempting to join the two: ask the model for the motion, hand the result to the
controller, and let the robot move through clutter without retraining either component.

A motion can resemble a step or a duck without clearing the intended obstacle. Even a reference
that clears the obstacle may lose clearance when tracked by a humanoid controller. This paper
separates these failures by measuring reference geometry, controller outcomes and achieved
clearance, for one frozen motion model (ARDY-G1 [1]) and one frozen whole-body controller
(SONIC [7]) on the Unitree G1. In our experiments the model decides when the foot lifts, early
and independently of where the box is; a stepping reference that clears a box somewhere along the
route rarely clears the box at the position the scene specifies. When a reference does clear the
box, the controller's evaluator stops the rollout within a fraction of a second of the reference
entering a long period in which neither foot meets a support test. A collision-free reference,
a completed tracking run and a successful traversal are three different results, and evidence
for one is not evidence for the next.

Scene2Motion is a measurement layer between the two frozen components. For each generated
reference it asks, in order: was the requested motion produced; does it occur at the obstacle
position the scene specifies; does it clear the obstacle geometry and pass a reference-only
screen that predicts whether the controller's evaluator will stop the rollout; and, after
tracking, do the recorded robot states still clear the obstacle. Where a reference fails on a
small, monotone geometric error, as ducking references do, the layer corrects it by adjusting
the model's own controls from the measured error. Where it fails the support screen, as the
stepping references do, the layer rejects it and records why. Every candidate becomes a record
with its measurements and, where applicable, its rejection reason.

**Contributions.** (i) An evaluation protocol for generated humanoid motions that separates
placement, reference clearance, tracking completion and clearance after tracking, with
obstacle-relative scoring at a fixed scene position. (ii) A reference-only screen that flagged
every early termination of the tested controller's evaluator on 64 stepping references, whose
prospective test on 128 fresh references is in progress. (iii) A measured clearance correction
for ducking references, evaluated against equal-budget resampling on 36 multi-beam scenes with
its wins and its losses reported. (iv) Motion records with rejection reasons, released with
provenance for every candidate.

Our strongest current contribution is an evaluation of where generated motions fail, plus a
promising correction method. We do not claim a working navigation system. No protocol in this
paper has yet shown a traversal with the obstacle present in the physics scene, and the screen
predicts one evaluator's stopping rule, not whether the robot would fall.

## II. Related Work

Scene2Motion studies how obstacle-relative reference clearance changes under a fixed humanoid
tracker, and evaluates interpretable screening and correction without retraining the motion
generator. Several capabilities it builds on already exist.

**Generated humanoid motion with scene placement.** ARDY [1] and Kimodo [2] generate
long-horizon humanoid motion from text and sparse kinematic controls. MotionBricks [4] goes
further and conditions on object-relative placement and scene-bound interaction keyframes;
MotionStreamer [3] streams compositional motion. What none of them reports, and what we
measure, is whether the requested behaviour clears a stated obstacle at a fixed route position
in a whole-body geometry check, and whether that clearance survives a fixed tracker.

**Motion through clutter.** *Moving Through Clutter* [12] collects scene-aware human motion in
virtual reality across 145 cluttered scenes and benchmarks retargeting and tracking; HumanoidPF
[11] learns obstacle-relative representations for crouching, hurdling and narrow passages. Both
start from the obstacle and state where retargeting or tracking fails. We adopt that structure
with a different input (a pretrained motion model rather than captured motion) and a fixed
controller rather than a learned perceptive policy.

**Physics-aware generation, verification and tracking.** CLoSD [5] closes the loop between a
diffusion planner and a controller; PhysDiff [6] projects generated motion onto physically
plausible motion; SONIC [7] tracks diverse references with one whole-body policy. TEXEDO [10]
selects among candidates with a learned dynamics verifier informed by tracking rollouts, and
SafeFlow [8] and BRIC [9] gate or guide generation with controller-aware checks. Our distinction
is therefore not "generate, then check". It is the specific measurements: an interpretable,
reference-only screen with a named prediction target (the evaluator's cutoff) tested by tracking
flagged and passed references alike; obstacle-relative clearance at a fixed position before and
after tracking; and a measured correction compared with equal-budget resampling, with the
proposer-dependent losses reported.

## III. Problem Setting and Evidence Levels

### A. Components

A frozen motion model $G_\theta$ maps a prompt $u$, a seed $z$ and sparse kinematic controls
$h$ to a reference $x_{\mathrm{ref}}$ at 25 fps. Obstacles are described in route coordinates:
a box of height $h_o$ at route position $s_o$ for stepping, beams at clearance $c_o$ for
ducking. A whole-body collision model of the G1 (Unitree's collision primitives with a measured
4 cm margin) scores any joint trajectory against the obstacle. A frozen controller $T_\phi$
tracks the reference in Isaac Sim; its recorded joint states are scored by the same collision
model. In every tracking protocol reported here the obstacle is absent from the physics scene.

### B. Evidence levels

| level | definition |
|---|---|
| produced | in a whole-body geometry check, the reference clears a 3 cm box at some tested position along the route |
| placed | the reference clears the obstacle at the position the scene specifies, not at a position chosen to fit the clip |
| collision-free reference | the reference clears the obstacle geometry with the stated margin |
| completed tracking run | the controller followed the reference to the end without the evaluator's tracking-error cutoff |
| clearance preserved after tracking | a reference that clears the obstacle yields a tracked trajectory that passes through the obstacle corridor, finishes beyond the obstacle, remains collision-free at the graded height, and satisfies the stated termination rule |

The last definition matters: a geometry query on the recorded states alone is not this
endpoint, because a robot that stops before the box looks collision-free. A sixth level,
traversal with the obstacle present in the physics scene, is not tested in this paper.

### C. The support test and the reference screen

A foot meets the *support test* at a frame when its sole is within 4.65 cm of the ground and its
planar speed is below 1.18 m/s (thresholds calibrated on a corpus of tracker-successful
references before any stepping experiment). Let $r_{\max}$ be the longest period in which
neither foot meets the test. The *reference screen* rejects a reference when
$r_{\max} > 0.20$ s (six or more frames). Its prediction target is the controller evaluator's
stopping rule: SONIC's release evaluation ends an episode when pelvis height error exceeds
0.25 m, pelvis orientation error exceeds 1.0 rad, an ankle or wrist height error exceeds 0.25 m,
or an ankle position error exceeds 0.2 m. The screen is a predictor of that rule. Failing the
support test does not establish zero contact force, a fall, or physical impossibility.

### D. Correcting measured clearance errors

For ducking, the model's root-height control produces a monotone response in overhead
clearance. Given a measured clearance error $e$ (positive when the body intrudes into the beam's
margin) and a forward-secant estimate $g'_+$ of the clearance response, the correction is
$\Delta q = e / |g'_+|$, applied at most twice, with the reference remeasured after each step and
the candidate rejected with its residual error if the response saturates. The correction acts on
the overhead channel only. Stepping motions have no scalar control with a monotone inverse in our
experiments and are not corrected.

## IV. Method: Scene2Motion

Fig. 1 **[FIG-1-PIPELINE]** shows the proposed layer. A route and an obstacle come from the
scene. The model produces a reference from a prompt, a root path along the route and, for
ducking, a root-height schedule from a proposer (a heuristic, a quadratic-programme teacher, or
a small temporal convolutional network trained on the teacher). The reference is measured in
route coordinates: the support test per frame, overhead and lateral clearance, and the
whole-body clearance profile (for each route position, the tallest box the body would clear
there). The obstacle-relative score is read at the scene's position, not at the best position.
In the proposed flow a reference then follows one of three paths: overhead deficit → correct and
remeasure; support violation → reject with reason; clear and passing → send to the controller,
whose recorded states are scored against the same geometry. Every candidate becomes a record:
scene, request, controls, reference measurements, screen result, correction log, evaluator
outcome, achieved-state endpoint, physics seed, and provenance (model revision and checkpoint
hash, runtime commit, robot model hash, controller commit and checkpoint).

**What has been evaluated.** The integrated flow above is the proposed system; the evidence in
§VI comes from separate component studies. We evaluate the support screen and the ducking
correction as separate components. To evaluate the screen, we track both flagged and passed
references; a flow that tracked only passing references could not test its own screen. The
ducking correction is evaluated on reference geometry; its effect on tracked clearance is the
obstacle-present experiment of §VIII.

Two design rules follow from the evidence levels. The obstacle position is fixed by the scene
before the reference is seen; a score at a position chosen after inspecting the clip is
reported separately and labelled. A rejected candidate is stored with its measurements, because
the rejection is a statement about this reference under this controller's evaluator, not about
the scene.

## V. Experimental Setup

**Models.** ARDY-G1 (25 fps, 52-frame horizon, released checkpoint, no weight updates); SONIC
release checkpoint with its `tracking/eval` termination configuration; Unitree G1 in Isaac Sim,
physics seed 0, one rollout per reference unless stated. One RTX 5080 (16 GB).

**Stepping.** One route, one scene. Prompt "A person steps over an obstacle." 64 fresh seeds.
Obstacle: a 5 cm (and 8 cm) box at $s_o = 1.2$ m. The position was chosen after inspecting
the 64 clips as the position that maximises exact clearance on a 0.05 m grid; the fresh-seed
replication in §VI-B uses the same position fixed in advance. Controls: the walk prompt on the
same route; a delayed prompt switch (walk → step at 2.1 s) with a sideways-step positive control.

**Ducking.** 36 out-of-distribution multi-beam scenes (three to six beams, predetermined heights
and gaps) × 8 seeds × three proposers, each in five arms: uncorrected, one correction, two
corrections, and best-of-two and best-of-three resampling with fresh seeds. All arms share a
maximum number of generations (three for two corrections and for best-of-three); the actual
number of generations differs by arm and is reported. Endpoints: collision-free reference, and
the 18 cm clearance margin.

**Units and intervals.** The scene is the inference unit for the multi-scene ducking study
(paired per-scene differences with a 36-scene cluster bootstrap); stepping rates are Wilson
intervals over seeds within one scene and are not generalised beyond it. Post hoc analyses are
labelled post hoc.

## VI. Results

### A. Placement: the stepping motion is produced, but not at the box

In a whole-body geometry check, 44 of 64 step-prompted references cleared a 3 cm box at some
tested position along the route (49 of 64 had positive clearance somewhere). Among those 49,
the reference root reached the location of maximum box clearance after a median of 1.4 s
(frame 35), and 40 of 49 within the first 50 frames. Fig. 2 **[FIG-2-FUNNEL]** shows the
counts at each evidence level for each way of asking. At the scene's box position, 12 of 64
clear a 5 cm box and 11 clear 8 cm. Allowing the box to move within ±0.25 m of its position to
fit each clip raises the 5 cm count to 20 of 64, which describes the clips' capability, not
placement; we report both, labelled.

The other ways of asking place the motion no better. Requesting a foot lift through the model's
position controls over-responds by 1.7–2.3× and yields references in which both feet fail the
support test for most of the lift. Requesting the lift through rotation controls aligned to the
gait phase yields a step 3.2 m from its anchor at half the prompt's amplitude, with negative
compliance to the request. Switching the prompt from walk to step during the rollout yields the
step in 3 of 16 references (against 6 of 8 from the start), 0.8–3.0 s late, none at the
predicted position. The free walk clears nothing (5 mm best box anywhere on the route).

### B. Tracking: no clearance is preserved, and a reference-only screen predicts the cutoff

All 64 references were tracked, flagged and passed alike. The evaluator stopped 53 rollouts; 11
completed. Under the full endpoint (passage through the corridor, finish beyond the obstacle,
collision-free at the graded height, no cutoff) none of the 12 references that clear the box
preserved clearance after tracking (0 of 64), and the count stays zero when the no-cutoff
condition is dropped. All 53 stopped rollouts were upright at the last archived state (pelvis
0.56–0.95 m; none below 0.5 m).

Long periods outside the foot-support test are associated with the tracking-error cutoffs. The
12 clearing references all lift during a period in which neither foot meets the test, lasting
0.44–1.04 s in nine of them and 2.4–3.1 s in three, while the pelvis rises only 0.02–0.26 m. A
simple flight-time comparison (that period is 1.3–15.6× longer than a ballistic flight for the
observed pelvis rise) motivates further physical validation; it is not a dynamics analysis and
does not establish that the motion cannot be performed. Fig. 3 **[FIG-3-SCREEN]** shows that
$r_{\max}$ separates stopped from completed rollouts with AUC 0.997 (bootstrap 0.987–1.00,
post hoc on these 64). The 0.20 s screen, fixed before these references existed, rejects all
53 stopped references and passes 8 of the 11 completed ones, rejecting three. The cutoff time
lands within ±0.2 s of the first such period in 47 of 53 rollouts (median +0.04 s). Maximum
foot lift alone gives AUC 0.92, and $r_{\max}$ still separates the 20 references with no
positive clearance (AUC 0.98), so the screen is not a lift detector. A post hoc sweep finds a
better threshold at eight frames (0.32 s: 51 of 53 rejected, 11 of 11 passed); it is reported
as a sweep, not as the screen.

**Prospective test (in progress).** 128 fresh references (32 seeds × four root-control
settings: free, pinned heading, pinned root height, both) were generated and scored, and the
screen's per-clip predictions were committed before any tracking run. The free setting
replicates the original (23 of 32 produced; 7 of 32 clear the box at the fixed position). No
setting yields a *valid step passing the support test* (a locally valid step and no period over
0.20 s outside the test; 0 of 32 in each), although pinning root height shortens the longest
such period (median 0.48 → 0.36 / 0.22 s). The screen passes 33 and rejects 95. Tracking
outcomes for all 128, flagged and passed, are pending: **[EXP-024-2X2]**, **[EXP-024-AUC]**.
Physical outcome with the cutoff rule removed, on the original 64 (fell / stalled / continued /
cleared): **[EXP-028-OUTCOME]**. Known-trackable walking and jumping controls, which test
whether the screen distinguishes problematic references from legitimate dynamic motion rather
than penalising motion without conventional foot support: **[EXP-024B-CONTROLS]**. Until these
land, the screen is validated retrospectively only.

### C. Repair: correcting measured clearance errors on ducking references

Table I summarises the 36-scene study (288 scene × seed pairs per arm).

| proposer | arm | collision-free | 18 cm margin met | mean generations |
|---|---|---|---|---|
| learned (TCN) | uncorrected | 72.9 % | 0.3 % | 1 |
| | one correction | 91.3 % | 27.8 % | — |
| | two corrections | **99.3 %** | **37.5 %** | 2.72 |
| | best of two (resample) | 76.4 % | 0.7 % | — |
| | best of three (resample) | 77.4 % | 1.4 % | 2.99 |
| QP teacher | uncorrected | 87.2 % | 0.7 % | 1 |
| | two corrections | 100 % | 41.7 % | — |
| | best of three | 97.2 % | 3.1 % | — |
| heuristic | uncorrected | 98.3 % | 53.5 % | 1 |
| | two corrections | 100 % | 64.6 % | 1.84 |
| | best of three | 100 % | **72.6 %** | 1.81 |

Table II gives paired per-scene differences, two corrections minus best-of-three resampling,
with 95 % intervals from a 36-scene cluster bootstrap (30,000 resamples). It is a descriptive
reanalysis of the same rows, not a new experiment, and claims nothing beyond this benchmark.

| proposer, endpoint | difference (pp) | 95 % interval | discordant pairs (correction only : resampling only) |
|---|---|---|---|
| learned (TCN), collision-free | +21.9 | +9.4 to +35.4 | 63 : 0 |
| learned (TCN), 18 cm margin | +36.1 | +25.0 to +47.6 | 104 : 0 |
| QP teacher, 18 cm margin | +38.5 | +28.1 to +48.6 | 112 : 1 |
| heuristic, 18 cm margin | −8.0 | −14.2 to −1.7 | 18 : 41 |

The supported conclusion is that measured correction improves reference clearance for the
learned proposer, but does not outperform resampling for every proposer. The learned proposer's
99.3 % collision-free rate is not a margin result: the 18 cm margin is met in 37.5 % of pairs.
For the heuristic proposer, whose uncorrected references are already almost always
collision-free, resampling reaches a higher margin rate than correction. "Same maximum number of
generations" is not equal computation: the learned proposer used a mean of 2.72 generations
under correction and 2.99 under resampling, the heuristic 1.84 and 1.81. The explanation that
measurement helps because it corrects a systematic error is plausible but untested; the
mechanism test is a fixed extra-crouch adjustment under the same generation budget, and if it
performs equally well the contribution will be framed accordingly.

All of Tables I and II concern reference geometry. An earlier tracking study of 526 ducking
references (859 rollouts) preserved clearance in none, under a protocol in which a 14 s clip cap
prevented most rollouts from reaching the beam; it is a limitation of that protocol, not
evidence about the corrections, and the obstacle-present experiment of §VIII is designed to
answer the question properly.

### D. Records with rejection reasons

Every candidate above is stored with its measurements. For the ducking pilot (300 procedural
scenes, one proposer, at most two corrections), 268 references were collision-free (192 meeting
the 18 cm margin, 76 below it), 6 were rejected after correction with their residual error, and
26 were refused before correction with a reason (saturated response or unreachable clearance).
For stepping, each of the 64 references carries: produced / not produced; clears at the scene
position / does not; $r_{\max}$ and the screen result; evaluator outcome and cutoff time; the
full clearance-after-tracking endpoint. The 128 fresh references carry the reference-side
fields and the screen's prediction; their evaluator outcome and clearance after tracking are
pending. A rejected record is a statement about this reference under this evaluator; it does not
label the scene untraversable. Whether these records improve a downstream navigation policy is a
separate question that this paper does not address.

## VII. Limitations

One motion model, one controller, one robot, physics seed 0. Stepping results come from one
route and one obstacle position, chosen after inspecting the clips; the fresh-seed replication
uses the same position, now fixed in advance. The obstacle is absent from the physics scene in
every tracking protocol; clearance after tracking is a replay of recorded states against the
geometry under the stated endpoint. The screen predicts the evaluator's stopping rule under the
release configuration; the physical outcome without that rule, and the known-trackable controls,
are pending. The corrections are kinematic; no executed benefit from correction has been shown,
and the mechanism baseline has not been run. The ducking tracking study was confounded by its
clip cap. No result here is a hardware result.

## VIII. Conclusion and Next Steps

A motion can resemble a step or a duck without clearing the intended obstacle, and a reference
that clears the obstacle may lose clearance when tracked. Measuring the reference against the
scene finds the placement failure before a rollout is spent, associates the tracking failure
with a reference-only feature, corrects the ducking references whose error is monotone and
geometric, and records the rest with a reason. Three experiments follow, in order: the
prospective screen evaluation with outcomes for every reference, cutoff-free rollouts, and
known-trackable walking and jumping controls, distinguishing evaluator cutoffs from falls; one
decisive ducking experiment with the beam present in the physics scene, comparing uncorrected
generation, resampling, measured correction and a simple fixed correction, reporting passage,
clearance, tracking error, falls and cost separately; and, as future work outside this paper,
whether records with rejection reasons help a navigation policy decide whether to duck, step
over or take another route.

## References

[1] ARDY. [2] Kimodo. [3] MotionStreamer. [4] MotionBricks. [5] CLoSD. [6] PhysDiff.
[7] SONIC / GR00T-WholeBodyControl. [8] SafeFlow. [9] BRIC. [10] TEXEDO. [11] HumanoidPF.
[12] Moving Through Clutter (2026). *(Full entries to be taken from the Codex draft's list.)*
