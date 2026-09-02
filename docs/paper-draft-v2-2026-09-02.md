<!--
Scene2Motion paper draft v2 (2026-09-02), restructured after the advisor's review
(docs/pi-advice-2026-09-02.md). Evidence cutoff: repository receipts at HEAD on 2026-09-02.
Bold bracketed tokens are slots for results that have not landed; they may be filled only from
the named receipt. Layout budget (ICRA 2027, 8 pages incl. references):
I 1.0 · II 0.6 · III 1.0 · IV 1.0 · V 0.6 · VI 2.2 · VII 0.6 · VIII + refs 1.0.
Vocabulary: docs/project-goal-2026-09-02.md. Every number here has a receipt path in
docs/REPORT.md or the ledger cited inline; check both before citing.
-->

# Scene2Motion: Evaluating Generated Humanoid Motions for Obstacle Traversal

**Anonymous Authors** · ICRA 2027 submission draft v2

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
do each at the place where the obstacle is. Humanoid controllers now track a wide range of
generated motions in open space, and text-conditioned motion models can produce a convincing
duck or a convincing step on request. It is tempting to join the two: ask the model for the
motion, hand the result to the controller, and let the robot move through clutter without
retraining either component.

This paper asks whether that works, and finds that it does not yet, for reasons that can be
measured. A generated stepping motion is not a step over *this* box. The model decides when the
foot lifts, and in our experiments it decides early and independently of where the box is. A
stepping reference that clears a box somewhere along the route rarely clears the box at the
position the scene specifies. When a reference does clear the box, the way it clears matters:
the motion model we study lifts the foot during a period when neither foot meets a support
test, and the controller's evaluator stops the rollout within a fraction of a second of that
period beginning. A collision-free reference, a completed tracking run, and a successful
traversal are three different results, and evidence for one is not evidence for the next.

We therefore treat the motion model as an instrument to be measured rather than a planner to
be trusted. Scene2Motion places a measurement layer between a frozen motion model (ARDY-G1
[1]) and a frozen whole-body controller (SONIC [7]) for the Unitree G1. The layer answers,
for each generated reference, four questions in order:

1. Was the requested motion produced?
2. Does it occur at the obstacle position the scene specifies?
3. Does the reference clear the obstacle geometry, and does it pass a screen that predicts
   whether the controller's evaluator will stop the rollout?
4. After tracking, do the recorded robot states still clear the obstacle?

Where the answer to question 3 is a small, monotone geometric error, as it is for ducking under
a beam, the layer corrects the reference by adjusting the model's own controls from the measured
error. Where the answer is a support violation, as it is for the stepping motions we obtained,
the layer rejects the reference and records why. Every candidate, accepted or rejected, becomes
a record with its measurements and its rejection reason.

**Contributions.** (i) An evaluation protocol for generated humanoid motions that separates
placement, reference clearance, tracking completion and clearance after tracking, with
obstacle-relative scoring at a fixed scene position. (ii) A reference screen, computed from the
reference alone, that predicted every early termination of the tested controller's evaluator on
64 stepping references and is being tested prospectively on 128 fresh ones. (iii) A clearance
correction for ducking motions, driven by measured overhead error, evaluated against
equal-budget resampling on 36 multi-beam scenes with its wins and its losses reported. (iv)
Motion records with rejection reasons, released with provenance for every candidate.

We do not claim a working navigation system. No protocol in this paper has yet shown a
traversal with the obstacle present in the physics scene, and the screen predicts one
evaluator's stopping rule, not whether the robot would fall.

## II. Related Work

**Generated humanoid motion.** ARDY [1], Kimodo [2], MotionStreamer [3] and MotionBricks [4]
generate long-horizon humanoid motion from text and sparse kinematic controls. Their conditioning
inputs are statistical requests: a prompt raises the probability of a motif, a waypoint pulls the
root toward a point. None of them scores the result against a scene, and none reports where along
a route the requested behaviour occurs. We measure exactly that.

**Motion through clutter.** *Moving Through Clutter* [12] collects scene-aware human motion in
virtual reality across 145 cluttered scenes and benchmarks retargeting and tracking, and
HumanoidPF [11] learns obstacle-relative representations for crouching, hurdling and narrow
passages. Both start from the obstacle and both state where retargeting or tracking fails. We
follow that structure with a different input: a pretrained motion model rather than captured
human motion, and a fixed controller rather than a learned perceptive policy.

**Physics-aware generation and tracking.** CLoSD [5] closes the loop between a diffusion planner
and a controller; PhysDiff [6] projects generated motion onto physically plausible motion; SONIC
[7] tracks diverse references with one whole-body policy. We keep both model and controller
frozen and ask what a cheap measurement layer between them can and cannot recover.

**Selection and screening.** TEXEDO [10], SafeFlow [8] and BRIC [9] sample candidates and
select or guide them with controller-aware checks. A generator followed by a filter is therefore
not our claim. Our contribution is the obstacle-relative evaluation at a fixed scene position,
the comparison of measured correction against equal-budget resampling, and a screen whose
prediction target (the evaluator's cutoff) is named and tested rather than assumed.

## III. Problem Setting and Evidence Levels

### A. Components

A frozen motion model $G_\theta$ maps a prompt $u$, a seed $z$ and sparse kinematic controls
$h$ to a reference $x_{\mathrm{ref}}$ at 25 fps. Obstacles are described in route coordinates:
a box of height $h_o$ at route position $s_o$ for stepping, a beam at clearance $c_o$ for
ducking. A whole-body collision model of the G1 (Unitree's collision primitives with a measured
4 cm margin) scores any joint trajectory against the obstacle. A frozen controller $T_\phi$
tracks the reference in Isaac Sim and its recorded joint states are scored by the same
collision model. In every tracking protocol reported here the obstacle is absent from the
physics scene; "clearance after tracking" means the recorded states, replayed against the
obstacle geometry, clear it.

### B. Evidence levels

We report five levels and never let one stand in for the next.

| level | definition |
|---|---|
| produced | the reference contains the requested motif (for stepping: a foot lift $\ge 3$ cm anywhere on the route) |
| placed | the reference clears the obstacle at the position the scene specifies, not at a position chosen to fit the clip |
| collision-free reference | the reference clears the obstacle geometry with the stated margin |
| completed tracking run | the controller followed the reference to the end without the evaluator's tracking-error cutoff |
| clearance preserved after tracking | the recorded robot states clear the obstacle geometry |

A sixth level, traversal with the obstacle present in the physics scene, is not tested in this
paper.

### C. The support test and the reference screen

A foot meets the *support test* at a frame when its sole is within 4.65 cm of the ground and its
planar speed is below 1.18 m/s (thresholds calibrated on a corpus of tracker-successful
references before any stepping experiment). Let $r_{\max}$ be the longest run of frames in
which neither foot meets the test. The *reference screen* rejects a reference when
$r_{\max} > 0.20$ s (six or more frames at 25 fps). The screen's prediction target is the
controller evaluator's stopping rule: SONIC's release evaluation ends an episode when pelvis
height error exceeds 0.25 m, pelvis orientation error exceeds 1.0 rad, an ankle or wrist height
error exceeds 0.25 m, or an ankle position error exceeds 0.2 m. The screen predicts that rule.
It does not predict whether the robot would fall, and it is not a physical impossibility test.

### D. Correcting measured clearance errors

For ducking, the model's root-height control produces a monotone response in overhead
clearance. Given a measured clearance error $e$ (positive when the body intrudes into the beam's
margin) and a forward-secant estimate $g'_+$ of the clearance response to the control, the
correction is $\Delta q = e / |g'_+|$, applied at most twice, with the reference remeasured after
each step and the candidate rejected with its residual error if the response saturates. The
correction acts on the overhead channel only. Stepping motions have no scalar control with a
monotone inverse in our experiments and are not corrected.

## IV. Method: Scene2Motion

Fig. 1 **[FIG-1-PIPELINE]** shows the layer. A route and an obstacle come from the scene. The
model produces a reference from a prompt, a root path along the route and, in the ducking
case, a root-height schedule from a proposer (a heuristic, a quadratic-programme teacher, or a
small temporal convolutional network trained on the teacher). The reference is measured in
route coordinates: foot heights, the support test per frame, overhead and lateral clearance, and
the whole-body clearance profile (for each route position, the tallest box the body would
clear there). The obstacle-relative score is read at the scene's position, not at the best
position. A reference then follows one of three paths: overhead deficit → correct and
remeasure; support violation → reject with reason; clear and passing → send to the controller.
The controller's recorded states are scored against the same geometry. Every candidate becomes a
record: scene, request, controls, reference measurements, screen result, correction log,
evaluator outcome, achieved-state clearance, physics seed, and provenance (model revision and
checkpoint hash, runtime commit, robot model hash, controller commit and checkpoint).

Two design rules follow from the evidence levels. First, the obstacle position is fixed by the
scene before the reference is seen; a score at a position chosen after inspecting the clip is
reported separately and labelled. Second, a rejected candidate is stored with its measurements,
because the rejection is a statement about this reference under this controller's evaluator, not
about the scene.

## V. Experimental Setup

**Models.** ARDY-G1 (25 fps, 52-frame horizon, released checkpoint, no weight updates);
SONIC release checkpoint with its `tracking/eval` termination configuration; Unitree G1 in
Isaac Sim, physics seed 0, one rollout per reference unless stated. One RTX 5080 (16 GB).

**Stepping.** One route, one scene. Prompt "A person steps over an obstacle." 64 fresh seeds.
Obstacle: a 5 cm (and 8 cm) box at $s_o = 1.2$ m. The position was chosen after inspecting
the 64 clips as the position that maximises exact clearance on a 0.05 m grid; a prospective
replication on 32 fresh seeds is reported in §VI-B. Controls: the walk prompt on the same route;
a delayed prompt switch (walk → step at 2.1 s) with a sideways-step positive control.

**Ducking.** 36 out-of-distribution multi-beam scenes (three to six beams, predetermined
heights and gaps) × 8 seeds × three proposers, each in five arms: uncorrected, one correction,
two corrections, and best-of-two and best-of-three resampling with fresh seeds. Endpoint:
collision-free reference, and the 18 cm clearance margin.

**Units and intervals.** The scene is the inference unit for the multi-scene ducking study
(cluster bootstrap over scenes); stepping rates are Wilson intervals over seeds within one scene
and are not generalised beyond it. Post hoc analyses are labelled post hoc.

## VI. Results

### A. Placement: the stepping motion is produced, but not at the box

Of 64 step-prompted references, 44 lift a foot at least 3 cm somewhere on the route. The lift
happens early: the crossing of the leading foot over the route's midline has median frame 35
(1.4 s), with 40 of 49 lifting clips lifting inside the first 50 frames. Fig. 2
**[FIG-2-FUNNEL]** shows the counts at each evidence level for each way of asking. At the
scene's box position, 12 of 64 clear a 5 cm box and 11 clear 8 cm. Allowing the box to move
within ±0.25 m of its position to fit each clip raises the count to 20 of 64 at 5 cm, which is
a statement about the clips' capability, not about placement; we report both, labelled.

The other ways of asking place the motion no better. Requesting a foot lift through the
model's position controls over-responds by 1.7–2.3× and produces a period with neither foot on
the ground. Requesting the lift through rotation controls aligned to the gait phase produces a
step 3.2 m from its anchor at half the prompt's amplitude, with negative compliance to the
request. Switching the prompt from walk to step during the rollout produces the step in 3 of 16
references (against 6 of 8 from the start), 0.8–3.0 s late, and none at the predicted position.
The free walk clears nothing (5 mm best box anywhere on the route).

### B. Tracking: no clearance survives, and a reference-only screen predicts the cutoff

All 64 references were tracked. The evaluator stopped 53 rollouts; 11 completed. None of the 12
references that clear the box preserve clearance after tracking (0 of 64 overall). At the last
recorded state every stopped robot is upright (pelvis 0.56–0.95 m; none below 0.5 m), so the
cutoff is the evaluator's tracking-error rule, not a fall.

The 12 clearing references all lift during a period when neither foot meets the support test,
lasting 0.44–1.04 s in nine of them and 2.4–3.1 s in three, while the pelvis rises only
0.02–0.26 m; the period is 1.3–15.6× longer than a ballistic flight for that rise. Fig. 3
**[FIG-3-SCREEN]** shows that $r_{\max}$ separates stopped from completed rollouts with AUC
0.997 (bootstrap 0.987–1.00, post hoc on these 64). The 0.20 s screen, whose threshold was
fixed before these references existed, rejects all 53 stopped references and passes 8 of the 11
completed ones, rejecting three. The cutoff time lands within ±0.2 s of the first such period in
47 of 53 rollouts (median +0.04 s). Maximum foot lift alone gives AUC 0.92, and $r_{\max}$
still separates the 20 non-lifting references (AUC 0.98), so the screen is not a lift detector.
A post hoc sweep finds a better threshold at eight frames (0.32 s: 51 of 53 rejected, 11 of 11
passed); it is reported as a sweep, not as the screen.

**Prospective test.** 128 fresh references (32 seeds × four root-control settings: free,
pinned heading, pinned root height, both) were generated and scored, and the screen's per-clip
predictions were committed before any tracking run. The free setting replicates the original:
23 of 32 produce the lift and 7 of 32 clear the box at the fixed position. No setting yields a
*valid step passing the support test* (0 of 32 in each), although pinning root height shortens
the unsupported period (median 0.48 → 0.36 / 0.22 s). The screen passes 33 and rejects 95.
Tracking results: **[EXP-024-2X2]**, **[EXP-024-AUC]**. Physical outcome with the cutoff rule
removed (fell / stalled / continued / cleared), on the original 64: **[EXP-028-OUTCOME]**.
A known-trackable jumping control: **[EXP-024B-JUMP]**. Until these land, the screen is
validated retrospectively only.

### C. Repair: correcting measured clearance errors on ducking references

Table I summarises the 36-scene study. Rates are over 288 scene × seed pairs.

| proposer | arm | collision-free | 18 cm margin met |
|---|---|---|---|
| learned (TCN) | uncorrected | 72.9 % | 0.3 % |
| | one correction | 91.3 % | 27.8 % |
| | two corrections | **99.3 %** | **37.5 %** |
| | best of two (resample) | 76.4 % | 0.7 % |
| | best of three (resample) | 77.4 % | 1.4 % |
| QP teacher | uncorrected | 87.2 % | 0.7 % |
| | two corrections | 100 % | 41.7 % |
| | best of three | 97.2 % | 3.1 % |
| heuristic | uncorrected | 98.3 % | 53.5 % |
| | two corrections | 100 % | 64.6 % |
| | best of three | 100 % | **72.6 %** |

Two corrections raise the learned proposer's collision-free rate from 72.9 % to 99.3 %, against
77.4 % for three-sample resampling; the paired discordance is 63 scene-seed pairs where only
correction succeeds against none where only resampling does. The same 99.3 % is *not* a margin
result: the 18 cm margin is met in 37.5 % of pairs after two corrections. For the heuristic
proposer, whose uncorrected references are already almost always collision-free, resampling
reaches a higher margin rate (72.6 %) than correction (64.6 %); measurement helps when the
proposal has a systematic error and adds little when it does not. All of Table I concerns
reference geometry. An earlier tracking study of 526 ducking references (859 rollouts) preserved
clearance in none, under a protocol in which a 14 s clip cap prevented most rollouts from
reaching the beam; it is a limitation of that protocol, not evidence about the corrections, and
the obstacle-present ducking experiment of §VIII is designed to answer the question properly.

### D. Records with rejection reasons

Every candidate above is stored with its measurements. For the ducking pilot (300 procedural
scenes, one proposer, at most two corrections), 268 references were collision-free (192 meeting
the 18 cm margin, 76 below it), 6 were rejected after correction with their residual error, and
26 were refused before correction with a reason (saturated response or unreachable clearance).
For stepping, each of the 64 (and 128 fresh) references carries: produced / not produced; clears
at the scene position / does not; $r_{\max}$ and the screen result; evaluator outcome and cutoff
time; clearance after tracking. A rejected record is a statement about this reference under this
evaluator; it does not label the scene untraversable. Whether these records improve a
downstream navigation policy is a separate experiment that this paper does not run.

## VII. Limitations

One motion model, one controller, one robot, physics seed 0. Stepping results come from one
route and one obstacle position, chosen after inspecting the clips; the fresh-seed replication
uses the same position, now fixed in advance. The obstacle is absent from the physics scene in
every tracking protocol; clearance after tracking is a replay of recorded states against the
geometry. The screen predicts the evaluator's stopping rule under the release configuration;
the physical outcome without that rule is pending. The corrections are kinematic; no executed
benefit from correction has been shown. The ducking tracking study was confounded by its clip
cap. No result here is a hardware result.

## VIII. Conclusion and Next Steps

A pretrained humanoid motion model can produce the right motion, but in our experiments it does
not produce it at the right place, and the way it clears the obstacle is one the tested
controller's evaluator stops. Measuring the reference against the scene finds these failures
before a rollout is spent, corrects the ones with a monotone geometric error, and records the
rest with a reason. Three experiments follow, in order: the prospective screen test and
cutoff-free rollouts with a known-trackable jumping control; one ducking experiment with the beam
present in the physics scene comparing uncorrected, corrected and resampled references over
predetermined heights, keeping collisions, failure to reach the beam and evaluator cutoffs
separate; and a test of whether the records, with their rejection labels, help a navigation
policy decide whether to duck, step over or take another route.

## References

[1] ARDY. [2] Kimodo. [3] MotionStreamer. [4] MotionBricks. [5] CLoSD. [6] PhysDiff.
[7] SONIC / GR00T-WholeBodyControl. [8] SafeFlow. [9] BRIC. [10] TEXEDO. [11] HumanoidPF.
[12] Moving Through Clutter (2026). *(Full entries to be taken from the Codex draft's list.)*
