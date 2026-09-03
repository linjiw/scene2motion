<!--
Scene2Motion paper draft v2.2 (2026-09-02), after the advisor's third review
(docs/pi-advice-2026-09-02-c.md, TEXEDO reading). Evidence cutoff: repository receipts at HEAD
on 2026-09-02. Bold bracketed tokens are slots for results that have not landed; fill only from
the named receipt. Layout budget (ICRA 2027, 8 pages incl. references):
I 1.0 · II 0.7 · III 1.0 · IV 0.8 · V 0.6 · VI 2.3 · VII 0.6 · VIII + refs 1.0.
Definitions, vocabulary and the claims table: docs/project-goal-2026-09-02.md.
Results are organised as four questions, not as experiment identifiers; the identifiers live in
docs/REPORT.md and the protocol files.
-->

# Scene2Motion: Evaluating Generated Humanoid Motions for Obstacle Traversal

**Anonymous Authors** · ICRA 2027 submission draft v2.2

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

A humanoid may execute a convincing ducking motion and still collide with a low beam if it
stands up before reaching the obstacle. Obstacle traversal therefore requires more than a
plausible motion or accurate tracking: the robot must produce sufficient clearance at the
obstacle's location and continue to the other side. Scene2Motion studies this gap by separating
reference clearance, controller execution, and traversal completion.

The gap is easy to hide. Humanoid controllers now track a wide range of generated motions in
open space, and text-conditioned motion models produce a convincing duck or a convincing step on
request. Joining the two is tempting: ask the model for the motion, hand the result to the
controller, and let the robot move through clutter without retraining either component. But a
motion can resemble a step or a duck without clearing the intended obstacle, and even a
reference that clears the obstacle may lose clearance when tracked.

Selecting well among generated candidates does not close the gap either. Recent work shows that
filtering candidates by predicted execution quality raises tracking success substantially with
both generator and controller frozen [10]. Executing a motion successfully and solving a
scene-specific task are, however, different requirements. Three references can all look relevant
to "duck under an obstacle": a stable duck that ends before the robot reaches the beam, a deep
crouch that clears the beam in the reference but cannot be tracked, and a tracked duck that
passes beneath the beam and reaches the other side. Only the third completes the traversal, and
distinguishing them requires the obstacle's position, the approach state and the required
destination — information a motion-and-language verifier does not receive.

This paper therefore asks: **which generated motions can this controller use to get through this
obstacle, from this starting state?** We instrument one frozen motion model (ARDY-G1 [1]) and
one frozen whole-body controller (SONIC [7]) on the Unitree G1, and measure four things
separately for each generated reference: whether the requested motion was produced; whether it
clears the obstacle at the position the scene specifies; whether it passes a reference-only
screen that predicts the controller evaluator's stopping rule; and whether, after tracking, the
recorded robot states still clear the obstacle under a traversal endpoint that requires passage
through the corridor and a finish beyond it.

We measure clearance at the specified obstacle position, rather than anywhere along the route,
because a generated motion can clear the same obstacle elsewhere on its path: in our stepping
family the model lifts the foot early and on its own schedule, so 44 of 64 references clear a
3 cm box somewhere while only 12 clear a 5 cm box where the scene puts it. We measure the
support phase of the reference, rather than trusting the tracker to discover problems, because
references that clear the box do so during long periods in which neither foot meets a support
test, and the evaluator stops those rollouts within a fraction of a second. And we report pool
coverage separately from selected success, because in this pool the set of references that clear
the obstacle and the set that complete tracking are **disjoint**: no selector, whatever its
ranking function, could have chosen a candidate that did both.

Where a reference fails on a small, monotone geometric error, as ducking references do, we
correct it by adjusting the model's own controls from the measured error, and compare against
resampling under the same generation budget. Where it fails the support screen, we reject it and
record why.

**Contributions.** (i) An evaluation protocol for generated humanoid motions that separates
production, obstacle-relative placement, tracking completion and clearance after tracking, with
scoring at a fixed scene position and a traversal endpoint that a robot stopping short cannot
satisfy. (ii) A coverage-versus-selection decomposition showing that, for the stepping family,
the limitation is the candidate pool rather than the choice among candidates. (iii) A
reference-only screen that flagged every early termination of the tested controller's evaluator
on 64 references, with a prospective test on 128 fresh references in progress. (iv) A measured
clearance correction for ducking references, evaluated against equal-budget resampling on 36
multi-beam scenes with its wins and its losses reported.

Our strongest current contribution is an evaluation of where generated motions fail, plus a
promising correction method. We do not claim a working traversal or navigation system. No
protocol in this paper has yet run with the obstacle present in the physics scene, and the screen
predicts one evaluator's stopping rule, not whether the robot would fall.

## II. Related Work

Scene2Motion studies how obstacle-relative reference clearance changes under a fixed humanoid
tracker, and evaluates interpretable screening and correction without retraining the motion
generator. Several capabilities it builds on already exist.

**Generated humanoid motion, including scene placement.** ARDY [1] and Kimodo [2] generate
long-horizon humanoid motion from text and sparse kinematic controls; MotionStreamer [3] streams
compositional motion. MotionBricks [4] goes further and conditions on object-relative placement
and scene-bound interaction keyframes. Object-relative conditioning is therefore not the missing
ingredient we claim to supply. What we measure, and these works do not report, is whether the
requested behaviour clears a stated obstacle at a fixed route position under a whole-body
geometry check, and whether that clearance survives a fixed tracker.

**Selection under a frozen generator and controller.** TEXEDO [10] samples candidate motions,
scores them with a dynamics verifier trained on controller rollouts, filters by predicted
execution quality and ranks the survivors by instruction match. With 32 candidates it raises
tracking success from 87.3 % to 98.4 % over a single sample while trading against the
instruction-match score, and it retrains neither component. Its lesson — identify two desirable
properties that conflict and state how the decision rule handles the conflict — is the one we
adopt. Its verifiers take motion and language, not scene geometry, and its reported hardware
results establish execution of selected motions rather than obstacle-relative traversal success;
that is a scope distinction, not a defect. Our study adds the third property those two must be
weighed against, namely completing the scene-specific task, and reports pool coverage separately
from selected success so that a selection result cannot be confused with a candidate-supply
result. SafeFlow [8] and BRIC [9] likewise gate or guide generation with controller-aware checks.

**Physics-aware generation and tracking.** CLoSD [5] closes the loop between a diffusion planner
and a controller; PhysDiff [6] projects generated motion onto physically plausible motion; SONIC
[7] tracks diverse references with one whole-body policy. We keep both model and controller
frozen and ask what a cheap measurement layer between them can and cannot recover.

**Motion through clutter.** *Moving Through Clutter* [12] collects scene-aware human motion in
virtual reality across 145 cluttered scenes and benchmarks retargeting and tracking; HumanoidPF
[11] learns obstacle-relative representations for crouching, hurdling and narrow passages. Both
start from the obstacle and state where retargeting or tracking fails. We adopt that structure
with a different input, a pretrained motion model rather than captured motion, and a fixed
controller rather than a learned perceptive policy.

## III. Problem Setting, Endpoints and Evidence Levels

### A. Components

A frozen motion model $G_\theta$ maps a prompt $u$, a seed $z$ and sparse kinematic controls $h$
to a reference $x_{\mathrm{ref}}$ at 25 fps. Obstacles are described in route coordinates: a box
of height $h_o$ at route position $s_o$ for stepping, beams at clearance $c_o$ for ducking. A
whole-body collision model of the G1 (Unitree's collision primitives with a measured 4 cm
margin) scores any joint trajectory against the obstacle. A frozen controller $T_\phi$ tracks the
reference in Isaac Sim; its recorded joint states are scored by the same collision model. In
every tracking protocol reported here the obstacle is absent from the physics scene.

### B. Local traversal is not navigation

**Local traversal** means passing through a specified opening or corridor; walking around it does
not satisfy the task. **Navigation** means reaching a destination through a scene, where walking
around an obstacle may be an excellent solution and ducking or stepping is required only when the
route demands it. Navigation additionally requires route choice, transitions between motions,
replanning and recovery. This paper measures local traversal only, and no result here establishes
navigation.

### C. Evidence levels

| level | definition |
|---|---|
| produced | in a whole-body geometry check, the reference clears a 3 cm box at some tested position along the route |
| placed | the reference clears the obstacle at the position the scene specifies, not at a position chosen to fit the clip |
| collision-free reference | the reference clears the obstacle geometry with the stated margin |
| completed tracking run | the controller followed the reference to the end without the evaluator's tracking-error cutoff |
| clearance preserved after tracking | a reference that clears the obstacle yields a tracked trajectory that passes through the obstacle corridor, finishes beyond the obstacle, remains collision-free at the graded height, and satisfies the stated termination rule |
| local traversal completion | the same, with the obstacle present in the physics scene, no prohibited contact, no fall, within the time limit — **not measured in this paper** |

The fifth definition matters: a geometry query on the recorded states alone is not an endpoint,
because a robot that stops before the box looks collision-free. In our tracked pool an unguarded
query reports 43 of 64 achieved clips clearing the 5 cm box; the guarded endpoint reports zero,
and the difference is robots that never reached the obstacle.

### D. Reporting rules

Every rate is over all assigned trials. A rule that rejects a candidate without executing it
scores that trial as a non-completion; where rejection avoids an unsafe attempt we report that
separately as an outcome class, because rejecting everything must not read as perfect task
performance. Outcomes are broken down as success, collision, fall, evaluator cutoff, timeout or
stall before the obstacle, and rejection. Progress is reported in world-space route distance, not
as pose similarity after subtracting the root position. Cost is reported as generation calls,
scoring seconds and executed seconds, because an equal maximum number of generations is not equal
computation.

### E. The support test and the reference screen

A foot meets the *support test* at a frame when its sole is within 4.65 cm of the ground and its
planar speed is below 1.18 m/s (thresholds calibrated on a corpus of tracker-successful
references before any stepping experiment). Let $r_{\max}$ be the longest period in which neither
foot meets the test. The *reference screen* rejects a reference when $r_{\max} > 0.20$ s. Its
prediction target is the controller evaluator's stopping rule: SONIC's release evaluation ends an
episode when pelvis height error exceeds 0.25 m, pelvis orientation error exceeds 1.0 rad, an
ankle or wrist height error exceeds 0.25 m, or an ankle position error exceeds 0.2 m. The screen
is a predictor of that rule. Failing the support test does not establish zero contact force, a
fall, or physical impossibility, and a calibrated score is not a physical guarantee.

### F. Correcting measured clearance errors

For ducking, the model's root-height control produces a monotone response in overhead clearance.
Given a measured clearance error $e$ (positive when the body intrudes into the beam's margin) and
a forward-secant estimate $g'_+$ of the clearance response, the correction is
$\Delta q = e / |g'_+|$, applied at most twice, with the reference remeasured after each step and
the candidate rejected with its residual error if the response saturates. Stepping motions have
no scalar control with a monotone inverse in our experiments and are not corrected.

## IV. Method: Scene2Motion

Fig. 1 **[FIG-1-PIPELINE]** shows the proposed layer. A route and an obstacle come from the
scene. The model produces a reference from a prompt, a root path along the route and, for
ducking, a root-height schedule from a proposer (a heuristic, a quadratic-programme teacher, or a
small temporal convolutional network trained on the teacher). The reference is measured in route
coordinates: the support test per frame, overhead and lateral clearance, and the whole-body
clearance profile, that is, for each route position the tallest box the body would clear there.
The obstacle-relative score is read at the scene's position, not at the best position. In the
proposed flow a reference then follows one of three paths: overhead deficit, correct and
remeasure; support violation, reject with reason; clear and passing, send to the controller,
whose recorded states are scored against the same geometry. Every candidate becomes a record:
scene, request, controls, reference measurements, screen result, correction log, evaluator
outcome, achieved-state endpoint, physics seed, and provenance.

**What has been evaluated.** The integrated flow above is the proposed system; the evidence in
§VI comes from separate component studies. We evaluate the support screen and the ducking
correction as separate components. To evaluate the screen, we track both flagged and passed
references; a flow that tracked only passing references could not test its own screen. The
ducking correction is evaluated on reference geometry; its effect on tracked clearance is the
obstacle-present experiment of §VIII.

## V. Experimental Setup

**Models.** ARDY-G1 (25 fps, 52-frame horizon, released checkpoint, no weight updates); SONIC
release checkpoint with its `tracking/eval` termination configuration; Unitree G1 in Isaac Sim,
physics seed 0, one rollout per reference unless stated. One RTX 5080 (16 GB).

**Stepping.** One route, one scene. Prompt "A person steps over an obstacle." 64 fresh seeds.
Obstacle: a 5 cm (and 8 cm) box at $s_o = 1.2$ m. The position was chosen after inspecting the
64 clips as the position maximising exact clearance on a 0.05 m grid; the fresh-seed replication
in §VI-C uses the same position fixed in advance. Controls: the walk prompt on the same route; a
delayed prompt switch (walk → step at 2.1 s) with a sideways-step positive control.

**Ducking.** 36 out-of-distribution multi-beam scenes (three to six beams, predetermined heights
and gaps) × 8 seeds × three proposers, each in five arms: uncorrected, one correction, two
corrections, and best-of-two and best-of-three resampling with fresh seeds. All arms share a
maximum of three generations; the number actually used differs by arm and is reported.
Endpoints: collision-free reference, and the 18 cm clearance margin.

**Units and intervals.** The scene is the inference unit for the multi-scene ducking study
(paired per-scene differences with a 36-scene cluster bootstrap); stepping rates are Wilson
intervals over seeds within one scene and are not generalised beyond it. Analyses chosen after
seeing outcomes are labelled post hoc.

## VI. Results

The results answer four questions in order: does the pool contain a suitable motion; could a
selector find one; does correction create one; and does any of it survive execution.

### A. Does the generated pool contain a motion for the specified obstacle?

In a whole-body geometry check, 44 of 64 step-prompted references cleared a 3 cm box at some
tested position along the route (49 of 64 had positive clearance somewhere). Among those 49, the
reference root reached the location of maximum box clearance after a median of 1.4 s (frame 35),
and 40 of 49 within the first 50 frames: the model produces the motion early and on its own
schedule. At the scene's box position, 12 of 64 clear a 5 cm box and 11 clear 8 cm. Allowing the
box to move within ±0.25 m to fit each clip raises the 5 cm count to 20 of 64, which describes
the clips' capability rather than placement; we report both, labelled. Fig. 2
**[FIG-2-FUNNEL]** gives the counts at each evidence level for each way of asking.

The other ways of asking place the motion no better. Requesting a foot lift through the model's
position controls over-responds by 1.7–2.3× and yields references in which both feet fail the
support test for most of the lift. Requesting the lift through rotation controls aligned to the
gait phase yields a step 3.2 m from its anchor at half the prompt's amplitude, with negative
compliance to the request. Switching the prompt from walk to step during the rollout yields the
step in 3 of 16 references, against 6 of 8 from the start, 0.8–3.0 s late, none at the predicted
position. The free walk clears nothing: the tallest box it clears anywhere on the route is 5 mm.

At the reference level, then, sampling helps: reaching 90 % coverage of a 5 cm clearance takes
about 12 fresh draws at the observed rate (13 at 8 cm).

### B. Could a selector have found it?

All 64 references were tracked, flagged and passed alike, because a study that tracked only
passing references could not test its own screen. Table I decomposes the pool at the specified
obstacle.

| pool of 64 candidates, 5 cm box at the specified position | count |
|---|---|
| clears the obstacle in the reference | 12 |
| completes tracking without an evaluator cutoff | 11 |
| **both** | **0** |
| passes the lateral corridor and finishes beyond the obstacle | 14 (11 of them without a cutoff) |
| never reaches the obstacle | 50 |
| satisfies the full clearance-after-tracking endpoint | 0 |

The two useful sets are disjoint. Every reference that clears the box was stopped by the
evaluator, and every rollout that completed came from a reference that does not clear the box.
Selected success is therefore zero for *every* selection rule over this pool, including an
offline oracle with access to all outcomes, and it stays zero at every candidate budget up to the
whole pool. Reference-level coverage improves with sampling while joint coverage does not move at
all. For this stepping family the limitation is the candidate pool, not the choice among
candidates, and a better verifier would not change the outcome. This is the decomposition we
recommend reporting before investing in selection machinery: pool coverage and selected success
answer different questions.

### C. Why the tracked references fail, and whether it can be predicted

Under the full endpoint none of the 12 clearing references preserved clearance after tracking
(0 of 64), and the count stays zero when the no-cutoff condition is dropped. All 53 stopped
rollouts were upright at the last archived state (pelvis 0.56–0.95 m; none below 0.5 m), so the
cutoff is a tracking-error rule firing, not an observed fall.

Scored against the whole traversal problem — start at the route origin, goal 7.2 m away, the
box at the specified position — every rollout receives one outcome class rather than a boolean.
Table I gives the breakdown; a boolean endpoint would merge the first two columns.

| box height | fell | hit the box | hit a wall | evaluator cutoff | timeout | stalled | **completed** |
|---|---|---|---|---|---|---|---|
| 3 cm | 0 | 21 | 0 | 43 | 0 | 0 | **0** |
| 5 cm | 0 | 21 | 0 | 43 | 0 | 0 | **0** |
| 8 cm | 0 | 22 | 0 | 42 | 0 | 0 | **0** |
| 12 cm | 0 | 25 | 0 | 39 | 0 | 0 | **0** |
| 20 cm | 0 | 30 | 0 | 34 | 0 | 0 | **0** |
| 30 cm | 0 | 31 | 0 | 33 | 0 | 0 | **0** |

At the 5 cm box, 21 of 64 achieved trajectories intersect the obstacle's volume and 43 are
stopped by the evaluator before anything else happens; the collision count rises to 31 at 30 cm.
Nothing falls, stalls or times out, and nothing completes the traversal at any height. Completion
requires passing the obstacle inside the corridor, so walking around it does not count. Because
these rollouts were tracked with the obstacle absent from the physics scene, a collision here
means the recorded motion intersects the box in replay: the robot never felt contact, and its
controller was never perturbed by one. Rates are over all assigned trials.

Long periods outside the foot-support test are associated with those cutoffs. The 12 clearing
references all lift during a period in which neither foot meets the test, lasting 0.44–1.04 s in
nine of them and 2.4–3.1 s in three, while the pelvis rises only 0.02–0.26 m. A simple
flight-time comparison — that period is 1.3–15.6× longer than a ballistic flight for the observed
pelvis rise — motivates further physical validation; it is not a dynamics analysis and does not
establish that the motion cannot be performed. Fig. 3 **[FIG-3-SCREEN]** shows that $r_{\max}$
separates stopped from completed rollouts with AUC 0.997 (bootstrap 0.987–1.00, post hoc on
these 64). The 0.20 s screen, fixed before these references existed, rejects all 53 stopped
references and passes 8 of the 11 completed ones, rejecting three. The cutoff lands within
±0.2 s of the first such period in 47 of 53 rollouts (median +0.04 s). Maximum foot lift alone
gives AUC 0.92, and $r_{\max}$ still separates the 20 references with no positive clearance
(AUC 0.98), so the screen is not a lift detector. A post hoc sweep finds a better threshold at
eight frames (0.32 s: 51 of 53 rejected, 11 of 11 passed), reported as a sweep, not as the screen.

**Prospective test, in progress.** 128 fresh references (32 seeds × four root-control settings:
free, pinned heading, pinned root height, both) were generated and scored, and the screen's
per-clip predictions were committed before any tracking run. The free setting replicates the
original: 23 of 32 produced, 7 of 32 clear at the fixed position. No setting yields a *valid step
passing the support test* (a locally valid step and no period over 0.20 s outside the test;
0 of 32 in each), although pinning root height shortens the longest such period (median
0.48 → 0.36 / 0.22 s). The screen passes 33 and rejects 95. Tracking outcomes for all 128,
flagged and passed, are pending: **[EXP-024-2X2]**, **[EXP-024-AUC]**. Physical outcome with the
cutoff rule removed, on the original 64: **[EXP-028-OUTCOME]**. Known-trackable walking and
jumping controls, which test whether the screen distinguishes problematic references from
legitimate dynamic motion rather than penalising motion without conventional foot support:
**[EXP-024B-CONTROLS]**. Until these land, the screen is validated retrospectively only.

### D. Does correction create a usable candidate?

Where selection has nothing to select, correction is the remaining lever. Ducking references
admit one because their clearance error is scalar-coupled and monotone. Table II gives the
36-scene study, 288 scene × seed pairs per arm.

| proposer | arm | collision-free | 18 cm margin met | mean generations used |
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

Table III gives paired per-scene differences, two corrections minus best-of-three resampling,
with 95 % intervals from a 36-scene cluster bootstrap (30,000 resamples). It is a descriptive
reanalysis of the same rows, not a new experiment, and claims nothing beyond this benchmark.

| proposer, endpoint | difference (pp) | 95 % interval | discordant pairs (correction only : resampling only) |
|---|---|---|---|
| learned (TCN), collision-free | +21.9 | +9.4 to +35.4 | 63 : 0 |
| learned (TCN), 18 cm margin | +36.1 | +25.0 to +47.6 | 104 : 0 |
| QP teacher, 18 cm margin | +38.5 | +28.1 to +48.6 | 112 : 1 |
| heuristic, 18 cm margin | −8.0 | −14.2 to −1.7 | 18 : 41 |

Measured correction improves reference clearance for the learned proposer, but does not
outperform resampling for every proposer. The learned proposer's 99.3 % collision-free rate is
not a margin result: the 18 cm margin is met in 37.5 % of pairs. For the heuristic proposer,
whose uncorrected references are already almost always collision-free, resampling reaches a
higher margin rate than correction. The benefit therefore depends on the starting proposer. An
equal maximum of three generations is not equal computation: the learned proposer used 2.72
generations under correction and 2.99 under resampling; the heuristic 1.84 and 1.81. Whether the
gain comes specifically from measurement is untested — a fixed extra-crouch adjustment under the
same budget is the mechanism test, and if it performs as well the contribution must be framed
accordingly.

All of Tables II and III concern reference geometry. Whether a corrected reference completes the
traversal is §VIII's experiment.

### E. Does the benefit survive execution?

Not yet answered for either family. For stepping, §VI-B and §VI-C give a measured zero under an
achieved-state replay endpoint. For ducking, an earlier tracking study of 526 references (859
rollouts) preserved clearance in none, under a protocol in which a 14 s clip cap prevented most
rollouts from reaching the beam; that is a limitation of the protocol, not evidence about the
corrections. No protocol in this paper places the obstacle in the physics scene. A deeper crouch
could improve geometry while making tracking harder, and that possible loss is a result we have
not yet measured.

### F. Records with rejection reasons

Every candidate is stored with its measurements. For the ducking pilot (300 procedural scenes,
one proposer, at most two corrections), 268 references were collision-free (192 meeting the 18 cm
margin, 76 below it), 6 were rejected after correction with their residual error, and 26 were
refused before correction with a reason. For stepping, each of the 64 references carries:
produced or not; clears at the specified position or not; $r_{\max}$ and the screen result;
evaluator outcome and cutoff time; the full clearance-after-tracking endpoint. The 128 fresh
references carry the reference-side fields and the screen's prediction, with evaluator outcome
and clearance after tracking pending. A rejected record is a statement about this reference under
this evaluator; it does not label the scene untraversable. Whether these records improve a
downstream navigation policy is a separate question this paper does not address.

## VII. Limitations

One motion model, one controller, one robot, physics seed 0. Stepping results come from one route
and one obstacle position, chosen after inspecting the clips; the fresh-seed replication uses the
same position, fixed in advance. The coverage decomposition of §VI-B is exact for the realised
pool of 64 at one scene and is not a general statement about the model. The obstacle is absent
from the physics scene in every tracking protocol, so no local traversal completion is measured
anywhere in this paper. The screen predicts the evaluator's stopping rule under the release
configuration; the physical outcome without that rule, and the known-trackable controls, are
pending. The corrections are kinematic; no executed benefit has been shown and the mechanism
baseline has not been run. The ducking tracking study was confounded by its clip cap. No result
here is a hardware result, and none of it establishes navigation.

## VIII. Conclusion and Next Steps

A motion can resemble a step or a duck without clearing the intended obstacle, and a reference
that clears the obstacle may lose clearance when tracked. Measuring the reference against the
scene finds the placement failure before a rollout is spent, associates the tracking failure with
a reference-only feature, and separates a selection problem from a candidate-supply problem: in
our stepping pool the references that clear the obstacle and the rollouts that complete tracking
do not overlap, so no ranking function would have helped. For ducking, correcting the measured
clearance error improves reference geometry for a learned proposer and does not beat resampling
for a well-shaped one.

Two experiments decide what this line of work becomes. First, complete the prospective screen
evaluation: outcomes for every reference, rollouts without the cutoff rule, and known-trackable
walking and jumping controls, so that evaluator cutoffs are distinguished from falls. Second, run
a shared-pool obstacle-position study for ducking with the beam present in the physics scene:
hold the instruction, start state, controller and candidate pool fixed, move the obstacle, and
compare random selection, a trackability-and-text rule of the kind TEXEDO proposes, geometry-only
selection, the combination, and an offline oracle, reporting pool coverage separately from
selected success and comparing extra sampling, a fixed extra-crouch adjustment and the measured
correction under one budget. If correction improves traversal completion, this becomes a method
contribution; if it improves only reference geometry, the narrower conclusion stands. Downstream
navigation policy is a separate research question and is not promised here.

## References

[1] ARDY. [2] Kimodo. [3] MotionStreamer. [4] MotionBricks. [5] CLoSD. [6] PhysDiff.
[7] SONIC / GR00T-WholeBodyControl. [8] SafeFlow. [9] BRIC. [10] TEXEDO. [11] HumanoidPF.
[12] Moving Through Clutter (2026). *(Full entries to be taken from the Codex draft's list.)*
