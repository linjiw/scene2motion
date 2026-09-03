<!--
Scene2Motion paper draft v2.3 (2026-09-03), after the advisor's third review
(docs/pi-advice-2026-09-02-c.md, TEXEDO reading) and the integration of EXP-030
(outputs/exp030_obstacle_present/receipt.json), the first campaign with the obstacle in the
physics scene. Evidence cutoff: repository receipts at HEAD on 2026-09-03. Bold bracketed tokens are slots for results that have not landed; fill only from
the named receipt. Layout budget (ICRA 2027, 8 pages incl. references):
I 1.0 · II 0.7 · III 1.0 · IV 0.8 · V 0.6 · VI 2.3 · VII 0.6 · VIII + refs 1.0.
Definitions, vocabulary and the claims table: docs/project-goal-2026-09-02.md.
Results are organised as four questions, not as experiment identifiers; the identifiers live in
docs/REPORT.md and the protocol files.
Figure numbers follow the asset names in docs/figures/ (fig2 funnel, fig3 channel response, fig4
cost curve, fig5 screen, fig6 screen across families, fig7 repair vs resampling, fig8 outcome
classes; fig1 pipeline is not drawn yet). First citation therefore runs 1-2-3-4-8-5-6-7 and the
numbers must be reordered to citation order at typesetting.
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
geometry. A paired study then placed the box in the physics scene for those same 64 references:
local traversal completion was 0 of 64 with a 5 cm box and 0 of 64 with a 20 cm box, against 1 of
64 in an obstacle-absent control arm on the same references, and the obstacle-absent replay
predicted the obstacle-present outcome class for 63 of 64 references at the 5 cm box (Cohen's
kappa 0.96). A foot-support screen flagged all 53 references whose rollouts triggered the
configured tracking-error cutoffs, while also rejecting three of the eleven runs that did not
terminate early. This analysis
predicts evaluator behavior, not whether the robot would fall.
For ducking motions, we use measured clearance errors to adjust the generation controls. Across
36 beam scenes and eight seeds, up to two repairs increased collision-free references from
72.9% to 99.3% for a learned proposal model, compared with 77.4% for resampling under the same
three-attempt limit. These improvements concern reference geometry. For stepping, local traversal
completion is now a measured zero with the obstacle present; for ducking, traversal completion has
not been measured at all, and the only execution evidence is a replay against beam geometry. The
results identify limits of the current generator–controller combination and show where measured
feedback improves clearance.
Prospective screen validation remains future work, as does obstacle-present execution for the
ducking pipeline.

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
one frozen whole-body controller (SONIC [7]) on the Unitree G1, and measure five things
separately for each generated reference: whether the requested motion was produced; whether it
clears the obstacle at the position the scene specifies; whether it passes a reference-only
screen that predicts the controller evaluator's stopping rule; whether, after tracking, the
recorded robot states still clear the obstacle under a traversal endpoint that requires passage
through the corridor and a finish beyond it; and, for the stepping pool, whether the robot
completes that traversal with the obstacle actually present in the physics scene.

We measure clearance at the specified obstacle position, rather than anywhere along the route,
because a generated motion can clear the same obstacle elsewhere on its path: in our stepping
family the model lifts the foot early and on its own schedule, so 37 of 64 references clear a 5 cm
box somewhere along the route while only 12 clear that same box where the scene puts it. We
measure the support phase of the reference, rather than trusting the tracker to discover problems,
because references that clear the box do so during long periods in which neither foot meets a
support test, and the evaluator stops those rollouts within a fraction of a second. And we report
pool coverage separately from selected success, because in this pool the set of references that
clear the obstacle and the set that complete tracking are **disjoint**: no selector, whatever its
ranking function, could have chosen a candidate that did both.

Where a reference fails on a small, monotone geometric error, as ducking references do, we
correct it by adjusting the model's own controls from the measured error, and compare against
resampling under the same generation budget. Where it fails the support screen, we reject it and
record why.

**Contributions.** (i) An evaluation protocol for generated humanoid motions that separates
production, obstacle-relative placement, tracking completion, clearance after tracking and local
traversal completion with the obstacle in the physics scene, with scoring at a fixed scene
position and a traversal endpoint that a robot stopping short cannot satisfy; the last level is
measured on the stepping pool, and the same study checks the obstacle-absent replay the stepping
execution results rest on against physics, per reference, at one box height. (ii) A
coverage-versus-selection decomposition showing that, for the stepping family, the limitation is
the candidate pool rather than the choice among candidates. (iii) A
reference-only screen that flagged every early termination of the tested controller's evaluator
on 64 references, with a prospective test on 128 fresh references in progress. (iv) A measured
clearance correction for ducking references, evaluated against equal-budget resampling on 36
multi-beam scenes with its wins and its losses reported.

Our strongest current contribution is an evaluation of where generated motions fail, plus a
promising correction method. We do not claim a working traversal or navigation system:
with the box in the physics scene, none of the 64 stepping references completed the traversal.
Most measurements here track the reference with the obstacle absent from the physics scene and
score the recorded states against its geometry afterwards; one campaign (§VI-D) puts the box in
the scene for the stepping pool, which both measures that zero directly and tests the replay
proxy the stepping execution results depend on. The screen predicts one evaluator's stopping rule,
not whether the robot would fall.

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
reference in Isaac Sim; its recorded joint states are scored by the same collision model. The
obstacle is absent from the physics scene in the campaigns behind most results reported here, and
the recorded states are scored against its geometry afterwards; one campaign (§VI-D) spawns the
box as a collidable body for the 64 stepping references, so that the robot contacts it and the
traversal outcome is measured rather than replayed.

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
| local traversal completion | the same, with the obstacle present in the physics scene, no prohibited contact, no fall, within the time limit — measured for the stepping pool in §VI-D: **0 of 64** with a 5 cm box and 0 of 64 with a 20 cm box |

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

A foot meets the *support test* at a frame when its sole is within 4.65 cm of the ground and
its planar speed is below 1.18 m/s (thresholds calibrated on 284 clips from the control arms of
an earlier position-channel study: 144 model references at 25 fps together with the 140
rollouts the controller completed from them, recorded at 50 Hz). The thresholds were frozen
before the 64 step-prompted references of §VI-A and the 128 fresh references of §VI-C were
generated. The position-channel and ducking corpora used for the cross-family checks in §VI-C
were generated before the calibration, and that position-channel study's control arms are the
calibration corpus itself, so those checks are reported in §VI-C as post hoc. Let $r_{\max}$ be
the longest period in which neither foot meets the test. The *reference screen* rejects a
reference when $r_{\max} > 0.20$ s. Its prediction target is the controller evaluator's
stopping rule: SONIC's release evaluation ends an episode when pelvis height error exceeds 0.25
m, pelvis orientation error exceeds 1.0 rad, an ankle or wrist height error exceeds 0.25 m, or
an ankle position error exceeds 0.2 m. The screen is a predictor of that rule. Failing the
support test does not establish zero contact force, a fall, or physical impossibility, and a
calibrated score is not a physical guarantee.

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
ducking correction is evaluated on reference geometry; its effect on tracked clearance awaits the
beam-present ducking study proposed in §VIII.

## V. Experimental Setup

**Models.** ARDY-G1 (25 fps, 52-frame horizon, released checkpoint, no weight updates); SONIC
release checkpoint with its `tracking/eval` termination configuration; Unitree G1 in Isaac Sim,
physics seed 0, one rollout per reference unless stated. One RTX 5080 (16 GB).

**Stepping.** One route, one scene. Prompt "A person steps over an obstacle." 64 fresh seeds.
Obstacle: a 5 cm (and 8 cm) box at $s_o = 1.2$ m. The position was chosen after inspecting the
64 clips as the position maximising exact clearance on a 0.05 m grid; the fresh-seed replication
in §VI-C uses the same position fixed in advance. The same 64 references were tracked a second
time on a controller build that spawns the box as a collidable body of the same geometry at
$s_o$, in three arms on identical references and the same physics seed: no box, a 5 cm box and a
20 cm box. Traversal scene: start at the route origin, goal at 7.2 m with a 0.5 m tolerance,
corridor half-width 1.4 m, no time limit configured. Controls: the walk prompt on the same route; a
delayed prompt switch (walk → step at 2.1 s). Its sideways-step positive control was refused at
its preregistered substrate gate — 0 of 8 references carried the sidestep composite against a
required 4 — so the delayed-switch arm has no working positive control. The arm itself was
retained by design, the absence of the step being an outcome rather than a gate.

**Ducking.** 36 out-of-distribution multi-beam scenes (three to six beams, predetermined heights
and gaps) × 8 seeds × three proposers, each in five arms: uncorrected, one correction, two
corrections, and best-of-two and best-of-three resampling with fresh seeds. Budgets differ by
arm: one generation uncorrected, two for one correction and for best-of-two, three for two
corrections and for best-of-three. Only the two-correction and best-of-three arms, which Table IV
compares, are equal-budget, and even there the number of generations actually used differs and is
reported. Endpoints: collision-free reference, and the 18 cm clearance margin.

**Units and intervals.** The scene is the inference unit for the multi-scene ducking study
(paired per-scene differences with a 36-scene cluster bootstrap); stepping rates are Wilson
intervals over seeds within one scene and are not generalised beyond it. Analyses chosen after
seeing outcomes are labelled post hoc.

## VI. Results

The results answer four questions in order: does the pool contain a suitable motion; could a
selector find one; does correction create one; and does any of it survive execution. Two
subsections are placed where their evidence sits rather than where the question falls: §VI-C
explains why the tracked stepping references fail, and §VI-D answers the fourth question for the
stepping pool, whose obstacle-present measurement belongs beside the replay it validates.

### A. Does the generated pool contain a motion for the specified obstacle?

In a whole-body geometry check, 44 of 64 step-prompted references cleared a 3 cm box at some
tested position along the route (49 of 64 had positive clearance somewhere). Among those 49,
the reference root reached the location of maximum box clearance after a median of 1.4 s (frame
35), and 40 of 49 within the first 50 frames: the model produces the motion early and on its
own schedule. At the scene's box position, 12 of 64 clear a 5 cm box and 11 clear 8 cm, while
37 of 64 clear that same 5 cm box somewhere along the route. Allowing the box to move to
whichever position suits each clip — the union over a 0.05 m grid of centres within ±0.25 m of
the scene's, that is the eleven positions from 0.95 m to 1.45 m — raises the 5 cm count to 24
of 64, and to 17 of 64 within ±0.10 m. Those are counts of what the clips can do somewhere, not
of placement: no single obstacle position achieves them, the best single position on that grid
is the scene's own with 12, and such a count must never be read as a success probability for a
fixed obstacle. We report both, labelled. Fig. 2 **[FIG-2-FUNNEL]** gives the counts at each
evidence level for each way of asking.

The other ways of asking place the motion no better; Fig. 3 **[FIG-3-CHANNELS]** collects their
responses. Requesting a foot lift through the model's position controls over-responds by
1.7–2.3× and yields references in which both feet fail the support test for most of the lift.
Requesting the lift through rotation controls aligned to the gait phase yields a step a median
2.30 m from its anchor (mean 2.06 m; 3.2 m is the worst of the eight seeds, and the
residual-anchored form of the same request gives a median 1.32 m) at half the prompt's amplitude,
with negative compliance to the request. Switching the prompt from walk to step during the rollout
yields the step in 3 of 16 references, against 6 of 8 from the start, 0.8–3.0 s late, none at the
predicted position; that arm's sideways-step positive control was refused at its substrate gate,
so its negative reading is bounded by a missing control. The free walk clears nothing: the tallest
box it clears anywhere on the route is 5 mm.

The position channel was also tracked, and its ladder is the clearest measurement we have of
the trade between clearing the obstacle in the reference and completing the tracking run. Over
six requested amplitudes, 24 fresh seeds each and one rollout per reference, the reference
geometry improves with the request — at a fixed probe position 3.6 m along the route the median
tallest box the body clears rises from 0 m at the smallest request to 0.32 m at the largest —
while completed tracking runs fall from 6 of 24 to 2 of 24, then 1 of 24, and then 0 of 24 at
every amplitude of 0.16 m and above. Matched control references on the same seeds complete 22
to 24 of 24 at every rung, so this is the request and not the route. No rung met the campaign's
preregistered survival condition — tracked success of at least 0.5 together with a median
clearance of at least 0.05 m at the clip's own lift peak, at 24 seeds — and its receipt records
the verdict "claim KILLED on this ladder", closing the position channel as a route to placed
clearance. Asking harder for the clearance buys reference geometry and spends execution.

At the reference level, then, sampling helps: reaching 90 % coverage of a 5 cm clearance takes
about 12 fresh draws at the observed rate (13 at 8 cm); Fig. 4 **[FIG-4-COST]** gives the
coverage curves and the staged/unstaged contrast. That budget belongs to a position chosen
after inspecting the clips. Scored at an unchosen position on the same route — 3.6 m, the same
route position as the ladder's fixed probe, scored here with this study's own 0.20 m-deep
obstacle box — the same 64 references clear the 5 cm box twice, 90 % coverage would take about
73 fresh draws instead of 12, and at 20 cm and above no reference in the pool clears at all.
The placement numbers in this section are therefore a best case for this generator on this
route, not a typical one, and the fresh-seed replication fixes the same position in advance so
that the next measurement is not.

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
rollouts were upright at the last archived state (pelvis 0.56–0.95 m; none below 0.5 m), though
one of the 53 has a torso up-axis below 0.92, a tilt of more than 23° from vertical, still clear
of the 0.70 up-axis this paper's outcome classes count as a fall. The cutoff is therefore a
tracking-error rule firing, not an observed fall.

Scored against the whole traversal problem — start at the route origin, goal 7.2 m away, the
box at the specified position — every rollout receives one outcome class rather than a boolean.
Table II gives the breakdown and Fig. 8 **[FIG-8-OUTCOMES]** the same replay classes across the
six box heights; a boolean endpoint would merge the first two columns. Both are computed from
rollouts recorded with the obstacle absent from the physics scene and scored afterwards against
its geometry; §VI-D gives the obstacle-present measurement.

| box height | fell | intersects the box (replay) | hit a wall | evaluator cutoff | timeout | stalled | **completed (replay)** |
|---|---|---|---|---|---|---|---|
| 3 cm | 0 | 21 | — | 43 | — | 0 | **0** |
| 5 cm | 0 | 21 | — | 43 | — | 0 | **0** |
| 8 cm | 0 | 22 | — | 42 | — | 0 | **0** |
| 12 cm | 0 | 25 | — | 39 | — | 0 | **0** |
| 20 cm | 0 | 30 | — | 34 | — | 0 | **0** |
| 30 cm | 0 | 31 | — | 33 | — | 0 | **0** |

At the 5 cm box, 21 of 64 achieved trajectories intersect the obstacle's volume and 43 are
classed as evaluator cutoffs; ten further rollouts were also stopped by the evaluator but are
counted under the intersection column, which precedes the cutoff class, so this column's 43 and
the 53 stopped rollouts named above are the same set less those ten. The collision count rises to 31
at 30 cm. No rollout met the fall criteria over its archived samples, and none satisfies the
completion class in replay at any height; with the obstacle absent from the physics scene that
column is not a local-traversal-completion measurement, and §VI-D gives the measurement that is.
Two columns are dashed here and in Table III, in both cases because the class could not fire: no
time limit was configured in either analysis, so the timeout class was not assessed, and the scene
carries a single obstacle box and no wall geometry, so a wall collision could not be recorded. A
dash there records the absence of a measurement, not a measured zero. The stall class is the
residual one, preempted by the collision and cutoff classes above it in the precedence order,
so its zero should be read together with the 50 of 64 rollouts that never reached the obstacle
at all. Completion requires passing the obstacle inside the corridor, so walking around it does
not count. Because these rollouts were tracked with the obstacle absent from the physics scene,
a collision here means the recorded motion intersects the box in replay: the robot never felt
contact, and its controller was never perturbed by one. Rates are over all assigned trials.

The fourth row of Table I and the last column of this table are consistent, and it is worth saying
why. At every box height, all 14 trajectories that pass the corridor and finish beyond the
obstacle also intersect its volume in replay, and exactly one rollout of the 64 reaches the goal
region — that one intersects the box as well. Reaching the far side of the obstacle's position is
not clearing it, which is what separates the corridor-passage row from the endpoint.

### D. Does the traversal complete with the obstacle in the scene?

Table II, and every execution measurement in this paper except the one that follows, tracks the
reference with the obstacle absent and scores the recorded states against its geometry
afterwards. That proxy is testable, and we tested it. The same 64 references were
tracked again on a controller build that spawns the box as a collidable body, in three arms
sharing the references and the physics seed: no box, a 5 cm box at the specified position, and a
20 cm box there. Table III gives the outcome classes over all 64 assigned trials in each arm.

| arm | fell | collides with the box (physics) | hit a wall | evaluator cutoff | timeout | stalled | **completed** |
|---|---|---|---|---|---|---|---|
| no box (control) | 0 | 0 | — | 54 | — | 9 | **1** |
| 5 cm box present | 0 | 20 | — | 44 | — | 0 | **0** |
| 20 cm box present | 0 | 30 | — | 34 | — | 0 | **0** |

Three scope notes belong with this table. First, unlike every other collision count in this paper,
this column counts collisions of a robot that had the box physically in its scene, so the states
scored are those of a robot that ran into it rather than of one that walked an empty world. The
class itself is still scored by the same conservative collision model, which inflates every scene
box by the 4 cm body margin, and no contact sensor was read: 18 of the 20 collisions at 5 cm
(0.040–0.055 m) and 25 of the 30 at 20 cm (0.040–0.068 m) penetrate at or beyond that margin, so
the collision primitives are resting on the box surface, while 2 of 20 and 5 of 30 are shallower
than the margin and need not be primitive contact at all. Second, the timeout and wall columns are
dashed rather than zero for the reason given under Table II: no time limit was configured and the
scene carries no wall geometry, so neither class could fire. Third, the control arm is scored
with no box in its scene at all, which is why rollouts that Table II
would class as intersecting the box appear here as cutoffs, as stalls, or — for the one that
reaches the goal — as a completion; the fall class, as in Table II, is evaluated only over the
archived samples, which for a cut-off rollout end at the cutoff.

Three predictions were registered before the first launch, and all three held.

*The control.* The no-box arm reproduces the earlier obstacle-absent campaign on this build: 63 of
64 references agree on the termination flag, against a preregistered threshold of 58 (54 stopped
here, 53 there; the single disagreement ran to the end of its reference in the earlier campaign
and was stopped after 120 of 397 recorded samples here). The build's box-spawning fix and the run
conditions are inert for the termination flag, though rollout length agrees less closely — 54 of
64 references match on valid length (0.736 to 0.913), ten differing, nine of them by one to ten
frames and the tenth being that flag disagreement — so the two runs match in outcome rather than
frame for frame. That is what licenses comparing this campaign against the earlier obstacle-absent
one on this pool; it says nothing about the other campaigns. One consequence is worth naming: the
reference the two runs disagree on is one of the eleven that completed tracking in the earlier
campaign, so the eleven completed runs of Table I are ten on this build. The disjointness of §VI-B
is unchanged by that, since none of the eleven clears the box either way, but comparisons that
turn on the exact count of completed runs are scoped to the earlier campaign.

*Completion.* Local traversal completion is 0 of 64 with the box present, at both heights (Wilson
0 to 0.057). With no box in the scene one of the 64 rollouts reached the goal region upright and
inside the corridor (1 of 64, Wilson 0.003 to 0.083); passing an obstacle is vacuous in that arm,
because there is no obstacle to pass, so this is a route-completion control rather than a
traversal. It is what gives the zero its meaning: the controller can carry one of these references
the length of the route, and the box removes that. That reference is instructive on its own. With
no box it reaches 7.63 m along the route and completes; with the 5 cm box it again reaches 7.64 m
and again runs uncut, but its closest approach violates the 4 cm clearance margin (3.6 cm
penetration, about 4 mm from the box surface), which the endpoint counts as a failure; the 5 cm
zero therefore turns on a margin violation for this one reference. With the 20 cm box it is
stopped at 1.51 m, at the obstacle.

*The proxy.* The obstacle-absent replay predicts the obstacle-present outcome class for 63 of 64
references (agreement 0.984, Wilson 0.917 to 0.997; Cohen's $\kappa$ 0.964, percentile bootstrap
over the 64 references 0.882 to 1.00). Rescoring the no-box arm's recorded states against the 5 cm
box gives the same split as Table II's 5 cm row — 21 classed as intersecting the box, 43 as
cutoffs — and of those 21, 20 collided with the box in physics, while one was stopped by the
evaluator short of it instead; that reference's replayed trajectory penetrates the box by
1.8 cm, so the
disagreement is a marginal case rather than a class error. All 43 the replay classed as cutoffs
were cutoffs in physics. The comparison has two classes only: neither labelling produced a
completion, so the agreement establishes that replay predicts which failure mode occurs, not that
replay would recognise a successful traversal. The success class was never exercised in either
direction, and the replay endpoint's zeros are not validated by this check.

The box changes the rollout where the rollout reaches it, and not otherwise. The paired change in
maximum achieved root position between the no-box and 5 cm arms has a median of 0.0 m and an
interquartile width below 1 mm; 8 of 64 references lose more than 0.05 m of progress, the largest
by 3.93 m, and one travels 4.52 m further with the box present than without it, a reminder that
the two arms are separate rollouts of a contact-rich system rather than perturbations of one.
With the box removed, the root of 50 of the 64 rollouts never passes the obstacle's position at
1.2 m; 51 do not with the 5 cm box and 60 do not with the 20 cm box. Because the box spans 1.1 to
1.3 m and the collision check is whole-body, 20 rollouts nevertheless registered a collision with
the 5 cm box.

Two conclusions follow, and no more. Local traversal completion has now been measured rather than
inferred, and with a box in the way it is zero. And the obstacle-absent replay endpoint used for
the stepping execution results of §VI-B and §VI-C agrees with physics on this pool at the 5 cm
box, so those results keep their meaning rather than inheriting a caveat; the ducking execution
evidence is a different behaviour family, a different pipeline and a different obstacle, and is
not covered. What does not follow is any claim of a traversal system, or any generalisation beyond
this route, this scene, this obstacle position and this box height: one physics seed, one rollout
per reference, one pool of 64. Nor does the proxy agreeing mean the obstacle is immaterial — 20
references collide with the 5 cm box and 30 with the 20 cm one — only that replay and physics
assign the same class to the same reference.
The outcome classes were scored under the first version of the traversal evaluator, and a
re-score under its corrected version is a separate versioned analysis.

Long periods outside the foot-support test are associated with those cutoffs. The 12 clearing
references all lift during a period in which neither foot meets the test, lasting 0.44–1.04 s in
nine of them and 2.4–3.1 s in three, while the pelvis rises only 0.02–0.26 m. A simple
flight-time comparison — that period is 1.3–15.6× longer than a ballistic flight for the observed
pelvis rise — motivates further physical validation; it is not a dynamics analysis and does not
establish that the motion cannot be performed. Fig. 5 **[FIG-5-SCREEN]** shows that $r_{\max}$
separates stopped from completed rollouts with AUC 0.997 (bootstrap 0.987–1.00, post hoc on
these 64). The 0.20 s screen, fixed before these references existed, rejects all 53 stopped
references and passes 8 of the 11 completed ones, rejecting three.

The two rules disagree completely on this pool. Of the 44 references that clear a 3 cm box
somewhere along the route — the whole producing set — not one passes the screen, and 43 of them
were stopped by the evaluator; of the eight references the screen does pass, all eight completed
their tracking runs. So on these 64 the screen is an almost perfect predictor of the controller's
evaluator and an exact rejector of the whole producing set — the 12 references that clear the box
where the scene puts it included, none of which passes the screen. That is the conflict this
paper reports rather than resolves: a rule that selects what the controller can execute is not a
rule that selects what solves the scene task, and here the two sets do not intersect.

The screen is not merely a lift detector, and the association is tight in time. The cutoff
lands within ±0.2 s of the reference's first period outside the support test in 47 of 53
rollouts (median +0.04 s); in 10 of the 53 the cutoff precedes that onset, three of them by
more than 0.1 s, so the feature is a predictor of the cutoff and not a demonstrated cause of
it. The maximum clearable whole-body box height anywhere on the route gives AUC 0.92 on its own
— a weaker separation than $r_{\max}$ — and $r_{\max}$ still separates the 20 references that
clear no 3 cm box anywhere (15 of them have no positive clearance at all) with AUC 0.98. A post
hoc sweep finds a better threshold at runs of eight frames or longer ($r_{\max} \ge 0.32$ s,
that is $r_{\max} > 0.28$ s: 51 of 53 rejected, 11 of 11 passed), reported as a sweep, not as
the screen.

**Does the screen only describe stepping?** Fig. 6 **[FIG-6-FAMILIES]** measures the same
feature on a second behaviour family, produced through a different conditioning pipeline: 526
ducking references from the heuristic proposer under zero, one and two corrections, whose
crouch comes from a root-height schedule and its measured correction rather than from a step
prompt, across 36 multi-beam scenes, with one tracking rollout scored per reference. There
$r_{\max}$ still ranks the evaluator's cutoffs above chance: AUC 0.674 pooled (36-scene cluster
bootstrap 0.625–0.721) and 0.694 within scene, above 0.5 in all seven preregistered strata
(0.578–0.707). The confound we expected does not account for it. Root speed ranks the same
outcomes at 0.441 (0.328–0.552, spanning chance; the references that completed tracking are the
faster ones), the pooled difference between the support feature and the speed feature is +0.233
(0.120–0.362), and the two correlate at r = −0.18. Within scene the same contrast is +0.09
(−0.02 to +0.21), spanning zero, with the speed feature itself rising to 0.601; the
within-scene figures are computed on the 13 of 36 scenes that carry at least five rollouts of
each outcome, so the separation from the speed confound is established pooled, not within
scene. Crouch depth separates the outcomes pooled (0.707) but falls to 0.565 within scene, so
most of what it carries is between-scene. Between the two stepping corpora the same feature
ranks cutoffs at 0.921 on the 144 lift arms of the position-channel ladder (135 cut off, 9
completed), and an 18-feature logistic fitted on one stepping corpus and applied to the other
ranks at 0.90–0.97 in both directions. The pooled ladder AUC of 0.974 across all 288 references
(139 cut off) spans both arms, of which the 144 control references are the calibration corpus
and carry 140 of the ladder's 149 completed runs, so it is reported for completeness rather
than as independent transfer; on those control references alone, of which only 4 were cut off,
the feature ranks at 0.513, which is why that corpus is reported as calibration and carries no
predictive claim. The screen is therefore not an artefact of the stepping family or of the
text-prompt channel; the learned and quadratic-programme proposers remain untested for it.

It is also much weaker there, and it is not a decision rule for ducking. At its calibrated 0.20 s
threshold it flags 313 of the 344 references whose rollouts ended at the evaluator's cutoff
(sensitivity 0.910, 0.875–0.936) but also 128 of the 182 whose rollouts completed (specificity
0.297, 0.235–0.367), because 441 of the 526 ducking references contain a period longer than 0.20 s
in which neither foot meets the support test. Used as an accept/reject rule on this family it
would reject most of what works; what transfers is the ranking, not the operating point. Two
scopes bound the result. The analysis is post hoc: the feature groups, their primaries and the
direction of the test were fixed before any feature was computed, but the outcomes were already
known. And in that tracking campaign only 200 of 859 rollouts reached the first beam, so most of
the cutoffs the feature ranks occurred before the obstacle.

**Prospective test, in progress.** 128 fresh references (32 seeds × four root-control settings:
free, pinned heading, pinned root height, both) were generated and scored, and the screen's
per-clip predictions were committed before any tracking run. The free setting replicates the
original: 23 of 32 produced, 7 of 32 clear at the fixed position. No setting yields a *valid step
passing the support test* — a locally valid step with no period over 0.20 s outside the test — in
any of its 32 references, and the four settings trade placement against the screen rather than
improving both. Pinning root height does shorten the longest period outside the support test, from
a median of 0.48 s in the free setting to 0.36 s, and to 0.22 s when heading is pinned as well; it
also costs the placement the screen was meant to protect, with references clearing the 5 cm box at
the specified position falling from 7 of 32 to 1 of 32, and to 2 of 32 with both pinned, while the
step is produced in 19 and 11 of 32 rather than 23. Pinning heading alone moves the other way on
both counts: 8 of 32 placed, 27 of 32 produced, and the longest period outside the support test
lengthens to a median of 0.74 s. The screen passes 33 of the 128 and rejects 95, and the settings
whose references pass it most often are the settings that place the motion least often. Which side
of that trade is right is exactly what the tracking outcomes decide, and those outcomes, for all
128 references, flagged and passed alike, are pending: **[EXP-024-2X2]**, **[EXP-024-AUC]**.
Physical outcome with the cutoff rule removed, on the original 64: **[EXP-028-OUTCOME]**.
Known-trackable walking and jumping controls, which test whether the screen distinguishes
problematic references from legitimate dynamic motion rather than penalising motion without
conventional foot support: **[EXP-024B-CONTROLS]**. Until these land, the screen is validated
retrospectively only.

### E. Does correction create a usable candidate?

Where selection has nothing to select, correction is the remaining lever. Ducking references
admit one because their clearance error is scalar-coupled and monotone. Table IV gives the
36-scene study, 288 scene × seed pairs per arm, and Fig. 7 **[FIG-7-REPAIR]** the paired view of
its equal-budget comparison. Every rate is over all assigned trials, so a candidate the loop
rejects rather than passing on for execution counts in the rejection column and never as a
success; here the two columns are exactly complementary, a candidate being rejected precisely when
its reference is not collision-free within the arm's budget.

| proposer | arm | collision-free | rejected before execution | 18 cm margin met | mean generations used |
|---|---|---|---|---|---|
| learned (TCN) | uncorrected | 72.9 % | 27.1 % | 0.3 % | 1 |
| | one correction | 91.3 % | 8.7 % | 27.8 % | — |
| | two corrections | **99.3 %** | 0.7 % | **37.5 %** | 2.72 |
| | best of two (resample) | 76.4 % | 23.6 % | 0.7 % | — |
| | best of three (resample) | 77.4 % | 22.6 % | 1.4 % | 2.99 |
| QP teacher | uncorrected | 87.2 % | 12.9 % | 0.7 % | 1 |
| | two corrections | 100 % | 0 % | 41.7 % | — |
| | best of three | 97.2 % | 2.8 % | 3.1 % | — |
| heuristic | uncorrected | 98.3 % | 1.7 % | 53.5 % | 1 |
| | two corrections | 100 % | 0 % | 64.6 % | 1.84 |
| | best of three | 100 % | 0 % | **72.6 %** | 1.81 |

Table V gives paired per-scene differences, two corrections minus best-of-three resampling,
with 95 % intervals from a 36-scene cluster bootstrap (30,000 resamples). It is a descriptive
reanalysis of the same rows, not a new experiment, and claims nothing beyond this benchmark.

| proposer, endpoint | difference (pp) | 95 % interval | discordant pairs (correction only : resampling only) |
|---|---|---|---|
| learned (TCN), collision-free | +21.9 | +9.4 to +35.4 | 63 : 0 |
| learned (TCN), 18 cm margin | +36.1 | +25.0 to +47.6 | 104 : 0 |
| QP teacher, collision-free | +2.8 | 0.0 to +6.9 | 8 : 0 |
| QP teacher, 18 cm margin | +38.5 | +28.1 to +48.6 | 112 : 1 |
| heuristic, collision-free | 0.0 | 0.0 to 0.0 | 0 : 0 |
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
accordingly. The two collision-free rows for the better-shaped proposers are ceiling effects worth
stating rather than omitting: the QP teacher's references are collision-free in 87.2 % of pairs
uncorrected and reach 100 % under correction against 97.2 % under resampling, a +2.8 pp difference
whose interval touches zero, and the heuristic's reach 100 % either way. The correction's
collision-free advantage is therefore a property of the proposer that starts furthest from the
constraint.

All of Tables IV and V concern reference geometry. Whether a corrected reference completes the
traversal is the beam-present study proposed in §VIII, not yet run.

### F. Does the benefit survive execution?

Not yet answered for ducking, and for stepping there was no benefit to survive: stepping
references are not corrected. §VI-B and §VI-C give the stepping pool a measured zero under an
achieved-state replay endpoint, and §VI-D a measured zero for local traversal completion itself
with the box in the physics scene. For ducking, an earlier tracking study of 526 references
preserved clearance in 0 of its 859 rollouts; 554 of those rollouts ended at the evaluator's
tracking-error cutoff and only 200 reached the first beam, so the endpoint was rarely reachable at
all, and the ~14 s clip length bounds the 305 rollouts that were not cut off.

That study also never touched the proposer whose repair result this paper reports. Its 859
rollouts came from the heuristic proposer only — uncorrected, one correction and two corrections —
so the learned proposer, whose collision-free rate rises from 72.9 % to 99.3 % under two
corrections, has never been tracked at all: it has evidence at the reference-geometry level and at
no level beyond it. The negative execution result bounds the heuristic pipeline; the headline
repair gain is untested in execution in either direction.

No measurement in this paper places the *beam* in the physics scene. The stepping campaign of
§VI-D does place its box there and measures local traversal completion at zero, but that result
covers a different behaviour family, a different conditioning pipeline and a different obstacle,
and it says nothing about whether a corrected ducking reference would complete a traversal. A
deeper crouch could improve geometry while making tracking harder, and that possible loss is a
result we have not yet measured.

### G. Records with rejection reasons

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

One motion model, one controller, one robot, physics seed 0. Stepping results come from one
route and one obstacle position, chosen after inspecting the clips; the fresh-seed replication
uses the same position, fixed in advance. The coverage decomposition of §VI-B is exact for the
realised pool of 64 at one scene and is not a general statement about the model. The obstacle
is absent from the physics scene in the campaigns behind every result here except one: the
stepping pool of §VI-D was tracked a second time with the box spawned as a collidable body, which
measured local traversal completion at zero and found the obstacle-absent replay agreeing with
physics on 63 of those 64 references at the 5 cm box. That validation is bounded in §VI-D — one
pool, one route, one obstacle position, one box height, under the first version of the traversal
evaluator — and does not extend to the ducking results, whose execution evidence remains a replay
against beam geometry. The screen predicts the evaluator's stopping rule under the release
configuration; the physical
outcome without that rule, and the known-trackable controls, are pending. The corrections are
kinematic; no executed benefit has been shown and the mechanism baseline has not been run. In
the ducking tracking study most rollouts ended at the evaluator's cutoff before the first beam,
and it covered the heuristic proposer only, so it bounds that pipeline rather than testing the
learned proposer's correction. No result here is a hardware result, and none of it establishes
navigation.

## VIII. Conclusion and Next Steps

A motion can resemble a step or a duck without clearing the intended obstacle, and a reference
that clears the obstacle may lose clearance when tracked. Measuring the reference against the
scene finds the placement failure before a rollout is spent, associates the tracking failure with
a reference-only feature, and separates a selection problem from a candidate-supply problem: in
our stepping pool the references that clear the obstacle and the rollouts that complete tracking
do not overlap, so no ranking function would have helped. For ducking, correcting the measured
clearance error improves reference geometry for a learned proposer and does not beat resampling
for a well-shaped one. Putting the box in the physics scene turns the stepping outcome from an
inference into a measurement: none of the 64 stepping references completed the local traversal,
one of them completed the same route in an obstacle-absent control arm, and the obstacle-absent
replay the stepping execution results rest on assigned the same outcome class as physics for 63
of the 64 references at the 5 cm box.

Two experiments decide what this line of work becomes. First, complete the prospective screen
evaluation: outcomes for every reference, rollouts without the cutoff rule, and known-trackable
walking and jumping controls, so that evaluator cutoffs are distinguished from falls. Second, run
a shared-pool obstacle-position study for ducking with the beam present in the physics scene:
hold the instruction, start state, controller and candidate pool fixed, move the obstacle, and
compare random selection, a trackability-and-text rule of the kind TEXEDO proposes, geometry-only
selection, the combination, and an offline oracle, reporting pool coverage separately from
selected success and comparing extra sampling, a fixed extra-crouch adjustment and the measured
correction under one budget.

The instrument for the second has been exercised at campaign scale for a floor box carried as a
per-motion table pose — the stepping study of §VI-D spawned a collidable box for all 64 references
across three arms and scored local traversal directly — and scored under the first version of the
traversal evaluator. Spawning an overhead beam, the rollout horizon and the corrected evaluator
still have to be validated before the ducking study can run, and nothing in the stepping result
predicts its outcome.

If correction improves local traversal completion, this becomes a method contribution; if it
improves only reference geometry, the narrower conclusion stands. Downstream navigation policy is
a separate research question and is not promised here.

## References

[1] K. Zhao *et al.*, “ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation,” *ACM Trans. Graph.*, 2026. [Online]. Available: <https://arxiv.org/abs/2607.08741>

[2] D. Rempe *et al.*, “Kimodo: Scaling Controllable Human Motion Generation,” 2026. [Online]. Available: <https://arxiv.org/abs/2603.15546>

[3] L. Xiao *et al.*, “MotionStreamer: Streaming Motion Generation via Diffusion-Based Autoregressive Model in Causal Latent Space,” in *Proc. IEEE/CVF ICCV*, 2025. [Online]. Available: <https://arxiv.org/abs/2503.15451>

[4] T. Wang *et al.*, “MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives,” *ACM Trans. Graph.*, 2026. [Online]. Available: <https://arxiv.org/abs/2604.24833>

[5] G. Tevet *et al.*, “CLoSD: Closing the Loop between Simulation and Diffusion for Multi-Task Character Control,” 2024. [Online]. Available: <https://arxiv.org/abs/2410.03441>

[6] Y. Yuan, J. Song, U. Iqbal, A. Vahdat, and J. Kautz, “PhysDiff: Physics-Guided Human Motion Diffusion Model,” in *Proc. IEEE/CVF ICCV*, 2023. [Online]. Available: <https://arxiv.org/abs/2212.02500>

[7] Z. Luo *et al.*, “SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control,” *Sci. Robot.*, vol. 11, no. 117, 2026. [Online]. Available: <https://arxiv.org/abs/2511.07820>

[8] H. Cho, S.-H. Kim, J. Kang, and D. Koo, “SafeFlow: Real-Time Text-Driven Humanoid Whole-Body Control via Physics-Guided Rectified Flow and Selective Safety Gating,” 2026. [Online]. Available: <https://arxiv.org/abs/2603.23983>

[9] D. Lim, M. Kim, J. Lim, and S. Kim, “BRIC: Bridging Kinematic Plans and Physical Control at Test Time,” in *Proc. AAAI*, 2026. [Online]. Available: <https://arxiv.org/abs/2511.20431>

[10] J. Cao, Y. Chen, Y. Song, M. Tomizuka, C. Li, and T. Tian, “TEXEDO: Test Time Scaling for Controller-Aware Language-Conditioned Humanoid Motion Generation,” 2026. [Online]. Available: <https://arxiv.org/abs/2606.22998>

[11] H. Xue *et al.*, “Collision-Free Humanoid Traversal in Cluttered Indoor Scenes,” 2026. [Online]. Available: <https://arxiv.org/abs/2601.16035>

[12] B. Wang, Y. Lu, L. Wang, L. Yu, and X. Xiao, “Moving Through Clutter: Scaling Data Collection and Benchmarking for 3D Scene-Aware Humanoid Locomotion via Virtual Reality,” 2026. [Online]. Available: <https://arxiv.org/abs/2603.05993>
