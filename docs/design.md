# Scene2Motion-G1 — design, findings, and where the research actually is

**Working thesis.** A large pretrained humanoid motion prior can be turned into a
*whole-body* navigation planner by learning how 3D scene geometry should modify its root
trajectory and its body pose, with a physics critic closing the kinematic-to-executable gap.

Everything marked **[measured]** was produced by a script in `experiments/` on this machine
and has a receipt in `outputs/<exp>/receipt.json`. Everything else is design or citation.

---

## 0. Read this first: the framing has to change

A recon pass over the 2025-26 literature (31 agents, adversarially verified) found that
several parts of the original proposal are **already published, with real-robot results**:

| Proposed contribution | Already done by |
|---|---|
| Whole-body G1 traversal — duck under beam, narrow gap, step over — **on hardware** | **Gallant** (arXiv 2511.14625), **HumanoidPF** (arXiv 2601.16035, real G1: Crouch-Pass 4/5, Hurdle-Pass 5/5, Side-Pass 5/5) |
| A physics-executability critic closing the kinematic→executable gap | **GenTrack** (arXiv 2608.01410) — frozen **SONIC** on **G1** as a lagged quality judge emitting execution rewards, via FlowGRPO. Same tracker, same robot, ~6 weeks before this work started |
| Motion prior + whole-body SDF guidance on real G1 | **BeyondMimic** (2508.08241) |
| Diffusion generator + RL tracker, terrain-aware G1, real deployment | **arXiv 2604.17335** |
| Scene-conditioning a motion diffusion model via adapters | **SceneAdapt** (2510.13044) |
| Scene-aware G1 traversal dataset + penetration benchmark | **MTC** (2603.05993) |

**So "Scene2Motion-G1: a G1 ducks under a beam" is not a paper.** It is two papers old and
both had hardware.

What survives, and what this codebase is now aimed at:

1. **Explicit, enumerable, controllable traversal strategies.** Current scene-aware
   humanoid systems optimise *execution of a single locally selected behaviour*; they do
   not explicitly construct and evaluate a set of topologically and morphologically
   distinct whole-body strategies for the same scene, start and goal. → **EXP-003**.

   **Do not claim that an RL policy *cannot* be multimodal** — a stochastic policy
   `a_t ~ π(a|s)`, a latent-conditioned policy `π(a|s,z)`, and hierarchical/mixture
   policies all can be, and a reviewer will kill the stronger claim on sight. The
   defensible contribution is that here the strategies are
   **explicit + enumerable + measurable + controllable**, rather than latent accidents of
   policy noise.
2. **Deliberative long-horizon whole-body planning vs. reactive local control.** Gallant
   perceives ±0.8 m; 2604.17335 predicts 0.5 s replanned at 4 Hz; the only true
   confined-space whole-body planner (2608.10220) takes 2–6 min/solve and is sim-only.
   ARDY's `future_constraints_proj` accepts constraints beyond the current generation
   window at 33 ms/step. → the anticipation result in §5 is the first evidence for this.
3. **Language as strategy *preference*, not goal specification.** The
   language-conditioned-humanoid cell is **not empty** — TANGO (whole-body VLA for
   language-conditioned traversal in cluttered indoor scenes), WOLF-VLA (2606.25591) and
   Humanoid-LLA already occupy it, so "first language-controlled humanoid navigation" is
   not available. What those systems express poorly is language that selects among
   *physically distinct ways of getting there* at fixed geometry and fixed goal:
   "take the shortest way" → duck under; "stay upright" → go around; "keep your arms free"
   → avoid the tucked passage. Formally, language modifies a cost over strategies,
   `J(C; l)`, rather than naming the goal.

---

## 1. The stack, as it actually stands

| piece | state |
|---|---|
| ARDY | installed editable in `/home/linjiw/ardy/.venv`, torch 2.11.0+cu128, RTX 5080 (sm_120) |
| ARDY-G1-RP-25FPS-Horizon52 | downloaded, generation verified end to end |
| Text encoder (LLM2Vec / Llama-3-8B) | **CPU** gradio service on `:9550` — it does not fit next to the diffusion model on 16 GB |
| MuJoCo 3.12 | drives exact collision against G1's own primitives |
| Kimodo + `indoor_nav_1k` | 1000 prior G1 clips, reusable |
| SONIC / IsaacLab | present but **never successfully run on this box** (0 `success_manifest.json` repo-wide); the shipped rollout script hard-codes a `miniconda3` path that does not exist here |

Cost **[measured]**: generation **0.18 s per 8 s clip** batched; collision evaluation
**~0.07 s/clip**; the full 112-scene × 3-planner × 2-variant sweep runs in **105 s**. The
data engine is not the bottleneck, so the experiment budget goes into breadth.

---

## 2. The constraint action space

ARDY conditions by overwriting slices of a 414-d per-frame feature vector, gated by a
per-`(frame, channel)` mask. `create_conditions` calls exactly five fills
(`ardy/motion_rep/reps/ardy_motionrep.py:295-299`) — there is no sixth, and `velocities`
and `foot_contacts` have no filler and are **unreachable**:

| channel | feature slice | maskable per | expresses |
|---|---|---|---|
| `root_2d` | `root_pos[0]`, `root_pos[2]` | frame | where to go |
| `root_y_pos` | `root_pos[1]` | frame | **duck** |
| `global_root_heading` | `[3:5]` | frame | **turn / sidle** |
| `global_joints_positions` | `local_joints_positions[5:104]` | (frame, joint) | **tuck / step over** |
| `global_joints_rots` | `global_rot_data[104:308]` | (frame, joint) | limb orientation |

Two properties are load-bearing and easy to get wrong (`ardy_motionrep.py:137` and `:366`,
which agree):

```
local_joints_positions[t, j] = global_pos[t, j] - pelvis_pos[t] + [0, pelvis_y[t], 0]
```

Root-relative **in the ground plane**, **absolute in height**, and *not* rotated into the
heading frame. A world-height target for a joint is therefore directly expressible, which
is exactly what overhead clearance needs.

**Traps.**
- Passing `cfg_weight` as a scalar silently sets the *constraint* CFG weight to 0.0
  (`ardy_model.py:274-275`) — constraints go inert while looking wired up. Always pass a
  2-tuple.
- Constraints are **soft**. Only the root slice is hard-infilled; body constraints are
  never infilled or clamped. This is why joint targets *overshoot* (0.19 m requested →
  0.27 m achieved) rather than saturating.
- ARDY's own constraint classes all copy from a reference *motion*. We author from a
  *plan*, so `scene2motion/constraints.py` speaks the `update_constraints` protocol
  directly.

---

## 3. Three geometric facts that decide whether any of this is valid — EXP-000

1. **G1's head reaches 23.6 cm above ARDY's highest joint. [measured]** ARDY's rig has 34
   joints; standing, its highest is the *shoulder* at 1.078 m. The robot's top is
   **1.314 m**. Every overhead clearance here is smaller than that gap, so a joint-derived
   collision model would certify head-first collisions as clear.
2. **The shipped collision capsules under-cover the real meshes by up to 3.56 cm.
   [measured]** Worst offenders: ankle_roll (3.56), hip_yaw (3.46), and — directly on our
   critical channel — **head_link, 3.29 cm past the head capsule**. `robot.BODY_MARGIN =
   0.04 m` inflates every body extent and every scene box, so "collision-free" means clear
   of the *real* geometry.
3. **The qpos reprojection residual is 1.3 mm mean / 16.2 mm max. [measured]** ARDY
   generates free 3D rotations; the G1 has single-axis hinges, and `MujocoQposConverter`
   projects onto them. We collision-check the *reprojected* pose — correct, since that is
   the pose the robot can actually hold — and this bounds how far the reference we would
   hand a tracker sits from the motion we validated.

The collision pipeline is therefore: ARDY → qpos → MuJoCo FK → distance queries against
Unitree's own primitives, inflated by the measured margin. No hand-rolled geometry.

One trap found and kept: `g1.xml` declares explicit `<pair>` contacts for each foot pad
against the ground plane, and explicit pairs **bypass** `contype`/`conaffinity`. Rather
than fight it we report foot-vs-floor separately — it is a free physical-consistency
signal.

---

## 4. What the frozen prior can be told to do — EXP-001 / 001b / 001c

Measured on ARDY-G1-RP-25FPS-Horizon52, 8 s clips, straight path, 3 seeds, always against a
**matched control** (same seed, same path, no adaptation).

| adaptation | channel | achievable | verdict |
|---|---|---|---|
| **duck** | `root_y_pos`, one scalar/frame | top of robot **1.29 m → 0.755 m**, i.e. **43–53 cm** of overhead clearance; knee 130°→77°; feet planted; forward progress unchanged; path error ~2–3 cm | **strong, monotone, clean** |
| **tuck** | arm joint positions | half-width **−5 to −6 cm** (a 53 cm slot → 41 cm), saturating at tuck≈0.7 | **weak, saturates** |
| **sidle** | `global_root_heading` | 83° of a requested 90° achieved — and **zero** half-width gain | **no gain alone** |
| **step over** | swing-leg joint positions | box height **0 → 16 cm** mean (23 cm best, 2.8 cm worst-seed) | **weak, high variance** |

Three of these were only understood correctly after a first wrong answer, and each mistake
is instructive enough to be worth recording:

- **Sidling makes G1 *wider*, not narrower.** Once the robot is side-on, its arm *swing
  arc* along the direction of travel becomes the corridor width, replacing the shoulder
  width it just removed. Sidling only helps combined with tucking, and even then the best
  measured configuration (sidle 90° + tuck 0.85) reaches 21.9 cm half-width vs 20.5 cm for
  tucking alone. **The lateral channel is essentially saturated.**
- **"Ducking makes you wider" — retracted, then re-confirmed.** EXP-001 first compared each
  clip against its own opening window, which is not a valid control (a gait's opening arm
  swing is wider than its steady state), and under matched control at n = 3 the effect
  vanished. That retraction was itself wrong. EXP-001d re-measured at **n = 20** and the
  coupling is real: mean half-width rises monotonically 0.310 m → 0.403 m across dip 0 →
  0.50, and the 90 % bound rises 0.378 m → 0.510 m. It was invisible at n = 3 because the
  per-level standard deviation is 4–7 cm and the effect is ~9 cm. Three seeds could neither
  establish it nor refute it, and treating the null as a finding was the mistake.
- **Step-over first measured ~0 cm at every requested lift height.** Not because the
  channel fails — targets are followed and overshot — but because only the *nearest* swing
  was lifted, leaving the trailing foot to walk through the obstacle. Lifting both legs
  across the two swings that bracket the crossing recovers a monotone 0 → 16 cm curve.

**The scoping consequence:** vertical adaptation is available from the frozen prior for
free; lateral and step-over adaptation are not. That is exactly where a V2 (adapters /
fine-tuning) would have to earn its keep, and it is a measured claim rather than a guess.

---

## 5. The planner contrast — EXP-002

`scene2motion/planner.py` runs three planners differing *only* in what body volume blocks a
cell: **PELVIS** (only the `[0.60, 0.95] m` band — what `(x, y, θ)` navigation reduces to),
**STANDING** (the full standing volume), and **ADAPTIVE** (A\* over `(x, y, body mode)`,
modes calibrated in §4, worst-case over seeds). 112 scenes, 5 families, 4 seeds/rung.

| planner | variant | plan feasible | goal | collision-free | **end-to-end** |
|---|---|---|---|---|---|
| PELVIS | either | 75.0 % | 100 % | **32.3 %** | 24.2 % |
| STANDING | either | **31.2 %** | 100 % | 100 % | 31.2 % |
| ADAPTIVE | path only | 68.8 % | 100 % | 52.3 % | 35.9 % |
| ADAPTIVE | **adapted** | 68.8 % | 100 % | **100.0 %** | **68.8 %** |

128 scenes, 6 families, 4 seeds/rung, 608 rows. Per-sample noise seeding, per-channel
dilation, and the EXP-004b-calibrated 0.25 s lead. These numbers supersede an earlier run
(80.7 % / 55.5 %) taken before the dilation defect in §10 was found.

**The failure mode has moved.** ADAPTIVE is now collision-free on *every* feasible plan, so
end-to-end success is bounded purely by plan feasibility (68.8 %). The system no longer
proposes motions that hit things; it declines scenes it cannot certify. That is the correct
direction for the error to lie, and it means the remaining headroom is in the conservatism of
the calibrated envelopes rather than in execution.

Each baseline fails in its own characteristic way and they fail to opposite sides: PELVIS
plans confidently and walks the body into obstacles (mean penetration 7.9 cm, mean minimum
clearance **−4.3 cm**); STANDING never collides and declares two scenes in three
infeasible. ADAPTIVE is 2.6x PELVIS and 1.8x STANDING end to end. The adaptation itself is
what does it — on the *same* plans, path-only is 51.1 % collision-free and adapted is
80.7 %.

Per family, the §4 calibration **predicts** the outcome, which is the strongest internal
evidence that the capability map is real:

| family | PELVIS e2e | STANDING e2e | ADAPTIVE e2e | matches |
|---|---|---|---|---|
| `overhead_beam` | 12.5 % | 0 % | **83.3 %** | strong duck channel |
| `partial_beam` | 25.0 % | 100 % | **100 %** | duckable *or* avoidable |
| `beam_and_gap` | 0 % | 0 % | **100 %** | duck + narrow in sequence |
| `pillar` (control) | 87.5 % | 100 % | **100 %** | needs no body adaptation |
| `narrow_gap` | 12.5 % | 0 % | 33.3 % | saturated lateral channel |
| `low_obstacle` | 0 % (100 % plan feasible, 0 % collision-free) | 0 % | 16.7 % | weak step-over channel |

`beam_and_gap` went 43.8 % → 100 % on the dilation fix alone: it is the one family needing
two different adaptations, and it was the family the defect destroyed.

**Reproducibility.** Fixed: clips are now seeded per sample from a hash of the scene id, so
the table does not depend on how the suite is chunked into batches (§11).

### Anticipation is not optional

The first ADAPTIVE run still collided on 4 of 6 beam heights (up to 11.8 cm), and the cause
is worth stating because it is a claim about the problem, not a bug report. A\* labels the
cells *physically under* the obstacle, so a naive rendering asks the prior to crouch during
the ~0.3 s it spends beneath the beam. The body cannot change envelope that fast.

Dilating each adaptation ±0.8 s — ducking *before* the beam, as a person does — takes the
same plans from 18.9 cm penetration to **0.0 cm on every rung**. The planner was correct in
space and wrong in time. This is direct evidence for surviving claim (2): whole-body
traversal needs *deliberative* commitment metres in advance, which a reactive controller
with a ±0.8 m perception window cannot express.

---

## 6. Physics executability — what is honest to claim

A kinematically plausible ARDY motion is not a trackable one. The ladder, cheapest first:

1. **foot-ground penetration and foot slip** — available now, no simulator. Raw ARDY G1
   motion penetrates the floor by **1.3–3.9 cm [measured]**, consistent with the upstream
   note that G1 post-processing is disabled.
2. **MuJoCo inverse dynamics** — joint-limit and torque violations, still no RL policy.
3. **SONIC / ProtoMotions rollout** — the only signal that supports "physically executable
   on G1", and **not yet demonstrated on this machine**.

Two findings make (3) less attractive than it first looked. SONIC is **blind to the scene**
— all 89 observation functions are motion-reference or proprioception, and the only height
map casts rays straight down, "so an overhead beam is invisible by construction". And
SceneBot (2606.27581) measures SONIC at **15 %** on terrain scene-interaction and **5 %**
on object scene-interaction. Reporting (1) as if it were (3) would be the easiest way to
write a paper that does not replicate, so the metric tables keep
**kinematic success / tracking success / hardware success** in separate columns
permanently.

**Correction (2026-08-28).** An earlier draft said SONIC had never run on this machine.
That was true of the `groot-wbc-sonic-sim-trackb` checkout the recon pass examined, and
false of the box: `/home/linjiw/isaaclab-install/env_isaaclab` drives
`/home/linjiw/lucid/GR00T-WholeBodyControl` and was observed running curriculum-robustness
evaluations on the GPU. A working IsaacLab+SONIC path therefore exists here; wiring an
ARDY reference into it is an integration task, not a bring-up. It should still be developed
**decoupled from the planner work**, so tracker debugging cannot stall EXP-004/005.

---

## 7. Experiment ladder

| id | question | status |
|---|---|---|
| EXP-000 | geometry audit: head gap, capsule coverage, qpos residual | **done** |
| EXP-001/b/c | what body envelopes can the frozen prior reach, per channel? | **done** |
| EXP-002 | PELVIS vs STANDING vs ADAPTIVE, end to end | **done** |
| EXP-003 | how many topologically distinct strategies does the prior realise per aperture? | **done** |
| EXP-003b | how few numbers does a strategy take, and can the prior recover one from a distal goal? | **done** |
| EXP-004 | counterfactual locality and temporal anticipation | running |
| EXP-001d | the envelope as a calibrated function with a real coverage bound | **done** |
| EXP-005a | is a 39-number constraint program expressive enough to be worth learning? | **done** |
| EXP-005b | does the mean of two valid strategies collide? (decides the model class) | **done** |
| EXP-005c | is the refusal boundary conservatism or physics? | **done** — 11.7 % recoverable |
| EXP-005d | **ORACLE@K — the kill gate** | **done — FIRED.** See §18 |
| EXP-005e | is the enumerator incomplete on harder scenes? | **done** — recall 0.325, and 93.6 % of misses are body-diversity at equal cost |
| EXP-005f | seed-noise floor; are the variants modes or a continuum? | **done** — 3.5 modes/scene; tuck and foot-order are below noise |
| EXP-005g | **BODY-ENUMERATE@K** — can a classical same-route enumerator cover them? | building |
| EXP-005 | learned **`p_φ(C \| S, s, g)`** — a *distribution* over constraint programs — vs the ADAPTIVE oracle | designing |
| EXP-006 | prior independence: the same constraint programs through Kimodo, as an external-validity baseline | planned |
| EXP-007 | physics: an ARDY reference through the working IsaacLab+SONIC path, decoupled from the planner work | planned |

---

## 8. Strategy multimodality — EXP-003, the result aimed at the surviving claim

For scenes that genuinely admit more than one route, strategies are **enumerated by
constrained re-planning** rather than hoped for from sampling: forbid the lateral band
containing the bypass and the planner must duck under; forbid every non-standing mode and
it must go around. Each is then generated with 4 seeds and scored exactly as in EXP-002.

| family | strategy | scenes | per-seed success | strategies realised per scene |
|---|---|---|---|---|
| `partial_beam` | `under` | 12 | **100 %** | **2.00** (2 in 12 of 12 scenes) |
| `partial_beam` | `around` | 12 | **100 %** | |
| `pillar` | `right` | 9 | 97 % | **1.78** (2 in 7 of 9 scenes) |
| `pillar` | `left` | 7 | 79 % | |

The two `partial_beam` strategies are not stylistic variants of one behaviour — they are
physically distinct traversals of the same aperture:

| | min pelvis height | max lateral excursion | path length |
|---|---|---|---|
| `under` | **0.415 m** | 0.033 m | 8.00 m |
| `around` | **0.756 m** | **0.578 m** | 8.46 m |

One frozen prior, one scene, two collision-free goal-reaching solutions that differ by
34 cm of crouch and 55 cm of lateral displacement, each realised on every seed.

**State this carefully.** The result is *not* that ARDY's sampling discovers the two
strategies — the strategies were supplied by constrained re-planning, and EXP-003b measures
what sampling alone does. It is also *not* that a competing policy could not be multimodal.
The claim is that here the alternatives are explicit, enumerable, and selectable, and that
the frozen prior faithfully instantiates whichever one it is handed. That is what licenses
putting the strategy variable **above** the prior rather than hoping to find it inside.

Caveat worth keeping: `pillar`'s left/right asymmetry (79 % vs 97 %) is real and
unexplained — the two routes should be near-mirror-images, so something in either the
scene construction or the prior's turning behaviour is not symmetric. That is EXP-004's
first job.

---

## 9. How few numbers is a strategy? — EXP-003b

The generator's output space had to be sized before it could be designed, so each enumerated
strategy was rendered at decreasing waypoint counts and regenerated. `dense` constrains every
frame (~215 at 25 fps); `wpN` constrains N evenly spaced frames; `goal_only` constrains the
last 0.4 s and nothing else. 704 clips, 177 s.

| program | numbers | duck strategy: collision-free | detour strategy: homotopy kept |
|---|---|---|---|
| dense | 860 | 100 % | 100 % |
| wp32 | 128 | 100 % | 100 % |
| wp16 | 64 | 97 % | 100 % |
| wp10 | 40 | 97 % | 100 % |
| **wp8** | **32** | **94 %** | **100 %** |
| wp6 | 24 | 84 % | 100 % |
| wp5 | 20 | **16 %** | 100 % |
| wp4 | 16 | 0 % | 100 % |
| wp2 | 8 | 0 % | 25 % |
| goal_only | 40 | 0 % | 50 % |

**The route is cheap; the body profile is expensive — the opposite of the naive assumption.**
Which side of the obstacle the robot passes survives down to 4 waypoints. The pelvis-height
profile that makes the duck actually work collapses between wp6 and wp5: 84 % → 16 %, with the
achieved minimum pelvis height rising from 0.354 m to 0.493 m as the crouch is averaged away.
At 25 fps over ~8.6 s clips, wp8 is one control point per ~1.1 s and wp6 per ~1.4 s, so the
adaptation needs a control point roughly every second and tolerates nothing sparser.

**Design consequence.** `p(C|S,s,g)` should emit on the order of **8–16 waypoints, 32–64
numbers** — 13–27× smaller than the dense program at ≥94 % of its collision-free rate. That is
a small enough object for a flow model to represent multimodally on a few thousand examples,
which is the whole reason for putting the generator in constraint space rather than motion
space.

**Commitment, measured.** Given only a distal goal (`goal_only`), the prior reaches the goal
every time and preserves the intended strategy only **50 %** of the time, collision-free **0 %**
— it walks the straight line and takes the beam with it. ARDY cannot recover a homotopy class
from a distal goal, which is the empirical case for making the strategy variable explicit
*above* the prior rather than hoping to elicit it from sampling.

One methodological trap found here and worth keeping: with a sparse program whose first
constrained frame is late in the clip, ARDY's `first_heading_angle` must NOT be read off
`heading[0]` — that is the heading at the first *constrained* frame, and importing it rotates
the whole clip to match a pose the robot should only reach at the end. `ConstraintSpec` now
carries `first_heading` separately.

---

## 10. Locality and anticipation — EXP-004 / EXP-004b

### Locality: the adaptation is surgical, and the metric can tell

Matched ladder rungs differ in one clearance parameter; a good planner should change its
behaviour only where the geometry demanded it. Differencing raw collision-primitive positions
does **not** measure that — it gives SNR < 1, because two clips of the same walk drift out of
gait phase and a swinging wrist moves further than a 10 cm beam change moves the body.

The metric is therefore the **clearance envelope** — top height, lateral half-width, pelvis
height — which is phase-robust and is exactly what an obstacle constrains. The null is the
same scene with the obstacle nudged 12 cm sideways: a scene edit that changes nothing about
what the body must do.

| family | variant | locality ratio | share of change inside the window |
|---|---|---|---|
| `overhead_beam` | adaptive | **4.32** | 0.41 |
| `overhead_beam` | global-duck strawman | 0.72 | 0.13 |
| `partial_beam` | adaptive | **3.86** | 0.36 |
| `partial_beam` | global-duck strawman | 0.61 | 0.11 |

The interaction window is 0.17 of the clip, so a uniformly smeared difference scores 1.0. The
adaptive renderer concentrates change ~4× inside the window; the strawman that ducks for the
whole clip scores *below* 1.0, i.e. anti-local. The metric separates them, which is the test
it had to pass.

### Anticipation: measured causally, not by onset

Onset detection is the wrong instrument here. `plan_to_spec` dilates the schedule by a
constant, ARDY hard-infills the root slice, and achieved onset therefore tracks commanded
onset to within a frame — it reads back our own constant. Worse, the global-duck strawman
scores a *larger* lead (3.75 s vs 1.31 s) purely by ducking throughout, so a bigger onset is
not even better.

So sweep the lead and watch collisions instead. Everything except `lead_s` is held fixed,
including the noise draw:

| lead_s | `overhead_beam` collision-free | `beam_and_gap` collision-free |
|---|---|---|
| 0.0 | 71 % | 64 % |
| 0.1 | 87 % | 75 % |
| **0.2** | **100 %** | 94 % |
| **0.3** | 100 % | **100 %** |
| 0.4 – 1.4 | 100 % | 100 % |
| 2.0 | 100 % | 75 % |

**0.2–0.3 s of lead is necessary and sufficient**, on top of the 0.6 s smoothing the renderer
already applies — so the ±0.8 s constant EXP-002 adopted sits comfortably on the plateau
rather than being load-bearing. The plateau is broad but not unbounded: at 2.0 s of lead
`beam_and_gap` degrades to 75 %, because its two adaptations are only ~3.1 s apart and begin
to blur into each other. Anticipation has an optimum, not a monotone benefit.

### A bug this experiment caught

The first run of the sweep showed `beam_and_gap` collapsing from 94 % at 0.2 s to 39 % at
0.3 s and beyond. The cause was in the renderer, not the prior: dilation operated on whole
*modes*, ranked by "most demanding", so any duck outranked a tuck and spreading a duck near
the gap **deleted the squeeze** — tuck frames went 16 → 6 → 0 as lead went 0.0 → 0.2 → 0.3 s.
An adaptation was being silently replaced by a different one. Dilation is now per-channel
(pelvis takes the window minimum, tuck/lift/sidle the maximum), tuck frames now grow 10 → 44
with lead, and the curve above is monotone. One of the EXP-005 design agents independently
found the same defect in 24 of 88 plans.

---

## 11. Infrastructure that had to be fixed first

**Per-sample noise seeding.** ARDY seeds once per generation *call*, so a sample's noise
depended on its position in the batch. Three independent analyses put the resulting floor at
the scale of the effects being measured — a 0.085 m null against a 0.137 m signal, and the
same nominal condition yielding worst-of-3 half-widths of 0.281 m in EXP-001b and 0.380 m in
EXP-001. Since ARDY's sampler is deterministic DDIM (η = 0), the only stochastic input is one
`torch.randn` for the initial latent, so `generate(seeds=[...])` now intercepts exactly that
call and fills it per sample. Verified identical across batch slots, different across seeds,
and reproducible across batch sizes.

**Cache-first text encoding.** The Llama-3-8B encoder is ~14 GB as a CPU service and ~16 GB
of VRAM locally, and ARDY's default `auto` mode falls back from one to the other *silently*.
When unrelated IsaacLab jobs took the GPU, the kernel OOM-killed the CPU service and two
experiments then died several minutes in on a CUDA OOM. The prompt cache is now
authoritative: no encoder is loaded unless a genuinely new prompt appears, and `encode`
raises with instructions rather than grabbing the GPU. Startup went 30 s → 3 s at 926 MB.

**Planner speed**, needed for the EXP-005 oracle. A\* expanded 8·M successors per pop (every
neighbour × every mode); splitting into *move* and *switch-in-place* gives 8 + (M−1) with no
loss of expressiveness. A necessary-condition flood fill now short-circuits infeasible scenes,
which is where essentially all the time went. A single plan is 0.01 s, and strategy
enumeration results are unchanged.

---

## 12. Is 39 numbers enough? — EXP-005a

`p_phi(C | S, s, g)` can only be as good as C. If compressing an oracle plan into a small
program loses enough fidelity that the decoded version no longer traverses the scene, a
perfect generator over that space is still useless. So this ran before any training code was
written — the cheapest experiment that could falsify the whole EXP-005 design.

Every evaluation scene, every enumerated strategy, rendered twice from the *same* plan and
the *same* noise seed: once through `plan_to_spec` and once through
`encode → 39 numbers → decode`. Both paths share `_limb_targets`, `_dilate_channel` and
`LEAD_S`, so any difference is compression, not two code paths disagreeing.

| family | n | oracle succeeds | program succeeds | mean route error |
|---|---|---|---|---|
| `overhead_beam` | 15 | 100 % | 100 % | 0.007 m |
| `beam_and_gap` | 12 | 100 % | 100 % | 0.012 m |
| `narrow_gap` | 6 | 100 % | 100 % | 0.079 m |
| `partial_beam` | 28 | 96 % | 96 % | 0.092 m |
| `pillar` | 34 | 91 % | 91 % | 0.138 m |
| `low_obstacle` | 3 | 67 % | 100 % | 0.007 m |
| **overall** | **98** | **94.9 %** | **95.9 %** | |

**Agreement 96.9 %, and of the three disagreements two favour the program.** The single case
where compression hurts is `pillar_off0.60 duck_light:R`, the largest route error in the set
(0.289 m), which produces 0.5 cm of penetration.

The lesson is about the metric, not the model. The compressed request differs from the oracle
by up to 0.29 m in L-infinity, which sounds disqualifying and is not: 13 cm is irrelevant in a
1.4 m corridor and fatal in a 0.5 m gap, and only generation plus collision checking can tell
which. Route error is concentrated on `pillar`, the family with the largest lateral detours,
where 16 uniform chord knots over an 8 m path under-resolve the turn. That is a bounded and
understood limitation with an obvious fix (more knots, or non-uniform placement near
obstacles) if it ever binds.

**Tension resolved — and my diagnosis of it was wrong.** Making `dip` continuous appeared to
invalidate A*'s lateral certificate, because the tabulated half-widths were non-monotone in
dip (`duck_deep` 0.497 m against `stand` 0.380 m). I predicted that was a worst-of-3-seeds
artefact. EXP-001d re-measured at n = 20 and it is not: half-width rises monotonically with
dip, 0.378 m → 0.510 m at the 90 % bound. Ducking really does make the robot wider. The
non-monotonicity in the old table was noise; the underlying *trend* was real and had the
opposite sign to what a "spurious outlier" story implied.

The fix is therefore not to discard the coupling but to model it. §13.

---

## 13. The capability envelope as a calibrated function — EXP-001d

`outputs/body_modes.json` was the worst of three seeds per mode. That is not a bound: it is a
36.8 %-content tolerance interval at 95 % confidence, and the same nominal condition produced
worst-of-3 half-widths of 0.281 m and 0.380 m in two experiments purely from which draws were
taken. So both axes were re-swept at **20 independent samples per level** and summarised by a
**split conformal upper bound** — the ⌈(n+1)(1−α)⌉-th order statistic, which is valid for a
fresh exchangeable sample. n = 20 supports a genuine 90 % bound; n = 3 supports none.

| dip (m) | top, mean | top, 90 % bound | half-width, mean | half-width, 90 % bound | sd |
|---|---|---|---|---|---|
| 0.00 | 1.314 | 1.336 | 0.310 | 0.378 | 0.044 |
| 0.15 | 1.147 | 1.220 | 0.323 | 0.402 | 0.052 |
| 0.30 | 1.021 | 1.072 | 0.327 | 0.411 | 0.054 |
| 0.45 | 0.885 | 0.931 | 0.379 | 0.453 | 0.063 |
| 0.50 | 0.823 | 0.879 | 0.403 | 0.510 | 0.074 |

`top(dip)` is monotone decreasing and clean. `half_width(dip)` is monotone **increasing** —
the arms come out for balance — by ~9 cm in the mean and ~13 cm at the bound. A deep duck
therefore needs a **1.02 m corridor**, not the 0.76 m a standing envelope implies, and the
planner must carry that coupling rather than treating the axes as independent.

Tuck is the opposite and is the axis worth spending on when a gap is tight: the bound falls
0.378 → 0.272 m and the standard deviation falls 0.044 → 0.012, so tucking makes the robot
both narrower *and* more repeatable.

`scene2motion/envelope.py` exposes this as `top(dip)` and `half_width(dip, tuck)` with
monotonicity enforced. **The claim it licenses, exactly:** for a request drawn from the
calibration distribution, the body's envelope exceeds the bound at most 10 % of the time. It
is a statement about the *envelope*, not about collision-freeness — the per-instance guarantee
remains the MuJoCo check, which is why those stay in separate columns.

Its weakest link is stated in the module: the two axes were calibrated *separately*, so
combining them assumes the tuck credit measured at nominal pelvis height still applies while
crouching. EXP-001's duck+tuck cells hint it does not fully. A joint 2-D sweep is the honest
way to settle it and has not been run.

---

## 14. What the model may be — EXP-005b

Flow matching in constraint space is the obvious choice for `p_phi(C | S, s, g)`, and a review
pass argued against it with a specific empirical claim: the arithmetic mean of two valid
strategies is *in collision*, so any objective whose conditional optimum is a mean emits an
infeasible program. That decides the architecture, so it was measured here rather than taken
on faith. Every corpus scene with two validated programs, interpolated between them:

| mix | 0.0 (strategy A) | 0.25 | **0.50 (mean)** | 0.75 | 1.0 (strategy B) |
|---|---|---|---|---|---|
| goal-reaching and collision-free | 94 % | 56 % | **33 %** | 44 % | 83 % |

Endpoints 89 %, midpoint **33 %**. The claim holds, and the failure is not a narrow band at
exactly 0.5 — it covers most of the interior. **The straight line between two ways through a
scene is mostly not a way through the scene**, which is what one would expect when the two
differ by a homotopy class.

State the consequence carefully, because the strong version is wrong. Flow matching and
diffusion are *not* mean-seeking at convergence; they model the full distribution and sample
from modes. What this measures is that the interpolation path is largely infeasible, so a
generative model over C must resolve its modes **sharply** rather than smear across them — a
set predictor with Hungarian matching gets that structurally, while a flow has to earn it from
data, and at 10³ pairs a smoothed velocity field will put terminal mass in exactly the gap
above. That is an argument about sample efficiency and inductive bias, not an impossibility
proof, and the cheap way to settle it is to train both heads on the same encoder and compare.

*Caveat on the endpoints.* They are 94 % and 83 % rather than 100 % because every mix shares
one nominal limb reference (from the scene's own adaptive plan) so the comparison is clean,
which costs each endpoint the reference it was validated against. The contrast is measured
within the same setup and is unaffected.

*Caveat on the parameterisation.* These numbers were taken on the 39-dimensional program,
before the design-spec corrections (§15) took it to 43 dimensions with arc-length decoding.
The conclusion is about geometry — the straight line between two homotopy classes leaves the
free space — so it should survive the reparameterisation, but the exact percentages are from
the old space and the experiment is queued for a re-run. The corresponding artefacts under
`outputs/` carry a `STALE.txt`.

---

## 15. What the EXP-005 design pass changed — and it is not a detail

Thirteen agents designed the learned proposer and attacked each other's designs. The
specification they produced re-measured this repo rather than trusting its documentation, and
its opening section reframes what the project can claim.

### The oracle no longer collides. It refuses.

Grouping `outputs/exp002/rows.jsonl`: adaptive/adapted is **88/88 plan-feasible, 1.000
collision-free, 1.000 goal-reaching**, and end-to-end success (68.8 %) *equals plan
feasibility exactly*. All 40 failures are `plan(...).feasible == False`, concentrated in
`low_obstacle h ≥ 0.08` (20), `narrow_gap w ≤ 0.50` (16), `overhead_beam h = 0.80` (4).

So "how much better can any proposer be?" is entirely "how many refusals are recoverable?",
and that was measured two independent ways. Relaxing the certified envelope by
`top −0.08, width −0.05` and re-planning the 128-scene suite takes feasibility **68.8 % →
75.0 %** — reproduced here exactly, with the gain in `overhead_beam` (20→24) and `narrow_gap`
(8→12) and *none* in `low_obstacle`. Generating deliberately off-lattice programs on the
refused scenes and MuJoCo-checking them agrees to the scene: `overhead_beam h=0.80` recovers
12/12, `narrow_gap w=0.50` recovers 10/12, and `narrow_gap w ≤ 0.42` and `low_obstacle
h ≥ 0.08` recover 0/36.

**Eight of the forty refusals are conservatism; thirty-two are physics. The ceiling for any
proposer on this suite is 75.0 %.**

### Three consequences, all binding

1. **End-to-end kinematic success cannot be the headline.** Six points of headroom, all of
   which a two-line envelope relaxation plus resampling also takes. A table leading with
   "learned 75 % vs oracle 68.8 %" would be dishonest by omission.
2. **`ORACLE-RELAXED@K` is a mandatory control**: plan against the relaxed envelope, propose,
   generate K seeds, keep any that verifies. It is about fifteen lines, and a reviewer will
   build it mentally whether or not it is in the table. EXP-005d builds it.
3. **The learned model's only structural advantages are** (i) it is a distribution with a
   calibrated per-program validity score, so K samples buy *coverage* rather than repetition;
   (ii) it can emit programs off the discrete mode lattice, which A\* structurally cannot;
   (iii) it is a differentiable, language-addressable interface. Each is measurable, and each
   has a stated result that kills it.

### The gate before any training code

If `ORACLE-ENUMERATE@8` already reaches the learned model's plausible traversal success *and*
its strategy coverage, then `p_phi` is distillation of a 0.13 s planner into a network and is
not a contribution. Two honest exits are pre-committed: re-aim at the calibration and
refusal-with-a-reason results plus language re-ranking, which need no training at all; or move
to scenes where enumeration is expensive or incomplete — four or more obstacles, dead ends,
non-monotone routes — and show `p_phi` covers strategies the enumerator's exclusion heuristic
misses. **That decision costs 20 minutes of GPU and is made before three hours of dataset.**

### Step 0 and step 1, done

Five measured defects in `program.py`, all corrected (`LAT_SCALE` 0.60 → 1.50 against a
measured 1.458 m maximum; the `sidle` column dropped as identically zero; chord-abscissa
sampling replaced by arc-length resampling, which was costing up to 0.30 m of along-track lag
no number of knots could remove; `N_LAT` 16 → 24; slots ordered by position rather than
intensity). `DIM_C` 39 → 43. Re-running EXP-005a:

| | program succeeds | agreement | route error, max | `pillar` mean |
|---|---|---|---|---|
| N_LAT = 16 | 95.9 % | 96.9 % | 0.289 m | 0.138 m |
| **N_LAT = 24** | **96.9 %** | **98.0 %** | **0.110 m** | **0.052 m** |

The compressed program now *beats* the oracle plan it compresses, and geometric error is down
62 %.

---

## 16. The certificate was optimistic, and fixing it costs 3 points

Rebuilding the mode table from the EXP-001d envelope rather than worst-of-3-seeds exposed
that the old table was not conservative at all — it was wrong in **both** directions, and the
dangerous direction landed on exactly the modes that do the work:

| mode | half-width, old → calibrated | top, old → calibrated |
|---|---|---|
| `duck_light` | 0.360 → 0.402 | 1.216 → 1.220 |
| `duck` | 0.375 → 0.420 | 1.127 → 1.120 |
| `duck_deep` | 0.497 → 0.420 | 1.017 → 1.034 |
| **`duck_max`** | **0.344 → 0.510** | **0.822 → 0.879** |

The planner was certifying `duck_max` against a 0.344 m half-width when the 90 % bound is
0.510 m — **16.6 cm optimistic** — and a 0.822 m top against 0.879 m. It has not produced
collisions, since EXP-002 is 100 % collision-free on feasible plans, but that is the *mean*
behaviour and a certificate is meant to bound the tail. A guarantee that holds on average is
not a guarantee.

Correcting it costs feasibility, as it must: **68.8 % → 65.6 %** on the 128-scene suite,
`beam_and_gap` 16/16 → 12/16. The earlier number rested in part on an optimistic bound; 65.6 %
is the one that comes with a coverage statement. A headline moving *down* under scrutiny is
the direction that should increase confidence in it.

`experiments/derive_modes.py`, which builds the old table, now refuses to run rather than
silently restoring a guarantee the system does not have.

**Process note.** That file was edited while the ORACLE@K gate was running. `MODES` loads at
import, so the in-flight run held the old table — a config a running experiment depends on
should not be edited mid-flight, and the gate was re-run rather than its result reused.

## 17. Refusal with a reason

Since every remaining failure is a refusal (§15), a refusal that names its cause is worth more
than one that does not. `Plan.refusal` reports the binding functional and the deficit:

| binding functional | refused scenes | example |
|---|---|---|
| `lateral_clearance` | 20 | `narrow_gap w=0.30`: scene offers 0.30 m, body needs 0.54 m — short 24.4 cm |
| `step_height` | 16 | `low_obstacle h=0.08`: box is 0.08 m, best certified step is 0.03 m — short 5.2 cm |
| `overhead_clearance` | 8 | `overhead_beam h=0.80`: clearance 0.80 m, body needs 0.88 m — short 7.9 cm |

The deficit is what separates "2.9 cm short of a certified duck" (recalibrate, or accept the
risk) from "0.30 m gap against a 0.54 m body" (a real limit of the prior), and it makes
EXP-005c's 32-of-40 claim checkable per scene rather than only in aggregate.

Two bugs in the first version, both of the *technically true and useless* kind, and both found
by reading its own output rather than by a test: it called a 2.6 m wall panel a step-over
problem short by 257 cm, and it measured lateral clearance per-obstacle so it could not see
the channel *between* two wall panels — which is exactly the geometry `narrow_gap` is made of.

---

## 18. The gate fired — EXP-005d

128 scenes, calibrated certificate, budget spread across each proposer's enumerated
strategies, every clip generated and verified against the robot's own collision geometry, and
strategy coverage measured from the MOTION rather than the commanded label.

| | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| **ORACLE-ENUMERATE** traversal success | 65.6 % | 65.6 % | 65.6 % | 65.6 % |
| **ORACLE-ENUMERATE** distinct verified strategies | 1.00 | **1.93** | 1.93 | 1.93 |
| **ORACLE-RELAXED** traversal success | 65.6 % | 71.1 % | 73.4 % | **75.0 %** |
| **ORACLE-RELAXED** distinct verified strategies | 0.94 | 1.69 | 1.83 | 1.89 |

The pre-committed kill criterion was: *if `ORACLE-ENUMERATE@8` already achieves the learned
model's plausible traversal success and its strategy coverage, then `p_phi` is distillation of
a 0.13 s planner into a network and is not a contribution.*

Both halves are met, and by a wide margin.

* **Traversal success is capped at 75.0 %** by EXP-005c — 32 of 40 refusals are physics, not
  conservatism. `ORACLE-RELAXED@8` reaches exactly 75.0 %. There is no headroom above it for
  any proposer, learned or otherwise.
* **Strategy coverage saturates at K=2.** `ORACLE-ENUMERATE` reaches **1.93 distinct verified
  strategies** on the 30 ambiguous scenes at K=2 and does not improve with four times the
  budget. Since an ambiguous scene has at least two strategies by construction, 1.93 is
  ~96 % of what is available.

Note also that traversal success for the certified enumerator is **flat in K**. Extra samples
buy nothing: a certified plan either works at K=1 or the scene is refused, and no budget
rescues a refusal. Sampling only helps the *relaxed* arm, and only because relaxation trades
the guarantee for reach.

**So a learned `p_phi(C | S, s, g)` has nothing left to win on this suite, on either axis.**
That is the result the gate existed to produce, and producing it cost 20 minutes of GPU
instead of three hours of dataset plus a training run.

### The two pre-committed exits

**(a) Re-aim at what is already in hand and needs no training**: the conformal capability
calibration and the finding that the previous certificate was optimistic by 16.6 cm on the
mode that does the most work (§16); refusal-with-a-reason (§17); and language as a re-ranker
over enumerated strategies, which is cheap because the motion set does not depend on the
instruction.

**(b) Move to scenes where enumeration is expensive or incomplete** — four or more obstacles,
dead ends, non-monotone routes — and show `p_phi` covers strategies the enumerator's exclusion
heuristic misses.

Exit (b) rests on a premise that has not been tested: that the enumerator *is* incomplete on
harder scenes. On this suite it is near-complete, which is exactly why the gate fired. That
premise is cheap to test on CPU and should be tested before it is adopted — adopting it
untested would repeat the mistake the gate was built to prevent.

---

## 19. EXP-005e: the enumerator is blind to BODY diversity, not to routes

Exit (b) said "move to scenes where enumeration is incomplete". That was a premise, so it was
tested: 150 harder scenes (1–5 obstacles, tighter corridors), the shipped heuristic against a
deliberately expensive reference search (40 restarts with random exclusions, random mode
subsets), both scored by the same signature.

| obstacles | scenes | heuristic finds | reference finds | recall |
|---|---|---|---|---|
| 1 | 23 | 1.43 | 4.48 | 0.314 |
| 2 | 15 | 1.73 | 4.33 | 0.371 |
| 3 | 10 | 1.50 | 5.30 | 0.319 |
| 4 | 12 | 1.42 | 5.00 | 0.339 |
| 5 | 4 | 1.50 | 7.25 | 0.185 |
| **all** | **150** | | | **0.325** |

**The misses are not contrived detours.** A random-restart reference can manufacture
technically-distinct-but-absurd routes, which would inflate incompleteness in exactly the
direction that keeps a learned model alive, so the cost of every miss was measured against the
best route in its scene: median ratio **1.005**, **73.5 % within 1.02×**, maximum 1.155 — and
the heuristic's *own* kept plans run p50 1.009, p95 1.136. The strategies it misses are, if
anything, cheaper than the ones it keeps. (The 1.35× "useful" filter turned out too loose to
discriminate at all; the raw distribution is the answer.)

### What is actually missing

Classifying 172 misses by what makes them new:

| | share |
|---|---|
| **new BODY, same route** | **93.6 %** |
| new combination of known parts | 3.5 % |
| new route AND new body | 2.3 % |
| new route only | 0.6 % |

The enumerator's exclusion rule operates on *corridors* — spatial regions — so it is built to
find different routes and is near-complete at that (2.9 % of misses involve a new route). It
is structurally blind to **a different body adaptation along the same route**, because A\* over
`(x, y, mode)` returns the cost-minimal mode assignment per route and never enumerates the
other assignments that also work.

That is precisely the axis this whole project is about, and precisely the axis a constraint
program covers by construction: `dip`, `tuck` and `lift` are continuous fields in C, while A\*
picks one point from a discrete lattice.

### This qualifies my own gate result, and I should say so plainly

EXP-005d concluded that coverage saturates at 1.93 and therefore a learned proposer has
nothing to win. That conclusion used `metrics.realised_signature`, which encodes body
diversity as **one bit** — ducked or not. On a two-route scene it can express at most four
values. It cannot see the axis EXP-005e just showed the enumerator is blind to.

So the gate splits:

* **Traversal-success half: fired, robustly.** The 75.0 % ceiling is real (32 of 40 refusals
  are physics), and `ORACLE-RELAXED@8` reaches exactly 75.0 %. No proposer beats it.
* **Coverage half: not settled.** It was measured with a signature too coarse to resolve
  body-adaptation diversity, which is where the incompleteness lives.

The fix is a finer *realised* morphology signature — achieved dip / tuck / lift quantised into
bands and read off the motion, not the label — and a re-run of the coverage arm. Until that is
done, the honest statement is that end-to-end traversal success is a dead end for a learned
model, and the strategy-coverage claim is **open**, with EXP-005e giving a concrete reason to
expect the enumerator to fall short on it.

---

## 20. EXP-005f: the seed-noise floor, and two axes that turn out to be unusable

Before any morphology coverage number can mean anything, three things have to be separated:
different body strategies, different amplitudes of one strategy, and ARDY's own seed-to-seed
scatter. The third sets the unit for the second, so it was measured first — 6 scenes x 36 body
programs (dip x tuck x lift over the reachable box) x 6 seeds, every quantity a paired delta
against a neutral-body control on the **same route and the same seed**.

### ARDY's sampling scatter is large

| channel | seed noise (q99 of \|delta − program mean\|) |
|---|---|
| `dh_top` | **0.122 m** |
| `dw_left` / `dw_right` | 0.175 / 0.158 m |
| `dz_foot_left` / `dz_foot_right` | 0.341 / 0.364 m |
| `t_lead` / `t_duration` | 2.29 s / 3.26 s |
| `dpsi` | 0.462 rad (26°) |

A body program does **not** reliably produce a specific envelope. 12 cm of scatter in achieved
top height is ~45 % of the strongest adaptation the prior offers (dip 0.30 gives 0.274 m), and
it is exactly why the capability calibration in §13 needed a conformal bound rather than a mean.

### Two axes are below their own noise

**Tuck never fires, in any scene, at any amplitude.** EXP-001b measured the lateral channel at
~5 cm; the seed noise on width is 16–18 cm. So the lateral adaptation is not merely weak — it
is **smaller than ARDY's own sample-to-sample width variation**, and no thresholding can
recover it. That retires the tuck axis as a controllable strategy dimension.

**Foot order is not controllable either.** The first version of the active-set signature
assigned left-first / right-first from a bare sign comparison, which gives every clip an order
and turns gait phase into a strategy label — it was inflating the distinct-strategy count from
3.5 to 4.2 per scene. Requiring the two feet to differ by more than the foot-channel noise
before assigning an order makes `order` **0 everywhere**: the prior does not controllably
choose which foot leads.

### What is actually there

| scene | valid programs | discrete strategies | eps-net @3σ | stability |
|---|---|---|---|---|
| `overhead_beam` 1.10 | 13 | 3 | 5 | 0.83 |
| `overhead_beam` 0.90 | 2 | 1 | 2 | 1.00 |
| `partial_beam` 1.05 | 22 | 6 | 14 | 0.89 |
| `narrow_gap` 0.70 | 17 | 3 | 6 | 0.90 |
| `beam_and_gap` 1.10 | 9 | 2 | 6 | 0.78 |
| `pillar` 0.30 | 21 | 6 | 9 | 0.90 |
| **mean** | **14.0** | **3.5** | **7.0** | **0.88** |

**A 36-program ladder collapses to 14 valid, then 7 distinguishable at 3σ, of which 3.5 are
distinct discrete strategies.** The median nearest-neighbour distance between adjacent ladder
points is 1.4–2.4σ, so the sweep samples finer than the prior can reliably deliver — precisely
the artefact the guidance warned would make a densely-sampled continuum look like a set of
strategies.

The surviving alphabet is combinations of **(duck, lift, yaw)** only, and mean stability is
0.88 with about three quarters of programs above the 0.8 addressability threshold.

### What this predicts about the gate, stated before it runs

This is a modest amount of morphological diversity: ~3.5 discrete modes per scene drawn from an
alphabet of at most eight. Covering that does not obviously require a learned model — a
classical enumerator that simply tries the (duck, lift, yaw) combinations at a fixed route
would plausibly get most of it cheaply. **I expect BODY-ENUMERATE@K to do well**, which would
push the project toward the guidance's Outcome 3: the contribution is the calibration, the
refusal decomposition, and the morphology benchmark, not the generator. EXP-005g decides it.

---

### A note on V2, if it happens

The original plan assumed scene cross-attention adapters. **ARDY has no cross-attention
anywhere** — an exhaustive `named_modules()` scan finds 32 `MultiheadAttention` modules,
all `self_attn`, all `_qkv_same_embed_dim=True`. Conditioning enters by per-token channel
concatenation into role-specific `Linear` layers summed under disjoint token masks
(`auto_latent_twostage_denoiser.py:210-232`). The native injection point is therefore a new
per-token `scene_proj` Linear summed into `root_stage_input` and `body_stage_input`,
zero-initialised so step 0 reproduces the frozen checkpoint exactly — not a cross-attention
hook and not a prefix token (the learned prefix is exactly 3 tokens and resizing it breaks
`strict=True` loading). The forward/backward path is differentiable as-is.

The root/body two-tower split the proposal hoped for **is** real: `root_model`
(73,520,148 params) then `body_model` (73,630,848), reachable at
`model.denoiser.model.root_model`.

---

## 21. EXP-005g: the gate fired against my own prediction

**Prediction recorded before the run** (section 20): *"~3.5 modes from an 8-symbol alphabet does
not obviously need a learned model, so I expect BODY-ENUMERATE@K to do well, which would push
this toward Outcome 3."*

**That prediction is wrong.** 30 scenes, 22 with a feasible route, `n_seeds=4`, `K=8`,
`eps` calibrated on the NULL-SEED arm at 3.99 calibrated units, 10 838 unique ARDY clips,
61 min wall-clock. Discrete and continuous MorphRecall@8, macro-averaged per scene, with a
scene-level paired bootstrap CI on the difference against NULL-SEED:

| arm | ARDY calls | disc@8 | vs NULL | cont@8 | vs NULL | addressable of 8 |
|---|---|---|---|---|---|---|
| A-KBEST | 36 | 0.583 | [+0.060, +0.365] | 0.349 | [+0.016, +0.247] | 4.5 |
| B-NOGOOD | 36 | **0.592** | [+0.123, +0.341] | 0.381 | [+0.063, +0.271] | 5.0 |
| C-WSWEEP | 36 | 0.505 | [−0.015, +0.286] | 0.273 | [−0.056, +0.175] | 4.2 |
| COMPOSITE | 37 | 0.541 | [+0.013, +0.328] | 0.304 | [−0.029, +0.207] | 2.9 |
| D-REFINE | 37 | 0.370 | [+0.000, +0.027] | 0.304 | [+0.020, +0.157] | 2.6 |
| COMPOSITE(x4) | 148 | **0.665** | [+0.136, +0.447] | 0.455 | [+0.120, +0.355] | 4.6 |
| D-REFINE(x4) | 148 | 0.595 | [+0.148, +0.318] | **0.551** | [+0.275, +0.411] | 5.0 |
| REF-RANDOM | 101 | 0.606 | [+0.168, +0.318] | 0.507 | [+0.212, +0.373] | 13.2 |
| NULL-SEED *(control)* | 69 | 0.361 | — | 0.207 | — | 7.0 |

**No arm reaches the pre-committed 0.90 on either axis, at any budget.** The best equal-call
arm covers 59 % of the discrete modes and 38 % of the eps-net; four times the calls buys
67 % / 55 %. By the guidance's pre-committed table this is the third row — *"it misses stable,
useful body components even at equal K"* — which is the row that licenses a learned proposer.

I am deliberately not claiming that yet, for three reasons visible in this same table.

**1. A third of the score is free.** NULL-SEED — the *same* program submitted eight times on
disjoint seeds — scores 0.361 discrete and 0.207 continuous. It proposes nothing. That is the
floor any arm must be read against, and it is why the CI column is against NULL-SEED rather
than against zero. C-WSWEEP's discrete advantage over resampling is not significant.

**2. The failure is family-localised, and its shape is diagnostic.**

| arm | narrow_gap (4) | overhead_beam (10) | partial_beam (8) |
|---|---|---|---|
| A-KBEST | 0.350 | 0.483 | 0.825 |
| B-NOGOOD | 0.400 | 0.563 | 0.725 |
| COMPOSITE(x4) | 0.550 | 0.603 | 0.800 |
| REF-RANDOM | **0.800** | 0.527 | 0.608 |
| NULL-SEED | 0.350 | 0.440 | 0.267 |

On `partial_beam` the classical enumerators work well (0.73–0.83 against a 0.27 floor). On
`narrow_gap` every classical arm equals the null floor exactly — they contribute *nothing* over
resampling — while random restarts reach 0.800. A method that cannot beat resampling on a
family where random search reaches 80 % has a **candidate-support** problem on that family, not
a selection problem. That is a different diagnosis from "learn a generator", and it is
family-specific.

One caveat on that row, in REF-RANDOM's favour and against my reading of it: every arm's
recall@8 is computed over its first eight candidates, but REF-RANDOM *proposes twenty-four* and
the run pays to evaluate all of them — 101 charged clips against the classical arms' 36. So it
is candidate-equal and not call-equal, which is precisely the accounting the guidance's
`ValidDiversityYield@B` exists to correct. Its 0.800 on `narrow_gap` is a real signal that the
support is reachable, but it is not a like-for-like win, and the yield table in EXP-005i is
where the two are put on the same currency.

**3. `addressable of 8` is the number that actually hurts.** A-KBEST, B-NOGOOD and C-WSWEEP
return eight candidates of which only 4.2–5.0 are addressable (feasible on ≥75 % of seeds *and*
Stability ≥ 0.8). They are not spending their budget on eight bodies; they are spending it on
four bodies and four coin flips. COMPOSITE and D-REFINE report feasibility 1.00 — but they
select using ARDY outcomes on the **same seeds they are scored on**, so that 1.00 is
post-selection until it is shown on held-out seeds. The guidance is right to flag this; it is a
real hole in the harness as run, not a hypothetical one.

### What the gate does *not* tell us, and the harness bug that hides it

Coverage of 0.59 is consistent with three completely different worlds — the pool lacks the
programs (**support**), the pool contains them and the heuristic picks wrong (**selection**), or
no program addresses the mode reliably at all (**ARDY addressability**). The prescribed
discriminator is POOL-ORACLE@K: the best size-K subset of the *union* of every arm's candidates,
chosen with hindsight from realised outcomes. It is not deployable; it is an upper bound that
names which model could possibly help.

It should have cost zero extra GPU time, because every candidate's per-seed validity, active
set and descriptor already existed in memory as `body_enumerate.Evaluation` objects during the
run. **The harness discarded them and wrote only per-arm summary statistics.** So POOL-ORACLE,
the commanded-vs-realised matrix, the allocation control, the failure decomposition and the
covariance sensitivity — five analyses the guidance asks for — are all blocked on data I had
and threw away.

That is a design error worth naming, because it is the same class as the metric bugs: the
experiment was written to *answer its question* rather than to *record its evidence*. The fix is
to emit a per-candidate ledger once and treat every downstream question as a re-analysis of that
file. The gate re-runs once with the ledger and with disjoint selection/evaluation seed blocks;
after that, none of these questions costs GPU time again.

---

## 22. The capability audit: what the API offers vs what the robot can be asked to do

The guidance's framing — *nominal constraint dimensions ≠ reliably addressable robot
capabilities* — turns out to be measurable channel by channel, and two of the measurements cost
no GPU time at all. They come from re-reading EXP-005f's 1 296 saved paired residuals.

### Yaw is not a control channel. It is a symptom.

The morphology descriptor's eighth channel, `dpsi`, has been in the counted alphabet since
EXP-005f. It should not have been, and the reason is in the program definition rather than in
any measurement: **no program can request a yaw.** `program.py:31` records that the `sidle`
field was removed because it was identically 0.000 in all 278 corpus programs, and `decode`
takes heading straight from the path tangent (`root_xz, heading = _path_channels(xy, fps)`).
Every body enumerator copies `base.lat` byte-identically — that is what makes the whole
same-route comparison valid — so within a scene **every candidate commands exactly the same
heading**, and any difference in realised `dpsi` is the decoder's response, never an
instruction.

What it responds to is duck and lift:

| | corr with realised `dpsi` |
|---|---|
| requested dip | **+0.276** |
| requested tuck | +0.023 |
| requested lift | **+0.515** |

and it is the least reliable bit in the signature:

| channel | seeds agreeing with the program's modal value | coin-flip programs |
|---|---|---|
| tuck | 1.000 | 0.0 % |
| liftL | 0.989 | 2.4 % |
| liftR | 0.970 | 6.2 % |
| duck | 0.953 | 9.0 % |
| **yaw** | **0.929** | **15.2 %** |

Counting it inflates the discrete alphabet by about a quarter (5.8 → 4.5 modal signatures per
scene across EXP-005f's six scenes). The gate's pre-committed metric keeps it, because changing
a metric after seeing its result is exactly the failure this project has been guarding against;
it is retired in the EXP-005i secondary analysis instead, and reported both ways.

This is the answer to the guidance's *"yaw still has to earn its place"*. It cannot earn a place
as a **capability**, because there is no channel through which to ask for one. That is a
stronger and cleaner negative than "it did not improve clearance".

### Lift is not dead. It is expensive.

Tuck and yaw are the two channels that measure nothing. Lift is the opposite problem — it
fires on ~48 % of clips, but asking for it costs most of the clip:

| requested | n | fraction valid (goal reached ∧ collision-free) |
|---|---|---|
| lift 0.00 | 108 | **0.634** |
| lift 0.35 | 108 | **0.204** |
| tuck 0.00 | 72 | 0.486 |
| tuck 0.40 | 72 | 0.410 |
| tuck 0.85 | 72 | 0.361 |
| neither tuck nor lift | 36 | **0.704** |
| tuck only | 72 | 0.600 |
| lift only | 36 | 0.269 |
| tuck **and** lift | 72 | **0.171** |

Dip runs the other way — validity *rises* with a deeper duck (0.333 at dip 0 to 0.593 at dip
0.50), which is what a beam scene should do: the duck is what makes the clip collision-free.

Both tuck and lift enter the same code path — `decode` writes the `global_joints_positions`
channel only when one of them clears its `ACTIVE` floor (`program.py:234`) — so limb targets
are the single place where a body request can turn a clip invalid. But the split above shows it
is not the *channel* that is fragile: tuck alone costs 10 points of validity, lift alone costs
44. Whether that is the prior refusing the posture or locomotion failing to reach the goal is a
question the new ledger's `goal_err` and `pen_frac` features answer directly, and section 23
reports it.

### The audit so far

| channel | commandable | measurable effect | cost to validity | verdict |
|---|---|---|---|---|
| dip / duck | yes | yes, large | **improves** it | the one solid capability |
| tuck | yes | **none** above width noise | −10 pts | dead channel, negative control |
| lift | yes | yes | **−44 pts** | real but expensive |
| yaw | **no** | correlated with duck/lift | n/a | side effect, not a capability |
| foot order | no | forced by a sign comparison | n/a | retired in EXP-005f |
| velocities, foot contacts | **no** — unreachable in the 414-d feature | n/a | n/a | out of the API |

Six nominal degrees of body freedom; one of them works cleanly.

### The headline funnel, rebuilt so that it is actually a funnel

The guidance proposes a figure of the form

    43-D program -> 36 requested -> 14 valid -> 7 distinguishable -> 3.5 stable strategies

It is the right figure and it is the strongest single claim this project has. But as written the
stages are **not nested**, and one of the numbers is not scale-invariant, so it cannot be drawn
as a funnel without fixing both.

*Not nested.* "7 distinguishable" is an ε-net over a continuous descriptor; "3.5 stable
strategies" counts distinct discrete active sets. Neither is a subset of the other — recomputed
on the same six scenes, the discrete-mode count (3.0) comes out **larger** than the ε-net count
in q99 units (2.5). A funnel whose fourth stage can exceed its third is measuring two different
things on one axis.

*Not scale-invariant.* At the same ε = 3, the ε-net has **6.2** members per scene in pooled
covariance-σ units and **2.5** in q99 units, because q99 ≈ 2.33 σ. The middle number of the
headline figure is a factor of 2.5 wide depending on a unit the figure does not state. This is
the guidance's own §6 point turned on the guidance's own figure.

The nested version, one filter chain, macro-averaged over EXP-005f's six scenes:

| stage | per scene | filter |
|---|---|---|
| requested programs | 36 | the dip × tuck × lift ladder |
| kinematically valid | 14.0 | goal reached ∧ collision-free on > half the seeds |
| valid **and** stable | 10.2 | one modal active set on ≥ 80 % of seeds |
| distinct discrete modes | **3.0** | distinct active sets among those |
| after dropping yaw | **2.2** | the uncommandable bit removed |

    36 requested  ->  14 valid  ->  10.2 stable  ->  3.0 modes  ->  2.2 commandable modes

Every stage is a subset of the one above it. The continuous ε-net belongs beside this chain as
a separate axis with its unit named, not inside it.

The claim that survives is *stronger* than the one it replaces, not weaker: a 43-dimensional
constraint interface, exercised across 36 deliberately spread requests, yields **about two
reliably commandable whole-body strategies per scene**. And the last two arrows are the ones a
reviewer will care about — they are where nominal interface dimensionality stops being
capability.
