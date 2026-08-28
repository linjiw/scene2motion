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

1. **Multimodality of traversal strategy from a large prior.** Every scene-aware humanoid
   competitor is a single deterministic policy: one geometry, one behaviour. Nobody has
   shown a pretrained prior yielding a *distribution* over topologically distinct
   strategies for the same aperture — duck vs. go around vs. turn sideways — nor scored
   that set. An RL policy structurally cannot produce this. → **EXP-003**.
2. **Deliberative long-horizon whole-body planning vs. reactive local control.** Gallant
   perceives ±0.8 m; 2604.17335 predicts 0.5 s replanned at 4 Hz; the only true
   confined-space whole-body planner (2608.10220) takes 2–6 min/solve and is sim-only.
   ARDY's `future_constraints_proj` accepts constraints beyond the current generation
   window at 33 ms/step. → the anticipation result in §5 is the first evidence for this.
3. **Joint text-semantic + 3D-geometric control on a physically tracked robot.** No robot
   system in the survey accepts language (Gallant: goal position; 2604.17335: a heading
   *vector only*; HumanoidPF: goal point). That cell is empty.

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
- **"Ducking makes you wider" was an artefact.** EXP-001 first compared each clip against
  its own opening window; a gait's opening arm swing is simply wider than its steady state.
  Under matched control the effect vanishes — duck and width are roughly decoupled.
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
| PELVIS | either | 75.0 % | 100 % | **28.1 %** | 21.1 % |
| STANDING | either | **31.2 %** | 100 % | 100 % | 31.2 % |
| ADAPTIVE | path only | 68.8 % | 100 % | 51.1 % | 35.2 % |
| ADAPTIVE | **adapted** | 68.8 % | 100 % | **80.7 %** | **55.5 %** |

128 scenes, 6 families, 4 seeds/rung, 608 rows, 141 s wall clock.

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
| `overhead_beam` | **0 %** | 0 % | **83.3 %** | strong duck channel |
| `partial_beam` | 25.0 % | 100 % | **100 %** | duckable *or* avoidable |
| `beam_and_gap` | 0 % | 0 % | **43.8 %** | duck + narrow in sequence |
| `pillar` (control) | 79.2 % | 100 % | 95.8 % | needs no body adaptation |
| `narrow_gap` | 16.7 % | 0 % | 16.7 % | saturated lateral channel |
| `low_obstacle` | 0 % (100 % plan feasible, 0 % collision-free) | 0 % | 4.2 % | weak step-over channel |

**Reproducibility caveat.** `torch.manual_seed` is set per generation *batch*, so a sample's
noise depends on its position in the batch, and changing the scene suite changes batch
composition. Between the two EXP-002 runs this flipped one `pillar` scene (83.3 % -> 79.2 %
for PELVIS). Per-sample seeding would remove the wobble and is worth doing before any
headline number is quoted to a decimal place.

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
write a paper that does not replicate, so the metric tables keep them in separate columns.

---

## 7. Experiment ladder

| id | question | status |
|---|---|---|
| EXP-000 | geometry audit: head gap, capsule coverage, qpos residual | **done** |
| EXP-001/b/c | what body envelopes can the frozen prior reach, per channel? | **done** |
| EXP-002 | PELVIS vs STANDING vs ADAPTIVE, end to end | **done** |
| EXP-003 | how many topologically distinct strategies does the prior realise per aperture? | **done** |
| EXP-004 | counterfactual locality: raise a beam, everything else fixed — does only the crouch change? | next |
| EXP-005 | learned `f_φ(S, s, g) → C` (V1) vs the ADAPTIVE oracle | after 003/004 |
| EXP-006 | physics critic from tracker rollouts — blocked on the SONIC invocation | deferred |

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
34 cm of crouch and 55 cm of lateral displacement, each realised on every seed. **A single
deterministic policy — which is what every scene-aware humanoid competitor in §0 is —
scores 1 here by construction.** This is the cleanest evidence so far for surviving claim
(1), and unlike the ducking figure it is not something the prior-art systems can produce at
all.

Caveat worth keeping: `pillar`'s left/right asymmetry (79 % vs 97 %) is real and
unexplained — the two routes should be near-mirror-images, so something in either the
scene construction or the prior's turning behaviour is not symmetric. That is EXP-004's
first job.

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
