# Scene2Motion-G1 — Research Report

> **Revalidation notice, 2026-08-30.** A sampling audit found that the per-sample seeded
> runner restarted each seed at every 52-frame autoregressive window. Results produced via
> long-clip `generate(seeds=...)` runs—including the Phase 4 hard-set table—used this legacy
> noise stream v1. They remain reproducible evidence about that controlled sampler, but they
> are not being treated as confirmatory evidence about ARDY's intended sampler until rerun
> under noise stream v2. Cache version 2 prevents accidental mixing. Scope and rerun order:
> [`revalidation-2026-08-30.md`](revalidation-2026-08-30.md).

**What was asked:** turn a frozen humanoid motion prior into a scene-conditioned *whole-body*
motion planner for the Unitree G1 — scene geometry plus start and goal in, whole-body motion out
that reaches the goal, avoids obstacles with the whole body, and is physically executable.

**What was found:** the frozen prior exposes a 43-dimensional constraint interface whose
*reliably commandable* subset, on this robot, is **narrow — duck depth, plus a weak
small-amplitude step-over.** Everything else is
either not requestable, not distinguishable from the prior's own sampling noise, structurally
capped by the robot's geometry, or not executable by a physics tracker. The planning decision
this supports is real but scalar: *duck, and by how much*. No learned generator, set-valued
proposer or reranker is justified.

**What the audit is worth, as a number.** The same 36 programs and the same generated clips,
counted four ways — each naive row a choice a careful person could make without doing anything
obviously wrong:

| how the capability is counted | modes/scene |
|---|---|
| 1 seed, any change > 1 mm | 9.00 |
| 1 seed, round 1 cm threshold | **10.00** |
| 1 seed, 1 cm, dropping clips that never validate | 6.67 |
| **6 seeds, paired, q99-calibrated, stability ≥ 0.8** | **1.67** |
| …and after the physics tracker | **~1** |

**A single-seed reading with a round threshold overstates this prior's body repertoire by 6×
kinematically and ~10× after physics.** That gap is the contribution
(`experiments/audit_delta.py`).

**What the project is actually a contribution to:** a **capability-auditing methodology** for
frozen generative priors used as robot planners — how to measure what such a prior will
*reliably* let you ask for, calibrated against its own stochasticity, with the controls needed to
keep the answer honest. Along the way it produced a catalogue of twelve measurement defects, ten
of which biased results *toward* the hypothesis under test.

Status: 57 commits, 12 experiment families, ~34 000 ARDY generations, one end-to-end physics
validation. Research log: `docs/design.md` (33 sections, chronological — later sections correct
earlier ones). This report is the current state, organised by topic rather than by date.

---

## 1. How the question changed three times

The project ran under three guidance documents, each of which moved the research object.

| | framing | what forced the change |
|---|---|---|
| **V0** | scene → whole-body motion, via a classical planner writing ARDY constraints | — |
| **V1** | learn `p(C \| S, s, g)`, a *distribution over constraint programs*, ARDY frozen as decoder | routes were already solved by classical search; the interesting object is the program |
| **V2** | keep classical routes, learn only `q(b \| S, r)` over *body* programs | EXP-005d: a learned proposer had nothing to win on traversal — the enumerator already reached the ceiling |
| **V3** | the real object is **addressability**: the map `c → (v, σ, m)` is stochastic and low-bandwidth | the NULL-SEED control showed a program has a *distribution* over descriptors, not a descriptor |

The V3 reframing is the one that stuck, and the rest of this report is organised around it.

---

## 2. What was built

### 2.1 Geometry and collision — `scene2motion/robot.py` (~300 lines)

Exact whole-body collision against the scene using **MuJoCo with Unitree's own collision
primitives** from `ardy/assets/skeletons/g1skel34/xml/g1.xml`, not a bounding box.

Two design decisions carry the rest of the project:

* **`BODY_MARGIN = 0.04`.** The shipped collision primitives *under-cover* the visual meshes.
  This is a measured correction, not a safety fudge. Without it, "collision-free" is a statement
  about capsules, not about the robot.
* **`_extent(g, n)` projects a capsule's surface onto a query direction** as
  `half_length·|a·n| + radius`. An early version used the radius alone, which silently reported
  a horizontal capsule as a sphere.

The head sits **23.6 cm above ARDY's highest joint**, so any clearance measured on the skeleton
alone is wrong by that much — the reason a real collision model was necessary at all.

`trajectory_report` separates *scene* collision (the planning metric) from *foot-versus-floor*
penetration (a physical-consistency signal about the generated motion), because `g1.xml`'s
`<pair>` contacts bypass `contype` filtering and would otherwise contaminate the first with the
second.

### 2.2 The constraint interface — `scene2motion/constraints.py`

The adapter from a plan to ARDY's conditioning. ARDY conditions by overwriting slices of a
**414-dimensional per-frame feature vector** under a per-`(frame, channel)` binary mask. Five
channels are writable:

| channel | feature block | expresses |
|---|---|---|
| `root_2d` | `root_positions` xz | where to go |
| `root_y_pos` | `root_positions` y | **duck** |
| `global_root_heading` | — | turn / sidle |
| `global_joints_rots` | `global_rot_data` | limb orientation |
| `global_joints_positions` | `local_joints_positions` | tuck / step-over |

`velocities` and `foot_contacts` are **not reachable** — a hard limit of the API, not a design
choice.

The single most important fact about this table was found late (§4.4): **the pose decoder reads
only `global_rot_data` and `root_positions`.**

### 2.3 The action space — `scene2motion/program.py`

A body program is **43 numbers**: 22 lateral chord knots + 4 adaptation slots × 5 fields
(`mid`, `half_width`, `dip`, `tuck`, `lift`) + speed. `from_vec` clips to `[-1, 1]` — *the box
is the capability set*, so an out-of-range request is impossible rather than merely bad.

There is **no yaw/sidle field**: it was identically 0.000 in all 278 corpus programs. That
absence turns out to matter (§4.3).

Decode resamples by **arc length**, not chord abscissa — the latter left up to 0.30 m of lag
that no number of knots removes.

### 2.4 Planning — `scene2motion/planner.py`

A* over `(x, y, body-mode)` with a move/switch successor split (`8·M → 8 + (M−1)`) and a
flood-fill early exit, giving ~0.01 s per plan. Modes come from a **conformally calibrated**
capability envelope (`envelope.py`, from EXP-001d at α = 0.1), so the planner refuses rather
than guesses when no mode fits — and `_diagnose_refusal` says *which* geometric constraint bound.

`_dilate_channel` spreads each adaptation channel **independently**. Dilating at mode level
instead deleted adaptations outright: spreading a duck near a gap took tuck from 16 → 6 → 0
frames and dropped `beam_and_gap` from 94 % to 39 %.

### 2.5 Generation — `scene2motion/runner.py`

Cache-first wrapper around ARDY. `_per_sample_noise` monkey-patches `torch.randn` to fill
row-by-row from per-sample generators, which is what makes **matched pairs** possible: DDIM at
η = 0 is deterministic, so the same seed with two different requests is a controlled comparison.
Before this existed, per-batch seeding invalidated every matched pair (0.085 m null against a
0.137 m signal).

### 2.6 Measurement — `morphology.py`, `metrics.py`, `funnel.py`

An 8-channel morphology descriptor (`dh_top`, `dw_left/right`, `dz_foot_left/right`, `t_lead`,
`t_duration`, `dpsi`), always as a **paired delta** against a neutral-body control on the same
route and the same seed. `d_morph` is a whitened distance in units of ARDY's own seed noise;
`epsilon_net` deduplicates before counting so a densely-sampled region cannot inflate coverage.

`funnel.py` computes the capability funnel identically across configurations, each thresholded on
its **own** measured q99, with `--q99_from` to hold the threshold fixed when the question is a
behaviour change rather than an addressability change.

### 2.7 Physics — `scene2motion/sonic_export.py`

ARDY qpos → SONIC motion-library. The load-bearing fact: **ARDY's `qpos[7:36]` is in the same
joint order as SONIC's 29-DOF G1**, verified name by name. `check_joint_order` re-asserts it at
export time, because a silent reordering would track a plausible but *different* motion and no
downstream assertion would catch it.

---

## 3. Experiments

| # | question | design | headline |
|---|---|---|---|
| 000 | does the collision model match the robot? | geometry audit vs meshes | head is 23.6 cm above the highest joint; `BODY_MARGIN = 0.04` |
| 001 | what can the prior be asked for at all? | 87 conditions, ducks × tucks × sidles | duck works; heading tracked ~90 % |
| 001b | how narrow can G1 get? | sidle × tuck grid | half-width floor ≈ 0.245 m |
| 001d | calibrate the envelope | 320 rows, conformal α = 0.1 | the old mode table was **16.6 cm optimistic** on `duck_max` |
| 002 | does mode-aware planning beat baselines? | 128 scenes, 3 planners | standing accepts 47.6 %; duck-only accepts 85.7 % but only **32.3 %** collision-free; mode-aware 81.5 % / **76.1 %** |
| 005c | are refusals recoverable? | 120 refused scenes, 8 seeds | 11.7 % recoverable; **`low_obstacle` 0/20** |
| 005d | does a learned proposer have room? | ORACLE@K, 128 scenes | ~1.0 strategies/scene — **the gate fired: no room on traversal** |
| 005e | is the enumerator complete? | 150 scenes vs 40 random restarts | recall 0.32; misses are *body*, not route |
| 005f | what is the seed-noise floor? | 6 scenes × 36 programs × 6 seeds | tuck below noise; foot order uncontrollable |
| 005g | can classical enumeration cover the body set? | 22 scenes, 8 arms, 15 778 clips | best equal-call **0.592** vs a **0.361** resampling floor — nowhere near the pre-committed 0.90 |
| 005h | is the diversity decision-relevant? | preference regret, 3 objective groups | **passes** at 45.5 % — but the winners are `none`/`duck` only (§4.6) |
| 005i | support, selection, or ARDY? | POOL-ORACLE on the ledger | **support**: a perfect reranker wins **0.023**; 0.398 unreachable |
| 006 | is it the inference setting? | 6 configs, families the gate never saw | stronger guidance is **strictly harmful**; 5 steps beats 10 |
| 008 | is it the channel? (arm) | POS vs ROT±, matched seeds | rotation **8.68 σ** vs position **1.57 σ**, which flips sign twice |
| 009 | is it the renderer? | OLD vs NEW, request held fixed | **6 %** of the gap, p = 0.46 — my hypothesis refuted |
| 010 | is it the channel? (leg) | POS vs ROT±, airborne-gated | position reaches **10.64 σ** — the channel is not uniformly bad |
| 011 | does any of it survive physics? | SONIC, one run per body | duck 1.000/0.750/0.625 by depth; **lift 0.000** |

---

## 4. Findings

### 4.1 The capability funnel

Three runs of the same code, same scenes, same seeds, changing only the configuration:

```
43-D constraint interface
  →  36  requested body programs
  →  13.2 kinematically valid        (goal reached, collision-free, > half the seeds)
  →  10.8 valid AND stable           (one modal active set on ≥ 80 % of seeds)
  →   1.7 distinct addressable modes
  →  1-2   dynamically executable     (duck, strong; small gated step-over, weak — §4.7)
```

The originally reported 3.00 modes/scene decomposes into **2.17 of behaviour** (once a
self-contradictory renderer request is removed) plus a **stale noise threshold**. Every stage of
this funnel has shrunk under scrutiny; none has ever grown.

### 4.2 The separating axis is the *representation*, not the model

`local_joints_positions[t,j] = global_pos − pelvis_pos + [0, pelvis_y, 0]` — **root-relative in
the ground plane, absolute in height.** That one line predicts which capabilities work:

| request | physical quantity | encoding | \|effect\|/sd |
|---|---|---|---|
| duck | height, root | absolute | works, 74–100 % |
| **lift** | height, joint | **absolute** | **10.64** |
| **tuck** | lateral extent | **root-relative, ground-plane** | **1.57, sign-flipping** |

A height request names a coordinate the feature vector holds directly. A lateral request names
one the model must reconcile against a root path it is generating concurrently — and it does so
badly. This is a statement about ARDY's *representation*, and it would transfer to any prior
with the same encoding.

### 4.3 Two channels are not requestable at all

**Yaw** — `program.py` has no sidle field, `decode` takes heading from the path tangent, and
every body enumerator copies `base.lat` byte-identically. So within a scene every candidate
commands the *same* heading, and realised `dpsi` is a side effect of duck (r = +0.28) and lift
(r = +0.52). It is also the least stable bit (75 % seed agreement) and inflated the counted
alphabet by ~25 %. *ARDY's heading channel itself works fine* (90° sidle tracked at 94 %) — the
limitation is our action space, not the prior.

**Velocities and foot contacts** are outside the writable API entirely.

### 4.4 The decoder reads two feature blocks; we were writing into a third

`ardy/exports/mujoco.py:156` — `dict_to_qpos` takes **only** `local_rot_mats` and
`root_positions`. So every collision check, half-width and capability number in this project is a
function of `global_rot_data` and `root_positions` and nothing else. `local_joints_positions` —
our sole channel for tuck and step-over — reaches the body only through the denoiser's *learned*
coupling between blocks.

The clean version of that theory is false and our own data says so: `global_root_heading` is also
unread, yet 90° sidle requests track at 94 %. Being unread is not sufficient for weakness. But it
explains why writing the unread channel with a **no-op** request (POS at amplitude 0.00, asking
for the arms' own nominal positions) still moves top height by 49.8 mm.

### 4.5 Tuck is capped by the robot, not the prior

Optimising half-width over all 14 arm DOFs with 60 restarts: minimum **0.2065 m**, bound by the
**thighs**. Tucking buys ~32 mm and then structure takes over. Against a 55 mm seed-noise σ, the
entire available effect is under 1 σ. No channel and no inference setting recovers it —
confirmed by EXP-006, where tuck fires on **0.000** of clips at every configuration.

### 4.6 The decision that remains is scalar

EXP-005h passes its pre-committed kill condition — 45.5 % of scenes [22.7, 68.2] have ≥ 2 modes
optimal under different declared preferences, against a 25 % threshold. But across 154
preference-winning slots the winners are:

| body | slots |
|---|---|
| `none` (plain walk) | 78 |
| `duck` | 70 |
| `yaw` (uncommandable) | 6 |
| **anything with a lift** | **0** |

So the diversity is **duck versus don't duck**. That is a real trade-off a planner must make, and
because it is carried entirely by duck it **survives the tracker** — unlike everything else here.
But it is one capability with an amplitude, not a set of strategies.

### 4.7 Physics removes the rest

| body | tracking success | accel_dist | head+hand error |
|---|---|---|---|
| neutral | 0.875 | 2.04 | 33.5 mm |
| duck-shallow | **1.000** | 1.90 | 29.7 mm |
| duck | 0.750 | 5.99 | 45.6 mm |
| duck-deep | 0.625 | 10.18 | 71.9 mm |
| **lift** *(two-legged raise, 0.35 m — see below)* | **0.000** | **28.31** | — |
| **duck + lift** *(same)* | **0.000** | 24.64 | — |

Duck degrades gracefully and monotonically with depth — that is what an executable adaptation
looks like.

**The lift rows are wrong, and the correction matters more than the original result.** Those
requests raised **both** legs at every frame in the window. `planner._limb_targets` gates the
lift **per side** on that leg already being airborne; my experiment code did not. A contact
analysis (EXP-013) found `mean_contacts = 0.00` — no foot on the ground at any frame. The
reference was ballistic, and no controller can track a jump.

Re-run with the shipped gating and an amplitude the prior does not over-drive:

| body, lift = 0.08 m | ground contacts | success | progress | accel_dist |
|---|---|---|---|---|
| neutral | 5.63 | 1.000 | 1.000 | 1.71 |
| two-legged raise *(what was tested before)* | 0.00 | **0.000** | 0.145 | 21.33 |
| **per-side gated** | **1.45** | **0.375** | **0.746** | **6.49** |

**Step-over is weakly executable, not absent.** Measured against the threshold this report
pre-committed to in §8.1 — withdrawal at ≥ 0.5 success over 16 seeds — 0.375 over 8 seeds does
**not** earn the strict withdrawal. But 0.000 and 0.375 are not the same claim, and the whole
difference is in how the request was formed. The boundary lies somewhere between 0.08 m (tracks,
1.45 contacts) and 0.35 m (does not, 0.05 contacts) and has not been located.

**A result that has to be restated.** The conformal kinematic envelope refused `low_obstacle`
0/20, and the tracker initially appeared to confirm it at lift 0/8 — two independent methods
agreeing. That agreement is now known to be partly coincidental: the tracker's 0/8 was measuring
a jump. The envelope's refusal still stands on its own terms and the tracker still says a
*large* step-over is unexecutable, but "a calibrated kinematic refusal predicted a dynamic one"
was a stronger claim than the evidence supports, and it is withdrawn.

### 4.8 The execution certificate is not yet measured

The planner certifies the *generated* motion. Existing SONIC runs establish obstacle-free
tracking survival and scalar pose error, but save no achieved qpos, so they cannot establish how
much scene clearance survives execution. The previously reported 85.6 mm threshold and 23.5%
shortfall are withdrawn as an execution claim:

* SONIC's `vr_3points` is torso plus two wrists, not head plus hands;
* `mpjpe_l` removes root translation and is an unsigned time/joint mean, not signed clearance
  loss along the active obstacle normal;
* `G1Body` already applies `BODY_MARGIN` to scene geometry, so adding 40 mm again double-counts
  it; and
* 23.5% was the share of generated clips below a proxy threshold, not an observed physical
  failure rate.

The needed quantity is paired and directional:

    loss = generated_clearance_with_margin - achieved_clearance_with_margin

Fit a one-sided held-out bound `tau(d)` on that loss, then require generated clearance
`>= tau(d)` for collision freedom or `>= 0.18 + tau(d)` to retain the 18 cm target. The new
achieved-state callback records the missing qpos, but no such archive has yet been generated.
See `docs/exec-gate-audit.md` for the evidence and locked data schema.

---

## 5. Methodology: the defect ledger

**First, a correction to this section's own headline.** The research log says "twelve defects,
ten favouring". An independent audit of that claim found both numbers wrong. *Twelve* is a
**running tally**, not a census: it was assembled from three places in the log and omits every
defect found before or after those three windows. A proper census gives **~24 distinct
measurement, metric, control and harness defects** at the same bundling, and ~32 if the bundles
are split the way the log splits its own. *Ten* is also off by one against the log's own
arithmetic, and the direction attribution of one bundle is wrong in the other direction — four of
the five `program.py` defects made the *compressed program* look lossier than it is, which tilts
**against** a learned proposer, not toward it.

The corrected statement: **of roughly 24 measurement defects, about 15 biased results toward the
hypothesis under test.** The skew is real and, at the census count, larger in absolute terms than
the running tally suggested. What is *not* reliable is any precise ratio, and this report should
not have quoted one — a tally accumulated across a research log is not a measurement, and I
applied less rigour to counting my own errors than to any experiment in this report.

That direction is not bad luck — each defect sat where I had a hypothesis and the metric had
freedom to agree with it.

| defect | direction | how it was caught |
|---|---|---|
| 1-bit coverage signature | favours | follow-up experiment questioned the gate |
| hardcoded noise σ in baseline D | favours | adversarial code review |
| moving hypervolume reference box | favours | unit check on a toy set, before any GPU |
| pool dedup stole candidates from arms | favours | self-review before data |
| allocation control sampled a pool no method sees | neutral | self-review before data |
| **POOL-ORACLE was a division identity** | **favours** | **multi-agent audit** |
| EXP-005h kill condition could not fire | favours | multi-agent audit |
| `_limb_targets` pinned heights to the un-ducked nominal | favours | multi-agent audit |
| whitener dropped the `1/√n_int` normalisation | favours | multi-agent audit |
| §I rescale divided out the correlation it tested | favours | multi-agent audit |
| hardcoded 2.33σ threshold in the live gate | favours | audit's *dropped* rank-3 tail |
| `sig_key([sg])` double-wrap emptied the matrix | favours | reading output that could not be true |

*(The table lists the twelve from the running tally, which is what the log recorded as it went.
It is a sample of the ~24, not the census.)*

Plus one defect in **third-party code** — ARDY's `space_timesteps` returns a schedule of length
`num_base_steps` regardless of what is requested, so asking for more than 10 denoising steps
indexes past the end and dies in a device-side assert — and four **integration** failures in the
SONIC bring-up (EULA prompt with no stdin, relative path resolved against the wrong cwd,
cross-checkout import, and `gear_sonic/trl` shadowing the installed `trl`).

**What actually worked as a defence**, in order of value:

1. **Null controls built before the number was read.** NULL-SEED (the same program K times on
   disjoint seeds) caught its own ε sitting inside ARDY's scatter. EXP-009's untouched-program
   rows reproduced at delta *exactly* 0.0000, proving isolation.
2. **Sign controls** — but only where the geometry supports them. EXP-008's arm sign control
   worked; I reused it on the leg in EXP-010 and it was invalid by construction, because rotating
   a chain hanging straight down raises the foot as `L(1−cos θ)`, which is *even in sign*.
3. **Positive controls.** EXP-009's first run reported +0.000 everywhere and its null *passed* —
   a one-sided control cannot distinguish "correctly isolated" from "nothing happened".
4. **Identity checks.** Printing `min(K,M)/M` beside the POOL-ORACLE row made a tautology visible
   instead of publishable.
5. **Adversarial multi-agent audit.** Five of twelve defects, including the worst.
6. **Recording predictions before running.** Five pre-registered predictions, four wrong, each
   informative *because* it was written down.

**And one claim in an earlier draft of this section was false.** It said the two defects that did
not favour the hypothesis "were caught by controls built before the numbers were read". They were
not. `sig_key` was caught by *disbelieving output* — a matrix in which the request does not change
the outcome is not credible — and EXP-009's inert run was caught only after noticing its null
control had **passed** while hiding the inertness, which is what prompted adding a positive
control. Neither was caught by a control that existed in advance. The lesson is the opposite of
the one I wrote: **a control you already have can hide the failure it was built to catch**, and
what actually worked was refusing to believe a number.

**The single highest-leverage engineering decision** was making experiments emit a
**per-candidate ledger** of raw evidence rather than summary statistics. The gate's first run
discarded its per-seed outcomes and put five analyses behind a second 61-minute GPU pass; once
the ledger existed, a hardcoded-threshold bug was repaired post-hoc with zero GPU time.

---

## 6. What this means

**For this system.** ARDY-G1 exposes a rich-looking interface — 5 channels, 43 program
dimensions, 36 spread requests — and delivers **one strong executable body axis and one weak
one**. The boundary between
what works and what does not is predictable from the *representation* (absolute-height versus
root-relative-lateral), from the *robot* (the thighs cap narrowing), and from the *action space*
(no yaw field), and **not** from anything one would call the model's competence.

**For the original goal.** Scene-conditioned whole-body traversal on G1 is achievable for
overhead clearance, and — through text rather than through the constraint interface — for the
lower half of the `low_obstacle` ladder. The 0/20 refusal of that family was correct *for the
action space it was calibrated over* and wrong about the robot.

**For the learned-model question.** Settled, negatively, three times over: a perfect reranker
wins 0.023; the bottleneck is candidate *support* rather than selection; and the surviving
decision is a scalar. Nothing here justifies a generative program model.

**What is genuinely reusable.** The audit methodology, the noise-calibrated distinguishability
metric, the per-sample-addressability-versus-expected-effect distinction, the ledger discipline,
and the control patterns in §5.

---

## 7. Honest limits

* **The funnel rests on 6 scenes**, the gate on 22, the tracker on 6 bodies × 8 seeds. Nothing
  here is a large-sample result.
* **One prior, one robot, one controller.** Every claim is about ARDY-G1-Horizon52 with SONIC.
* **"Lift is 0/8 executable" is a statement about *this* controller.** SONIC trains on retargeted
  human capture; an out-of-distribution reference and an infeasible one fail identically from the
  outside. Separating them needs a tracker with step-over in its data.
* **Scenes are synthetic ladders**, not scanned environments.
* **The rotation channel was never made into an action space** — it was tested in two ad-hoc
  experiments and shown better for the arm, comparable for the leg.
* **`mpjpe` is a proxy** for the clearance that actually matters; the achieved states SONIC would
  need to dump for a direct measurement are not saved.

---

## 8. Next steps

### 8.0 What has already been done rather than recommended

Four experiments were run *while writing this report*, because each could change what the report
says and none was expensive. Three of the four overturned or corrected something.

| | question | outcome |
|---|---|---|
| **EXP-012** | does a lift commanded through the *rotation* channel track, given that the position channel buys foot height by raising the pelvis 122 mm — a hop? | **hypothesis refuted.** Pelvis-still rotation lift also tracks 0/8. But it gets 37 % through the clip against position's 1.9 %, and halves `accel_dist` 26.7 → 13.6, so the hop was about half the problem |
| **EXP-013** | is the lift reference *dynamically* infeasible, or just unfamiliar to SONIC? (a controller-independent ZMP test) | answered a different and more important question — see below |
| **EXP-014** | does the *shipped* gated step-over track? | at 0.35 m both are airborne — but **at 0.08 m the gated one keeps 1.45 contacts and tracks 0.375**, overturning §4.7 |
| amplitude sweep | is 0.35 m simply too big an ask? | ARDY **over-responds ~2.5×** (an 80 mm request yields +197 mm of foot rise) and contact falls monotonically: 5.34 → 1.34 → 0.27 across 0.08 → 0.35 m |

**EXP-013 found an error in my own experiments, and it is the most important thing in this
section.** Computing foot contacts on the lift reference gave `mean_contacts = 0.00` over the
whole interaction window — *no foot on the ground at any frame*. A contactless reference is
ballistic and no controller can track it. The cause was mine: `planner._limb_targets` gates the
lift **per side** on that leg already being airborne, and EXP-011/012/013 all raised **both**
legs. **"Lift is not executable" was a claim about a jump.**

That is the fourth time in this project that a capability was declared absent and the cause was
how it was asked for. It is why §8.1 leads where it does.

### 8.1 Tier 1 — resolves an open claim, costs hours

**(a) Finish the contact-consistent step-over ladder — partly done, and it already overturned
§4.7.** Two amplitudes are now tracked: 0.35 m fails (0.000, 0.05 contacts) and 0.08 m succeeds
partially (**0.375**, 1.45 contacts). The pre-committed bar was ≥ 0.5 over 16 seeds, which 0.375
over 8 does not clear, so what remains is to run the intermediate amplitudes (0.12, 0.16, 0.20,
0.25) at 16 seeds and locate the boundary. **This is now the highest-value experiment in the
project**, because it decides whether the executable repertoire is one axis plus a fringe case or
genuinely two.

**(b) Elicit the behaviour with TEXT, address it with the ROOT.** *The best idea to come out of
the next-steps panel, and one I had not considered.* ARDY is text-conditioned, and the root
channels are the ones that demonstrably work. So: prompt for the behaviour ("a person steps over
an obstacle"), constrain **only** the root path, and never touch the body slices at all. This
tests whether the frozen prior *contains* a supported, contact-consistent step-over that our
constraint interface was simply unable to ask for — which is the strongest remaining threat to
the report's central claim. Matched per-sample seeds against a plain-walk prompt; kill condition
is failure to separate swing height from the walk baseline by more than the q99 seed noise.
Cost: one session.

Both are cheap and both can overturn §4.7. **They should run before anything in Tier 2.**

### 8.2 Tier 2 — needed before this is publishable

**(c) Scale the funnel.** It rests on **6 scenes**. Pre-register a family-stratified suite with
enough seeds to *certify* rather than estimate — EXP-007 established that τ = 0.8 needs n ≥ 14
unanimous seeds and the funnel used 6. Pre-commit the kill condition: if the 95 % scene-level
upper bound on addressable modes/scene exceeds 3.0, the headline is not supportable. ~2 GPU-hours.

**(d) Build the rotation-channel action space.** §28 of the log concluded exactly this and the
project never acted on it — EXP-008 and EXP-010 tested the channel in two ad-hoc experiments and
it was never made into a program. Note §29 narrowed *why* this matters: the axis is height versus
lateral, so rotation is expected to help the arm and to buy pelvis-stillness for the leg, not to
transform the funnel. Worth doing precisely because the expectation is now specific enough to be
wrong.

**(e) Restate every headline as an interval with a reproduction script.** No number should appear
that a committed script cannot re-derive from a committed ledger. `funnel.py` and
`audit_delta.py` are the model; most other numbers in this report are not yet re-derivable that
way.

### 8.2b The finding that reorders everything above

§4.9 changes which experiments matter. If a text prompt reaches a capability the constraint
interface cannot express, then the priority is no longer "audit the constraint interface more
carefully" but **"map what the text conditioning can and cannot be asked for, and whether it can
be steered precisely enough to plan with."** Three things follow, and they outrank (c) and (d):

* **A prompt ablation.** One prompt was tried because one prompt was in the cache. Run a set —
  duck, step over, sidle, crouch-and-walk, and compound requests — against a fixed root path, and
  build the same commanded→realised matrix for the *text* channel that §4.3 built for the
  constraint channels. This needs the CPU text-encoder service, which is the only infrastructure
  cost in this entire list.
* **Is text addressable?** +57 mm with sd 44.9 mm is a behaviour, not a control. The whole
  apparatus of this project — paired deltas, noise calibration, stability — applies unchanged and
  asks the sharper question: can you request *12 cm* of clearance, or only "step over"?
* **Text × constraint composition.** The planner needs "duck here, then step over that". Text
  names one behaviour; the root channels place it in space. Whether the two compose is the
  question the original project premise assumed away.

### 8.3 Tier 3 — generalisation, and what the method is worth to anyone else

**(f) A second prior, or failing that a second skeleton.** §4.2 makes the project's boldest
claim — that height-works / lateral-fails is a property of the *representation*, not of ARDY —
and it rests on one checkpoint. *The panel corrected my brief here: there is no Kimodo checkout
on this machine; `~/lucid-sonic` is a different project's data root.* The cheap substitute is
ARDY's own **Core and SOMA skeletons** (`ardy/model/registry.py`), which share the representation
but not the robot, and so separate "the encoding does this" from "G1 does this". Kill condition:
if a second skeleton's lateral channel reaches ≥ 5 σ with a stable sign, the representation
explanation is wrong.

**(g) Package the audit as a tool.** The reusable object is three callables: a paired-delta
descriptor, a null-calibrated distinguishability threshold, and a stability filter. Validating it
by having it **reproduce this project's own wrong answers** — the 10.00 naive count and the 1.67
calibrated one — is a better acceptance test than any synthetic case.

### 8.4 What I would not do

**Not a learned proposer, in any form.** Settled three ways: a perfect reranker wins 0.023, the
bottleneck is candidate support rather than selection, and the surviving decision is scalar.

**Not more inference tuning.** EXP-006 closed it: the one knob that should have helped makes seed
scatter 37× worse, and the only free win (5 steps over 10) is already adopted.

**Not a duck-only planner as a research contribution.** It is a day of engineering with a
calibrated rule already derived (§4.8) and nothing left to learn.

### 8.5 The honest assessment

*With one more day:* (a) and (b). Both can overturn the central negative claim, and it would be
poor practice to publish a negative result while its two cheapest refutations sit unrun.

*With one more month:* (c) and (f) — scale the funnel to a pre-registered suite and replicate on
a second skeleton. Those two convert "we measured this on one prior, one robot, six scenes" into
a claim about representations, which is the only version of this work that generalises.

*The thing most likely to be wrong — and it was.* This section originally named §4.7, that lift
is unexecutable, as the claim most likely to fail, on the grounds that *every previous
capability-absent claim in this project failed to survive the experiment that asked properly.*
Experiment (a) was run before this report was finished and duly overturned it: a per-side gated
step-over at 0.08 m tracks 0.375 where the two-legged raise at 0.35 m tracked 0.000. Five for
five.

*So the thing most likely to be wrong now* is the remaining half of the same claim — that the
step-over is only **weakly** executable. It rests on 8 seeds at a single amplitude, below the
0.5-over-16-seeds bar this report set for itself. The amplitude ladder in (a) is unfinished, and
the boundary between 0.08 m (works) and 0.35 m (does not) is unlocated. If the band is wider than
it currently looks, the executable repertoire is two solid axes and several conclusions in §6
soften further.

*One methodological note for whoever continues.* A claim in the panel that fed this section — that
the scene suite labels only two adaptation types — was **false**: the 128-scene suite carries six
(`none` 44, `duck` 32, `step_over` 20, `narrow` 16, `detour` 12, `sidle` 4). It was checked before
being written down, which is the only reason it is not in this report as a finding.

---

## 4.9 (late addition) Text reaches what the constraint interface cannot

Three arms; **A and B differ only in the prompt** — identical root constraints, identical
per-sample seeds, no body slice written in either.

| arm | Δ foot peak | sd | ground contacts | tracked success |
|---|---|---|---|---|
| A `"A person walks forward."` | +0.0 mm | — | 2.49 | **0.875** |
| **B `"A person steps over an obstacle."`** | **+56.9 mm** | 44.9 | **2.80** | **0.625** |
| C walk prompt + gated position lift | +362.1 mm | 33.7 | **0.12** | **0.000** |

The walk baseline peaks at ~0.117 m, so arm B reaches ~0.174 m — clearing the bottom three rungs
of the `low_obstacle` ladder (`0.02, 0.08, 0.15, 0.22, 0.30, 0.38`), the family the conformal
envelope refused **0/20**.

**The failure mode of the constraint interface is instructive.** Asked for a foot 0.35 m higher,
ARDY delivers a foot 0.36 m higher *and a robot in flight*. The request is honoured; the
**behaviour** is not — because "step over" means "raise one foot while the other stays planted",
and the position channel has no way to say the second half. Text does, because the prior learned
the coordinated behaviour from data.

**Limits, plainly.** One prompt, one behaviour, 8 seeds, no ablation — it was tried because it
happened to be in the embedding cache. +57 mm is real but modest and its spread is large (sd
44.9), so text gives a *behaviour*, not an addressable control: you cannot ask for 12 cm. It
tracks at 0.625 against the walk's 0.875, so it is not free. And it does not rescue composability
— the planner needs "duck by 0.31 m here, then step over that", and nothing here shows text can
be steered that precisely.

---

# Phase 4 — Generate → Verify → Repair → Select

Phases 1–3 built an open loop: plan a route, write a duck schedule against a fitted model of
the body, hand it to the frozen prior, and hope. Phase 3's convex teacher halved excess crouch
at 100% collision-free on single-beam scenes, which was the first genuine improvement in the
project — and it was measured entirely inside the distribution the surrogate was fitted on.

The table in this phase is a **legacy noise-stream-v1 result pending exact v2 replication**.
Its generate→verify→repair architecture remains the hypothesis under test; its numerical
effect sizes are not current headline claims.

Phase 4 closes the loop. The system now generates, measures what the motion *actually*
cleared, corrects locally from that measurement, regenerates, and reverifies.

## 9. The verification primitive

`G1Body.trajectory_report` returns one scalar minimum over a whole clip. That answers "did
this collide" and is useless for "where", which is what a local repair needs.
`verify/trace.py` produces clearance as a function of route position on the same 64-sample
grid the schedule lives on, so a deficit at sample *i* maps to a command at sample *i* with no
re-indexing.

Frames map to route position by the robot's own along-route travel, not by frame index. The
prior accelerates from rest and settles at the goal, so a frame-indexed trace would shift the
deficit relative to the schedule by a variable amount and put the repair in the wrong place.

**Measuring it immediately corrected the design.** The first real trace reported a 3-beam
scene at 76 mm of total clearance against a 180 mm target — while its overhead clearance was
341 mm. The robot was squeezing past a wall, not grazing a beam. A repair loop driven by the
undifferentiated minimum would have answered that by crouching, which buys nothing.

Contacts are therefore split by contact-normal direction:

| quantity | meaning | who can fix it |
| --- | --- | --- |
| collision | any clearance < 0 | nobody; reject |
| overhead deficit | headroom below target | the duck schedule |
| lateral deficit | side clearance below target | the route (Phase 4B) |

`MARGIN_M = 0.18` is judged against the overhead channel, which is what the Phase 3 scheduler
always solved against. Judging it against a whole-scene minimum would fail routes the system
was never asked to keep clear — the shipped single-beam demo clips sit at 0.13–0.17 m of total
clearance purely from corridor width.

## 10. The repair operator

    e(s)  = max(0, target − measured_overhead(s))
    dq(s) = e(s) / |g'(q(s))|
    dq    = anticipate(dq)          shifted early, because the body lags
    q'    = smooth(clip(q + dq, 0, 1))

`|g'|` is a **forward** secant over `[q, q+h]`, not a centred one, because `g` is convex over
most of its range and a centred secant borrows steep gain *behind* the current command that a
correction increasing it cannot use. Measured against the exact inverse for a 5 cm deficit,
the forward secant lands within 4 mm of command over `q ∈ [0, 0.7]` where the centred one
undershoots by up to 27% — expensive when the loop stops after two iterations.

The slope is floored at 0.15 m/unit. Near saturation the fitted gain goes flat and dividing by
it turns a 1 cm deficit into a demand for infinite crouch. Where the floor binds the repair is
deliberately partial, and the next verification says whether that was enough.

Anticipation lead is `3·tau·v`, derived from the measured lag rather than the planner's
hand-set `LEAD_S`. **The obvious refinement was tested and refuted**: sweeping the lead over
1.5, 3, 5 and 8 tau, 3 and 5 are identical (1.000 collision-free / 0.417 margin), 1.5 is
worse, and 8 is worse still on margin (0.250) while costing more crouch.

## 11. Genuine OOD failures of the final m018 model

The honesty constraint on this phase was that no repair demo may reuse the quarantined
0.12-margin model's failures. Sweeping the **final m018 system** over 3–6 beam
corridor-blocking scenes (training stopped at 2 beams) found real ones: **4 overhead
collisions from −32 to −105 mm and 11 target-margin shortfalls out of 36.**

## 12. Experiment D — five methods on hard scenes

36 corridor-blocking 3–6 beam scenes, all through the real frozen prior. Collision-free rate
and target-margin satisfaction reported separately, because they are separate claims.

| method | coll-free | meets 0.18 | ARDY calls | peak dip | excess crouch | min overhead |
| --- | --- | --- | --- | --- | --- | --- |
| heuristic | 0.972 | 0.583 | 2.0 | 36.7 cm | 8.3 cm | 18.5 cm |
| qp | 0.889 | 0.028 | 2.0 | 26.8 cm | −1.6 cm | 5.6 cm |
| tcn | 0.750 | 0.028 | 2.0 | 30.1 cm | 1.8 cm | 3.2 cm |
| tcn+1 | 0.944 | 0.333 | 3.9 | 40.4 cm | 12.0 cm | 13.6 cm |
| tcn+2 | **1.000** | 0.417 | 5.3 | 42.0 cm | 13.7 cm | 16.5 cm |

Repair never regressed a scene in any paired comparison. `tcn → tcn+2` fixes 9 collisions and
breaks 0 (McNemar exact *p* = 0.0039) and fixes 14 margin failures breaking 0 (*p* = 0.0001).
One-shot collision-free degrades monotonically with beam count — 1.00, 0.78, 0.67, 0.56 at 3,
4, 5, 6 beams — and two repairs restore 1.000 at every count.

**Two results cut against the system and are kept.**

The Phase 3 surrogate is badly optimistic out of distribution. Planning against the fitted
gain for an 0.18 m margin, the QP teacher and the TCN each achieve it on 1 of 36 scenes,
landing at 5.6 cm and 3.2 cm of actual mean headroom. The margin was swept on single-beam
scenes and does not transfer.

Repair buys safety with crouch, not efficiency. The heuristic reaches 18.5 cm of clearance
from 36.7 cm of peak dip; the repaired TCN reaches 16.5 cm from 42.0 cm. The heuristic ducks
broader and shallower, which converts better through a first-order body. **Phase 3's
halved-excess-crouch win does not survive out of distribution.** Repair corrects depth where
the schedule is already shaped; it does not reshape a schedule whose shape was wrong.

12 of the 21 `tcn+2` runs that end collision-free but short of target are saturated at
`DIP_MAX`, so their residual is a limit of the frozen prior's command range, not a repair
failure. The loop reports those as `accepted_margin`, never as success.

## 13. Experiment E — route selection that prices the body

`J_body(r) = ∫(u² + 6(du/ds)² + 25(d²u/ds²)²) ds`, `J(r) = w_L·L + w_B·J_body − w_C·C_min`,
with preferences as weight configurations rather than hard-coded route rules. Regret is
measured on a common scale: every selector's chosen route is rescored with the QP schedule
*for that route*, so a bad number means a bad route choice, not a worse schedule.

| selector | agreement with oracle | mean regret | median | max |
| --- | --- | --- | --- | --- |
| oracle_qp | 1.000 | 0.00 | 0.00 | 0.00 |
| heuristic (length) | 0.139 | 33.71 | 13.74 | 223.10 |
| tcn_body | 0.611 | 2.70 | 0.00 | 25.84 |
| tcn_verify | 0.611 | 2.70 | 0.00 | 25.84 |

Route length agrees with the body-aware reference on 5 of 36 scenes. Its failure mode repeats
across the sweep: it takes a route 50 cm shorter that requires a 42 cm sustained crouch over
one that requires no duck at all.

| k routes | TCN batched | QP sequential | speedup |
| --- | --- | --- | --- |
| 8 | 60.5 ms | 2.02 s | 33× |
| 32 | 60.7 ms | 8.93 s | 147× |
| 128 | 61.6 ms | 40.0 s | 649× |

The TCN's cost is flat in *k*; the QP pays ~312 ms of SLSQP per candidate. A learned body cost
is what makes scoring a large candidate set affordable at all.

`tcn_verify` reached 1.000 collision-free and 0.694 margin satisfaction at 3.7 ARDY calls per
scene, repairing on 53%. **It never fell back to a second route** — its first choice was always
salvageable within two repairs on this scene family. The fallback path is implemented and
unit-tested but this sweep did not exercise it, so its agreement and regret are identical to
`tcn_body` by construction.

## 14. Experiment F — rate bounds and a dimensioned objective

`r_down` and `r_up` were measured from the prior rather than assumed: for each of 483 cached
clips, take the MuJoCo top-of-body height per frame, invert it through the fitted gain, and
take the 95th percentile of each sign of its time derivative.

**r_down = 1.293, r_up = 1.363 command units per second** over ~60k frame transitions. They
are symmetric — which independently confirms, at four orders of magnitude more data, the
Phase 3 claim that was withdrawn: recovery is not slower than descent.

**The bounds never bind.** Over 42 solves the worst step the smoothness terms produce is 0.112
against a cap of ~0.32. `default` and `default+rates` are bit-identical. Implemented,
measured, reported as changing nothing.

The dimensioned objective `Σ[αq² + β((dq)/dt)²]dt` has no jerk term and so no `1/dt³` factor
to swamp everything — but the Phase 3 weights still do not transfer, since its rate/effort
ratio is `β/(αdt²)`. Recalibrating β to 0.375 (matching the per-sample ratio at the 0.9 m/s
reference speed):

| objective | boundary by speed (0.6 / 0.9 / 1.2 m/s) | CV in distance | CV in time |
| --- | --- | --- | --- |
| per-sample (Phase 3) | 3.0 3.0 3.4 m = 5.00 3.33 2.83 s | **0.060** | 0.249 |
| dimensioned β=0.375 | 2.4 3.4 4.2 m = 4.00 3.78 3.50 s | 0.221 | **0.054** |

Each objective is ~6% variable in its own natural unit and ~25% in the other. **Experiment B
is not overturned** — it accurately measured the teacher it tested. What changes is the
interpretation: the distance-fixed boundary was a consequence of summing raw differences over
a distance grid, not evidence that the body's cost is distance-based.

## 15. Experiment G — more coverage does not fix multi-beam failure

Trained the same 14,257-parameter TCN on a dataset covering 0–3 beams with 4, 5 and 6 held out
entirely, then reran Experiment D with it.

| method | m018 (0–2 beams) | | hard (0–3 beams) | |
| --- | --- | --- | --- | --- |
| | coll-free | meets .18 | coll-free | meets .18 |
| tcn | 0.750 | 0.028 | 0.722 | 0.028 |
| tcn+1 | 0.944 | 0.333 | 0.972 | 0.222 |
| tcn+2 | 1.000 | 0.417 | 1.000 | 0.333 |

Paired McNemar finds no significant difference on any method or metric; the largest
discordance is 4 vs 0 (*p* = 0.125). The reason is in the same table: **the QP teacher itself
satisfies the target on 1 of 36 of these scenes.** An imitation model cannot beat its teacher,
so covering more of the teacher's outputs cannot lift a ceiling the teacher sets. Held-out
accuracy does improve as expected (test MAE 18.4 / 37.3 / 50.5 mm at 4 / 5 / 6 beams against a
112 mm demand-only control), so the model is learning the teacher well. The teacher is what is
wrong out of distribution.

This is the argument for the whole phase. Repair does not imitate anything — it corrects from
measured clearance, which is why it reaches 1.000 where more data does not.

## 16. Demo V3.1

Default public layer is **AUTO**: propose with the Phase-3 TCN, generate, measure, repair,
regenerate, reverify. The three one-shot layers moved behind an Advanced disclosure and now
explicitly state that they make no margin claim, because they never checked.

A scene is labelled *repaired* only when the accepted clip's schedule hash equals the last
repair's output. Cache keys carry the schedule hash and the repair iteration, so a pre-repair
clip cannot be served for a post-repair request even when route, seed, model and scene are
identical.

Deep links, both on the final m018 model:

- **genuine repair** — `?height=1.05&width=2.25&n_beams=5&gap=2.5&preference=shortest&body_layer=auto&auto=1`
  TCN proposal collides at −7.4 mm; one repair reaches +182.1 mm and clears the 180 mm target.
- **no repair needed** — `?height=0.95&width=2.25&n_beams=3&gap=1.5&preference=shortest&body_layer=auto&auto=1`
  verified at +207.2 mm on the first attempt.

## 17. Phase 4 defect ledger

| # | defect | how it surfaced | resolution |
| --- | --- | --- | --- |
| 15 | clearance trace conflated overhead with lateral | a 3-beam scene read 76 mm total / 341 mm overhead | split contacts by normal direction; margin judged on overhead |
| 16 | centred secant undershoots the deficit by up to 27% | compared against the exact inverse of `g` | forward secant over the direction the repair moves |
| 17 | `mean_ardy_calls` measured cache state and included an unused path reference | tcn+2 reported *fewer* calls than tcn+1 | remove the unused generation; report one candidate call per attempt, preserve legacy count, and separate cache hits |
| 18 | narrow forbidden bands generated no distinct routes | A* routed around them and returned the same path | force routes through a slot by forbidding its complement |
| 19 | dataset RNG seeded from `hash(split)` | Python salts str hashes per process | sha256-derived seed, pinned by a test |

Defect 19 has a provenance consequence worth stating plainly: the committed m018 dataset
**cannot be regenerated byte-for-byte from source** — only verified against its recorded
content hash, which is what the provenance chain actually checks. Builds from here are
reproducible.

## 18. What Phase 4 changes about the conclusion

The measurement phase established that ARDY exposes roughly one or two executable body
strategies per scene. Phase 3 showed a convex teacher could schedule one of them well *inside
its fitted distribution*. Phase 4 shows that the fitted distribution is where the whole
approach lives or dies: out of it, the surrogate is optimistic by enough to collide, the
learned model faithfully inherits that optimism, and more training data cannot help because
the teacher is the ceiling.

What does work is measuring. Two bounded corrections driven by actual geometry take a system
from 0.750 to 1.000 collision-free on scenes it was never trained for, with zero regressions,
at 2.7× the ARDY calls. That is a smaller and less elegant claim than "the learned planner
generalises", and it is the one the evidence supports.
