# Scene2Motion-G1 — Research Report

**What was asked:** turn a frozen humanoid motion prior into a scene-conditioned *whole-body*
motion planner for the Unitree G1 — scene geometry plus start and goal in, whole-body motion out
that reaches the goal, avoids obstacles with the whole body, and is physically executable.

**What was found:** the frozen prior exposes a 43-dimensional constraint interface whose
*reliably commandable* subset, on this robot, is **one axis — duck depth**. Everything else is
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
  →  ~1   dynamically executable     (duck; lift is 0/8 under the tracker)
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
| **lift** | **0.000** | **28.31** | — |
| **duck + lift** | **0.000** | 24.64 | — |

Duck degrades gracefully and monotonically with depth — that is what an executable adaptation
looks like. Lift is zero of eight, twice.

**The strongest single result in the project** is that two methods sharing no code, no
assumptions and no data agree: the conformal kinematic envelope refused `low_obstacle` 0/20, and
a trained RL controller in physics independently reports lift 0/8. *A calibrated kinematic
refusal predicted a dynamic one.*

### 4.8 The collision-free certificate does not survive execution

The planner certifies against the *generated* motion. The controller realises a duck with 45.6 mm
of head/hand error. The margin actually required is `BODY_MARGIN + tracking error`:

| body | required | certified plans falling short |
|---|---|---|
| duck | 85.6 mm | **23.5 %** |
| duck-deep | 111.9 mm | **34.5 %** |

The requirement **grows with the adaptation**, so a deeper duck buys headroom and spends some of
it back on execution error — something nothing in the pipeline priced until now. This yields a
concrete, calibrated planning rule and is the first one in this project derived from *executed*
rather than generated motion.

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
dimensions, 36 spread requests — and delivers **one executable body axis**. The boundary between
what works and what does not is predictable from the *representation* (absolute-height versus
root-relative-lateral), from the *robot* (the thighs cap narrowing), and from the *action space*
(no yaw field), and **not** from anything one would call the model's competence.

**For the original goal.** Scene-conditioned whole-body traversal on G1 is achievable for
overhead clearance and nothing else. `low_obstacle` should be refused, and two independent
methods agree it should be.

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
| **EXP-014** | does the *shipped* gated step-over track? | both gated and ungated are airborne; the shipped renderer's lift also produces 0.05 mean contacts |
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

**(a) Finish the contact-consistent step-over ladder.** The amplitude sweep is a dose-response,
not a boundary: contact degrades smoothly rather than collapsing at a threshold, and no amplitude
has been *tracked* except 0.35 m. Track the ladder. **Kill condition:** if some amplitude both
preserves contact and tracks at ≥ 0.5 over 16 seeds, "step-over is unavailable" is withdrawn and
the executable repertoire is two axes, not one.

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

*The thing most likely to be wrong:* §4.7, that lift is unexecutable. It has already survived two
command mechanisms and a contact analysis, but every previous version of a capability-absent
claim in this project failed to survive the experiment that asked properly, and (a) and (b) are
that experiment.

*One methodological note for whoever continues.* A claim in the panel that fed this section — that
the scene suite labels only two adaptation types — was **false**: the 128-scene suite carries six
(`none` 44, `duck` 32, `step_over` 20, `narrow` 16, `detour` 12, `sidle` 4). It was checked before
being written down, which is the only reason it is not in this report as a finding.
