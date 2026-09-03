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
*reliably commandable* subset, on this robot, is **narrow — duck depth.** The "weak
small-amplitude step-over" earlier drafts of this report carried was an 8-seed, v1-sampler
reading; the completed 24-seed ladder under the intended sampler killed it (§19). Everything
else is either not requestable, not distinguishable from the prior's own sampling noise,
structurally capped by the robot's geometry, or not executable by a physics tracker. The
planning decision this supports is real but scalar: *duck, and by how much*. No learned
generator, set-valued proposer or reranker is justified.

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
keep the answer honest. Along the way it produced a catalogue of roughly **30** measurement
defects, most of which biased results *toward* the hypothesis under test (the count and its
scoping are reconciled in §5).

Status: 80 commits, 12 experiment families, ~34 000 ARDY generations, one end-to-end physics
validation. Research log: `docs/design.md` (39 sections, chronological — later sections correct
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
  →   1   dynamically executable      (duck; the gated step-over died on the v2 ladder — §19)
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

*(Update, 2026-08-31.)* The 0.375 was an 8-seed reading under the quarantined v1 sampler.
EXP-1C ran the full amplitude ladder at 24 seeds under noise stream v2 with the same per-side
gating and **no amplitude survived**: best success 0.25 at 0.05 m, falling to 0.0 by 0.16 m,
while matched controls held 0.92–1.00. **"Weakly executable" is withdrawn — see §19.**

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

**One number, and its scoping — because this report and the paper draft were quoting two.**
The ~24 census above covers the *audit phase* only — the research log as it stood when the
census was taken on 2026-08-29 (at the census's own bundling; ~32 if split). Since then the
ledger has grown by exactly six enumerable entries: the five Phase 4 defects in §17 (#15–19)
and the noise-stream v1 sampler defect (per-window latent replay,
`revalidation-2026-08-30.md`). 24 + 5 + 1 = **~30 project-wide**, which is the number the
frozen paper draft carries. The convention from here on: **~30** when counting the project,
**~24** only when explicitly scoped to the audit phase. Both remain "~" numbers — the census
bundling is a judgement call, not a measurement.

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
dimensions, 36 spread requests — and delivers **one strong executable body axis** (the weak
second axis this section once carried did not survive the v2 ladder — §19). The boundary between
what works and what does not is predictable from the *representation* (absolute-height versus
root-relative-lateral), from the *robot* (the thighs cap narrowing), and from the *action space*
(no yaw field), and **not** from anything one would call the model's competence.

**For the original goal.** Scene-conditioned whole-body traversal on G1 is achievable for
overhead clearance. The text route to the `low_obstacle` ladder is weaker than first reported:
the elicited lift is behaviourally real but lands ~0.14 m from where a placed obstacle needs it
and clears a 0-cm box in 8/8 seeds (§20), so the 0/20 refusal of that family stands until
EXP-016 shows the behaviour can be *placed*.

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
genuinely two. *(Done — EXP-1C, §19: 24 seeds × 6 amplitudes under v2, no amplitude survived.
The repertoire is one axis.)*

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
soften further. *(Resolved the other way — §19: the band is empty at 24 seeds under v2. The
"weakly executable" half of the claim is withdrawn, and the streak of capability-absent claims
failing on a better request ends at five.)*

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

*(Update, 2026-08-31.)* A spatial re-analysis of these same clips (EXP-015b, §20) found the
elicited lift **behaviourally real but spatially unplaced**: against a virtual obstacle at a
fixed route coordinate, whole-body box clearance is 0.0 m in 8/8 seeds at every probed height,
with the swing peak landing ~0.14 m from the obstacle. Text elicits the behaviour; nothing yet
places it.

---

# Phase 4 — Generate → Verify → Repair → Select

Phases 1–3 built an open loop: plan a route, write a duck schedule against a fitted model of
the body, hand it to the frozen prior, and hope. Phase 3's convex teacher halved excess crouch
at 100% collision-free on single-beam scenes, which was the first genuine improvement in the
project — and it was measured entirely inside the distribution the surrogate was fitted on.

The tables in this phase are **legacy noise-stream-v1 results**. The exact v2 replication has
now landed (§21): the generate→verify→repair architecture survives it, with one honest
exception (equal-budget resampling beats repair on margin for the heuristic proposer). The v1
numerical effect sizes below are kept for the record and are not current headline claims —
quote §21.

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

---

# Phase 5 — Noise stream v2: what survived replication

Everything in this part ran under the corrected per-sample sampler (`NOISE_STREAM_VERSION = 2`,
`CACHE_VERSION = 2`; scope and locked rerun order in
[`revalidation-2026-08-30.md`](revalidation-2026-08-30.md)) except where a **v1 quarantine
banner** says otherwise. Two of the four rerun items on the locked list are done and both
changed a conclusion; the third landed with an honest wrinkle; the fourth (the audit-cell
re-check of the 6× counting result) has not run.

## 19. EXP-1C: the step-over ladder was finished, and the claim is dead

§4.7 and §8.1(a) left the position-channel step-over "weakly executable" — 0.375 tracked
success over 8 seeds at one amplitude, under the v1 sampler — with the amplitude boundary
unlocated. EXP-1C (`experiments/exp001c_step_over.py`, `outputs/exp1c_stepover/`) located it:
**24 seeds × 6 lifts** (0.05–0.28 m), per-seed `planner._limb_targets` gating on each seed's
*own* control clip, matched controls at every rung, noise stream v2, 735.5 s.

| lift request | overshoot | contacts (lift / ctrl) | bilateral flight | SONIC success (lift / ctrl) |
|---|---|---|---|---|
| 0.05 m | 2.32× | 0.35 / 3.55 | 0.906 | **0.250** / 1.000 |
| 0.08 m | 1.98× | 0.25 / 3.03 | 0.948 | 0.083 / 1.000 |
| 0.12 m | 1.88× | 0.29 / 3.60 | 0.942 | 0.042 / 0.958 |
| 0.16 m | 1.92× | 0.20 / 2.18 | 0.958 | **0.000** / 0.917 |
| 0.20 m | 1.76× | 0.28 / 3.23 | 0.933 | 0.000 / 0.958 |
| 0.28 m | 1.68× | 0.12 / 3.18 | 0.975 | 0.000 / 1.000 |

The pre-committed kill condition — some amplitude with tracked success ≥ 0.5 **and** median
clearance-at-peak ≥ 0.05 m at 24 seeds — fired at no rung; the receipt's verdict field reads
**"claim KILLED on this ladder"** with an empty `survived_amplitudes` list. The failure mode is
the same at every amplitude: ARDY over-responds 1.68–2.32× and puts the reference into
bilateral flight on 91–98 % of gated frames, while the matched controls track at 0.92–1.00.
There is no amplitude band where the position channel buys a placed, contact-consistent
step-over — the 0.375 was a v1-sampler, 8-seed artefact of the one amplitude that was tried.

This supersedes `outputs/exp014_small` (single-seed gating defect;
`review-2026-08-30-codex-changeset.md`) and it is the sixth time a step-over claim changed when
the experiment asked properly — this time in the *unfavourable* direction, which is worth
noting: the elicitation principle of §35 (design.md) cuts both ways.

## 20. EXP-015b: the text-elicited step-over is behaviourally real and spatially useless

> **v1 quarantine.** This re-analysis scores the *saved EXP-015 clips*, which were generated
> under the legacy v1 sampler. It is diagnostic, not confirmatory
> (`receipt.json: status = post_hoc_exploratory`); the held-out v2 test is EXP-016, authored
> (`experiments/exp016_semantic_geometric_stepover.py`) but not yet run.

§4.9 reported that seven words of prompt raise the swing foot +57 mm while preserving contact.
EXP-015b (`outputs/exp015b_spatial_reanalysis/`) asks the question EXP-015's design did not: did
that lift happen *where an obstacle would need it*? A virtual corridor-spanning box at a fixed
route coordinate (x = 3.8 m, 0.2 m deep), whole-body clearance from exact collision geometry:

| arm | box clear-rate @ 0.05/0.08/0.12 m | \|phase error\| | swing foot at crossing |
|---|---|---|---|
| A walk text | 0.0 / 0.0 / 0.0 | 0.080 m | 0.024 m |
| B step-over text | **0.0 / 0.0 / 0.0** | **0.144 m** | 0.055 m |
| C walk + position lift | 0.0 / 0.0 / 0.0 | 0.140 m | 0.352 m |

**Max box height cleared by the whole body: 0.000 m, in 8/8 seeds, in every arm.** Arm B's lift
is real (+31 mm of swing foot at the crossing over arm A, paired) but its peak lands ~0.14 m
from the obstacle, and 0.14 m of phase error at 0.9 m/s is the difference between stepping over
a box and stepping on it. So the §4.9 result decomposes cleanly: **text supplies the behaviour,
nothing supplies the placement.** That is precisely the semantic × geometric composition
question, and it is what EXP-016 is built to answer.

## 21. Table 2 landed: proposer × feedback × equal-budget resampling, 8 seeds, v2

The §12 table was one seed under the v1 sampler. Its exact replication — plus the control the
field usually omits — is now committed: `outputs/phase4e_architecture_v2_s8/experiment.json`,
36 OOD scenes (3–6 beams × heights 0.95/1.05/1.20 × gaps 1.5/2.5/3.5) × 8 paired seeds
(100–107) × 15 arms = **4 320 rows**, 2 241.9 s, provenance frozen per row (commit `d54565b`,
checkpoint and dataset hashes, `noise_stream_version: 2`). The resampling arms regenerate with
the proposal held byte-identical at the same generation budget, so "feedback helps" cannot be
confused with "more samples help". Cells: collision-free / meets-0.18 m / mean generations:

| proposer | one-shot | +1 repair | +2 repairs | best-of-2 | best-of-3 |
|---|---|---|---|---|---|
| heuristic | .983 / .535 / 1.0 | 1.000 / .622 / 1.5 | 1.000 / .646 / 1.8 | .993 / .656 / 1.5 | 1.000 / **.726** / 1.8 |
| qp | .872 / .007 / 1.0 | 1.000 / .326 / 2.0 | 1.000 / .417 / 2.7 | .944 / .017 / 2.0 | .972 / .031 / 3.0 |
| tcn | .729 / .003 / 1.0 | .913 / .278 / 2.0 | .993 / .375 / 2.7 | .764 / .007 / 2.0 | .774 / .014 / 3.0 |

Three things, two expected and one not:

* **The v1 architecture conclusion replicates.** Repair dominates resampling wherever the
  proposal itself is wrong: for the TCN, +2 repairs beats equal-budget best-of-3 .993 vs .774
  collision-free (paired discordance **63 vs 0** rows), and .375 vs .014 on margin (104 vs 0);
  for the QP, .417 vs .031 on margin (112 vs 1). Resampling an optimistic schedule redraws the
  noise around the same mistake.
* **The honest wrinkle: for the heuristic proposer, resampling beats repair on margin.**
  Equal-budget best-of-3 reaches **0.726** margin satisfaction against repair's **0.646**
  (row-paired discordance 41 vs 18 in resampling's favour; both reach 1.000 collision-free).
  A proposal that is already *shaped* right and generously deep has nothing for repair to fix
  that seed luck cannot also fix — repair earns its cost exactly where the proposer is the
  weak link, and the report should not pretend otherwise.
* **tcn+2 is 0.993, not the v1 table's 1.000** (286/288; Wilson 95 % lower bound at n = 288 is
  0.987 for a true 1.000). "Two repairs restore 1.000 at every beam count" was a one-seed
  statement; under 8 seeds it holds for heuristic+1/+2 and qp+1/+2 but not for the TCN.

Seeds are nested in scenes, so the scene (n = 36) is the inference unit for any confirmatory
claim; the row-level discordances above are stated as ledger counts, not tests.

## 22. Corpus pilot v2: the loop as a data engine, measured

`outputs/corpus_pilot_v2/` — 300 randomized scenes through the full accept loop (heuristic
proposer, ≤ 2 repairs, seed 7), **301.9 s wall-clock** on one consumer GPU: **192 accepted,
76 accepted-margin, 6 rejected, 26 refused-with-reason**. That is 268 kinematically verified
traversal records — with per-record manifest, schedule hashes and v2 cache keys — in five
minutes, and 0 records silently dropped. The throughput tiers the paper draft commits to
(generated / kinematically scored / accepted / SONIC-executed) are measurable from this
manifest for the first three. The fourth is now measured by EXP-1B (§25), with a negative
result that must not be conflated with kinematic acceptance.

## 23. Redirection, 2026-08-31: the audit becomes infrastructure; the method is RAMP

The sixth guidance document (`guidance-2026-08-31-ramp-refocus.md`, review verbatim in
`guidance-2026-08-31-ramp-refocus-full.md`) redirects the project. The audit-shaped paper
draft (`paper-draft-v0.md`) is **frozen as the baseline technical report**, and everything
this report measures — Table 2, the τ(d) execution gate, the cross-prior audit — is
reclassified as **baseline, regression test and evaluation infrastructure** for a method
contribution: **Scene2Motion-RAMP** (Response-Adaptive Motion Programs) — event-aligned,
prior-compatible motion programs, coherent adaptation residuals transported onto the current
seed's own phase-matched nominal, and response-conditioned repair that modifies the program
from the observed generation failure. The bar the guidance sets is explicit: the next result
that matters is whether coherent residual + response adaptation turns placed step-over,
lateral squeeze and composed traversal from failures (§19, §20) into stable executable
successes — not another defect count or overstatement factor. This report remains the ledger
of record for the baseline numbers; RAMP work is documented from its own drafts forward.

## 24. Kimodo-G1 replication: numbers real, artifacts lost — a provenance gap

The frozen paper draft (`paper-draft-v0.md:228–237`) quotes a reduced audit on Kimodo-G1-RP-v1:
naive counts 8.0/9.0/9.0 vs calibrated 2.0 — a 4.5× overstatement — from 84 clips in 173 s.
**No `outputs/` directory backs those numbers.** The provenance hunt (2026-08-31) found the
run: it executed on this machine on 2026-08-31 ~04:40 UTC from a Claude session working in
`/home/linjiw/ardy`, and wrote everything to that session's **temporary scratchpad**
(`/tmp/claude-1000/-home-linjiw-ardy/f4440d67-…/scratchpad/kimodo/`, including
`audit_out/receipt.json` with `n_clips: 84`, `wall_clock_s: 172.7`, counts matching the draft
digit-for-digit). That directory **no longer exists** — `/tmp` scratchpads are session-scoped
and were cleaned. The numbers survive only in the session transcript
(`~/.claude/projects/-home-linjiw-ardy/f4440d67-ed27-4331-be07-dc169754a80c.jsonl`), which
captured the receipt fields and per-program console output verbatim, and the generating
scripts (`kimodo_runner.py`, `kimodo_reduced_audit.py`) are recoverable from the same
session's subagent transcript. So the result is *attested but not re-derivable*: every other
number in this report traces to a committed ledger; this one traces to a chat log. Until the
audit is re-run with its receipt landed under `outputs/`, the Kimodo row does not meet this
project's own evidence bar and must be labelled transcript-sourced wherever it is quoted.
Full trail and rerun checklist:
[`kimodo-provenance-2026-08-31.md`](kimodo-provenance-2026-08-31.md).

## 25. EXP-1B completed: the execution gate fails before clearance calibration

The recovered, resumable achieved-state campaign is complete and curated at
`outputs/exp1b_execution_clearance_v2/`: **859** kinematically accepted Phase-4E selections,
36 scenes, 526 unique clips, 24 SONIC launches, noise-stream/cache v2. The recovery combines
180 schema-v1 rows (the post-reset teleport frame dropped at load) and 679 schema-v2 rows (the
frame excluded at write); hashes and both code commits are pinned in `receipt.json`.

This is a strong negative execution result. **554/859** rollouts terminated or missed the
progress criterion; **200/859** reached the first obstacle; **0/859** passed the last obstacle;
and **0/859** were executed traversal successes. The same endpoint is zero for heuristic
(0/283), heuristic+1 (0/288), and heuristic+2 (0/288). Deep requested crouches preserve local
posture much better than global root motion: the robot generally stalls before or among the
beams while the reference continues down the route. The frame-level diagnosis and exclusion
of an environment-origin bug are in
[`exp1b-frame-diagnosis-2026-08-31.md`](exp1b-frame-diagnosis-2026-08-31.md).

Consequently, the predeclared execution-calibrated acceptance rule is **not fit**:
`gate.json` reports `insufficient_data`, because no rollout passed the last obstacle and hence
no row provides a valid, uncensored clearance-loss observation. This is not an instrumentation
failure. It is evidence that execution modeling must be hierarchical: first predict
reach/survival and forward-progress retention, then model clearance loss conditional on reaching
the interaction. It also prevents the paper from using kinematic acceptance as a proxy for
physical traversal success.

## 26. RAMP step-event foundation and exp017 harness landed, 2026-08-31; E1 remains unrun

The first method implementation after the §23 redirection is now present under
`scene2motion/ramp/`, deliberately separate from the frozen 43-D audit representation in
`scene2motion/program.py`. It is a strict **v1 step-event foundation**, not evidence that RAMP
already solves step-over:

* `step_phase.py` derives takeoff, physical-foot apex, landing, swing side, and contralateral
  stance evidence from exact qpos foot envelopes and locked height/speed support masks. It
  rejects truncated cycles, bilateral flight, excessive penetration, insufficient lift, and
  inadequate stance support instead of synthesizing a phase label.
* `phase.py` aligns complete source phase windows and inverts a held-out target's measured
  phase trace onto the packet grid. Both source and target receipts retain the measured
  support fractions, locked threshold/window, named stance side, and evidence source.
* `packet.py` extracts a causally paired absolute/residual packet on identical adapted phase
  queries and renders either arm onto the same held-out nominal gait. The residual is formed
  in hierarchy-local rotations relative to an independently aligned neutral source; source
  route heading is explicit, and the generated body-heading feature is not misused as route
  yaw. The two arms are required to have identical adapted event, phase knots, taper, route
  frame, skeleton, generator/checkpoint/seed/noise-stream provenance, and exact ARDY
  channel/frame/joint support. Rendering requires a target-phase receipt and uses the held-out
  nominal root height rather than a constant surrogate.

The deliberately narrow scope matters. V1 implements the physical contact contract for one
unilateral step event; it does **not** yet implement duck/double-support, squeeze/turn,
multi-event composition, or a body-heading conditioning program. The ARDY heading feature is
left free in the paired step pilot so route canonicalization does not erase a donor's body-yaw
residual.

The CPU-testable exp017 orchestration now exists at
`experiments/exp017_ramp_residual_stepover.py`, with fail-closed tests in
`tests/test_exp017.py`. A completed candidate-pool run spends exactly `2D + K + 2NP`
frozen-prior samples: matched adapted/neutral donor sources, a fully archived pool of `K`
immutable nominals, and paired absolute/residual calls for the first `N` commonly eligible
seeds at fixed seed-independent scenes. It freezes and hashes every program and planned arm
before final sampling; requires identical prompt, seed, CFG, diffusion settings, target-phase
receipt and ARDY support for each pair; archives source/nominal qpos plus replayable nominal
substrates; and reports eligibility separately from descriptive paired effects. It does not
attach a confidence interval to crossed placements of one box geometry.
It refuses a dirty worktree, non-empty output directory, non-v2 sampler, malformed calibration
identity/provenance, stalled source/nominal clips, incomplete physical cycles, a protocol-hash
mismatch, or unequal channel support. Canonical experiment/program/output identities bind the
checkpoint, code, threshold receipt, prompts, settings, clips, packet/program/support hashes,
and outputs without creating a cyclic manifest hash.
The final receipt separately anchors `rows.jsonl`, the immutable attempt plan, the durable
per-arm attempt ledger, the output-identity set, and both logical and archive hashes for
`qpos.npz`. Launched and returned sample counts are separate at every generation stage; a
thrown call produces a returned lower bound and conservative charge rather than an invented
exact spend.

This remains **implementation evidence, not a capability result**. Three real ARDY attempts
have produced source/eligibility ledgers but no final arm, and no SONIC comparison exists. The
next dependency is the locked `D=4, K=8, N=2, P=1` exploratory pool in
[`ramp-e1-protocol.md`](ramp-e1-protocol.md); equal-budget SONIC replay is still required
before an execution claim. The local response optimizer, RepairNet, hierarchical execution
model, and RAMP-aware route cost remain unimplemented and must not be described as landed.

## 27. First exp017 GPU preflight failed before arm generation; diagnostics hardened

The committed `D=1, N=1, P=1` preflight ran under noise-stream v2 with donor/evaluation seeds
2600/2800 and the locked calibration receipt. It failed closed in source discovery after
exactly two ARDY samples. Both the adapted step prompt and paired neutral walk for seed 2600
had empty physical-cycle lists; the enumerator reported each foot as one unsupported run from
frame 0 through 199, so no phase-alignable source pair existed. The run generated zero nominal
and zero absolute/residual final samples. This is an eligibility failure, **not** evidence
against either packet representation.

The failure ledger is `outputs/exp017_ramp_preflight_d1_n1_p1/receipt.json` plus
`donor_candidates.jsonl`. It exposed a diagnostic artifact gap: rejected donor qpos were not
archived, so the first ledger cannot distinguish whether height, speed, or both caused the
support failure. The harness now writes all candidate adapted/neutral qpos to
`donor_qpos.npz` before any progress/phase gate, binds logical and archive hashes into donor
rows and success/failure receipts, and records support/clearance/speed summaries that are
explicitly excluded from selection. Thresholds were unchanged; that historical run had
`K=N=1`, so its then-written `2D + N + 2NP` accounting is numerically identical to the current
`2D + K + 2NP` contract.
Future failed receipts also carry a canonical pre-source run identity binding code,
checkpoint, threshold receipt, prompts/settings, budget, splits, scenes, and route; the first
already-completed v1 failure predates that hardening and remains labeled accordingly.

Before seeing any additional donor outcome, the next diagnostic is locked as
`D=4, N=1, P=1` with sequential donor seeds 2600–2603, evaluation seed 2800, obstacle
`x=3.6 m`, and the same calibration/gates. It spends eight samples if source discovery fails
or eleven if it reaches the paired arms. A successful schema preflight would still not be a
capability result or license an execution claim.

## 28. Sequential donor discovery passed; nominal target assignment failed before arms

The predeclared `D=4, N=1, P=1` attempt ran from clean commit `7b488f3` and spent nine ARDY
samples: eight matched donor-source samples and nominal seed 2800, with zero paired-arm calls.
Seeds 2601, 2602, and 2603 admitted physical adapted/neutral cycle pairs. The deterministic
selector chose seed 2603 and successfully constructed the absolute/residual packet pair from
a left-swing adapted event with 0.232 m relative lift and full contralateral support. Source
eligibility therefore exists under the locked calibration; the prior D=1 failure was a
single-seed donor issue, not a universal absence of source steps.

The run then stopped because no target cycle for nominal seed 2800 admitted the locked bounded
one-to-one assignment at `x=3.6 m`. No manifest, absolute output, residual output, or packet
comparison exists. The durable ledger is
`outputs/exp017_ramp_preflight_d4_n1_p1/`, including all eight donor qpos, source-cycle
receipts, selected packet arrays, full pre-source provenance, and partial-spend receipt.

Nominal qpos were not yet persisted before that exception, so the binding sub-gate—phase
alignment, ±8-frame placement shift, or packet-window boundary—was not identifiable from this
ledger. The harness now archives every nominal before gates, preserves later batch members,
records per-cycle phase/shift/window checks, and anchors `nominal_rows.jsonl` logically and by
file hash in success/failure receipts and the pre-final manifest. The exact `D=4, N=1, P=1`
design is locked for one diagnostic replay; no gate or sample selection changes.

## 29. Exact replay identifies fixed-placement addressability as the binding gate

The exact-design replay ran from clean commit `dad93dd` with the same `D=4, N=1, P=1`, donor
and evaluation seeds, scene at `x=3.6 m`, prompts, calibration, sampler, and assignment gates.
It spent nine ARDY samples and again produced no final arms. The selected adapted/neutral clip
hashes, paired source qpos, left-swing donor event (0.232 m relative lift), and packet payload
reproduced the preceding retry exactly; only code-revision-bearing provenance changed.

The newly complete nominal ledger makes the failure identifiable. Seed 2800 reached a final
route-progress ratio of 1.013 and yielded one complete left-swing cycle at frames 31/34/38
with 0.047 m relative lift and full right-foot stance. Its five source-to-target phase errors
were all zero, and packet frames 98--100 fit inside the 200-frame clip. Fixed obstacle placement
required root frame 99, however, so the event center would have to move from apex frame 34 by
`+65` frames. This exceeds the locked `+/-8` local-shift bound. The durable evidence is
`outputs/exp017_ramp_preflight_d4_n1_p1_v2/`; its receipt anchors all donor and nominal qpos,
the finalized nominal row, experiment identity, and partial spend.

This is a refusal at the pre-arm scene-addressability gate, not evidence about absolute or
residual packet performance. Moving the obstacle or widening the bound after observing the
failure would change the scientific question. Instead, the next exploratory pilot was
redesigned—after this eligibility observation and before any arm outcome—as a frozen ordered
nominal pool with `D=4, K=8, N=2, P=1`. All seeds 2800--2807 are screened under unchanged
common gates, all screening calls are charged, the first two eligible seeds are selected
without ranking or replacement, and pool exhaustion stops the run. Its successful budget is
`2D + K + 2NP = 20`; it reports `E/K` eligibility separately from the paired conditional arm
effect. This design does not support an unconditional seed-robustness claim, and low pool
coverage is negative evidence rather than a reason to extend the pool.

## 30. Candidate-pool harness frozen before the first final-arm outcome

The exploratory `D=4, K=8, N=2, P=1` harness was implemented and independently reviewed
before any absolute/residual arm was generated. It archives and classifies every candidate in
the ordered 2800--2807 pool under the unchanged progress, physical phase/contact, fixed-scene
assignment, packet-window, and `+/-8`-frame shift gates. Only if `E >= N` does it select the
first two eligible seeds; otherwise it selects no partial cohort and stops after the complete
16-sample source-plus-pool spend. Seed 2800 and its `+65`-frame refusal remain in the ledger.

The successful budget is exactly `2D + K + 2NP = 20`. `nominal_selection.json` hash-binds
`E/K`, the first-eligible prefix, selected coverage `N/E`, attrition reasons, and the planned
`NP` denominator per arm. Before final sampling, `attempt_plan.json` freezes all `2NP` arm
identities into the manifest. `attempts.jsonl`, successful rows, and qpos are then persisted
incrementally, so generation, conversion, or scoring failures stay in the denominator and
cannot trigger candidate replacement. Every generation stage records launched versus returned
samples; on an exception, exact spend is null and the receipt gives a returned lower bound plus
a conservative launched charge. All `K` nominal `global_rot_mats` and `smooth_root_pos` arrays
are archived for later response-optimizer replay but do not enter eligibility or ranking.

The repository-wide CPU suite passed 251 tests, and independent review found no remaining
P0/P1 issues. This licenses only the locked exploratory GPU launch at
`outputs/exp017_ramp_pool_d4_k8_n2_p1`; it does not license a representation, execution, or
planner claim before that ledger is inspected.

## 31. Frozen nominal pool exhausts at 1/8 eligibility; no packet arm is generated

The locked `D=4, K=8, N=2, P=1` run executed from clean commit `3e4ba7f` and stopped exactly
at its preregistered pool-exhaustion rule. It launched and returned eight donor-source plus
eight nominal samples (`16` exact total), made zero paired-arm calls, selected no partial
cohort, and wrote no final program or outcome row. All logical, file, qpos-content, and nominal-
substrate hashes were independently rechecked against the receipt.

Only seed 2805 was eligible (`E/K=1/8`), using a left-swing apex at frame 95 and a required
`+8`-frame shift exactly on the locked boundary. Seeds 2800, 2804, and 2806 had physical target
cycles but failed bounded fixed-placement assignment; their nearest accepted-side shifts were
`+65`, `+28`, and `-15` frames. Seeds 2801, 2802, 2803, and 2807 had no left-swing cycle under
the locked target contract. The frozen attrition ledger therefore reads four `phase_cycle`,
three `target_assignment`, and one `eligible`. The durable artifact is
`outputs/exp017_ramp_pool_d4_k8_n2_p1/`.

This is negative evidence about the v1 event-placement substrate, not an absolute-versus-
residual packet outcome. Offline same-qpos diagnostics found 25 phase-alignable right donor
pairs but no additional right-side target satisfying source phase, window, and `+/-8`
placement; bilateral support would still give `E/K=1/8`. A target-only lift sensitivity from
0 to 4 cm exposed additional ordinary-walk cycles but also left `E/K=1/8`; fixed placement,
not render-window or lift threshold, remained binding. These are retrospective mechanism
checks, not new arm results.

The pool will not be extended and its gate will not be relaxed. The method implication is that
RAMP v1 still waits for a nominal gait cycle to occur near fixed scene progress. The next
representation must instead couple route timing/root-speed scheduling to the selected gait
phase while keeping the obstacle fixed, then use new disjoint calibration/donor/evaluation
seeds. Exp018 response optimization remains blocked until that redesign produces at least one
placed residual response.

## 32. Route-phase calibration lands on its third preregistration; warp identities frozen

The route-timing redesign demanded by §31 is now calibrated and held-out validated. Three
preregistered campaigns ran on 2026-09-01, each 72 fresh paired samples (16+8 seeds × 3
speeds, nine batches of eight, ~24 s wall clock each), each fail-closed, all three committed
with full evidence:

- **v1 (seeds 3200s/3300s) refused at calibration.** Two calibration seeds produced no
  physically-complete unilateral swing at any speed, so the "every seed supplies background"
  requirement was unsatisfiable. Diagnosis (`analyze_calibrate_ramp_route_phase_v1.py`):
  24 % of clips have zero support-consistent complete cycles; seed-level washout is
  speed-invariant (the paired design shows it); a single fixed 3.6 m event admits a warp for
  only 25 % of valid cycles under the [0.6, 1.2] m/s envelope; endpoint-speed coverage rules
  were sized above the measured washout rate. Artifact:
  `outputs/calibrate_ramp_route_phase/`.
- **v2 (seeds 3400s/3500s) refused at held-out validation.** The five-point placement set
  and attrition-tolerant calibration worked (Pmin 0.042 m vs 0.041 informative from v1;
  15/16 seeds evidenced), and separation passed where evidenced (right Q25 0.0605 m vs
  floor 0.0503 m) — but per-seed reference-stratum warp feasibility measured 3/7 against
  the gated 5/7 (pooled 9/14 ≈ 0.64). The descriptive strata showed the same fresh seeds
  feasible at other speeds (union 6/8). Artifact: `outputs/calibrate_ramp_route_phase_v2/`.
- **v3 (seeds 3600s/3700s) passed every gate.** The speed stratum became an outcome-free
  program parameter — selected per seed/side by the frozen minimum-deformation key with
  reference preferred on ties, the same move the placement set made for event position —
  and the gates became rate-honest pooled quantities. Result: Pmin 0.042 m (third
  replication); caps 0.19 m/s² / 1.2 m/s³ / 0.17 m/s from exactly 10/16 contributing
  seeds; validation with zero washouts, zero background exceedances, 7/8 program-covered
  seeds, and pooled separation Q25 0.0459 m against the 0.042 m floor with balanced sides
  (6 left / 7 right, 13 signals). Artifact: `outputs/calibrate_ramp_route_phase_v3/`.

Two substrate properties measured by the refusals now bind the E1 pilot design: (a)
nominal-substrate attrition — ~6 % of seeds are all-speed washouts and ~20 % of clips are
unusable at any single speed, so the nominal pool budget must be sized for it; (b) the
selected strata skew heavily to fast (11/13 selected programs) because faster gaits put
more apexes inside the feasibility window — the pilot's nominal substrates will mostly walk
fast, and any speed-sensitivity of packet transport must be read against that skew, not
assumed away.

The passing receipt authorizes exactly one thing: the route-warped packet pilot at frozen
Pmin, caps, placement set, and per-seed (stratum, placement, program) triples, with pool
budget from the measured rates. It is not evidence that any packet representation improves
traversal. Protocols: `docs/ramp-route-phase-calibration-protocol{,-v2,-v3}.md`; refusal
records: `docs/ramp-route-phase-calibration-refusal-2026-09-01{,-v2}.md`.

## 33. Three placement levers measured; the packet's own constructibility is the binding limit

With the route-phase foundation calibrated (§32), the E1 packet comparison was attempted
on the two remaining placement levers. Both were preregistered, staged so a placement
failure could not be confounded with a packet effect, and both produced measured negatives
before any arm was generated. All evidence is committed.

**exp018 — route warping (`docs/ramp-e1-route-warped-protocol.md`).** Stage A regenerated
six calibrated-selected seeds under their own outcome-free route-progress warps. Only
**1/6** still placed a complete left swing at the obstacle inside the unchanged ±8-frame
gate, though all six tracked their routes (progress 0.996–1.003): the prior re-plans its
gait when root-path timing changes, and two seeds lost swing structure entirely. The
mechanism is a route-length coupling inherited from the calibration — the program
reparameterizes a fixed 7.2 m route while cycles are observed at 4.776/7.200/9.552 m, so
the "minimum-deformation" selection displaces conditioning by up to 2.4 m. Artifact:
`outputs/exp018_route_warped_stepover/`; note:
`docs/ramp-exp018-route-warp-negative-2026-09-01.md`.

**exp019 — gait-matched placement (`docs/ramp-e1-gait-matched-protocol.md`).** Inverting
the lever (obstacle at `route_progress(apex) + foot_offset` on the nominal's own route)
makes the required shift exactly zero and reaches **12/16** placement eligibility. But
only **5/16** seeds are *constructible*: the donor's step-over swing is 11 frames while
the pool's median walk swing is 8 and 97 % are shorter, so the packet's ±2-frame source
window compresses below one integer frame on one side of most target apexes (24 render
collapses), while longer targets fail phase alignment (15). Constructibility occupies a
narrow 8–9-frame band. Widening the source window is not an option: at half-window 3 and 4
**no donor pair in the bank is phase-alignable at all**. Artifacts:
`outputs/exp019_gait_matched_stepover{,_v2}/`; note:
`docs/ramp-exp019-constructibility-2026-09-01.md`.

**What this means for the method.** Across three placement designs the frozen prior has
now been measured to resist *scene-specified* event placement in every available way:
waiting for a gait event at a predeclared obstacle (1/8), warping route timing to move the
root to one (1/6), and — even when placement is conceded to the gait — transporting a
step-over packet onto the prior's own walk swings (5/16). The first two bound the
placement lever; the third bounds the *representation* itself, because a step-over donor's
swing is intrinsically longer than a walk swing. These rates belong in the paper next to
whatever the absolute-vs-residual contrast eventually shows: they say how often any
coherent packet can be applied at all, independent of whether the residual form is better
than the absolute one.

**Two latent defects were also found and fixed** by running the real path for the first
time: exp017's channel-usage allowlist named ConstraintSpec builder keys
(`root_2d`/`root_y_pos`/`global_joints_rots`) rather than the model's `slice_dict` names
(`root_pos`/`global_rot_data`), so it would have aborted *any* real arm run and survived
only because exp017 never reached rendering; and exp019's first attempt selected
placements with the phase-observability enumerator while the transport consumes
`step_phase` cycles, which is now prevented by an outcome-free constructibility probe that
runs the real assignment and both renders before a candidate is eligible.

## 34. The first completed E1 packet comparison ran (superseded reading; see §35)

The rate-honest rerun (K=32 fresh seeds 4000–4031, N=8) cleared its gate at **17/32**
constructible seeds and, for the first time in the E1 family, **generated and scored every
planned arm**: 8 nominal (free), 8 absolute, 8 residual, with the full 120-sample budget
spent exactly. Artifact: `outputs/exp019_gait_matched_stepover_v3/` (blocked only in the
summary stage by a `float(None)` on `lead_matches_donor_side`, a reporting defect; all 24
arm rows are complete and hash-anchored).

The result does **not** adjudicate representation, and the free nominal arm is what shows
why:

| metric (8 paired seeds) | nominal | absolute | residual |
|---|---:|---:|---:|
| whole-body box clearance (m) | 0.0000 | 0.0000 | 0.0000 |
| obstacle min clearance (m) | −0.0869 | −0.0916 | −0.0903 |
| obstacle collision-free | 0/8 | 0/8 | 0/8 |
| kinematic step success | 0/8 | 0/8 | 0/8 |

Every arm collides, including the unmodified nominal walk. Residual − absolute is noise on
every endpoint (box clearance exactly 0 for both; min clearance median +0.4 mm, 5/3 split).
Both packets make crossing-position error *worse* than nominal (+0.15 m).

**Correction (2026-09-01, see §35).** This section first read the all-zero nominal as
proof of an unwinnable scene. That was wrong. The box probe requires the whole body to
clear the box across the whole trajectory, and a walking swing foot sweeps through every
route position at low height, so **an ordinary walk scores zero at every x on its route** —
the nominal arm scoring zero is the correct baseline, not a defective placement. The
donor step-over scores 0.30 m on the same probe, so the endpoint discriminates properly.
The footfall-clearance rule added here is a mild scene-quality improvement, not the
validity fix it was described as.

The placement rule now also requires the expanded footprint to contain no support-phase
footfall of either foot, which is outcome-free and cheap to satisfy.

The free nominal arm remains permanent and was still what forced this section to be
re-read: it is the control that separates "the packet failed" from "the endpoint is
insensitive". Here it did the latter job.

## 35. E1a lands: the packet elicits the step-over reliably but places it 1–3 m away

The gait-matched pilot completed at K = 64 — `outputs/exp019_gait_matched_stepover_v7/`,
status `complete`, **216/216 samples exact**, 13/64 eligible, N = 8 evaluated with three
arms each. Full write-up: `docs/ramp-e1a-result-2026-09-01.md`.

**The endpoint discriminates.** The selected donor step-over clears a 0.3013 m box; its
own neutral WALK clip and 7 of 8 evaluated nominals clear 0.0000 m at every position on
their routes. A walking gait scores zero because the swing foot sweeps through every x at
low height — so a zero nominal is the expected baseline, which corrects §34's reading of
the same measurement as an unwinnable scene.

**At the commanded obstacle, all three arms score identically zero** (box clearance
0.0000, collision-free 0/8, step success 0/8). Residual − absolute is noise on every
endpoint; both packets slightly worsen min clearance versus nominal (−0.090 / −0.096 vs
−0.079 m) and break the swing-side match the nominal had (absolute 5/8 wrong, residual
3/8).

**But the packets are not inert.** Scanning each generated clip's whole route, **8/8**
absolute and **8/8** residual arms contain a real step-over-like lift (mean best clearance
0.070 / 0.061 m) against **1/8** for the nominal — and that lift sits a median **2.33 m**
(absolute) / **1.62 m** (residual) from where it was commanded.

So the coherent packet **elicits the behavior and fails to place it** — the same failure
mode exp015 found for text prompts (box clearance 0.0, random placement, 8/8), now
reproduced through the strongest structured channel we have: the target swing identified
from measured physical gait phase, the obstacle anchored to that swing's own apex on the
nominal's own route, a center shift of exactly zero, and both arms rendered with identical
channel support. Spatial placement is therefore not a text-conditioning artefact; it
survives phase-matched coherent packet transport.

Under exp017's preregistered decision rule the residual arm shows no clearance advantage
and degrades another gate, so the stated fallback applies: stop expanding the open-loop
packet path and pivot to correcting the realized response. The positive reading is that
this is exactly a response-space error — signed, directional, measurable from the
generated clip — which is the signal a response-conditioned repair policy consumes. E1's
negative motivates E2 directly rather than ending the line.

## 36. The packet is a style cue — and it is worse than the prompt alone

Four controls on top of §35, three of them replays of the archived 24 clips (arm clips
byte-identical, 16/16) and one an 8-sample matched control. Full note:
`docs/ramp-e1a-controls-2026-09-01.md`.

**Placement does not respond to the request.** Regressing realized lift position on
requested obstacle position gives slope −0.44 (absolute) and +0.02 (residual) where
perfect placement is 1.0. The signed error is not consistently signed (4 after / 4 before,
sd 2.43 m), so no open-loop lead exists. The two arms land within 0.25 m of each other in
5/8 seeds despite different payloads — the lift belongs to the seed, not the request. Any
clearance at the obstacle: 0/16, Wilson95 upper 0.19. The error is not stride-quantized
(mean |frac| 0.277 vs 0.25 uniform), so selection is not rescued by lattice structure.

**The command is never satisfied.** Compliance at the constrained (frame, joint) pairs —
1 met, 0 stayed at nominal, negative moved away — is **−0.26** (absolute) and **−0.44**
(residual) for rotations: the clips end further from the command than the nominal was.
Root height, the historically clean channel, is +0.20/+0.14 on a 2.3 cm request at the
tracking noise floor. Sweeping lags −40…+80 frames, **0/16 clips match the command better
than nominal at any lag**; best lag is median +1 and correlates *negatively* with placement
error (r = −0.48) where delayed execution predicts +1. The packet is not late — it is
unread.

**The world cannot be anchored by shifting the start.** ARDY is not translation-equivariant
along the route: offsetting by 0.5/1.3 m gives max |qpos − shifted| of 1.4–3.5, with the
start pinned near the origin and the endpoint moving. A route offset is absorbed as a
speed change — the same mechanism that made exp018's warp re-plan the gait.

**The control that matters (exp020, 8 samples).** Same seeds, strata, routes and endpoint
vector, STEP prompt with route conditioning only:

| arm | elicits | mean lift | max lift | clears obstacle |
|---|---:|---:|---:|---:|
| nominal (WALK) | 3/8 | 0.0019 m | 0.0050 m | 0/8 |
| **text only (STEP)** | **7/8** | **0.1213 m** | **0.2118 m** | **3/8** |
| STEP + absolute packet | 8/8 | 0.0721 m | 0.1377 m | 0/8 |
| STEP + residual packet | 8/8 | 0.0609 m | 0.1439 m | 0/8 |

Paired per seed, text-only beats the packets on amplitude by a median **+6.0 cm** and
**+6.8 cm** (6/8 seeds each), and clears a 5 cm box in 3/8 scenes where both packet arms
clear 0/8. **The coherent packet is not neutral but harmful**: it adds elicitation the
prompt already gives, halves the amplitude, and removes the prompt's only successes —
exactly what the negative compliance predicts, since its rotation targets sit off the
prior's own coordinated manifold.

This answers E2's first two planned steps without spending a campaign on them: placement
gain is zero, and a re-anchoring fixed-point loop presumes a command honored somewhere,
which the lag sweep rules out. What survives is measure-then-select. exp021 measures the
joint distribution of (lift position, lift height) over 64 text-conditioned clips to size
it. Two harness changes adopted: the box-height endpoint is reported **graded**
(0.03–0.30 m), and the tracker belongs in the endpoint before any 5–7 cm claim, since the
kinematic endpoint is ambiguous at that amplitude.

## 37. The prior emits the behavior once, early — which makes it placeable by staging

exp021 (`outputs/exp021_elicited_lift_distribution_v2/`, complete, 64/64 samples) measures
the joint distribution of (lift position, lift height) for text-conditioned clips on a
fixed route. Full note: `docs/ramp-exp021-addressable-region-2026-09-01.md`.

**Amplitude was never the ceiling.** Elicitation is 0.77; lift heights reach a **0.400 m**
maximum with q75 0.184 m, and 31/64 clips clear an 8 cm box somewhere. E1a's 0.06–0.07 m
was the packet suppressing the prior's own behavior, not the prior's limit.

**Timing is the constraint, and it is sharp.** Converted to clip frames, lifts have median
**frame 34** (1.36 s), q10–q90 21–54, with **80 % inside the first 50 of 200 frames**;
restricting to lifts ≥ 0.08 m does not move it (median 33). Only 3 of 49 lifts occur after
frame 60. **The prior expresses the prompt once, shortly after its context is established,
then returns to walking.** This single fact explains the whole E1 family: the event's
*when* is set by the rollout rather than the conditioning, so no channel specifying *where*
can move it — and exp015's "text places randomly" was never random, it was early.

**There is a narrow addressable region.** Per-clip hit rate against a scene obstacle
(peak within 0.25 m) is 0.25–0.31 at 1.0–1.4 m of route, and **exactly zero beyond 3.4 m**.
Optimizing the target gives per-clip 0.312 at 5 cm and 0.266 at 8 cm, so **N = 7–8
generator calls buy 90 % success** — against 0/16 for the packet at its obstacles
(Wilson95 upper 0.19). Averaged over obstacles spread across the whole route the rate
falls to 0.04–0.10 and the budget rises to N ≈ 16–32, which is why the headline table
understates the achievable method.

**The method that follows is staging, not control.** Walk to a point ≈ 1.2 m before the
obstacle, generate a fresh clip there under the prompt, measure box clearance at the
obstacle, accept or resample. This is distinct from translation-equivariance, which is
separately measured false (§36): the offset must come from re-planning the approach, not
from shifting an existing clip's route. Scope is kinematic, one prompt, one route, one
depth; the history/horizon configuration is the obvious confound on the frame-34 window
and is untested; and 5–8 cm is exactly where tracking previously ate the margin, so the
tracker must enter the endpoint before any of this is called a traversal.

## 38. EXP-022A: the exp021 references do not survive SONIC at the staged obstacle

`outputs/exp022_exact_tracking_bridge/` (schema `exp022a-exact-tracking-bridge-v1`, status
`complete`, 64/64 SONIC rollouts, zero new ARDY samples, Scene2Motion commit `291f2ec`, two
launches of 32 environments, physics seed 0). Full note:
`docs/ramp-exp022-exact-tracking-result-2026-09-01.md`.

**Exact-centre audit first.** §37's `0.312/0.266` rates came from a lower-bound + tolerance
calculation rather than an exact-probe union — the exact ±0.25 m unions measured later are 24/64
and 23/64 (`outputs/analysis_exact_centre_cost_curve/receipt.json` → `tolerant_union`), so the two
families of number are not comparable and 0.375 ≠ 0.312 is not a fresh non-reproduction. They used a ±0.25 m placement-tolerant
endpoint chosen post hoc on the same clips. Replaying the exact `BoxHeightProbe(1.2, 0.20)`
against the archived exp021 qpos gives **12/64** clears at 5 cm and **11/64** at 8 cm (per clip
0.19/0.17, N for 90 % = 12/13, not 7–8). The "7–8 calls" budget and the findings-page staged
demo are withdrawn as method results; they remain addressable-window and capability analyses.

**Tracking retention is zero.** Every reference that clears the staged box kinematically is lost
after SONIC: paired retention 0/13, 0/12, 0/11, 0/7, 0/6, 0/2 at 3/5/8/12/20/30 cm. The achieved
endpoint requires a non-terminated rollout, passage through the obstacle corridor, finishing
beyond it, and exact whole-body collision freedom at the graded height; removing the
non-termination guard recovers nothing. SONIC terminated 53/64 rollouts; the 11 survivors are
all non-clearing references. An unguarded collision query on achieved states reports 43/64
"clears" at 5 cm — robots that stopped before the hypothetical box — and must never be quoted.

**Boundary.** The box was absent from Isaac; this is achieved-state replay against the
collision model, on root-XZ-only references under the pinned SONIC evaluation configuration.
It does not isolate amplitude, the unconstrained root height/heading and speed distribution of
this pool, or the tracker. It does close "stage, then select, then SONIC" under the current
pipeline: more samples cannot repair a zero reference-to-achieved retention rate. The kinematic
addressability map and its sampling cost stay as diagnostics; EXP-023 (prompt handoff) remains
the decisive native-interface timing test.

## 39. EXP-023: a delayed prompt does not move the step event

`outputs/exp023_prompt_handoff/` (schema `exp023-prompt-handoff-v1`, status `complete`,
32/32 samples, fresh paired seeds 4500–4507, four B=8 schedule calls, 16/16 Horizon52 window
calls). Full note: `docs/ramp-exp023-prompt-handoff-result-2026-09-01.md`.

Each seed produced four prompt schedules through ARDY's released `autoregressive_step`
interface at its GUI-default minimum history (one four-frame token): all-WALK, STEP from
frame 0, WALK→STEP at frame 52, and WALK→STEP at frame 104, with byte-identical shared
prefixes and corresponding-window noise (verified exact on features and decoded qpos).
Both preregistered gates passed (`step_0` events 6/8 ≥ 4; `all_walk` seeds with any event
0/8 ≤ 1). **The delayed arms produced no whole-body-clearable (≥ 3 cm) step event in the
96-frame post-onset window in 0/8 seeds each**, against 6/8 for the same seeds prompted from
frame 0; the delayed arms' largest unfiltered clearable height was 0.030 m once and 0.000 m in
13 of 16 rows. Every trajectory completed the route (progress 0.994–1.006, route MAE
≤ 0.035 m), so this is a missing behaviour, not a derailed rollout. No delayed event exists to
regress on; per protocol the planned-denominator contrast is the result and no binary verdict
is assigned at n = 8.

The exact fixed box at the frozen-latency prediction (frame onset + 34) cleared 1/8 at 3 cm
and 0/8 at 5 cm for `step_0` (events fell at frames 18–64), and 0/8 everywhere for the delayed
arms and the WALK control, with 8/8 traversal in every arm — a prospective, fresh-seed negative
for fixed-latency staging beside §38's exact-centre audit.

**Scope.** Horizon52 checkpoint, minimum-history handoff, one route and speed, kinematic
only. Longer history crops, the Horizon8 checkpoint, re-issued prompts and latent-state APIs
are untested interface contracts. The campaign cannot distinguish "the prompt is read only
while the rollout context is established" from "a four-frame handoff attenuates the prompt".

**Provenance.** The host process was killed during analysis after 5/32 rows; the committed
resume script re-scored the durable archives through byte-identical frozen sources (source
manifest, G1 model, runtime and prompt-cache identities revalidated; the five archived rows
recomputed identically), regenerating nothing. The interrupted receipt and rows are preserved
and hashed in the final receipt.

**What this closes.** With exp017/018/019/020 (no spatial channel moves the event) and §37
(the event lands early on its own), the native-interface timing claim is landed negative and
scoped: for this checkpoint at the released minimum history, neither a spatial constraint nor
a later prompt changes *when* the STEP behaviour occurs.

## 40. 2026-09-02: the three-pillar framing, the shared host gate, and the first A0 analysers

The PI guidance of 2026-09-02 asks for the same three contributions to be presented as a
**trackability contract**, a **measured-deficit closed-loop repair operator**, and a **verified
data engine**, with two guardrails: never overclaim (no "all-terrain planner"; "evaluator
cutoff" and "dynamic loss of balance" distinguished in every sentence) and never present an
autopsy without the method. `docs/framing-2026-09-02-contract-repair-engine.md` binds that
narrative to receipts. Three of its statements needed tightening to stay inside the evidence:
"execution-*certified*" becomes "execution-*audited*" (zero retained traversals in both
families); "≈10⁵ verified clips per GPU-day" becomes a tiered extrapolation (corpus pilot v2:
268 kinematically verified duck records in 301.9 s → 7.7 × 10⁴ per GPU-day at that tier, 0 at
the executed tier); and "verified execution margins τ(d)" cannot be offered (EXP-1B:
`insufficient_data`). Pillar 2 is carried by the duck-axis Table 2 (§21) and by the *refusal* of
the float (0/44 lifting exp021 clips pass the gate, Wilson upper bound 0.08), not by any new
method run before the deadline.

**Host gate.** `scene2motion/host_gate.py` measures free VRAM, available RAM and concurrent
Isaac processes and is called by every EXP-02x driver before it touches its output directory;
the report is bound into the receipt. On 2026-09-02 the host failed it all day (two co-tenant
Isaac jobs: ≈ 4 GB free VRAM, 6–8 GB available RAM), so nothing was launched; every driver was
built and tested on CPU instead.

**Threshold convention pinned (claims hygiene).** The contract receipt's sweep flags
`max_unsupported_run_s > thr`: 0.20 → 53/53 terminated flagged and 3/11 survivors flagged;
0.28 → 51/53 and 0/11; 0.32 → 46/53 and 0/11. The quoted post hoc optimum "≥ 8 frames (0.32 s):
51/53 and 11/11" is therefore the rule `run > 0.28 s`, i.e. `run ≥ 0.32 s` on the 0.04 s grid.
The EXP-024 draft wrote it as `> 0.32 s`, which would have preregistered a different (46/53)
rule; corrected before preregistration. The calibrated gate remains `run > 0.20 s` (≥ 6 frames);
the onset-alignment statistics use runs ≥ 5 frames. Plan §6 now states all three.

**A0(c) exact-centre cost curve** (`experiments/analyze_exact_centre_cost_curve.py` →
`outputs/analysis_exact_centre_cost_curve/`; 64 clips × 121 centres (0.60–6.60 m, 0.05 m) ×
{3, 5, 8} cm; 23,232 exact `BoxHeightProbe.clears` calls, 286 s CPU). The recomputed hits at
1.2 m and 3.6 m equal EXP-022A's reference rows clip for clip (asserted). Under the exact
endpoint, 1.2 m is the *unique* maximum on the grid for 5 cm (12/64) and 8 cm (11/64); the
3 cm maximum is 15/64 at 1.15 m; at the 3.6 m control every height is 2/64. The curve is an
addressability analysis of one archive on one route and its maximum is a post hoc selection;
EXP-022A's paired guarded retention is zero at both measured centres. Fig. 4 (`experiments/
fig_cost_curve.py`) and Fig. 5 (`experiments/fig_contract_gate.py`, the float and the gate) are
regenerated from these committed outputs only, and each writes the numbers it draws beside
itself (`docs/figures/fig{4,5}_numbers.json`).

**A0(a) event frames landed** (`experiments/analyze_event_frames.py` →
`outputs/analysis_event_frames/`, 743 s CPU; EXP-021's lift profile recomputed byte-identically,
64/64 archived rows agree). Over the 49 lifting exp021 clips the root-crossing frame of the
tallest lift region has median 35 (q10–q90 20.8–55.4; 1.4 s), **40/49** inside the first 50
frames (Wilson 0.69–0.90) and 4/49 after frame 60; the nominal-speed conversion gives 42/49
and 4/49 (median 34.4). The plan's "43/49" is corrected. Two nuances the paper must carry:
(i) **the prompt does not lift exactly once** — 30/49 exp021 clips contain more than one
positive lift region (1/2/3/5/6/8 regions in 19/17/6/4/2/1 clips); "early" is a statement about
the *tallest* region in exp021 (40/49) and about the *first clearable* lift in exp023; (ii)
under the windowed streaming interface (exp023 `step_0`, STEP re-issued every 52-frame window)
the whole-clip scan finds 1–4 regions per clip with the tallest region *late* (root crossing at
frames 106–190) in 5/8, while the archived 96-frame-window detector's first ≥ 3 cm event is at
frames 18–64 in 6/8 (7/8 lift ≥ 3 cm somewhere in the clip; s4507's 0.12 m lift at frame 106
fell outside the window). The delayed arms are unchanged and stronger: on the whole 200-frame
scan `step_52` has two sub-3 cm lifts (frames 72 and 112) and **no ≥ 3 cm lift anywhere**,
`step_104` none at all, `all_walk` none — EXP-023's zero is not an artefact of its 96-frame
window.

**Built, not yet run (CPU-tested drivers; launch order in the framing note §3):** EXP-023b
(`experiments/exp023b_prompt_switch_control.py`), EXP-028
(`experiments/exp028_termination_free_rollouts.py`), EXP-024
(`experiments/exp024_reference_contract.py`), and the A0(a) event-frame analyser
(`experiments/analyze_event_frames.py`). Their protocols stay DRAFT until the commit that
precedes each first sample.

## 41. EXP-023b: the SQUEEZE control is refused at its substrate gate, and the delayed STEP prompt replicates at 3/8

Run 2026-09-02 07:35 EDT beside two co-tenant Isaac jobs (ARDY-only host gate, 7.9 GB free),
32/32 samples, seeds 4640–4647 spent (`outputs/exp023b_prompt_switch_control/`, result note
`docs/ramp-exp023b-prompt-switch-result-2026-09-02.md`). The preregistered sidestep composite
fired in **0/8** `squeeze_0` clips (heading deviation ≤ 28°, lateral excursion ≤ 0.07 m, no foot
crossing), so the substrate gate refused the measurement: under a dense `root_xz` route the
SQUEEZE prompt is executed as a lifting gait (5/8 step events from frame 0, 2/8 clearing a 5 cm
box at the predicted centre; 0.34 rad RMS from walking), not as a sidle. The positive control
is inconclusive and must be re-designed on fresh seeds.

The delayed arm is the finding: `step_52` elicited the step in **3/8** fresh seeds (latencies
0.84–3.0 s after the switch; 0.04–0.10 m clearances) where EXP-023 saw 0/8. Pooled, a STEP
prompt delivered at frame 52 elicits the step in 3/16 (Wilson 0.07–0.43) against 6/8 from
frame 0; none of the delayed events cleared a box at the predicted centre. Under the
preregistered replication rule the abstract's "a later prompt does not elicit the step" and the
"tied to the rollout origin" mechanism are **withdrawn**; the surviving statement is "less
often, later, and unplaced". The handoff-transmission statistic (delayed prompts move the
post-switch joints by 0.14–0.20 rad RMS; prefix exactly zero) removes attenuation as the
leading alternative. `all_walk` specificity held (0/8 sidestep, 0/8 step events).

## 42. EXP-024 kinematic stage: the float under every native root contract; predictions frozen before SONIC

Fresh seeds 4600–4631 × {free, pin_h, pin_y, pin_yh} (128 references, 16 paired B=8 calls),
scored on CPU and the per-clip gate predictions committed before any launch
(`outputs/exp024_reference_contract/predictions.jsonl`, sha `18a2fb14…`, HEAD blob `51e1a5a`;
note `docs/ramp-exp024-kinematic-stage-2026-09-02.md`). The `free` arm replicates exp021
prospectively (elicitation 23/32 = 0.72 vs 44/64 = 0.69 at ≥ 3 cm; exact 5 cm at the
preregistered 1.2 m centre 7/32 = 0.22 vs 12/64 = 0.19): P2 passes. Every arm is constructible
by its preregistered criterion, and **no clip in any arm is contact-consistent** (0/32 × 4), so
P4 is NO-GO before tracking: the prompt elicits a bilateral no-support run under all four native
root contracts. Pinning root height shortens the median run (0.48 → 0.36 / 0.22 s), lowers the
root peak (0.955 → 0.78 m) and elicitation (0.72 → 0.59 / 0.34) but leaves 16–24/32 flagged;
pinning heading alone lengthens the run (0.74 s) and raises elicitation (27/32). P1's 2×2 has
95 flagged and 33 passed references waiting for the SONIC stage. Process note: the score stage
was blocked twice by provenance refusals unrelated to the data (agent worktrees under
`.claude/`, then the driver's own resume path); the resume path added in `a95e8f6` preserves
both blocked attempts and re-scored eight rows byte-identically.


## 43. Coverage, not selection: the clearing and the tracking-completing sets are disjoint

Motivated by the advisor's third review (`docs/pi-advice-2026-09-02-c.md`, Experiment B) and
computed from committed EXP-022A rows by `experiments/analyze_pool_coverage.py`
(`outputs/analysis_pool_coverage/summary.json`). At the staged centre x = 1.2 m over the 64-clip
exp021 pool, 5 cm box: 12 references clear, 11 rollouts complete without an evaluator cutoff,
and **0 do both**; 14 achieved trajectories pass the lateral corridor and finish beyond the
obstacle (11 of them uncut), 50 never reach the obstacle, and 0 satisfy the guarded endpoint.
The two useful sets being disjoint means selected success is zero for *every* selection rule over
this pool, including an offline oracle, at every candidate budget up to 64 — the limitation is
the candidate pool, not the ranking function. Reference-level coverage does respond to sampling:
N90 is 12 fresh draws at 5 cm and 13 at 8 cm (binomial convention, reproducing §37's figure), or
11 and 12 when sub-sampling this fixed pool (hypergeometric); the analyser reports both side by
side so the two conventions are never confused. At the unstaged control centre x = 3.6 m the
reference count is 2/64 and the joint count is likewise 0. Scope: obstacle absent from Isaac,
achieved-state replay, one route, one scene, physics seed 0, one rollout per reference. This is
a descriptive decomposition of existing rows, not a new campaign, and it does not generalise
beyond this pool; EXP-029 (`docs/ramp-exp029-selection-vs-coverage-protocol.md`) is the
prospective, beam-present version for the duck family.

## 44. Engineering findings that unblock EXP-029 and EXP-024b (2026-09-02, CPU only)

Read from the SONIC checkout at `ca86b5e` while the GPU was held by a co-tenant; nothing here
has been executed, and each item names what must still be measured.

**The obstacle can be put in the physics scene.** `gear_sonic/envs/manager_env/
modular_tracking_env_cfg.py` lines 595–670 spawn a `RigidObjectCfg` at `{ENV_REGEX_NS}/Table`
from `sim_utils.CuboidCfg` with `kinematic_enabled=True`, `collision_enabled=True` and
`activate_contact_sensors=True`, sized by `config["table_size"]` and placed by
`config["table_position"]` (identity quaternion on the cuboid path). The node is
`manager_env.config`, so the launch overrides are `++manager_env.config.add_table=true`,
`++manager_env.config.table_position=[x,0,z]`, `++manager_env.config.table_size=[sx,sy,sz]`. The
release checkpoint's saved config has no `add_table` key, so the branch is off by default. The
contact sensor is what would make **collision an observed outcome class** rather than a geometry
replay. The (x, y, z) mapping of `table_size` is inferred from `CuboidCfg` and the code's
width/depth/thickness naming and must be confirmed by measuring the spawned prim.

**Two defaults are incompatible with our route, and both are the same failure mode as the old
14 s clip cap.** `manager_env.config.env_spacing` defaults to **2.0 m** while our route is
**7.2 m**; since the obstacle is spawned once per environment at that environment's origin, a
robot walking the route would cross neighbouring environment origins and meet *their* boxes.
`manager_env.config.episode_length_s` defaults to **10.0 s** against references of about 8.3 s,
leaving little margin for the robot to actually reach the obstacle. Both must be set explicitly
and recorded; EXP-029's §2.2–2.5 validate them rather than assuming.

**The known-trackable jump control exists but needs a format conversion.** The checkout ships
`gear_sonic_deploy/reference/example/tired_one_leg_jumping_R_001__A359` (500 timesteps, 29
joints, CSV: `joint_pos/vel`, `body_pos/quat/lin_vel/ang_vel_w` over 14 tracked bodies). It is in
the **deployment** reference format, not the `motions.pkl` our eval bridge consumes via
`scene2motion/sonic_export.py`, so EXP-024b needs a conversion that preserves physical duration
and the joint ordering. That the asset exists settles the availability question the advisor
raised; it does not settle trackability under our eval configuration.

**Reuse note.** EXP-029 must not reimplement the endpoint: `exp022.score_trajectory` already
computes passage, lateral corridor, finish-beyond, stall and graded clearance, and
`exp028.fall_detection` / `classify_outcome` already implement the preregistered
`fell > stalled > walked_through > cleared` ordering with the 0.50 m pelvis and 0.70 up-axis
bounds. EXP-029 adds an observed-collision class from the table contact sensor on top of them.

**Analysers now carry tests.** `tests/test_analyze_pool_coverage.py` (13) pins the coverage
arithmetic, both N90 conventions and the disjointness decomposition;
`tests/test_analyze_repair_paired_bootstrap.py` (17) pins the truth coercion that a first
attempt got wrong, the pairing, the sign of the losing comparison and bootstrap determinism.
Suite: 595 passing.

## 45. Scene-level traversal outcomes: the pool fails by colliding, not only by being cut off

`scene2motion/traversal_eval.py` (new) scores a tracked trajectory against a whole traversal
problem — the scene's **start, goal and obstacles** — and returns one outcome class from
`completed / collided_obstacle / collided_wall / fell / cutoff / timeout / stalled`, plus
`rejected` for an assigned trial that was screened out and never executed. Collision is split
into obstacle and corridor wall so that "hit the beam" and "walked around and hit a wall" are
never one number; completion requires passing the obstacle **inside the corridor**, so walking
around is a failure (local traversal, not navigation). Fall thresholds are EXP-028's
preregistered constants, pinned by a test; `exp022.score_trajectory` is untouched and the landed
receipts are unaffected.

`experiments/analyze_traversal_outcomes.py` applies it to all 64 archived EXP-022A rollouts with
start (0, 0), goal (7.2, 0) and the box at x = 1.2 m
(`outputs/analysis_traversal_outcomes/summary.json`):

| box height | fell | collided with the box | collided with a wall | evaluator cutoff | timeout | stalled | **completed** |
|---|---|---|---|---|---|---|---|
| 3 cm | 0 | 21 | 0 | 43 | 0 | 0 | **0** |
| 5 cm | 0 | 21 | 0 | 43 | 0 | 0 | **0** |
| 8 cm | 0 | 22 | 0 | 42 | 0 | 0 | **0** |
| 12 cm | 0 | 25 | 0 | 39 | 0 | 0 | **0** |
| 20 cm | 0 | 30 | 0 | 34 | 0 | 0 | **0** |
| 30 cm | 0 | 31 | 0 | 33 | 0 | 0 | **0** |

This is strictly more informative than §38's zero. The frozen bridge could only say "0/64
retained", which merges a robot that drove its body through the box with one that never left the
first metre. Under the scene endpoint, **21 of 64 achieved trajectories intersect a 5 cm box at
the specified position**, rising to 31 at 30 cm, and 43 are stopped by the evaluator before
anything else happens. Nothing falls, nothing stalls, nothing times out, and nothing completes at
any height. Collision outranks the cutoff in the precedence, so the 21 include 10 rollouts the
bridge also counted as terminated (53 = 43 + 10) and the 11 that were never cut off.

**Scope, and it is a real limit.** These rollouts were tracked with the obstacle *absent* from
Isaac, so `collided_obstacle` means the recorded motion intersects the box's volume in replay:
the robot never felt contact and its controller was never perturbed by one. It is a statement
about the achieved motion, not about contact dynamics. EXP-029 is the obstacle-present version.

**Tracker checkpoint, settled.** NVIDIA's model card lists three released G1 checkpoints and
assigns the **default release** (`sonic_release/last.pt`, sha `e6bdab3f…`, 10 future frames at
20 ms) to *motion tracking*; the low-latency and v1.1 variants are aimed at teleoperation and
VLA execution, v1.1 being heading-normalised and wrist-pose augmented rather than a better
tracker. Every tracking number in this project therefore already uses the official
motion-tracking checkpoint, by the vendor's own assignment rather than by default. **SONIC v1.1
is now downloaded** (`sonic_v1_1/last.pt`, 1.14 GB, sha `af24831a…`, with its `config.yaml` and
`model_config.yaml`; same `terrain_type: trimesh`, `episode_length_s: 10.0`) so that a
controller-generality arm can re-run the same references under a second released controller and
answer whether the zero is checkpoint-specific. That arm has not been run.

## 46. EXP-026: the reference screen ranks duck-family cutoffs too, weakly, and speed does not

Preregistered (`docs/ramp-exp026-duck-contract-protocol.md`, `f52ba5c`, amended `8291721` before
any feature existed), run on CPU in 17 s over the 526 committed EXP-1B duck references (first
rollout per clip, 344 terminated, 36 scenes) with the identical feature code the step family
used; result note `docs/ramp-exp026-duck-contract-result-2026-09-03.md`,
ledger `outputs/analysis_duck_contract/`.

Pooled AUC with a cluster bootstrap over scenes: **contact (longest no-support run) 0.674
(0.625–0.721)**, crouch depth 0.707 (0.632–0.770), **speed 0.441 (0.328–0.552)**. Within scene,
where route length and beam geometry are constant (13 of 36 scenes evaluable, 1,144 pairs):
contact **0.694**, speed 0.601, crouch 0.565. The plan's decision rule — transfer only if the
contact AUC exceeds the speed AUC — is satisfied on both measures: contact − speed is +0.233
(CI 0.120–0.362) pooled and +0.092 (CI −0.018–0.210) within scene, so the ordering is established
pooled and only suggested once the confound is removed. Contact is the only primary above 0.5 in
**all seven** preregistered strata (three dip bins 0.578–0.693, four beam counts 0.663–0.707).

**The confound the plan named in advance is refuted.** Speed does not predict duck cutoffs: its
interval spans chance, it is at or below 0.5 in five of seven strata, and survivors are *faster*
than terminated references (median max root planar speed 1.87 vs 1.75 m/s). The 14 s clip cap
made every reference fast; it did not decide which were cut off. Crouch depth, the best pooled
predictor, is largely a between-scene effect: 0.707 pooled falls to 0.565 within scene and to
0.45–0.62 inside two of three dip bins. The three primaries are near-independent (contact vs
speed **−0.18**), so the contact feature is not a repackaged speed.

**What it does not license.** The transfer is directional and much weaker than the step family's
0.997, and the screen is not usable as a duck accept/reject rule at its calibrated threshold:
sensitivity 0.910 (0.875–0.936) but specificity 0.297 (0.235–0.367), and **441 of 526 duck
references (84 %) have a no-support run longer than 0.20 s** — the crouch pipeline produces
floats too, and every one of the 526 references has some no-support run. Nothing here says a
passing duck reference is trackable; EXP-1B's endpoint was 0/859 traversals. Post hoc on a
completed campaign, one physics seed, one rollout per clip, schema-v1 archives, one tracker.

The screen's cross-family evidence is now two actuation channels within the step family plus one
different behaviour family, which is what lets the paper call it a property of this controller's
evaluator applied to references rather than an artefact of stepping.


## 47. The obstacle can be put in the tracker's scene, and the robot is stopped by it

The project has never executed a rollout with the obstacle *present* in physics: every executed
result replays achieved states against our collision model with the box absent from Isaac. §44
found the spawn mechanism by reading the checkout. Running it exposed three things reading did
not (`experiments/probe_obstacle_present.py`, `outputs/probe_obstacle_present/`; operational
probes, no seeds spent, not campaign evidence).

**The spawn pose does not survive a reset.** Whenever a table exists, `commands.py:3134` rewrites
its pose per environment on every reset — from the motion's own `table_pos` / `table_quat` plus
that environment's origin when the motion carries them, and otherwise from a fallback that puts
it at the object's position with `z = 0.76` and no environment offset (`:3149-3181`). Motion
pickles from `scene2motion.sonic_export.write_motion_pkl` carry no table metadata, so setting
`table_position` on the command line alone would have placed the obstacle somewhere else and the
campaign would have measured a box that was not where it said. The working route is **per-motion
table metadata inside the pickle**. Geometry, from the code: `CuboidCfg` size is the full x/y/z
extents and the position is the centre, so a box of height *h* on the floor is
`size=[depth_x, width_y, h]` at `pos=[x, y, h/2]`.

**`add_table=true` without `add_object=true` crashed the checkout.** `right_hand_wrist_links` was
assigned only inside the `add_object` branch while the table-to-robot contact sensor used it
unconditionally (`modular_tracking_env_cfg.py:729` vs `:755`), so a table used as a plain scene
obstacle raised `UnboundLocalError` before the simulator started. The obvious workaround,
`add_object=true`, fails differently: it spawns a missing `data/wheelchair.usd`. The fix is to
keep that sensor inside the `add_object` branch, where its filter list is defined and where
comparing table contact against object contact means anything.

**With the fix, the obstacle is real.** Paired launches over two archived EXP-021 references,
identical but for the box, physics seed 0:

| reference | max root x, no box | max root x, box | cut off, no box | cut off, box |
|---|---|---|---|---|
| s4434 | 6.06 m | **0.29 m** | no (397 frames) | yes (283 frames) |
| s4459 | 0.98 m | **0.38 m** | yes (52) | yes (45) |

A 0.30 m box at x = 0.5 m stops the robot in front of it. s4434 is the decisive one: without the
box it walks 6.06 m of the route and is never cut off; with the box it stops at 0.29 m, short of
the box's front face at 0.40 m. So the obstacle spawns where it was asked for, the robot collides
with it, and the collision changes the outcome. **This is the first physical-obstacle evidence in
the project**, and it is what EXP-029 needs to measure local traversal completion rather than
infer it from replay.

**An ordering constraint this creates.** The fixed file is in the SONIC core source manifest that
every receipt binds equal to EXP-022A's `44e98c45…`, so EXP-028 and EXP-024's tracked stage — which
must stay comparable to EXP-022A and never set `add_table` — have to run against the pinned,
unpatched tracker, and EXP-029 against the patched one with its own declared baseline. The
EXP-028 driver refused to launch on the patched checkout, which is the guard working as designed.

**Host RAM, not VRAM, is the resource that bites.** A launch that started with 13.6 GiB available
drove the host to 376 MiB and the kernel OOM-killer took a browser process. Both probes now
refuse to start below a RAM floor and kill their own launch if available RAM collapses; the
campaign gate's RAM threshold already carried the measured 6.8 GiB consumption.

**Commit pins for this path.** The tracker-fork fix is `7c63c53`, preserved on the branch
**`exp029-obstacle-present`**; it is **reverted on `research/practice-utility` by `350cae1`** so
that the pinned core-source manifest again equals EXP-022A's `44e98c45…` and EXP-028 / EXP-024's
tracked stage can launch. EXP-029 must run against `7c63c53` (cherry-pick or check out the
branch) and declare its own baseline; the two families are never compared across that boundary.
Verified after the revert: `exp028.tracker_identity()` reports the expected manifest.

**Three contact quantities, never merged.** No table contact sensor is instantiated by this
configuration. The table-to-robot sensor lives inside the `add_object` branch — that is exactly
where this fix put it — and `add_object=true` still fails on the missing `data/wheelchair.usd`;
the other table sensor, `table_to_hand_contact_sensor`, is gated on `robot_has_hands`
(`"43dof" in robot_type or "hand" in robot_type`) while the release config's robot type is
`g1_model_12_dex`, and it filters right hand/wrist links, which could never see foot or shin
contact with a floor box. So spawning the box does **not** by itself make sensor contact
available. EXP-029 must still record and report separately: **physical contact with the obstacle**
(which needs a contact-sensor path that does not yet exist), **violations of the conservative
replay clearance model** (which inflates obstacles by 4 cm, so it is a different event and a
stricter one), and **prohibited floor contact**. Collapsing them would reintroduce exactly the
ambiguity §45 removed.

## 48. EXP-030: obstacle in the physics scene — local traversal measured, replay proxy validated

Preregistered in `docs/ramp-exp030-obstacle-present-stepping-protocol.md` (sha
`c1849036…`, bound into the receipt before the first launch). **This is the first campaign in
this project to put the obstacle in the physics scene**; every executed result before it — EXP-1B,
EXP-1C, EXP-011/012/014, EXP-022A, §45 — tracked with the box absent from Isaac and replayed the
achieved states against our collision model. Six launches of 32 over the **same 64 archived
EXP-021 references EXP-022A tracked**, physics seed 0, one rollout per reference, 192 rollouts
requested and 192 returned, no new ARDY samples and no seeds spent. All three arms — including
the obstacle-absent control — ran on the patched worktree `/home/linjiw/lucid/GR00T-WBC-exp029`
(branch `exp029-obstacle-present`, contains `7c63c53`), which declares its own tracker baseline
rather than asserting EXP-022A's `44e98c45…` manifest. Scene: start (0, 0), goal (7.2, 0),
corridor half-width 1.4 m, box at x = 1.2 m, depth 0.20 m, width 2.8 m from
`stepover_eval.step_scene`, carried as per-motion `table_pos`/`table_quat` inside the motion
pickle (§47). Ledger `outputs/exp030_obstacle_present/` (`receipt.json`, `summary.json`,
192-row `rows.jsonl`).

**Outcome classes over all 64 assigned trials per arm** (`summary.arms`; the precedence is
`rejected > fell > collided_obstacle > collided_wall > cutoff > timeout > stalled > completed`,
and nothing was rejected, nothing fell and nothing hit a corridor wall in any arm):

| arm | obstacle | completed | evaluator cutoff | stalled | collided with the box |
|---|---|---|---|---|---|
| `absent` | none | **1** | 54 | 9 | 0 |
| `present_05` | 5 cm box at x = 1.2 m | **0** | 44 | 0 | 20 |
| `present_20` | 20 cm box at x = 1.2 m | **0** | 34 | 0 | 30 |

**Local traversal completion is now measured, not inferred: 0 of 64 with the obstacle present**
(Wilson 0–0.057 at both heights), against **1 of 64 without it** (Wilson 0.003–0.083; the
completing reference is `s4434`). That single completion is what gives the zero its meaning: the
controller can carry one of these references down this route to the goal, and the obstacle is
what removes that. Local traversal, not navigation — the endpoint requires passing the obstacle
**inside the corridor**, so walking around would have been a failure, and no rate here is a
navigation success rate.

**The three preregistered predictions all held** (`summary.predictions`).

*P1, the control (threshold ≥ 58/64).* The `absent` arm reproduces EXP-022A on **63 of 64**
termination flags; the terminated counts are 53 (EXP-022A) and 54 (EXP-030 `absent`), and the one
disagreement is `s4459` (397 valid frames and uncut in EXP-022A, cut off at 120 here). Valid-frame
counts agree on 54 of 64 (0.844, 0.736–0.913) and are reported alongside; the preregistered rule
was the flag. So the fork fix and the run conditions are inert at the resolution this campaign
needs, and comparison across the two campaigns stands.

*P2, completion.* 0 of 64 in both present arms, as predicted. The prediction carried a
consequence — any completion would have been the project's first measured local traversal and
would be named — and none occurred.

*P3, the proxy.* The obstacle-absent replay predicts the obstacle-present class on **63 of 64**
references: agreement **0.984** (Wilson 0.917–0.997), Cohen's **κ = 0.964** (percentile bootstrap
over the 64 references, 2000 resamples, seed 30, no degenerate resample; CI 0.882–1.0), against a
preregistered floor of 0.80 agreement and κ ≥ 0.6. Confusion, replay-inferred class in rows and
physics-measured class in columns (`summary.q1_proxy_check.confusion`):

| replay-inferred ↓ / physics-measured → | collided with the box | evaluator cutoff |
|---|---|---|
| collided with the box (21) | 20 | 1 |
| evaluator cutoff (43) | 0 | 43 |

Per class, agreement of the measured label is 20/20 for collision and 43/44 for the cutoff. The
single disagreement is `s4410`, whose replay intersection was the shallowest in the pool (1.8 cm
into the margin-inflated box) and which the physics arm merely cut off: the proxy's one error sits
at its own margin.

**How much the obstacle changed the rollout** (`summary.paired_progress_change`, `absent` minus
`present_05` maximum achieved root x, all 64 paired, none excluded): median **0.0 m**, IQR width
0.5 mm, **8 of 64** references losing more than 0.05 m of progress (`s4408`, `s4418`, `s4419`,
`s4428`, `s4440`, `s4452`, `s4453`, `s4463`), maximum +3.93 m and minimum **−4.52 m**. The median
of zero is the expected shape, not a null result: most of these rollouts are cut off in the first
metre and never reach x = 1.2 m, so the box cannot touch them. Where the robot did get there the
effect is large and it runs both ways — `s4463` loses 3.94 m (5.28 → 1.34 m) and `s4459` gains
4.52 m (1.73 → 6.25 m), its rollout running the full 397 samples uncut with the box present after
being cut off at 120 without it.

**What "collided with the box" means here, and it is not what it meant in §45.** The box was
physically in the scene, so these achieved states were produced by a robot that ran into a real
obstacle and was stopped by it — not by a replay of a motion recorded in an empty world. The
signature is in the depths: in the present arms the penetration into the margin-inflated box
piles up **at** the 4 cm margin (18 of the 20 collisions at 5 cm lie in 0.040–0.055 m and 25 of
the 30 at the 20 cm height in 0.040–0.068 m, i.e. the collision primitives are essentially
resting on the box surface), whereas the obstacle-absent replay of the same references drives 19
of its 21 intersections to 0.062–0.075 m, through the volume the box would have occupied. **The
class is still scored by the same conservative model** (`traversal_eval.evaluate_traversal` →
`G1Body.trajectory_report`, with every scene box inflated by `BODY_MARGIN` = 4 cm). No table
contact sensor was instantiated in this configuration — the table-to-robot sensor lives inside the
`add_object` branch, which EXP-030 does not enable (and `add_object=true` still fails on the
missing `data/wheelchair.usd`, §47), and the hand sensor needs a 43-dof/hand robot type — so
physical sensor contact was not measured at all here, and §44's three quantities stand with the
sensor one still unavailable: physical sensor contact, replay-clearance violation and floor
contact remain distinct, and this campaign reports the second, measured under physics with the
obstacle present. Collisions shallower than the 4 cm
margin may not be primitive contact at all: two of the twenty at 5 cm (`s4462` at 2.7 cm, also
cut off, and `s4434` at 3.6 cm) and five of the thirty at 20 cm (0.0015–0.0195 m). One of them
matters for the headline: **`s4434` is the one reference that reached the goal inside the
corridor, uncut, in the 5 cm arm** (397 frames, final goal distance 0.44 m) and is classified
`collided_obstacle` only because collision outranks completion. P2's zero at 5 cm therefore rests
on a margin violation for that reference; settling it needs a contact-sensor path that does not
yet exist.

**What it licenses.** Two things, and only these. First, local traversal completion has been
**measured** with a real obstacle and it is zero, with an obstacle-absent control that completes
once, so the zero is attributable to the obstacle rather than to the route or the controller
being unable to walk it. Second, the obstacle-absent replay endpoint that every earlier executed
result in this paper depends on is **validated against physics on this pool** (0.984, κ 0.964),
so those results keep their meaning instead of inheriting a stated caveat about a proxy nobody had
tested.

**What it does not license.** No claim of a traversal system. No generalisation beyond this route,
this scene and this obstacle position — one physics seed, one rollout per reference, one prompt
family. Nothing about ducking. The proxy is validated on 64 references from one pool at one box
height, not as a general property of replay scoring. And the completion zero is an outcome for
*these* references under *this* controller, not a statement that the task is impossible.

**Scope to state whenever the numbers are quoted.** Scored under `traversal_eval` **version 1**
(recorded as `summary.scene.evaluator_version`); a re-score under the corrected evaluator is a
separate versioned analysis and must say so. The **timeout class is "not assessed", not zero**:
no wall-clock deadline was preregistered (`time_limit_s: null`), so its count of zero carries no
information. Rates are over all 64 assigned trials per arm, never over an executed or accepted
subset, so rejecting references could not read as success. "Cut off" is the tracker evaluator's
stopping rule, not a fall — nothing in any arm met the fall criterion.


## 49. EXP-025: the float crosses the prior boundary on a thin margin, the early timing does not

Preregistered in `docs/ramp-exp025-kimodo-cross-prior-protocol.md` (sha `3d8a47bc…`, bound into
the receipt before the first sample, with a three-point amendment also written before any sample),
run 2026-09-03 in 24.7 min on the clean worktree `93323a09`; result note
`docs/ramp-exp025-kimodo-cross-prior-result-2026-09-03.md`, ledger
`outputs/exp025_kimodo_cross_prior/`. **The project's first cross-prior campaign.** 64 STEP and 64
WALK references from **Kimodo-G1-RP-v1** (HF `3020ad8c…`, checkout `1aece8c1…`) — a released
**offline, non-autoregressive** G1 prior, one denoising pass and no history window, 30 fps — on
seeds 4700–4763, paired per seed, scored by the same committed analysers the ARDY family uses with
the support thresholds read from the frozen exp016 calibration receipt (`f6dba8be…`). Exact
accounting 128/128 over 16 of 16 generate invocations, latent pairing verified in every chunk. The
STEP embedding is the **byte-identical** vector ARDY used, copied under the same sha1 key, with the
two LLM2Vec wrappers equal as normalised ASTs and no encoder loaded. **Kinematic only: tiers are
generated 128, kinematically scored 128, SONIC-executed 0** — nothing here is an execution outcome,
and the 0.20 s rule is the reference screen for predicted tracking cutoffs, not a physical verdict.

**The two preregistered rules split, and the split is the finding: the float crosses the
architecture boundary — 19 of 23 elicited references, 0.826, on a Wilson interval (0.629–0.930)
that straddles the preregistered 0.80 bar — and the early timing does not.**

*Timing → `ardy_window_attributed_to_autoregressive_rollout_context`.* Over the 41 STEP clips with
a lift position, Kimodo puts the lift inside the first 2.0 s in **1 of 41 = 0.024**
(0.004–0.126), median lift time **3.87 s** by root crossing (q10–q90 2.47–5.77) and 3.77 s by
nominal-speed conversion, against ARDY's **40/49 = 0.816** (0.686–0.900) and 42/49, median 1.40 s.
Both event-time definitions agree (`definitions_agree: true`), and so does the companion elicited
denominator (0/23, 0–0.143); over all 64 assigned trials it is 1/64. The rule's branch threshold is
a fraction **at or below 0.4** and the measured value is 0.024, so "the prompted behaviour lands
early" is scoped to ARDY's autoregressive rollout, not to released G1 priors. The distributions
barely overlap — ARDY's q90 (2.22 s) is below Kimodo's q10 (2.47 s) — and the single early Kimodo
clip (`s4720_step`, 1.73 s) misses the 3 cm elicitation bar by 0.3 mm.

*Screen → `screen_generalises_to_released_g1_priors`, thinly.* **19 of 23** elicited Kimodo clips
exceed the calibrated 0.20 s support screen: **0.826, Wilson 0.629–0.930**, against the
preregistered **0.80** threshold; the secondary `> 0.28 s` cut flags the same 19; ARDY's comparator
is 44/44. **The point estimate meets the bar and the interval does not settle it** — the Wilson
interval straddles 0.80 and one clip flips the verdict (18/23 = 0.783). The uncertainty is about
how many clips, not where the line is: the four passing clips run 0.000–0.100 s and the 19 flagged
ones 0.367–2.967 s, with **no elicited clip between 0.100 and 0.367 s**, so any threshold in that
band returns the same count. Ballistic ratio over the 19 flagged elicited clips is 1.35–9.85
(different set, same shape as the 1.3–15.6 of the 12 ARDY references clearing 5 cm at x = 1.2 m).
So an offline, non-autoregressive prior produces the same float.

**Elicitation is much lower, which is what bounds the screen rule's denominator.** Whole-body-
clearable lift ≥ 3 cm *somewhere along the route* — an anywhere-measure, never a clearance rate at
a specified obstacle — is **23/64 = 0.359** (0.253–0.482) against ARDY's 44/64 = 0.688, an interval
that excludes ARDY's point estimate; any positive lift is 41/64. The free WALK control is a hard
floor: **0/64** elicited, 0/64 with any lift, and a maximum longest no-support run over all 64 WALK
clips of exactly **0.000 s**, so 0/64 are flagged. Paired over the 64 seeds: 23 step-only
discordant pairs, 0 walk-only, 41 concordant; median paired differences +0.0143 m lift height and
+0.183 s longest run, descriptive with no interval claimed on the difference.

**Denominators differ between the two rules by design** (`decisions.denominator_rule`). The timing
rule is reported over clips with a lift position because that is the denominator ARDY's comparator
was measured on (40/49, 42/49 over the 49 exp021 clips with any positive lift); the screen rule
keeps the elicited denominator because the protocol names it ("≥ 80 % of Kimodo's elicited clips").
Both are reported beside rates over all 64 assigned trials — 19/64 = 0.297 elicited-and-flagged,
31/64 = 0.484 flagged in total — so no accepted subset can read as a rate.

**Placement, kept separate from production.** Exact whole-body clearance at the two preregistered
centres is **0/64 at every graded height at the staged x = 1.2 m**, and 4/64 at 3 cm, 3/64 at 5 and
8 cm at the unstaged x = 3.6 m. This is a measured non-clearance, not a coverage artefact: the
envelope swept both centres in 64/64 clips (`not_reached: 0`, 120/120 scan points). The cause is
placement — elicited lifts land at 1.91–5.79 m (median 3.29 m) and the lowest lift anywhere in the
arm sits at 1.576 m (`lift_position_any_lift_m.min`, `s4720_step`) — not route error (smooth-root
path MAE median 0.032 m, progress ratio 0.982).
Produced 23/64, placed at the specified obstacle 0/64, tracked: not attempted.

**What it licenses.** The screen's target behaviour is not an artefact of one architecture. With
**EXP-026** (§46, duck family, contact AUC 0.674 against a speed confound at 0.441) the screen now
has evidence of **two different kinds** — a ranking against real tracking outcomes for the duck
family, property recurrence with no tracking outcomes at all for the second prior — across **two
behaviour families and two priors**. It is not a 2 × 2: only ARDY varies the behaviour family and
only the step family varies the prior. Each row carries its own limit:
EXP-026's transfer is directional and much weaker than the step family's 0.997, and **EXP-025's
margin is thin and its Wilson interval straddles its own threshold**. And the early-window finding
is now correctly scoped to the autoregressive rollout rather than to released G1 priors.

**What it does not license.** Nothing about execution: no rollout was run, Kimodo references have
never been tracked, and no number here says a Kimodo reference would or would not be followed by
the controller or bears on falling. No placement claim (0/64 at the specified obstacle), no local
traversal and no navigation claim — no obstacle was in any physics scene. And the two verdicts must
not be merged into a single "cross-prior" statement.

**Scope.** One prior checkpoint, one prompt per arm, one straight route, one scene, 64 seeds,
kinematic only. Timing is compared in seconds and the support thresholds are fps-free, because
Kimodo runs at 30 fps and ARDY at 25. Route error is measured against `smooth_root_pos`, as the
amendment requires, since `smooth_root_2d` constrains the ADMM-smoothed root and not the raw
pelvis. The WALK arm uses Kimodo's own prompt text, so it is this prior's elicitation floor, not a
cross-prior WALK comparison. Three departures from the protocol's letter are recorded: the host
gate ran with relaxed thresholds
(4 GiB VRAM / 8 GiB RAM, `require_no_concurrent_isaac: false`, measured 8,077 MiB free VRAM and
11,237 MiB available RAM with no Isaac process running) rather than the ≥ 12 GB / ≥ 18 GB the
protocol names; the campaign was launched from the sibling worktree
`/home/linjiw/scene2motion-exp030` at the same commit; and the protocol's prose secondary cut
("the post hoc 0.32 s cut") is implemented in its fps-free form, `> 0.28 s`, which cannot change
the count on this corpus (no elicited clip runs between 0.100 s and 0.367 s). Part B, the reduced
capability audit rerun, is out of scope and was not run, so the 4.5× / 6× counting rows stay transcript-sourced. Seeds
4700–4763 are spent; 4800–4927 remain reserved for EXP-027.
