# EXP-029 protocol — shared-pool obstacle-position study: selection, coverage, correction

**Status: preregistered** (2026-09-03, before any sample). Arms, scene conditions, candidate
pool, seed block, endpoints, planned denominators, primary comparison, decision rules, kill and
refusal conditions, budget and statistics are fixed below. From this commit the file is frozen:
later changes are dated amendments appended to the section they touch, never edits in place, and
the file's sha256 is bound into the receipt before the first launch (EXP-030's `protocol_identity`
pattern). Motivated by the advisor's third review (`docs/pi-advice-2026-09-02-c.md`, Experiments
A/B/C). The retrospective half of Experiment B for the stepping family is already answered from
committed rows by `experiments/analyze_pool_coverage.py`; this protocol covers the prospective
ducking study.

**Schedule reality — read this before planning around it.** The driver, the per-motion table-pose
generalisation, the spawn-pose read-back, the fixed extra-crouch proposer and their tests do not
exist yet (§15). This campaign **will not run before the ICRA 2027 deadline (2026-09-15)**, and it
did not run before the Sep 6 number freeze. Nothing in the submission may be written as though it
had. It is the experiment that decides whether the paper's final story is an *evaluation
contribution* (where generation, selection and tracking fail for obstacle-relative tasks) or a
*method contribution* (a selection or correction mechanism improves actual local traversal). Until
it runs, the paper ships as the evaluation contribution, and §VIII cites this file as a
preregistered plan, not as pending results.

## 1. The question

Which generated motions can this controller use to get through this obstacle, from this starting
state? Concretely, three sub-questions on one shared candidate pool:

- **A. Does scene information change which motion should be selected?** Hold the instruction,
  starting state, controller and candidate pool fixed; vary the obstacle position; compare
  selectors that do and do not see the scene.
- **B. Is the failure selection, or a missing candidate?** Report pool coverage (does any
  candidate complete the traversal) separately from selected success (does the deployed rule
  pick one that does).
- **C. Does measured correction create a usable candidate when selection cannot?** Compare extra
  sampling, a fixed extra-crouch adjustment, and the measured correction, under one budget.

## 2. Gating engineering (must be done and validated before any arm is generated)

### 2.1 The beam in the physics scene — MEASURED 2026-09-03, and the first recipe was wrong

Every executed result before this date tracked with the obstacle **absent** and replayed achieved
states against our collision model. That is no longer a limitation: an obstacle can now be put in
the SONIC scene and the robot feels it. Three things had to be corrected first, and two of them
invalidate the recipe this protocol carried on 2026-09-02.

**(a) The spawn pose does not survive a reset.** `add_table=true` spawns
`{ENV_REGEX_NS}/Table` as a `CuboidCfg` with `kinematic_enabled`, `collision_enabled` and
`activate_contact_sensors`, but `commands.py` rewrites the table pose per environment on **every
reset**: from the motion's own `table_pos` / `table_quat` when the motion carries them (plus that
environment's origin), and otherwise from a fallback that puts it at the object's position with
`z = 0.76` and no environment offset. Setting `++manager_env.config.table_position` on the
command line — what this protocol originally specified — therefore places the obstacle somewhere
else entirely. **The working route is per-motion table metadata written into the motion pickle**,
which makes the cached branch fire and puts one obstacle per environment at the intended place.

**(b) `add_table` without `add_object` crashed before the simulator started.**
`table_to_robot_contact_sensor` filters on `right_hand_wrist_links`, which is defined only inside
the `add_object` branch, while the sensor itself sat one level out inside the `add_table` branch.
A table used as a plain scene obstacle rather than a manipulation surface therefore raised
`UnboundLocalError` during env-cfg construction. Fixed in our fork by moving the sensor inside
the branch that defines its filter and that gives it meaning (it compares table contact against
*object* contact for grasp termination). **Fork commit `7c63c53` must be pinned in every EXP-029
receipt**, and the campaign must refuse to run against a checkout without it.

**(c) `add_object=true` is not a workaround.** It tries to spawn `data/wheelchair.usd`, which the
checkout does not ship, and it would put an irrelevant object in the scene. Do not use it.

Geometry convention, confirmed against the spawned prim: `table_size` is the cuboid's **full
x, y, z extents** and `table_position` is its **centre**, so a box of height *h* resting on the
floor is `size=[depth_x, width_y, h]` at `z = h/2`; a beam leaving clearance *c* is
`size=[depth_x, width_y, t]` at `z = c + t/2`.

**Verified on 2026-09-03** (`experiments/probe_obstacle_present.py`; operational probes, no
seeds spent, not campaign evidence). Paired launches over two archived EXP-021 references,
identical but for a 0.30 m box, physics seed 0:

| reference | max root x, no box | max root x, box | cut off, no box | cut off, box |
|---|---|---|---|---|
| s4434 | 6.06 m | **0.29 m** | no (397 frames) | yes (283 frames) |
| s4459 | 0.98 m | **0.38 m** | yes (52) | yes (45) |

s4434 is decisive: without the box it walks 6.06 m of the route and is never cut off; with a box
at x = 0.5 m it stops at 0.29 m, short of the box's front face at 0.40 m. The obstacle spawns
where it was asked for, the robot collides with it, and the collision changes the outcome. This
is the first physical-obstacle evidence in the project
(`outputs/probe_obstacle_present/report_h0.3.json` → `comparison.per_motion`, marked
`campaign_evidence: false`).

**Commit pins.** The fork fix is `7c63c53`, preserved on branch **`exp029-obstacle-present`** and
**reverted on `research/practice-utility` by `350cae1`**, because the patched file sits in the
core-source manifest every receipt binds equal to EXP-022A's `44e98c45…`: EXP-028 and EXP-024's
tracked stage must run against the unpatched tracker to stay comparable, and EXP-029 against the
patched one with its own declared baseline. Never compare across that boundary.

**Amendment, 2026-09-03 (after EXP-030), to the two-checkout rule.** `CLAUDE.md`'s "Two tracker
checkouts" ruling makes this concrete and EXP-030 executed it: the execution root is the dedicated
worktree **`/home/linjiw/lucid/GR00T-WBC-exp029`** (branch `exp029-obstacle-present`, HEAD
containing `7c63c53`), the legacy checkout is **refused** as an execution root, and **both** the
obstacle-absent and the obstacle-present arms run there so the comparison is internally
consistent (`outputs/exp030_obstacle_present/receipt.json` → `design.execution_root`, which also
records `legacy_root_refused`). EXP-029 declares its own tracker baseline the same way: the
receipt records the checkout commit and the full core-source manifest and does **not** assert
equality with `44e98c45…`.

**Host RAM, not VRAM, is the binding resource.** A launch that started with 13.6 GiB available
drove the host to 376 MiB and the kernel OOM-killer took an unrelated process. Both probes now
refuse to start below a RAM floor and kill their own launch if available RAM collapses; EXP-029
inherits that guard.

**Amendment, 2026-09-03 — the host requirement is now measured, not inherited.** The plan of
record's "≥ 12 GiB free VRAM, ≥ 18 GiB available RAM, no concurrent Isaac process" was never
measured and blocked every SONIC campaign for a day. `experiments/probe_sonic_vram.py` measured a
32-environment launch — the campaign configuration — at **3,769 MiB peak VRAM and ≈ 6,810 MiB of
host RAM, 61.6 s, completing beside four concurrent Isaac processes**
(`scene2motion/host_gate.py` → `SONIC_MEASURED_NEED`, report
`outputs/probe_sonic_vram/report_envs32.json`). EXP-029 therefore uses the measured preset
`host_gate.SONIC_LAUNCH_GATE` — **≥ 5,500 MiB free VRAM, ≥ 9,500 MiB available RAM,
`require_no_isaac: false`** — evaluated before the output directory is touched and bound into the
receipt per launch, with concurrent Isaac processes **recorded, not gated**, exactly as EXP-030
did (`outputs/exp030_obstacle_present/receipt.json` → `provenance.initial_host_resource_gate` and
`launches.*.host_resource_gate`).

**Not yet verified, and 2.4–2.6 still gate the campaign:** both probe motions were cut off early
and their roots never passed 0.96 m against a box at 1.2 m, so this establishes *physical
effect*, not that a traversal can succeed. Contact attribution is also still to be built: the
table's contact sensor is enabled but the probe does not yet read it, and EXP-029 must record and
report **physical contact with the obstacle**, **violations of the conservative replay clearance
model** (which inflates obstacles by 4 cm and is therefore not the same event), and **prohibited
floor contact** as three separate quantities, never merged. — **Superseded in part: see the
amendment immediately below.** The sensor-contact quantity is unavailable and is no longer
required; the separation of the remaining quantities stands.

**Amendment, 2026-09-03 (after EXP-030) — the contact-sensor quantity is not available, and this
protocol no longer requires it.** EXP-030 established that the table-to-robot contact sensor lives
inside the `add_object` branch, which an obstacle-present campaign does not enable, and that
`add_object=true` still fails on the missing `data/wheelchair.usd`; the hand sensor additionally
needs a 43-dof robot type (REPORT §48). So **no physical sensor contact was measured by EXP-030
and none will be measured by EXP-029.** The three quantities stay distinct and EXP-029 reports the
second of them — a violation of the conservative replay clearance model
(`traversal_eval.evaluate_traversal` → `G1Body.trajectory_report`, every scene box inflated by
`BODY_MARGIN = 0.04 m`), **measured under physics with the obstacle in the scene**, which is a
different event from the same model applied to a motion recorded in an empty world. Prohibited
floor contact is not scored at all: the scoring scene carries no floor box and the tracker's own
terrain carries the floor. Every statement of an EXP-029 collision must name which of the three it
is, and the shallow-collision caveat EXP-030 hit applies here too — an intersection shallower than
the 4 cm margin may not be primitive contact.

### 2.2 Risk found while reading: environment spacing versus route length

`manager_env.config.env_spacing` defaults to **2.0 m**, and our route is **7.2 m** long. Because
the obstacle is spawned once per environment at that environment's origin, a robot walking the
full route would cross several neighbouring environment origins and meet **their** boxes. Either
raise the spacing past the route length plus a margin
(`++manager_env.config.env_spacing=12.0` or more), or shorten the route, or run fewer
environments per launch. This must be settled before the pool is generated, and the resolved
spacing recorded in the receipt. Whether neighbouring props actually intrude is inferred from
per-environment prim paths and shared-scene physics; 2.4 measures it instead of assuming.

**Settled, 2026-09-03: `env_spacing = 16.0 m`.** EXP-030 ran at 12.0 m over a 7.2 m route
(`outputs/exp030_obstacle_present/receipt.json` → `design.env_spacing_m`) with no sign of a
neighbour's box on the route. EXP-029's route is longer — start (0, 0), goal (8.5, 0), §10 — so
the spacing rises to 16.0 m, leaving ≥ 6.5 m of clear ground past the goal. The value is passed as
`++manager_env.config.env_spacing=16.0`, recorded in the receipt, and checked by V1 (§2.4).

### 2.3 Risk found while reading: episode length versus time to reach the obstacle

`manager_env.config.episode_length_s` defaults to **10.0 s**. Our references are 200–208 frames
at 25 fps, about 8.3 s, and the robot must still reach the obstacle inside the episode. The
earlier duck tracking study reached 0 traversals in 859 rollouts, with 554 ending at the
evaluator's tracking-error cutoff and only 200 reaching the first beam
(`outputs/exp1b_execution_clearance_v2/receipt.json` → `outcomes.terminated` 554,
`outcomes.reached_first_obstacle` 200, `outcomes.passed_last_obstacle` 0); the ~14 s clip cap
bounds only the 305 rollouts that were **not** cut off, so it is a second, smaller mechanism and
not what stopped most rollouts. Episode length is nevertheless the same *class* of constant and
can bind here. Set `episode_length_s` explicitly, record it, and validate with 2.5.

**Settled, 2026-09-03: `episode_length_s = 25.0 s`, and the 14 s clip cap no longer binds.** The
earlier duck study's references were built for multi-beam scenes whose goal is
`max(8.0, x_last + 3.5)` metres — up to about 25 m at six beams and a 3.5 m gap
(`scene2motion/demo/scene_builder.py`) — so at the pipeline's 0.9 m/s nominal speed the route
needed far more than the 14 s cap in `demo/ardy_runner.py::MAX_DURATION_S`, and the reference
simply stopped part-way. EXP-029 uses **one** beam and a fixed 8.5 m route: at 0.9 m/s that is
about 9.4 s, i.e. **236 of the 350 available frames** (computed from `n_frames`, `SPEED = 0.9`,
`MAX_DURATION_S = 14.0` and 25 fps in `scene2motion/demo/ardy_runner.py` for a straight 8.5 m
path; a plan with lateral detours is longer but still well inside the cap), so the cap is not
reached and the reference covers the whole route by construction. The episode is set to 25.0 s — about 2.7× the reference
duration, the same ratio EXP-030 used (20.0 s against ≈ 8.3 s references) — and passed as
`++manager_env.config.episode_length_s=25.0`. The traversal deadline is a separate, tighter number
(§3): `time_limit_s = 20.0 s`, so a rollout that runs out of time is distinguishable from one that
runs out of episode.

### 2.4 Validation: the box is where we asked, and it collides

Spawn a box of known size at a known position, then, from the simulation state rather than from
the configuration, confirm its pose and extent, confirm that a rollout aimed through it registers
contact on the table sensor, and confirm that a rollout aimed beside it does not. Confirm also
that no neighbouring environment's box lies on the route. Dump the resolved scene configuration
into the receipt.

**Made exact, 2026-09-03 — V1, and its threshold.** One launch of 8 environments, 8 walking
references (seeds 5340–5347, §10), each carrying a **different** `table_pos` drawn from the 12
scene conditions, with the beam **raised off the floor** (this is a ducking study: the cuboid
floats at `z = c + t/2`, which no probe has yet exercised).

- **V1a, pose.** Read each environment's `Table` prim pose from the simulation state at the first
  and the last step of the episode. Every environment must be within **0.01 m** and **0.01 rad**
  of its own motion's `table_pos` / `table_quat` plus that environment's origin, at both samples.
  8 of 8 required. The last-step check is what tests that a kinematic floating beam does not
  drift or fall; the first-step check is what tests the per-motion route.
- **V1b, extent.** The prim's reported x/y/z extents must equal the launch's `table_size`
  `[0.24, 2.25, 0.25]` to 1 mm.
- **V1c, neighbours.** No other environment's `Table` prim may lie within the corridor
  `|y| ≤ 1.2 m` over `x ∈ [-1.0, 9.5] m` of any environment's own frame. Checked from the prim
  poses, not inferred from spacing.
- The **table contact sensor is not available** (amendment in §2.1), so the "registers contact on
  the table sensor" clause of the original text is **not** part of V1 and no EXP-029 statement
  depends on it. The physical-effect check is V1d instead: with the beam lowered to an underside
  of 0.30 m — below the standing pelvis, so it blocks rather than requires a duck — a walking
  reference's maximum achieved root x must stop before the beam's near face in 8 of 8, the same
  signature `s4434` produced in the 2026-09-03 probe. V1d is one-sided — a rollout cut off early
  satisfies it trivially — which is why V2 below, with the beam raised out of the way, is what
  proves the route is reachable at all.

Failure of any of V1a–V1d stops the campaign and is recorded as a refusal
(`raised_beam_pose_not_held`, `table_extent_mismatch`, `neighbour_prop_on_route`,
`no_physical_effect`). It is not worked around by lowering the tolerance.

### 2.5 Validation: the horizon reaches the obstacle

A plain walking reference on the route must reach and pass the obstacle position within the
episode in at least 7 of 8 rollouts, with the obstacle raised out of the way. If it cannot, no
negative traversal result from this protocol means anything.

**Made exact, 2026-09-03 — V2, and its threshold.** The same 8 walking references (seeds
5340–5347), one launch, beam underside raised to **1.60 m** — above the 1.35 m standing top
recorded as `stand_top_m` in `outputs/phase4e_architecture_v2_s8/experiment.json`, and the
maximum `BeamParams.clamped()` allows — at the farthest condition, `x = 5.0 m`. **≥ 7 of 8**
rollouts must reach `max root x ≥ 5.0 m` inside `episode_length_s`. Below 7, refusal
`horizon_does_not_reach_the_beam`.

### 2.6 Validation: the evaluator can score a success

A plain walking reference under a beam high enough not to interfere must be classified as a
completed local traversal by the outcome classifier. If the evaluator cannot produce a success on
a case that plainly is one, its zeros are uninterpretable. The outcome classes and the fall
detector already exist and are preregistered in
`experiments/exp028_termination_free_rollouts.py` (`fall_detection`, `classify_outcome`,
ordering `fell > stalled > walked_through > cleared`, pelvis below 0.50 m or up-axis below 0.70
counting as a fall); EXP-029 extends that classifier with an **observed collision** class from
the table contact sensor rather than replacing it.

**Made exact, 2026-09-03 — V3, and a correction to the last sentence.** V3 is scored on the same
V2 rollouts: **≥ 7 of 8** must be classified `completed` by
`scene2motion.traversal_eval.evaluate_traversal` under the exact criteria of §3. Below 7, refusal
`evaluator_cannot_score_a_success`. The classifier EXP-029 uses is `traversal_eval`, whose
`OUTCOMES` precedence is `rejected > fell > collided_obstacle > collided_wall > cutoff > timeout >
stalled > completed` and whose fall constants `FALL_PELVIS_Z_M = 0.50` / `FALL_UP_Z = 0.70` are
the EXP-028 constants, pinned by a test; `exp022.score_trajectory` stays frozen and untouched.
**There is no observed-collision class from a table contact sensor** — that sensor is unavailable
(§2.1 amendment) — so `collided_obstacle` here is the conservative replay clearance model applied
to achieved states produced under physics with the beam present, and every report of it says so.

A failure at any of 2.4, 2.5 or 2.6 stops the campaign and is recorded as a refusal; it is not
worked around by loosening the endpoint.

### 2.7 What EXP-030 settled, and what EXP-029 inherits from it (added 2026-09-03)

EXP-030 (`docs/ramp-exp030-obstacle-present-stepping-protocol.md`,
`outputs/exp030_obstacle_present/`, REPORT §48) is the stepping-family obstacle-present study. It
ran six launches of 32 over the 64 archived EXP-021 references, physics seed 0, one rollout each,
192 requested and 192 returned, on the patched worktree. Five things it settled are now premises
of this protocol rather than open questions.

1. **The method to reuse.** Per-motion `table_pos` / `table_quat` inside the motion pickle; the
   cuboid convention of §2.1; paired absent/present arms in one campaign; `traversal_eval` as the
   endpoint; the attempt/resume discipline; the measured host gate. §15 lists the exact functions.
2. **The replay proxy is validated for stepping.** Obstacle-absent replay predicted the
   obstacle-present class on **63 of 64** references — agreement 0.984 (Wilson 0.917–0.997),
   Cohen's κ 0.964 (percentile bootstrap over the 64 references, 2,000 resamples, seed 30, CI
   0.882–1.0) — against preregistered floors of 0.80 and 0.60
   (`receipt.json` → `summary.q1_proxy_check`, `summary.predictions.P3`). The single miss,
   `s4410`, is the shallowest replay intersection in the pool.
3. **Local traversal completion with a real obstacle is 0 of 64** at both 5 cm and 20 cm (Wilson
   0–0.057), against **1 of 64** in the obstacle-absent control (`s4434`)
   (`summary.arms.*.local_traversal_completion`). The single control completion is what makes the
   zero attributable to the obstacle rather than to the route — EXP-029 keeps an absent arm for
   exactly that reason (§11).
4. **The fork fix is inert.** The `absent` arm reproduced EXP-022A on 63 of 64 termination flags
   (`summary.p1_absent_vs_exp022a.termination_flag.n_agreeing`), so the patched worktree does not
   change the tracker's behaviour at the resolution these campaigns need.
5. **Two scope habits to copy.** The evaluator version is *recorded*, not asserted
   (`summary.scene.evaluator_version` = 1; a re-score under a corrected evaluator is a separate
   versioned analysis), and an unassessed class is reported as "not assessed", not as zero
   (`summary.scene.timeout_note`: no deadline was configured, so EXP-030's timeout zero carries no
   information). EXP-029 fixes a deadline (§3) so its timeout class *is* assessed whenever the
   evaluator supports it.

**Does EXP-029 still need its own proxy check? Yes — a reduced, free one, and it is not a gate.**
EXP-030's own scope statement is that the proxy is "validated on 64 references from one pool at
one box height, not as a general property of replay scoring", and three things differ here.
(a) The obstacle is **overhead**: a contact removes head or torso clearance and knocks the upper
body, rather than stopping the feet against a floor box, so the physics response the proxy must
predict is not the one that was validated. (b) EXP-030's agreement was carried by the 43 of 64 pairs that were `cutoff` on both sides
(`outputs/exp030_obstacle_present/receipt.json` → `summary.q1_proxy_check.confusion.matrix`;
the present arm labels 44 `cutoff` and 20 `collided_obstacle`), so most pairs agreed trivially;
EXP-029's references are built to cover the whole route (§2.3), so the proxy is tested on a
harder set. (The separate figure of 50 references that never reached the obstacle is the EXP-022A
replay analysis, `outputs/analysis_pool_coverage/summary.json` → `results[0].n_never_reached_obstacle`,
used as such in the §5 coverage table — not an EXP-030 quantity.) (c) EXP-030's
one error sat at its own margin, which makes its error rate a property of how many references sit
near the margin — a pool property, not a general one. The check costs **no extra rollouts**: the
absent arm is already required as the completion control, so its achieved states are scored
against each condition's beam and compared with the physics-measured class of the same reference
at that condition (§12, P7). It is **descriptive replication in a second behaviour family**, not
a gate: a low agreement does not stop the campaign, it attaches a stated caveat, with the
direction of the bias named, to ducking results that rest on replay.

## 3. Endpoints (defined before the pool is generated)

**Local traversal completion** (primary): the robot passes beneath the beam within the specified
lateral corridor, finishes beyond it, does not contact the beam or the floor with a prohibited
body part, does not fall (pelvis below 0.5 m or orientation error beyond the stated bound), and
does so within the time limit. Walking around the beam does not satisfy this task; the corridor
is part of the definition.

This is *local traversal*, deliberately not *navigation*. Navigation asks whether the robot
reaches a destination through a scene, where walking around an obstacle can be an excellent
solution. Navigation adds route choice, transitions between motions, replanning and recovery,
and no result here establishes any of them.

**The exact criteria, fixed 2026-09-03** — `scene2motion.traversal_eval.TraversalCriteria` with
`goal_tolerance_m = 0.5`, `corridor_half_width_m = 1.2` (the demo family's `CORRIDOR_HALF`),
`time_limit_s = 20.0`, `fall_pelvis_z_m = 0.50`, `fall_up_z = 0.70` and
`required_clearance_m = 0.0`, evaluated against the **traversal scene** of §10 (the beam box
alone) by `evaluate_traversal(qpos, scene, terminated=..., sample_dt_s=0.02, criteria=...)`.
`sample_dt_s` is a keyword argument of that call, **not** a field of `TraversalCriteria` — the
six names above are the whole dataclass, and a driver that passes `sample_dt_s` to the
constructor will raise.
There is no separate allowed-contact body set: the collision test is `G1Body.trajectory_report`
over the whole-body primitive set with `BODY_MARGIN = 0.04 m`, so any body primitive intersecting
the margin-inflated beam is a prohibited contact. Prohibited **floor** contact is not scored —
the scoring scene carries no floor box — and no EXP-029 sentence may imply it was. `collided_wall`
cannot be produced, because the traversal scene carries no wall boxes (the physics scene can hold
only one cuboid, §10); departure from the corridor is captured by `passed_within_corridor`
instead, and a rollout that finishes beyond the beam outside the corridor is a **non-completion**,
never a success. The **timeout class is assessed only when the driver's recorded
`traversal_eval` evaluator version supports a completion-time deadline**; under an evaluator that
does not, the class is reported as "not assessed", exactly as EXP-030 reported it, and the driver
records the version it actually ran (EXP-030's `evaluator_version()` pattern, §15).

**Secondary, reported separately, never merged into the primary:**

| measure | definition |
|---|---|
| reference clearance | the reference clears the beam at the specified position, whole-body collision model, margin stated |
| completed tracking run | no evaluator cutoff (the release termination configuration) |
| world-space progress | route distance covered, in metres, not root-relative pose similarity |
| outcome class | success / collision / fall / evaluator cutoff / timeout or stall before the beam / rejected before execution |
| cost | generation calls, scoring seconds, executed rollout seconds, wall clock |

**Evidence levels stay separate** (`docs/project-goal-2026-09-02.md`): produced → placed →
collision-free reference → completed tracking run → clearance preserved after tracking → local
traversal completion. No EXP-029 table may let one stand for the next, and the two Experiment C
stages of §6 are two of these levels, not two views of one.

**Reporting rule.** Every rate is over **all assigned trials**. A rule that rejects a candidate
without executing it scores that trial as a non-completion; if rejection avoided an unsafe
attempt, that benefit is reported separately as an outcome class. Rejecting everything must
never read as perfect task performance. `traversal_eval.OUTCOMES` already carries `rejected` as
the class a driver records for an assigned trial that was screened out and never executed, and
§4's screen-using selectors are the rules that produce it.

## 4. Experiment A — same motions, different obstacle positions

**Design.** One instruction, one starting state, one route. A single shared pool of N candidate
references generated once. M beam positions along the route, fixed in advance (e.g. early,
middle, late relative to the gait cycle the model produces). Every selector chooses from the
*same* pool for every position; only the obstacle moves.

**Fixed 2026-09-03:** N = 36 candidates (§10), and the obstacle varies over **12 scene
conditions** — six beam positions `x ∈ {3.0, 3.4, 3.8, 4.2, 4.6, 5.0} m` crossed with two beam
underside heights `h ∈ {1.05, 0.95} m` (§10). Every one of the 36 candidates is executed at every
one of the 12 conditions, so the whole 36 × 12 outcome table is measured and every selector's
choice is looked up in it rather than costing its own rollout.

| selector | information it uses | note |
|---|---|---|
| `random` | none | lower bound |
| `screen_only` (draft name: trackability-only) | predicted tracking quality | our reference screen for predicted tracking cutoffs; ranks by longest period outside the support test |
| `screen_filter_then_random` (draft name: text-and-trackability, TEXEDO-style) | predicted tracking quality + instruction match | **not TEXEDO**: our own predictor, their decision shape, see §8 — and with one instruction the semantic term is constant, so the arm reduces to a hard screen filter, below |
| `geometry_only` | reference clearance at the specified beam position | scene-aware, controller-blind |
| `geometry_and_screen` (draft name: geometry + trackability) | both | the combination this project argues for |
| `oracle` (draft name: offline oracle) | observed traversal outcome of every candidate | privileged diagnostic, not a deployable method; rollout budget disclosed |

**The exact rules, fixed before any sample.** Every selector sees only the budget-k prefix of the
pool (§5) and returns one candidate or the decision `reject`; ties break on ascending pool seed,
so every rule is deterministic.

1. `random` — uniform over the prefix, `numpy.random.default_rng(129)` drawn once per (condition,
   budget) in a fixed order (129, not the bootstrap's 29, so the two streams are never confused).
   Never rejects.
2. `screen_only` — ranks ascending by `max_unsupported_run_s`, the longest period with neither
   foot meeting the support test, computed by the committed feature code behind
   `experiments/analyze_trackability_contract.py` against the thresholds frozen in
   `outputs/exp016_threshold_calibration/receipt.json` (sha `f6dba8be…`, 284 calibration clips).
   Picks the smallest. Never rejects: it is a ranker.
3. `screen_filter_then_random` — keeps candidates with `max_unsupported_run_s ≤ 0.20 s` (the
   calibrated screen), then ranks the survivors by instruction match. **Every candidate in this
   family is produced under the single cached prompt "A person walks forward."
   (`scene2motion/demo/ardy_runner.py::PROMPT`), so the semantic term is constant and cannot
   order anything**; the arm therefore falls back to the `random` rule among survivors, and is
   reported under that name so no reader mistakes it for a text-ranking result. If no candidate
   passes the filter it returns `reject`. A genuine instruction-ranking arm needs a second
   instruction and the four resolutions of §8; it is out of scope here and is not implied by any
   EXP-029 sentence.
4. `geometry_only` — ranks descending by the reference's minimum overhead clearance at the
   condition's beam (`verify/trace.clearance_trace` against the **reference scene** of §10, the
   same quantity `min_overhead_m` the repair loop acts on), picks the largest. Never rejects.
5. `geometry_and_screen` — among candidates with `max_unsupported_run_s ≤ 0.20 s`, picks the
   largest minimum overhead clearance at the condition's beam. If none passes the screen it
   returns `reject`: where no candidate passes a screen the honest options are rejection,
   resampling or replanning, never quietly executing the best-scoring candidate anyway
   (`docs/project-goal-2026-09-02.md`, metrics discipline). A `reject` is an assigned trial scored
   as a non-completion, with the rejection reported separately as its own outcome class.
6. `oracle` — any candidate in the prefix whose measured outcome at that condition is `completed`;
   if none, the one with the greatest world-space progress. Privileged diagnostic. Its rollout
   budget is the whole 36 × 12 table and is disclosed as such wherever it appears.

Because the whole outcome table is measured, evaluating a selector costs no rollouts — which also
means a new selector could be invented after seeing the outcomes. **These six are the
preregistered set; any selector added later is post hoc and is labelled so wherever it appears.**

A selector that sees only the motion and the instruction returns the same choice when the
obstacle moves; a scene-aware selector can change its choice. **Whether changing it improves
traversal completion is the experiment**, not an assumption.

**Planned denominators.** 12 assigned trials per selector per budget — one per scene condition —
with the full 36 × 12 = 432 measured present-arm rollouts underneath, plus the 36 absent-arm
rollouts of §11. Rates are over all 12, `reject` included as a non-completion. Reported at every
budget of §5, and additionally at the reference-clearance level (does the selected candidate
clear at the specified beam), which is a different evidence level and is never merged with
completion.

**Primary comparison (the campaign's one primary).** `geometry_and_screen` minus `screen_only` on
**local traversal completion**, paired by scene condition, at the full pool (k = 36). Direction:
the scene-aware rule completes at least as often. Decision rule and thresholds in §12 (P5).

**Figure this produces:** the same candidate that completes the traversal at one beam position
and fails at another, from measured rollouts, not an illustration.

## 5. Experiment B — coverage versus selected success

Computed on the same rollouts, at nested candidate budgets k = 1, 2, 4, 8, … , N, where the
budget-k pool is a prefix of the budget-2k pool so that increasing the budget does not silently
change the sample set.

- **Pool coverage(k):** fraction of (scene, position) conditions where at least one of the k
  candidates completes the traversal.
- **Selected success(k):** fraction where the deployed selector's choice completes it.

**Fixed 2026-09-03:** budgets **k = 1, 2, 4, 8, 16, 32, 36**, and the prefix order is
**round-robin over the 12 conditioning conditions in ascending seed index** — candidate 1 is the
first seed of condition 0, candidate 2 the first seed of condition 1, …, candidate 13 the second
seed of condition 0, and so on (§10). Fixing the order in advance is what keeps a prefix from
being all one conditioning position. Coverage is also reported at the **reference-clearance**
level (does any of the k candidates clear at the specified beam), separately and never merged.

Read as: high coverage with low selected success is a selector problem; low coverage at every
budget is a candidate problem, and no ranking function can fix it; clearing references that fail
in execution is a controller-compatibility problem.

**Already answered retrospectively for the stepping family** (`outputs/analysis_pool_coverage/`),
under the replay endpoint, one route, one scene, physics seed 0:

| at the specified obstacle, 5 cm, n = 64 | count |
|---|---|
| reference clears at the specified position | 12 |
| completes tracking without a cutoff | 11 |
| **both** | **0** |
| passes the corridor and finishes beyond | 14 (11 of them without a cutoff) |
| never reaches the obstacle | 50 |
| satisfies the full traversal endpoint | 0 |

The two sets are disjoint, so no selector over this pool could have succeeded whatever its
ranking function. Sampling more references raises reference clearance (90 % coverage at about 12
fresh draws at 5 cm) and leaves joint coverage at zero for every budget up to the whole pool.
For the stepping family the limitation is the candidate pool, not the selection rule. Scope: that
protocol tracked with the obstacle absent; EXP-029's beam-present endpoint is the proper test.

**Amendment, 2026-09-03 — the stepping row is no longer replay-only.** EXP-030 measured the same
pool with the box **in the physics scene** and found local traversal completion **0 of 64** at
5 cm and at 20 cm, with 1 of 64 in the obstacle-absent control
(`outputs/exp030_obstacle_present/receipt.json` → `summary.arms.*.local_traversal_completion`).
So for stepping the coverage answer now holds at the executed level as well, and EXP-029's
contribution is the same decomposition for a **second behaviour family**, with the obstacle
present, over a grid of obstacle positions rather than one.

## 6. Experiment C — correction versus sampling

Four arms under one budget of at most three generations each, on the same beams:

1. single generation, uncorrected;
2. best-of-three resampling with fresh seeds;
3. fixed extra-crouch adjustment (a constant offset, no measurement) applied once, then a second
   time if still deficient;
4. measured correction, Δq = e / |g′₊|, at most twice.

Arm 3 is the mechanism test: if a constant offset performs as well as the measured correction,
then measurement is not what is producing the gain and the contribution must be framed
accordingly.

**The exact arm configurations, fixed 2026-09-03.** All four use the learned proposer
(`outputs/duck_model_v3_m018`, the `tcn` schedule key), the cached prompt, `SPEED = 0.9`,
`DIFFUSION_STEPS = 5`, target margin `MARGIN_M = 0.18 m`, preference `shortest`, and the
verify/repair loop in `scene2motion/verify/loop.py`. Arm names are the ledger's own:

| arm | rule | budget (`max_adapted_generations`) | committed counterpart |
|---|---|---|---|
| `C1_single` | one generation, no feedback | 1 | phase4e `tcn` (0.7292 collision-free, 0.2708 rejected) |
| `C2_resample3` | verify/select over seeds `s`, `s+10000`, `s+20000` (`loop.resample_seeds`, `RESAMPLE_SEED_STRIDE = 10000`), predeclared `_sample_score` order | 3 | phase4e `tcn-resample3` (0.7743; mean 2.99 generations) |
| `C3_fixed_crouch2` | up to two constant-offset corrections | 3 | **new code**, §15 |
| `C4_measured2` | up to two measured corrections, `verify.repair.repair`, `MAX_REPAIRS = 2` | 3 | phase4e `tcn+2` (0.9931 collision-free, 0.375 margin; mean 2.72 generations) |

**`C3_fixed_crouch2` in full, so it is a mechanism control and not a straw man.** It runs the
identical loop to `C4_measured2` — same trigger (a measured deficit against the same target), same
deficit support window, same anticipation kernel `anticipation_kernel(..., LEAD_TAUS = 3.0)`, same
`_smooth`, same clip to [0, 1], same two-iteration bound — and differs in exactly one place: the
per-sample magnitude is a **constant** instead of `e / max(|g′|, SLOPE_FLOOR)`. The constant is
**Δdip = 0.1084 m**, i.e. `Δq = 0.1084 / DIP_MAX = 0.2168` command units, taken from
`outputs/phase4e_architecture_v2_s8/experiment.json` → `repair_stats.mean_magnitude_m` (the mean
commanded extra dip over the **1,925** committed repair steps of that ledger, `n_repair_steps`).
Fixing it from a prior committed receipt, rather than from EXP-029's own data, is what keeps the
arm from being tuned: the two arms command the *same average* extra crouch and differ only in
whether the amount is measured per sample.

Reported in two stages, never merged: (i) does the arm improve obstacle-relative reference
clearance; (ii) does the selected motion complete the traversal after tracking. A deeper crouch
may improve geometry while making tracking harder; that loss is a result, not a nuisance.

**Planned denominators.** 12 conditions × 8 seeds = **96 assigned trials per arm**, 384 in total,
each executed once. Stage (i) is scored on the arm's finally selected reference against the
**reference scene**; stage (ii) on its single rollout against the **traversal scene**. Both are
over all 96, with `rejected` outcomes counted as non-completions.

**Budgets differ by arm and the report must say so — this corrects the "one budget of at most
three generations each" in this section's opening line.** `C1_single` spends one generation; the
other three spend up to three. Only `C2_resample3`, `C3_fixed_crouch2` and `C4_measured2` are
equal-budget and only those three may be compared as such; `C1_single` is the uncorrected
reference point and is never presented as an equal-budget comparator
(`experiment.json` → `method_specs[*].max_adapted_generations` = 1 / 3 / 3 / 3 for these arms).

**Cost is reported as actual generation calls, scoring seconds and executed seconds**, because
"same maximum number of generations" is not equal computation. In the existing kinematic study
the learned proposer used a mean of 2.72 generations under correction and 2.99 under resampling.

## 7. Statistics

The scene (route + beam position) is the inference unit; rates carry cluster-bootstrap intervals
over scenes, and paired arm differences are per-scene differences with the same bootstrap.
Coverage curves are exact for the realised pool and are labelled as such. Planned denominators:
a missing rollout is a failure. Any analysis chosen after seeing outcomes is labelled post hoc.

**Fixed 2026-09-03.** The inference unit is the **scene condition** (beam position × underside
height): **12 clusters**. Cluster bootstrap over conditions, **30,000 resamples, seed 29**
(distinct from EXP-030's κ bootstrap seed 30 so the two are never confused); the resample count and
seed are bound into the receipt. Within-condition rates carry Wilson 95 % intervals over that condition's own
denominator: 36 references for the pool arms of Experiments A and B, 8 seeds for each Experiment C
arm. Paired arm and selector differences are per-condition differences with the same
bootstrap and are **descriptive** at 12 clusters — no significance is claimed.

**Fixed 2026-09-03 — what "descriptive" means where P5 and P6 use an interval as a threshold.**
P5 and P6 call a difference material only when the cluster-bootstrap interval excludes 0. That
interval is a **preregistered decision instrument, descriptive but binding**: it is written down
before any sample so the decision cannot be chosen after seeing the result, and it is *not* a
significance test. At 12 clusters it carries no p-value, no error-rate guarantee and no
inferential claim, and it must never be reported as "significant" or as evidence that an effect
is absent. In particular P6's "answered **no**" is the preregistered decision *this campaign
committed to at this budget*, not a demonstration that measured correction and a fixed crouch are
equivalent; an interval that includes 0 at 12 clusters is equally consistent with an effect this
design cannot resolve, and the write-up must say so wherever that branch is taken. Both intervals
are reported with their point difference, their per-condition counts and their discordant pairs,
so a reader can see the decision's whole basis. Cohen's κ for the
proxy check carries a percentile bootstrap over references within condition and over conditions
pooled, with degenerate resamples excluded and counted, exactly as EXP-030 reported it. Coverage
curves are exact for the realised pool of 36 and are labelled as such, never as a probability for
a fresh draw; where an N90-style number is quoted it names the convention it uses. Every rate
names its denominator and its evidence level, and a class the design did not assess is reported
as "not assessed", never as zero.

## 8. Using TEXEDO as a baseline: cautions

The `text-and-trackability` arm above is **TEXEDO-shaped, not TEXEDO**: it reuses the decision
structure (filter by predicted execution quality, then rank by instruction match) with our own
support-screen predictor, which avoids the confounds below. A genuine comparison with the
released system is a stretch goal and must resolve all four first:

1. **Paper rule versus released code.** The paper filters by a dynamics threshold and then ranks
   semantically; the released CLI combines the dynamics score with a normalised semantic
   distance. These are different rules. Reproduce one and say which.
2. **Motion format.** TEXEDO expects 50 Hz, 36-dimensional G1 motions with a specified joint
   order and quaternion convention; our references are 25 Hz. Conversion must preserve physical
   duration; relabelling the frame rate changes the motion.
3. **Trajectory length.** Their dynamics scorer defaults to the first 1,024 frames. Verify the
   beam encounter falls inside the scored window.
4. **Checkpoint versus retraining.** Published checkpoints exist, but the dynamics-training
   rollout labels are not in the public dataset. Using their checkpoint and retraining on our own
   labels are different experimental conditions and must not be reported as one.

**Note, 2026-09-03.** §4 fixes that arm as `screen_filter_then_random` because this family is
produced under a single instruction, so the semantic half of the decision shape has nothing to
order. The four cautions above are unchanged and remain the conditions on any future
instruction-ranking arm; EXP-029 claims no comparison with TEXEDO.

## 9. To settle before this becomes preregistered — SETTLED 2026-09-03

Pool size N and beam positions M with their exact coordinates; the beam heights and thickness;
the corridor width and the prohibited-contact body set; **`env_spacing` and `episode_length_s`,
both of which currently default to values incompatible with a 7.2 m route (§2.2, §2.3)**; the
time limit; the fall bound (the exp028 constants unless changed deliberately); the seed block
(**reserve 5300–5555**, disjoint from every block in CLAUDE.md); the SONIC launch budget and the
host-resource gate; and the kill conditions. All three validations in §2.4–2.6 must pass before
the pool is generated. Every arm's driver lives in `experiments/` and is resumable by receipt,
and reuses `exp022.score_trajectory` and exp028's `fall_detection` / `classify_outcome` rather
than reimplementing the endpoint.

**Resolutions.** Each item, and where it is now fixed:

| item | resolved to | where |
|---|---|---|
| pool size N | 36 references, learned proposer, uncorrected | §10 |
| beam positions M | 6 positions × 2 heights = 12 scene conditions | §10 |
| exact coordinates | `x ∈ {3.0, 3.4, 3.8, 4.2, 4.6, 5.0} m`, beam centre `y = 0.15 m`, `z = h + 0.125 m` | §10 |
| beam heights, thickness | undersides `h ∈ {1.05, 0.95} m`; full extents `0.24 × 2.25 × 0.25 m` | §10 |
| corridor width | `corridor_half_width_m = 1.2` (the demo family's `CORRIDOR_HALF`) | §3, §10 |
| prohibited-contact body set | the whole-body primitive set at `BODY_MARGIN = 0.04 m`; no floor scoring; no contact sensor | §2.1 amendment, §3 |
| `env_spacing` | 16.0 m | §2.2 |
| `episode_length_s` | 25.0 s | §2.3 |
| time limit | `time_limit_s = 20.0 s`, assessed only under an evaluator that supports it | §3 |
| fall bound | the EXP-028 constants, unchanged: pelvis 0.50 m, up-axis 0.70 | §3 |
| seed block | 5300–5347 and 5400–5407 used, 5500–5555 reserved for a post-refusal resize, plus the derived resampling seeds 15400–15407 and 25400–25407 | §10 |
| SONIC launch budget | 31 planned launches, ≤ 38 before the campaign stops | §11 |
| host-resource gate | the measured `host_gate.SONIC_LAUNCH_GATE` preset | §2.1 amendment |
| kill conditions | eight of them, with their thresholds | §13 |
| validations | V1a–V1d, V2, V3, each with a threshold and a named refusal | §2.4–2.6 |

Two items in the original list changed rather than being filled in: the **contact-sensor**
quantity is unavailable and is no longer required (§2.1 amendment), and **`exp028`'s
`classify_outcome` is not the classifier** — `scene2motion.traversal_eval` is, carrying the same
fall constants pinned by a test, while `exp022.score_trajectory` stays frozen (§2.6 amendment,
§15).

## 10. The pool, the scene conditions and the seed block (added 2026-09-03)

**Scene family.** The single-beam preset of the demo builder,
`scene2motion.demo.scene_builder.build(BeamParams(beam_height=h, beam_width=2.25, n_beams=1))` —
the family the learned proposer and the response fit were built and evaluated on
(`outputs/phase4e_architecture_v2_s8/experiment.json` → `width_m` 2.25, `heights` [0.95, 1.05,
1.2], `n_beams` [3, 4, 5, 6]), so only the beam count and its position leave the training
envelope, and the beam position is the independent variable. Two deliberate departures from the
builder's defaults, both fixed for every condition:

- **the beam is translated in x** to the condition's position instead of the builder's fixed
  `BEAM_X = 4.0`, and
- **the goal is pinned at `x = 8.5 m` for every condition** rather than
  `max(GOAL_X, x_last + 3.5)`, so the route length, the plan and the frame count do not change
  with the beam position. Without this the independent variable would move two things at once.

Derived geometry, from the builder's committed constants (`CORRIDOR_HALF = 1.2`, `WALL_T = 0.15`,
`BEAM_THICK_X = 0.12`, `BEAM_THICK_Z = 0.125`): the beam spans from the left wall inward, giving
box half-extents `(0.12, 1.125, 0.125)` at centre `(x, 0.15, h + 0.125)` — full extents
`0.24 × 2.25 × 0.25 m` — and a bypass of **0.225 m** between the beam's right edge and the right
wall. That bypass is far narrower than the robot, so "walking around" is not physically available
and the corridor rule of §3 is not doing hidden work; it is still stated, because the endpoint is
local traversal and the corridor is part of its definition.

**Two scenes, and they are not interchangeable.**

- **Reference scene** — walls and beam, the full builder output. Used for *reference* geometry:
  clearance traces, the deficit the repair loop acts on, `min_overhead_m` / `min_clearance_m`.
  This is how the phase4e corpus was scored, so Experiment C's stage (i) stays comparable to it.
- **Traversal scene** — the beam box alone, no walls. Used for scoring *achieved* states through
  `traversal_eval.evaluate_traversal`, because the physics scene can hold exactly one cuboid per
  environment (`add_table` spawns one `Table` prim) and there are no walls in it. Scoring achieved
  states against walls that were never there would invent collisions; EXP-030 made the same choice
  (`receipt.json` → `design.scene`, one obstacle, corridor as a rule).

**The 12 scene conditions**, indexed `c = 0…11` in this fixed order (height outer, position
inner):

| c | beam underside h (m) | beam centre x (m) | table_pos (m) | pool seeds |
|---|---|---|---|---|
| 0 | 1.05 | 3.0 | (3.0, 0.15, 1.175) | 5300, 5301, 5302 |
| 1 | 1.05 | 3.4 | (3.4, 0.15, 1.175) | 5303, 5304, 5305 |
| 2 | 1.05 | 3.8 | (3.8, 0.15, 1.175) | 5306, 5307, 5308 |
| 3 | 1.05 | 4.2 | (4.2, 0.15, 1.175) | 5309, 5310, 5311 |
| 4 | 1.05 | 4.6 | (4.6, 0.15, 1.175) | 5312, 5313, 5314 |
| 5 | 1.05 | 5.0 | (5.0, 0.15, 1.175) | 5315, 5316, 5317 |
| 6 | 0.95 | 3.0 | (3.0, 0.15, 1.075) | 5318, 5319, 5320 |
| 7 | 0.95 | 3.4 | (3.4, 0.15, 1.075) | 5321, 5322, 5323 |
| 8 | 0.95 | 3.8 | (3.8, 0.15, 1.075) | 5324, 5325, 5326 |
| 9 | 0.95 | 4.2 | (4.2, 0.15, 1.075) | 5327, 5328, 5329 |
| 10 | 0.95 | 4.6 | (4.6, 0.15, 1.075) | 5330, 5331, 5332 |
| 11 | 0.95 | 5.0 | (5.0, 0.15, 1.075) | 5333, 5334, 5335 |

The position grid is 0.4 m spacing over a 2.0 m span centred on the builder's own `BEAM_X = 4.0`,
chosen **from the scene, before any clip exists**, so it cannot be a position fitted to the
motions — the failure `CLAUDE.md` records for the stepping study's `x = 1.2 m`. The grid is a
design choice, not a measurement: it is stated here rather than justified by a number, and the
campaign additionally reports the full reference-clearance-versus-position curve over the grid so
a reader can see whether the span was wide enough. Both undersides are below the 1.35 m standing
top (`stand_top_m` in `outputs/phase4e_architecture_v2_s8/experiment.json`), so every condition
requires a duck.

**The pool: 36 references.** Three seeds per condition, generated once by the **learned proposer,
uncorrected** — phase4e's `tcn` specification, one generation, no repair — against that
condition's reference scene, then **merged into one shared pool of 36 that every selector sees at
every condition**. The merge is what gives selection a real problem: 3 of the 36 were built for the
condition being tested and 33 were not. The uncorrected learned proposer is chosen because its
committed collision-free rate on the multi-beam corpus is 0.7292 with 0.2708 rejected
(`experiment.json` → `summary.tcn`), i.e. a pool that genuinely mixes clearing and non-clearing
candidates rather than one that is all of either.

**Prefix order for the coverage budgets** (§5): round-robin over conditions by seed index —
`(c0,s0), (c1,s0), …, (c11,s0), (c0,s1), …, (c11,s2)` — fixed here, before generation.

**Seed block: 5300–5555, plus the derived resampling seeds.** Disjoint from every block in
`CLAUDE.md`'s spent list (whose Phase-4 demo/corpus block ends at 5299, and whose highest other
block is EXP-027's 4800–4927):

| use | seeds | n |
|---|---|---|
| shared pool, Experiments A and B | 5300–5335 | 36 |
| validation walking references, V1–V3 | 5340–5347 | 8 |
| Experiment C, paired across all four arms | 5400–5407 | 8 |
| Experiment C, `C2_resample3` derived draws (`s + 10000`, `s + 20000`) | 15400–15407, 25400–25407 | 16 |
| held for a resize on fresh seeds after a recorded refusal (house rule 2) | 5500–5555 | 56 |

The derived resampling seeds are named here so they are reserved rather than silently consumed:
`loop.resample_seeds(s, 3, 10000) = [s, s+10000, s+20000]`. `CLAUDE.md`'s spent-seed list must
gain 5300–5347, 5400–5407, 15400–15407 and 25400–25407 when this campaign runs, and 5500–5555 only
if the resize is used. Physics seed is **0**, one rollout per (reference, condition), as in
EXP-022A and EXP-030.

## 11. Arms, launch plan and budget (added 2026-09-03)

**Executed arms.**

| arm | obstacle in physics | rollouts | why |
|---|---|---|---|
| `validate` | per-motion, V1/V2/V3 poses | 24 | the three gates of §2.4–2.6: one launch of 8 for V1a–V1c at the condition poses, one of 8 for V1d at a 0.30 m blocking underside, one of 8 for V2/V3 at a 1.60 m raised underside |
| `absent` | none | 36 | the completion control that makes a present-arm zero attributable to the obstacle (EXP-030's lesson), and the source of the replay proxy comparison |
| `present` | the condition's beam | 432 | the 36 × 12 outcome table behind Experiments A and B |
| `repair` | the condition's beam | 384 | Experiment C's four arms × 12 conditions × 8 seeds |

Total **876 rollouts** in **31 launches** of ≤ 32 environments (3 validation, 2 absent, 14
present, 12 repair; the last launch of an arm is short). Because the cuboid **size** is identical
in every condition and only the **pose** varies, and the pose is carried per motion inside the
pickle, one launch may mix conditions freely — this is why the present arm packs into 14 launches
rather than one per condition. `table_size` is a launch-level override and is constant at
`[0.24, 2.25, 0.25]`; the CLI `table_position` is recorded but never relied on (§2.1(a)), and V1a
is what proves the per-motion route worked.

**Staged execution with a stop rule between stages.** Stage 0 validation (V1, V2, V3) → Stage 1
the `absent` arm → Stage 2 the `present` grid → Stage 3 the `repair` arms. **Stage 1 is a gate:**
if no reference in the pool completes the local traversal with the beam absent, the remaining 26
launches are not spent (§12, P3). Each stage writes its rows and receipt before the next begins.

**Budget.**

| resource | planned | cap before the campaign stops |
|---|---|---|
| ARDY generations (all v2 sampler) | 8 validation + 36 pool + ≈ 905 Experiment C (expected; 960 maximum) ≈ **949**, at most 1,004 | 1,255 |
| SONIC launches | 31 | 38 |
| SONIC rollouts | 876 | — |
| tracking wall clock | ≈ 26–49 min at the measured 50–95 s per ≤ 32-environment launch, plus host-gate waits | — |
| kinematic scoring | ≈ 1,000 clips at 2–7 s ≈ 35–120 min CPU | — |
| VRAM / host RAM per launch | ≈ 3.8 GiB / ≈ 6.8 GiB measured (`SONIC_MEASURED_NEED`) | gate: ≥ 5,500 MiB free VRAM, ≥ 9,500 MiB available RAM |

Experiment C's expectation uses the committed means: 1 generation for `C1_single`, 3 for
`C2_resample3` (realised 2.99), and 2.72 for `C4_measured2`, with `C3_fixed_crouch2` budgeted at
the same 2.72 for planning and capped at 3.

## 12. Predictions and decision rules, fixed before any sample (added 2026-09-03)

Each carries a threshold, a direction and the consequence of failing, so that no outcome can be
reinterpreted afterwards.

**P1 — the beam spawns where it was asked and stays there (V1).** 8 of 8 environments within
0.01 m / 0.01 rad of their own motion's pose at the first *and* last step; extents within 1 mm;
no neighbour's prop on the route; and, at a 0.30 m underside, 8 of 8 walking rollouts stopped
before the beam's near face. *Consequence if failed:* refusal, campaign stops, nothing is
generated. This is the one item with no committed precedent — every validated spawn so far has
been a box resting on the floor — so it is the first thing run.

**P2 — the horizon reaches the beam and the evaluator can score a success (V2, V3).** With the
beam raised to a 1.60 m underside, ≥ 7 of 8 walking rollouts reach `max root x ≥ 5.0 m` and ≥ 7 of
8 are classified `completed`. *Consequence if failed:* refusal; a negative traversal result would
be uninterpretable, so none is reported.

**P3 — the controller can carry this pool down this route at all (Stage 1 gate).** The `absent`
arm completes the local traversal for **≥ 1 of 36** references. *Direction:* completion with the
beam present is expected to be ≤ completion absent. *Consequence if failed:* the campaign stops
after Stage 1 and reports exactly what it measured — that under this controller no reference in
this pool completes this route even with the beam absent — as a controller-and-pool result at the
"completed tracking run / local traversal completion" evidence levels. Experiments A and C are
then declared unanswerable with this generator–controller pair rather than answered. This is the
campaign's principal risk: the earlier duck study tracked 859 rollouts and **0** passed the last
obstacle, with 554 cut off by the evaluator and only 200 reaching the first beam
(`outputs/exp1b_execution_clearance_v2/receipt.json` → `outcomes`), though under a protocol whose
14 s clip cap §2.3 removes by construction.

**P4 — completion with the beam present.** Predicted **0 completions across all 432 present-arm
trials**, following EXP-030's 0 of 64 at two box heights and the 0 of 859 above. *Consequence if
it does not hold:* any completion is the project's first measured local traversal in the ducking
family and is reported prominently, naming the reference, the condition and the arm, with its
rollout preserved.

**P5 — coverage versus selection (the primary comparison).** If pool coverage at k = 36 is
**0 in all 12 conditions**, the selector comparison is declared **not evaluable** and Experiment A
is reported as a candidate-pool result — the shape the stepping pool already produced, where the
clearing and the tracking-completing sets are disjoint so no ranking function could have
succeeded. If coverage is ≥ 1 condition, the primary comparison is evaluated:
`geometry_and_screen` minus `screen_only` on completion, paired over the 12 conditions, called
**material at ≥ +2 of 12 conditions** with a cluster-bootstrap interval excluding 0 (a
preregistered decision instrument, descriptive and non-inferential — §7), and called a null
otherwise. *Direction:* scene-aware selection is predicted to complete at least as often;
a negative difference of the same size is reported just as prominently.

**P6 — the mechanism, Experiment C stage (i).** `C4_measured2` minus `C3_fixed_crouch2` on
collision-free reference clearance at the specified beam, paired over the 12 conditions:
**material at ≥ +10 pp** with a bootstrap interval excluding 0 (the same preregistered,
non-inferential decision instrument — §7). *Consequence if the interval includes 0:* the paper
says that on this design a constant offset of the same average magnitude did as well as the
measured one, the claim "correction helps because it fixes a systematic error" is answered
**no at this budget**, and the repair contribution is reworded to "a deeper crouch helps" — with
the accept-the-null caveat of §7 attached, never as a demonstration that the two are equivalent. *Consequence if it
holds:* that claim is answered yes, at reference-geometry level only, and stage (ii) decides
whether anything survives tracking.

**P7 — the replay proxy in a second behaviour family.** The `absent` arm's replay-inferred class
— its achieved states scored against that condition's beam — agrees with the physics-measured
class of the **same reference** at the **same condition** on **≥ 80 %** of the 36 references, with
**κ ≥ 0.6** pooled: the same floors EXP-030 preregistered, so the two campaigns are directly
comparable. The two labels come from two different rollouts of one reference under the same
physics seed, exactly as in EXP-030, and the comparison is paired per reference. Reported per
condition (36 pairs each) and pooled over the 12 conditions with a cluster bootstrap; **not a
gate**. *Consequence if failed:* ducking results that rest on replay scoring inherit a stated
caveat naming the direction of the bias, and EXP-030's validation is explicitly scoped to the
stepping family and the floor-box geometry.

## 13. Kill and refusal conditions (added 2026-09-03)

1. **Wrong tracker.** Refuse to launch unless the execution root is
   `/home/linjiw/lucid/GR00T-WBC-exp029`, its HEAD contains `7c63c53`, and an independent source
   check confirms the `add_table`-without-`add_object` fix is present. The legacy checkout is
   refused as an execution root. Record the checkout commit and the full core-source manifest, and
   do not assert equality with EXP-022A's `44e98c45…`.
2. **Validation failure.** Any of V1a–V1d, V2, V3 below threshold: stop, record the refusal
   (`docs/ramp-exp029-refusal-<date>.md`), generate nothing. Never re-run a failed validation on
   the same seeds with a looser threshold; resize on the 5500–5555 seeds in a fresh output
   directory under an amended protocol.
3. **Stage 1 gate.** `absent` completions = 0 of 36: stop after Stage 1, report as P3 states.
4. **Dirty tree, non-empty output directory, wrong sampler.** Refuse on a dirty `git status`, a
   non-empty `--out` for a fresh campaign, or `noise_stream_version != 2`. A refused attempt is
   preserved beside the ledger; resumption happens only through the documented resume path.
5. **Host gate.** Refuse unless `host_gate.SONIC_LAUNCH_GATE` passes; this is a resumable pause,
   never a discarded launch, and concurrent Isaac processes are recorded per launch.
6. **Launch integrity.** Any launch returning non-zero, archiving fewer rollouts than requested,
   or whose log's termination table does not match the release evaluator: stop and preserve. Do
   not re-run a completed arm with different settings. A second failure of the same launch stops
   the campaign for inspection.
7. **Budget overrun.** More than 38 SONIC launches or more than 1,255 ARDY generations: stop.
8. **Pool shortfall.** Fewer than 30 of the 36 pool references produced and verified: stop,
   record the refusal, and resize on fresh seeds from 5500–5555 in a fresh directory.

If a stage is killed during analysis, it is finished with a resume script that re-scores the
archives through byte-identical sources — the `experiments/exp023_prompt_handoff_resume_analysis.py`
pattern — never by regenerating.

## 14. Provenance bound into the receipt (added 2026-09-03)

Persisted **before** the first generation, in EXP-021's `persist()` / EXP-030's ledger order, and
revalidated after every launch: this protocol file's sha256; the project commit and clean status;
the ARDY HF revision `059b8007…` and denoiser sha `0c16ac26…`, the ARDY runtime commit `693f74d`
and its Python source manifest; `g1.xml` sha `5d76cf92…`; the frozen threshold receipt
`outputs/exp016_threshold_calibration/receipt.json` (sha `f6dba8be…`); the tracker checkout
commit, branch and full core-source manifest **as EXP-029's own baseline**, plus the checkpoint
`sonic_release/last.pt` sha `e6bdab3f…` — the vendor's motion-tracking release, as used by every
comparable campaign; the learned proposer directory `outputs/duck_model_v3_m018` and the response
fit it uses; the clip-cache version; the phase4e receipt sha that supplies `C3_fixed_crouch2`'s
constant; the resolved scene configuration dumped from the simulator, including `env_spacing`,
`episode_length_s` and every environment's read-back table pose; the recorded `traversal_eval`
evaluator version; the exact ordered batch plan (row IDs, seeds, conditions, spec hashes, chunk
boundaries) and the exact `actual_ardy_samples` with launched/returned accounting; the
cluster-bootstrap resample count and seed. Rows are written before any SONIC stage, and every
launch record carries its host-gate report.

## 15. What EXP-029 reuses from EXP-030, and what must be built (added 2026-09-03)

**Reused as committed — do not reinvent** (`experiments/exp030_obstacle_present.py` unless noted):

- `require_execution_root`, `read_tracker_fix` / `require_tracker_fix`, `tracker_identity`,
  `bound_tracker_identity`, `resolve_release_bundle`, `core_source_hashes` — the two-checkout rule
  and the campaign's own tracker baseline.
- `build_sonic_command`, `sonic_subprocess_env`, `sonic_launcher`, `shared_overrides` — the
  launcher, which is EXP-022A's with an explicit checkpoint.
- `table_spec` (the cuboid convention: full extents, centre position), `write_arm_motion_pkl`,
  `validate_motion_pkl`, `ensure_motion_pkl` — per-motion table metadata and the deterministic
  export.
- `run_or_resume_launch`, `_write_process_result`, `_validate_process_result`,
  `_check_log_terminations`, `_rollout_check` — the attempt discipline, resume-by-receipt and the
  release-evaluator check.
- `evaluator_version`, `CollisionCache`, `score_rollout`, `_zero_length_record` — scoring, including
  the rule that a rollout with no alive sample still occupies an assigned trial.
- `scene2motion.traversal_eval` (`evaluate_traversal`, `summarise`, `OUTCOMES`,
  `TraversalCriteria`, the EXP-028 fall constants); `scene2motion.host_gate`
  (`SONIC_LAUNCH_GATE`, `SONIC_MEASURED_NEED`, `require_host_resources`);
  `scene2motion.sonic_state_export` (the achieved-state callback, schema v2);
  `experiments/analyze_pool_coverage.py` (the coverage-versus-selected-success decomposition);
  `scene2motion.verify.loop` / `.repair` / `.trace`, `scene2motion.learn.predictor`,
  `scene2motion.optim` (Experiment C's committed arms).
- `exp022.score_trajectory` stays **frozen and untouched**; it is not this campaign's endpoint.

**Must be built, and this is why the campaign cannot run in time:**

1. `experiments/exp029_selection_vs_coverage.py` — the staged, resumable driver, EXP-030 shaped,
   with tests in `tests/test_exp029_selection_vs_coverage.py` and its ledger at
   `outputs/exp029_selection_vs_coverage/`.
2. **Per-motion table poses within one launch.** EXP-030's writer applies one pose to every motion
   in a launch; EXP-029 needs a per-motion mapping (the size stays constant per launch, so only
   the pose generalises), with the validator extended to match.
3. **The spawn-pose read-back for V1a–V1c.** No existing path reads the `Table` prim's pose or
   extents out of the simulation state; EXP-030 inferred physical effect from behaviour instead.
   This is the largest unknown in the build.
4. **The raised-beam case.** Every validated spawn so far is a box resting on the floor. A
   kinematic cuboid floating at `z = c + t/2` is what V1a's last-step check exists to test.
5. **`C3_fixed_crouch2`.** A constant-offset correction beside `verify.repair.repair` — a new
   function, never a modification of the measured one, with a unit test pinning that it reduces to
   a constant `Δq` over the same dilated support.
6. **Beam translation and a pinned goal** in the scene builder path, plus the two-scene split of
   §10.
7. **The six selectors and the analysis** over the 36 × 12 table, including the coverage curves at
   the fixed prefixes.

## 16. Scope, and what this campaign cannot license (added 2026-09-03)

One tracker build, one controller checkpoint, one route, one corridor, one beam geometry, six beam
positions, two beam heights, one physics seed, one rollout per (reference, condition), one motion
prior, one instruction. Rates are within these 12 conditions and are not generalised. A completed
**local traversal** is not navigation and never shares a label with it: passing outside the
corridor is a failure here, and route choice, motion transitions, replanning and recovery are
untouched. `collided_obstacle` is a violation of the conservative replay clearance model measured
under physics with the beam present, not a contact-sensor reading and not the replay of a motion
recorded in an empty world; a violation shallower than the 4 cm margin may not be primitive
contact. A cutoff is the tracker evaluator's stopping rule, not a fall; where a rollout is stopped,
the honest statement is what the last archived state shows. Nothing here licenses a traversal
system, a claim about a second controller, or a transfer of the reference screen beyond the two
families it has been measured on — on ducking it ranks cutoffs only weakly (pooled AUC 0.674
against a speed feature's 0.441, specificity 0.297 at the calibrated 0.20 s operating point,
`outputs/analysis_duck_contract/receipt.json` → `summary.decision.pooled`, `summary.screen`), which
is why it is a ranking feature for selection here and not an accept/reject rule.

## 17. Schedule reality, and what stays open (added 2026-09-03)

**What this protocol does not fix, and why it is still preregisterable.**

- **Whether a raised cuboid holds its pose.** Untested; V1a is the test and its failure is a
  refusal, not a redesign. The design does not branch on the outcome, so nothing about the
  campaign changes depending on what V1a finds.
- **Physical contact attribution.** The table contact sensor is unavailable (§2.1 amendment). The
  endpoint is defined without it, so this is a stated measurement limitation rather than an open
  design choice; if the sensor path is ever built it is a new, versioned analysis.
- **Which `traversal_eval` version scores the run.** Recorded, not asserted (§3, §2.7), following
  EXP-030. Under a version without a completion-time deadline the timeout class is reported as
  "not assessed"; the rest of the endpoint is unaffected.
- **A genuine TEXEDO comparison.** Explicitly out of scope (§4, §8); the arm is defined as a
  screen filter and named so.
- **Whether the learned proposer and the response fit extrapolate to a translated beam.** In-family
  by construction — same builder, width, thickness and height grid — with beam position as the
  independent variable and beam count reduced to one. Reported as scope, not as a gate.

**Timing.** The build list in §15 is several engineer-days, and the campaign is a supervised day
of staged launches on top of it. It cannot be built, validated and run before **2026-09-15**. The
ICRA submission therefore reports this file as a preregistered plan and nothing else; the claims
table rows "scene-aware selection changes and improves the choice", "correction helps because it
fixes a systematic error" and "correcting the reference improves achieved clearance and completes
traversal" stay marked **not run**.
