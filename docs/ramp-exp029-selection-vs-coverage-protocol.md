# EXP-029 (DRAFT protocol) — shared-pool obstacle-position study: selection, coverage, correction

**Status: draft, not preregistered.** House rule 1 requires arms, seeds, gates, budget and kill
conditions fixed before the first sample; §9 lists what must be settled first. Motivated by the
advisor's third review (`docs/pi-advice-2026-09-02-c.md`, Experiments A/B/C). The retrospective
half of Experiment B for the stepping family is already answered from committed rows by
`experiments/analyze_pool_coverage.py`; this protocol covers the prospective ducking study.

**Schedule reality.** This campaign requires engineering that does not exist yet (§2) and will
not land before the Sep 6 number freeze. It is the experiment that decides whether the paper's
final story is an *evaluation contribution* (where generation, selection and tracking fail for
obstacle-relative tasks) or a *method contribution* (a selection or correction mechanism
improves actual traversal). Until it runs, the paper ships as the evaluation contribution.

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
is the first physical-obstacle evidence in the project.

**Commit pins.** The fork fix is `7c63c53`, preserved on branch **`exp029-obstacle-present`** and
**reverted on `research/practice-utility` by `350cae1`**, because the patched file sits in the
core-source manifest every receipt binds equal to EXP-022A's `44e98c45…`: EXP-028 and EXP-024's
tracked stage must run against the unpatched tracker to stay comparable, and EXP-029 against the
patched one with its own declared baseline. Never compare across that boundary.

**Host RAM, not VRAM, is the binding resource.** A launch that started with 13.6 GiB available
drove the host to 376 MiB and the kernel OOM-killer took an unrelated process. Both probes now
refuse to start below a RAM floor and kill their own launch if available RAM collapses; EXP-029
inherits that guard.

**Not yet verified, and 2.4–2.6 still gate the campaign:** both probe motions were cut off early
and their roots never passed 0.96 m against a box at 1.2 m, so this establishes *physical
effect*, not that a traversal can succeed. Contact attribution is also still to be built: the
table's contact sensor is enabled but the probe does not yet read it, and EXP-029 must record and
report **physical contact with the obstacle**, **violations of the conservative replay clearance
model** (which inflates obstacles by 4 cm and is therefore not the same event), and **prohibited
floor contact** as three separate quantities, never merged.

### 2.2 Risk found while reading: environment spacing versus route length

`manager_env.config.env_spacing` defaults to **2.0 m**, and our route is **7.2 m** long. Because
the obstacle is spawned once per environment at that environment's origin, a robot walking the
full route would cross several neighbouring environment origins and meet **their** boxes. Either
raise the spacing past the route length plus a margin
(`++manager_env.config.env_spacing=12.0` or more), or shorten the route, or run fewer
environments per launch. This must be settled before the pool is generated, and the resolved
spacing recorded in the receipt. Whether neighbouring props actually intrude is inferred from
per-environment prim paths and shared-scene physics; 2.4 measures it instead of assuming.

### 2.3 Risk found while reading: episode length versus time to reach the obstacle

`manager_env.config.episode_length_s` defaults to **10.0 s**. Our references are 200–208 frames
at 25 fps, about 8.3 s, and the robot must still reach the obstacle inside the episode. The
earlier duck tracking study (859 rollouts, 0 traversals) was confounded by a 14 s clip cap that
stopped most rollouts before the obstacle, so this is the same failure mode in a different
constant. Set `episode_length_s` explicitly, record it, and validate with 2.5.

### 2.4 Validation: the box is where we asked, and it collides

Spawn a box of known size at a known position, then, from the simulation state rather than from
the configuration, confirm its pose and extent, confirm that a rollout aimed through it registers
contact on the table sensor, and confirm that a rollout aimed beside it does not. Confirm also
that no neighbouring environment's box lies on the route. Dump the resolved scene configuration
into the receipt.

### 2.5 Validation: the horizon reaches the obstacle

A plain walking reference on the route must reach and pass the obstacle position within the
episode in at least 7 of 8 rollouts, with the obstacle raised out of the way. If it cannot, no
negative traversal result from this protocol means anything.

### 2.6 Validation: the evaluator can score a success

A plain walking reference under a beam high enough not to interfere must be classified as a
completed local traversal by the outcome classifier. If the evaluator cannot produce a success on
a case that plainly is one, its zeros are uninterpretable. The outcome classes and the fall
detector already exist and are preregistered in
`experiments/exp028_termination_free_rollouts.py` (`fall_detection`, `classify_outcome`,
ordering `fell > stalled > walked_through > cleared`, pelvis below 0.50 m or up-axis below 0.70
counting as a fall); EXP-029 extends that classifier with an **observed collision** class from
the table contact sensor rather than replacing it.

A failure at any of 2.4, 2.5 or 2.6 stops the campaign and is recorded as a refusal; it is not
worked around by loosening the endpoint.

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

**Secondary, reported separately, never merged into the primary:**

| measure | definition |
|---|---|
| reference clearance | the reference clears the beam at the specified position, whole-body collision model, margin stated |
| completed tracking run | no evaluator cutoff (the release termination configuration) |
| world-space progress | route distance covered, in metres, not root-relative pose similarity |
| outcome class | success / collision / fall / evaluator cutoff / timeout or stall before the beam / rejected before execution |
| cost | generation calls, scoring seconds, executed rollout seconds, wall clock |

**Reporting rule.** Every rate is over **all assigned trials**. A rule that rejects a candidate
without executing it scores that trial as a non-completion; if rejection avoided an unsafe
attempt, that benefit is reported separately as an outcome class. Rejecting everything must
never read as perfect task performance.

## 4. Experiment A — same motions, different obstacle positions

**Design.** One instruction, one starting state, one route. A single shared pool of N candidate
references generated once. M beam positions along the route, fixed in advance (e.g. early,
middle, late relative to the gait cycle the model produces). Every selector chooses from the
*same* pool for every position; only the obstacle moves.

| selector | information it uses | note |
|---|---|---|
| random | none | lower bound |
| trackability-only | predicted tracking quality | our support screen; ranks by longest period outside the support test |
| text-and-trackability (TEXEDO-style) | predicted tracking quality + instruction match | **not TEXEDO**: our own predictor, their decision shape, see §8 |
| geometry-only | reference clearance at the specified beam position | scene-aware, controller-blind |
| geometry + trackability | both | the combination this project argues for |
| offline oracle | observed traversal outcome of every candidate | privileged diagnostic, not a deployable method; rollout budget disclosed |

A selector that sees only the motion and the instruction returns the same choice when the
obstacle moves; a scene-aware selector can change its choice. **Whether changing it improves
traversal completion is the experiment**, not an assumption.

**Figure this produces:** the same candidate that completes the traversal at one beam position
and fails at another, from measured rollouts, not an illustration.

## 5. Experiment B — coverage versus selected success

Computed on the same rollouts, at nested candidate budgets k = 1, 2, 4, 8, … , N, where the
budget-k pool is a prefix of the budget-2k pool so that increasing the budget does not silently
change the sample set.

- **Pool coverage(k):** fraction of (scene, position) conditions where at least one of the k
  candidates completes the traversal.
- **Selected success(k):** fraction where the deployed selector's choice completes it.

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

Reported in two stages, never merged: (i) does the arm improve obstacle-relative reference
clearance; (ii) does the selected motion complete the traversal after tracking. A deeper crouch
may improve geometry while making tracking harder; that loss is a result, not a nuisance.

**Cost is reported as actual generation calls, scoring seconds and executed seconds**, because
"same maximum number of generations" is not equal computation. In the existing kinematic study
the learned proposer used a mean of 2.72 generations under correction and 2.99 under resampling.

## 7. Statistics

The scene (route + beam position) is the inference unit; rates carry cluster-bootstrap intervals
over scenes, and paired arm differences are per-scene differences with the same bootstrap.
Coverage curves are exact for the realised pool and are labelled as such. Planned denominators:
a missing rollout is a failure. Any analysis chosen after seeing outcomes is labelled post hoc.

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

## 9. To settle before this becomes preregistered

Pool size N and beam positions M with their exact coordinates; the beam heights and thickness;
the corridor width and the prohibited-contact body set; **`env_spacing` and `episode_length_s`,
both of which currently default to values incompatible with a 7.2 m route (§2.2, §2.3)**; the
time limit; the fall bound (the exp028 constants unless changed deliberately); the seed block
(**reserve 5300–5555**, disjoint from every block in CLAUDE.md); the SONIC launch budget and the
host-resource gate; and the kill conditions. All three validations in §2.4–2.6 must pass before
the pool is generated. Every arm's driver lives in `experiments/` and is resumable by receipt,
and reuses `exp022.score_trajectory` and exp028's `fall_detection` / `classify_outcome` rather
than reimplementing the endpoint.
