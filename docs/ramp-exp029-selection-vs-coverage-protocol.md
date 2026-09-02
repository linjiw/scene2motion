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

1. **The beam must exist in the physics scene.** Every existing SONIC path in this repository
   tracks with the obstacle absent and replays achieved states against the geometry afterwards.
   A static box can plausibly be spawned through SONIC's `add_table` `CuboidCfg` path; that path
   is **unvalidated here**. Validation: spawn a beam of known height and position, confirm from
   the simulation state that a rollout which should collide does collide and one which should
   pass does pass, and dump the resolved scene configuration into the receipt.
2. **The rollout must be long enough to reach the beam.** The earlier duck tracking study
   (859 rollouts, 0 traversals) was confounded by a 14 s clip cap that stopped most rollouts
   before the obstacle. Validation: the walking control below reaches and passes the beam
   position within the horizon in at least 7 of 8 rollouts.
3. **The traversal evaluator must be validated on a known-trackable positive control.** A plain
   walking reference under a beam high enough not to interfere must be scored as a completed
   traversal. If the evaluator cannot score a success, no negative result from it is meaningful.

A failure at any of these three stops the campaign and is recorded as a refusal; it does not get
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

Pool size N and beam positions M with their exact coordinates; the beam heights; the corridor
width and the prohibited-contact body set; the time limit; the fall bound; the seed block
(**reserve 5300–5555**, disjoint from every block in CLAUDE.md); the SONIC launch budget and the
host-resource gate; the kill conditions; and which of the three gating validations in §2 must
pass before the pool is generated. Every arm's driver lives in `experiments/` and is resumable
by receipt.
