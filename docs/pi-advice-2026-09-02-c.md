# PI advice, third review — TEXEDO as a reference point (received 2026-09-02)

Verbatim record. Read against the 16-page TEXEDO v2 paper (arXiv 2606.22998v2), its figures and
tables, and its released selection code. Adopted in `docs/project-goal-2026-09-02.md` (rev. 3),
`docs/paper-draft-v2-2026-09-02.md` (v2.2), `docs/index.html`, and the protocol
`docs/ramp-exp029-selection-vs-coverage-protocol.md`. The retrospective coverage analysis it
motivates is `experiments/analyze_pool_coverage.py` →
`outputs/analysis_pool_coverage/summary.json`.

---

**TEXEDO gives us a useful starting point: choosing a motion the robot can execute is not the same as choosing a motion that solves the obstacle task.**

I read the 16-page v2 paper, including its appendices, checked the key figures and tables, and inspected the released selection code. I used the Xiao-inspired skills to connect its argument to your research questions—not to treat its design choices as rules we must copy.

## 1. What TEXEDO teaches us

Its central idea is straightforward:

1. Generate several candidate motions.
2. Predict how well the fixed controller can execute each one.
3. Measure how well each motion matches the instruction.
4. Filter by predicted execution quality, then choose the best text match.

The dynamics verifier learns from controller rollouts. Neither the generator nor the controller needs to change during selection.

The most instructive result is the tradeoff—not simply the improvement:

| Selector | Tracking success | Text-alignment score |
|---|---:|---:|
| Single generated sample | 87.3% | 5.722 |
| Trackability only, 32 candidates | 99.0% | 4.924 |
| Text alignment only, 32 candidates | 88.5% | 6.110 |
| TEXEDO, 32 candidates | 98.4% | 6.054 |

These are simulation results; text alignment is evaluated with a VLM judge. TEXEDO improves tracking success by **11.1 percentage points** over a single sample, while balancing the competing criteria. It does not win every column.

The research lesson is:

> A good paper can identify two desirable properties that conflict, then show why its decision rule handles that conflict.

For Scene2Motion, the additional property is **completing the scene-specific task**.

## 2. Where our problem becomes different

Consider three illustrative motions:

- A stable duck that ends before the robot reaches the beam.
- A deep crouch that clears the beam in the reference but cannot be tracked.
- A tracked duck that passes beneath the beam and reaches the other side.

All can look relevant to "duck under an obstacle." Only the third completes the traversal.

TEXEDO's verifiers use motion and language, without an explicit scene-geometry input. Its reported hardware results establish execution of 30 selected motions, not a benchmark of obstacle-relative traversal success. That is a scope distinction, not evidence that it cannot be extended.

Our research question should therefore become:

> **Which generated motions can this controller use to get through this obstacle, from this starting state?**

That introduces obstacle position, dimensions, approach state, and the required destination.

One important choice: **navigation and traversal should not share an ambiguous success definition.**

- **Local traversal:** pass through a specified opening or corridor. Walking around it does not satisfy the task.
- **Navigation:** reach a destination through a scene. Walking around an obstacle may be an excellent solution; stepping or ducking is required only when the task or route demands it.

I would establish local traversal first. Navigation later adds route choice, transitions between motions, replanning, and recovery. Successful isolated clips do not establish those capabilities.

## 3. Three experiments that would most benefit Scene2Motion

These are proposed tests—not claims about results we already have.

### Experiment A: Same motions, different obstacle positions

**Question:** Does scene information change which motion should be selected?

Keep the instruction, starting state, controller, and candidate pool fixed. Evaluate that pool against obstacle positions chosen in advance—for example, a beam encountered earlier or later along the same route.

Compare:

| Selector | Information used |
|---|---|
| Random candidate | No ranking |
| TEXEDO-style selection | Predicted tracking quality + text |
| Geometry-only selection | Reference clearance at the specified obstacle |
| Controller + geometry selection | Predicted tracking quality + scene clearance |
| Offline oracle | Observed traversal outcomes for every candidate |

The last row is a privileged diagnostic, not a deployable method.

For the fixed-pool test, a selector receiving identical motion and text inputs has no new information when only the obstacle moves. A scene-aware selector can change its decision. **Whether that change improves traversal is the experiment.**

Evaluate the selected motions with the obstacle present in simulation. Fix the earlier horizon/cutoff problems first, and include a known-trackable positive control.

This would make a strong opening figure: the same candidate motion succeeds in one placement and fails in another. Show actual measured examples, not just an explanatory illustration.

### Experiment B: Is the problem selection—or missing candidates?

**Question:** Does a successful traversal exist in the generated pool?

For increasing candidate budgets, report both:

- **Pool coverage:** how often at least one candidate completes the traversal.
- **Selected success:** how often the deployed selection completes it.

This separates two very different research problems:

- **High coverage, poor selected success:** improve the selector.
- **Low coverage even with a large pool:** selection has little to work with; investigate correction, conditioning, or generation.
- **Clear references but failed executions:** investigate controller compatibility and how tracking changes clearance.

Use nested candidate pools so increasing the budget does not silently change the entire sample set. Define the oracle as the **best observed outcome in that pool**, with its rollout budget disclosed.

This is particularly useful for your stepping results. Before investing in a larger verifier, establish whether useful stepping candidates are present at all.

### Experiment C: Can correction create a useful candidate?

**Question:** Does measured correction help when selection alone is insufficient?

For ducking, compare:

- Additional sampling.
- A fixed extra-crouch adjustment.
- Your measured correction.

Give them the same maximum generation budget and disclose actual generation calls, scoring cost, and elapsed time.

Measure two stages separately:

1. Does correction improve obstacle-relative reference clearance?
2. Does the corrected motion complete traversal after tracking?

A deeper crouch might improve geometry while making tracking harder. That possible loss is scientifically important.

If correction improves traversal, the contribution becomes stronger than "we selected a better-looking motion." If it improves only reference geometry, retain that narrower conclusion.

## 4. Borrow the evaluation discipline—not the success label

TEXEDO defines progress as the fraction of reference frames completed. Its position error is root-relative, and its reported tracking errors are conditioned on successful rollouts. These choices should not be silently reused as navigation measures.

For Scene2Motion, I would prioritize:

- **Traversal completion:** reaches beyond the obstacle within the specified corridor and time limit, without prohibited contact or a fall.
- **Navigation completion:** reaches the goal under the stated route-choice rules.
- **World-space progress and clearance:** not only pose similarity after subtracting root position.
- **Outcome breakdown:** collision, fall, evaluator cutoff, timeout/stall, and rejection.
- **Cost:** generation and scoring latency, executed duration, and interventions.

Report success over **all assigned trials**. If rejection avoids an unsafe attempt, report that benefit separately; rejecting everything must not look like perfect task performance.

Likewise, keep a learned score distinct from a physical guarantee. TEXEDO's paper selects the highest dynamics score when no candidate passes its threshold. For obstacle navigation, I would instead explicitly investigate rejection, resampling, or replanning as task-level alternatives.

## 5. Writing techniques worth adopting

### Start with the mismatch the reader can picture

The strongest opening for your paper is not a list of generators, verifiers, and correction modules. It is a robot performing the right action in the wrong place.

Suggested introduction opening:

> A humanoid may execute a convincing ducking motion and still collide with a low beam if it stands up before reaching the obstacle. Obstacle traversal therefore requires more than a plausible motion or accurate tracking: the robot must produce sufficient clearance at the obstacle's location and continue to the other side. Scene2Motion studies this gap by separating reference clearance, controller execution, and traversal completion.

That gives each later measurement a reason to exist.

### Make experiments answer questions

Organize results around:

1. Do generated pools contain motions suitable for the specified obstacle?
2. Can a selector identify them?
3. Does correction improve the available motions?
4. Does the benefit survive controller execution?

Experiment IDs belong in reproducibility records, not as the reader's main explanation of the paper.

### Make each design choice necessary

Instead of:

> "Our framework contains a scene module, support module, and repair module."

Explain:

> "We measure clearance at the specified obstacle because a motion can clear the same obstacle elsewhere along its route."

Then give the experiment that tests whether this measurement improves decisions.

### Let the result determine the final story

Two outcomes can support different, honest papers:

- **Evaluation contribution:** where generation, selection, and tracking fail for obstacle-relative tasks.
- **Method contribution:** a specific selection or correction mechanism improves actual traversal under defined conditions.

Do not promise the second before measuring it. Keep the current title while the experiments resolve that distinction.

A suitable project goal is:

> **Determine when generated humanoid motions can complete obstacle traversal, and test whether scene-aware selection and measured correction improve completion under a fixed controller and computation budget.**

## 6. Practical cautions before using TEXEDO as a baseline

I found several details worth resolving before running comparisons:

- **Paper versus released selector:** Equation 5 uses threshold filtering followed by semantic ranking. The released CLI instead combines the dynamics score with a normalized semantic distance. Reproduce the paper rule or clearly label the released-code variant; do not assume they are equivalent. (`pipeline/select_best_of_n.py`, pinned revision `8c5beb5e`.)
- **Motion format:** TEXEDO expects 50 Hz, 36-dimensional G1 motions with specified joint ordering and quaternion convention. The ARDY setup in your reviewed draft uses 25 Hz. Preserve physical duration when converting; simply changing the frame-rate label changes the motion. (`docs/FORMAT.md`.)
- **Long trajectories:** the released dynamics scorer defaults to the first 1,024 frames. Check that the relevant obstacle encounter is actually included. (`pipeline/score.py`.)
- **Reproduction versus adaptation:** checkpoints are published, but the reproduction guide says the dynamics-training rollout labels are not included in the public dataset. Using their checkpoint and retraining with your own labels are different experimental conditions. (`docs/REPRODUCE.md`.)

**My recommendation:** begin with the shared-pool, obstacle-position experiment for ducking, after validating the traversal evaluator. It can tell us whether scene-aware selection is sufficient, whether correction is necessary, or whether the candidate generator/controller pair is the main limitation. That answer should guide both the next implementation and the paper's central claim.
