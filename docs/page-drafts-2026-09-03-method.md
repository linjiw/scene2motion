# Page draft — method and workflow narrative (2026-09-03)

Draft prose for the project page's method section. Vocabulary and evidence levels follow
`docs/project-goal-2026-09-02.md` (framing of record). Every number below is traceable to a
committed artifact under `outputs/`; the receipt path and JSON key for each one is listed in
the final section, **Receipts**. Nothing here is sourced from prose in another document.

**Status this section must preserve.** This is an evaluation contribution plus a correction
method that works on reference geometry. It is not a working traversal system. With the box
actually present in the physics scene, 0 of 64 references completed the traversal.

---

## 1. Start at the obstacle

A box 0.20 m deep and 2.8 m wide sits across a 7.2 m corridor at x = 1.2 m. A Unitree G1
starts at the origin, facing down the corridor. The question the project asks is:

> **Which generated motions can this controller use to get through this obstacle, from this
> starting state?**

That is *local traversal*: pass through the specified corridor, finish beyond the box, no
prohibited contact, no fall. Walking around the box is a failure here, not a clever solution.
Local traversal is not navigation — navigation asks whether the robot reaches a destination,
and going around is often the right answer to that question. The two never share a success
label, and this page only measures the first.

Stating the task from the obstacle rather than from the software matters because the two give
different answers. A text-prompted humanoid motion prior will happily produce a step-over. On
64 draws it produced one that clears a 3 cm box *somewhere along the route* 44 times, and one
that clears a 5 cm box somewhere 37 times. At the position the scene actually specifies, the
5 cm count is 12. Producing the behaviour and placing it are different results.

And the last result in the chain is the one that decides the task. When the box was finally
put into the physics scene and all 64 references were tracked, local traversal completion was
0 of 64 with a 5 cm box and 0 of 64 with a 20 cm box. In the obstacle-absent control arm, run
on the same references at the same time, 1 of 64 completed. That single completion is what
makes the zero attributable to the obstacle rather than to the controller having a bad day.

---

## 2. Driving a released motion model without training it

The generator is NVIDIA's released **ARDY-G1** humanoid motion prior — autoregressive
diffusion, 25 fps, the Horizon52 checkpoint — and it is **frozen**. No weights are trained,
fine-tuned or adapted anywhere in this project. It occupies about 0.93 GB of GPU memory and is
driven entirely through its own published conditioning interface.

That interface is a text prompt plus five channels the model will let you overwrite, frame by
frame, in the feature vector it denoises (`scene2motion/constraints.py`):

| channel | what it expresses |
|---|---|
| `root_2d` | where to go — the pelvis ground path, maskable per frame |
| `root_y_pos` | pelvis height — the duck channel |
| `global_root_heading` | turn or sidle |
| `global_joints_rots` | limb orientation, per (frame, joint) |
| `global_joints_positions` | joint targets — tuck, step-over |

One property of that interface explains a large fraction of the project's findings, and it is
a fact about the released model, not a modelling choice of ours. **The decoder poses the body
by forward kinematics from the rotation channel.** The joint-position channel is not read when
the pose is decoded. A world-height target for a foot is therefore expressible, but it steers
generation only indirectly: the denoiser must produce a feature vector whose position block
matches the request, and nothing forces the rotation block to agree. Every step-over and tuck
this project requested through the position channel went by that indirect route.

The model also ships a streaming interface — `autoregressive_step` — which is what the
released interactive demo uses to change the prompt window by window. We drive it at the
demo's own default: keep the accepted transcript immutable and complete, show each
continuation only its last token (four frames), slice future constraints from the same global
history start, and append only the returned suffix. That is what makes "switch the prompt from
walking to stepping partway down the corridor" an implementable experiment rather than a
thought experiment.

Because nothing is trained, every negative result here is a statement about the **released
conditioning interface**, not about a fine-tune that might have gone wrong. That is the whole
reason the audit is worth reading.

One correction was required before any of it could be measured. ARDY seeds its sampler once
per generation call, so a sample's noise depended on its *position in the batch* — which
silently breaks every matched-pair comparison, since two clips differing only in scene
geometry also differ in their noise draw. `scene2motion/runner.py` gives each batch element
its own advancing noise stream keyed by its own seed (`NOISE_STREAM_VERSION = 2`). Clips
generated before that fix are quarantined and never pooled with clips generated after it.

---

## 3. The measurement chain: from a clip to a record

Six stages turn a scene into one record. Each names the module that does the work, and each
appends fields to the same record rather than replacing it.

**Route layer** — `scene2motion/scenes.py`, `scene2motion/planner.py`. A* over
(x, y, body mode) produces a route and a body-mode preference through the scene. When no route
exists, the planner does not fail silently: it emits a diagnosed refusal naming which
functional clearance was short and by how much.

**Actuation layer** — `scene2motion/program.py`, `scene2motion/constraints.py`. The route plus
a prompt becomes a request in the five channels above: a per-frame pelvis path, optionally a
pelvis-height schedule, optionally sparse joint targets. The request is hashed and recorded, so
"what was the model actually asked" is answerable after the fact.

**Generation** — `scene2motion/runner.py`. Batched, one advancing noise stream per seed,
prompt embeddings served from a cache so the 8B text encoder is never loaded implicitly.

**Whole-body collision check** — `scene2motion/robot.py`. `G1Body` runs MuJoCo forward
kinematics on Unitree's own collision primitives for the G1, inflated by a measured
**`BODY_MARGIN` of 0.04 m**, and reports clearance out to 0.6 m. Contacts are split by contact
normal into *overhead* and *lateral* deficits, because an undifferentiated whole-scene minimum
answers "crouch" to a corridor that is simply too narrow — the first real trace measured a
3-beam scene at 76 mm of total clearance against a 180 mm target while its overhead clearance
was 341 mm. The robot was squeezing past a wall, not grazing a beam.

**Obstacle-centred clearance** — `scene2motion/stepover_eval.py`. `BoxHeightProbe` mutates the
half-height of the single box geom and asks whether the whole trajectory is collision-free
against it, over all scored frames, with the body margin included. Clearance is graded across
0.03–0.30 m rather than reported as one threshold. The centre is **the position the scene
specifies**. A ±r tolerance window — "did it clear the box at any centre within 25 cm" — is a
different quantity, and the receipt that computes it says so in its own label: it lets the
obstacle move to the clip's own lift after the clip has been seen, which the scene does not
grant. Both numbers are published; only the exact-centre one is a placement rate.

**Support test and reference screen** — same module, thresholds frozen in a calibration
receipt before any stepping reference was generated. A foot meets the support test at a frame
when its sole is within 4.65 cm of the ground and its planar speed is below 1.18 m/s;
thresholds were calibrated on 284 clips from the control arms of an earlier study (144 model
references at 25 fps plus the 140 achieved rollouts recorded at 50 Hz). The **reference
screen** rejects a reference whose longest period with neither foot meeting the support test
exceeds 0.20 s. Its prediction target is named and narrow: the controller evaluator's stopping
rule. It is not a fall detector, a contact-force claim, or a statement of physical
impossibility.

**Execution and achieved-state replay** — `scene2motion/sonic_export.py`,
`sonic_state_export.py`. References are exported to the SONIC whole-body tracker's motion
format and tracked in Isaac from a released checkpoint; the achieved robot state is archived
at 50 Hz and replayed against the same collision model that scored the reference. Until
EXP-030 the obstacle was **absent** from the physics scene, so "clearance after tracking" was a
geometry query on recorded states — a robot that stops before the box looks collision-free.
EXP-030 put the box in the scene, which is why its collision counts mean the robot actually
contacted it.

### Why the chain has to keep the levels apart

The chain exists because each stage answers a different question and the answers do not
follow from one another. For one pool of 64 step-prompted references at the specified 5 cm
box:

| evidence level | count |
|---|---|
| produced (clears a 3 cm box somewhere on the route) | 44 / 64 |
| placed (clears the 5 cm box where the scene puts it) | 12 / 64 |
| completed tracking run (no evaluator cutoff) | 11 / 64 |
| clearance preserved after tracking | 0 / 64 |
| local traversal completion, obstacle present in physics | 0 / 64 |

The 12 that clear and the 11 that complete a tracking run are **disjoint sets**. No selection
rule over this pool could have succeeded, whatever it ranked by — which is a candidate
problem, not a selector problem, and the two require different fixes. This is why coverage
(does the pool contain a usable motion at all) is reported separately from selected success
(does the deployed rule find one) everywhere on this page.

---

## 4. Correcting measured errors

When a reference comes back short of clearance, the system does not re-solve the whole
schedule against the same fitted surrogate that just missed. It corrects the surrogate locally
using the measurement that exposed it.

**Measure the deficit where the route is, not where the frame index is.**
`scene2motion/verify/trace.py` reports clearance as a function of route position on the same
64-sample grid the command schedule lives on, so a deficit at sample *i* maps to a command at
sample *i* with no re-indexing. The prior accelerates from rest and settles at the goal; a
frame-indexed trace would put the correction in the wrong place by a variable amount.

**The operator** (`scene2motion/verify/repair.py`):

```
e(s)  = max(0, target − measured_overhead(s))     metres of missing headroom
dq(s) = e(s) / |g′(q(s))|                          command needed to buy it back
dq    = anticipate(dq)                             shifted early, because the body lags
q′    = smooth(clip(q + dq, 0, 1))
```

**Directional gain inversion.** `|g′|` is a **forward** secant over `[q, q+0.12]`, not a
centred one, because the response curve is convex over most of its range and a centred secant
borrows steep gain *behind* the current command that a correction increasing it cannot use.
Measured against the exact inverse for a 5 cm deficit, the forward secant lands within 4 mm of
command over q ∈ [0, 0.7], where the centred one undershoots by up to 27 % — expensive when
the loop stops after two iterations.

**A slope floor, and what it means when it binds.** The gain is floored at 0.15 m per unit
command. Near saturation the fitted gain goes flat, and dividing a 1 cm deficit by a flat
slope demands infinite crouch. Where the floor binds, the correction is deliberately partial:
it moves as far as the measured response supports and the next verification says whether that
was enough. In the 36-scene campaign the floor bound on 0 of 1,925 repair steps, so it is a
guard that did not fire rather than a load-bearing approximation.

**Lag-derived anticipation.** The body is a first-order lag with time constant τ, so a
correction that begins where the deficit begins arrives after the beam. The lead is `3·τ·v`,
derived from the measured τ rather than from the planner's hand-set constant, and applied as a
running maximum over a window that starts early and releases late. The obvious refinement was
tested and refuted: sweeping the lead over 1.5, 3, 5 and 8 τ, 3 and 5 are identical, 1.5 is
worse, and 8 is worse on margin while costing more crouch.

**Monotone by construction.** The deficit is clipped at zero, so the command change is
non-negative, and the anticipation is a running maximum. The correction can therefore only add
crouch where headroom was short; it cannot buy overhead clearance by giving something up
elsewhere. That is checked, not assumed: across the paired comparisons, two corrections on the
learned proposer fixed 9 collisions and broke 0 (McNemar exact p = 0.0039) and fixed 14 margin
failures breaking 0 (p = 0.0001). Repair never regressed a scene.

**Bounded, and three-valued.** Two iterations, then the route is rejected. An unbounded loop
against a stochastic prior converges to whatever the seed happened to do; two attempts is
enough to fix a surrogate error and not enough to launder a route that was never feasible. The
outcome is not pass/fail but `accepted` (collision-free and meets the target margin),
`accepted_margin` (collision-free, below target, budget exhausted) and `rejected`.

**Refusal with a quantified deficit.** Separately from repair, the response model answers
"is this clearance reachable at all?" — and `False` means refuse, not duck harder. A refused
scene is a record, not an absence, and it carries the number: one pilot scene was refused with
`overhead_clearance` short by **0.167 m**, scene value 1.169 m against a best available
1.336 m, at the third partial beam, 9.15 m along the route.

### Where it wins, and where it loses

Paired scene-cluster bootstrap over 36 beam scenes × 8 seeds, two corrections minus
best-of-three resampling, descriptive:

| proposer, endpoint | difference | 95 % interval |
|---|---|---|
| learned (TCN), collision-free reference | **+21.9 pp** | +9.4 to +35.4 |
| learned (TCN), 18 cm margin | **+36.1 pp** | +25.0 to +47.6 |
| QP teacher, 18 cm margin | **+38.5 pp** | +28.1 to +48.6 |
| heuristic, 18 cm margin | **−8.0 pp** | −14.2 to −1.7 |

The learned proposer goes from 72.9 % collision-free to 99.3 % under two corrections, against
77.4 % for resampling under the same three-generation cap. For the heuristic proposer, which
is already 100 % collision-free without help, plain resampling **beats** correction on the
18 cm margin: 72.6 % against 64.6 %. The supported conclusion is the narrow one — measured
correction improves reference clearance for the learned proposer, and does not outperform
resampling for every proposer.

Three boundaries belong next to that table and must not be dropped:

- Equal caps are not equal computation. Both compared arms are capped at three generations,
  but correction actually used a mean of 2.72 and resampling 2.99.
- The learned proposer's 99.3 % collision-free rate is **not** its margin rate, which is
  37.5 %. Those are different endpoints and are reported separately.
- All of this is **reference geometry**. The duck family's tracked endpoint is 0 of 859
  rollouts, of which 554 hit the evaluator's cutoff and only 200 reached the first beam at
  all. Correction improving a reference is not correction improving a traversal, and the
  experiment that would test the difference has not been run.

---

## 5. The record engine

The interesting use of a frozen motion prior here is **as a generator of data, not as a
planner**. The loop above — plan, request, generate, verify against the scene, correct,
screen, execute, replay — emits one record per scene attempt, and the record is the product.

**What a record contains** (`docs/dataset-engine-record-schema.md` is the specification;
this is its shape, not a new one):

- *identity* — scene id, seed, physics seed, sampler version, cache version
- *scene* — full obstacle geometry, route endpoints, preference, requested margin
- *request* — prompt, proposer, repair budget, constraint-spec hash, which channels were written
- *response* — the full-body reference trajectory, archived for every clip including failures
- *clearance traces* — overhead and lateral clearance along the route; exact obstacle-centred
  clearance at each graded height
- *repair* — iterations used, command change per iteration, anticipation lead, whether the
  slope floor bound, and the three-valued outcome
- *refusal* — the located, quantified reason an instance is unsolvable
- *screen features* — 18 reference features including the longest unsupported period, plus the
  screen flag
- *execution* — evaluator outcome, achieved-state archive, corridor passage, finish beyond the
  obstacle
- *provenance* — every hash in section 6 below

**The engine never drops a record.** Refusals and rejections are rows carrying their deficit,
not absences. In the 300-scene pilot, all 300 scenes are accounted for.

**Tiered accounting, never one number.** A single "records per GPU-day" headline is exactly
the number that would mislead, because the tiers differ by two orders of magnitude in cost and
by everything in what they establish.

| tier | ducking | stepping |
|---|---|---|
| generated | 300 scenes in **301.9 s** on one RTX 5080 (heuristic proposer, ≤ 2 corrections) | 64 references (elicitation pool); 128 references (screen test, 1,194 s); 128 samples from a second released prior (1,482 s) |
| kinematically verified | **268** collision-free records — 192 at the 0.18 m target, 76 collision-free below it; 6 rejected; 26 refused with a named deficit | 12 of 64 clear the 5 cm box where the scene specifies |
| screen-accepted | see the launch-budget filter below | 8 of 64 pass the calibrated screen; 33 of 128 in the prospective test |
| tracker-executed | 859 rollouts over 526 references; 0 preserved clearance | 192 rollouts across 6 launches, 25.6–53.0 s per ≤ 32-env launch (255.0 s total); **0 of 64** local traversal completions with the obstacle present |

If a throughput figure is wanted, it must carry its tier and its clock: at the pilot's
measured rate, ≈ 7.7 × 10⁴ **kinematically verified** duck records per GPU-day, extrapolated
from 268 records in 301.9 s. That is not an executed number and must never be quoted as one.

**The screen as a launch-budget filter.** This is the engine's one demonstrated economic use.
Tracking is the expensive tier; the screen is a reference-only computation. On the stepping
pool, the calibrated 0.20 s screen would have skipped every launch the evaluator cut off —
53 of 53 — at the cost of 3 of the 11 that completed a tracking run. The post hoc cut at eight
frames or longer skips 51 of 53 at no cost to survivors, and is reported as a sweep, never as
the operating rule. The prospective test is in flight and honest about its order: 128
references were generated, the screen flag for each was written to a file **before** any
tracker launch (95 flagged at 0.20 s, 81 at 0.28 s, so 33 pass), and the tracking outcomes are
pending. The screen is a predictor of the evaluator's rule; a flagged reference is not a
predicted fall.

**Does the screen generalise?** Two tests, both reported with their weakness. Across a second
*behaviour* family — ducking, 526 references over 36 scenes — the contact feature reaches AUC
0.674 pooled and 0.694 within scene, above chance in all seven strata, while the obvious
confound is refuted: the speed feature scores 0.441. But the transfer is weak, not strong:
specificity at the calibrated threshold is 0.297, and 441 of the 526 duck references are
flagged. Across a second *released model* — an offline, non-autoregressive prior — the two
preregistered rules split. The unsupported period crosses the architecture boundary on a thin
margin (19 of 23 elicited clips exceed the screen, 0.826 against a 0.80 threshold, with a
Wilson interval of 0.629–0.930 that straddles it), and the early-lift timing does not cross at
all (1 of 41 clips with a lift places it inside 2.0 s, against 40 of 49 for the autoregressive
model). Neither model puts the behaviour where the scene asks: 0 of 64 references from the
second prior clear any tested height at the specified position.

**What is claimed.** The schema, the tiers and the records — including the negatives with
their signed deficits and locations. Not "a dataset for training", until a downstream result
exists.

---

## 6. How quality is actually controlled

Every practice below is a house rule the harnesses enforce, and each one is followed by an
artifact showing it actually happening rather than being aspired to.

**Preregister, then generate.** The obstacle-present campaign wrote three predictions with
numeric thresholds and a named consequence for each failure before the tracker ran. All three
held: the control arm reproduces the earlier campaign on 63 of 64 termination flags (threshold
was 58); local traversal completion is 0 of 64 in both obstacle arms; and the obstacle-absent
replay predicts the obstacle-present outcome class on 63 of 64 references — agreement 0.984,
Cohen's κ 0.964, bootstrap 0.882–1.0. The third one is what licenses reading the earlier
obstacle-absent work at all, and it was declared as a falsifiable prediction, not discovered
afterwards. The screen's prospective test committed per-reference flags to a file before any
launch.

**Fail closed, and record the refusal.** The sideways-step positive control was refused at its
own preregistered substrate gate: 0 of 8 sidestep composites observed against a required
minimum of 4. The receipt's status is `refused`, with the reason named. Nothing was rerun on
those seeds with a looser gate; the refusal is the result. The route-phase calibration went
through three preregistrations and two written refusals before it produced a number.

**Planned denominators, so rejecting everything cannot read as success.** Rates are over **all
assigned trials**. The obstacle-present campaign reports 64 assigned trials per arm and 0
rejected before execution, alongside the outcome breakdown — 1 completed / 54 cutoff /
9 stalled with no box; 0 completed / 44 cutoff / 20 contacted the box at 5 cm; 0 completed /
34 cutoff / 30 contacted the box at 20 cm. The cross-prior campaign writes the rule into its
own receipt: a clip that never swept a box has a null outcome there and is counted as not
clearing it, never as a pass.

**Exact obstacle-centred endpoints, not tolerant windows.** The cost-curve receipt publishes
both and refuses to let them be confused. At the specified position the 5 cm count is 12 of
64; the ±0.10 m union is 17 and the ±0.25 m union is 24, and the receipt's own label says the
union is an addressability analysis and never a fixed-obstacle success probability. It goes
further and flags a numerical trap: two integers in the union block coincide with older,
voided figures one height-step away, so any citation must name both the height and the window
radius.

**Provenance bound before generation and revalidated after.** Each campaign binds the
generator's HF revision and denoiser hash, the ARDY runtime commit plus a 54-file source
manifest, the collision model's `g1.xml` hash, the frozen threshold receipt's hash, the
tracker commit and checkpoint hash, the resolved termination configuration, the protocol hash
and a host-resource gate report — and then re-checks after every launch that the project
commit, the tracked diff, the source archives and the tracker identity are all unchanged. Six
launches, six clean revalidations.

**Seeds are never reused.** Spent blocks are ledgered project-wide, and receipts carry an
explicit `seeds_spent_and_must_not_be_reused` list.

**v1 and v2 samplers are never pooled.** The runner pins `NOISE_STREAM_VERSION = 2` and the
harnesses raise if a campaign is launched under anything else, on a dirty working tree, or
into a non-empty output directory. Clips generated under the earlier per-window latent-replay
defect are quarantined as audit evidence.

**Independent adversarial verification.** On 2026-09-03 the manuscript, the figures and the
vocabulary rules were audited against the committed evidence by eight independent auditors,
with every medium or high finding attacked by two further independent verifiers and kept only
if it survived both. **17 findings survived; 49 were refuted.** The refutation rate is the
point — most apparent discrepancies were an auditor misreading a receipt. The survivors
changed real numbers on this page: the like-for-like elicitation figure is 37 of 64 clearing a
5 cm box somewhere against 12 at the specified position (not 44 against 12, which compares
different box heights), and the ±0.25 m union is 24 of 64, not the 20 an earlier draft
carried. A standing table of numbers that do not reproduce from the archives is maintained in
the repository's own orientation file and is treated as authoritative over any prose.

---

## 7. Suggested diagram

**Title:** *From scene to record — and where the records fall out.*

A left-to-right flow in one band, with a second band beneath it showing what leaves the
pipeline at each stage. Render as inline SVG, or as an ordered list with the drop-outs as
indented sub-items if SVG is too heavy.

**Main band** — six boxes, left to right, each labelled with its module:

1. `Route` — A* over (x, y, body mode) · `scenes.py`, `planner.py`
2. `Request` — five writable channels + prompt · `constraints.py`, `program.py`
3. `Generate` — frozen ARDY-G1, per-seed noise stream · `runner.py`
4. `Measure` — whole-body collision vs the scene, 4 cm margin; obstacle-centred graded
   clearance · `robot.py`, `stepover_eval.py`
5. `Correct` — measured deficit → command change, ≤ 2 iterations · `verify/repair.py`
6. `Execute` — released tracker, achieved state archived at 50 Hz, replayed against the same
   collision model · `sonic_export.py`, `sonic_state_export.py`

A short return arrow loops from **Measure** back to **Correct → Generate**, labelled
"≤ 2 corrections". A separate small box hangs below **Measure**, labelled `Screen — longest
unsupported period > 0.20 s`, with its arrow going *into* **Execute** as a filter, not as a
verdict. Label that arrow "launch-budget filter: skipped 53 of 53 cutoffs, cost 3 of 11
survivors".

**Drop-out band** — beneath each stage, one downward arrow into a small grey tray, each tray
labelled with the record class that ends there and, where a number exists, the count:

- under `Route`: **refused** — unreachable clearance, with a located deficit (26 of 300 in the
  pilot)
- under `Correct`: **rejected** — still colliding after the budget (6 of 300)
- under `Measure`: **accepted_margin** — collision-free but below target (76 of 300)
- under `Screen`: **flagged** — predicted evaluator cutoff (56 of 64 stepping references)
- under `Execute`: **cutoff / contacted the box / stalled** (the obstacle-present outcome
  breakdown)

The visual point of the second band is that nothing is discarded: every tray is a row in the
record set, carrying the reason and, where it is measurable, the size of the deficit.

**Optional right-hand rail — the evidence ladder.** Five rungs, bottom to top, each with the
64-reference count beside it, drawn so that no rung's number can be read as another's:
produced 44 → placed 12 → completed tracking run 11 → clearance preserved after tracking 0 →
local traversal completion with the obstacle present 0. Draw the "placed 12" and "completed
tracking run 11" rungs side by side with a small **∅** between them, annotated "disjoint —
no selection rule over this pool could succeed".

**What the diagram must not do:** it must not draw a single happy path from scene to a
completed traversal. There is no such path in the measured evidence, and a diagram that
implies one is the single most misleading thing this page could publish.

---

## Receipts

Every number in the draft above, with the artifact and JSON key it comes from. Paths are
relative to `/home/linjiw/scene2motion/`.

### Obstacle present in the physics scene (§1, §6)

`outputs/exp030_obstacle_present/receipt.json`

| number | key |
|---|---|
| box 0.20 m deep, 2.8 m wide, at x = 1.2 m | `design.obstacle.depth_m`, `.width_m`, `.x_m` |
| corridor half-width 1.4 m, goal 7.2 m, tolerance 0.5 m | `design.scene.corridor_half_width_m`, `.goal_xy_m`, `.goal_tolerance_m` |
| local traversal definition (corridor, beyond, collision-free, upright; not navigation) | `summary.arms.*.local_traversal_completion.definition` |
| no box: 1 completed / 54 cutoff / 9 stalled, 64 assigned | `summary.arms.absent.outcomes`, `.n_assigned_trials` |
| 5 cm: 0 completed / 44 cutoff / 20 contacted, Wilson 0–0.057 | `summary.arms.present_05.outcomes`, `.local_traversal_completion.wilson95` |
| 20 cm: 0 completed / 34 cutoff / 30 contacted | `summary.arms.present_20.outcomes` |
| "contacted" means actual physical contact | `summary.interpretation_guard` |
| 0 rejected before execution, all arms | `summary.arms.*.n_rejected_before_execution` |
| P1 held: 63 of 64 termination flags agree, threshold 58 | `summary.predictions.P1`, `summary.p1_absent_vs_exp022a.termination_flag` |
| P2 held: 0 completions with the obstacle | `summary.predictions.P2` |
| P3 held: agreement 0.984, κ 0.964, CI 0.882–1.0 | `summary.predictions.P3` |
| 192 rollouts requested and returned; 64 archived references reused, 0 new samples | `sonic_rollouts_requested`, `sonic_rollouts_returned`, `reused_archived_ardy_samples`, `actual_ardy_samples` |
| launch times 25.6–53.0 s, 255.0 s total, 32 envs each | `launches.*.elapsed_s`, `.n_rollouts` (sum computed over the six launches) |
| provenance bound: g1.xml sha `5d76cf92…`, body margin 0.04 m, ARDY commit `693f74d`, 54-file manifest | `provenance.project.physical_model`, `provenance.project.runtime.fields` |
| post-launch revalidation clean on all six launches | `post_launch_revalidation.*` |

### Elicitation and placement (§1, §3, §6)

- 44 of 64 clear 3 cm somewhere; 37 of 64 clear 5 cm somewhere —
  `outputs/exp021_elicited_lift_distribution_v2/receipt.json` → `summary.n_clearing["0.03"]`,
  `["0.05"]`
- prompt, route and budget — `outputs/exp021_elicited_lift_distribution/receipt.json` →
  `design.prompt`, `design.conditioning`, `design.budget`, `design.graded_heights_m`
- 12 of 64 clear the 5 cm box at x = 1.2 m; 11 complete a tracking run; 0 do both; 50 never
  reach the obstacle — `outputs/analysis_pool_coverage/summary.json` →
  `results[0].by_height["0.05"].n_reference_clears`,
  `results[0].n_completes_tracking`,
  `results[0].by_height["0.05"].n_reference_clears_and_completes_tracking`,
  `results[0].n_never_reached_obstacle`
- exact-centre counts at x = 1.2 m (13 / 12 / 11 at 3 / 5 / 8 cm); ±0.10 m union 17 and
  ±0.25 m union 24 at 5 cm; the label forbidding a fixed-obstacle reading; the numerical-trap
  warning — `outputs/analysis_exact_centre_cost_curve/receipt.json` →
  `at_exp022a_centres.staged.reference_exact_hits`,
  `tolerant_union.historical_rates_disambiguation.union_counts_by_window`,
  `tolerant_union.label`, `tolerant_union.historical_rates_disambiguation.the_collision`
- 0 preserved clearance after tracking at every graded height —
  same receipt → `at_exp022a_centres.staged.paired_guarded_retention.*.achieved_guarded_clear`

### The screen (§3, §5)

`outputs/analysis_trackability_contract/receipt.json`

| number | key |
|---|---|
| support thresholds 0.0465 m / 1.175 m/s, screen at 0.20 s | `inputs.threshold_receipt.support_height_m`, `.support_speed_mps`, `.max_unsupported_run_s` |
| AUC 0.997 (CI 0.987–1.0) for the longest unsupported period | `summary.exp021_step.single_feature_auc.max_unsupported_run_s` |
| 53 of 53 cutoffs flagged, 3 of 11 survivors flagged (sensitivity 1.00, specificity 0.727) | `summary.exp021_step.gate_0p2s_primary` |
| post hoc ≥ 8 frames: 51 of 53, 0 false flags | `summary.exp021_step.sweep_max_unsupported_run_s` (threshold_s 0.28) |
| 8 of 64 pass the calibrated screen, and all 8 completed their tracking run | `summary.exp021_within_calibrated_gate` = `{"n": 8, "survived": 8}` |
| 0 of the 44 references that clear a 3 cm box pass the screen | `summary.exp021_lift_ge_3cm.n_within_calibrated_gate` |
| 56 of 64 flagged (the diagram's drop-out count) | derived as 64 − `summary.exp021_within_calibrated_gate.n` |
| 53 terminated / 11 survived, with 3 and 0 flagged survivors at the two thresholds | `docs/figures/fig5_numbers.json` → `panel_a` |
| threshold corpus: 284 clips, 144 references + 140 rollouts | `outputs/exp016_threshold_calibration/receipt.json` → `calibration.n_clips`, `corpus` |

Prospective test: `outputs/exp024_reference_contract/receipt.json` → `actual_ardy_samples` 128;
`stages.predict.primary_flagged` 95, `.secondary_flagged` 81 (so 33 pass at 0.20 s);
`predictions.written_before_sonic` true; `stages.analyze.status` `planned`; `wall_clock_s`
1194.1. The per-reference flags are in
`outputs/exp024_reference_contract/predictions.jsonl`.

### Screen transfer (§5)

- ducking: contact AUC 0.674 pooled / 0.694 within scene, speed 0.441, verdict, 526 clips /
  36 scenes — `outputs/analysis_duck_contract/receipt.json` → `summary.decision.pooled`,
  `.within_scene`, `summary.n_clips`, `summary.n_scenes`
- specificity 0.297; 441 flagged of 526 (128 survivors + 313 terminated) —
  same receipt → `summary.screen.calibrated_0p20s`
- above chance in all seven strata — same receipt → `summary.strata.dip_bins.strata[*].auc.contact`
  (3 strata) and `summary.strata.route_classes.strata[*].auc.contact` (4 strata)
- second released prior: 19 of 23 elicited clips above the screen, 0.826, Wilson 0.629–0.930,
  threshold 0.80 — `outputs/exp025_kimodo_cross_prior/receipt.json` →
  `decisions.screen_rule.float_primary_0p20s`, `.threshold_fraction`
- timing does not cross: 1 of 41 inside 2.0 s vs 40 of 49 —
  same receipt → `decisions.timing_rule.definitions.nominal_speed.first_2s`,
  `decisions.timing_rule.ardy_reference.root_crossing`
- 0 of 64 clear any tested height at the specified position —
  same receipt → `summary.arms.step.exact_clearance.staged.*`
- 128 samples in 1,481.9 s — same receipt → `actual_kimodo_samples`, `wall_clock_s`

### Correction (§4)

- forward secant window, slope floor 0.15, `MAX_REPAIRS = 2`, lead `3·τ`, smoothing window —
  `scene2motion/verify/repair.py` (`SLOPE_WINDOW_Q`, `SLOPE_FLOOR`, `MAX_REPAIRS`, `LEAD_TAUS`,
  `SMOOTH_WIN`); derivation and the 4 mm / 27 % comparison, the lead sweep, and the 76 mm /
  341 mm trace — `docs/REPORT.md` §9–§10
- three-valued outcome and the "repaired" contract — `scene2motion/verify/loop.py` docstring
- refuse rather than duck harder — `scene2motion/optim/response.py` `DuckResponse.clears`
- slope floor bound on 0 of 1,925 repair steps —
  `outputs/phase4e_architecture_v2_s8/experiment.json` → `repair_stats.slope_floor_bound_steps`,
  `.n_repair_steps`
- paired differences +21.9 / +36.1 / +38.5 / −8.0 pp with intervals —
  `outputs/analysis_repair_paired_bootstrap/summary.json` → `results[*].paired_difference_pp`,
  `.bootstrap_95_pp`; 36 scenes, 8 seeds, 30,000 resamples, `descriptive_only: true`
- 72.9 % → 99.3 % vs 77.4 %; margin 37.5 % vs 1.4 %; heuristic 64.6 % vs 72.6 %; mean
  generations 2.72 vs 2.99 — `outputs/phase4e_architecture_v2_s8/experiment.json` →
  `summary.tcn.collision_free_rate` 0.7292, `summary["tcn+2"].collision_free_rate` 0.9931,
  `summary["tcn-resample3"].collision_free_rate` 0.7743,
  `summary["tcn+2"].margin_satisfaction_rate` 0.375,
  `summary["heuristic+2"].margin_satisfaction_rate` 0.6458,
  `summary["heuristic-resample3"].margin_satisfaction_rate` 0.7257,
  `summary["tcn+2"].mean_ardy_calls` 2.72, `summary["tcn-resample3"].mean_ardy_calls` 2.99;
  budget caps at `method_specs[*].max_adapted_generations`
- McNemar p = 0.0039 and p = 0.0001, repair never regressed a scene — `docs/REPORT.md` §12
  (Experiment D, same campaign)
- duck tracked endpoint: 0 of 859 preserved clearance, 554 cutoffs, 200 reached the first beam
  — `outputs/exp1b_execution_clearance_v2/receipt.json` → `outcomes.executed_success`,
  `.terminated`, `.reached_first_obstacle`, `.passed_last_obstacle`

### Record engine (§5)

- 300 scenes in 301.9 s; 192 accepted + 76 accepted_margin = 268 collision-free; 6 rejected;
  26 refused; heuristic proposer; ≤ 2 repairs; sampler v2 —
  `outputs/corpus_pilot_v2/receipt.json` → `n_scenes`, `wall_clock_s`, `counts`, `proposer`,
  `max_repairs`, `noise_stream_version`
- the worked refusal (deficit 0.1672 m, scene value 1.1688 m, best available 1.336 m,
  `partial_beam_3` at x = 9.154 m) — `outputs/corpus_pilot_v2/manifest.jsonl`, first row,
  `refusal.*`
- ≈ 7.7 × 10⁴ kinematically verified duck records per GPU-day — extrapolated as
  268 × 86400 / 301.9 from the two keys above; tier and clock stated wherever it appears
- record schema and tier discipline — `docs/dataset-engine-record-schema.md` §2–§3

### Model and interface (§2)

- five writable channels and the forward-kinematics-from-rotations fact —
  `scene2motion/constraints.py` module docstring (which cites
  `ardy/motion_rep/reps/ardy_motionrep.py:284`, `:137`, `:366`, `:373`)
- frozen checkpoint, 25 fps, Horizon52; per-sample noise stream; prompt cache; history crop —
  `scene2motion/runner.py` (`NOISE_STREAM_VERSION`, `_per_sample_noise`, `ArdyRunner.__init__`,
  `.encode`, `.generate`, `.generate_prompt_schedule`)
- ≈ 0.93 GB of GPU memory held by the model —
  `outputs/exp024_reference_contract/receipt.json` →
  `stages.generate.runner_release.cuda_memory_allocated_bytes` 935,241,728
- body margin 0.04 m, clearance reported to 0.6 m — `scene2motion/robot.py` `BODY_MARGIN`,
  `CLEARANCE_MARGIN`
- box probe geometry (0.20 m deep, 2.8 m wide corridor-spanning box; graded heights; margin
  included) — `scene2motion/stepover_eval.py` `step_scene`, `BoxHeightProbe.clears`,
  `.metadata`

### Quality control (§6)

- refused positive control: `status: "refused"`, `refusal_reason:
  "squeeze0_substrate_gate_failed"`, 0 observed against 4 required of 8 planned —
  `outputs/exp023b_prompt_switch_control/receipt.json` → `status`, `refusal_reason`,
  `measurement_gates.squeeze0_substrate`
- denominator rule in the cross-prior campaign —
  `outputs/exp025_kimodo_cross_prior/receipt.json` → `summary.arms.step.coverage.note`,
  `.denominator`, `.rule`
- seeds ledgered as spent — same receipt → `seeds_spent_and_must_not_be_reused`, `spent_seeds`
- sampler pinned to v2 — `scene2motion/runner.py` `NOISE_STREAM_VERSION = 2`
- 8 auditors, 2 verifiers per finding, 17 survived / 49 refuted, and the two corrections it
  forced — `docs/audit-2026-09-03-paper-readiness.md` (header; items 2 and 6)

### Figures already built (read the companion `*_numbers.json` before citing)

`docs/figures/fig2_channel_funnel.png` (channel funnel), `fig3_channel_response.png`,
`fig4_cost_curve.png`, `fig5_contract_gate.png` (the screen), `fig6_screen_across_families.png`,
`fig7_repair_vs_resampling.png`, `fig8_traversal_outcomes.png`.

Two cautions carried over from those files, both of which affect this section:

- `fig6_numbers.json` covers three ARDY corpora plus the duck family — it does **not** include
  the second released prior, so it cannot illustrate the cross-model result in §5.
- `fig8_numbers.json` is the obstacle-**absent** replay (its outcome classes come from a
  geometry query on recorded states), so it must not be captioned with the obstacle-present
  counts in §1. Its own `annotation.goal_region_clips` is the single reference `s4434`, which
  is also the one completion in the obstacle-absent arm of the obstacle-present campaign.
