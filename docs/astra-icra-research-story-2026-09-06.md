# From a clearance audit to a complete robotics paper

September 6, 2026. Research design and claim audit; proposed experiments are not results.

The paper should answer one concrete question: **Can a bounded change to native
motion constraints make a frozen humanoid pipeline clear an obstacle and keep
moving, beyond a strong simple command with the same preview budget?** The present
record identifies this problem convincingly. It has not yet established a broadly
superior obstacle-present correction method.

Our working system is an execution-aware native motion compiler: scene-relative
constraints → frozen ARDY-G1 → frozen SONIC preview → measured clearance and
progress → at most one native correction → exact second preview → select or
abstain. This is an offline compilation procedure with privileged simulated state
and known geometry. It is not online feedback control on a real robot, a learned
tracking transfer function, or a dynamic-feasibility guarantee.

## What is already established

| Question | Evidence and denominator | What the comparison supports | Boundary |
| --- | --- | --- | --- |
| Does a convincing motion imply passage? | EXP-030: 0/64 stepping-present attempts; EXP-031: 0/2 repaired-present attempts | Reference clearance and tracking completion are insufficient proxies | Specific tested carriers/obstacles; no impossibility claim |
| Can native ducking work physically in the simulator? | A5 fixed3/default: 17/32 operational local passages, eight motion groups, one beam geometry | A constructive local passage baseline exists | Geometric endpoint; positive robot-contact sensitivity and full-route completion remain open |
| Can smaller native edits supply better candidates? | A10: fixed shallow raw14/48 versus nominal7/48 on12 reused development scenes; eight gains, one loss | Native depth matters; nominal-preserving selection is a serious comparator | Obstacle-absent geometry/progress qualification, not present traversal |
| Is the full correction better than that comparator? | A12 selected13/48 versus shallow15/48; A15 common-context full4/12 versus shallow4/12, nominal2/12 | Full correction does not establish incremental benefit | A12 cross-context confound is explicit; A15 narrows it |
| Are paired previews comparable? | A13:64/64 replay controls exact; at the observed early onset equal applied actions precede differing post-state. A14:39/39 input invariants and64/64 exact paired replays | Fixing the complete background gives a controlled comparison procedure | No identified physics kernel or global independence guarantee; one useful target per32-env job |
| Does removing phase correction help? | A16: depth-only5/12, shallow/full4/12, nominal2/12; one paired gain, no losses | A narrow development lead, worth a discriminating test | Same12 reused scenes, one u0 each; no significance, present or held-out claim |
| Does the lead extend beyond the pilot? | A17: both distinct candidates fail;35 selected pairs proven equivalent;64 new operational rollouts | Zero selected incremental gain on36 remaining reused units; the frozen rule is closed | Absolute edited-method rates over36 unknown;35 further jobs unexecuted |

Sources: [A10](astra-a10-results-2026-09-06.md),
[A12](astra-a12-results-2026-09-06.md), [A13](astra-a13-result-2026-09-06.md),
[A14](astra-a14-result-2026-09-06.md), [A15](astra-a15-result-2026-09-06.md),
[A16](astra-a16-result-2026-09-06.md),
[A17 result](astra-a17-result-2026-09-06.md) and [protocol](astra-a17-remaining-development-protocol.md).
Counts from different rows must not be added into a single success rate. Operational
rollouts include context workload; they are not independent scientific trials.

A16's useful mechanistic example is s00/u0: retaining depth0.30 m and recovery1.38 m,
but restoring onset1.17→1.08 m, changes inflated clearance +14.94→+10.65 mm and
continuation0.426→0.501 m. It crosses the existing0.45 m progress gate while giving
up4.29 mm of clearance. This supports coupled clearance/progress measurement.
It does not prove a universal phase law or positive forward-velocity constraint.

## Position relative to current primary work

ARDY supplies native long-horizon kinematic constraints and already demonstrates
ARDY+SONIC on G1. We must credit that base capability and identify what our
scene-relative qualification/correction adds. Connecting the two released models
is not our novelty claim. [ARDY, SIGGRAPH2026](https://research.nvidia.com/labs/sil/projects/ardy/).

SONIC supplies general humanoid motion tracking. Our change is above its frozen
controller interface; we have not improved or retrained its tracking policy.
[SONIC official project](https://nvlabs.github.io/GEAR-SONIC/).

CLoSD already couples autoregressive motion diffusion to a tracking controller
with feedback from physics for multiple character tasks. The general idea of
closing the diffusion–physics loop is therefore established. Our candidate
contribution is the narrower frozen-robot, obstacle-relative compilation problem,
including clearance/progress tradeoffs and context-controlled qualification.
That distinction is a research hypothesis, not demonstrated superiority to CLoSD.
[CLoSD paper](https://arxiv.org/abs/2410.03441).

These primary sources establish the main overlap; this is not an exhaustive
novelty search. A direct quantitative comparison requires matching embodiment,
task, observations, trained assets and compute. Until a faithful port is available,
use these systems as related work and name that limitation. Do not put an
unimplemented port in a results table.

## Proposed contributions, with their acceptance conditions

1. **A joint execution qualification problem for frozen humanoid motion priors.**
   Specify obstacle-relative whole-body clearance, corridor passage, deadline,
   complete continuation, and abstention over all assigned tasks. Distinguish
   reference tests, absent previews and obstacle-present execution.
2. **A bounded native correction and selection algorithm.** Keep ARDY and SONIC
   frozen; change only declared native controls. The claimed improvement must beat
   nominal-preserving fixed shallow and an equally budgeted search baseline in
   prospective obstacle-present comparisons. A16 alone cannot support this item
   as an effectiveness claim; A17 can close its present depth-only candidate.
3. **A controlled empirical evaluation and reusable execution records.** Expose
   complete input/runtime identities, aliases, failed launches, rejected tasks,
   paired outcomes, scene grouping and cost. A14 supports the evaluation apparatus.
   Data utility becomes a separate contribution only after a learner experiment.

A suitable eventual title is *Execution-Aware Native Motion Compilation for
Humanoid Obstacle Traversal*. Until constructive superiority is established,
retain the current *Scene2Motion: Evaluating Generated Humanoid Motions for
Obstacle Traversal* framing. A title promising robust all-motion traversal,
contact safety, or policy gains would outrun the present evidence.

## Experiments that close the argument

| Stage | Concrete design and consumed artifact | Primary question / stop rule | Output used by the next stage |
| --- | --- | --- | --- |
| A17 development decision | Frozen references, original slots/backgrounds, common nominal fallback; decisive pair then conditional complete36-unit comparison | Does depth-only exceed fixed shallow? Tie/loss closes this rule; no more tuning these units | Recorded positive or negative method decision, with unknown absolute outcomes retained |
| Contact apparatus | Controlled G1 primitive–beam interactions at known separation, penetration and grazing; positive and negative controls; sensor body/path, time support and impulse units fixed before launch | Do foot/shin/torso contact sensors detect deliberately induced contacts and reject separated controls? Missing positives block contact-free labels | Validated force/impulse/event mapping and geometric/contact confusion table |
| Constructive present pilot | Freeze strongest supported compiler, nominal, shallow and equal-budget search; identical scene/seed/slot contracts; select exclusively from absent previews | Does the selected motion actually pass with the obstacle present and continue? Failed/abstained tasks count against all-assigned passage | Paired endpoint/cost table and final method freeze or negative decision |
| Scene confirmation | Fresh scene groups with preregistered height, beam length and position ranges; reserve geometry combinations and motion seeds; scene-cluster inference | Does improvement survive unseen geometry without retuning? Do not expand a failed frozen method | Held-out single-beam success, failures, uncertainty and clearance calibration |
| Multi-beam transfer | Freeze per-beam scheduler and route endpoint before testing; report local beam passages and complete-route success separately | Does recovery from the first beam support the next? Any unobserved suffix is not a route pass | Route-level generalization evidence |
| Stepping successor | Separate protocol: carrier-specific nominal tracking response, task-space swing/root envelopes, reduced-order support check; preserve open-space and obstacle-present controls | Can an advanced lift survive tracking and cross the box? Opposite carrier lead/lag excludes a universal time shift | Positive physical-simulation stepping evidence or a scoped limitation |
| Optional data utility | Train the same modest learner on raw versus execution-qualified/repaired trajectories; matched training size and compute; scene/group splits before filtering | Does the data improve all-assigned closed-loop performance on untouched scenes? Include execution-filtered-only to separate filtering from repair | A downstream utility claim, if supported |

Exact contact poses, new seeds, scene grids and per-stage budgets belong in separate
executable protocols before any sampling. The table is a research plan, not an
unlimited campaign authorization or a completed benchmark.

A17 failed its strict lead gate. Stop promoting its adaptive rule and carry the
simple native compiler into contact/present validation; the next artifact is the
[A18 contact-sensitivity design](astra-a18-g1-contact-design.md). A replacement
correction must have a new mechanism and a fresh development protocol. One useful
candidate is a route-progress-indexed height schedule with explicitly varied
native forward timing, tested for candidate coverage before fitting a response
model. Merely adding a velocity inequality to a reference trajectory does not
constrain the frozen controller's achieved velocity. A support QP likewise cannot
be called a guarantee without its model assumptions and a predictive validation.

The user’s broad motion/scene objective remains the program direction. A complete
focused paper need not claim every motion family or include an imitation learner.
It does need a clear technical delta, a working evaluated system, a strong simple
baseline, meaningful generalization and costs. If we choose a dataset-utility
paper instead, then downstream utility is central rather than optional.

## Manuscript and deadline plan

The official ICRA2027 call gives September15,2026 as the paper deadline, an8-page
limit including references, and double-anonymous review. Video windows are
August5–September9 and September17–22. These are venue rules, not evidence of
submission readiness. [Official call and submission details](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/).

Allocate the8 pages provisionally: introduction/problem example0.75, related
work0.6, formulation0.65, method1.4, evaluation setup0.8, results/ablations2.0,
limitations/conclusion0.5, references1.3. Lead with an obstacle-present example,
then the algorithm and its strongest comparison. Use experiment IDs only as
receipt references. A main figure should show clearance and continuation together;
a method diagram alone cannot demonstrate the claimed capability.

Immediate work: close A17 and publish the evidence; then validate contact and the
present compiler before opening scene confirmation. By September10, reassess the
actual constructive result and freeze the claim scope. Use September11–14 for
completed tables, reproducible figures, limitations, anonymous manuscript and
video checks; do not invent missing gains to meet the date. Submission remains a
separate user decision. No hardware result, universal safety guarantee or
statistically significant generalization is currently supported.
