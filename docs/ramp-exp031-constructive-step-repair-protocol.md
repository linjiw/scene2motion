# EXP-031 — constructive step repair and the first obstacle-present traversal attempt

**Status:** draft protocol, written 2026-09-03 before any repaired reference was sent to SONIC.
It must be changed to `preregistered` and committed, with the final source hashes and evaluator-v2
commit filled in, before the first tracker launch.  CPU development on the archived references is
explicitly exploratory; no population or traversal claim is made from it.

## 1. The question

Can an obstacle-relative, support-screened edit turn a reference that the frozen SONIC tracker
has previously completed into a reference that also clears and traverses a 5 cm corridor-spanning
box at the fixed scene position?

The operational break is already measured.  In the frozen 64-reference EXP-021 pool, 12
references clear the 5 cm box at `x = 1.2 m`, 11 complete the obstacle-absent tracking run, and
the two sets are disjoint.  EXP-030 then measured `0/64` local traversal completions with that box
present.  Selection cannot join the two sets.  EXP-031 changes the reference.

## 2. The changed component

ARDY-G1, its weights, its generated `qpos`, SONIC, the scene, the start state, the goal, and the
support thresholds remain fixed.  The new component is
`scene2motion.step_repair.repair_step_reference`:

1. Measure each physical foot's forward envelope, bottom clearance and planar speed.
2. At the fixed obstacle, keep a supporting foot on its current side of the margin-inflated box.
3. Lift an overlapping swing foot above the inflated box top plus an 8 mm target buffer.
4. Dilate and smooth those targets over time.
5. Solve bounded six-joint IK independently for each leg and each active frame, then smooth the
   resulting joint deformation with a Gaussian whose sigma is fixed at 1 frame.  Root pose,
   upper body, duration and frame rate are byte-identical to the input.
6. Re-measure whole-body collision and the frozen support rule.  Admit the result only when it is
   collision-free, remains at or below `0.20 s` longest unsupported duration, stays within a
   `0.50 rad` maximum leg-joint edit, and raises no matched joint-time velocity magnitude by more
   than `2 rad/s`.  The raw IK target residual must remain below 5 mm and the post-smoothing
   residual below 25 mm; the latter is reported beside the final geometric clearance.

This is reference-space surgery after generation, not a claim that ARDY's native channels learned
obstacle placement.  Passing the projection is not a dynamic guarantee; obstacle-present rollout
completion is the endpoint.

## 3. Development observation, separated from the experiment

The operator and its numerical settings were developed on the eight EXP-021 references that pass
the frozen support screen.  This makes all reference-level rates on those eight exploratory.
Before any tracker launch, the code-level regression test pins one concrete observation:
`s4434` changes from colliding to whole-body collision-free at 5 cm, retains a longest unsupported
run of `0.20 s`, changes no root or upper-body value, and remains inside the edit/residual budgets.

An exploratory all-eight check found two candidates meeting every pre-execution admission rule:
`s4408` and `s4434`.  This list is recorded before execution and cannot be changed after observing
SONIC.  The other six support-passing references and all 56 screen-failed references remain rows
in the record set with their rejection/refusal reasons; they are not
silently removed from any pool-level denominator.

The controller-consumed float32 arrays are also frozen by content before execution:
`s4408 = 0a12895f270031247a4f89205978b88079a005f8a27b529adb154da7543ccf89` and
`s4434 = b6dc740973ad4404c1b3b736683339e3fecf5490de7a7cd6c9ca8c7b50c0fa3e`, using the
preparation harness's dtype/shape/byte hash.  A changed array is a new method version, not a resume.

Because the historical tracking outcomes of this pool are already public, this pilot can establish
only engineering viability: whether the reference-level bridge can produce any obstacle-present
completion.  A success does not estimate a fresh-pool success probability.

## 4. Pilot design

### 4.1 Scene and controller

- Unitree G1; frozen SONIC release motion-tracking checkpoint.
- Patched obstacle-present tracker worktree and table-spawn path already validated by EXP-030.
- Start `(0, 0)`, goal `(7.2, 0)`, tolerance `0.5 m`.
- Corridor half-width `1.4 m`; walking around is failure.
- Box centre `x = 1.2 m`, depth `0.20 m`, width `2.8 m`, height `0.05 m`.
- One rollout per `(reference, arm)` at physics seed 0 for this engineering pilot.
- Release evaluator plus `traversal_eval` version 2 for outcome measurement.  No time limit is
  introduced; timeout remains not assessed.

The final driver must use EXP-030's patched-checkout identity, table-pose validation, deterministic
motion export, host gate, resumable launch receipts, and achieved-state archive callback.  It must
not compare an arm run on the patched checkout with one run on the legacy checkout.

### 4.2 Paired arms

Every arm contains the same two predeclared reference keys, in the same order:

| arm | reference | obstacle in Isaac | question |
|---|---|---:|---|
| `raw_absent` | original | no | does the pilot reproduce the known executable substrate? |
| `repaired_absent` | repaired | no | did surgery itself destroy controller execution? |
| `raw_present_05` | original | yes | paired obstacle-present baseline under this launch |
| `repaired_present_05` | repaired | yes | primary constructive endpoint |

The existing EXP-030 rows are context, not replacements for these paired control arms.

### 4.3 Outcome hierarchy

For every assigned trial, report independently:

1. reference collision-free at the specified box;
2. passes the frozen support screen;
3. completed tracking run;
4. achieved clearance after tracking;
5. local traversal completion with the obstacle present.

The exclusive outcome label and independent event flags from evaluator v2 are both stored.
Collision, evaluator cutoff, fall, stall, corridor exit and completion are never collapsed.

## 5. Frozen pilot decisions

**P1 — first closure.** `repaired_present_05` produces at least one local traversal completion
among the two assigned references.  A completion must pass the corridor, finish within `0.5 m`
of the goal after passing, remain upright, and have no violation of the 4 cm margin-inflated
whole-body collision model.  This is the project's first constructive milestone, not a rate claim.

**P2 — surgery survival.** For any repaired reference whose `raw_absent` rollout completes,
report whether `repaired_absent` also completes.  No minimum is imposed; a loss directly weakens
the method and is retained.

**P3 — obstacle effect.** Report paired outcome and forward-progress changes from
`repaired_absent` to `repaired_present_05`.  No median is interpreted as an independent-sample
estimate at `n = 2`.

If P1 fails, the result is a failed engineering pilot.  The preserved rows determine the next
mechanism: IK tracking loss calls for a windowed smoothness/velocity projection; remaining achieved
collision calls for achieved-state feedback; early cutoff without either calls for a stronger
controller-conditioned constraint.  The same two primary rollouts are never repeated under relaxed
criteria.  A changed method uses a new version, a fresh output directory and a new protocol.

## 6. Required mechanism ablation before a paper-level method claim

The pilot is intentionally the smallest test of closed-loop viability.  After any successful pilot,
freeze the repair and use fresh references for a preregistered comparison:

| arm | purpose |
|---|---|
| raw reference | frozen-prior baseline |
| longitudinal placement only | tests obstacle-relative foot placement |
| swing-height only | tests clearance without placement repair |
| full foot-envelope IK | tests the combined mechanism |
| equal-budget independent resampling | separates feedback from additional prior calls |

The primary metric is obstacle-present local traversal completion over all assigned fresh trials.
Reference clearance, screening, tracking completion and achieved clearance remain secondary rungs.
The scene, not a frame, is the independent unit once multiple obstacle scenes are introduced;
physics seeds are repeats within a reference/scene condition.

The 20 cm box is a declared stress condition, not part of P1.  The current development operator
does not produce an admitted 20 cm candidate on the eight archived substrates, so launching those
rejected references would only turn a pre-execution failure into wasted simulation.  A 20 cm claim
requires a larger swing-foot workspace or windowed whole-leg trajectory optimization and its own
predeclared comparison.

## 7. Record and dataset fields

All 64 source references receive a record, including screen refusals and projection rejections.
The record binds:

- source campaign, seed, original qpos hash and repaired qpos hash;
- scene geometry and 4 cm body margin;
- support thresholds and longest unsupported run before/after;
- per-foot active window, longitudinal and vertical target magnitudes;
- IK calls, function evaluations, pre/post-smoothing target residuals, matched joint-time speed
  increase, diagnostic acceleration increase, and joint-change budgets;
- whole-body signed clearance and penetration before/after;
- pre-execution terminal state and reason;
- for launched candidates, controller/evaluator identity, physics seed, achieved-state archive,
  event flags, exclusive outcome and local-traversal endpoint.

This is one slice of Scene2Motion-DB: paired failures and repairs with explicit provenance.  It is
not described as training data or a first-of-its-kind dataset until a downstream baseline and a
current literature audit support those claims.

## 8. Launch blockers and provenance

Before changing this document to `preregistered`:

1. The pending evaluator-v2 work in `scene2motion/robot.py`,
   `scene2motion/traversal_eval.py` and its tests must be committed by its owner.
2. The additive `robot.py` change must re-lock both exact-clearance analysers in that same commit,
   as required by `CLAUDE.md`.
3. The EXP-031 driver and tests must land on a clean worktree and bind this protocol's hash, the
   source qpos archive, threshold receipt, G1 XML, operator source, tracker core manifest and
   checkpoint.
4. Repaired qpos and admission rows must be written and hashed before any SONIC launch.
5. The SONIC host gate must pass without relaxation.

No robot experiment has been run under this protocol as of its draft timestamp.
