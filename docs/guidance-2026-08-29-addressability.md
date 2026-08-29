# Guidance received 2026-08-29 — "The project has found its real problem"

Saved verbatim. This is the third guidance document and it re-centres the research object for
the second time. Guidance 1 chose ARDY over Kimodo and reformulated the target as a
distribution over constraint programs; guidance 2 factorised that into route + morphology and
froze the route; this one says the real object is **addressability** — the map from a requested
program to what the frozen prior actually does with it, which is stochastic, low-bandwidth and
only partly controllable.

---

## The discovery

The mapping from a requested program to the resulting body behaviour is stochastic,
low-bandwidth, and only partly addressable:

    c --[ARDY, random seed xi]--> (v, sigma, m)

* `c` requested constraint program
* `v` whether the generated motion is valid and collision-free
* `sigma` realised discrete behaviour (duck / lift / yaw)
* `m` continuous achieved morphology
* `xi` ARDY's sampling randomness

36 requested programs do not produce 36 reliably different behaviours. They collapse:

    36 requested -> 14 valid -> 7 distinguishable -> 3.5 stable strategies/scene

The nominal dimensionality of the API is much larger than the **effective controllable
capability set** of the frozen model.

Tuck was not "slightly weak"; its effect was smaller than the prior's own stochastic width
variation. Foot order was not merely unstable; the old sign-based signature mathematically
forced every clip to receive an order even when no ordering signal existed. Retiring both axes
prevents the benchmark from measuring noise and calling it intelligence.

The three metric failures caught — one-bit signature, hardcoded noise scale, moving
hypervolume box — would all have produced favourable evidence for a learned generator.

## Main recommendation

**Finish EXP-005g exactly as precommitted, without changing its metric or threshold after
seeing the smoke result.**

Before training any generative model, use the 005g outputs to distinguish three very different
bottlenecks:

    candidate support   vs.   candidate selection   vs.   ARDY addressability

The smoke result around 0.65-0.69 recall does not yet tell us which one is responsible.

---

## 1. Treat this as an uncertain quality-diversity problem

The NULL-SEED control shows a program does not possess one fixed descriptor. It possesses a
**distribution over descriptors**: `c -> p(v, sigma, m | S, r, c)`.

The uncertain-QD literature makes exactly this distinction: when descriptors and quality vary
across repeated evaluations, one-off evaluation favours lucky outliers and can create archives
full of apparently novel but unreproducible solutions. It therefore evaluates expected quality,
descriptor distributions, and reproducibility separately. (arXiv:2302.00463)

The primary scientific quantity should not be raw morphology diversity. It should be
**addressable morphology diversity**. For a scene-route pair `x = (S, r)`, a mode `z` is
addressable when some program's LOWER-CONFIDENCE success probability clears a threshold:

    A_tau(x, z) = 1[ exists c :  P_lower(v=1, sigma=z | x, c) >= tau ]

A reasonable initial `tau` is the existing stability threshold, 0.8, fixed before final
testing. Then:

    AddressableCoverage@K = #{reference modes reliably hit by K programs}
                            / #{addressable reference modes}

This is subtly different from realised recall using seed means. A program that hits
"duck + lift" on four seeds and "duck only" on four seeds is not a reliable way to command
either mode, even though its mean descriptor might sit between them.

Keep the current 005g metric unchanged for the gate; add addressable coverage as a stricter
secondary analysis using held-out seeds.

## 2. Add POOL-ORACLE@K before deciding what to learn

Take the union of all candidates proposed by A-KBEST, B-NOGOOD, C-WSWEEP, D-REFINE and
COMPOSITE. Call it `P(x)`. Use the ACTUAL REALISED OUTCOMES to compute the best size-K subset:

    C_K^pool-oracle = argmax_{C subset of P, |C| = K} AddressableCoverage(C)

Not a deployable method — an upper bound that says what kind of learned model could help.

| Full-gate result | Scientific meaning | Correct next model |
| --- | --- | --- |
| Best classical baseline reaches the precommitted 0.90 | Explicit enumeration is enough | No learned proposer |
| Classical stays low but `POOL-ORACLE@8 >= 0.90` | Good programs are already in the pool, the heuristic chooses poorly | Learn a **selector / reranker** |
| `POOL-ORACLE@8 < 0.90` but the full reference has more modes | Candidate pool lacks the needed continuous programs | Learn a **mode-conditioned inverse program** |
| Even the full continuous/reference oracle cannot address the modes reliably | ARDY is the bottleneck | Adapt the prior, or drop the claimed capability |
| Coverage exists but EXP-005h shows no useful trade-offs | Diversity has no downstream decision value | Keep the calibration/planning paper; kill the morphology-generator claim |

Without it, a neural generator might appear to help when all that was needed was a feasibility
classifier choosing among candidates the classical methods already produced.

## 3. Equalise the real budget: ARDY calls, not returned candidates

Verify that COMPOSITE and D-REFINE obtain eight accepted programs using **exactly eight ARDY
generations**. If they generate, reject invalid outcomes, and continue until eight valid
candidates remain, their real budget is larger. The rejected clips must count.

    B_ARDY = total generated clips        (primary budget)
    K_returned = number of accepted outputs   (NOT the budget)

    ValidDiversityYield@B = #{distinct, stable, valid realised modes} / B_ARDY

For every arm retain: CPU proposals considered; ARDY clips generated; MuJoCo checks; invalid
clips rejected; stable valid modes returned; wall-clock.

A method that returns four useful bodies from eight ARDY calls is fundamentally better than one
that returns four useful bodies after generating thirty clips and hiding twenty-two rejections.

## 4. Program-versus-seed allocation control

> Given eight ARDY calls, should we try eight programs once, or fewer programs with several
> seeds each?

Compare `8x1`, `4x2`, `2x4`, `1x8` (programs x seeds per program) at fixed total budget B = 8.
Measure valid success probability; stable mode coverage; useful/Pareto coverage; minimum calls
to reach each coverage level. This separates **program diversity** from **stochastic retry**,
and gives the learned method a proper baseline: it must beat not only eight classical programs
but the best classical allocation between programs and resampling.

## 5. Split candidate selection from held-out evaluation

Because outcome noise is large, the same seeds must not both choose and score a program:

    selection seeds  intersect  evaluation seeds  =  empty

e.g. seeds 0-3 estimate feasibility and choose/refine; seeds 4-7 evaluate stability and
addressability. Otherwise a method appears perfectly feasible simply because it retained
programs that were lucky on the same samples used for reporting. This is especially important
for COMPOSITE and D-REFINE: their feasibility 1.00 is highly meaningful if it transfers to
held-out seeds, and merely post-selection correctness if it does not.

For the paper, report a commanded-versus-realised matrix:

    P(sigma_realised | sigma_requested, scene family)

Rows are the eight requested duck/lift/yaw combinations; columns the stable realised
combinations, with an explicit "inactive/none" state. This may be one of the strongest figures
in the paper because it directly shows what the frozen prior can and cannot be reliably asked
to do.

## 6. Preserve the calibrated distance, clean up its statistical language

The per-channel scale is a q0.99 residual quantile, so do not call it "3 sigma" unless the
distance literally uses a standard deviation. Name it

    s_j = Q_0.99( |m_j - mbar_{c,j}| )

and call distances "calibrated units" / "q99-normalised" / "noise-normalised".

Check whether the residual channels are CORRELATED — duck, yaw and lift can induce correlated
changes in top height, side extent and foot position, and a diagonal distance double-counts the
same fluctuation. Robust alternative:

    d_noise(m_i, m_j) = sqrt( (m_i - m_j)^T Sigma_seed^{-1} (m_i - m_j) )

with a regularised covariance from same-program, different-seed residuals. The current
calibrated metric stays primary for 005g (it was precommitted); covariance-aware distance is a
sensitivity analysis. The result is convincing if both preserve the conclusion.

Also test whether the noise scale is conditional on mode combination, constraint amplitude,
route curvature, scene family, interaction duration. If some groups have 2-3x the variance, use
the global scale for the conservative primary result and group-conditional calibration as a
secondary analysis. Do not silently replace the global metric after observing gains.

## 7. EXP-005h needs one more conceptual correction

Fixing the moving hypervolume box was necessary, but seven objectives create a different risk.
As the objective count grows, Pareto dominance becomes sparse and most candidates become
mutually nondominated; hypervolume becomes increasingly sensitive to normalisation, reference
choice and numerical estimation.

> **Do not make seven-dimensional Pareto-front size or hypervolume the primary evidence that
> morphology alternatives are useful.**

With seven objectives and 8-14 candidates the likely failure is not that the front collapses to
one point. It is that almost everything appears nondominated.

**Hierarchy.** First treat as HARD CONSTRAINTS: goal reached, collision-free, mode stability
>= 0.8. Then aggregate into three interpretable groups:

    f(c) = [ f_safety, f_motion_burden, f_task_preference ]

* `f_safety = -clearance_quantile + lambda * P(invalid across seeds)`
* `f_motion_burden` = posture deviation + motion roughness + contact inconsistency + excess
  joint velocity/acceleration. Until a physics tracker runs, label "foot slip" as **kinematic
  contact inconsistency**, not physical slip.
* `f_task_preference` = integrated duck depth + integrated lift + integrated yaw + adaptation
  duration.

Report three complementary metrics: 3-D hypervolume; set sparsity/coverage; and most
importantly **preference regret**. For a fixed set of predeclared preference vectors W:

    R(C_K) = E_{w ~ W} [ min_{c in C_K} w^T f(c)  -  min_{c in C*} w^T f(c) ]

Preferences: maximise clearance; remain upright; minimise leg lifting; minimise heading change;
minimise overall adaptation burden; maximise smoothness. A body alternative is decision-relevant
when it wins under at least one meaningful preference or robustness condition — not merely
because its descriptor differs by a calibrated distance.

**Precommitted kill condition.** The morphology-diversity story weakens substantially if fewer
than ~25 % of feasible scenes contain at least two stable modes optimal under different
predeclared preferences; or if the K=8 set does not materially reduce normalised preference
regret over the best single neutral program. Accompany with a paired scene-level CI.

**Hypervolume implementation.** Use the reference feasible corpus (not the evaluated arm) for
ideal/nadir; use the same Sobol/QMC points for every arm within a scene; report Monte-Carlo
standard error; do not interpret differences smaller than estimator uncertainty. Common random
numbers are the next step after the fixed-box correction.

## 8. Yaw still has to earn its place

Duck clearly changes the usable vertical envelope. Lift may support low-obstacle traversal,
though its seed noise is extremely large. Yaw is a reliably different orientation only if it
changes something downstream that matters — earlier, sidling alone did not reduce effective
corridor width and sometimes exposed the arm swing in the travel direction.

EXP-005h should explicitly test whether yaw improves any of: minimum clearance; future
manipulation orientation; robustness under asymmetric corridors; tracker stability; semantic
preference satisfaction. If not, yaw is a stable style/orientation variation rather than a
useful traversal capability — an important negative result.

Keep tuck and foot order in the audit as **negative-control channels**, but remove them from
active planning, learned-output dimensions and diversity counts.

## 9. Inference-control sensitivity before declaring the noise intrinsic

The current capability map is for ARDY-G1 Horizon52, 10 denoising steps, the current constraint
guidance setting and postprocessing. ARDY's official implementation exposes the number of
denoising steps and separate text and constraint classifier-free guidance weights, and ships
both G1 Horizon52 and Horizon8 checkpoints.

Do not rerun the full gate or tune on its test scenes. Use a small INDEPENDENT sensitivity suite
of representative duck / lift / yaw / combined / neutral-control programs. Compare: current
config; more denoising steps; stronger constraint guidance; optionally Horizon8 vs Horizon52.
Measure q0.99 seed scatter, `P(sigma_realised = sigma_requested)`, `P(valid)`, motion
roughness/contact consistency, generation latency.

The question is not "can we tune until the paper improves?" It is: **is the apparent low control
bandwidth inherent to the trained prior, or partly an inference-setting trade-off between
natural diversity and constraint adherence?** Three informative outcomes: scatter large across
settings (prior-level non-addressability); stronger guidance reduces noise without damaging
quality (adopt it); stronger guidance improves control but destroys quality (report an explicit
controllability-naturalness frontier — itself a contribution).

## 10. If learning survives, do not start with diffusion or flow matching

The emerging learning problem is much simpler: given a scene, route and requested morphology
mode, predict a program that ARDY will realise validly and stably. Start with a **stochastic
addressability model**:

    g_psi(x, c) -> [ P_psi(v=1), P_psi(sigma | v=1), mu_psi(m), Sigma_psi(m), fhat_psi ]

Train on SEED-LEVEL outcomes, not program means. Select a candidate set by maximising reliable
mode coverage:

    J(C_K) = sum_z w_z [ 1 - prod_{c in C_K} (1 - p_lower_psi(v=1, sigma=z | x, c)) ]
             - lambda * redundancy

Use calibrated LOWER confidence probabilities so unstable programs are not preferred for
occasionally producing a rare mode.

* **Stage A — learned reranking.** Train `g_psi` only to rank the union of classical candidates.
  Correct when `POOL-ORACLE@8` is high. First baselines deliberately simple: logistic
  regression or gradient-boosted trees; a small MLP; then a route-profile encoder if needed. A
  large transformer or diffusion model is unnecessary unless simple models fail.
* **Stage B — mode-conditioned inverse program.** If the pool lacks support, train
  `f_phi(x, z) -> chat_z`; enumerate the small mode alphabet, predict one or several continuous
  programs per mode, score with `g_psi`. Guarantees symbolic diversity by construction and
  concentrates learning on amplitude, onset, offset, anticipation, duration, continuous
  foot/root constraints.
* **Stage C — generative only if necessary.** Justified only if, WITHIN one requested mode `z`,
  the valid program set has multiple disconnected continuous components a deterministic inverse
  or small K-head set predictor cannot cover (e.g. early/shallow and late/deep adaptation both
  useful and disconnected). The current eight-symbol structure does not support that yet.

## 11. The full 005g result should decompose failure, not just report recall

For every missed reference mode, classify:

    symbol missing
    symbol proposed, parameters wrong
    program valid but mode unstable
    requested mode realised but collision occurs
    mode stable but not decision-relevant

which determines the next method: improve enumeration / inverse program or continuous optimiser
/ addressability model, stronger guidance, or prior adaptation / feasibility model / remove from
the target set. Do it per family and macro-average by family. The 30-scene run is appropriate as
a decision gate; the final paper claim should return to the larger 150-scene suite with
completely frozen metrics and code. Use scene-level paired bootstrap intervals, because programs
and seeds inside one scene are correlated.

## 12. Dynamic tracking has become more important

Two reference programs may be distinct and collision-free in exported kinematics yet, after
tracking, converge to the same posture, lose clearance, destabilise, produce different fall
rates, or have very different tracking errors.

    TrackedAddressability(z) = P[ v_dynamic = 1, sigma_tracked = z ]

Start with a small tracker suite: neutral walk, duck, lift, yaw, two combined modes. Independent
ARDY seeds. First without obstacles, then with the scene geometry. The tracker can turn
morphology diversity into genuine decision value — several kinematically valid candidates are
useful when one is dynamically executable and another is not. Conversely, if the tracker erases
or fails most distinctions, that is a clean kill result.

## Recommended experiment sequence

    EXP-005g full gate, unchanged
      -> POOL-ORACLE@{1,2,4,8} + fixed-ARDY-call yield + program-vs-seed allocation
      -> EXP-005h: preference regret + 3-D utility front
      -> small ARDY guidance/step sensitivity
      -> one of { classical capability planner, learned reranker/addressability model,
                  mode-conditioned inverse proposer, targeted prior adaptation }
      -> tracker validation

Do not begin diffusion/flow training before POOL-ORACLE and decision relevance are known.

## The strongest emerging paper framing

Not "a generative model produces multiple humanoid traversal strategies", but:

### **Addressable Motion: Calibrating and Planning with a Stochastic Humanoid Motion Prior**

    nominal constraint dimensions  !=  reliably addressable robot capabilities

Contributions: (1) capability auditing — missing geometry, dead control channels,
sampling-induced false diversity; (2) noise-calibrated morphology — distinguishability relative
to the prior's own stochastic variation; (3) capability-aware planning — body programs only from
empirically addressable modes, with explicit refusal; (4) budget-aware valid diversity — stable,
collision-free morphology coverage per ARDY generation; (5) a learned addressability component
only if it earns its place; (6) physical validation — whether the calibrated morphology set
survives motion tracking.

The headline figure:

    43-D program
       -> 36 requested variants
       -> 14 kinematically valid
       -> 7 distinguishable above decoder noise
       -> 3.5 stable addressable strategies
       -> ? dynamically executable strategies

Directional judgement: a **learned feasibility/addressability model has a plausible
contribution**, while a full generative program model still has to justify itself. The full 005g
result, POOL-ORACLE, and the corrected 005h preference analysis will distinguish those two
possibilities cleanly.

### References cited
* arXiv:2302.00463 — Uncertain Quality-Diversity: evaluation methodology and new methods
* arXiv:2603.04053 — adaptive KKT-based indicator for convergence in multi-objective optimisation
* arXiv:2602.07764 — preference-conditioned multi-objective RL
* github.com/nv-tlabs/ardy — `scripts/interactive_demo/generation.py`, denoising steps and
  separate text / constraint CFG weights, G1 Horizon52 and Horizon8 checkpoints
