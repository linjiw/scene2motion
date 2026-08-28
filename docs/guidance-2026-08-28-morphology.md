# Research guidance, 2026-08-28: factor traversal into ROUTE and MORPHOLOGY

Verbatim direction from the lead researcher, received after EXP-005e. Recorded because it
redefines the research object and pre-commits several gates.

## The judgment

The EXP-005d correction *strengthens* the project. Two claims were separated correctly:

    traversal success is saturated        BUT      morphological coverage is unresolved

That distinction is now the centre of the project. The ordinary planner is nearly complete
over WHERE the robot goes and structurally incomplete over HOW it configures its body along
that route. That 93.6 % of missed candidates use the same route with a new body realisation is
"exactly the clean factorization we were hoping to discover."

**One further correction before training:** a low raw recall over continuous dip/tuck/lift
samples is *not yet* evidence of missing strategies. It may be evidence that one continuous
morphology manifold has been sampled densely. EXP-005f must distinguish
(1) genuinely different body strategies, (2) continuous variation within one strategy,
(3) seed-induced variation, (4) alternatives with no decision value.

## The new factorization

Stop writing the learned model as `p_phi(C | S, s, g)` — that asks one model to relearn the
route problem, which is saturated. Instead:

    r_{1:T} = RoutePlanner(S, s, g)                     classical, keep it
    b_{1:K} ~ q_phi(b | S, r_{1:T}, l)                  LEARNED, body only
    M_k     = ARDY(r, b_k)                              frozen prior

with `b = {dip, tuck, left/right lift, heading/sidle, onset, offset}`.

The new scientific object is the **feasible body set**

    F_B(S, r) = { b : ARDY(r, b) is collision-free and reaches the goal }

The current A* returns ONE point from it, `b* = argmin_{b in F_B} J(b)`. The research goal is
a small set `{b_1..b_K}` covering the *useful* morphology alternatives at a small budget.

This matches ARDY's architecture: root motion explicit, body in a latent, root predicted
before body, sparse long-horizon constraints on both.

## EXP-005f: bands must not be the primary metric

Fixed bands make the conclusion depend on bin width — a smooth feasible interval
`dh in [0.15, 0.45]` becomes ten "strategies" with ten bands and two with two. Make it
hierarchical.

**Layer 1, discrete morphology strategy.** Achieved ACTIVE-SET signature per obstacle
interaction: `(duck active?, tuck active?, left lift?, right lift?, lift order, large yaw?)`.
Thresholds come from matched-control noise, not round numbers:

    duck active  <=>  dh > Q_0.99(|dh_matched_control|) + m_physical

**Layer 2, continuous within-mode descriptor.**

    m = [dh, dw_L, dw_R, dz_footL, dz_footR, t_lead, t_duration, dpsi]

using the full inflated MuJoCo geometry that survived EXP-000, never ARDY joints.

## Estimate the morphology noise floor before counting anything

For each program b, several fixed per-sample seeds give `mu_b` and `Sigma_b`. Compare
realisations by a whitened distance

    d_morph(b_i, b_j) = sqrt( (mu_i - mu_j)^T Sigma_seed^{-1} (mu_i - mu_j) )

in units of ARDY-noise standard deviations. Require `d_morph > 2..3` before treating a
continuous difference as real. Also report signature stability

    Stability(b) = max_sigma P_z[ sigma(M(b,z)) = sigma ]

with a pre-committed threshold `Stability >= 0.8` for a mode to enter the reference set, and
aggregate signature-flip rate below ~5 %. "A program that produces duck on half the seeds and
duck+tuck on the other half is diverse, but it is not controllably addressable."

## Compare everything against matched controls

    Delta m(b, z) = m(M(r, b, z)) - m(M(r, b_0, z))

same route, same seed, `b_0` the neutral-body program. Without this the signature will
rediscover the confounds EXP-001 already corrected (the false "ducking makes you wider").

## Do not compute recall over raw corpus programs

Raw recall makes a densely-sampled duck region look ten times more important. Instead
deduplicate into an epsilon-net `E_eps(S,r)` and macro-average PER SCENE:

    MorphRecall@K_eps = (1/|E_eps|) sum_{m* in E_eps} 1[ min_{k<=K} d_morph(m*, m_k) <= eps ]

Report three quantities separately: route recall; discrete morphology-mode recall; continuous
within-mode coverage. Keep the old one-bit ducked/not metric as an ablation showing how a
coarse signature falsely implies saturation.

## The real gate is not training — it is BODY-ENUMERATE@K (EXP-005g)

EXP-005e indicts the CORRIDOR EXCLUSION RULE, not classical search in general. A reviewer will
immediately ask why not enumerate different mode assignments at a fixed route. Four baselines:

  A. k-best search over the full (x, y, body mode) graph, grouped by route AFTER search
  B. body no-good cuts: forbid the previous body schedule, or require
     `d_morph(b, b_prev) >= eps`, leaving the route available
  C. weight sweep over `J = w_d J_dip + w_t J_tuck + w_l J_lift + w_c J_clearance`
  D. continuous body-only refinement at fixed route (Sobol / CEM / CMA-ES / coordinate)

Strongest classical baseline = k-best body skeletons + continuous refinement. Compare at equal
**ARDY calls** (K in {1,2,4,8}); report planner CPU separately. Report area under K, not just
K=8.

### Pre-committed interpretation

| Result | Decision |
|---|---|
| BODY-ENUMERATE@8 reaches >=90 % discrete-mode recall AND >=90 % continuous/Pareto coverage | do not claim a learned proposer is needed |
| finds discrete modes, covers the continuous frontier poorly | learn continuous refinement or preference conditioning |
| misses stable, useful body components even at equal K | train a set-valued proposer |
| covers everything but needs far more ARDY calls | learning justified by candidate efficiency |
| covers everything at negligible extra cost | the contribution is the abstraction/benchmark, not the generator |

## EXP-005h: are the missing variants decision-relevant?

Different is not useful. Duck 31/32/33 cm are numerically distinct and scientifically empty
unless they differ in robustness, energy, visibility, tracking, arm use or preference. Use a
multi-objective vector

    f(b) = [L_path, -clearance, E_posture, E_smooth, A_arm_restriction, H_head_reduction, R_tracking]

A candidate is decision-relevant if it is Pareto-nondominated, optimal for some reasonable
preference vector, more robust under geometry uncertainty, more trackable, or uniquely
satisfies a semantic instruction. Report ParetoRecall@K and hypervolume, not count-based
diversity. Also report the paired WITHIN-ROUTE cost difference
`dJ = J(b_miss) - J(b_selected_on_same_route)`, which is more informative than comparing to
the globally best route.

## Three possible outcomes

1. **Several disconnected stable modes** (deep duck / shallow duck+tuck / step-over) ->
   a set-valued model is justified; the paper is about morphological multimodality.
2. **One connected Pareto manifold** (more duck <-> less tuck) -> diffusion unnecessary; a
   preference-conditioned deterministic `b = f_phi(S, r, w)` is cleaner and sampling `w`
   traces the frontier. The paper is about morphology-frontier generation.
3. **Many numerical variants, little utility** -> do not train. The paper is capability
   calibration, conservative whole-body planning, explicit refusal, route/body failure
   decomposition, and the morphology benchmark.

## If learning survives: set predictor first, not diffusion

K learned proposal queries over a shared encoder, Chamfer-style set loss

    L_set = L_coverage + lam_p L_precision + lam_d L_diversity + lam_q L_quality
    L_coverage  = (1/|E|) sum_{b* in E} min_k d_C(b*, b_k)
    L_precision = (1/K)   sum_k min_{b* in E} d_C(b_k, b*)

Move to flow/diffusion only if the K-head collapses, the feasible set has complex disconnected
components, or smooth stochastic interpolation is genuinely valuable — and then apply it to
the BODY program only, never the route and never the full G1 motion.

## Route-aligned scene representation

Do not feed a global occupancy map. Sample a body-clearance TUBE along route arc length s:

    x(s) = [c_top(s), c_L(s), c_R(s), h_low_obstacle(s), kappa(s), v(s)]

and predict body channels as functions of arc length:

    [d(s), t(s), l_L(s), l_R(s), psi(s)]

converting arc length to time only afterwards via the speed profile. This directly protects
against the chord-versus-arc-length bug already found: geometry lives in arc length,
anticipation and execution convert to time afterwards.

## Do not train on raw oracle frequency

There is no natural probability over the oracle's programs; a naive fit learns the sampling
procedure. Build targets from robust feasible candidates, morphology-deduplicated, preferably
the Pareto/quality-diverse subset, balanced across discrete modes:

    q*(b | S, r)  proportional to  1[valid] * w_coverage(b) * w_quality(b)

Call it a **quality-diverse morphology proposer**, not a calibrated probabilistic planner.

## Experiment queue

| exp | question |
|---|---|
| EXP-005f | are the missed body variants stable modes or a densely sampled continuum? |
| EXP-005g | can a classical same-route body enumerator recover them? (`BODY-ENUMERATE@{1,2,4,8}`) |
| EXP-005h | are the missing variants decision-relevant? (Pareto recall, hypervolume) |
| EXP-006a | can a simple learned K-set proposer amortise morphology coverage? |
| EXP-006b | does it generalise rather than memorise the procedural generator? |
| EXP-006c | does geometry change only the necessary body behaviour? |
| EXP-007 | are distinct morphology candidates dynamically useful? |
| EXP-008 | can language select among useful same-route alternatives? |

All programs for one (scene, route) pair must stay in ONE split, or the model sees the same
route geometry in train and test with only a different body label.

## Counterfactual tests matter more now

At fixed route, vary one scene parameter continuously. Lower beam -> `dh_dip` nondecreasing.
Narrower gap -> `dw_tuck` nondecreasing. Higher obstacle -> `dz_lift` increasing until the
prior saturates. Measure monotonicity, onset, duration, change outside the interaction window,
switch threshold, hysteresis. "A model that produces many morphology samples but behaves
non-monotonically under simple geometric perturbations has learned the dataset's sampling
artifacts, not bodily affordances."

## The tracker's stronger role

Same-route morphology diversity now has a dynamical reason: deep duck may destabilise, shallow
duck+tuck may track better, aggressive lift may break swing timing. Measure
`TrackMorphPreservation = 1[sigma(M_tracked) = sigma(M_reference)]` and `TrackedMorphRecall@K`.
"The planner supplies several geometrically valid body realizations so that the tracker-aware
verifier can select one that remains executable" — stronger than diversity for visual variety.

## Framing

> Existing route planners can find where a humanoid should move while collapsing the set of
> ways its body can traverse the same corridor. We factor traversal into route and morphology
> planning and generate a small, quality-diverse set of realized body programs for a frozen
> motion prior.

Provisional title: **Same Path, Different Body: Morphology-Set Planning with Frozen Humanoid
Motion Priors**

## Stated probabilities

- 80 % that the route–morphology factorization is a real and useful finding
- 60 % that a learned body-program proposer beats the current corridor enumerator
- 35–45 % that a full diffusion/flow model is necessary after a strong same-route classical
  enumerator and a simple K-head set predictor
