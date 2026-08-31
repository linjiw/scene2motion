# Guidance received 2026-08-31 — refocus: from audit paper to Scene2Motion-RAMP

Sixth guidance document; the central redirection of the project. Received with the
instruction to treat Table 2 / τ(d) / Kimodo audit as **baseline, regression test and
evaluation infrastructure**, and to make the paper's contribution a method that *actively
expands the prior's usable capability in scenes*. Actioned: the audit-shaped draft
(`paper-draft-v0.md`) is **frozen as the baseline technical report**; new work proceeds under
the RAMP program below. Original review appended verbatim at the end.

## Working summary (EN)

**Diagnosis.** The current draft's engineering evidence is strong but its identity slid back
to the evaluation paper we explicitly did not want: audit first contribution, defect
catalogue third, and §4/6/7/8/9 all about measuring/refusing/reproducing. A reviewer will
summarize it as "a rigorous ARDY interface evaluation plus a heuristic closed-loop repair."
The measurement system is the most mature part; the thinnest part is a method that extends to
multiple traversal behaviors. *The draft writes the diagnosis as the paper and the treatment
as future work.*

**The real problem statement.** u is not an action; it is a soft condition on a stochastic,
phase- and history-dependent generator followed by a tracker:
u\* = argmin_u E_{z,ξ} J(S, T(G_θ(u, z, h))). The scientific problem is **online system
identification and inverse control of a frozen, stochastic, context-dependent generative
motion actuator**. The audit exposed this; the paper must contribute the
response-adaptive inverse controller.

**Strawman risk.** ARDY/Kimodo position themselves as controllable generation / authoring,
not scene planners (CLoSD is the one claiming diffusion-as-planner). Reframe the opening:
priors provide a powerful motion substrate but no scene-grounded action space; the missing
piece is a method converting scene objectives into prior-compatible, executable motion
programs.

**Why generate–verify–repair alone is not the endgame.** Texedo (controller-aware verifier
selection over SONIC, plug-and-play across generators), BRIC (test-time controller
adaptation + signal-space guidance), ReactiveBFM (closed-loop replanning), SafeFlow
(physics-guided generation + safety gating) already cover select/reject/adapt-tracker.
Scene2Motion must show it **modifies a scene-aligned structured motion program based on the
observed generation failure** — and that the current scalar duck repair does not generalize
to which-foot, where-peak, stance-keeping, sidle-vs-tuck, left-vs-right, multi-event
composition, or text placement.

**The method: Scene2Motion-RAMP (Response-Adaptive Motion Programs).**
1. **Event-aligned programs**: per obstacle e_j = (s_j path progress, Δs_j extent, geometry,
   body region, mode m_j, target gait/contact phase φ_j, continuous params a_j). Step-over is
   "the foot in swing phase when the robot reaches s_j peaks its clearance arc at the
   obstacle, stance foot keeps contact" — never "frame 104, left foot +8 cm".
2. **Coherent adaptation residuals**: ΔP = P_adapt ⊖ P_nominal extracted from a real
   adaptation (pelvis vertical, swing-arc, hip/knee/ankle rotations, root-speed, torso/arm
   counterbalance, timing), then transported onto the *current seed's own* phase-matched
   nominal walk with warp W_{γ,τ,ρ} (amplitude, path shift, time scale) — preserving the
   seed's gait style, swing side, velocity, heading, history. MTC (348 traj / 145 scenes) is
   the natural residual source; MotionBricks means "primitives" alone are not novel — the
   novelty is scene-aligned residual programs corrected by realized response.
3. **Response-conditioned repair**: r_k = Φ(S, u_k, x_ref, x_exec) (signed clearance trace,
   collision body/position, achieved dip, widths, swing peak, stance loss, root deviation,
   execution loss, repair history); Δu_k = π_φ(S, u_k, r_k). First a **local response
   optimizer** (paired-stream finite differences → local Jacobian → small QP over ~7 event
   parameters with packet-coherence constraint) as teacher/baseline; then distill into
   **RepairNet**. Key ablation: scene-only vs program-conditioned vs response-conditioned vs
   response-conditioned-without-coherent-projection.
4. **Repairability-aware curriculum** over response complexity (L0 monotone duck → L1
   phase-sensitive step-over → L2 multi-region squeeze → L3 multi-event → L4 discrete+continuous
   → L5 cross-prior), sampling the *repairable frontier* (one-shot fails, optimizer fixes in
   1–2), with D_repair = weighted clearance/phase/contact/execution/mode deficits.

**Four deciding experiments.** E1 residual-packet pilot (duck / step-over / squeeze ×
{synthetic, absolute packet, residual packet, residual+1, residual+2 response repairs};
step-over endpoint = obstacle-centered whole-body box clearance + crossing error + correct
swing foot + stance contact + SONIC success — never foot peak). E2 feedback vs equal-budget
best-of-B (B∈{1,2,3}; RepairNet-3 vs Best-of-3; Texedo makes verifier-selection a mandatory
baseline). E3 ARDY→Kimodo zero-shot **response transfer** (response-conditioned repair should
transfer where open-loop command prediction does not; claim scoped to two released G1
priors). E4 route–body joint planning (C(π) = L + λĴ_adapt + η p̂_exec-fail + κ N̂_calls) —
otherwise call it a local whole-body adaptation planner.

**Claim fixes required in the frozen baseline (and any successor):**
1. "43-D collapses to one axis" → scope to tested request families/sampler/embodiment/metrics.
2. "Teacher is the ceiling" → "this fitted QP response model stayed optimistic under this
   shift; its distilled student couldn't recover information absent from that teacher" — no
   claim that better teachers/response-conditioned learners can't.
3. "Exact collision geometry" → "the robot's simulation collision model with a separately
   measured coverage margin".
4. "Repairs every proposal failure" → report X/Y under fixed budget, never a systemic
   guarantee.
5. "~10⁵ verified traversals/GPU-day" → split throughput tiers: raw generated / kinematically
   scored / accepted records / SONIC-executed; report measured wall-clock, acceptance, GPU
   config.
6. Defect catalogue → concise provenance paragraph + appendix, not a headline contribution.
7. Dataset claim needs **downstream utility** (CLAW/GenTrack already generate data): the
   natural downstream experiment is RepairNet itself — pipeline emits (scene, program,
   response, repair, outcome) transitions; show success ↑ / calls ↓ with data scale.
8. **Statistics**: seeds are nested in scenes — scene is the inference unit. Scene-level
   means + cluster bootstrap (or mixed-effects logistic, scene as random effect); McNemar
   only as auxiliary; conformal calibration split by scene/topology; prefer
   "execution-calibrated acceptance rule" over "certificate".

**Paper skeleton.** Title: *Scene2Motion: Response-Adaptive Motion Programs for Frozen
Humanoid Priors* (alt: *Event-Aligned Adaptation of Frozen Humanoid Motion Priors for 3D
Traversal*); "A frozen motion prior is not a planner" survives only as an intro punch line.
Three contributions: (1) prior-compatible scene motion programs; (2) response-adaptive repair
(optimizer + RepairNet); (3) execution-aware cross-prior planning. Audit → motivating
analysis; defect catalogue/ledgers → appendix; §9 → RepairNet data engine.

**Execution order.** 1) residual-packet pilot (decides: representation bottleneck vs absent
capability); 2) local response optimizer (multi-axis feedback vs best-of-N); 3) RepairNet with
the curriculum; 4) SONIC execution model inside program+route optimization; 5) ARDY→Kimodo
transfer. If residual packets still fail → output-space residual adapter
x_adapt = x_nominal ⊕ A_ψ(S, x_nominal) or per-window latent optimization — keep designing
the stronger method, do not retreat to the audit paper.

**The bar:** the next result that matters is whether **coherent residual + response
adaptation turns placed step-over, lateral squeeze, and composed traversal from failures into
stable, executable successes.**

## Original (verbatim)

真正能把 Scene2Motion 推向强方法论文的中心转变是：从"审计 prior 有哪些可靠通道"，转向"学习如何通过 event-aligned、prior-compatible motion programs 和 realized-response feedback，主动扩展 prior 在场景中的可用能力"。当前 Table 2、τ(d)、Kimodo audit 仍然值得完成，但它们应当成为新方法的 baseline、regression test 和 evaluation infrastructure。接下来最重要的结果，不再是 6×、10×或者又发现了几个 defect，而是：coherent residual + response adaptation 能否把 placed step-over、lateral squeeze 和组合 traversal 从失败变成稳定、可执行的成功。

[Full review as received — sections 一 through 十二 and 最终判断 — preserved verbatim in
`guidance-2026-08-31-ramp-refocus-full.md`; key formulas: u* = argmin_u E_{z,ξ} J(S, T(G_θ(u,z,h)));
e_j = (s_j, Δs_j, g_j, b_j, m_j, φ_j, a_j); ΔP = P_adapt ⊖ P_nominal;
P_target = P_nominal ⊕ W_{γ,τ,ρ}(ΔP); r_k = Φ(S, u_k, x_ref, x_exec);
Δu_k = π_φ(S, u_k, r_k); local Jacobian QP with packet constraint;
D_repair = Σ w_i d_i; C(π) = L + λĴ_adapt + ηp̂_exec-fail + κN̂_calls.
References: ARDY 2607.08741 · CLoSD 2410.03441 · Texedo 2606.22998 · BRIC 2511.20431 ·
MTC 2603.05993 · MotionBricks 2604.24833 · CLAW 2604.11251.]
