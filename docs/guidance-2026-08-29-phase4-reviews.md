# Guidance received 2026-08-29 — two reviews of the Phase 4 report

Saved verbatim. This is the fourth guidance document. It arrived after the Phase 4 report
(`docs/REPORT.md` §9–18) and consists of two independent reviews. The project's response and
the resulting Phase 5 plan are in `docs/response-2026-08-29-phase5.md`.

Inline tags such as "NVIDIA Research" and "NYU Shanghai RITS" in Review 1 are source
attributions from the reviewer's tooling, not authorship marks.

---

# Review 1

This is unusually disciplined work — the closed-loop Phase 4 result and the counting-methodology finding are both genuinely exportable, and the claim ledger puts most published robotics evaluations to shame. Before the assessment, one piece of context worth having on the table: ARDY was trained directly conditioned on text labels and kinematic constraints sampled from ground-truth poses, which mechanistically explains your Arm A/B/C result — a constraint pattern that never co-occurs in mocap (one foot 35 cm up, mid-walk, nothing else changed) is out-of-distribution for the conditioning channel itself, so the model honors it literally and completes incoherently, while text indexes whole behavior modes the dataset actually contains. Your "the interface is not the model" finding isn't just an empirical surprise; it falls straight out of how the conditioning was trained. That should reorder your priorities, and mostly you've already reordered them correctly.
NVIDIA Research

## What holds up

Three claims look solid to me. First, the verify→repair→select result (0.750 → 1.000 collision-free OOD, zero regressions, exact McNemar) is the real contribution, and the structural argument underneath it — an imitator can't exceed its teacher, but measurement-driven correction doesn't imitate anything — is crisp and will generalize beyond this system. Your retraining-doesn't-help control is exactly the right falsification attempt and it makes the claim much stronger. Second, the 6× counting delta is a legitimate methodological result on its own; the deepest lesson in the document — a capability-absent claim is only as strong as the best request anyone has tried — is precisely the elicitation problem the LLM-evals community has been circling, and your document demonstrates it six times over in a motor domain. Third, the height-absolute/lateral-relative representation account is a falsifiable mechanism with a clean prediction, correctly flagged as resting on one checkpoint.

## Where I'd push back

There's a buried comparison the report doesn't foreground, and a reviewer will. Put the heuristic row next to tcn+2: the heuristic gets 0.972 collision-free, 0.583 margin satisfaction, 2.0 calls, 36.7 cm dip; your full learned stack plus two repairs gets 1.000, 0.417, 5.3 calls, 42.0 cm dip. The entire Phase 2–3 apparatus nets you one collision in 36 scenes over a heuristic, at 2.65× the compute, worse margin satisfaction, and deeper crouch. The missing arm is obvious and cheap: heuristic + repair. The repair operator corrects from measured geometry, so it should be proposer-agnostic — if wrapping the heuristic reaches 1.000 with fewer repairs and better margins, then verification is the load-bearing component and the proposer barely matters, which is a sharper paper than the one you have. If it doesn't, you've justified the TCN. Either outcome is worth having, and it runs on the existing 36-scene harness.

Second, scope the headline. The funnel characterizes the 43-D constraint interface under one frozen prompt — and your own Arm B is the counterexample showing the model contains at least one more executable strategy reachable by another channel. "A frozen prior delivers 1–2 strategies" invites exactly the generalization the report elsewhere refutes. Every instance of that claim should read "the constraint interface, under the walk prompt."

Third, your certificate isn't a certificate. 23.5% of collision-free plans lack the 86 mm needed to survive tracking error, and the requirement grows with duck depth — so the accept gate in the repair loop is checking the wrong threshold. Make required clearance depth-dependent (base + κ·depth, with κ fit from tracker residuals) and re-emit the entire Phase 4 table under the executable criterion. This is a one-line change to the gate and it converts "collision-free" into something SONIC can actually cash. While you're there, re-run the withdrawn kinematic-refusal→dynamic-refusal correspondence with the corrected step-over gating; if it survives, you get back a cheap pre-filter you shouldn't have had to give up.

Finally, small statistical debts, in your own currency: the 0.375 step-over on 8 seeds carries a Wilson 95% interval of roughly 0.14–0.69, and "zero regressions in 36 paired scenes" only bounds the true regression rate below ~8% by the rule of three. Neither invalidates anything; both belong in the text, given the standard the document sets for itself.

## The program I'd run next

1. Proposer ablation under repair (days). Heuristic+repair and QP+repair vs TCN+repair, same scenes, same harness. This is a system-architecture decision and a claim-sharpening experiment in one, and it's the cheapest high-information move available.
2. Executable acceptance gate (days). Depth-dependent clearance threshold in the verify loop; re-score Phase 4; re-test the refusal correspondence.
3. Locate the step-over amplitude boundary (days). You flag this yourself as the claim most likely to fall next. Sweep ~6 amplitudes in [0.05, 0.35 m], ≥24 seeds per point with per-side gating and contact analysis, pre-registered. Either the boundary is useful (a commandable range) or the 0.375 dissolves — both matter.
4. Map the text channel with your existing machinery (1–2 weeks). This is your Tier 2, and I'd promote it: text prompting is a native, first-class conditioning channel in ARDY, alongside root paths, full-body keyframes, and sparse joint constraints — you'd be characterizing the front door, not a hack. Prompt battery of 20–40 (verb variants, amplitude modifiers, object descriptions), paired seeds, null-calibrated thresholds, and the sharp question exactly as you framed it: is text addressable ("step 12 cm over") or only nominal ("step over")? +56.9 mm with sd 44.9 says the answer isn't obvious in either direction.
NVIDIA Research
5. Probe full-body keyframe conditioning — this is the one channel your plan doesn't list, and I think it may be the fastest route to a third strategy. Your diagnosis of the step-over failure was that the sparse position channel "cannot say the second half" — raise one foot while the other stays planted. A full-body keyframe at the crossing instant can say both halves at once, and it's a documented constraint type. If a single mid-crossing keyframe yields a contact-preserving step-over, you get a spatially placeable step-over without first solving text×space composition — which would partially answer your Tier 2 item 6 for free. Your limits section admits the rotation channel was never built into an action space; this is that experiment, generalized.
6. Text × constraint composition (after 4–5). My prediction: the failure mode will be timing/placement — text names the behavior but the lift lands at a phase uncorrelated with the beam. Measure phase alignment between text-induced lift and obstacle position explicitly; keyframes or root-speed shaping are the candidate rescues.
7. Second skeleton, then scene realism, then n (2–4 weeks). The Core/SOMA replication is the right test of the representation claim. Separately, the funnel's 6 scenes and the beam-corridor hard set are the two external-validity soft spots — push the funnel toward 30+ scenes and add non-ladder geometry (door frames, low tables, a scanned mesh) before anyone else does it to you.
8. Package the audit toolkit exactly as you propose — the acceptance test of reproducing your own wrong answers (10.00 naive, 1.67 calibrated) is the best spec line in the document. On the surrogate question (your item 1): I'd formalize the division you already discovered rather than refit globally. Your repair step dq = e(s)/|g′| is a secant/Newton iteration on realized clearance; the surrogate's only real job is the initial slope. "Teacher proposes, measurement certifies" is a cleaner system statement than a better-fitted teacher. And your repair-count predictor (item 3) is probably a small classifier on first-trace features — 12 of 21 residual failures saturate at the command limit, so the saturation flag alone likely predicts "unreachable" well. Implement the lateral refusal at the route level (deficit > ~32 mm structural cap → prune candidate) so it composes with the learned body-cost selector you already have.

On publication: this is two papers. The systems paper — closed-loop verification and repair over a frozen motion prior, with the proposer ablation as the clincher — fits CoRL/RSS, with LangWBC, Harmon, UniAct, MaskedMimic, and CLoSD as the comparison set for the text-channel piece. The methods paper — the 6× counting result, the elicitation lesson, the packaged audit — stands alone and will travel beyond robotics. Keep the claim ledger as an appendix in both; it's the most trust-building artifact you have. And note the timing works for you: ARDY shipped in July and SONIC was open-sourced in early 2026 as part of the GR00T whole-body control platform, so a rigorous audit-then-close-the-loop treatment of this exact stack has a window where it will be the reference point rather than a follow-up.
NYU Shanghai RITS

If I had to compress all of this to two sentences: the crux engineering experiment is heuristic+repair, because it decides whether anything learned belongs in your deployed system; the crux research question is whether ARDY's text and keyframe channels are addressable and spatially composable, because that's the only observed door out of a 1.5-strategy action space.

---

# Review 2

## Overall judgment

This is now **real research rather than a sequence of demos**. The strongest evidence is not the 79 commits or 34,000 generations; it is that the project repeatedly invalidated its own preferred explanations, preserved negative results, and eventually discovered that the central problem was not "learning a better duck schedule" but **measuring the behavior actually produced by a stochastic, only partially addressable motion prior**.

My central assessment is:

> **Phase 4 is scientifically meaningful, but the project is not yet a convincing general "scene-to-whole-body-motion" system. It is currently a strong study of how to audit and safely exploit a frozen motion generator, demonstrated mainly through crouching under obstacles.**

As a narrowly framed paper, it is already credible. As a broad Scene2Motion paper claiming general obstacle traversal, it needs one more focused phase: **unlock at least one second robust behavior through text–constraint composition, and move the final certificate from generated kinematics to tracked execution.**

The Claude preview itself did not load for me, so this assessment uses the complete status report you pasted, together with the official ARDY, SONIC, and related-work sources.

## What you have genuinely discovered

### 1. The capability-audit methodology is a real contribution

The 6× gap between a naive capability count and the paired, noise-calibrated, multi-seed count is not merely an implementation anecdote. It exposes a common but serious error in evaluating generative robot priors:

* one generated clip is mistaken for a repeatable capability;
* any visible displacement is mistaken for command responsiveness;
* failed generations are silently removed;
* stochastic variation is mistaken for controllability;
* kinematic validity is mistaken for executable capability.

Your null-controlled threshold, paired seeds, stability gate, nested funnel, and raw candidate ledger form a coherent evaluation protocol. ARDY's own paper demonstrates broad text and kinematic controllability, but its quantitative evaluation centers on generated-motion quality, joint-constraint error, trajectory error, and foot skating—not scene collision safety or downstream physical execution. Your audit therefore asks a materially different robotics question.

The right claim is not yet "all frozen priors are overestimated by 6×." It is:

> **For ARDY-G1 under this tested program family, commonly used single-seed and fixed-threshold evaluation choices overestimate repeatably addressable robot behaviors by approximately 6×.**

To turn that into a general methodology paper, replicate the result on a second generator.

### 2. Generate → verify → repair is the first part that actually solves the deployment mismatch

The most important conceptual move in Phase 4 was refusing to treat the requested schedule as evidence. You measure the generated clip against the scene, separate overhead from lateral contacts, and apply a bounded correction based on the observed deficit.

That changes the system from scene → planned command → hope, to scene → command → generated motion → measured residual → repair or refusal.

The zero-regression paired result against the one-shot TCN is strong. It also explains why adding imitation data did not help: the TCN learned the teacher more accurately, while repair obtained new information from the actual generator output.

This distinction is increasingly important because closely related work generally resolves generation–execution mismatch through training. RLPF and PhysMoDPO use physical feedback to post-train motion generators, while other recent systems keep a generator frozen but fine-tune the tracker or train a perception-conditioned controller around it. Your distinctive position is **weight-frozen, black-box, per-instance adaptation through measurement**, which is genuinely different.

### 3. "The tested interface is not the model's behavior support" is probably your highest-upside finding

The step-over prompt result is more important than the Phase 2 CNN or Phase 3 QP:

* a manually imposed high foot target produced literal height compliance but destroyed contact;
* a semantic prompt produced a smaller lift, but preserved a recognizable coordinated behavior;
* therefore, failure through one conditioning channel does not imply that the learned motion distribution lacks the behavior.

I would phrase the result slightly differently from "the interface is not the model":

> **The behavior support of a motion prior is larger than the set that is stably addressable through any single control channel.**

That gives you a principled reason to combine channels: text specifies the coordinated behavior manifold; kinematic constraints specify location, timing, direction, and geometry; verification determines whether the requested composition actually survived generation.

That is a much stronger research direction than learning another schedule network.

### 4. The project has identified where learning helps and where it does not

The negative result from the TCN is useful because it separates two kinds of amortization: learning an output from a flawed teacher reproduces the teacher's ceiling; learning an expensive but valid quantity, such as route cost, can be useful if the labels reflect the true downstream outcome.

That second qualification matters. Your current route oracle is QP-on-every-route, but the QP response model is known to become badly optimistic out of distribution. Therefore, agreement with that oracle does not yet demonstrate agreement with the **actual ARDY-generated and verified route ranking**. The route ranker may still be useful, but it needs to be relabeled or validated using realized motion costs.

## The major weaknesses reviewers will notice

### 1. The repaired TCN does not currently beat the strongest practical baseline

This is the most urgent issue.

On the 36-scene hard set:

| Method          | Collision-free | Meets 18 cm | Peak dip | ARDY calls |
| --------------- | -------------: | ----------: | -------: | ---------: |
| Heuristic       |          35/36 |       21/36 |  36.7 cm |        2.0 |
| TCN + 2 repairs |          36/36 |       15/36 |  42.0 cm |        5.3 |

The repaired TCN wins exactly one additional collision-free scene, while losing on margin attainment, crouch efficiency, and query cost. Its improvement over **one-shot TCN** is statistically persuasive, but superiority over the **best existing method** is not established.

A reviewer will immediately ask: why did you repair the TCN rather than repair the heuristic?

You need the following comparison before doing anything more elaborate: heuristic; heuristic + one repair; heuristic + two repairs; QP + repairs; TCN + repairs; all under identical query budgets and seed policies.

It is quite possible that heuristic + one repair will dominate the current Phase 4 method. That would not destroy the paper. It would simplify and strengthen it: the contribution would become the verifier and repair loop, not the learned scheduler.

Also, write "36/36 observed successes," not an unqualified "100% reliable." With 36 successes, the two-sided exact 95% lower confidence bound on the underlying rate is only about **90.3%**.

### 2. "Whole-body obstacle traversal" currently overstates the demonstrated capability

The robust capability is crouching. Step-over is an emergent, weakly executable result at one amplitude, and lateral narrowing is mostly a detection-and-refusal problem.

So the current system has: one robust continuously controllable adaptation (ducking); one preliminary discrete behavior (small step-over); one mostly unavailable behavior (narrowing/squeezing); route-level avoidance when body adaptation is unavailable.

That is not yet a general whole-body traversal repertoire. HumanoidPF and Perceptive Humanoid Parkour already demonstrate crouching, squeezing, hurdling, climbing, and long-horizon perceptive traversal, including real-world deployment. Your comparative strength is not breadth; it is exploiting an arbitrary frozen prior without training it for the environment.

Until step-over becomes stable, I would use language such as "scene-conditioned clearance adaptation with a frozen humanoid motion prior" rather than "general whole-body obstacle traversal."

### 3. The 43-dimensional audit is not an audit of the entire ARDY constraint interface

ARDY officially supports: root paths and waypoints; sparse joint positions; sparse joint rotations; full-body keyframes; arbitrary combinations of those signals.

Your own limits section says the rotation channel never became a systematic action space. Therefore, the result should be described as a capability audit of the **tested root-and-position program family**, not of the entire kinematic interface.

This matters especially because the final pose is rotation-driven through forward kinematics. A step-over may be much more naturally preserved by sparse hip/knee/ankle rotations or full-body keyframes than by commanding a single foot position.

### 4. "Height works because of the representation" is plausible, but not yet causally established

The representation analysis is insightful, but there is another explanation at least as plausible:

> **The successful commands are on or near the learned coordination manifold, while the failed commands specify isolated coordinates without the correlated contacts, rotations, timing, and support transitions that existed in training.**

ARDY was trained using kinematic constraints sampled from ground-truth motions. Its official constraint demo similarly samples constraints from motion files rather than constructing arbitrary, independently manipulated coordinates. Body constraints are supplied as masked conditioning signals, while root constraints receive a more direct overwrite mechanism.

This suggests three competing mechanisms:

1. **Representation geometry:** absolute height is easier than root-relative lateral displacement.
2. **Decoder/action-path asymmetry:** root and rotation-related information has a stronger path to realized pose.
3. **Constraint-manifold compatibility:** coherent motion-derived constraints work; isolated synthetic coordinates do not.

You can distinguish them experimentally. At present, the report sometimes treats the first mechanism as settled when the evidence supports it as a strong hypothesis.

### 5. The kinematic certificate still fails at the only level that ultimately matters

Your own tracker result says that 23.5% of kinematically certified plans do not retain adequate clearance during execution. That means Phase 4 closed the **generator loop**, but not yet the complete **robot loop**.

The actual system should be: propose → generate → kinematically verify → repair → track in physics → execution verify → accept/refuse.

SONIC is designed as a broad motion-tracking foundation model, but a general tracker does not guarantee that an externally supplied reference is scene-valid. Consequently, the Phase 4 hard set eventually needs tracked execution results, not only collision checks on the ARDY clip.

### 6. "More training data cannot fix it" is too broad

What you demonstrated is narrower and more useful: more imitation data from the same biased teacher did not fix the teacher's out-of-distribution error. Training data grounded in actual ARDY generations, physical tracking outcomes, or successful prompt–constraint compositions could absolutely fix parts of the problem. The issue is not learning itself; it is what target the model is learning.

## The better central thesis

> **A frozen motion prior should be treated as a stochastic, partially addressable motion actuator—not as a scene planner. Reliable scene-conditioned use requires calibrated capability discovery, semantic behavior selection, geometric placement, output-level verification, bounded repair, and explicit refusal.**

That unifies all four phases: Phase 1 shows that nominal command dimensions are not equivalent to addressable capabilities. Phase 2 shows that imitating a scheduler does not improve beyond the scheduler. Phase 3 shows that a global response surrogate becomes unreliable under scene shift. Phase 4 shows that observing the realized output enables local correction. Phase 5 should show that text and structured constraints can jointly unlock a second robust behavior.

The method diagram should become: Scene + route → behavior prompt → sparse motion scaffold → ARDY → directional verifier → repair / switch / refuse → SONIC execution verifier.

## The three experiments to run immediately

### Experiment 0: Close the baseline hole

Before building another model, run every proposer through the same repair operator:

| Proposer              | No repair | 1 repair | 2 repairs |
| --------------------- | --------: | -------: | --------: |
| Heuristic             |         ✓ |        ✓ |         ✓ |
| QP                    |         ✓ |        ✓ |         ✓ |
| TCN                   |         ✓ |        ✓ |         ✓ |
| Seed/sample selection |         ✓ |        — |         — |

Evaluate at fixed budgets of approximately 1, 2, 3, and 5 ARDY calls. Report: observed kinematic collision rate; target-margin rate; integrated crouch, not only peak dip; goal completion; query count and wall-clock latency; worst-scene and CVaR-style performance; number of refusals.

This experiment decides whether the TCN belongs in the final method. Do not protect it. If heuristic + repair wins, make that the system.

### Experiment 1: Test coherent constraints rather than isolated coordinates

For the same small step-over, compare five conditions with matched obstacle location and target foot clearance:

1. walk prompt only;
2. step-over prompt only;
3. manually constructed foot-position constraint;
4. constraints extracted from a successful text-generated step-over clip;
5. text prompt plus sparse rotation/keyframe scaffold extracted from that clip.

Measure: swing-foot peak; stance-foot displacement; number and duration of supporting contacts; bilateral airborne time; root vertical impulse; ARDY seed stability; SONIC tracking success; final obstacle clearance.

This experiment directly distinguishes "position representation is weak" from "the manual command was off-manifold." It may also give you the second executable strategy with remarkably little new machinery.

### Experiment 2: Execute the full 36-scene hard set through SONIC

For each final accepted kinematic candidate, run several physics seeds and record: body–obstacle collision; fall or termination; goal completion; executed overhead and lateral clearance; tracking error by body region; contact deviation; loss of clearance from kinematic to physical execution.

Then fit a simple calibrated relationship such as c_exec = c_kin − e_track(a, ȧ, strategy, duration), where adaptation magnitude and strategy type condition the uncertainty. The goal is not initially to train a complex dynamics model. It is to learn a conservative lower bound such as Pr(c_exec ≥ c_min) ≥ 1 − α. That gives the word "certificate" a defensible meaning.

## Phase 5: Verified semantic–geometric composition

### 5A. Build a prompt atlas, but evaluate addressability rather than prompt quality

Use a small predeclared set of behavior families: normal walk; crouch walk; step over a low obstacle; walk sideways; squeeze through a narrow opening; lift one leg while continuing forward.

For each behavior, use several paraphrases and the same seed set. Record a descriptor vector containing: root-height profile; left/right foot peaks; contact sequence; body half-width; yaw; forward progress; trackability.

Separate two questions. Discrete addressability: can text reproducibly select a qualitatively different contact and posture pattern? Continuous addressability: can phrases such as "small," "medium," "high," or numerical obstacle heights continuously control the realized amplitude?

My prior is that text will work better for the first than the second. That is not a failure. It implies a clean division of labor: text selects the behavior mode; geometric constraints scale and place it.

Prevent prompt fishing by using: discovery prompts for development; held-out paraphrases for evaluation; identical seeds; predeclared descriptors and thresholds.

### 5B. Prompt → scaffold → scene placement

1. Generate several text-only examples of "step over a low obstacle."
2. Select clips with valid unilateral support and good SONIC trackability.
3. Extract a sparse canonical scaffold: hip, knee, and ankle rotations; pelvis height; swing-foot phase; supporting-foot contact interval.
4. Align the scaffold temporally to the obstacle position along the route.
5. Scale only a small number of parameters: step height; swing duration; root progression; perhaps knee flexion.
6. Give ARDY both the semantic prompt and sparse scaffold.
7. Verify, repair, or refuse.

This would turn the text result from an interesting anecdote into an actual planning primitive. Text discovers a coordinated motion manifold; constraints anchor that manifold to a particular scene.

### 5C. Replace the fixed global surrogate with a local response model

The Phase 3 response surrogate failed because it assumed a global, stationary relationship between command and realized clearance. Phase 4 already contains the seed of the replacement: learn the response locally from the generated result.

For one control variable, use a secant or trust-region update: u_{k+1} = clip(u_k + (c_target − c_k) / (∂c/∂u)^). For multiple controls, estimate a small local Jacobian and solve min ‖Δu‖²_W subject to predicted overhead, lateral, contact, and saturation constraints.

The important behaviors are: if sensitivity is strong, repair; if sensitivity is near zero, switch mode or route; if the command is saturated, refuse; if the contact pattern changes discontinuously, shrink the trust region.

Use common random numbers—identical ARDY seeds for paired perturbations—to reduce generator noise when estimating sensitivities.

### 5D. Learn from realized outcomes, not from the heuristic teacher

The next learned model should not predict the heuristic's schedule. It should predict the **distribution of what ARDY and SONIC will actually do**.

Construct D = {S, r, p, u, y_generated, y_executed}. Train a small probabilistic model to estimate: lower quantile of clearance; probability of contact corruption; probability of tracker failure; expected repair count; probability that the command is saturated or unreachable.

A modest 1-D scene encoder or tiny ensemble is enough. The key is the supervision, not model scale. At planning time, optimize a conservative prediction such as the fifth-percentile clearance, then still verify the selected candidate. This gives learning a meaningful role: amortize expensive queries while preserving measurement as the final authority. Evaluate it on query efficiency and calibration, not schedule MAE.

I would not use RL yet. The current decision space is low-dimensional, generation is expensive, and the verifier supplies direct numerical residuals. Supervised probabilistic response modeling plus active querying is simpler and more sample-efficient.

### 5E. Add a second prior before making general claims

The cleanest second generator is probably **Kimodo-G1**: same robot skeleton; closely related text and constraint interface; different offline-generation architecture; publicly released G1 model. Kimodo supports text, keyframes, joint positions and rotations, paths, and waypoints, making it suitable for applying the same audit protocol.

Run a reduced audit: naive count; null-calibrated count; prompt versus position constraint; on-manifold versus manually constructed constraints; generate–verify–repair success.

ARDY currently has released Core and G1 checkpoints; the official repository still labels SOMA as forthcoming, so Core can test representation effects now, while Kimodo-G1 is a better immediate cross-model test.

## The right benchmark expansion

The next scene set should not merely contain more beams. It should vary the **decision structure**: overhead-only scenes where duck magnitude is continuous; floor obstacles where step-over timing and contact matter; mixed over–under sequences requiring mode composition; lateral passages where the system should reroute or refuse; alternate-route scenes where a longer upright path competes with a shorter crouched path; obstacle arrangements unseen at the topology level, not only unseen dimensions.

Split by complete scene families. Holding out beam count while sharing the same procedural grammar is useful, but it is weaker than holding out topology.

For a less synthetic evaluation, the Moving Through Clutter framework provides embodiment-consistent motions and 145 cluttered scenes designed specifically for geometry-induced humanoid adaptation. Even a carefully selected subset would substantially improve ecological validity.

## Recommended acceptance gates

**Gate A: Baseline validity.** The proposed method must improve at least one meaningful Pareto axis over heuristic + identical repair: fewer queries at equal execution safety; less crouch at equal safety; higher safety at equal query budget; better calibrated refusal. Otherwise, remove the learned scheduler and present verification/repair as the method.

**Gate B: Second strategy.** Step-over should achieve, on held-out obstacle geometries: stable unilateral contact structure; no pathological bilateral flight; materially better execution success than constraint-only lifting; repeatability across prompts and seeds. A target such as at least 70–80% tracked success would make it a meaningful planner action rather than an isolated positive example.

**Gate C: Composition.** Text + scaffold should significantly outperform both text alone (which lacks precise placement) and constraints alone (which lack coordinated semantics). This factorial comparison is crucial.

**Gate D: Execution certificate.** On held-out scenes, the acceptance rule should achieve a low false-safe rate. Report a risk–coverage curve: coverage = fraction of candidates accepted; risk = collision/failure rate among accepted candidates. A verifier that refuses difficult cases can be valuable, but refusal must be measured rather than hidden.

**Gate E: Cross-prior transfer.** At least the qualitative audit finding—naive capability inflation, channel differences, or benefit from output verification—should reproduce on Kimodo or another prior.

## How I would restructure the paper

Possible title: **A Frozen Motion Prior Is Not a Planner: Verified Scene-Conditioned Humanoid Motion Generation**. Or, after step-over succeeds: **Scene2Motion: Verified Semantic–Geometric Composition for Frozen Humanoid Motion Priors**.

Main contributions: (1) a calibrated capability-audit protocol that distinguishes nominal controls, stochastic effects, repeatable addressability, and physical executability; (2) a black-box generate–verify–repair/refuse planner that corrects realized scene violations rather than trusting predicted commands; (3) an analysis of conditioning-channel complementarity, showing that semantic prompts can elicit coordinated behaviors not reliably expressed by isolated geometric constraints; (4) an execution-aware acceptance model connecting generated kinematic clearance to tracked physical safety; (5) optionally, cross-prior validation on ARDY and Kimodo.

What should move out of the center: Phase 2's TCN should become an instructive negative ablation unless it wins under fixed query budgets. Phase 3's QP should be presented as a proposal mechanism, never as a certificate. The route scorer should remain secondary until its oracle is grounded in actual generated outcomes. The tally of approximately 29 measurement defects is valuable for the project website and reproducibility appendix, but the paper should distill it into methodological defenses rather than narrating every internal mistake. Avoid spending too much paper space proving that the team was self-critical.

## My concrete recommendation

Freeze Phase 4 as a reproducible baseline. Do not add another generic neural scheduler.

1. Run heuristic/QP/TCN with identical repair and fixed query budgets.
2. Test text-generated, rotation/keyframe-based, on-manifold step-over scaffolds.
3. Run every final hard-set candidate through SONIC and calibrate execution risk.
4. Train a small response-and-risk predictor from actual ARDY/SONIC outcomes.
5. Replicate the reduced audit on Kimodo-G1.
6. Only then build the multi-strategy scene planner.

There are two honest outcomes. If text plus sparse rotation constraints turns step-over into a repeatable executable behavior, the project becomes a compelling semantic–geometric composition system with verified multi-strategy planning. If it does not, the project is still valuable, but the strongest paper is narrower: a rigorous study showing that frozen generative motion priors have a much smaller addressable robot repertoire than nominal interfaces suggest, and that measured test-time correction is more dependable than teacher imitation under distribution shift.

Both are publishable stories. The important progress you have already made is that the project no longer needs to pretend those are the same claim.

References cited by Review 2:
[1] https://arxiv.org/html/2607.08741v1 — ARDY
[2] https://arxiv.org/html/2506.12769v1 — (RLPF / PhysMoDPO context)
[3] https://arxiv.org/html/2601.16035v1 — Collision-Free Humanoid Traversal in Cluttered Indoor Scenes
[4] https://research.nvidia.com/labs/sil/projects/ardy/
[5] https://arxiv.org/abs/2511.07820 — SONIC
[6] https://research.nvidia.com/labs/sil/projects/kimodo/
[7] https://github.com/nv-tlabs/ardy
[8] https://arxiv.org/html/2603.05993v1 — Moving Through Clutter
