# Response to the Phase 4 reviews, and the Phase 5 program

*Written 2026-08-30 against the two reviews saved verbatim in
`guidance-2026-08-29-phase4-reviews.md`. Every claim below was checked against the committed
ledgers, the code, or the ARDY/SONIC/Kimodo checkouts; where a reviewer's number or mechanism
could not be verified it is marked so. File references are to the working tree at HEAD `d54565b`
unless noted.*

---

## 0. The situation this response is written into

Three things happened between the report the reviewers read and this document.

1. **Both reviews arrived.** They agree on the two crux questions — *does anything learned
   belong in the deployed system once repair exists?* and *are the text and keyframe channels
   addressable and spatially composable?* — and on the ordering: proposer ablation first, then
   an executable acceptance gate, then the step-over amplitude boundary, then the text channel.
2. **A concurrent implementation session (Codex, uncommitted in the working tree as of 01:06
   on 2026-08-30) started acting on the same reviews.** It has: (a) found a sampler defect —
   `_per_sample_noise` re-seeded a fresh `torch.Generator` on every intercepted `randn`, and
   ARDY draws a fresh initial latent per 52-frame autoregressive window (`ardy_model.py:481`
   inside `_generate_window`, called per `auto_step` at `:630`), so every window of a long clip
   received the *same* latent row; (b) bumped `NOISE_STREAM_VERSION` and `CACHE_VERSION` to 2;
   (c) removed the dead path-reference generation in `verify/loop.py`; (d) withdrawn §4.8;
   (e) written a SONIC achieved-state export callback (`sonic_state_export.py`, unrun);
   (f) built `verify/experiment_architecture.py` (the proposer × feedback matrix with an
   equal-budget resampling control) and `experiments/exp016_semantic_geometric_stepover.py`;
   (g) recorded all of this in `revalidation-2026-08-30.md` and `exec-gate-audit.md`.
   I verified (a), (c) and the two mechanism claims behind (d) independently; they hold.
3. **An independent fact-check of both reviews was run** (nine read-only agents over the
   ledgers and code, a dedup pass, then a two-lens refutation pass on every critique). The
   surviving findings are folded into §§2–4; the ones that did not survive are listed in §8.

The rest of this document is therefore not "what I would do next" in a vacuum. It is what to
keep, what to change, and in what order, given that the sampler is being fixed and two of the
four first-tier experiments are already scaffolded.

---

## 1. Verdict in five lines

- **The reviewers' two crux calls are right and I accept both.** But each has to be
  re-specified before it is worth running, and the reasons were invisible from the report.
- **Every Phase 4 number is a single ARDY seed (100) under a sampler that was not ARDY's.**
  The heuristic-versus-repair comparison both reviewers foreground is one scene at −3.7 mm,
  inside one clip's seed noise. Running the proposer ablation "with identical seed policies"
  would inherit n = 1 and decide an architecture question on a coin flip.
- **The text step-over that both reviewers build their research direction on is a foot-peak
  delta, not a clearance.** Under the project's own whole-body box metric the step-over prompt
  clears 0.0 m in every seed (two independent re-analyses), lands its swing at an uncontrolled
  position (peak-x sd ≈ 0.70 m, same as the walk), and the one composition datum already in
  the log (EXP-001c `both`, n = 4) leans against rather than for. Text is a door; it is not yet
  a door out.
- **The executable-gate recommendation is right in spirit and wrong in every detail:** the
  23.5 % comes from the Phase 1 gate ledger, the 85.6 mm threshold double-counts
  `BODY_MARGIN`, the "head + hand" error is SONIC's torso + wrists, and the Phase 4 gate already
  demands more than the execution requirement. Under the corrected criterion heuristic and
  tcn+2 tie at 32/36. No κ can be fit until achieved states exist.
- **One paper, not two**, with the R2 fork as the honest outcome. The systems half cannot
  stand alone while it is single-seed and kinematic-only; the methods half needs a second
  generator — which, contrary to §8.3, is already on this machine (Kimodo-G1-RP-v1, weights
  cached, 994 clips generated).

---

## 2. What I agree with, and verified

| Reviewer claim | Status | Where checked |
|---|---|---|
| Heuristic 0.972 / 0.583 / 2.0 calls / 36.7 cm; tcn+2 1.000 / 0.417 / 5.3 / 42.0 cm; "one collision in 36" | confirmed | `outputs/phase4a_repair/experiment.json` (m018, Exp D); heuristic's one collision is `h0.950_w2.250_n6_g3.50` at −3.7 mm with peak dip at `DIP_MAX` |
| McNemar 9/0 (p = 0.0039), 14/0 (p = 0.0001); retraining control shows no difference (largest 4 vs 0, p = 0.125) | confirmed, recomputed | paired by scene on `phase4a_repair` vs `phase4d_hard` |
| 12 of 21 residual `accepted_margin` runs sit at the command limit | confirmed | 12 rows at `peak_dip_m == 0.5` |
| Wilson 3/8 ≈ [0.14, 0.69]; rule of three 3/36 ≈ 8 %; Clopper–Pearson 36/36 ≥ 90.3 % | arithmetic confirmed; denominators wrong (§4.6) | — |
| The loop is proposer-agnostic, so heuristic + repair "runs on the existing harness" | confirmed | `verify/loop.py run(scene, p, q0, resp, cache, …)` takes a schedule, not a proposer; `experiment_repair.py:46-49` is a three-dict edit. Codex's `experiment_architecture.py` already does this |
| Scope the funnel headline to "the constraint interface, under the walk prompt" | confirmed, with a correction | the prompt was *not* fixed in every experiment: EXP-001b used a side-step prompt, EXP-001c three prompts. The funnel (EXP-005f) was under the walk prompt |
| R2: root constraints are overwritten into the sample, body constraints are concatenated conditioning | confirmed from code | `auto_latent_twostage_denoiser.py:204-207` infills the root slice at every denoiser call; body constraints enter only as `[latent_body, root, observed_body, mask]` features. The body is a 128-d FSQ latent during denoising — there is no explicit body block to overwrite |
| R1: ARDY was trained on text labels and constraints sampled from ground-truth motion | confirmed from the paper; no training code shipped | arXiv 2607.08741 §3.5 |
| R2: the official demo samples body constraints from motion files | confirmed | `README.md:80,119,161-167`; root constraints can be authored by mouse, body/EE constraints cannot |
| Kimodo-G1 is released, same skeleton, keyframe/EE/path/waypoint interface | confirmed — **and REPORT §8.3 is false**: `/home/linjiw/kimodo` exists, `nvidia/Kimodo-G1-RP-v1` is cached (1.1 GB), 994 G1 clips with constraints were generated here on 08-28, and `design.md:67` already listed it when §8.3 was written | `~/.cache/huggingface/hub`, `kimodo/data/indoor_nav_1k/generate.log`; `g1.xml` md5-identical to ARDY's, so `G1Body`, `sonic_export` and the funnel reuse unchanged |
| Core released, SOMA "coming soon" | confirmed | `ardy/model/registry.py:23-35`, `README.md:73` |
| RLPF / PhysMoDPO post-train the generator; HumanoidPF and PHP show crouch/squeeze/hurdle/climb on real G1; Moving Through Clutter = 145 VR-collected scenes | confirmed | PhysMoDPO is 2603.13228 (R2's [2] is RLPF only); MTC's release is announced, not located — do not plan on it |
| "More training data cannot fix it" is too broad | confirmed, and sharper than R2 states | see §4.8: on collisions the student is *below* its teacher |
| Off-manifold / constraint-family compatibility is a live third mechanism | confirmed, with a documentary detail R2 did not have | ARDY's official "sparse joint rotations" are chain-base only — ankle, wrist, pelvis (`skeleton/base.py:153-159`); the official foot constraint is a coherent packet (ankle + toe positions, ankle + pelvis rotations, root xz/y/heading from one real frame). The project's lift wrote leg-joint positions + root — exactly the isolated-coordinate pattern |

The mechanism R2 asked for behind §4.2 also exists in the code and the report never cited it:
stage 2 of the denoiser predicts the body latent conditioned on a *local root* of
`[heading rate, planar velocity, absolute root y]` (`motion_rep/base.py:103-147`). Height is
handed to the body decoder absolutely; the ground plane only as velocity. That is the
representation account, one layer deeper than `local_joints_positions`.

---

## 3. Where I push back on the reviewers

### 3.1 A "full-body keyframe" is not a new channel (R1 item 5, R2 weakness 3 / Exp 1 cond. 5)

`FullBodyConstraintSet.update_constraints` writes `global_joints_positions` for all 34 joints
plus `root_2d`, `root_y_pos`, `global_root_heading` — and the comment at
`ardy/constraints.py:165` reads `# global rotations are not used here`. Kimodo's is identical.
A mid-crossing keyframe is therefore dense writing into the same position block the project
already audited (§4.4: the decoder poses from rotations; positions reach the body only through
the learned latent), and it cannot say "stance foot planted" because `foot_contacts` is
unwritable. R1's item 5 as written re-tests the position channel with more mask.

The channel that *is* un-built is `global_joints_rots` for the legs. The project's adapter
writes it (`scene2motion/constraints.py:168-177`); EXP-010 measured it as the cleanest signal
in the project (ROT− 30°: +271.8 mm foot rise, sd 6.6 mm, pelvis −7 mm); and it has only ever
been tracked as an ungated two-legged raise at 0.35 m (EXP-012). Note also that hip/knee
rotations lie *outside* ARDY's official constraint family (chain-base only), so an "on-manifold
rotation scaffold" is on the motion manifold but off the trained constraint family — R2's
condition 5 tests two things at once unless the arm uses the official foot-EE packet.

### 3.2 The executable gate is not a one-line change, and its inputs are wrong (R1 push-back 3, R2 Exp 2 / Gate D)

Five separate problems, each verified:

1. **Population.** The 23.5 % / 34.5 % figures are computed over the 5,479 collision-free
   clips of the *Phase 1 gate ledger* (`design.md:2003`), not over any Phase 4 plan.
2. **Double count.** `G1Body` inflates every scene box and body extent by
   `BODY_MARGIN = 0.04` (`robot.py:44-51, 100, 194`), so every clearance in every ledger is
   already net of the mesh correction. §32's "40 + 45.6 = 85.6 mm" adds it again. Against
   ledger clearance the requirement is tracking error alone; on the same 5,479 clips that is
   9.7 % below 45.6 mm, not 23.5 %.
3. **Wrong body group.** SONIC's `vr_3points` is `torso_link` + the two wrist-yaw links
   (`im_eval_callback.py:557`, "use torso_link instead of head"). The report's "head + hands
   45.6 mm" is torso + wrists, root-relative, unsigned, time-averaged — not a head clearance
   loss in any frame.
4. **The Phase 4 gate is already stricter than the execution requirement.** `MARGIN_M = 0.18`
   exceeds 0.072 + anything at every depth. The executable question only bites on the
   `accepted_margin` bucket (24/36 of tcn+2 outcomes), and re-scoring it is a ledger filter,
   zero GPU. Result under tracking-error-only, depth-interpolated from EXP-011
   (33.5 / 29.7 / 45.6 / 71.9 mm at dips 0 / 0.20 / 0.35 / 0.50): **heuristic 32/36, tcn+2
   32/36, paired 1–1.** Under R1's literal "40 mm + error": 28 vs 30. Under a flat 45.6 mm:
   33 vs 35. No threshold reorders the proposers; tcn+2's one-scene collision lead dissolves
   because it crouches deeper and therefore needs more clearance.
5. **No κ is fittable.** Three duck depths × 8 seeds, shallow arm non-monotone (29.7 < 33.5),
   proxy metric, no achieved states saved (`metrics_eval.json` holds scalars only). Codex's
   `exec-gate-audit.md` reaches the same verdict and specifies the right target: signed
   clearance loss `c_ref − c_exec` against the same margin-inflated geometry, one-sided
   conformal bound `τ(d)`, two gates (`≥ τ(d)` executable; `≥ 0.18 + τ(d)` retains target).
   The callback that produces the achieved qpos exists and has not run.

So: keep the 0.18 m proposal margin, add an *execution* acceptance tier, and do not fit
anything until Experiment 2 (§6, 1B) has produced paired reference/achieved clearances.

### 3.3 R2's Experiment 1 is circular as designed, and Gate B is uncalibrated

Conditions 4/5 extract constraints from ARDY's own successful clip and evaluate on the seeds
that produced it; on-manifold-by-construction against synthetic-by-construction, confounded
with amplitude and number of constrained coordinates. "Matched target foot clearance" is
unattainable because the position channel over-responds 1.8–2.5× (0.20 m → +362 mm;
0.08 m → +197 mm) and text delivers +57 mm with sd 45. Tracked success at n = 8 cannot
separate 5/8 from 3/8 (Fisher p = 0.62); 55–100 seeds per arm for 80 % power at the effect
sizes on the table. Gate B's 70–80 % bar sits *above* the duck's own tracked rate at working
depth (0.750) — applied symmetrically it fails the flagship strategy. State Gate B as
non-inferiority to the duck at matched clearance, with n fixed by power, and with **box
clearance at the obstacle as the primary endpoint** (see §4.4 for why).

Codex's `exp016` already fixes the donor/evaluation seed split and uses exact box clearance;
what it still lacks is a rotation-only arm and an official-foot-EE-packet arm (§6, 2A).

### 3.4 Text cannot be placed in time (R1 item 6, R2 5B)

The text encoder returns one pooled 4096-d vector per prompt (`llm2vec_wrapper.py:93-94`,
`lengths = 1`), injected as a single prefix token before self-attention over motion tokens.
Within `Ardy.__call__` — what `runner.py` uses — one prompt covers every autoregressive window.
Prompt switching exists only through `autoregressive_step`, at window granularity: 52 frames
= 2.08 s on the checkpoint the project uses, **8 frames = 0.32 s on `ARDY-G1-RP-25FPS-Horizon8`,
which is already in the local HF cache.** R1's prediction that composition fails on
timing/placement is structurally grounded; the only native placement mechanisms are
constraints or window boundaries. The Horizon8 checkpoint is the cheapest untested lever for
text × space and neither reviewer nor the report mentions it.

### 3.5 "Saturated ⇒ unreachable/refuse" is an action-space artefact (R1 item 8, R2 5C/5D)

`DIP_MAX = 0.50` is the program's full-scale constant (`optim/response.py:25`). The fitted
response is *steepest* in its last segment (1.05 m/unit vs 0.67 overall), the slope floor never
bound (`repair_stats.slope_floor_bound_steps = 0`), and no experiment has ever commanded
beyond 0.5 m (EXP-001d's dip ladder stops there). Across the 180 Exp G rows, 39 end at
`DIP_MAX`; 37 are collision-free and 12 meet 0.18. A saturation flag would refuse most of the
scenes it fires on. The prior is not the limit either: `design.md:1788` records an emergent
0.578 m duck under the Horizon8 checkpoint. One cheap sweep past 0.5 m (matched seeds) has to
precede any refusal rule — and duck-deep already tracks only 0.625, so the real ceiling may be
the tracker's. Treat saturation as a feature on the Gate D risk–coverage curve, not a rule.

### 3.6 Core cannot test the representation claim with this harness (R2 5E, R1 item 7)

Every descriptor is read off inflated MuJoCo G1 primitives and `robot.py` hardcodes `g1.xml`;
`cskel27/29` ships joints and a T-pose, no collision model, no tracker. Core needs a new body
model and a rescaled scene ladder before its lateral kill condition means anything. Kimodo-G1
reuses everything unchanged. Order: Kimodo first, Core as a joint-position-proxy test later.

### 3.7 SONIC is an offline stage, not a loop element (R2 method diagram)

One Isaac launch costs ~35–45 s for 8 clips (startup + 398 steps at ~17 it/s), roughly
independent of `num_envs`; an ARDY attempt costs 0.4 s. Experiment 2 offline is cheap
(36 scenes × 5 methods × 3 physics seeds ≈ 15–20 min). Inside the two-iteration repair loop
or the demo it is 50–100× the rest of the loop.

### 3.8 R2's 5D premise is wrong about *why* Phase 3 failed

The Phase 3 gain `g` was fitted on realised ARDY top heights (EXP-001d, 220 rows), not on the
heuristic. It failed on *scene shift* (multi-beam), not on supervision source. "Learn from
realised outcomes" without a pre-registered OOD split on the covariate that broke Phase 3
(beam count × gap) reproduces §15 with quantiles. Its best case saves under two generations
per scene.

### 3.9 Two papers → one paper (R1)

The systems result is 36 synthetic scenes, one grammar, one seed, a sampler that was not
ARDY's, kinematic only. A standalone systems paper would have to defend a single-seed 36/36
against the methods paper's own 6× lesson — the split makes each paper the other's referee.
R2's single narrower paper, with its two honest outcomes as the fork, is what the evidence
supports. Revisit the split only if the Kimodo replication of both the counting result and
verify/repair succeeds.

---

## 4. What neither reviewer could see from the report

### 4.1 Every Phase 4 row is seed 100

`outputs/phase4a_repair/experiment.json` and `phase4d_hard/experiment.json`: `seeds: [100]`,
180 rows each; both failure searches, the lead sweep and Exp E likewise. §12 never states it.
The report's own Phase 1–3 protocol — six paired seeds, ~37 mm per-clip σ on top height, "a
single-seed reading overstates by 6×" — was not applied to the phase that produces the
headline. The heuristic-vs-tcn+2 gap is one scene at −3.7 mm; 8 of 28 second repairs correct
1–21 mm deficits. The whole 180-row grid re-runs in ~2 min per seed at 0.4 s per generation;
eight seeds is a lunch break. **This, not the choice of proposer, is what Experiment 0
decides.**

### 4.2 The sampler was not ARDY's (found by Codex; verified)

See §0(2). Single-window clips (≤ 52 frames) are unaffected; every long `seeds=` clip —
the entire Phase 4 hard set at 292–350 frames, EXP-011–015 — repeated one latent across
6–7 windows. Paired comparisons within v1 remain internally valid (matched seeds, deterministic
DDIM); what is unknown is whether the effect sizes and the seed-noise floors survive the
intended sampler. Codex's position — keep v1 results as audit evidence, re-run under v2 before
any number is headline — is the right one. The 6× counting result (EXP-005f) is the highest
priority re-check because it is the methods paper.

### 4.3 The reported ARDY calls are 2× the necessary cost (fixed by Codex; verified)

`loop.py:125` generated a path-only reference clip per attempt and never read it
(`spec_from_dip` takes no reference; the runner is stateless per call). Every "ARDY calls"
figure in §12 is `2 × attempts`. Real cost: one-shot proposers 1, tcn+1 ≈ 2.0, tcn+2 ≈ 2.6.
Ratios survive (2.64×); absolute budgets halve; R2's budgets of 1/3/5 were unattainable under
the old accounting. Defect 17 replaced a cache artefact with a dead-code artefact.

### 4.4 The text step-over is a gait change, not a step-over

Three independent lines of evidence, none in the report:

- **EXP-001c already ran the prompt-only arm** with the same prompt and the same root-only
  design (24 draws) and scored `max_box_h_m` 0.000–0.007 m under `probe_box` — the tallest
  0.25 m-deep corridor-spanning floor box the *whole body* clears, the metric on which
  `max_step` and the conformal `low_obstacle` refusal were calibrated.
- **Re-probing the EXP-015 arm-B clips** with the box at each clip's *own* foot-peak x gives
  0.0 m in 8/8 seeds under both the margin-inclusive and the margin-free probe (an earlier pass
  read one seed at 0.019 m; the refutation pass corrected it to 0.0). Peak-x sd is ≈ 0.70 m
  (walk arm: ≈ 0.72 m).
- **Codex's `exp015b` re-analysis**, done separately: max box height 0.0 in all three arms;
  swing foot at the crossing 55 mm (B) vs 24 mm (A); `local_step_success_rate` 0.0.

So "+57 mm clears the bottom three rungs" and "half of the 0/20 refusal is wrong" (§4.9, §36)
are withdrawn: the refusal stands on its own metric. What survives is *behaviour elicitation*
— text changes the gait (3.6 σ on foot peak, contacts up, tracks 0.625) — which is what §36's
own limits paragraph said. Two further corrections: arm C's request was 0.20 m
(`exp015/receipt.json: lift_for_C = 0.2`), not the 0.35 m the report and both reviewers
repeat; and **composition has already been tested once** — EXP-001c's `both` arm (step-over
prompt + gated leg targets) scored 0.010 / 0.012 / 0.021 / 0.051 / 0.079 m against
0.005 / 0.018 / 0.027 / 0.057 / 0.094 m for the same targets under the walk prompt, n = 4,
pre-fix per-batch seeding: lower at 4 of 5 levels, paired differences not significant. It is
weak evidence and it is the only composition evidence, and it does not point R2's way. R2's 5B
step 2 ("select text clips with valid unilateral support") presupposes text step-overs that, by
the box metric, do not exist.

### 4.5 The hard set was partly selected on the TCN's failures

§11's 36 scenes and §12's 36 scenes overlap in 12. The failure search ran first under
`preference=clearance` (one lateral collision), then under `shortest` over
h ∈ {0.85, 0.95} × w ∈ {1.45, 2.25}; all four overhead collisions fell in one cell
(h = 0.95, w = 2.25). Exp D then fixed w = 2.25 and `shortest`, and extended to h = 1.05, 1.20
(never searched). The tcn one-shot 0.750 is a selected-cell estimate; the heuristic was never
searched against. Experiment 0 needs a blind grid.

### 4.6 "Zero regressions" is largely by construction

The loop stops on first accept, so scenes that passed one-shot are never regenerated; the
repair is monotone non-decreasing in command (`e ≥ 0`, clip). Only 1/36 tcn scenes met the
margin one-shot, so "fixes 14, breaks 0" has a structurally empty cell; for collisions the
at-risk n is 26 (27 one-shot collision-free, minus the one accepted scene that is never
regenerated), giving a rule-of-three bound of ~11.5 %, not ~8 %. The empirical content is "a
deeper/longer duck never created a new collision on 26 repaired scenes". In the proposer
ablation the heuristic has 21 regression opportunities and the TCN 1 — the column is not
comparable across arms unless stated per opportunity.

### 4.7 Exp E is on a different scene family and its baseline is a straw man

Exp E's 36 scenes are 1–6 beams × two widths at gap 2.5; 18 of them can be walked around
(18/18 met the target with a zero-duck route), 12 are in-distribution. Its 0.694 margin is
14/24 = 0.583 on the 3–6-beam subset; on the 12 scenes shared with Exp D, tcn_verify chose the
same route as `shortest` on 9 and reached 2/12 (Exp D tcn+2: 4/12). The "heuristic" selector is
argmin route length and picks a zero-duck route 0/36 times; the Phase 1 mode-cost A* — which
already prices "a deep duck outweighs a 0.5 m detour" — was never run as a selector. Every
scene yields exactly 3 routes, so the k = 32/128 latency rows time cycled duplicates. §13
should not sit under §12.

### 4.8 "The teacher is the ceiling" holds for margin, not for collisions

QP one-shot is collision-free on 32/36, TCN on 27/36; paired, 6 scenes QP-clean/TCN-collides
vs 1 reverse; on those 6 the TCN ducks *deeper* at the peak yet clears −37…−105 mm where the
QP clears +1…+49 mm. That is a shape/timing imitation error the per-sample MAE does not
capture. §15/§18 need splitting: margin failure is teacher-limited; collision failure is partly
student-limited.

### 4.9 The tracked step-over experiments gated the lift on seed 0's gait

`exp014.py:93-127` and `exp015.py:119` compute `_limb_targets` once from the control clip of
seed 0 and reuse that spec for all eight seeds, so seven seeds were gated against the wrong
gait phase. The 0.375 (3/8, `outputs/exp014_small`, lift 0.08 m) was measured under a defective
request; the amplitude sweep may move it either way. Per-seed gating is a prerequisite for any
step-over number.

### 4.10 Process debts

`design.md` ends at §37 with no Phase 3/4 entry and no pre-registered predictions for Exp
D–G; `LEAD_TAUS` and its sweep were added after Exp D and tuned on the evaluation scenes; the
demo's two deep links are the single one-shot margin success and one of twelve one-repair
successes, hard-coded in the acceptance test; the REPORT header says 57 commits / 33 sections
(actual 79 / 37); §8.3's Kimodo sentence was false when committed.

---

## 5. Corrections owed to REPORT.md

Codex's working tree already makes the first four; the rest are open.

| § | change |
|---|---|
| 4.8 | withdraw 85.6 mm / 23.5 %: population, double count, body group (done in working tree) |
| 12, 17, 18 | one call per attempt; `legacy_ardy_calls`; amend defect 17 (done) |
| header, 9 | v1-sampler notice; Phase 4 table "pending v2 replication" (done) |
| 32 (log) | §38 correction (done) |
| 12 | state `n_seeds = 1`; name the scene set and its provenance vs §11; add an "executable (tracking-error-only)" column; say next to the 1.000 headline that 21 of the 36 rows are `accepted_margin` (the column itself is correctly labelled); report at-risk denominators; drop "restores 1.000 at every count" (9/9, CP lower 0.66) |
| 13 | state the scene family; remove the comparability with §12; label the latency rows as 3 distinct routes |
| 15, 18 | split teacher-limited (margin) from student-limited (collision) |
| 4.9, 36 | withdraw "clears the bottom three rungs" / "half the refusal is wrong"; fix arm C to 0.20 m; cite EXP-001c `prompt` and `both` |
| 4.7, 8.0 | note the seed-0 gating defect in EXP-014/015 |
| 8.3 | Kimodo exists; SOMA unavailable; Core has no body model |
| 4.2 | "height-absolute is sufficient for addressability; lateral-through-position is unresolved because the only narrowing test is structurally capped"; cite the local-root conditioning |
| 5 | add defects 20 (per-window noise replay), 21 (dead reference call), 22 (seed-0 gating), 23 (`vr_3points` mislabel), 24 (`BODY_MARGIN` double count) |
| header | 79 commits, 37 sections |

---

## 6. Phase 5

**Standing rules for every experiment below.** Sampler v2 and cache v2 only. ≥ 8 paired seeds
per cell (24 for any step-over rate; 55 per arm for any factorial that claims an interaction).
Predictions and kill conditions written into `design.md` *before* the run. Blind grids, both
corridor widths, all three preferences, heights on both sides of the searched cell. Box
clearance at the obstacle is the primary endpoint for anything called a step-over; tracked
success is secondary. Budgets are attempts per scene. Saturation is a pre-registered split,
never a refusal. Every table carries its at-risk denominators.

### Tier 0 — this week, mostly in flight

- **Commit the freeze.** The working tree holds the sampler fix, cache v2, the dead-call
  removal, the §4.8 withdrawal and the two audit documents. Commit them as one provenance
  boundary before any v2 clip is generated, and re-emit the Phase 4 table from the ledger
  with the executable column so the v1 baseline is fully characterised before it is replaced.
- **Write the pre-registration.** One `design.md` section per Tier 1 experiment: prediction,
  kill condition, seed count, grid, primary endpoint.

### Tier 1 — days; each can change what the paper is

**1A. Proposer × feedback matrix (R1 item 1, R2 Exp 0; scaffolded as `phase4e`).**
{heuristic, QP, TCN} × {0, 1, 2 repairs} plus the equal-budget resampling control
(byte-identical proposal, best-of-k seeds) — Codex's design, which is better than either
reviewer's because it separates "feedback" from "more samples". Required additions: 8 seeds;
blind grid (h ∈ {0.85, 0.95, 1.05, 1.20}, w ∈ {1.45, 2.25}, three preferences); saturation
split; executable column; per-opportunity regression counts; integrated crouch and CVaR.
Cost ≈ 15 min GPU per seed-set. *Prediction:* heuristic+1 ≈ tcn+2 on collision-free and ahead
on margin and crouch; resampling ≈ one repair. *Kill:* if the learned scheduler is not on the
Pareto front on any axis at any budget, it leaves the system and Phase 2–3 become an ablation.

**1B. Executed clearance on the hard set (R2 Exp 2; callback written, unrun).**
Run `sonic_state_export` on the 1A accepted candidates, 3 physics seeds (`seed:` in
`sonic_release/config.yaml`; DR events randomise friction/COM), reconstruct qpos by joint
*name*, score with `clearance_trace` against the same margin-inflated scene, treat termination
before the obstacle as failure. Fit `τ(d)` one-sided conformal on `c_ref − c_exec`; report the
risk–coverage curve (Gate D). ≈ 1–2 days engineering, ~1 GPU-hour. *Kill:* if `τ(d)` is not
monotone in dip or its 90 % bound exceeds 0.18 − 0.04 anywhere in range, the 0.18 m target is
itself not executable and the proposal margin must rise.

**1C. Step-over, three cheap runs, no encoder needed (R1 items 3 & 6; R2 Gate B).**
(i) Re-run the 0.08 m gated lift with *per-seed* gating, 24 seeds, box clearance + contacts +
tracked — the 0.375 either survives or dissolves. (ii) Amplitude ladder 0.05–0.35 m at 24
seeds with the same readouts, reported against *realised* foot peak (the channel over-responds
1.8–2.5×). (iii) The composition test that EXP-001c only half ran: 2 × 2 {walk, step-over
prompt} × {no lift, 0.08 m gated lift}, 16 paired seeds, obstacle present, box clearance
primary. All three prompts are cached; ~4 SONIC launches. *Kill for (iii):* if
`both ≤ targets` again, R2's 5B premise fails and the paper narrows to R2's second outcome
without waiting for 2A.

**1D. Kimodo-G1 reduced audit (R2 5E / Gate E; promoted from Tier 3).**
`KimodoRunner` behind the `ArdyRunner` interface (key rename `root_2d → smooth_root_2d`, seed
shim, 30 fps export), then: naive vs calibrated count on the EXP-005f cells, duck
addressability, one generate–verify–repair sweep on the 1A grid. 3.1 s/clip at 100 steps;
2–4 days engineering, 2–3 GPU-hours. *Kill:* if the naive/calibrated ratio on Kimodo is < 2×,
the 6× result is an ARDY fact, not a methodology fact, and the methods paper loses its
generality claim.

### Tier 2 — one to two weeks

**2A. Semantic × geometric step-over (`exp016`, scaffolded).** Keep Codex's donor/evaluation
seed split and box-clearance endpoint. Add two arms: a *rotation-only* scaffold (per-side
gated hip/knee/ankle `rot_targets` at 0.08–0.16 m equivalents — the un-built channel of §3.1)
and the *official foot-EE packet* (ankle + toe positions, ankle + pelvis rotations, root, from
a donor frame — on-manifold *and* in-family). ≥ 24 seeds per cell; 55 if Gate C is to be
claimed. Add a Horizon8 arm: switch the prompt at the window preceding the obstacle. *This is
the experiment that decides whether the paper is A or B.*

**2B. Representation vs manifold, on the axis that can separate them.** Lateral *widening*
(the cap does not bind), position channel, matched seeds: targets copied from the same seed's
own wide-arm clip vs synthetic offsets of equal magnitude. ≥ 5 σ for copied only ⇒ manifold;
both ≈ 1.5 σ ⇒ representation; both ≥ 5 σ ⇒ §4.2 was a cap artefact.

**2C. Route selection with a real baseline.** Mode-cost A* vs `tcn_body` vs `oracle_qp` with
*all* candidates generated and verified (108 routes ≈ 3 min), a candidate generator that yields
> 3 routes, regret rescored on realised clearance/crouch. If mode-cost A* matches `tcn_body`,
the TCN's last role is batched scoring latency and it leaves the paper.

**2D. DIP_MAX sweep** beyond 0.5 m, matched seeds, tracked at the top — before any
saturation-driven refusal or 32 mm lateral pruning rule.

**2E. Text battery (R1 item 4, R2 5A)** — after 1C(iii), and after the SONIC training job
(PID 1027952, 7.7 GB RSS) frees RAM: the CPU Llama-3-8B encoder needs ~14 GB and 14 GB is
available. 20–40 prompts, discovery/held-out paraphrase split fixed before generation,
24 paired seeds, descriptor vector + box clearance. Question: discrete vs continuous
addressability. Kimodo's 300-prompt cache (same LLM2Vec lineage) may be reusable if pooling is
verified identical.

### Tier 3 — before submission

Topology hold-out and non-ladder geometry (door frame, low table, one scanned mesh); Core as a
joint-position-proxy replication with a new descriptor definition; package the audit toolkit
with its own wrong answers (10.00 / 1.67) as the acceptance test; related-work positioning
against BRIC (2511.20431: frozen planner + test-time adaptation, sim), SafeFlow (2603.23983:
test-time gating before a G1 tracker, no scene, no repair), 2604.17335 and HumanoidPF
(capability competitors on real G1).

### The fork

- **Paper A** (if 2A's rotation-only or foot-EE-packet arm yields a placed, contact-preserving
  step-over that tracks at non-inferiority to the duck): *verified semantic–geometric
  composition over a frozen prior*, with 1A/1B as the systems core and the audit as the method.
- **Paper B** (otherwise): *a frozen motion prior is not a planner* — the calibrated audit,
  the 6× (or Kimodo-replicated) counting result, verify–repair–refuse on the one executable
  axis with an execution certificate, and the elicitation lesson stated as a limit.

Both need 1A, 1B and 1D. Only A needs 2A to succeed. Nothing below Tier 1 should run before
1A–1C have reported.

---

## 7. Claim status after this response

| claim | status |
|---|---|
| Single-seed naive counting overstates the addressable repertoire by ~6× (EXP-005f) | **headline, pending v2 re-check**; generality pending Kimodo (1D) |
| Verify → repair takes one-shot TCN from 0.750 to 1.000 collision-free on 36 OOD scenes with zero regressions | **v1, seed 100, 27 at-risk scenes**; pending 1A at 8 seeds under v2 |
| Heuristic + repair beats or matches the repaired TCN | **prediction**, to be tested in 1A |
| The 0.18 m certificate is not executable; 23.5 % fall short | **withdrawn**; replaced by "9.7 % of Phase-1 gate clips lie below the torso-proxy error", and by 1B's `τ(d)` |
| Text reaches a step-over the constraint interface cannot express | **withdrawn as a clearance claim**; retained as "text elicits a higher-swing gait (3.6 σ, n = 8) that does not clear a placed obstacle" |
| Text × constraint composition | **open hypothesis**; one prior negative (EXP-001c `both`, n = 4); decided by 1C(iii) and 2A |
| Height-absolute / lateral-relative representation account | **strong hypothesis**, now with a mechanistic basis (local-root conditioning); separable from the manifold account only by 2B |
| A correctly gated 0.08 m step-over tracks 0.375 | **suspect** (seed-0 gating defect, n = 8, v1); re-run in 1C(i) |
| No learned proposer is justified | **stands for schedule proposal**; for route selection pending 2C |
| Kimodo is unavailable on this machine (§8.3) | **false** |

---

## 8. How this response was produced, and what did not survive

Nine read-only agents fact-checked both reviews against the ledgers and the three checkouts,
costed every proposed experiment against the harness, and attacked the reviewers from a
statistics, a systems and a research-strategy lens; a completeness pass looked for what neither
reviewer could see. Their 30 findings were merged to 20 and each was sent to two independent
refuters (evidence lens, materiality lens). **None of the 20 was refuted on evidence.** Eight
survived both lenses; the other twelve were judged true but plan-neutral — in most cases
because the plan in §6 or Codex's working tree already absorbs them (the dead call is removed;
the exec-gate audit already withdraws §4.8; R2 already recommends Kimodo). The refuters also
corrected three of the critics' numbers, which are the values used above (re-probe 0.0 in 8/8,
peak-x sd ≈ 0.70, at-risk n = 26). Every finding used above was additionally verified by me
directly on the ledger or the code. Findings that were raised and dropped: "the m018 Exp D
ledger was overwritten by Exp G" (false — `phase4a_repair` holds it, committed in `0f50d3a`);
"the 0.375 result has no receipt" (false — `outputs/exp014_small/receipt.json`, committed in
`6d08c15`); "the tracking error is head + hands" (inherited from the report; it is torso +
wrists). Two of Codex's findings (the sampler defect; `vr_3points`) were independently verified
from `ardy_model.py:481/630` and `im_eval_callback.py:557`.
