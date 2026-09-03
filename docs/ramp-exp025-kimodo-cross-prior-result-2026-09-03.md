# EXP-025 result — the float crosses the prior boundary on a thin margin, the early timing does not

**Run:** 2026-09-03, 24.7 min wall clock, one GPU generation stage plus CPU scoring
(`outputs/exp025_kimodo_cross_prior/`, receipt `status`/`stage` both `complete`,
`sample_count_exact: true`, `blocked: false`). Exact accounting 128/128: 128 samples planned,
launched, returned and converted to qpos over 16 of 16 generate invocations
(`query_accounting`), latent pairing verified in every one of the 16 chunks
(`stages.generate.latent_audit`), 128/128 scored (`stages.score`). Protocol
`docs/ramp-exp025-kimodo-cross-prior-protocol.md` (sha `3d8a47bc…`, **preregistered 2026-09-03
before the first sample**, with a three-point amendment also written before any sample), bound in
`provenance.protocol`. Driver `experiments/exp025_kimodo_cross_prior.py` (`6a20590a…`) with the
recovered runner `experiments/kimodo_recovered/kimodo_runner.py` (`7e3d4b25…`); clean worktree at
`93323a09`, revalidated after generation (`provenance.post_generation_identity_revalidation`: git,
checkpoint, sources, physical model, prompt cache and runner contract all unchanged). Prior:
**Kimodo-G1-RP-v1**, HF revision `3020ad8c…`, `model.safetensors` sha `e18c1de7…`, Kimodo checkout
`1aece8c1…` with 83 source files hashed. Generation ran under `/home/linjiw/kimodo/.venv/bin/python`
and scoring under `$S2M_PY`; the two interpreter runtimes are recorded, not equated
(`interpreter_runtime_compared_across_stages: false`).

**No tracker ran.** Tiers (`summary.tiers`): generated 128, kinematically scored 128,
**SONIC-executed 0**. Nothing in this note is an execution outcome, and the 0.20 s rule is the
calibrated **reference screen for predicted tracking cutoffs**, not a physical verdict
(`decisions.scope`).

## 1. What was asked

The step-family findings — the prompted behaviour appears early in the rollout, and it contains a
long period during which neither foot meets the calibrated support test — were both measured on
**one** prior: ARDY-G1, autoregressive, 25 fps, Horizon52. Two readings compete. Either these are
properties of released text-conditioned G1 priors, or they are properties of that architecture.
EXP-025 is the project's first cross-prior campaign and it separates them by running the same
measurement on **Kimodo-G1-RP-v1**: the same released skeleton and the same LLM2Vec text
conditioning, but an **offline, non-autoregressive** prior — one denoising pass, no history window
(`campaign_design.prior`), 30 fps.

Locked design: 64 STEP and 64 WALK references on seeds **4700–4763**, **paired per seed** (each
B = 8 call holds 4 seeds × both prompts, so a pair shares one call and one per-sample latent,
`campaign_design.batch_composition_rationale`); straight 7.2 m route at 0.9 m/s, 240 frames,
7.97 s of frame span; dense `smooth_root_2d` path constraint with root height and heading free —
the `free` contract exp021 used; 100 DDIM steps, cfg (2.0, 2.0), first heading 0, post-processing
bypassed, `noise_stream_version` 2. The WALK arm is the free nominal arm and the elicitation floor
(house rule 9).

The comparison is clean at the text boundary: **the STEP embedding is the byte-identical vector
the ARDY family used**, copied from the ARDY family's `outputs/text_cache.npz` in the sibling worktree
`/home/linjiw/scene2motion-exp030` (§3; source file sha `330c6996…`) under the same `sha1` key
(`ced85ba8…`, content `b378a64e…`), with the ARDY and Kimodo LLM2Vec wrappers compared as
normalised ASTs — identical, `3da6f640…` on both sides — and identical preset kwargs
(`provenance.prompt_cache.fields.encoder_equivalence`). No text encoder was loaded. The WALK arm
uses Kimodo's own cached prompt, "A person walks forward at a steady pace.", from the indoor_nav
cache, which is **not** the ARDY family's WALK text.

Scoring used the committed analysers the ARDY family uses: `box_height_profile` / `G1Body` (body
margin included) for lift and exact clearance, and `analyze_trackability_contract.features` for the
support features, with thresholds read from the frozen exp016 calibration receipt (sha
`f6dba8be…`; support height 4.649 cm, planar speed 1.175 m/s, `stages.score.support_thresholds`).
Thresholds are in seconds and therefore fps-free, and every event time is reported in seconds, never
in frames, because the two priors run at different rates.

Two decision rules were preregistered (`campaign_design.decision_rules`):

- **Timing.** First-2 s fraction ≥ 0.7 → the early window generalises; **≤ 0.4 → the ARDY window is
  attributed to autoregressive rollout context**; in between, report both distributions with no
  mechanism claim.
- **Screen.** **≥ 80 %** of Kimodo's *elicited* clips above the calibrated 0.20 s screen → the
  statement generalises to released G1 priors; otherwise it stays ARDY-scoped.

No arm expansion after outcomes (`decisions.no_arm_expansion_after_outcomes.held: true`).

## 2. Result

**The two rules split, and the split is the finding: the float crosses the architecture boundary
— 19 of 23 elicited references, 0.826, on a Wilson interval (0.629–0.930) that straddles the
preregistered 0.80 bar — and the early timing does not.** An offline, non-autoregressive prior
given the identical STEP embedding produces the same long period outside the support test that
the autoregressive prior does, so the behaviour the screen keys on is not an ARDY artefact; but it
places that behaviour
late and spread out, so "the prompted behaviour lands early" belongs to ARDY's autoregressive
rollout and not to released G1 priors. These are two separate statements about two separate
properties and they must not be merged into one cross-prior claim.

### 2.1 Kimodo elicits the behaviour, far less often than ARDY, and never in the control arm

All rates over **all 64 assigned trials** in the arm; Wilson 95 %. "Elicited" means the reference
clears a **3 cm box somewhere along the route** in the whole-body geometry check — an
*anywhere-on-the-route* measure, never a clearance rate at a specified obstacle.

| measure | Kimodo STEP | Kimodo WALK | ARDY STEP (exp021) |
|---|---|---|---|
| elicited (3 cm box somewhere on the route) | **23/64 = 0.359** (0.253–0.482) | **0/64 = 0** (0–0.057) | 44/64 = 0.688 (0.566–0.788) |
| any positive whole-body-clearable lift | 41/64 = 0.641 (0.518–0.747) | 0/64 = 0 (0–0.057) | 49/64 = 0.766 (0.649–0.853) |
| longest run outside the support test, max over the arm | 2.97 s | **0.000 s** | — |

Kimodo elicits the behaviour at about half ARDY's rate, and the Wilson interval 0.253–0.482
excludes ARDY's 0.688. That is a result in its own right and it also **bounds what the screen
rule's 23-clip denominator can carry** (§2.3).

The control arm is a hard floor, not a soft one: **no WALK clip has any positive lift, any lift
height above 0.000 m, or any period at all outside the support test** — the maximum longest
no-support run over all 64 WALK clips is exactly 0.000 s, so 0/64 are flagged by the screen.
Paired over the 64 seeds (`summary.paired_step_minus_walk`): 23 step-only discordant pairs, **0
walk-only**, 41 concordant. Median paired difference (step − walk): lift height +0.0143 m, longest
no-support run +0.183 s, smooth-root path MAE +0.0130 m. Descriptive only — **no interval is
claimed on a paired difference** (`interval_claimed_on_difference: false`, house rule 7).

### 2.2 Timing rule → `ardy_window_attributed_to_autoregressive_rollout_context`

Denominator: the STEP clips that **have a lift position** (any positive whole-body-clearable lift),
n = 41 — the denominator the ARDY comparator is computed on (§2.4).

| event-time definition | Kimodo inside first 2.0 s | median lift time | q10–q90 | ARDY comparator (exp021, n = 49) |
|---|---|---|---|---|
| root crossing (primary) | **1/41 = 0.024** (0.004–0.126) | **3.87 s** | 2.47–5.77 s | **40/49 = 0.816** (0.686–0.900), median 1.40 s (q10–q90 0.83–2.22) |
| nominal speed | **1/41 = 0.024** (0.004–0.126) | **3.77 s** | 2.36–5.67 s | 42/49 = 0.857 (0.733–0.929), median 1.37 s |

`definitions_agree: true`: both definitions give the same fraction, the same verdict, and the same
verdict on the companion elicited denominator, where it is **0/23 = 0** (0–0.143,
`outcome_on_elicited_denominator`). Over all 64 assigned STEP trials it is 1/64 = 0.016
(0.003–0.083). The rule's branch threshold is **at or below 0.4**; the measured value is **0.024**,
so the outcome key is `ardy_window_attributed_to_autoregressive_rollout_context`.

The two distributions barely overlap: ARDY's q90 (2.22 s) sits below Kimodo's q10 (2.47 s). This is
not a marginal call in either direction — the single Kimodo clip inside the window, `s4720_step` at
1.73 s, has a lift of 0.0297 m and so misses the 3 cm elicitation bar by 0.3 mm, which is why the
elicited denominator reads 0/23.

Missing event times are handled conservatively and were not needed: a clip with a lift position but
no defined event time would count as **not** inside the window and would never be dropped from the
denominator; there are none (`n_with_lift_position_without_event_time: 0`).

### 2.3 Screen rule → `screen_generalises_to_released_g1_priors`, by a thin margin

Denominator: the **elicited** STEP clips (lift ≥ 3 cm), n = 23 — the denominator the protocol names
for this rule.

| cut | flagged | rate | Wilson 95 % | preregistered threshold |
|---|---|---|---|---|
| `max_unsupported_run_s > 0.20 s` (calibrated, primary) | **19/23** | **0.826** | **0.629–0.930** | ≥ 0.80 |
| `> 0.28 s` (the step family's post hoc optimum, secondary) | 19/23 | 0.826 | 0.629–0.930 | — |
| ARDY comparator: elicited clips above 0.20 s | 44/44 | 1.000 | 0.920–1.000 | — |

**The point estimate meets the preregistered bar; the interval does not settle it.** 0.826 clears
0.80 by 2.6 points, and the Wilson interval **straddles the threshold** (0.629–0.930). One clip
moves the verdict: 18 of 23 would be 0.783, below the bar. The denominator is 23 only because
Kimodo elicits at 0.359 (§2.1), so this is small-sample uncertainty about *how many* clips, and it
should be quoted with the interval every time.

The uncertainty is **not** about where the line sits. At the clip level the split is wide: the four
passing clips have longest runs of 0.000, 0.067, 0.067 and 0.100 s, and the 19 flagged ones run
0.367–2.967 s. **No elicited clip has a longest run between 0.100 s and 0.367 s**, so any threshold
in that band returns the same 19 of 23 — which is why the secondary cut is identical, and why the
protocol's prose wording of that cut ("0.32 s") and the receipt's implementation of it (`> 0.28 s`,
the fps-free form) cannot disagree on this corpus.

Character of the flagged runs, for the elicited clips: median longest run 0.667 s (q10–q90
0.073–1.593, max 2.967). **Ballistic ratio** — run duration ÷ the ballistic flight time of the
pelvis rise inside that run, so above 1 means the run outlasts a ballistic flight of its own rise —
is **1.35–9.85** across the 19 flagged elicited clips, against **1.3–15.6** for the 12 ARDY
references that clear 5 cm at x = 1.2 m (`outputs/analysis_trackability_contract`). The two ranges
are computed on different sets (Kimodo's is elicited-and-flagged; ARDY's is clearing-at-the-
obstacle) and the comparison is one of shape, not of matched populations. Root peak over the
elicited clips is 0.803–1.657 m against a WALK arm of 0.771–0.794 m (mean 0.779, sd 0.004).

Reported over **all 64 assigned STEP trials** so that a small accepted subset cannot read as a
rate: 31/64 = 0.484 (0.366–0.604) of the arm is flagged in total (elicited or not), and the
headline's own numerator over the same 64 trials — elicited **and** flagged — is **19/64 = 0.297**
(0.199–0.418).

### 2.4 Why the two rules use different denominators

By design, and the receipt states it (`decisions.denominator_rule`, `decisions.denominators`):

- The **timing** rule is reported over the clips that *have a lift position* — any positive
  whole-body-clearable lift — because that is the ARDY comparator this campaign is calibrated
  against: the "80–86 % inside 2.0 s" the protocol quotes is 40/49 (root crossing) and 42/49
  (nominal) over the **49** exp021 clips with any positive lift. Changing the denominator would
  compare against a number ARDY was never measured on.
- The **screen** rule keeps the **elicited** denominator, because the protocol names it explicitly:
  "≥ 80 % of Kimodo's elicited clips".
- Both are reported beside the rate over **all 64 assigned trials**, and the timing verdict is
  additionally reported on the elicited denominator, where it is unchanged.

### 2.5 Placement: the lift never lands at the specified obstacle

Both obstacle centres were fixed in the protocol before any sample (`campaign_design.obstacles`:
"staged" x = 1.2 m, "unstaged" x = 3.6 m), depth 0.20 m, graded heights. Exact whole-body clearance
at the centre, over all 64 assigned STEP trials:

| centre | 3 cm | 5 cm | 8 cm | 12 / 20 / 30 cm |
|---|---|---|---|---|
| x = 1.2 m (staged) | **0/64** (0–0.057) | **0/64** | **0/64** | 0/64 |
| x = 3.6 m (unstaged) | 4/64 = 0.063 (0.025–0.150) | 3/64 = 0.047 (0.016–0.129) | 3/64 = 0.047 | 0/64 |

This is a measured non-clearance, not a coverage artefact: the whole-body envelope **swept both
centres in 64 of 64 clips** (`summary.arms.step.coverage`, `not_reached: 0`, forward extent
7.04–7.70 m, 120 of 120 scan points swept in every clip), and the coverage guard would have
recorded an unswept centre as null rather than as a pass. The cause is placement — the elicited
clips lift at 1.91–5.79 m (median 3.29 m) and **the lowest lift anywhere in the arm, elicited or
not, sits at 1.576 m** (`lift_position_any_lift_m.min`, clip `s4720_step`). Route following is not
the excuse: smooth-root path MAE median 0.032 m (max
0.122 m), progress ratio median 0.982, better than the ≈ 0.09 m the protocol's risk section warned
about.

Three evidence levels, kept apart: **produced** 23/64, **placed at the specified obstacle** 0/64 at
x = 1.2 m, **tracked** — not attempted.

## 3. What this licenses, and what it does not

- **Licensed: the float is not an ARDY artefact.** A released, offline, non-autoregressive prior,
  conditioned on the byte-identical STEP embedding and scored by the identical committed feature
  code, puts 0.826 (0.629–0.930) of its elicited references above the calibrated 0.20 s screen,
  against 44/44 for ARDY. Read with **EXP-026** (`outputs/analysis_duck_contract/receipt.json`),
  which found the same feature ranking duck-family cutoffs above chance in every stratum, the
  screen's target behaviour has evidence of **two different kinds** — a ranking against real
  tracking outcomes for the duck family, and property recurrence with no tracking outcomes at all
  for the second prior — across **two behaviour families and two priors**. It is not a 2 × 2: only
  ARDY varies the behaviour family and only the step family varies the prior. Each row carries its
  own honest limit: EXP-026's transfer is directional and much weaker than the step
  family's (AUC 0.674 vs 0.997), and EXP-025's margin is thin with a Wilson interval that straddles
  its own threshold.
- **Licensed: the early window is scoped to the architecture.** "The prompted behaviour appears
  early in the rollout" survives as a statement about ARDY's autoregressive rollout — 0.024 against
  a ≤ 0.4 branch threshold is not a near miss — and must not be written as a property of released
  G1 priors. The corresponding withdrawal is of generality, not of the ARDY measurement.
- **Not licensed: anything about execution.** No rollout was run in this campaign. The screen
  predicts the controller evaluator's stopping rule; it does not establish zero contact force,
  physical impossibility, or anything about a robot falling, and nothing here says a Kimodo
  reference would or would not be followed by the tracker. Kimodo references have never been
  tracked at all.
- **Not licensed: a clean "generalises" headline.** Quote 0.826 with 0.629–0.930 and with the fact
  that one clip crosses the bar (18/23 = 0.783). The point estimate meets the preregistered rule;
  the interval leaves it open.
- **Not licensed: any placement or traversal claim.** 0/64 at the specified obstacle at every
  graded height. No obstacle was in any physics scene, no rollout was executed, and no number here
  is a local-traversal or a navigation rate.
- **Not licensed: merging the two verdicts.** They are answers to different questions on different
  denominators, and the reason they differ is recorded in the receipt rather than smoothed over
  (§2.4).
- **Not run: Part B.** The 84-clip reduced capability audit rerun is declared out of scope for ICRA
  2027 in the protocol and was not executed (`campaign_design.part_b_reduced_audit`); the 4.5× /
  6× counting rows stay transcript-sourced.

**Scope to state whenever these numbers are quoted.** One prior checkpoint, one prompt per arm, one
straight route, one scene, 64 seeds, kinematic only. Kimodo runs at 30 fps and ARDY at 25 fps, so
every timing comparison is in seconds and every support threshold is fps-free; no frame count is
compared across the two. The `smooth_root_2d` channel constrains the ADMM-smoothed root, not the
raw pelvis (0.06 m smoother margin), and route error is measured against `smooth_root_pos` as the
amendment requires — measuring against the raw pelvis would read high by construction and bias the
cross-prior comparison against Kimodo. The amendment quoted the 0.06 m smoother margin as the
bound; the campaign measures the gap itself at about **2.6 cm** (median per-clip
`pelvis_minus_smooth_root_mae_m` 0.026 m over the STEP arm in `rows.jsonl`; median pelvis-path MAE
0.058 m against 0.032 m for the smoothed root). The WALK arm uses Kimodo's own prompt text, so
it is an elicitation floor for this prior, not a cross-prior WALK comparison. Three departures from
the protocol's letter are recorded rather than glossed: (i) the host gate ran with **relaxed
thresholds** — 4 GiB free VRAM
and 8 GiB available RAM with `require_no_concurrent_isaac: false`, not the ≥ 12 GB / ≥ 18 GB the
protocol's Gates section names — and passed with 8,077 MiB free VRAM, 11,237 MiB available RAM and
no concurrent Isaac process observed (`host_resource_gate.generate`); (ii) the campaign was
launched from the sibling worktree `/home/linjiw/scene2motion-exp030` at the same commit
`93323a09`, from which the copied STEP cache and the exp016 threshold receipt are bound by content
hash; (iii) the protocol's prose names the secondary cut "the post hoc 0.32 s cut" and the receipt
implements it in its fps-free form, `> 0.28 s` (`decisions.screen_rule.secondary_s`). On this
corpus the two cannot disagree — no elicited clip has a longest run between 0.100 s and 0.367 s
(§2.3) — and both return 19/23.

## 4. Ledger

`outputs/exp025_kimodo_cross_prior/` — `receipt.json`, `summary.json`, `rows.jsonl` (128 rows, one
per clip with its full contract-feature vector, elicitation, exact boxes, coverage, timing and
screen prediction), `clip_records.jsonl` (128), `qpos.npz` (128 arrays), `smooth_root.npz` (128),
`noise_audit.json` (16 chunk records), `kimodo_text_cache.npz` (2 entries). Every artefact carries
a file hash and a logical hash in `evidence_anchors`. Seeds **4700–4763 are spent and must not be
reused** (`seeds_spent_and_must_not_be_reused: true`, `unlaunched_locked_seeds: []`); 4800–4927
remain reserved for EXP-027.

Claim → receipt key:

| claim | key |
|---|---|
| timing verdict, thresholds, both definitions agree | `decisions.timing_rule` (`outcome`, `thresholds`, `definitions_agree`, `definitions.*`) |
| screen verdict, 19/23, Wilson, secondary cut, ARDY 44/44 | `decisions.screen_rule` (`outcome`, `float_primary_0p20s`, `float_secondary_0p28s`, `ardy_reference`) |
| why the denominators differ | `decisions.denominator_rule`, `decisions.denominators` |
| elicitation 23/64, any-lift 41/64, WALK 0/64 | `summary.arms.step.elicitation`, `.any_lift`, `summary.arms.walk.*` |
| rates over all 64 assigned trials | `decisions.screen_rule.elicited_float_over_all_assigned_step_trials`, `.over_all_assigned_step_trials`, `summary.arms.step.lift_time_s.*.within_first_2s_over_all_assigned` |
| lift-time medians and quantiles | `summary.arms.step.lift_time_s.root_crossing`, `.nominal_speed` |
| longest-run distribution, ballistic ratios, root peak | `summary.arms.step.screen.max_unsupported_run_s_elicited`, `rows.jsonl` `reference.contract_features`, `summary.arms.step.root_z_max_m` |
| exact clearance at 1.2 m and 3.6 m, coverage | `summary.arms.step.exact_clearance`, `summary.arms.step.coverage` |
| lift positions and route fidelity | `summary.arms.step.lift_position_m`, `.route_fidelity` |
| paired step − walk, no interval on the difference | `summary.paired_step_minus_walk` |
| no tracker ran | `summary.tiers`, `decisions.scope`, `campaign_design.stage_scope` |
| identical STEP embedding, encoder never loaded | `provenance.prompt_cache.fields` (`encoder_loaded: false`, `encoder_equivalence`) |
| support thresholds from the frozen calibration | `stages.score.support_thresholds` (receipt sha `f6dba8be…`) |
| exact sample accounting | `query_accounting`, `sample_count_exact`, `stages.generate.latent_audit` |
