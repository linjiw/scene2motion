# EXP-024 protocol — reference-contract ablation of the prompt-elicited step (DRAFT, not yet preregistered)

**Status:** draft written 2026-09-01 (evening) from the plan of record
(`docs/plan-2026-09-01-icra2027.md` §3). It becomes preregistered when it is committed with a
`Status: preregistered` line and its sha256 is bound into the campaign receipt before the first
sample. Seeds **4600–4631** are reserved for this campaign and must not be reused. No
result-dependent arm, threshold, seed, or endpoint change is permitted after generation starts.

## Question and scope

EXP-022A found that no prompt-elicited step-over reference (exp021 pool: STEP prompt, route
`root_xz` only, root height and heading free) retained clearance after SONIC's evaluation
terminations, and named three un-isolated causes: step amplitude, the unconstrained root
height/heading/speed distribution of that pool, and the tracker. The post-hoc contract analysis
(`outputs/analysis_trackability_contract/`) then showed that every exact-clearing clip executes
its lift inside a bilateral **no-support phase that is not ballistic** (0.44–3.12 s for a pelvis
rise of only a few centimetres to ~0.26 m — a float, not a supported step and not a hop) and that
the longest no-support run separates evaluator-terminated from surviving rollouts (AUC 0.997 on
the exp021 pool; the pipeline's calibrated 0.2 s gate flags 53/53 terminated and passes 8/11
survivors). "Terminated" is an evaluator cutoff (pelvis/orientation/ankle-wrist thresholds), not a
measured fall.

EXP-024 asks two preregistered questions on fresh seeds:

1. **Prospective contract test.** Does the post-hoc contract predict SONIC termination on clips
   it has never seen? (The exp021 analysis was post hoc; this is the confirmation.)
2. **Contract ablation.** Is the float a property of the STEP prompt regardless of the native root
   contract, or of leaving root height and heading unconstrained? Does any native contract
   produce at least one step that is contact-consistent, exactly clears the staged box, and is
   retained after tracking?

Scope: one released checkpoint (`nvidia/ARDY-G1-RP-25FPS-Horizon52`), one prompt, one straight
7.2 m route at the reference speed, one box depth (0.20 m), kinematic scoring plus SONIC
achieved-state replay with the obstacle absent from Isaac. No cross-prior, hardware, or
contact-rich execution claim.

## Locked generation design

Prompt: `A person steps over an obstacle.` (STEP, cached embedding; content hash bound).
Route: `calibrate_ramp_route_phase.route_xz_for_speed(REFERENCE_SPEED_MPS)` — 200 frames at
25 fps, 7.2 m, 0.9045 m/s. Sampler: noise stream v2, 5 diffusion steps, cfg (2.0, 2.0),
`first_heading=0.0`. Four arms differ only in the root contract:

| arm | `root_xz` | `root_y` | `heading` | replicates |
|---|---|---|---|---|
| `free` | dense route | `None` | `None` | exp021 / exp023 contract |
| `pin_y` | dense route | 0.78 m constant (nominal pelvis) | `None` | — |
| `pin_h` | dense route | `None` | 0.0 constant | — |
| `pin_yh` | dense route | 0.78 m constant | 0.0 constant | exp1c / exp015 contract |

Seeds 4600–4631 (32), identical across arms. Generation is locked into **16 calls of B = 8 =
2 seeds × 4 arms** (seeds (4600, 4601), (4602, 4603), …), so every same-seed comparison sits in
one call and receives identical per-sample noise (the exp023 pairing contract; the noise audit is
part of the receipt). 128 frozen-prior samples in total. The ledger (`rows.jsonl` skeleton,
batch plan, seeds, prompt/spec hashes) is persisted before the first call (exp021's
`persist()` pattern).

## Endpoints (all computed on the reference clips before any SONIC launch; rows written first)

Per clip, in this order, with planned denominators (a missing value is a failure, never
dropped):

1. **Elicitation** — a whole-body-clearable lift ≥ 0.03 m anywhere on the route
   (`analyze_e1a_placement.box_height_profile`, 120 scan points, the exp021 definition) with its
   position `lift_x_m` and height.
2. **Exact clearance** — `BoxHeightProbe(x, 0.20).probe(qpos)` at the staged centre **x = 1.2 m**
   and the mid-route control **x = 3.6 m**, graded at 0.03/0.05/0.08/0.12/0.20/0.30 m. Never a
   ±r tolerant centre.
3. **Contract features** — exactly the feature set of
   `experiments/analyze_trackability_contract.py::features` (calibrated support thresholds from
   the hash-locked exp016 receipt): `max_unsupported_run_s`, `bilateral_flight_frac`,
   `mean_support_feet`, `root_z_max`, `ballistic_ratio`, and the rest. The **primary
   preregistered predictor** is the pipeline's calibrated gate `max_unsupported_run_s > 0.20 s`
   (independent provenance: frozen in the exp016 receipt before exp021 existed); the post-hoc
   optimum `> 0.32 s` (8 frames) is the **secondary** predictor. **Per-clip predictions under
   both rules are written to the ledger for all 128 clips, hashed, and committed before any
   SONIC launch.** No refitting on EXP-024 data before the prospective test is scored.
4. **Local-step gates** — `stepover_eval.evaluate_local_step` at x = 1.2 m with the calibrated
   thresholds; `local_step_success` and the 13 gate booleans.
5. **Route fidelity** — progress ratio and route MAE (as exp023), to show pinning does not derail
   the rollout.
6. **Manipulation check (constructibility of the pinned arms)** — per clip, root-height MAE and
   max deviation from 0.78 m (`pin_y`, `pin_yh`) and heading MAE / range from 0 (`pin_h`,
   `pin_yh`). E1a measured root-height compliance of only +0.20/+0.14 on a 2.3 cm request, so
   pinning may not hold. **An arm is constructible only if its median `root_z_range` ≤ 0.10 m
   (`pin_y`, `pin_yh`) or its median heading range ≤ 10° (`pin_h`).** A non-constructible arm is
   reported as such and excluded from P3/P4 (its clips are still tracked and scored for P1).

Then SONIC on **all 128 clips** (never only winners): four launches of 32 environments, physics
seed 0, the EXP-022A bridge harness (`exp022_exact_tracking_bridge` pattern: motion pkl per
launch, `SonicStateExportCallback` schema v2, `terrain_type` recorded, tracker commit and core
source hashes bound and revalidated). Achieved endpoint per clip = EXP-022A's guarded retention
at x = 1.2 m and x = 3.6 m: non-terminated, passage inside the lateral corridor, finishing beyond
the obstacle, exact whole-body clearance at each graded height. Raw achieved `exact_clears`
are recorded but never reported as execution clearance.

## Preregistered predictions

- P1 (contract, primary rule 0.20 s): among all 128 clips, ≥ 90 % of those flagged by the
  calibrated gate (`> 0.20 s`) terminate, and ≤ 30 % of those it passes terminate (exp021 gave
  53/53 and 3/11). Prospective AUC of the single feature ≥ 0.90 with a bootstrap interval. The
  secondary 0.32 s rule is scored identically and reported beside it, never instead of it.
- P2 (`free` replicates exp021): elicitation in [0.55, 0.95] (exp021: 0.766, Wilson95
  [0.65, 0.85]); exact 5 cm clearance at 1.2 m in [0.06, 0.35] (exp021: 12/64).
- P3 (`pin_y`, `pin_yh`): median `root_z_max` ≤ 0.85 m and median `max_unsupported_run_s`
  shorter than `free`'s on the same seeds in ≥ 20/32 paired seeds; elicitation may fall (no
  prediction on its value).
- P4 (open): at least one arm yields ≥ 3/32 clips that are simultaneously contact-consistent
  (`local_step_success` true, `max_unsupported_run_s ≤ 0.20 s`), exactly clear 5 cm at 1.2 m,
  and are retained after tracking. **We do not predict P4 either way.**

## Decision rules (fixed in advance)

- **Contract confirmed** if P1 holds; then C2 of the paper reports the contract as
  prospectively validated. If P1 fails, the contract is reported as post hoc only and the
  failure mode (which clips) is diagnosed without refitting for the paper.
- **Prescriptive contract (GO)** if P4 holds in any constructible arm: the paper reports that
  arm's contract as the reference-quality condition under which a prompt-elicited step is
  placeable and trackable, with its rate and interval, and states the four-arm multiplicity of
  the ≥ 3/32 rule (a single arm reaching 3/32 by chance at a true rate of 0.02 has probability
  ≈ 0.03 per arm, ≈ 0.11 over four arms; the rate itself, not the pass/fail, is the reported
  quantity). **Diagnostic only (NO-GO)** otherwise: the paper states that under the four native
  root contracts tried, the prompt elicits a non-ballistic float and no clip was simultaneously
  placed, contact-consistent and retained.
- **Replication rule for the `free` arm:** it is the prospective replication of exp021's exact
  12/64 at the now-preregistered x = 1.2 m (expect ≈ 6/32, Wilson-compatible with 19 %). If its
  elicitation falls outside [0.55, 0.95] the campaign is reported as a replication failure of
  exp021 (with the batch-shape sensitivity noted in `CLAUDE.md` as the first suspect), P1 is
  still scored, and no arm is rerun.
- No arm is added, dropped or rerun on the same seeds after seeing outcomes. A harness defect is
  fixed and rerun into a fresh directory with the failed attempt preserved.

## Gates (fail closed)

Clean worktree at launch; empty output directory; `runner.noise_stream_version == 2`;
threshold receipt sha `f6dba8be…` verified; generator/runtime/model identities bound and
revalidated after generation; exact accounting 128 launched = 128 returned (a partial return
stops the campaign, is recorded, and is not topped up); tracker commit, checkpoint sha
(`e6bdab3f…`), **core source manifest equal to EXP-022A's (`44e98c45…`, unchanged since exp1b's
`fb57e86`; asserted, not assumed)** and the **resolved termination config dumped into the
receipt**; each launch's return code 0 and 32/32 rollouts archived; rows and the per-clip gate
predictions written and hashed before the SONIC stage; every row carries the seed, arm, spec
hash, qpos sha256 and its planned-denominator fields. **Host-resource gate:** launch only when
`nvidia-smi` reports ≥ 12 GB free and `free -g` reports ≥ 18 GB available, with no concurrent
Isaac job; both values are recorded in the receipt (a co-tenant training job was holding
6–7 GB of VRAM on 2026-09-01).

## Statistics

Seed is the pairing unit across arms; there is one scene, so no scene-level inference is
claimed. Wilson 95 % intervals on every per-arm rate; paired differences across arms as counts
(exact McNemar as auxiliary); the prospective contract test as a 2×2 table with Wilson intervals
on sensitivity and specificity. Physics seed 0 only: report the exp1b test–retest disagreement
(same clip, different launch: terminated differs in 51/179) as the ceiling any single-seed
tracker outcome carries. Post-hoc analyses of EXP-024 data are allowed only if labelled as such
and kept out of the decision rules.

## Budget

128 samples in 16 B=8 calls (≈ 10 s generation); reference scoring ≈ 128 × 3–6 s (profile scan
dominates) ≈ 8–13 min; SONIC 4 launches × 52–94 s ≈ 4–6 min plus Isaac start-up; ARDY must be
freed from the GPU before Isaac. Total < 30 GPU-minutes. Disk: 128 × (200×36) float32 ≈ 3.7 MB
references plus 4 achieved archives.

## Code changes required (before preregistration)

| file | change | effort |
|---|---|---|
| `experiments/exp024_reference_contract.py` (new) | driver: locked batch plan (2 seeds × 4 arms per call), spec builders for the four contracts, `persist()`-before-generate ledger, reference scoring (items 1–5), SONIC bridge reuse from `exp022_exact_tracking_bridge`, guarded achieved endpoint, receipt with predictions P1–P4 evaluated mechanically | ~1 day |
| `experiments/analyze_trackability_contract.py` | expose `features()` and the 0.30 s / 0.20 s rules as importable functions (no behaviour change) | 1 h |
| `tests/test_exp024_reference_contract.py` (new) | CPU tests: batch plan pairing, arm spec channel support (`pin_y` writes root_y_pos only; `pin_h` writes heading only), planned-denominator accounting, decision-rule evaluation on synthetic rows | 2–3 h |
| this document | flip to `Status: preregistered`, bind sha in receipt | — |

## Risks and confounds (stated in advance)

- Pinning root height at 0.78 m may suppress the step itself (a real step needs some pelvis
  motion); that is an informative outcome, not a defect.
- The SONIC checkout HEAD has moved since exp1b (practice-utility commits only); the tracker's
  core source manifest is unchanged from `fb57e86` through the current HEAD, so every SONIC
  result in the paper shares one core source, one checkpoint and one termination configuration.
  The receipt asserts this rather than assuming it.
- The obstacle is absent from Isaac; "retained" means achieved-state replay against the collision
  model, on a mm-rough trimesh floor; "terminated" is the evaluator's cutoff, whose physical
  meaning is established separately by EXP-028 (termination-free rollouts on these same clips).
- Elicitation under `free` may not replicate exp021 exactly (batch-shape sensitivity noted in
  `CLAUDE.md`); P2's interval is deliberately wide.
