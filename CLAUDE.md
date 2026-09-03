# CLAUDE.md — Scene2Motion-G1

Orientation for agents working in this repository. Last full audit: 2026-09-01 (Claude's 28-agent
read plus an independent handoff/endpoint audit). When this file and a doc disagree, this file is
newer; when this file and a receipt disagree, the receipt wins and this file must be fixed.

## What the project is

A frozen NVIDIA **ARDY-G1** humanoid motion prior (autoregressive diffusion, 25 fps, Horizon52
checkpoint, text prompt + sparse kinematic constraints) is driven as a scene-conditioned traversal
module for the **Unitree G1**. A MuJoCo whole-body collision model (Unitree's own primitives +
a measured 4 cm margin, `scene2motion/robot.py`) scores every clip; the **SONIC** whole-body
tracker (GR00T-WholeBodyControl) executes references in Isaac. One consumer GPU (RTX 5080, 16 GB).
Target venue: **ICRA 2027** (Seoul; deadline 2026-09-15, 8 pages including references,
double-anonymous). Fallbacks: IROS 2027 (2027-03-01), RSS 2027 (~late Jan 2027).

The plan of record is `docs/plan-2026-09-01-icra2027.md` (paper positioning, experiment schedule,
preregistrations E2/E3/E4, claims hygiene). Read it before proposing work. **Framing of record (2026-09-02, advisor review): `docs/project-goal-2026-09-02.md`** (title
*Scene2Motion: Evaluating Generated Humanoid Motions for Obstacle Traversal*, adopted abstract,
evidence levels, plain-language vocabulary; the advice itself is `docs/pi-advice-2026-09-02.md`,
the restructured manuscript `docs/paper-draft-v2-2026-09-02.md`; second review
`docs/pi-advice-2026-09-02-b.md`: shorten explanations, not definitions — "44/64" is a whole-body
3 cm box clearance somewhere on the route, not a foot height; "1.4 s" is the root reaching the
maximum-clearance location, not a foot crossing; "clearance preserved after tracking" needs corridor
passage + finish beyond + collision-free + termination rule; the screen is a predictor of the evaluator's
rule, never a physical verdict — say "upright at the last archived state", not "did not fall"; the
128 EXP-024 tracking outcomes are pending; paired repair differences from
`experiments/analyze_repair_paired_bootstrap.py`; third review `docs/pi-advice-2026-09-02-c.md`
(TEXEDO): name the conflict between "the controller can execute it" and "it solves the scene
task"; report **pool coverage separately from selected success** — for the exp021 pool the
clearing set (12) and the tracking-completing set (11) are **disjoint**, so no selector could
have succeeded (§43, `analyze_pool_coverage.py`); distinguish **local traversal** (through the
specified corridor; walking around does not count) from **navigation** (reach a destination;
walking around may be the best answer) and never share a success label; report rates over **all
assigned trials** so rejecting everything cannot read as success; organise results as questions,
not experiment IDs; EXP-029 protocol `docs/ramp-exp029-selection-vs-coverage-protocol.md`). Use its vocabulary: "reference
screen for predicted tracking cutoffs", not "trackability contract"; "correcting measured
clearance errors", not "measured-deficit repair"; "valid step passing the support test" for
EXP-024's contact-consistent column; no throughput number in the abstract. Lead with the obstacle,
separate evidence levels (produced / placed / collision-free reference / completed tracking run /
clearance preserved after tracking), keep the losing comparisons (heuristic proposer: resampling
72.6 % margin > repair 64.6 %; TCN 99.3 % collision-free is not its 37.5 % margin rate). The
older three-pillar presentation
followed the framing of 2026-09-02 (trackability contract / measured-deficit repair /
verified data engine) reconciled against receipts in `docs/framing-2026-09-02-contract-repair-engine.md`:
"execution-**audited**", never "certified"; tiered throughput, never one 10⁵ number; no τ(d) claim.
Gate thresholds: the calibrated gate flags `max_unsupported_run_s > 0.20 s` (≥ 6 frames); the post hoc
optimum is `> 0.28 s` = **≥ 0.32 s = 8 frames** (51/53, 11/11) — writing it "> 0.32 s" gives 46/53.

## State of the research (2026-09-01, EXP-030 added 2026-09-03)

**Landed finding (kinematic, v2 sampler).** Under the STEP prompt ("A person steps over an
obstacle."), 44/64 exp021 clips lift the leading foot over a whole-body-clearable box of ≥ 3 cm
(49/64 with any positive lift; the paper's "elicits" is the ≥ 3 cm count), usually early, some
reaching the 0.40 m probe cap. Event frames now come from committed code
(`experiments/analyze_event_frames.py` → `outputs/analysis_event_frames/`): root-crossing median
frame 35 (q10–q90 21–55; 1.4 s), 40/49 lifting clips inside the first 50 frames (Wilson
0.69–0.90), 4/49 after frame 60; nominal-speed conversion 42/49 and 4/49. Predeclared-obstacle waiting was
1/8 (exp017), route warping preserved the intended gait in 1/6 (exp018), and coherent
phase-aligned rotation packets show negative compliance (−0.26/−0.44), no matching lag in
[−40, +80], and suppress the prompt's amplitude (exp019 v7 + exp020). ARDY is not
translation-equivariant along the route. **EXP-023 (2026-09-01) and EXP-023b (2026-09-02) tested prompt handoff:** switching
WALK→STEP at frame 52 or 104 through the released minimum-history `autoregressive_step`
interface produced 0/8 whole-body-clearable steps in EXP-023 but **3/8 at frame 52 in EXP-023b**
(fresh seeds; latencies 0.84–3.0 s; none clears a box at the predicted centre), against 6/8 from
frame 0; pooled 3/16. "A later prompt does not elicit the step" and "tied to the rollout origin"
are **withdrawn**; say "less often, later, unplaced". EXP-023b's SQUEEZE positive control was
refused at its substrate gate (0/8 sidestep under the pinned root; SQUEEZE lifts in 5/8 instead);
delayed prompts move the post-switch joints by 0.14–0.20 rad RMS, so the handoff transmits.
Longer history crops, Horizon8 and re-issued prompts remain untested contracts — scope every
statement (`docs/ramp-exp023-prompt-handoff-result-2026-09-01.md`,
`docs/ramp-exp023b-prompt-switch-result-2026-09-02.md`).

**Exploratory lead, not a landed method: stage, then select.** Exp021's historical,
uncommitted post-hoc `0.312/0.266` analysis at 5/8 cm allows the *box centre* to move anywhere
within ±0.25 m after seeing the clip. At the exact
fixed centre `x=1.2 m`, exact `BoxHeightProbe.clears` gives only 12/64 and 11/64: implied N90 is 12/13, not
7/8. The target was selected post hoc, and the findings-page staged demo moves the obstacle to a
chosen clip's lift. Preserve these as addressable-window/capability analyses; prospective fresh
seeds and exact obstacle-centred scoring are required before calling staging a method result.

**Tracking result and remaining hole.** EXP-022A replayed all 64 archived exp021 clips through
SONIC. All 12/11 references that clear the staged 5/8 cm box were lost: guarded achieved-state
retention is 0 at every height, 53/64 rollouts terminated, and even dropping the non-termination
guard recovers no trajectory that both passes and clears. Raw achieved `exact_clears` is vacuous
for nonarrivals and must never be quoted as execution clearance. See
`docs/ramp-exp022-exact-tracking-result-2026-09-01.md`. EXP-023 then closed the last native-interface
hole (above). Both decisive tests are done; the revised plan of record
(`docs/plan-2026-09-01-icra2027.md`, evening revision) reframes the paper as a measurement paper
(addressability audit + calibrated reference-quality gate + tiered dataset) and orders the
remaining campaigns by information per engineer-hour with a Sep 6 GO/NO-GO: EXP-023b (WALK→SQUEEZE
prompt-switch positive control, must), EXP-028 (termination-free SONIC rollouts + physics-seed
re-roll, must), EXP-024 (reference-contract ablation and prospective 0.2 s-gate test, must),
EXP-024b (physical-jump tracking control, should), EXP-025 (Kimodo cross-prior; no encoder needed —
copy ARDY's cached STEP embedding), EXP-027 (prompt battery; needs the CPU encoder), EXP-026
(duck-family contract, CPU). Draft protocols are under `docs/ramp-exp02*-protocol.md`; every
SONIC protocol carries a host-resource gate (≥ 12 GB free VRAM, ≥ 18 GB available RAM) and dumps
the resolved termination config. Seeds reserved: 4600–4631 (EXP-024), 4640–4647 (EXP-023b),
4700–4763 (EXP-025), 4800–4927 (EXP-027). Do not run
`experiments/exp016_semantic_geometric_stepover.py` unchanged: it is an older donor/scaffold
factorial and lacks the current fail-closed/resume safeguards. If a campaign is killed during
analysis, finish it with a resume script that re-scores the archives through byte-identical
sources (pattern: `experiments/exp023_prompt_handoff_resume_analysis.py`), never by regenerating.

**Why the tracker's evaluator rejects the elicited step (post hoc, CPU, 2026-09-01/02;
`experiments/analyze_trackability_contract.py` → `outputs/analysis_trackability_contract/`).** The
STEP prompt makes this prior lift the leading foot inside a **bilateral no-support phase that is
not ballistic** — a *float*, not a supported step and **not a hop** (SONIC tracks real jumps; its
README lists jumping and the checkout ships a one-leg-jump reference). In the 12 clips that exactly
clear the staged 5 cm box both feet leave the calibrated support envelope for 0.44–1.04 s (nine)
or 2.4–3.1 s (three) while the pelvis rises 0.02–0.26 m inside the run (ballistic ratio 1.3–15.6);
root peak 0.97 m mean vs 0.78 m in free-root WALK on the same route (exp023 `all_walk`). The
longest no-support run separates evaluator-terminated from surviving rollouts with AUC 0.997
(bootstrap CI 0.987–1.00; LOO logistic over 18 features 0.976); the **calibrated 0.2 s gate**
(exp016 receipt, frozen before exp021) flags 53/53 terminated (sensitivity 1.00, CI 0.93–1.00)
and passes 8/11 survivors (specificity 0.73, CI 0.43–0.90); the post hoc optimum ≥ 8 frames
(0.32 s) gives 51/53 and 11/11 and is reported as a sweep, never as the gate. It is not "did it
lift": lift height alone gives 0.92 and the run feature still separates the 20 non-lifting STEP
clips (10 terminated; AUC 0.98). 0/44 lifting clips pass the gate; the 8 that pass all survived.
**"Terminated" is an evaluator cutoff, not a fall:** SONIC's `tracking/eval` override ends the
episode on pelvis height error > 0.25 m, orientation > 1.0 rad, ankle/wrist height error > 0.25 m
or ankle position error > 0.2 m (firing term not logged); termination lands within 0.2 s of the
reference's first ≥ 0.2 s no-support onset in 47/53 (median +0.04 s; 11/12 clearing clips), and at
the last archived sample every terminated robot is upright (pelvis 0.56–0.95 m, 0/53 below 0.5 m)
while the reference's higher foot is a median 0.40 m in the air. Transfer: exp1c lift arms (144
clips, 9 survivors) AUC 0.92; cross-family logistic 0.92/0.90. The exp1c **control** arms are the
gate's calibration corpus (tracker-successful selection; 59/144 advanced < 1 m) and carry no
predictive claim. Scope: physics seed 0, one rollout per clip, achieved-state replay, one scene;
label it post hoc — **EXP-024 is its prospective test and EXP-028 (terminations disabled) gives
the physical outcome class.** **Cross-behaviour transfer (EXP-026, 2026-09-03, CPU,
`outputs/analysis_duck_contract/`):** on the 526 duck references (crouches from a different
pipeline; 344 of the first rollouts terminated) the longest no-support run ranks cutoffs at
**0.674 pooled (0.625–0.721)** and **0.694 within scene**, above 0.5 in **all seven** strata,
while the speed confound the plan feared is **refuted** (0.441, CI spans chance; survivors are
*faster*) and crouch depth is largely between-scene (0.707 → 0.565). Contact vs speed r = −0.18.
So the screen is not a stepping artefact — but it is much weaker here and is **not** a duck
accept/reject rule (specificity 0.297; 441/526 duck references are floats).
The exp023 `step_0` fresh-seed events show the same runs
(0.16–1.76 s) but were not tracked. Tracked v2 step-family references: 352 (64 + 288), not 416;
v2 generations: 5,393 (5,361 + EXP-023's 32).

**Obstacle-present execution is now possible (2026-09-03, REPORT §47).** `add_table=true` spawns a
collidable cuboid (`size` = full x/y/z extents, `pos` = centre), but the pose is rewritten per
environment on every reset, so the obstacle must be carried as **per-motion `table_pos`/`table_quat`
inside the motion pickle**. The checkout crashed on `add_table` without `add_object`
(`right_hand_wrist_links` unbound); the fix keeps the table-to-robot contact sensor inside the
`add_object` branch. Measured: a 0.30 m box at x = 0.5 m stops the robot in front of it (s4434
walks 6.06 m uncut without the box, stops at 0.29 m cut off with it). That file is in the pinned
SONIC core manifest, so **EXP-028/EXP-024 must run against the unpatched tracker and
EXP-029/EXP-030 against the patched one with their own baseline.** Host RAM (~6.8 GiB per
launch), not VRAM (~3.8 GiB), is the binding resource; an unguarded launch OOM-killed a browser.

**EXP-030 landed — the first campaign with the obstacle in the physics scene (2026-09-03,
REPORT §48, `outputs/exp030_obstacle_present/`).** Six launches of 32 over the same 64 archived
EXP-021 references EXP-022A tracked, physics seed 0, one rollout each, all three arms on the
patched worktree. Two results the next agent must not re-derive:
**(1) local traversal completion is now MEASURED, not inferred — 0/64 with a 5 cm box and 0/64
with a 20 cm box, against 1/64 (`s4434`) in the obstacle-absent control**, which is what makes the
zero attributable to the obstacle rather than to the route. Classes over all 64 assigned trials:
absent 1 completed / 54 cutoff / 9 stalled / 0 collided; present_05 0 / 44 / 0 / 20; present_20
0 / 34 / 0 / 30. Nothing fell, nothing hit a wall; **timeout is "not assessed", not zero** (no
deadline configured). **(2) the obstacle-absent replay proxy is validated on this pool** — it
predicts the obstacle-present class on 63/64, agreement 0.984 (Wilson 0.917–0.997), κ 0.964
(bootstrap 0.882–1.0); the one miss (`s4410`) is the shallowest replay intersection. So every
earlier replay-scored executed result keeps its meaning instead of inheriting a proxy caveat. The
`absent` arm also reproduces EXP-022A on 63/64 termination flags, so the fork fix is inert. Paired
progress (absent − present_05 max root x): median 0 m, 8/64 lose > 0.05 m, extremes +3.93/−4.52 m.
Scored under `traversal_eval` **version 1** (`summary.scene.evaluator_version`); a re-score under
the corrected evaluator is a separate versioned analysis. **`collided_obstacle` means something
different here than in §45:** the box was physically present and stopped the robot (penetrations
pile up at the 4 cm margin, i.e. on the box surface) instead of a replayed motion passing through
an absent box — say so wherever the two are compared. It is still the conservative clearance
model, not the table contact sensor (still unread), and two of the twenty 5 cm collisions are
sub-margin, one being `s4434`, which otherwise reached the goal inside the corridor uncut. Not
licensed: any traversal system, any generalisation beyond this route / scene / obstacle position,
anything about ducking; one physics seed, one rollout per reference.

**Dead lines — do not propose again as the next native-interface experiment:** coherent packets /
RepairNet / local response optimizer / re-anchoring through the unread rotation packet; route
warping; fixed-frame predeclared-obstacle waiting; position-channel lifts (bilateral flight);
lateral squeeze (tuck ≤0.68σ; sidling widens the G1); "certificate" wording; "~34,000
generations"; "10⁵ traversals/GPU-day" as one number; the 6× audit as headline; "ARDY/Kimodo
position themselves as planners" (strawman — CLoSD is the diffusion-as-planner claimant).
Optimization outside the released native conditioning interface exists in prior work; bound it
out rather than claiming a frozen prior is inherently uncontrollable.

**ARDY API facts that matter:** `model.__call__` accepts `init_history_sequence` (history handoff;
not combinable with `crop_history_length`), and NVIDIA's interactive demo re-prompts window by
window — a "walk, then switch the prompt" experiment is implementable (E3). The
`ARDY-G1-RP-25FPS-Horizon8` checkpoint (0.32 s windows) is in the local HF cache, untested.
EXP-023 locks the released GUI-default minimum history (one four-frame token): keep the full
accepted transcript immutable, pass only its last token to `autoregressive_step`, slice future
constraints from the same global history start, and append only the returned suffix. Do not
replace accepted frames with the model's re-encoded/reconstructed history.

## Layout

| path | what |
|---|---|
| `scene2motion/robot.py` | `G1Body`: MuJoCo FK + collision vs Unitree primitives; `BODY_MARGIN=0.04`, `CLEARANCE_MARGIN=0.6` |
| `scene2motion/runner.py` | `ArdyRunner`: batched generation, prompt-embedding cache, `NOISE_STREAM_VERSION=2` per-sample noise |
| `scene2motion/constraints.py` | `ConstraintSpec` → ARDY's 5 writable channels (root_2d, root_y_pos, heading, joint rots, joint positions) |
| `scene2motion/stepover_eval.py` | `BoxHeightProbe`, `motion_metrics`, 13-gate `evaluate_local_step`, calibrated thresholds |
| `scene2motion/ramp/` | gait-phase detector, phase observability (Pmin 0.042 m), route-progress timing, coherent packets (packets are dead as a channel; the detectors are reusable) |
| `scene2motion/sonic_export.py`, `sonic_state_export.py` | qpos → SONIC motion pkl; achieved-state archive callback (schema v2) |
| `scene2motion/traversal_eval.py` | scene-level endpoint: start/goal/obstacles -> outcome (`completed`/`collided_obstacle`/`collided_wall`/`fell`/`cutoff`/`timeout`/`stalled`/`rejected`); corridor rule makes walking around a failure; rates over ALL assigned trials; EXP-028's fall constants, `exp022.score_trajectory` untouched |
| `scene2motion/scenes.py`, `planner.py`, `program.py` | Phase-1 scene families, A* over (x, y, body mode), 43-D constraint program (frozen audit representation) |
| `scene2motion/verify/`, `optim/`, `learn/`, `demo/` | Phase 2–4 duck-under-beam stack: generate→verify→repair→select loop, QP scheduler, TCN, demo app; v1-era except `phase4e_architecture_v2_s8` |
| `experiments/` | one script per campaign; each writes `rows.jsonl` + `receipt.json` (exact sample accounting). Post hoc CPU analysers: `analyze_trackability_contract.py` (float + gate), `analyze_exact_centre_cost_curve.py` (A0c), `analyze_event_frames.py` (A0a), `analyze_duck_contract.py` (EXP-026), `analyze_pool_coverage.py`, `analyze_traversal_outcomes.py`; figure scripts `fig_contract_gate.py` (Fig. 5), `fig_cost_curve.py` (Fig. 4) read committed outputs only and run under `/home/linjiw/isaaclab-install/env_isaaclab/bin/python` (the only local interpreter with matplotlib; `$S2M_PY` has none) |
| `scene2motion/host_gate.py` | shared host-resource gate (≥ 12 GiB free VRAM, ≥ 18 GiB available RAM, no Isaac process for SONIC); every EXP-02x driver calls it before touching `--out` and binds the report |
| `outputs/<campaign>/` | receipts, rows, `qpos.npz` archives (every clip, including gate failures) |
| `scene2motion/demo_outputs/clips` | **the duck-family reference cache** (8,978 content-addressed `.npy`+`.json` pairs, cache v2): the only surviving copy of EXP-1B's 526 duck references, keyed by `clip_key` in `outputs/exp1b_execution_clearance_v2/rows.jsonl` (the `run/` cache and the exp1b achieved archives are gone). Do not delete; EXP-026 depends on it |
| `docs/REPORT.md` | research ledger §1–39 (§35–39 current); `docs/design.md` is the pre-RAMP log |
| `docs/site/` | findings page ("When the Prior Steps"); build chain below |
| `docs/ramp-*.md` | preregistered protocols, results, refusals of the RAMP campaigns |

## Environment and commands

```bash
source env.sh            # S2M_ROOT, ARDY_ROOT=/home/linjiw/ardy, KIMODO_ROOT=/home/linjiw/kimodo,
                         # S2M_PY=$ARDY_ROOT/.venv/bin/python, PYTHONPATH, text-encoder env
$S2M_PY -m pytest tests -q                      # CPU tests (~400 collected); no GPU needed
$S2M_PY experiments/exp021_elicited_lift_distribution.py --out outputs/<new_dir>   # GPU campaign pattern
```

- **Prompt cache** `outputs/text_cache.npz` (gitignored) holds exactly three prompts: WALK
  "A person walks forward.", STEP "A person steps over an obstacle.", SQUEEZE "A person steps
  sideways through a narrow gap." `ArdyRunner.encode` **raises** on any other prompt. New prompts
  need the CPU Llama-3-8B service (~14 GB RAM): `$S2M_PY /home/linjiw/ardy/scripts/run_text_encoder_server.py --device cpu`,
  then `ArdyRunner(cache_path=..., text_encoder=True).encode([...])`. It does not fit beside the
  diffusion model on the 16 GB GPU.
- **SONIC**: separate checkout `/home/linjiw/lucid/GR00T-WholeBodyControl` (branch
  research/practice-utility; exp1b pinned commit `fb57e86`; HEAD has moved — re-pin in every
  receipt), python `/home/linjiw/isaaclab-install/env_isaaclab/bin/python` (`source
  /home/linjiw/isaaclab-install/env.sh`), checkpoint `sonic_release/last.pt`, launched as a
  subprocess: pattern `experiments/exp1b_execution_clearance.py::run_sonic` (adds `++seed`) and
  `experiments/exp011_tracked_addressability.py::run_sonic`; callback
  `scene2motion.sonic_state_export.SonicStateExportCallback`. 52–94 s per ≤36-env launch.
  Free ARDY from the GPU before launching Isaac (`del runner; gc.collect(); torch.cuda.empty_cache()`).
  **The obstacle is absent from Isaac in every SONIC path except EXP-030**: outside it, "executed
  clearance" means achieved qpos replayed against our collision model. SONIC eval terrain is a
  mm-rough trimesh (`sonic_release/config.yaml: terrain_type: trimesh`). A floor box **is** spawned
  via SONIC's `add_table` CuboidCfg path — validated and used by EXP-030 on the patched worktree
  (per-motion `table_pos`/`table_quat` in the motion pickle; REPORT §47–48).
- **Findings page**: `$S2M_PY experiments/export_demo_motions.py` →
  `MUJOCO_GL=glfw $S2M_PY experiments/render_demo_videos.py` (EGL/OSMesa fail here; needs ffmpeg) →
  `$S2M_PY docs/site/build.py` → `docs/site/index.html` + `artifact.html`.
- **Kimodo-G1**: checkout `/home/linjiw/kimodo` (venv `/home/linjiw/kimodo/.venv/bin/python`, no
  mujoco), HF `nvidia/Kimodo-G1-RP-v1`, 1,000-clip `data/indoor_nav_1k` corpus. The reduced-audit
  scripts are recoverable from the session transcript
  `~/.claude/projects/-home-linjiw-ardy/f4440d67-ed27-4331-be07-dc169754a80c.jsonl` (subagent
  `agent-ac06b53618f8a2379`); see `docs/kimodo-provenance-2026-08-31.md`.

## House rules (all enforced by the RAMP harnesses; keep them)

1. **Preregister, then generate.** Every campaign has a `docs/ramp-*-protocol.md` (or the plan doc)
   with arms, seeds, gates, budget and kill conditions written before the first sample.
2. **Fail closed; never rerun a gate outcome on the same seeds with a looser gate.** Diagnose,
   record the refusal (`docs/ramp-*-refusal-*.md`), resize on FRESH seeds in a fresh output dir.
   Only a harness defect is rerun, beside the preserved failed attempt.
3. **Clean worktree, empty output dir, v2 sampler.** Harnesses raise on a dirty `git status`,
   a non-empty `--out`, or `runner.noise_stream_version != 2`. Commit before you launch.
4. **Provenance is bound and revalidated**: generator HF revision + denoiser sha, ARDY runtime commit
   + source manifest, `g1.xml` sha, the threshold receipt
   `outputs/exp016_threshold_calibration/receipt.json` (sha `f6dba8be…`), tracker commit + ckpt,
   exact ordered batch plan (row IDs, seeds, prompts/spec hashes, chunk boundaries), and exact
   `actual_ardy_samples` with launched/returned accounting. Persist the ledger *before*
   generation (exp021's `persist()` pattern); write rows before any SONIC stage. Noise stream v2
   makes latent rows batch-position independent, but end-to-end GPU byte identity remains
   batch-shape/order sensitive. A 2026-09-01 live diagnostic reproduced the original first B=8
   exp021 batch and found singleton seed 4400 different, but no committed replay receipt exists;
   treat this as an operational warning, not citable evidence.
5. **Drivers live in the repo, never in a session scratchpad**, and long launches must be
   resumable (skip completed launch dirs by receipt/rc). A reboot wiped a scratchpad driver mid-run
   on 2026-08-31.
6. **v1 and v2 clips are never pooled.** Anything generated before commit `00d6e3e`
   (2026-08-31) with `seeds=` is quarantined audit evidence (the per-window latent-replay defect,
   `docs/revalidation-2026-08-30.md`).
7. **Seeds are the resampling budget within a scene; the scene is the inference unit.** Wilson
   intervals on every rate; cluster bootstrap / mixed-effects across scenes for any headline;
   descriptive paired differences at n=8 with no interval claimed on the difference.
8. **Endpoint is obstacle-centred whole-body box clearance** (collision model, margin stated),
   never swing-foot peak or clearance at *any* centre inside a tolerance window; report graded
   heights (0.03–0.30 m). A ±r addressability analysis is allowed only when named as such and
   must never supply a fixed-obstacle success probability.
9. **Free nominal arm + outcome-free constructibility probe** in every comparison (each caught an
   invalid comparison in exp019).
10. **Every throughput or dataset number names its tier**: generated / kinematically scored /
    accepted / SONIC-executed, with wall-clock and GPU.

## Working alongside another session (2026-09-03 ruling)

More than one agent session works this repo at once. Two sessions independently diagnosed and
wrote up the same obstacle-present milestone on 2026-09-03, producing a duplicated REPORT §47.
The owner's ruling:

1. **Split ownership, and hand off explicitly.** Whoever is mid-flight on a file finishes and
   commits it; the other session does not touch overlapping files until that commit lands and the
   handoff is stated. Check `git status` before editing anything you did not write.
2. **One canonical ledger section per finding.** If you find a duplicate, keep the
   better-evidenced account and fold in what the other uniquely carried; do not renumber history.
3. **A v2 re-analysis is a new, versioned analysis.** Never overwrite a receipt to reflect a
   changed rule; the original stays, and the new one says which rule produced it.
4. **Exactly one campaign poller, owned by the reporting session.** Confirm no existing owner or
   active monitor before arming (`pgrep -f 'launch_when_host_free.sh|launch_legacy_chain.sh'`).
   Use `experiments/launch_legacy_chain.sh`, which re-checks every precondition before each
   launch: clean worktree, no other owner or running campaign, tracker manifest equal to the
   frozen baseline, and the prescribed gates (**≥ 12 GiB free VRAM, ≥ 18 GiB available RAM, no
   concurrent Isaac process**). Gates are waited for, never relaxed; LUCID is never interrupted;
   a provenance or campaign failure stops the chain for inspection.

## Two tracker checkouts, and which campaign uses which

The `add_table` fix (`7c63c53`) changes a file inside the SONIC **core-source manifest** that
every receipt binds equal to EXP-022A's `44e98c45…`. Leaving it applied makes EXP-028 and
EXP-024's tracked stage refuse to launch — correctly, since they must stay comparable to
EXP-022A.

| campaign | checkout | tracker state |
|---|---|---|
| EXP-022A, EXP-024 tracked stage, EXP-028 | `/home/linjiw/lucid/GR00T-WholeBodyControl` | **unpatched**; fix reverted by `350cae1`; manifest must equal `44e98c45…` |
| EXP-029 / EXP-030 (obstacle present) | `/home/linjiw/lucid/GR00T-WBC-exp029` (worktree, branch `exp029-obstacle-present`) | **patched at `7c63c53`**; declares its own baseline |

Both the obstacle-absent and obstacle-present arms of an obstacle-present study use the *patched*
worktree, so the comparison is internally consistent. **Never switch a checkout underneath a
running poller**, and never compare results across the patch boundary.

## Seeds already spent (choose disjoint blocks; first free block 4600+)

900–907 (exp011); 1400–1923 (exp1c, threshold calibration); 1500–1507 (exp015); 2200–2211,
2400–2407 (exp016 defaults); 2600–2603 (donors); 2800–2807 (exp017); 3200–3215/3300–3307,
3400–3415/3500–3507, 3600–3615/3700–3707 (calibration v1–v3); 3800–3815 (exp018); 3900–3915,
4000–4031, 4100–4131, 4200–4231, 4300–4363 (exp019 pools); 4400–4463 (exp021); 4500–4507 (exp023); 4640–4647 (exp023b); exp020 reused
exp019's eight selected seeds; Phase-4 demo/corpus seeds 100–107 and 5000–5299; Kimodo audit
100–105 / 200–205; physics seed 0 (exp1b).

## Numbers that do NOT reproduce from the archives (fix before citing)

| where it appears | says | archive says |
|---|---|---|
| findings page, E2 draft, REPORT §37 | fixed staged rate 0.312/0.266; N≈7–8 | historical **lower-bound + tolerance** figures (the conservative `BoxHeightProbe` lower bound inside a ±0.25 m window), **not** exact-probe unions: the exact ±0.25 m unions are 24/64 (5 cm) and 23/64 (8 cm) and the exact ±0.10 m unions 17/64 and 15/64 (`outputs/analysis_exact_centre_cost_curve/receipt.json` → `tolerant_union.windows`); exact x=1.2 m is 12/64 and 11/64; N90=12/13. Do not read 0.375 ≠ 0.312 as a fresh non-reproduction — the two were computed differently. Always name **height and radius** together: 20/64 and 17/64 appear in both ledgers one height-step apart |
| findings-page staged demo | demonstrates prospective staging | obstacle is moved post hoc to the selected clip's lift; capability illustration only |
| findings page, E1a note, memory | nominal WALK lift "1/8" | 3/8 seeds with a 5 mm lift, 0/8 above |
| findings page, E1a note | residual displacement "1.62 m" | 1.32–1.59 m in metres; 1.61 *strides* |
| findings page, E2 draft | "gated step-over tracked at 0.375" | retracted v1 8-seed artefact; EXP-1C killed every amplitude |
| findings page ledger | "Pmin 0.042 m replicated 3×" | Q95 background replicated 3×; frozen 0.042 in v2 and v3 only |
| exp021 note, REPORT §37, plan (first revision) | "3 of 49 lifts after frame 60", "80 % in first 50", "43/49" | committed analyser: 4/49 after frame 60; 40/49 (82 %) by root crossing, 42/49 (86 %) by nominal conversion (`outputs/analysis_event_frames/`) |
| findings page | "1,152 trajectories", "~7.7 s per clip" | 1,150 unique; 7.2 s incl. scoring |
| findings page | "15 preregistered campaigns fail closed" | exp021 v1 was an interrupted run, not a refusal |
| REPORT §4.9, design §36 | arm C "asked for a foot 0.35 m higher" | request was 0.20 m |
| REPORT, paper-v0/v1 | "~34,000 generations" | unsourced; ≈37k receipt-reconstructible, 5,361 under v2 |
| exp021 doc | "best target 1.2 m" | tie across 1.05–1.25 m under the ±0.25 m tolerant endpoint; under the exact endpoint (`outputs/analysis_exact_centre_cost_curve/`, 0.05 m grid) 1.2 m is the unique 5 cm / 8 cm maximum (12/64, 11/64; 3 cm maximum 15/64 at 1.15 m) — still chosen post hoc on the same clips |
| first evening plan / CLAUDE.md revision | "the elicited step is a hop"; "terminates at take-off in every clearing clip" | non-ballistic float (ballistic ratio 1.3–15.6); termination within 0.2 s of the first no-support onset in 11/12 clearing clips, before it in 10/53 overall; every terminated robot upright |
| first evening plan | "416 tracked references"; "5,361 v2 generations" | 352 tracked step-family references (64 + 288; exp023's 32 untracked); 5,393 v2 generations incl. EXP-023 |

## Stale documents (banner or rewrite before anyone reads them as current)

- `README.md`: method status ends at exp017 and points at packets/RepairNet as next steps —
  the reverse of what landed.
- `docs/site/` findings page: its staged cost, timing absolutes, process counts, and dataset
  totals contain the audit errors listed above; do not cite or publish until regenerated.
- `docs/ramp-e2-protocol-draft.md`: never preregistered; its N=8 budget and v1 tracking premise
  are superseded by the exact-centre audit and the plan of record.
- `docs/paper-draft-v1-ramp.md`: C1/C2 (packets + RepairNet) are closed; keep only its evidence
  table, statistics protocol and scoping notes.
- `docs/paper-draft-v0.md`: frozen audit-shaped baseline report; v1-sampler numbers.
- `docs/ramp-e1-protocol.md`: "reviewed and ready" predates exp017 pool/exp018/exp019.
- `docs/design.md`: pre-RAMP thesis; §4 step-over row (16 cm mean / 23 cm best / 3 seeds) is
  unsupported by `outputs/exp001c`.
- Memory files under `~/.claude/projects/-home-linjiw-scene2motion/memory/`: the E1→E5 execution
  order and the three numbers above were corrected on 2026-09-01.

## External pins

ARDY repo `/home/linjiw/ardy` at `693f74d`; checkpoint `nvidia/ARDY-G1-RP-25FPS-Horizon52`
snapshot `059b8007…` (denoiser sha `0c16ac26…`); `g1.xml` sha `5d76cf92…`; SONIC **default release** `sonic_release/last.pt` sha `e6bdab3f…` is the vendor's
**motion-tracking** checkpoint per the model card (low-latency and v1.1 are teleoperation/VLA
variants, not better trackers) — keep using it and say so; **v1.1 downloaded 2026-09-02**
at `sonic_v1_1/last.pt` sha `af24831a…` for an unrun controller-generality arm. SONIC exp1b historical
commit `fb57e86`, current checkout observed at `dd0fd61` (bind and revalidate the exact commit and
core source hashes in every new receipt), ckpt `sonic_release/last.pt` sha `e6bdab3f…`; Kimodo checkout `1aece8c`, HF snapshot
`3020ad8c…`; GPU RTX 5080 16 GB, driver 595.84. Generation ≈0.2 s/clip batched, scoring 2–7 s/clip
(box-height profile scan dominates); SONIC 52–94 s per ≤36-env launch.
