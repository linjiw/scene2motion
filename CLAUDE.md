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
preregistrations E2/E3/E4, claims hygiene). Read it before proposing work.

## State of the research (2026-09-01)

**Landed finding (kinematic, v2 sampler).** Under the STEP prompt ("A person steps over an
obstacle."), exp021's 64 clips contain one dominant liftable region, usually early, with some
clips reaching the 0.40 m probe cap. The quoted median frame ≈34 and first-50 fraction remain provisional until a
committed script derives event frames from the archived qpos. Predeclared-obstacle waiting was
1/8 (exp017), route warping preserved the intended gait in 1/6 (exp018), and coherent
phase-aligned rotation packets show negative compliance (−0.26/−0.44), no matching lag in
[−40, +80], and suppress the prompt's amplitude (exp019 v7 + exp020). ARDY is not
translation-equivariant along the route. **Prompt scheduling/history handoff is still untested,**
so do not say that no conditioning operation can move event timing.

**Exploratory lead, not a landed method: stage, then select.** Exp021's historical,
uncommitted post-hoc `0.312/0.266` analysis at 5/8 cm allows the *box centre* to move anywhere
within ±0.25 m after seeing the clip. At the exact
fixed centre `x=1.2 m`, exact `BoxHeightProbe.clears` gives only 12/64 and 11/64: implied N90 is 12/13, not
7/8. The target was selected post hoc, and the findings-page staged demo moves the obstacle to a
chosen clip's lift. Preserve these as addressable-window/capability analyses; prospective fresh
seeds and exact obstacle-centred scoring are required before calling staging a method result.

**The two holes.** No prompt-elicited v2 step-over has ever been replayed through SONIC, and no
WALK→STEP history handoff has tested whether prompt onset resets the early-event clock. EXP-022A
tracks all 64 archived exp021 clips as a post-hoc bridge; EXP-023 tests delayed prompting on fresh
paired seeds. Do not run `experiments/exp016_semantic_geometric_stepover.py` unchanged: it is an
older donor/scaffold factorial and lacks the current fail-closed/resume safeguards.

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
| `scene2motion/scenes.py`, `planner.py`, `program.py` | Phase-1 scene families, A* over (x, y, body mode), 43-D constraint program (frozen audit representation) |
| `scene2motion/verify/`, `optim/`, `learn/`, `demo/` | Phase 2–4 duck-under-beam stack: generate→verify→repair→select loop, QP scheduler, TCN, demo app; v1-era except `phase4e_architecture_v2_s8` |
| `experiments/` | one script per campaign; each writes `rows.jsonl` + `receipt.json` (exact sample accounting) |
| `outputs/<campaign>/` | receipts, rows, `qpos.npz` archives (every clip, including gate failures) |
| `docs/REPORT.md` | research ledger §1–37 (§35–37 current); `docs/design.md` is the pre-RAMP log |
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
  **The obstacle is absent from Isaac in every existing SONIC path**: "executed clearance" means
  achieved qpos replayed against our collision model. SONIC eval terrain is a mm-rough trimesh
  (`sonic_release/config.yaml: terrain_type: trimesh`). A floor box could be spawned via SONIC's
  `add_table` CuboidCfg path (unvalidated).
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

## Seeds already spent (choose disjoint blocks; first free block 4500+)

900–907 (exp011); 1400–1923 (exp1c, threshold calibration); 1500–1507 (exp015); 2200–2211,
2400–2407 (exp016 defaults); 2600–2603 (donors); 2800–2807 (exp017); 3200–3215/3300–3307,
3400–3415/3500–3507, 3600–3615/3700–3707 (calibration v1–v3); 3800–3815 (exp018); 3900–3915,
4000–4031, 4100–4131, 4200–4231, 4300–4363 (exp019 pools); 4400–4463 (exp021); exp020 reused
exp019's eight selected seeds; Phase-4 demo/corpus seeds 100–107 and 5000–5299; Kimodo audit
100–105 / 200–205; physics seed 0 (exp1b).

## Numbers that do NOT reproduce from the archives (fix before citing)

| where it appears | says | archive says |
|---|---|---|
| findings page, E2 draft, REPORT §37 | fixed staged rate 0.312/0.266; N≈7–8 | those are ±0.25 m tolerant rates; exact x=1.2 is 12/64 and 11/64; N90=12/13 |
| findings-page staged demo | demonstrates prospective staging | obstacle is moved post hoc to the selected clip's lift; capability illustration only |
| findings page, E1a note, memory | nominal WALK lift "1/8" | 3/8 seeds with a 5 mm lift, 0/8 above |
| findings page, E1a note | residual displacement "1.62 m" | 1.32–1.59 m in metres; 1.61 *strides* |
| findings page, E2 draft | "gated step-over tracked at 0.375" | retracted v1 8-seed artefact; EXP-1C killed every amplitude |
| findings page ledger | "Pmin 0.042 m replicated 3×" | Q95 background replicated 3×; frozen 0.042 in v2 and v3 only |
| exp021 note, REPORT §37 | "3 of 49 lifts after frame 60", "80 % in first 50" | 4 and 80–86 % depending on frame conversion; frames are not archived and no script converts them |
| findings page | "1,152 trajectories", "~7.7 s per clip" | 1,150 unique; 7.2 s incl. scoring |
| findings page | "15 preregistered campaigns fail closed" | exp021 v1 was an interrupted run, not a refusal |
| REPORT §4.9, design §36 | arm C "asked for a foot 0.35 m higher" | request was 0.20 m |
| REPORT, paper-v0/v1 | "~34,000 generations" | unsourced; ≈37k receipt-reconstructible, 5,361 under v2 |
| exp021 doc | "best target 1.2 m" | tie across 1.05–1.25 m; chosen post hoc on the same clips |

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
snapshot `059b8007…` (denoiser sha `0c16ac26…`); `g1.xml` sha `5d76cf92…`; SONIC exp1b historical
commit `fb57e86`, current checkout observed at `dd0fd61` (bind and revalidate the exact commit and
core source hashes in every new receipt), ckpt `sonic_release/last.pt` sha `e6bdab3f…`; Kimodo checkout `1aece8c`, HF snapshot
`3020ad8c…`; GPU RTX 5080 16 GB, driver 595.84. Generation ≈0.2 s/clip batched, scoring 2–7 s/clip
(box-height profile scan dominates); SONIC 52–94 s per ≤36-env launch.
