# EXP-028 protocol — termination-free SONIC rollouts and physics-seed re-roll

**Status:** preregistered (2026-09-02 07:30 EDT, before the first launch; draft written
2026-09-01 evening after the internal review of the plan of record). The committed sha256 of
this file is bound into the receipt. No new ARDY samples; consumes archived clips only.

## Why

Every "tracked" outcome in the paper so far means "SONIC's evaluation configuration ended the
episode on a tracking-error threshold". In EXP-022A no terminated robot had fallen: at the last
valid sample the achieved pelvis was 0.56–0.95 m high and upright, and the firing term is not
logged. A reviewer who trains trackers reads "53/64 terminated" as an evaluator cutoff, not a
physical failure, and "none retain clearance" as a statement about non-arrival (every clearing
reference was cut off 0.1–1.0 m before the box). EXP-028 turns the evaluator outcome into a
physical outcome class per clip, and supplies the step family's own test–retest ceiling, which
today is borrowed from the duck family (exp1b: 51/179 re-rolls disagree on termination).

## Locked design

Part A — termination-free rollouts, physics seed 0:
- Clips: the 64 archived exp021 references (`outputs/exp021_elicited_lift_distribution_v2/qpos.npz`)
  and, once generated, the 128 EXP-024 references. Motion pickles regenerated from the archived
  qpos with `scene2motion.sonic_export.write_motion_pkl` and hash-bound.
- Launch: the EXP-022A bridge harness and checkpoint (`sonic_release/last.pt`, sha `e6bdab3f…`),
  same core source manifest (`44e98c45…`, unchanged from exp1b's `fb57e86` through the current
  HEAD; asserted), same terrain, with every tracking-error termination disabled. Because
  `eval_agent_trl.py` merges the CLI config on top of the checkpoint config, a Hydra `~` delete
  cannot remove terms inherited from `sonic_release/config.yaml`; instead raise each threshold to
  an unreachable value (`++manager_env.terminations.anchor_pos.params.threshold=1e6`, likewise
  `anchor_ori_full`, `ee_body_pos`, `foot_pos_xyz`) so only `motion_time_out` remains, and **dump
  the resolved termination config into the receipt**. A two-motion smoke launch must show the
  resolved config before the campaign launch.
- Chunks of 32 environments: 2 launches (exp021) + 4 launches (EXP-024).

Part B — physics-seed re-roll under the release evaluation terminations:
- The 64 exp021 references at physics seeds 1 and 2 with the EXP-022A configuration unchanged
  (4 launches). Per-clip termination agreement across seeds 0–2 is the step family's test–retest
  ceiling; the EXP-022A retention is restated over 3 × 12 = 36 rollouts of the 12 exact-clearing
  clips.

## Endpoints

Per clip (Part A), from the achieved-state archive replayed against the collision model:
- **Physical outcome class** (preregistered, mutually exclusive, in this order):
  `fell` — pelvis z < 0.50 m or up-vector z < 0.70 at any sample; `stalled` — never reaches
  x = 1.10 m (box front edge) within the reference duration; `walked_through` — reaches and
  passes the box position with whole-body collision at the graded height; `cleared` — passes
  within the lateral corridor, finishes beyond, and is exactly collision-free at the graded
  height (the EXP-022A guarded endpoint without the non-termination guard).
- Achieved max root x, max lateral deviation, pelvis height minimum, the calibrated contract
  features on the *achieved* trajectory (50 Hz; thresholds are in seconds), and the per-sample
  tracking-error terms recomputed offline against the 0.25 m / 1.0 rad / 0.25 m / 0.2 m
  thresholds so that the paper can state which term would have fired and when.
- Part B: terminated flag and valid length per (clip, physics seed); agreement matrix.

## Predictions and decision rules

- No prediction on the outcome-class mix; it is descriptive.
- The title may keep "Not Tracked" (or its equivalent) only if ≥ 9/12 exact-clearing references
  end as `fell` or `walked_through` with terminations disabled; if most `cleared`, the paper's
  trackability claim is rewritten as "SONIC's evaluation cutoff rejects the reference before the
  box; with the cutoff disabled the robot [outcome]", and the contract is reframed as predicting
  the evaluator, not physical failure.
- Test–retest: report the agreement fraction with a Wilson interval; if agreement < 0.80, every
  single-seed zero in the paper is labelled a lower bound only.

## Gates, budget, statistics

Clean worktree; empty output dir; tracker commit, checkpoint sha, core source manifest and
resolved termination config bound and revalidated; rows written per launch before the next
launch; every launch's return code 0 with 32/32 archives; host-resource gate (≥ 12 GB free VRAM,
≥ 18 GB available RAM, no concurrent Isaac job) recorded. Budget: 10 launches × 52–94 s ≈
10–16 GPU-minutes plus Isaac start-up. Rates with Wilson 95 % intervals; one scene; physics
seeds stated.

## Amendment, 2026-09-03 — the host-resource gate is now the measured SONIC preset

The launch condition in this protocol was inherited from the plan of record (≥ 12 GiB free VRAM,
≥ 18 GiB available RAM, no concurrent Isaac process) and had never been measured. It blocked
every SONIC campaign for a full day while a co-tenant training job held the GPU.
`experiments/probe_sonic_vram.py` measured what one launch actually needs
(`outputs/probe_sonic_vram/`, an operational probe: no seeds spent, no campaign directory
touched, nothing citable as a result): at 2, 16 and **32** environments — 32 being this
campaign's own configuration — a launch peaked at **3,631 / 3,727 / 3,769 MiB** of VRAM and
consumed about **6,810 MiB** of host RAM, completing in 49–62 s with return code 0 **beside four
concurrent Isaac processes**. VRAM is dominated by the Isaac Sim baseline and barely grows with
the environment count; host RAM is the binding resource.

The gate therefore becomes `scene2motion.host_gate.SONIC_LAUNCH_GATE` = **≥ 5,500 MiB free VRAM,
≥ 9,500 MiB available RAM, Isaac co-tenants recorded but not gated** (~1.5× the measured VRAM
peak; ≥ 2.7 GiB of RAM left after a launch's measured consumption).

This changes **when** a launch may start, not what is measured: no seed, arm, endpoint, threshold
or analysis in this protocol is affected, and the resolved termination configuration, checkpoint,
tracker commit and chunking are unchanged. Because launches now run beside co-tenant Isaac
processes, every receipt records the co-tenants observed at launch, and that co-tenancy is stated
as scope wherever the campaign's timings are reported.
