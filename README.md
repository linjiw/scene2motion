# Scene2Motion-RAMP

> **Front door (2026-09-02).** Start with [`docs/SETUP.md`](docs/SETUP.md) to install
> ARDY, Isaac Lab/SONIC and this repo on a new machine and to continue the experiments, then
> `CLAUDE.md` (orientation, house rules, pins), the plan of record
> [`docs/plan-2026-09-01-icra2027.md`](docs/plan-2026-09-01-icra2027.md), and the ledger
> [`docs/REPORT.md`](docs/REPORT.md) §40–42. The current finding: the STEP prompt elicits the
> step (44/64 at ≥ 3 cm) but places it at the fixed obstacle in only 12/64, and every clearing
> clip does so inside a bilateral no-support *float* that SONIC's evaluator cuts off; a
> calibrated 0.2 s no-support gate predicts that cutoff (AUC 0.997, post hoc). EXP-024's
> prospective test of the gate is generated, scored and frozen; its SONIC stage is pending.
> Latest figures: `docs/figures/`. Everything below this banner is retained as history: the
> packet/RepairNet method it describes failed closed on 2026-09-01.

**Response-Adaptive Motion Programs for Frozen Humanoid Priors.** Scene2Motion converts a
frozen motion prior into a scene-conditioned whole-body traversal module by placing coherent,
phase-aligned adaptation programs on a route and, in the next method stage, repairing those
programs from the motion response the prior actually realizes.

> **Method status (2026-09-01).** The strict v1 step-event representation and its paired
> absolute-vs-residual exp017 harness are implemented and CPU-tested. The first real `D=1`
> preflight failed closed after two donor samples. A sequential `D=4` retry found three
> eligible donor pairs and selected seed 2603, then stopped after nominal seed 2800 could not
> satisfy the locked bounded target assignment; no final arm was generated. An exact-design
> replay reproduced the source clips and exposed the binding gate: seed 2800 had exact phase
> alignment and a valid render window, but fixed placement required a `+65`-frame shift beyond
> the locked `+/-8` bound. These are infrastructure/eligibility results, not packet outcomes.
> A reviewed candidate-pool harness froze `K=8` nominal candidates, reported addressability
> separately from the conditional arm effect, and durably accounted for every planned arm. The
> locked run found only one eligible nominal (`E/K=1/8`), selected no partial cohort, and made
> zero final-arm calls. This is negative event-placement evidence, not a packet comparison. The
> evidence and method implications are recorded in
> [`docs/ramp-e1-protocol.md`](docs/ramp-e1-protocol.md). Duck, squeeze, response optimization,
> RepairNet, cross-prior transfer, and execution-aware route cost remain later stages.

> **Revalidation in progress (2026-08-30).** The repository's per-sample seeded runner used
> the correct seed but restarted it at every 52-frame autoregressive window. Long clips made
> through `generate(seeds=...)` therefore repeated a latent window instead of advancing the
> RNG stream. The runner and clip-cache identity are now versioned at noise stream/cache v2;
> historical v1 results below are retained as audit evidence but are not confirmatory ARDY
> results until rerun. See [`docs/revalidation-2026-08-30.md`](docs/revalidation-2026-08-30.md).

**Start with [`docs/paper-draft-v1-ramp.md`](docs/paper-draft-v1-ramp.md)** for the method
framing and evidence ledger, [`docs/ramp-e1-protocol.md`](docs/ramp-e1-protocol.md) for the
next experiment, and [`docs/REPORT.md`](docs/REPORT.md) for the baseline research record.
The audit-shaped v0 draft is retained as a technical report and evaluation infrastructure,
not as the identity of the proposed paper.

## Layout

| path | what |
|---|---|
| `scene2motion/scenes.py` | procedural scene families, built as **counterfactual ladders** (one clearance-critical dimension swept, nuisance parameters pinned by seed) |
| `scene2motion/robot.py` | collision checking against the G1 simulation collision model via MuJoCo FK, with a separately measured coverage margin |
| `scene2motion/constraints.py` | the ARDY-native constraint **action space** — what the frozen prior can be asked for |
| `scene2motion/ramp/` | physical step-phase receipts, common-phase alignment, and paired coherent absolute/residual motion packets |
| `scene2motion/planner.py` | PELVIS / STANDING / ADAPTIVE planners; A\* over `(x, y, body mode)` |
| `scene2motion/runner.py` | batched ARDY generation with a prompt-embedding cache and qpos export |
| `experiments/exp017_ramp_residual_stepover.py` | fail-closed paired step-over representation pilot (`2D + K + 2NP` samples) |
| `experiments/` | one script per experiment; each writes `rows.jsonl` + `receipt.json` |
| `outputs/body_modes.json` | body envelopes **derived from measurement**, not assumed |

## Completed diagnostic replay

```bash
source env.sh
$S2M_PY experiments/exp017_ramp_residual_stepover.py \
  --out outputs/exp017_ramp_preflight_d4_n1_p1_v2 \
  --n_donors 4 --n_nominal_candidates 1 --n_seeds 1 --obstacle_x 3.6 \
  --threshold_calibration_receipt outputs/exp016_threshold_calibration/receipt.json
```

Exp017 is single-shot and refuses a dirty worktree or non-empty output directory. This replay
used the same donor seeds 2600–2603, evaluation seed 2800, fixed scene, calibration, and gates
as the failed second preflight; only failure-evidence logging changed. It stopped after nine
frozen-prior samples, with no final-arm generation. The ledger is
`outputs/exp017_ramp_preflight_d4_n1_p1_v2/`. The subsequent exploratory pilot used a
predeclared nominal-candidate pool under unchanged gates so eligibility coverage and the
conditional arm effect would remain separate endpoints.

## Completed candidate-pool preflight

```bash
source env.sh
$S2M_PY experiments/exp017_ramp_residual_stepover.py \
  --out outputs/exp017_ramp_pool_d4_k8_n2_p1 \
  --n_donors 4 --n_nominal_candidates 8 --n_seeds 2 \
  --obstacle_x 3.6 \
  --threshold_calibration_receipt outputs/exp016_threshold_calibration/receipt.json
```

This run kept donor seeds 2600--2603 and screened the complete ordered pool 2800--2807 under
the unchanged fixed-placement and `+/-8`-frame gates. It would select the first two eligible
seeds only if at least two existed. It stopped after exactly 16 frozen-prior samples because
only seed 2805 was eligible; four candidates lacked a valid target phase cycle and three failed bounded
fixed-placement assignment. No arm was generated, and the pool will not be extended. Offline
same-gate checks show that right-foot donor support and lower target-only lift thresholds do not
rescue this pool; the next method step must actively couple route progress to gait phase rather
than rejection-screen more seeds.

## Legacy baseline experiments

```bash
source env.sh
$S2M_PY experiments/exp000_geometry_audit.py        # ~5 s   geometry sanity, sets BODY_MARGIN
$S2M_PY experiments/exp001_capability_envelope.py   # ~20 s  duck / tuck / sidle envelopes
$S2M_PY experiments/exp001b_min_halfwidth.py        # ~13 s  how narrow can it get
$S2M_PY experiments/exp001c_step_over.py            # ~31 s  how tall a box it clears
$S2M_PY experiments/derive_modes.py                 #        -> outputs/body_modes.json
$S2M_PY experiments/exp002_planner_contrast.py      # ~105 s the planner contrast
$S2M_PY experiments/exp003_multimodality.py         # ~143 s strategy multimodality
```

`derive_modes.py` must be re-run after any EXP-001* change: the planner only claims
envelopes the prior was measured to reach, aggregated **worst-case over seeds**.

## Historical v1 kinematic baseline numbers (pending v2 revalidation)

- Reference-motion traversal success in the simulation collision model: **68.8 %**, against
  24.2 % (pelvis-only) and 31.2 % (standing) (EXP-002, 128 scenes). Among plans marked
  feasible, adaptive reference motions are **100 % collision-free under that model**. These
  are not SONIC execution rates.
- On overhead-beam scenes: 12.5 % → **83.3 %**; on `beam_and_gap` (two adaptations in sequence): 0 % → **100 %**.
- Handed either of two enumerated strategies for the same aperture, the frozen prior
  instantiates both: **2.00 distinct strategies realised per ambiguous scene**, every seed
  (EXP-003). The strategies come from constrained re-planning, not from sampling.
- Duck is worth **43–53 cm** of overhead clearance; the lateral channel is saturated at
  **~5 cm**; step-over reaches ~16 cm mean but 2.8 cm worst-case.
