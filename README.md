# Scene2Motion-G1

Turning NVIDIA **ARDY**'s pretrained humanoid motion prior into a scene-conditioned
*whole-body* traversal planner for the Unitree G1, and measuring honestly what that buys.

**Start with [`docs/design.md`](docs/design.md)** — findings, corrected framing, and the
experiment ladder. Section 0 explains why the obvious framing ("a G1 ducks under a beam")
is already two papers old, and what survives.

## Layout

| path | what |
|---|---|
| `scene2motion/scenes.py` | procedural scene families, built as **counterfactual ladders** (one clearance-critical dimension swept, nuisance parameters pinned by seed) |
| `scene2motion/robot.py` | exact G1 collision via MuJoCo FK on ARDY-exported qpos, using Unitree's own primitives + a measured safety margin |
| `scene2motion/constraints.py` | the ARDY-native constraint **action space** — what the frozen prior can be asked for |
| `scene2motion/planner.py` | PELVIS / STANDING / ADAPTIVE planners; A\* over `(x, y, body mode)` |
| `scene2motion/runner.py` | batched ARDY generation with a prompt-embedding cache and qpos export |
| `experiments/` | one script per experiment; each writes `rows.jsonl` + `receipt.json` |
| `outputs/body_modes.json` | body envelopes **derived from measurement**, not assumed |

## Running

```bash
source env.sh
# the Llama-3-8B text encoder must be up on CPU (it does not fit beside the model on 16 GB):
#   .venv/bin/python scripts/run_text_encoder_server.py --device cpu
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

## Headline numbers

- Whole-body traversal end-to-end success: **68.8 %**, against 24.2 % (pelvis-only) and 31.2 % (standing) (EXP-002, 128 scenes). Adaptive is **100 % collision-free on every feasible plan** — the remaining failure is refusing scenes, never colliding in them.
- On overhead-beam scenes: 12.5 % → **83.3 %**; on `beam_and_gap` (two adaptations in sequence): 0 % → **100 %**.
- Handed either of two enumerated strategies for the same aperture, the frozen prior
  instantiates both: **2.00 distinct strategies realised per ambiguous scene**, every seed
  (EXP-003). The strategies come from constrained re-planning, not from sampling.
- Duck is worth **43–53 cm** of overhead clearance; the lateral channel is saturated at
  **~5 cm**; step-over reaches ~16 cm mean but 2.8 cm worst-case.
