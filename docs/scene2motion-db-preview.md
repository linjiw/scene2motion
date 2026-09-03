# Scene2Motion-DB release-preview protocol

**Status:** implemented and locally validated; not a public redistribution.

## Claim this artifact supports

The committed 300-scene ducking pilot can be exported as a machine-checked record corpus in
which all assigned scenes remain present, scene-feasibility refusals are distinct from generated
reference outcomes, signed overhead deficits remain distinct from collision penetration, and
every included motion payload has a content hash and array schema.

This is an artifact and accounting claim. It is not evidence that a downstream model benefits
from the data, that the split measures geometry generalization, or that any record completes
controller execution.

## Build

```bash
PYTHONPATH=. /home/linjiw/ardy/.venv/bin/python \
  experiments/export_scene2motion_db_preview.py \
  --out outputs/scene2motion_db_preview_v1
```

The default is metadata-only. `--include-clips` copies the 268 collision-free reference payloads,
but it must not be used for public distribution until dataset licensing and third-party generated
motion redistribution terms are resolved.

The command refuses an existing output path. It validates the receipt denominator and outcome
counts, row and seed identity, outcome semantics, every NPZ array name/shape/dtype/finiteness,
and the presence and content hash of every advertised clip before writing anything. The output
is built in a temporary directory and renamed into place only after validation.

A recipient can validate a built package without the source corpus:

```bash
PYTHONPATH=. /home/linjiw/ardy/.venv/bin/python \
  experiments/export_scene2motion_db_preview.py \
  --out outputs/scene2motion_db_preview_v1 \
  --validate-only
```

This checks the package hash manifest, record and outcome counts, evidence tiers, split membership,
scene-ID disjointness, and included motion payload hashes.

## Split and leakage boundary

The preview supplies an outcome-stratified 70/15/15 development split. Scene IDs are disjoint,
and hashing fixes membership deterministically. Stratification keeps the six rare `rejected`
records represented in all three partitions. Because scene geometry ranges overlap, this is not
an OOD split and cannot support a generalization claim. A public benchmark must preregister an
independent scene-family holdout before evaluating a learned consumer.

## The next discriminating dataset experiment

Train one feasibility discriminator twice under the same architecture, optimizer, budget, and
scene split: first on successful/collision-free references only, then on the complete records
including signed refusals, margin shortfalls, and residual collisions. Evaluate both on a
preregistered geometry-OOD set and report per-failure-mode recall with the all-assigned
denominator. That single-factor comparison tests whether the negative records add utility; a
training-loss curve on this preview would not.

Before that experiment, add the execution-labelled records from the frozen EXP-024 prospective
screen campaign and EXP-028 cutoff-free campaign as a separate evidence tier. Never infer those
labels from the pilot's reference geometry.
