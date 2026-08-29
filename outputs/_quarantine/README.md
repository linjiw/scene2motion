# Quarantined Phase 3 artifacts — DO NOT USE

Preserved as evidence, not deleted.

## `duck_dataset_v3_MIXED_margin_train018_test012/`

A margin-0.18 rebuild was interrupted by session teardown partway through. The directory was
left with split files from two different builds:

| file | mtime | build |
|---|---|---|
| `train.npz` (n=286) | 18:12:03 | margin **0.18** |
| `dev.npz` (n=8) | 18:12:07 | margin **0.18** |
| `test.npz` (n=114) | 18:06:27 | margin **0.12** |
| `meta.json` (`margin_m: 0.12`, counts train=281) | 18:06:27 | margin **0.12** |

Training and evaluation would have been at *different clearance margins*, and `meta.json`
described neither state — it recorded 281 train samples for a file containing 286. The failure
mode is why the replacement builder writes metadata only after every split completes and
validates counts against the files on disk.

## `duck_model_v3_PROVISIONAL_trained_on_012/`

The TCN reported in the previous session (test MAE 8.1 mm, 3-beam 17.2 mm). It was trained
BEFORE the rebuild started, so it is a margin-**0.12** model throughout — self-consistent, but
trained against a margin that `verify_margin.py` subsequently showed produces actual collisions
(80 % collision-free, worst clearance -18 mm). Superseded, not corrected.
