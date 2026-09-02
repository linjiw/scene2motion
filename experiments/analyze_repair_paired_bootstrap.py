"""Paired, scene-clustered comparison of two corrections vs three-sample resampling.

Descriptive reanalysis of the committed rows in
``outputs/phase4e_architecture_v2_s8/experiment.json`` (36 scenes x 8 seeds x 15 arms).  For
each proposer and endpoint it reports the mean per-scene paired difference (two corrections
minus best-of-three resampling, percentage points) with a 95 % percentile interval from 30,000
bootstrap resamples of the 36 scenes (the scene is the inference unit), plus the discordant
scene-seed counts.  It is not a new experiment and claims nothing beyond this benchmark.

Run:  $S2M_PY experiments/analyze_repair_paired_bootstrap.py
Writes outputs/analysis_repair_paired_bootstrap/summary.json (and prints the table).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/phase4e_architecture_v2_s8/experiment.json"
OUT = ROOT / "outputs/analysis_repair_paired_bootstrap/summary.json"
N_BOOT = 30_000
SEED = 0
COMPARISONS = [
    ("tcn", "collision_free", "learned (TCN): collision-free reference"),
    ("tcn", "meets_target", "learned (TCN): 18 cm margin met"),
    ("qp", "collision_free", "QP teacher: collision-free reference"),
    ("qp", "meets_target", "QP teacher: 18 cm margin met"),
    ("heuristic", "collision_free", "heuristic: collision-free reference"),
    ("heuristic", "meets_target", "heuristic: 18 cm margin met"),
]


def _truthy(value) -> float:
    return 1.0 if (value is True or str(value).lower() == "true") else 0.0


def scene_seed_matrix(rows, method: str, field: str, scenes: list[str]) -> np.ndarray:
    table: dict[str, dict[int, float]] = {s: {} for s in scenes}
    for row in rows:
        if row["method"] == method:
            table[row["scene_id"]][int(row["seed"])] = _truthy(row[field])
    return np.array([[table[s][k] for k in sorted(table[s])] for s in scenes])


def main() -> None:
    record = json.loads(SOURCE.read_text())
    rows = record["rows"]
    scenes = sorted({r["scene_id"] for r in rows})
    rng = np.random.default_rng(SEED)
    results = []
    for proposer, field, label in COMPARISONS:
        a = scene_seed_matrix(rows, f"{proposer}+2", field, scenes)
        b = scene_seed_matrix(rows, f"{proposer}-resample3", field, scenes)
        per_scene = (a - b).mean(axis=1)
        idx = rng.integers(0, len(scenes), size=(N_BOOT, len(scenes)))
        boots = 100.0 * per_scene[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        entry = {
            "proposer": proposer, "endpoint": field, "label": label,
            "arm_a": f"{proposer}+2", "arm_b": f"{proposer}-resample3",
            "rate_a": float(a.mean()), "rate_b": float(b.mean()),
            "paired_difference_pp": float(100.0 * per_scene.mean()),
            "bootstrap_95_pp": [float(lo), float(hi)],
            "discordant_a_only": int(((a == 1) & (b == 0)).sum()),
            "discordant_b_only": int(((a == 0) & (b == 1)).sum()),
        }
        results.append(entry)
        print(f"{label:42s} {entry['paired_difference_pp']:+6.1f} pp  "
              f"[{lo:+.1f}, {hi:+.1f}]  a-only {entry['discordant_a_only']:3d}  "
              f"b-only {entry['discordant_b_only']:3d}")
    summary = {
        "analysis": "paired scene-cluster bootstrap, two corrections minus best-of-three resampling",
        "descriptive_only": True,
        "n_scenes": len(scenes), "n_seeds": int(a.shape[1]), "n_boot": N_BOOT, "rng_seed": SEED,
        "source": {"path": str(SOURCE.relative_to(ROOT)),
                   "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
