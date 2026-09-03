"""EXP-026: does the reference screen predict duck-family cutoffs, or is it speed?

Preregistered in ``docs/ramp-exp026-duck-contract-protocol.md`` (groups, primaries, direction
and decision rule fixed in the plan of record at commit ``0379d47`` before any feature existed).
No new samples and no GPU: this re-scores the committed EXP-1B duck references on CPU.

The reference screen — reject a reference whose longest bilateral no-support run exceeds 0.20 s —
predicts the controller evaluator's stopping rule on the step family through two actuation
channels.  Both are motions that leave the ground.  This asks whether it also ranks cutoffs in
the duck family, where the motion is a crouch, against the confound the plan named in advance:
EXP-1B's 14 s clip cap forced reference speeds up to ~1.8 m/s, and speed alone may explain them.

Outcome labels were public before this analysis (REPORT §25), so this is a **post hoc analysis of
a completed campaign**, exactly like the step-family contract analysis.

Run:  $S2M_PY experiments/analyze_duck_contract.py
Writes outputs/analysis_duck_contract/{receipt.json,rows.jsonl}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments import analyze_trackability_contract as atc  # noqa: E402

SCHEMA_VERSION = "exp026-duck-contract-v1"
PROTOCOL_PATH = "docs/ramp-exp026-duck-contract-protocol.md"
ROWS_PATH = REPO / "outputs/exp1b_execution_clearance_v2/rows.jsonl"
CLIP_CACHE = REPO / "scene2motion/demo_outputs/clips"
DEFAULT_OUT = REPO / "outputs/analysis_duck_contract"

N_CLIPS = 526
FPS = 25.0
EXPECTED_CACHE_VERSION = 2
EXPECTED_NOISE_STREAM_VERSION = 2

#: group -> (primary scalar, also-reported members).  Direction is fixed: a *higher* value is
#: hypothesised to predict a cutoff, and the AUC is never flipped to whichever side scores higher.
GROUPS: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "speed": ("max_root_planar_speed", ("mean_root_planar_speed",)),
    "crouch": ("peak_dip_m", ("ref_min_overhead_m", "root_z_min")),
    "contact": ("max_unsupported_run_s", ("bilateral_flight_frac", "mean_support_feet")),
}
PRIMARY_CONTACT = "contact"
PRIMARY_SPEED = "speed"

SCREEN_S = 0.20          # the calibrated reference screen
POSTHOC_S = 0.28         # the step family's post hoc optimum (>= 0.32 s = 8 frames)
DIP_BINS: tuple[tuple[float, float], ...] = ((0.25, 0.35), (0.35, 0.45), (0.45, 0.5001))
MIN_STRATUM_N = 20
MIN_PER_OUTCOME = 5
N_BOOT = 2000
BOOT_SEED = 20260903

SOURCE_FILES = (
    PROTOCOL_PATH,
    "experiments/analyze_duck_contract.py",
    "experiments/analyze_trackability_contract.py",
    "scene2motion/stepover_eval.py",
    "scene2motion/robot.py",
)


def _repo_relative(path: Path) -> str:
    """Repo-relative when the file lives under the repo, else the absolute path (tests)."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _json_safe(value: float) -> float | None:
    """JSON has no NaN: a correlation against a constant column is reported as null."""
    value = float(value)
    return value if np.isfinite(value) else None


class DuckContractRefusal(RuntimeError):
    """A preregistered input or accounting condition failed; nothing is claimed."""


# ------------------------------------------------------------------------------ inputs


def first_rollout_per_clip(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One record per unique ``clip_key``, the first in file order (preregistered)."""
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["clip_key"])
        if key not in seen:
            seen[key] = dict(row)
    return list(seen.values())


def load_clip(cache: Path, key: str, row: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Reference qpos and its sidecar, refusing any provenance mismatch."""
    npy, meta_path = cache / f"{key}.npy", cache / f"{key}.json"
    if not npy.is_file() or not meta_path.is_file():
        raise DuckContractRefusal(f"clip {key} is missing from the cache {cache}")
    meta = json.loads(meta_path.read_text())
    if int(meta.get("cache_version", -1)) != EXPECTED_CACHE_VERSION:
        raise DuckContractRefusal(f"clip {key} has cache_version {meta.get('cache_version')}")
    if int(meta.get("noise_stream_version", -1)) != EXPECTED_NOISE_STREAM_VERSION:
        raise DuckContractRefusal(f"clip {key} is not a noise-stream-v2 clip")
    if float(meta.get("fps", 0.0)) != FPS:
        raise DuckContractRefusal(f"clip {key} has fps {meta.get('fps')}, expected {FPS}")
    if str(meta.get("scene_id")) != str(row["scene_id"]):
        raise DuckContractRefusal(
            f"clip {key} sidecar scene {meta.get('scene_id')!r} != row {row['scene_id']!r}")
    if not np.isclose(float(meta["peak_dip_m"]), float(row["peak_dip_m"]), atol=1e-9):
        raise DuckContractRefusal(
            f"clip {key} peak_dip_m {meta['peak_dip_m']} != row {row['peak_dip_m']}")
    qpos = np.load(npy)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2 or not np.isfinite(qpos).all():
        raise DuckContractRefusal(f"clip {key} qpos has shape {qpos.shape} or is non-finite")
    if int(meta.get("n_frames", len(qpos))) != len(qpos):
        raise DuckContractRefusal(f"clip {key} n_frames disagrees with the array")
    return np.asarray(qpos, dtype=float), meta


def beam_count(scene_id: str) -> int | None:
    """Beam count parsed from a scene id like ``demo_partial_beam_h0.950_w2.250_n3_g1.50``."""
    for part in str(scene_id).split("_"):
        if part.startswith("n") and part[1:].isdigit():
            return int(part[1:])
    return None


# --------------------------------------------------------------------------- statistics


def pooled_auc(values: Sequence[float], terminated: Sequence[bool]) -> float:
    """P(value of a terminated clip > value of a surviving clip), ties counted as one half."""
    v = np.asarray(values, dtype=float)
    y = np.asarray(terminated, dtype=bool)
    if v.shape != y.shape:
        raise ValueError("values and terminated must align")
    return atc.auc(v[y], v[~y])


def cluster_bootstrap_auc(values: Sequence[float], terminated: Sequence[bool],
                          scenes: Sequence[str], *, n_boot: int = N_BOOT,
                          seed: int = BOOT_SEED) -> dict[str, Any]:
    """Cluster bootstrap over scenes: resample scenes with replacement, pool their clips."""
    v = np.asarray(values, dtype=float)
    y = np.asarray(terminated, dtype=bool)
    scene_arr = np.asarray([str(s) for s in scenes])
    names = sorted(set(scene_arr.tolist()))
    index = {name: np.flatnonzero(scene_arr == name) for name in names}
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(int(n_boot)):
        picked = rng.integers(0, len(names), len(names))
        idx = np.concatenate([index[names[i]] for i in picked])
        yy = y[idx]
        if yy.all() or not yy.any():
            continue
        draws.append(atc.auc(v[idx][yy], v[idx][~yy]))
    if not draws:
        return {"ci95": None, "n_resamples_used": 0, "n_scenes": len(names)}
    return {"ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "n_resamples_used": len(draws), "n_scenes": len(names)}


def within_scene_auc(values: Sequence[float], terminated: Sequence[bool],
                     scenes: Sequence[str], *, min_per_outcome: int = MIN_PER_OUTCOME
                     ) -> dict[str, Any]:
    """AUC inside each scene, where route length and beam geometry are constant.

    Reported as the pair-count-weighted mean over evaluable scenes (a scene contributes in
    proportion to its terminated x surviving pairs) and as the unweighted mean.
    """
    v = np.asarray(values, dtype=float)
    y = np.asarray(terminated, dtype=bool)
    scene_arr = np.asarray([str(s) for s in scenes])
    per_scene: list[dict[str, Any]] = []
    for name in sorted(set(scene_arr.tolist())):
        idx = np.flatnonzero(scene_arr == name)
        yy = v[idx], y[idx]
        pos, neg = int(yy[1].sum()), int((~yy[1]).sum())
        record = {"scene_id": name, "n": len(idx), "terminated": pos, "survived": neg}
        if pos >= min_per_outcome and neg >= min_per_outcome:
            record.update({"evaluable": True, "auc": atc.auc(yy[0][yy[1]], yy[0][~yy[1]]),
                           "pairs": pos * neg})
        else:
            record.update({"evaluable": False, "auc": None, "pairs": 0,
                           "reason": f"fewer than {min_per_outcome} of an outcome"})
        per_scene.append(record)
    ok = [r for r in per_scene if r["evaluable"]]
    if not ok:
        return {"weighted_mean_auc": None, "unweighted_mean_auc": None,
                "n_evaluable_scenes": 0, "n_scenes": len(per_scene), "per_scene": per_scene,
                "min_per_outcome": int(min_per_outcome)}
    weights = np.asarray([r["pairs"] for r in ok], dtype=float)
    aucs = np.asarray([r["auc"] for r in ok], dtype=float)
    return {
        "weighted_mean_auc": float((aucs * weights).sum() / weights.sum()),
        "unweighted_mean_auc": float(aucs.mean()),
        "n_evaluable_scenes": len(ok), "n_scenes": len(per_scene),
        "total_pairs": int(weights.sum()), "min_per_outcome": int(min_per_outcome),
        "per_scene": per_scene,
    }


def cluster_bootstrap_within_scene_difference(
    a_values: Sequence[float], b_values: Sequence[float], terminated: Sequence[bool],
    scenes: Sequence[str], *, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
    min_per_outcome: int = MIN_PER_OUTCOME) -> dict[str, Any]:
    """Cluster bootstrap of the weighted within-scene AUC difference ``a - b``."""
    scene_arr = np.asarray([str(s) for s in scenes])
    names = sorted(set(scene_arr.tolist()))
    index = {name: np.flatnonzero(scene_arr == name) for name in names}
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    y = np.asarray(terminated, dtype=bool)

    def weighted(values: np.ndarray, picks: Sequence[int]) -> float | None:
        num = den = 0.0
        for i in picks:
            idx = index[names[i]]
            yy = y[idx]
            pos, neg = int(yy.sum()), int((~yy).sum())
            if pos < min_per_outcome or neg < min_per_outcome:
                continue
            pairs = float(pos * neg)
            num += atc.auc(values[idx][yy], values[idx][~yy]) * pairs
            den += pairs
        return None if den == 0 else num / den

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(int(n_boot)):
        picked = rng.integers(0, len(names), len(names))
        pa, pb = weighted(a, picked), weighted(b, picked)
        if pa is None or pb is None:
            continue
        draws.append(pa - pb)
    if not draws:
        return {"ci95": None, "n_resamples_used": 0}
    return {"ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "n_resamples_used": len(draws)}


def cluster_bootstrap_difference(a_values: Sequence[float], b_values: Sequence[float],
                                 terminated: Sequence[bool], scenes: Sequence[str],
                                 *, n_boot: int = N_BOOT, seed: int = BOOT_SEED
                                 ) -> dict[str, Any]:
    """Cluster bootstrap of the pooled AUC difference ``a - b`` on the same resamples."""
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    y = np.asarray(terminated, dtype=bool)
    scene_arr = np.asarray([str(s) for s in scenes])
    names = sorted(set(scene_arr.tolist()))
    index = {name: np.flatnonzero(scene_arr == name) for name in names}
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(int(n_boot)):
        picked = rng.integers(0, len(names), len(names))
        idx = np.concatenate([index[names[i]] for i in picked])
        yy = y[idx]
        if yy.all() or not yy.any():
            continue
        draws.append(atc.auc(a[idx][yy], a[idx][~yy]) - atc.auc(b[idx][yy], b[idx][~yy]))
    if not draws:
        return {"ci95": None, "n_resamples_used": 0}
    return {"ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "n_resamples_used": len(draws)}


def screen_table(runs_s: Sequence[float], terminated: Sequence[bool], threshold_s: float
                 ) -> dict[str, Any]:
    """Sensitivity / specificity of ``run > threshold`` with Wilson intervals."""
    v = np.asarray(runs_s, dtype=float)
    y = np.asarray(terminated, dtype=bool)
    flagged = v > float(threshold_s)
    tp, fn = int((flagged & y).sum()), int((~flagged & y).sum())
    fp, tn = int((flagged & ~y).sum()), int((~flagged & ~y).sum())
    return {
        "threshold_s": float(threshold_s),
        "flagged_terminated": tp, "terminated": tp + fn,
        "flagged_survivors": fp, "survivors": fp + tn,
        "sensitivity": (tp / (tp + fn)) if (tp + fn) else None,
        "sensitivity_wilson95": atc.wilson(tp, tp + fn) if (tp + fn) else None,
        "specificity": (tn / (tn + fp)) if (tn + fp) else None,
        "specificity_wilson95": atc.wilson(tn, tn + fp) if (tn + fp) else None,
    }


def strata_table(records: Sequence[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], Any],
                 label: str) -> dict[str, Any]:
    """Per-stratum pooled AUCs for every group primary, with the evaluability rule applied."""
    out: dict[str, Any] = {"stratified_by": label, "strata": []}
    buckets: dict[Any, list[Mapping[str, Any]]] = {}
    for record in records:
        buckets.setdefault(key(record), []).append(record)
    for name in sorted(buckets, key=lambda value: (value is None, value)):
        members = buckets[name]
        y = [bool(m["terminated"]) for m in members]
        entry: dict[str, Any] = {"stratum": name, "n": len(members),
                                 "terminated": int(sum(y)), "survived": int(len(y) - sum(y))}
        if (len(members) < MIN_STRATUM_N or entry["terminated"] < MIN_PER_OUTCOME
                or entry["survived"] < MIN_PER_OUTCOME):
            entry.update({"evaluable": False, "auc": None,
                          "reason": f"n < {MIN_STRATUM_N} or an outcome < {MIN_PER_OUTCOME}"})
        else:
            entry.update({"evaluable": True, "auc": {
                group: pooled_auc([m["features"][primary] for m in members], y)
                for group, (primary, _) in GROUPS.items()}})
        out["strata"].append(entry)
    return out


def decide(pooled: Mapping[str, Any], within: Mapping[str, Any]) -> dict[str, Any]:
    """The plan's rule: transfer only if the contact AUC exceeds the speed AUC."""
    pooled_contact = float(pooled[PRIMARY_CONTACT]["auc"])
    pooled_speed = float(pooled[PRIMARY_SPEED]["auc"])
    pooled_transfer = pooled_contact > pooled_speed
    within_contact = within[PRIMARY_CONTACT]["weighted_mean_auc"]
    within_speed = within[PRIMARY_SPEED]["weighted_mean_auc"]
    within_transfer = (None if within_contact is None or within_speed is None
                       else bool(float(within_contact) > float(within_speed)))
    if within_transfer is None:
        verdict = "pooled_only_no_evaluable_scene"
        agree = None
    else:
        agree = bool(pooled_transfer) == bool(within_transfer)
        if agree:
            verdict = ("contract_transfers_to_the_duck_family" if pooled_transfer
                       else "speed_limited_and_confounded_by_the_14s_clip_cap")
        else:
            verdict = "pooled_and_within_scene_disagree_claim_limited_to_what_they_share"
    return {
        "rule": ("plan of record, EXP-026 row: counts as contract transfer only if the "
                 "contact-feature AUC > the speed-feature AUC; else speed-limited and "
                 "confounded by the 14 s cap"),
        "pooled": {"contact_auc": pooled_contact, "speed_auc": pooled_speed,
                   "contact_minus_speed": pooled_contact - pooled_speed,
                   "transfer": bool(pooled_transfer)},
        "within_scene": {"contact_auc": within_contact, "speed_auc": within_speed,
                         "contact_minus_speed": (None if within_transfer is None else
                                                 float(within_contact) - float(within_speed)),
                         "transfer": within_transfer},
        "pooled_and_within_scene_agree": agree,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------------- run


def build_records(rows: Sequence[Mapping[str, Any]], *, cache: Path = CLIP_CACHE,
                  body: Any | None = None, thresholds: Mapping[str, Any] | None = None,
                  feature_fn: Callable[..., Mapping[str, Any]] | None = None,
                  progress: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """One scored record per unique clip: provenance, committed row fields, contract features."""
    if thresholds is None:
        thresholds = atc.load_support_thresholds()
    if feature_fn is None:
        feature_fn = atc.features
    if body is None:
        from scene2motion.robot import G1Body
        body = G1Body(None)
    sup_h = float(thresholds["support_height_m"])
    sup_v = float(thresholds["support_speed_mps"])
    records: list[dict[str, Any]] = []
    for index, row in enumerate(first_rollout_per_clip(rows)):
        key = str(row["clip_key"])
        qpos, meta = load_clip(cache, key, row)
        feats = dict(feature_fn(body, qpos, sup_h, sup_v, FPS))
        # Committed row/meta fields the protocol names inside the crouch group.
        feats["peak_dip_m"] = float(row["peak_dip_m"])
        feats["ref_min_overhead_m"] = float(row["ref_min_overhead_m"])
        records.append({
            "clip_key": key,
            "scene_id": str(row["scene_id"]),
            "beam_count": beam_count(row["scene_id"]),
            "method": str(row["method"]),
            "seed": int(meta["seed"]),
            "repair_iteration": int(meta.get("repair_iteration", 0)),
            "route_len_m": float(row["route_len_m"]),
            "n_frames": int(len(qpos)),
            "duration_s": float(len(qpos) / FPS),
            "terminated": bool(row["terminated"]),
            "progress": float(row["progress"]),
            "clip_npy_sha256": atc.sha256_file(cache / f"{key}.npy"),
            "clip_json_sha256": atc.sha256_file(cache / f"{key}.json"),
            "features": feats,
        })
        if progress is not None and (index + 1) % 50 == 0:
            progress(f"scored {index + 1} clips")
    return records


def summarise(records: Sequence[Mapping[str, Any]], *, n_boot: int = N_BOOT) -> dict[str, Any]:
    """Every preregistered endpoint, computed from the scored records."""
    y = [bool(r["terminated"]) for r in records]
    scenes = [r["scene_id"] for r in records]
    primaries = {group: [r["features"][primary] for r in records]
                 for group, (primary, _) in GROUPS.items()}

    pooled: dict[str, Any] = {}
    within: dict[str, Any] = {}
    for group, (primary, others) in GROUPS.items():
        values = primaries[group]
        pooled[group] = {
            "primary": primary, "auc": pooled_auc(values, y),
            "cluster_bootstrap": cluster_bootstrap_auc(values, y, scenes, n_boot=n_boot),
            "also_reported": {name: pooled_auc([r["features"][name] for r in records], y)
                              for name in others},
        }
        within[group] = {"primary": primary, **within_scene_auc(values, y, scenes)}

    contact, speed = primaries[PRIMARY_CONTACT], primaries[PRIMARY_SPEED]
    runs = [r["features"]["max_unsupported_run_s"] for r in records]

    def spread(name: str) -> dict[str, Any]:
        term = np.asarray([r["features"][name] for r in records if r["terminated"]], dtype=float)
        surv = np.asarray([r["features"][name] for r in records if not r["terminated"]],
                          dtype=float)
        def stat(values: np.ndarray) -> dict[str, float] | None:
            if not len(values):
                return None
            return {"n": int(len(values)), "median": float(np.median(values)),
                    "q10": float(np.quantile(values, 0.1)),
                    "q90": float(np.quantile(values, 0.9)),
                    "min": float(values.min()), "max": float(values.max())}
        return {"terminated": stat(term), "survived": stat(surv)}

    names = [primary for primary, _ in GROUPS.values()]
    matrix = np.asarray([[r["features"][name] for name in names] for r in records], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(matrix, rowvar=False)

    return {
        "n_clips": len(records),
        "n_terminated": int(sum(y)),
        "n_scenes": len(set(scenes)),
        "pooled_auc_by_group": pooled,
        "pooled_auc_is_the_leave_one_scene_out_auc": (
            "an unfitted scalar scores a held-out clip identically whichever scene is held out, "
            "so the pooled leave-one-scene-out AUC of endpoint (1) is the whole-sample AUC; "
            "the within-scene co-primary is the confound-controlled measure"),
        "within_scene_auc_by_group": within,
        "contact_minus_speed_pooled": cluster_bootstrap_difference(
            contact, speed, y, scenes, n_boot=n_boot),
        "contact_minus_speed_within_scene": cluster_bootstrap_within_scene_difference(
            contact, speed, y, scenes, n_boot=n_boot),
        "screen": {"calibrated_0p20s": screen_table(runs, y, SCREEN_S),
                   "step_family_posthoc_0p28s": screen_table(runs, y, POSTHOC_S)},
        "float_fraction": {
            "n_with_run_over_0p20s": int(sum(1 for v in runs if v > SCREEN_S)),
            "n_with_any_no_support_run": int(sum(1 for v in runs if v > 0)),
            "n": len(runs),
        },
        "strata": {
            "dip_bins": strata_table(
                records,
                lambda r: next((f"[{lo:.2f}, {hi:.2f})" for lo, hi in DIP_BINS
                                if lo <= r["features"]["peak_dip_m"] < hi), "out_of_range"),
                "peak_dip_m"),
            "route_classes": strata_table(records, lambda r: r["beam_count"], "beam_count"),
        },
        "distribution_by_outcome": {name: spread(name) for name in names},
        "primary_correlation": {
            "order": names,
            "pearson": [[_json_safe(value) for value in row] for row in correlation],
        },
        "decision": decide(pooled, within),
    }


def run(*, out: Path | str = DEFAULT_OUT, rows_path: Path = ROWS_PATH,
        cache: Path = CLIP_CACHE, n_boot: int = N_BOOT,
        body: Any | None = None, thresholds: Mapping[str, Any] | None = None,
        feature_fn: Callable[..., Mapping[str, Any]] | None = None,
        expected_n_clips: int | None = N_CLIPS,
        progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    started = time.monotonic()
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise DuckContractRefusal(f"refusing non-empty output directory {out}")
    rows = [json.loads(line) for line in Path(rows_path).read_text().splitlines() if line]
    records = build_records(rows, cache=cache, body=body, thresholds=thresholds,
                            feature_fn=feature_fn, progress=progress)
    if expected_n_clips is not None and len(records) != expected_n_clips:
        raise DuckContractRefusal(
            f"planned denominator {expected_n_clips}, scored {len(records)}")
    if thresholds is None:
        thresholds = atc.load_support_thresholds()
    summary = summarise(records, n_boot=n_boot)
    receipt = {
        "schema": SCHEMA_VERSION,
        "experiment": "exp026_duck_contract",
        "status": "complete",
        "post_hoc": True,
        "interpretation": (
            "Post hoc analysis of the completed EXP-1B duck campaign: the groups, primaries, "
            "direction and decision rule were fixed in the plan of record (0379d47) and this "
            "protocol before any feature was computed, but the outcome labels were already "
            "public. One physics seed, one rollout per clip, schema-v1 archives, one tracker. "
            "EXP-1B's 14 s clip cap is a property of that campaign, not of ducking: a negative "
            "result bounds this corpus, not the behaviour."),
        "protocol": {"path": PROTOCOL_PATH,
                     "sha256": atc.sha256_file(REPO / PROTOCOL_PATH)},
        "inputs": {
            "rows": {"path": _repo_relative(rows_path),
                     "sha256": atc.sha256_file(Path(rows_path)),
                     "n_rows": len(rows)},
            "clip_cache": {"path": _repo_relative(cache),
                           "n_clips_used": len(records)},
            "support_thresholds": dict(thresholds),
        },
        "provenance": {
            "code": atc.git_state(),
            "source_sha256": {name: atc.sha256_file(REPO / name) for name in SOURCE_FILES},
            "python": sys.version.split()[0], "numpy": np.__version__,
        },
        "design": {
            "groups": {group: {"primary": primary, "also_reported": list(others)}
                       for group, (primary, others) in GROUPS.items()},
            "direction": "higher value predicts a cutoff; the AUC is never flipped",
            "screen_threshold_s": SCREEN_S, "step_family_posthoc_threshold_s": POSTHOC_S,
            "dip_bins": [list(b) for b in DIP_BINS],
            "min_stratum_n": MIN_STRATUM_N, "min_per_outcome": MIN_PER_OUTCOME,
            "n_boot": int(n_boot), "boot_seed": BOOT_SEED,
            "unit": "one reference clip; the scene is the inference unit for every interval",
        },
        "summary": summary,
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    receipt["wall_clock_s"] = float(time.monotonic() - started)
    receipt["evidence_anchors"] = {
        "rows": {"path": "rows.jsonl", "n_rows": len(records),
                 "file_sha256": atc.sha256_file(out / "rows.jsonl")}}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    receipt = run(out=Path(args.out), n_boot=args.n_boot,
                  progress=lambda message: print(message, flush=True))
    summary = receipt["summary"]
    print(json.dumps({
        "n_clips": summary["n_clips"], "n_terminated": summary["n_terminated"],
        "pooled": {group: value["auc"] for group, value in summary["pooled_auc_by_group"].items()},
        "within_scene": {group: value["weighted_mean_auc"]
                         for group, value in summary["within_scene_auc_by_group"].items()},
        "screen": summary["screen"]["calibrated_0p20s"],
        "decision": summary["decision"],
        "wall_clock_s": receipt["wall_clock_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
