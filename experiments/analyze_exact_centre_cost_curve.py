"""A0(c): exact obstacle-centred cost curve over the completed EXP-021 archive.

For every box centre on a fixed grid along the route and every graded height, the script
replays each archived exp021 clip against ``BoxHeightProbe(x, 0.20).clears(qpos, h)`` — the
exact whole-body collision endpoint (house rule 8) — and reports the per-centre hit rate with a
Wilson interval, the independent best-of-N curve, and N90/N95.  The EXP-022A paired guarded
retention (zero at every height) is overlaid at the two centres it measured, and the
recomputed exact hits at those centres are asserted equal to EXP-022A's reference rows so the
two ledgers cannot drift apart.

Every centre other than the preregistered/post hoc ones is descriptive: the curve is an
addressability analysis of one archive, named as such.  Its maximum is a post hoc selection
and must never be quoted as a prospective fixed-obstacle success probability.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import analyze_exp021_exact_addressability as e21  # noqa: E402

SCHEMA_VERSION = "exact-centre-cost-curve-v1"
DEFAULT_OUT = Path("outputs/analysis_exact_centre_cost_curve")
DEFAULT_EXP022_DIR = Path("outputs/exp022_exact_tracking_bridge")

OBSTACLE_DEPTH_M = 0.20
CENTRE_GRID_M = tuple(float(x) for x in np.round(np.arange(0.60, 6.60 + 1e-9, 0.05), 3))
HEIGHTS_M = (0.03, 0.05, 0.08)
BEST_OF_N = (1, 2, 4, 8, 16, 32)
ROUTE_LENGTH_M = 7.2
REFERENCE_SPEED_MPS = 0.9045
FPS = 25.0


def _git_state(repo: Path) -> dict[str, Any]:
    import subprocess
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                              check=False).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def exact_hit_matrix(
    arrays: Mapping[str, np.ndarray],
    keys: Sequence[str],
    centres_m: Sequence[float],
    heights_m: Sequence[float],
    *,
    probe_factory: Callable[[float, float], Any],
    depth_m: float = OBSTACLE_DEPTH_M,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Boolean array (centre, height, clip) of exact whole-body clearance."""
    hits = np.zeros((len(centres_m), len(heights_m), len(keys)), dtype=bool)
    for ci, x in enumerate(centres_m):
        probe = probe_factory(float(x), float(depth_m))
        for hi, h in enumerate(heights_m):
            for ki, key in enumerate(keys):
                hits[ci, hi, ki] = bool(probe.clears(np.asarray(arrays[key], dtype=float),
                                                     float(h)))
        if progress is not None:
            progress(f"centre {x:.2f} m done ({ci + 1}/{len(centres_m)})")
    return hits


def summarize_curve(
    hits: np.ndarray,
    centres_m: Sequence[float],
    heights_m: Sequence[float],
    *,
    best_of_n: Sequence[int] = BEST_OF_N,
) -> list[dict[str, Any]]:
    """One descriptive row per (centre, height)."""
    if hits.ndim != 3 or hits.shape[:2] != (len(centres_m), len(heights_m)):
        raise ValueError("hits must be (centre, height, clip)")
    total = int(hits.shape[2])
    if total == 0:
        raise ValueError("no clips")
    rows: list[dict[str, Any]] = []
    for ci, x in enumerate(centres_m):
        for hi, h in enumerate(heights_m):
            successes = int(hits[ci, hi].sum())
            rate = successes / total
            low, high = e21.wilson_interval(successes, total)
            rows.append({
                "centre_x_m": float(x),
                "height_m": float(h),
                "nominal_arrival_frame": float(x / REFERENCE_SPEED_MPS * FPS),
                "nominal_arrival_s": float(x / REFERENCE_SPEED_MPS),
                "successes": successes,
                "total": total,
                "exact_hit_rate": rate,
                "wilson95": [low, high],
                "independent_plugin_best_of_n": {
                    f"N={n}": e21.independent_best_of_n(rate, int(n)) for n in best_of_n
                },
                "independent_plugin_n90": e21.calls_for_target(rate, 0.90),
                "independent_plugin_n95": e21.calls_for_target(rate, 0.95),
            })
    return rows


def curve_extrema(rows: Sequence[Mapping[str, Any]], heights_m: Sequence[float]) -> dict[str, Any]:
    """Post hoc maxima per height — labelled as selection, never as a prospective rate."""
    out: dict[str, Any] = {}
    for h in heights_m:
        level = [row for row in rows if row["height_m"] == float(h)]
        best = max(row["successes"] for row in level)
        tied = [row["centre_x_m"] for row in level if row["successes"] == best]
        out[f"h={h:g}"] = {
            "max_successes": int(best),
            "total": int(level[0]["total"]),
            "max_rate": best / level[0]["total"],
            "tied_centres_m": tied,
            "label": "post hoc maximum over the scanned grid on the same clips; descriptive",
        }
    return out


def load_exp022_overlay(exp022_dir: Path) -> dict[str, Any]:
    """Bind EXP-022A's summary and reference rows; return per-label centre, retention, hits."""
    summary_path = exp022_dir / "summary.json"
    rows_path = exp022_dir / "reference_rows.jsonl"
    receipt_path = exp022_dir / "receipt.json"
    for path in (summary_path, rows_path, receipt_path):
        if not path.is_file():
            raise ValueError(f"missing EXP-022A artifact: {path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete":
        raise ValueError("EXP-022A summary is not complete")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    labels: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row["obstacle_label"])
        entry = labels.setdefault(label, {"centre_x_m": float(row["obstacle_x_m"]),
                                          "exact_clears_by_seed": {}})
        if entry["centre_x_m"] != float(row["obstacle_x_m"]):
            raise ValueError(f"EXP-022A label {label} has inconsistent centres")
        entry["exact_clears_by_seed"][int(row["seed"])] = {
            str(float(h)): bool(v) for h, v in row["exact_clears"].items()}
    retention = summary["paired_reference_to_achieved_retention"]
    for label, entry in labels.items():
        if label not in retention:
            raise ValueError(f"EXP-022A retention lacks label {label}")
        entry["paired_guarded_retention"] = {
            str(h): {
                "reference_clear": int(v["reference_clear"]),
                "achieved_guarded_clear": int(v["achieved_guarded_clear"]),
                "n_paired": int(v["n_paired"]),
            } for h, v in retention[label].items()
        }
    return {
        "dir": str(exp022_dir),
        "file_sha256": {p.name: e21.file_sha256(p) for p in (summary_path, rows_path, receipt_path)},
        "interpretation_guard": summary.get("interpretation_guard"),
        "labels": labels,
    }


def assert_overlay_consistency(
    hits: np.ndarray,
    centres_m: Sequence[float],
    heights_m: Sequence[float],
    seeds: Sequence[int],
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    """The recomputed exact hits must equal EXP-022A's reference rows at its centres."""
    checks: dict[str, Any] = {}
    for label, entry in overlay["labels"].items():
        x = float(entry["centre_x_m"])
        matches = [ci for ci, c in enumerate(centres_m) if abs(float(c) - x) < 1e-9]
        if len(matches) != 1:
            raise ValueError(f"EXP-022A centre {x} m is not on the scanned grid")
        ci = matches[0]
        per_height: dict[str, Any] = {}
        for hi, h in enumerate(heights_m):
            key = str(float(h))
            mine = [bool(v) for v in hits[ci, hi]]
            theirs = [bool(entry["exact_clears_by_seed"][int(s)][key]) for s in seeds]
            if mine != theirs:
                disagree = [int(s) for s, a, b in zip(seeds, mine, theirs) if a != b]
                raise ValueError(
                    f"exact hits at {label} x={x} h={h} disagree with EXP-022A for seeds "
                    f"{disagree}")
            per_height[key] = {"successes": int(sum(mine)), "agrees_with_exp022a_rows": True}
        checks[label] = {"centre_x_m": x, "per_height": per_height}
    return checks


def run(
    *,
    archive: Path = e21.DEFAULT_ARCHIVE,
    exp022_dir: Path = DEFAULT_EXP022_DIR,
    out: Path = DEFAULT_OUT,
    centres_m: Sequence[float] = CENTRE_GRID_M,
    heights_m: Sequence[float] = HEIGHTS_M,
    probe_factory: Callable[[float, float], Any] | None = None,
    locked_manifest: Mapping[str, Any] | None = e21.LOCKED_ARCHIVE_MANIFEST,
    scoring_identity_fn: Callable[[], Mapping[str, Any]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[1]
    receipt, seeds, arrays, locked_hashes = e21._load_validated_archive(archive, locked_manifest)
    injected = probe_factory is not None
    if probe_factory is None:
        from scene2motion.stepover_eval import BoxHeightProbe
        probe_factory = BoxHeightProbe
    if scoring_identity_fn is None:
        scoring = dict(e21._locked_scoring_identity()) if not injected else {"injected_probe": True}
    else:
        scoring = dict(scoring_identity_fn())
    keys = [f"s{seed}" for seed in seeds]
    overlay = load_exp022_overlay(Path(exp022_dir))
    hits = exact_hit_matrix(arrays, keys, centres_m, heights_m, probe_factory=probe_factory,
                            progress=progress)
    consistency = assert_overlay_consistency(hits, centres_m, heights_m, seeds, overlay)
    rows = summarize_curve(hits, centres_m, heights_m)
    result = {
        "schema": SCHEMA_VERSION,
        "analysis": "exact_centre_cost_curve",
        "status": "complete",
        "post_hoc": True,
        "interpretation": (
            "Exact obstacle-centred whole-body clearance of the 64 archived exp021 STEP clips "
            "at every scanned centre and graded height (BoxHeightProbe.clears, 4 cm body "
            "margin, 0.20 m box depth). An addressability analysis of one archive on one "
            "route; its maximum is a post hoc selection on the same clips. EXP-022A's paired "
            "guarded retention at its two centres is overlaid: zero at every height."
        ),
        "definitions": {
            "exact_hit": "BoxHeightProbe(x, 0.20).clears(qpos, h) over all 200 scored frames",
            "centre_grid_m": [float(x) for x in centres_m],
            "heights_m": [float(h) for h in heights_m],
            "obstacle_depth_m": OBSTACLE_DEPTH_M,
            "best_of_n": list(BEST_OF_N),
            "nominal_arrival": "x / 0.9045 m/s * 25 fps (route-nominal, not the root trajectory)",
        },
        "inputs": {
            "exp021_archive": str(archive),
            "exp021_locked_file_sha256": locked_hashes,
            "exp021_qpos_content_sha256": receipt.get("evidence_anchors", {}).get("qpos", {}).get("content_sha256"),
            "exp022a": overlay,
        },
        "scoring_identity": scoring,
        "provenance": {"code": _git_state(repo),
                       "this_script_sha256": e21.file_sha256(Path(__file__)),
                       "e21_analyzer_sha256": e21.file_sha256(Path(e21.__file__))},
        "consistency_with_exp022a_reference_rows": consistency,
        "n_clips": len(keys),
        "n_centres": len(centres_m),
        "n_heights": len(heights_m),
        "n_exact_clears_calls": int(len(centres_m) * len(heights_m) * len(keys)),
        "extrema_post_hoc": curve_extrema(rows, heights_m),
        "at_exp022a_centres": {
            label: {
                "centre_x_m": entry["centre_x_m"],
                "reference_exact_hits": {
                    h: next(r["successes"] for r in rows
                            if abs(r["centre_x_m"] - entry["centre_x_m"]) < 1e-9
                            and r["height_m"] == float(h))
                    for h in heights_m
                },
                "paired_guarded_retention": entry["paired_guarded_retention"],
            } for label, entry in overlay["labels"].items()
        },
        "wall_clock_s": None,
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "curve.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    np.savez_compressed(out / "exact_hits.npz", hits=hits,
                        centres_m=np.asarray(centres_m, dtype=float),
                        heights_m=np.asarray(heights_m, dtype=float),
                        seeds=np.asarray(seeds, dtype=int))
    result["wall_clock_s"] = float(time.monotonic() - started)
    result["evidence_anchors"] = {
        "curve": {"path": "curve.jsonl", "n_rows": len(rows),
                  "file_sha256": e21.file_sha256(out / "curve.jsonl")},
        "exact_hits": {"path": "exact_hits.npz",
                       "file_sha256": e21.file_sha256(out / "exact_hits.npz")},
    }
    (out / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=str(e21.DEFAULT_ARCHIVE))
    parser.add_argument("--exp022-dir", default=str(DEFAULT_EXP022_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    result = run(archive=Path(args.archive), exp022_dir=Path(args.exp022_dir),
                 out=Path(args.out), progress=lambda msg: print(msg, flush=True))
    print(json.dumps({
        "status": result["status"],
        "extrema_post_hoc": result["extrema_post_hoc"],
        "at_exp022a_centres": result["at_exp022a_centres"],
        "wall_clock_s": result["wall_clock_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
