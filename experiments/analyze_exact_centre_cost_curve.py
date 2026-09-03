"""A0(c): exact obstacle-centred cost curve over the completed EXP-021 archive.

For every box centre on a fixed grid along the route and every graded height, the script
replays each archived exp021 clip against ``BoxHeightProbe(x, 0.20).clears(qpos, h)`` — the
exact whole-body collision endpoint (house rule 8) — and reports the per-centre hit rate with a
Wilson interval, the independent best-of-N curve, and N90/N95.  The EXP-022A paired guarded
retention (zero at every height) is overlaid at the two centres it measured, and the
recomputed exact hits at those centres are asserted equal to EXP-022A's reference rows so the
two ledgers cannot drift apart.

The receipt also carries ``tolerant_union``: for each graded height and each window radius in
{0.10, 0.25} m about the preregistered centre x = 1.2 m, the number of references that clear
the box at *any* scanned centre inside the window (each reference counted once), beside the
exact count at the centre itself.

Every centre other than the preregistered/post hoc ones is descriptive: the curve is an
addressability analysis of one archive, named as such.  Its maximum is a post hoc selection
and must never be quoted as a prospective fixed-obstacle success probability, and neither may
any tolerant-window union — the window moves the obstacle to the clip after seeing the clip.
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

SCHEMA_VERSION = "exact-centre-cost-curve-v2"
DEFAULT_OUT = Path("outputs/analysis_exact_centre_cost_curve")
DEFAULT_EXP022_DIR = Path("outputs/exp022_exact_tracking_bridge")

OBSTACLE_DEPTH_M = 0.20
CENTRE_GRID_M = tuple(float(x) for x in np.round(np.arange(0.60, 6.60 + 1e-9, 0.05), 3))
HEIGHTS_M = (0.03, 0.05, 0.08)
BEST_OF_N = (1, 2, 4, 8, 16, 32)
PREREGISTERED_CENTRE_M = 1.2
UNION_RADII_M = (0.10, 0.25)
TOLERANT_UNION_LABEL = (
    "Addressability/capability analysis, never a fixed-obstacle success probability. A "
    "reference is counted here if it clears the box at ANY scanned centre inside the window, "
    "which means the obstacle is allowed to move to that clip's own lift after the clip has "
    "been seen; the scene does not grant that tolerance. The fixed-obstacle number is "
    "exact_at_centre_successes in the same row, and only that number may be quoted as a rate "
    "of clearing an obstacle where it actually is. wilson95_descriptive_only carries no "
    "inferential claim: the success event it summarises was defined by scanning centres "
    "after seeing the clips, so the interval describes this archive's spread and is not a "
    "confidence interval for any prospective rate. The inferential interval lives on the "
    "exact-centre rows of curve.jsonl."
)

HISTORICAL_RATES_DISAMBIGUATION = {
    "why_this_is_here": (
        "Two integers in this block coincide numerically with the voided historical EXP-021 "
        "rates one height-step away, so any citation must name BOTH the height and the "
        "window radius."
    ),
    "historical": {
        "source": "docs/ramp-exp021-addressable-region-2026-09-01.md:82,88-92",
        "method": (
            "a lower-bound + placement-tolerance calculation, NOT a union of exact "
            "BoxHeightProbe replays over scanned centres"
        ),
        "rates_of_64": {"h=0.03": 0.344, "h=0.05": 0.312, "h=0.08": 0.266},
        "counts_of_64": {"h=0.03": 22, "h=0.05": 20, "h=0.08": 17},
        "status": "voided for citation by CLAUDE.md's non-reproducing table",
    },
    "this_receipt": "see union_counts_by_window below, computed from this run's hit matrix",
    "the_collision": (
        "The historical 5 cm and 8 cm counts (20 and 17 of 64) reappear in this receipt as "
        "the +-0.10 m unions one height-step down, at 3 cm and 5 cm -- different quantities, "
        "at different heights, computed by a different method. Compare "
        "union_counts_by_window against the historical counts_of_64 before citing either. "
        "No row of this receipt reproduces 0.312 or 0.266, so no row of it may be described "
        "as 'the +-0.25 m tolerant rate' that those historical numbers came from."
    ),
}
ROUTE_LENGTH_M = 7.2
REFERENCE_SPEED_MPS = 0.9045
FPS = 25.0


RUN_TREE_NOTE = (
    "provenance.code describes the tree this process actually ran in, which is not always "
    "the main worktree: the locked-scoring-identity guard refuses to run against a modified "
    "scene2motion/robot.py or scene2motion/stepover_eval.py, so when the main tree carries "
    "unrelated uncommitted edits the analysis is re-run in a clean linked worktree checked "
    "out at the recorded commit, with this script copied in. dirty_paths then names exactly "
    "what differed from that commit, and is_linked_worktree says which tree it was. The "
    "numbers are bound by scoring_identity.source_sha256, inputs.exp021_locked_file_sha256 "
    "and inputs.exp021_qpos_content_sha256 -- not by the tree being clean."
)


def porcelain_paths(porcelain: str) -> list[str]:
    """Paths out of ``git status --porcelain`` lines (``XY <path>``, ``R  <old> -> <new>``).

    The two status columns and their separating space are a fixed three-character prefix, so
    the raw output must NOT be stripped before it is split -- a leading space on an unstaged
    modification is part of the record, and stripping it eats the first character of the path.
    """
    paths: set[str] = set()
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        path = line[3:] if len(line) > 3 else line.strip()
        if " -> " in path:
            path = path.split(" -> ")[-1]
        paths.add(path.strip().strip('"'))
    return sorted(paths)


def _git_state(repo: Path) -> dict[str, Any]:
    import subprocess
    def run(*args: str, raw: bool = False) -> str:
        out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                             check=False).stdout
        return out if raw else out.strip()
    porcelain = run("status", "--porcelain", raw=True)
    git_dir = run("rev-parse", "--absolute-git-dir")
    linked = bool(git_dir) and Path(git_dir).resolve() != (Path(repo) / ".git").resolve()
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(porcelain.strip()),
        "dirty_paths": porcelain_paths(porcelain),
        "tree_path": str(Path(repo).resolve()),
        "is_linked_worktree": linked,
    }


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


def tolerant_union(
    hits: np.ndarray,
    centres_m: Sequence[float],
    heights_m: Sequence[float],
    *,
    centre_m: float = PREREGISTERED_CENTRE_M,
    radii_m: Sequence[float] = UNION_RADII_M,
) -> dict[str, Any]:
    """Union over a +-r window of scanned centres — capability, never placement.

    For each graded height and each window radius, count the references that clear the box at
    *any* scanned centre with ``|x - centre_m| <= r``.  A reference that clears at several
    centres inside the window is counted once.  The exact count at ``centre_m`` travels in the
    same row so the tolerant number can never be read as a fixed-obstacle success probability;
    the block's ``label`` says so in words.  Fails closed if the window centre is off the
    scanned grid, if the hit matrix has the wrong shape, or if there are no clips.  The
    empty-window, union-below-exact and non-monotone-in-radius raises are invariants that
    cannot fire once the centre is on the grid and the radii are nested: they are defensive,
    not exercised fail-closed behaviour.
    """
    if hits.ndim != 3 or hits.shape[:2] != (len(centres_m), len(heights_m)):
        raise ValueError("hits must be (centre, height, clip)")
    total = int(hits.shape[2])
    if total == 0:
        raise ValueError("no clips")
    centres = np.asarray(centres_m, dtype=float)
    on_grid = [ci for ci, x in enumerate(centres) if abs(float(x) - float(centre_m)) < 1e-9]
    if len(on_grid) != 1:
        raise ValueError(f"union centre {centre_m} m is not on the scanned grid")
    ci0 = on_grid[0]
    windows: dict[str, Any] = {}
    previous: tuple[float, dict[str, int]] | None = None
    for radius in sorted(float(r) for r in radii_m):
        mask = np.abs(centres - float(centre_m)) <= radius + 1e-9
        if not mask.any():  # invariant; unreachable once the centre is on the grid
            raise ValueError(f"window +-{radius} m around {centre_m} m has no scanned centre")
        counts: dict[str, int] = {}
        by_height: dict[str, Any] = {}
        for hi, h in enumerate(heights_m):
            cleared_anywhere = np.asarray(hits[mask, hi, :]).any(axis=0)
            successes = int(cleared_anywhere.sum())
            exact = int(hits[ci0, hi].sum())
            if successes < exact:  # invariant; a union over centres including centre_m
                raise ValueError(
                    f"union at r={radius} h={h} is below the exact count at {centre_m} m")
            low, high = e21.wilson_interval(successes, total)
            counts[f"h={h:g}"] = successes
            by_height[f"h={h:g}"] = {
                "height_m": float(h),
                "union_successes": successes,
                "total": total,
                "union_rate": successes / total,
                # Descriptive only: the event was defined by scanning centres after seeing
                # the clips, so this interval carries no inferential claim (house rule 8).
                "wilson95_descriptive_only": [low, high],
                "exact_at_centre_successes": exact,
                "exact_at_centre_rate": exact / total,
                "extra_references_bought_by_the_window": successes - exact,
            }
        if previous is not None:
            prev_radius, prev_counts = previous
            for key, value in counts.items():
                if value < prev_counts[key]:  # invariant; the radii are sorted and nested
                    raise ValueError(
                        f"union at r={radius} is smaller than at r={prev_radius} for {key}")
        previous = (radius, counts)
        windows[f"r={radius:.2f}"] = {
            "radius_m": radius,
            "window_m": [round(float(centre_m) - radius, 6),
                         round(float(centre_m) + radius, 6)],
            "scanned_centres_m": [float(x) for x in centres[mask]],
            "n_scanned_centres": int(mask.sum()),
            "by_height": by_height,
        }
    disambiguation = dict(HISTORICAL_RATES_DISAMBIGUATION)
    disambiguation["union_counts_by_window"] = {
        key: {h: row["union_successes"] for h, row in window["by_height"].items()}
        for key, window in windows.items()
    }
    disambiguation["exact_counts_at_centre"] = {
        h: row["exact_at_centre_successes"]
        for h, row in next(iter(windows.values()))["by_height"].items()
    }
    return {
        "label": TOLERANT_UNION_LABEL,
        "historical_rates_disambiguation": disambiguation,
        "definition": (
            "union_successes = the number of archived references for which "
            "BoxHeightProbe(x, 0.20).clears(qpos, h) is true at at least one scanned grid "
            "centre x with |x - centre_m| <= radius_m; each reference counts once however many "
            "centres inside the window it clears at."
        ),
        "centre_m": float(centre_m),
        "centre_provenance": (
            "1.2 m is the post hoc maximum of this same curve at 5 and 8 cm and the centre "
            "preregistered for EXP-024; the window is a tolerance laid on top of it, not a "
            "second, independent measurement"
        ),
        "total": total,
        "windows": windows,
    }


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
    union_centre_m: float = PREREGISTERED_CENTRE_M,
    union_radii_m: Sequence[float] = UNION_RADII_M,
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
                       "run_tree_note": RUN_TREE_NOTE,
                       "this_script_sha256": e21.file_sha256(Path(__file__)),
                       "e21_analyzer_sha256": e21.file_sha256(Path(e21.__file__))},
        "consistency_with_exp022a_reference_rows": consistency,
        "n_clips": len(keys),
        "n_centres": len(centres_m),
        "n_heights": len(heights_m),
        "n_exact_clears_calls": int(len(centres_m) * len(heights_m) * len(keys)),
        "extrema_post_hoc": curve_extrema(rows, heights_m),
        "tolerant_union": tolerant_union(hits, centres_m, heights_m,
                                         centre_m=union_centre_m, radii_m=union_radii_m),
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
        "tolerant_union": result["tolerant_union"],
        "at_exp022a_centres": result["at_exp022a_centres"],
        "wall_clock_s": result["wall_clock_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
