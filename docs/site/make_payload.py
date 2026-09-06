"""Assemble ``docs/site/_payload.js`` from committed artefacts only.

The findings page draws three things: skeleton tracks, MuJoCo videos and two charts.
This script is the missing link between the producers and ``build.py``:

* motion tracks come from ``outputs/demo_motions.json`` (``experiments/export_demo_motions.py``);
* videos come from ``outputs/demo_videos.json`` (``experiments/render_demo_videos.py``);
* every charted number is read here from a committed analysis ledger, never retyped:

  - the event-time histogram from ``outputs/analysis_event_frames/receipt.json``
    (root-crossing frame of the maximum-clearance region, the 49 lifting exp021 clips);
  - the exact-position clearance curve from
    ``outputs/analysis_exact_centre_cost_curve/curve.jsonl`` (each clip scored against a
    box at that exact position -- never a tolerance window around it).

It writes ``outputs/figure_data.json`` (the chart numbers with the sha256 of every input)
and then ``docs/site/_payload.js``.  Run it before ``build.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

EVENT_FRAMES = REPO / "outputs/analysis_event_frames/receipt.json"
COST_CURVE = REPO / "outputs/analysis_exact_centre_cost_curve/curve.jsonl"
DEMO_MOTIONS = REPO / "outputs/demo_motions.json"
DEMO_VIDEOS = REPO / "outputs/demo_videos.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timing_figure() -> tuple[list[dict], dict]:
    """Event-time histogram and its statistics, exactly as the committed analyser reports them.

    The stratum is the 49 exp021 clips with positive whole-body box clearance somewhere on
    the route, and the statistic is where the *root* reaches the location of maximum
    clearance -- not a detected foot-crossing event.
    """
    receipt = json.loads(EVENT_FRAMES.read_text())
    stratum = receipt["summary"]["exp021"]["lifting_all"]["A_root_crossing"]
    histogram = stratum["histogram_10_frame_bins"]
    edges = histogram["edges_frames"]
    bins = [
        {"bin": edges[i], "label": f"{edges[i]}–{edges[i + 1] - 1}", "count": count}
        for i, count in enumerate(histogram["counts"])
    ]
    stats = {
        "median_frame": stratum["frames"]["median"],
        "median_s": stratum["seconds"]["median"],
        "q10": stratum["frames"]["q10"],
        "q90": stratum["frames"]["q90"],
        "n_first50": stratum["inside_first_50_frames"]["count"],
        "first50_wilson95": stratum["inside_first_50_frames"]["wilson95_of_planned"],
        "n_after60": stratum["after_frame_60"]["count"],
        "n_lifts": stratum["n_present"],
        "n_clips": receipt["summary"]["exp021"]["planned_clips"],
        "n_lift_ge_3cm": receipt["summary"]["exp021"]["lift_ge_3cm_clips"],
        "definition": receipt["definitions"]["A_root_crossing_frame"],
    }
    return bins, stats


def clearance_figure() -> tuple[list[dict], dict]:
    """Exact-position clearance rate along the route, and where each height peaks."""
    rows = [json.loads(line) for line in COST_CURVE.read_text().splitlines() if line.strip()]
    centres = sorted({row["centre_x_m"] for row in rows})
    by_height: dict[float, dict[float, dict]] = {}
    for row in rows:
        by_height.setdefault(row["height_m"], {})[row["centre_x_m"]] = row
    curve = [
        {
            "centre_m": centre,
            "h03": by_height[0.03][centre]["successes"],
            "h05": by_height[0.05][centre]["successes"],
            "h08": by_height[0.08][centre]["successes"],
        }
        for centre in centres
    ]
    peaks = {}
    for height, per_centre in sorted(by_height.items()):
        best = max(per_centre.values(), key=lambda row: (row["successes"], -row["centre_x_m"]))
        peaks[f"{height}"] = {
            "best_centre_m": best["centre_x_m"],
            "successes": best["successes"],
            "total": best["total"],
            "n90_fresh_draws": best["independent_plugin_n90"],
            "at_1p2_m": per_centre[1.2]["successes"],
            "at_3p6_m": per_centre[3.6]["successes"],
        }
    meta = {
        "n_clips": rows[0]["total"],
        "centres": len(centres),
        "centre_step_m": round(centres[1] - centres[0], 3),
        "range_m": [centres[0], centres[-1]],
        "endpoint": "exact whole-body box clearance at that position; never a tolerance window",
        "peaks": peaks,
    }
    return curve, meta


def figure_data() -> dict:
    timing, timing_stats = timing_figure()
    clearance, clearance_meta = clearance_figure()
    return {
        "sources": {
            "event_frames": {
                "path": str(EVENT_FRAMES.relative_to(REPO)),
                "sha256": sha256(EVENT_FRAMES),
            },
            "exact_centre_cost_curve": {
                "path": str(COST_CURVE.relative_to(REPO)),
                "sha256": sha256(COST_CURVE),
            },
        },
        "timing": timing,
        "timing_stats": timing_stats,
        "clearance": clearance,
        "clearance_stats": clearance_meta,
    }


def build(write_figures: bool = True) -> Path:
    figures = figure_data()
    if write_figures:
        destination = REPO / "outputs/figure_data.json"
        destination.write_text(json.dumps(figures, separators=(",", ":")) + "\n")
        print(f"wrote {destination.relative_to(REPO)}")

    motions = json.loads(DEMO_MOTIONS.read_text())
    videos = {
        key: {
            "uri": entry["data_uri"],
            "w": entry["width"],
            "h": entry["height"],
            "bytes": entry["bytes"],
            "fps": entry["fps"],
        }
        for key, entry in json.loads(DEMO_VIDEOS.read_text()).items()
    }
    payload = {"motions": motions, "figures": figures, "videos": videos}
    out = HERE / "_payload.js"
    out.write_text("window.S2M=" + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"wrote {out.relative_to(REPO)} ({out.stat().st_size / 1024:.0f} KB)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figure-data", action="store_true",
                        help="do not rewrite outputs/figure_data.json")
    args = parser.parse_args()
    build(write_figures=not args.no_figure_data)


if __name__ == "__main__":
    main()
