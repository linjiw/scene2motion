"""Scene-level traversal outcomes for the tracked stepping pool (start, goal, obstacle).

Re-scores the archived EXP-022A achieved states against a full traversal problem — a start, a
goal, and a corridor-spanning obstacle at the specified position — and reports the outcome
breakdown the project reports going forward: completed / collided with the obstacle / collided
with a wall / fell / evaluator cutoff / timeout / stalled, over **all assigned trials**.

Why this adds something the frozen bridge does not: the bridge reports a boolean
``achieved_replay_clear_after_passing``, so a rollout that reached the obstacle and intersected
it is indistinguishable from one that never arrived. Both are simply "not retained". Here they
are separate classes, which is what tells a reader whether the pipeline fails by colliding or by
never getting there.

**Scope, and it matters.** These rollouts were tracked with the obstacle *absent* from the
physics scene, so ``collided_obstacle`` means the achieved trajectory intersects the obstacle's
geometry when replayed against it — the robot never actually felt the box, and its controller
was never perturbed by one. That is a statement about the recorded motion, not about contact
dynamics. The obstacle-present version is EXP-029
(`docs/ramp-exp029-selection-vs-coverage-protocol.md`). One route, one scene, physics seed 0,
one rollout per reference; descriptive, no new samples.

Run:  $S2M_PY experiments/analyze_traversal_outcomes.py
Writes outputs/analysis_traversal_outcomes/summary.json and prints the breakdown.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from scene2motion import traversal_eval as te
from scene2motion.stepover_eval import step_scene

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "outputs/exp022_exact_tracking_bridge"
ACHIEVED_ROWS = BRIDGE / "achieved_rows.jsonl"
OUT = ROOT / "outputs/analysis_traversal_outcomes/summary.json"

OBSTACLE_X_M = 1.2
OBSTACLE_DEPTH_M = 0.20
#: `step_scene` makes the box corridor-spanning; the bridge's lateral corridor is its half-width.
CORRIDOR_HALF_WIDTH_M = 1.4
ROUTE_LENGTH_M = 7.2
GOAL_TOLERANCE_M = 0.5
SAMPLE_DT_S = 0.02
HEIGHTS_M = (0.03, 0.05, 0.08, 0.12, 0.20, 0.30)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_achieved(bridge: Path = BRIDGE) -> dict[str, dict[str, Any]]:
    """Every archived achieved trajectory, keyed by motion key, from all launch chunks."""
    clips: dict[str, dict[str, Any]] = {}
    for archive in sorted(bridge.glob("launches/*/attempt-*/eval/achieved_qpos.npz")):
        with np.load(archive, allow_pickle=False) as data:
            qpos, lengths = data["qpos"], data["valid_lengths"]
            terminated, keys = data["terminated"], data["motion_keys"]
            for i, key in enumerate(keys):
                name = str(key)
                clips[name] = {
                    "qpos": np.asarray(qpos[i][: int(lengths[i])], dtype=float),
                    "terminated": bool(terminated[i]),
                    "archive": str(archive.relative_to(bridge)),
                }
    return clips


def scene_for(height_m: float) -> Any:
    """The traversal problem: start at the route origin, goal at its end, box between them."""
    scene = step_scene(OBSTACLE_X_M, height_m, OBSTACLE_DEPTH_M)
    return replace(scene, start=(0.0, 0.0), goal=(ROUTE_LENGTH_M, 0.0),
                   meta={"corridor_half": CORRIDOR_HALF_WIDTH_M})


def evaluate_pool(clips: dict[str, dict[str, Any]], height_m: float) -> list[dict[str, Any]]:
    scene = scene_for(height_m)
    criteria = te.TraversalCriteria(goal_tolerance_m=GOAL_TOLERANCE_M,
                                    corridor_half_width_m=CORRIDOR_HALF_WIDTH_M)
    records = []
    for name in sorted(clips):
        clip = clips[name]
        record = te.evaluate_traversal(clip["qpos"], scene, terminated=clip["terminated"],
                                       sample_dt_s=SAMPLE_DT_S, criteria=criteria)
        record["motion_key"] = name
        records.append(record)
    return records


def main() -> None:
    clips = load_achieved(BRIDGE)
    if not clips:
        raise SystemExit(f"no achieved archives under {BRIDGE}")

    by_height: dict[str, Any] = {}
    for height in HEIGHTS_M:
        records = evaluate_pool(clips, height)
        summary = te.summarise(records)
        summary["obstacle_height_m"] = height
        summary["per_clip"] = [
            {k: r[k] for k in ("motion_key", "outcome", "passed_obstacle", "reached_goal",
                               "collided_obstacle", "obstacle_min_clearance_m",
                               "progress_fraction", "tracker_terminated")}
            for r in records
        ]
        by_height[f"{height:g}"] = summary

    result = {
        "analysis": "scene-level traversal outcomes for the tracked stepping pool",
        "descriptive_only": True,
        "scope": ("obstacle ABSENT from the physics scene: 'collided_obstacle' means the "
                  "achieved trajectory intersects the obstacle geometry in replay, not that "
                  "the robot felt contact; one route, one scene, physics seed 0"),
        "scene": {"obstacle_x_m": OBSTACLE_X_M, "obstacle_depth_m": OBSTACLE_DEPTH_M,
                  "corridor_half_width_m": CORRIDOR_HALF_WIDTH_M,
                  "start_m": [0.0, 0.0], "goal_m": [ROUTE_LENGTH_M, 0.0],
                  "goal_tolerance_m": GOAL_TOLERANCE_M},
        "n_clips": len(clips),
        "sources": {"achieved_rows": {"path": str(ACHIEVED_ROWS.relative_to(ROOT)),
                                      "sha256": _sha256(ACHIEVED_ROWS)},
                    "archives": sorted({c["archive"] for c in clips.values()})},
        "by_height": by_height,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    names = [n for n in te.OUTCOMES if n != "rejected"]
    print(f"{len(clips)} tracked rollouts, start (0, 0) -> goal ({ROUTE_LENGTH_M}, 0), "
          f"box at x = {OBSTACLE_X_M} m\n")
    print(f"{'height':>7}  " + "  ".join(f"{n:>17}" for n in names) + "   completed_rate")
    for key, summary in by_height.items():
        counts = summary["outcomes"]
        print(f"{key:>7}  " + "  ".join(f"{counts[n]:>17}" for n in names)
              + f"   {summary['completion_rate']:.3f}")
    print("\nwrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
