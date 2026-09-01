"""EXP-021: the prior's spontaneous step-over distribution, and what it costs to select.

E1a plus its controls left exactly one placement mechanism standing.  The coherent packet
neither places (gain about zero) nor helps (it halves the prompt's own amplitude and
removes its 3/8 clearance of a 5 cm box); the route warp re-plans the gait; the model is
not translation-equivariant, so the world cannot be anchored by shifting the start.  What
remains is measure-then-select: generate under the prompt, measure where the lift actually
lands and how high it is, and choose.

Sizing that method needs one thing this repository does not yet have -- the joint
distribution of (lift position, lift height) for text-conditioned clips on a fixed route.
This measures it on K fresh seeds and derives, for a scene-specified obstacle, the
best-of-N curve a selection planner would face:

    P(at least one of N clips clears a box of height h within radius r of x*)

Reported for graded heights, because the operating envelope discovered in E1a is 5-7 cm
rather than the 8 cm the earlier scenes assumed, and a 5 cm cable is the common real
obstacle.  Kinematic stage only; no tracker claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import calibrate_ramp_route_phase as cal  # noqa: E402
from experiments import exp017_ramp_residual_stepover as e17  # noqa: E402
from experiments import exp019_ramp_gait_matched_stepover as e19  # noqa: E402
from experiments.analyze_e1a_placement import (  # noqa: E402
    box_height_profile,
    lift_location,
    lift_side,
)
from scene2motion.constraints import ConstraintSpec  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.stepover_eval import foot_kinematics_series  # noqa: E402


SCHEMA_VERSION = "exp021-elicited-lift-distribution-v1"
FAILURE_SCHEMA_VERSION = "exp021-elicited-lift-distribution-failure-v1"

POOL_SEEDS = tuple(range(4400, 4464))
BATCH_SIZE = 8
# 6.6 m of scannable route at 120 points is 5.5 cm resolution, well under the 0.28 m
# expanded footprint the probe integrates over; the 400-point default is wasted here and
# dominates wall clock (one collision sweep per point per clip).
SCAN_POINTS = 120
GRADED_HEIGHTS_M = (0.03, 0.05, 0.08, 0.12, 0.20, 0.30)
SELECTION_RADII_M = (0.10, 0.25, 0.50)
BEST_OF_N = (1, 2, 4, 8, 16, 32)


class DistributionAbort(RuntimeError):
    """Fail-closed stop after durable evidence has been written."""


def clears(profile_xs: np.ndarray, profile_heights: np.ndarray,
           target_x: float, radius_m: float, height_m: float) -> bool:
    """Does this clip clear a box of `height_m` somewhere within `radius_m` of x*?"""
    window = np.abs(profile_xs - target_x) <= radius_m
    return bool(np.any(profile_heights[window] >= height_m))


def best_of_n_curve(
    profiles: list[tuple[np.ndarray, np.ndarray]],
    targets: np.ndarray,
    *,
    radius_m: float,
    height_m: float,
) -> dict[str, float]:
    """Per-clip hit rate at a scene-specified obstacle, and the implied best-of-N.

    The per-clip rate is averaged over candidate obstacle positions, so it answers "if a
    scene puts the obstacle somewhere on this route, how often does one sampled clip clear
    it" -- which is the quantity a selection planner spends calls against.
    """
    per_clip = np.asarray([
        np.mean([clears(xs, hs, float(t), radius_m, height_m) for t in targets])
        for xs, hs in profiles
    ], dtype=float)
    rate = float(per_clip.mean())
    curve = {
        f"N={n}": (1.0 - (1.0 - rate) ** n) if rate < 1.0 else 1.0
        for n in BEST_OF_N
    }
    return {"per_clip_rate": rate, **curve}


def run_distribution(
    *,
    out: str | Path,
    runner=None,
    body=None,
    code_state_fn=cal._git_state,
    cache_path: str | Path = "outputs/text_cache.npz",
) -> dict:
    output = Path(out)
    if output.exists() and any(output.iterdir()):
        raise DistributionAbort(f"refusing nonempty output directory: {output}")
    repo = Path(__file__).resolve().parents[1]
    code = dict(code_state_fn(repo))
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stage = "preflight"
    rows: list[dict] = []
    qpos_archive: dict[str, np.ndarray] = {}
    counters = {"generate_invocations": 0, "samples_launched": 0,
                "samples_returned": 0}
    receipt: dict = {
        "schema": SCHEMA_VERSION,
        "experiment": "exp021_elicited_lift_distribution",
        "status": "running", "complete": False, "blocked": False, "stage": stage,
        "design": {
            "question": (
                "joint distribution of (lift position, lift height) under text "
                "conditioning on a fixed route, and the best-of-N curve it implies"
            ),
            "prompt": e19.STEP,
            "conditioning": "route root_xz only; the reference-speed 7.2 m straight route",
            "pool_seeds": list(POOL_SEEDS),
            "graded_heights_m": list(GRADED_HEIGHTS_M),
            "selection_radii_m": list(SELECTION_RADII_M),
            "budget": f"exactly {len(POOL_SEEDS)} frozen-prior samples",
            "stage_scope": "kinematic only; no tracker claim",
        },
        "query_accounting": counters,
        "provenance": {},
    }

    def persist() -> None:
        cal._write_jsonl(output / "rows.jsonl", rows)
        cal._persist_qpos(output / "qpos.npz", qpos_archive)
        receipt["stage"] = stage
        receipt["query_accounting"] = dict(counters)
        receipt["evidence_anchors"] = {
            "rows": {"path": "rows.jsonl", "n_rows": len(rows),
                     "logical_sha256": cal._json_hash(rows),
                     "file_sha256": cal._sha256(output / "rows.jsonl")},
            "qpos": {"path": "qpos.npz", "n_arrays": len(qpos_archive),
                     "content_sha256": (cal._array_hash(qpos_archive)
                                        if qpos_archive else None)},
        }
        receipt["wall_clock_s"] = float(time.monotonic() - started)
        cal._write_json(output / "receipt.json", receipt)

    try:
        if code.get("dirty") is not False:
            raise ValueError("exp021 requires an exactly clean git worktree")
        receipt["provenance"]["code"] = code
        thresholds, dependency = cal._load_physical_threshold_dependency(
            Path("outputs/exp016_threshold_calibration/receipt.json"))
        receipt["provenance"]["physical_threshold_dependency"] = dependency
        runner = runner or ArdyRunner(cache_path=cache_path)
        if int(runner.noise_stream_version) != cal.NOISE_STREAM_VERSION:
            raise ValueError("exp021 requires ARDY noise_stream_version == 2")
        body = body or G1Body(None)
        route = cal.route_xz_for_speed(cal.REFERENCE_SPEED_MPS)
        spec = ConstraintSpec(root_xz=route, heading=None, root_y=None,
                              first_heading=0.0)

        stage = "generation"
        profiles: list[tuple[np.ndarray, np.ndarray]] = []
        for start in range(0, len(POOL_SEEDS), BATCH_SIZE):
            seeds = list(POOL_SEEDS[start:start + BATCH_SIZE])
            counters["generate_invocations"] += 1
            counters["samples_launched"] += len(seeds)
            persist()
            returned = runner.generate(
                [e19.STEP] * len(seeds), [spec] * len(seeds), cal.N_FRAMES,
                cal.DIFFUSION_STEPS, cfg_weight=cal.CFG_WEIGHT, seeds=seeds)
            counters["samples_returned"] += len(returned)
            if len(returned) != len(seeds):
                raise ValueError("runner returned the wrong number of samples")
            for seed, sample in zip(seeds, returned):
                qpos = np.asarray(runner.to_qpos(sample), dtype=float)
                key = f"s{seed}"
                qpos_archive[key] = np.asarray(qpos, dtype=np.float32)
                xs, heights = box_height_profile(
                    qpos, route, e19.OBSTACLE_DEPTH_M, n_points=SCAN_POINTS)
                profiles.append((xs, heights))
                lift = lift_location(xs, heights)
                row = {
                    "seed": seed, "prompt": e19.STEP,
                    "sample_sha256": e17._sample_hash(sample),
                    "qpos_content_sha256": cal._array_hash({key: qpos_archive[key]}),
                    "progress_ratio": float(
                        e17._prescribed_progress_ratio(qpos, route)),
                    **lift,
                }
                if lift["lift_x_m"] is not None:
                    row["lift_side"] = lift_side(
                        qpos, foot_kinematics_series(body, qpos, cal.FPS),
                        lift["lift_x_m"], e19.OBSTACLE_DEPTH_M)
                row["clears_height"] = {
                    f"{h:g}": bool(lift["lift_height_m"] >= h)
                    for h in GRADED_HEIGHTS_M
                }
                rows.append(row)
            persist()

        stage = "summary"
        targets = np.linspace(1.5, 5.7, 43)
        selection = {
            f"h={h:g}": {
                f"r={r:g}": best_of_n_curve(
                    profiles, targets, radius_m=r, height_m=h)
                for r in SELECTION_RADII_M
            }
            for h in GRADED_HEIGHTS_M
        }
        heights = np.asarray([r["lift_height_m"] for r in rows], dtype=float)
        positions = np.asarray(
            [r["lift_x_m"] for r in rows if r["lift_x_m"] is not None], dtype=float)
        receipt["summary"] = {
            "n_clips": len(rows),
            "elicitation_rate": float(np.mean(heights > 0)),
            "lift_height_quantiles_m": {
                q: float(np.quantile(heights, float(q)))
                for q in ("0.25", "0.5", "0.75", "0.9", "1.0")
            },
            "n_clearing": {
                f"{h:g}": int(np.sum(heights >= h)) for h in GRADED_HEIGHTS_M
            },
            "lift_position_m": {
                "mean": float(positions.mean()) if positions.size else None,
                "sd": float(positions.std(ddof=1)) if positions.size > 1 else None,
                "quantiles": {
                    q: float(np.quantile(positions, float(q)))
                    for q in ("0.1", "0.25", "0.5", "0.75", "0.9")
                } if positions.size else None,
            },
            "target_positions_m": [float(t) for t in targets],
            "selection": selection,
        }

        stage = "complete"
        receipt.update({"status": "complete", "complete": True, "stage": stage,
                        "actual_ardy_samples": counters["samples_returned"]})
        persist()
        return receipt
    except Exception as exc:
        receipt.update({
            "schema": FAILURE_SCHEMA_VERSION, "status": "blocked", "blocked": True,
            "failed_stage": stage, "error_type": type(exc).__name__, "error": str(exc),
            "actual_ardy_samples": counters["samples_returned"],
        })
        persist()
        if isinstance(exc, DistributionAbort):
            raise
        raise DistributionAbort(str(exc)) from exc


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/exp021_elicited_lift_distribution")
    parser.add_argument("--cache-path", default="outputs/text_cache.npz")
    args = parser.parse_args(argv)
    receipt = run_distribution(out=args.out, cache_path=args.cache_path)
    summary = receipt["summary"]
    print("=" * 78)
    print(f"Spontaneous step-over lift under text conditioning, {summary['n_clips']} clips")
    print("=" * 78)
    print(f"  elicitation rate (any lift): {summary['elicitation_rate']:.2f}")
    print("  lift height quantiles (m): " + ", ".join(
        f"q{q}={v:.3f}" for q, v in summary["lift_height_quantiles_m"].items()))
    print("  clips clearing each box height: " + ", ".join(
        f"{h}m:{n}" for h, n in summary["n_clearing"].items()))
    position = summary["lift_position_m"]
    if position["quantiles"]:
        print(f"  lift position (m): mean {position['mean']:.2f}, sd {position['sd']:.2f}, "
              + ", ".join(f"q{q}={v:.2f}" for q, v in position["quantiles"].items()))
    print()
    print("=" * 78)
    print("Selection: P(>=1 of N clips clears height h within r of a scene obstacle)")
    print("=" * 78)
    for h in GRADED_HEIGHTS_M:
        block = summary["selection"][f"h={h:g}"]
        for r in SELECTION_RADII_M:
            curve = block[f"r={r:g}"]
            if curve["per_clip_rate"] <= 0:
                continue
            print(f"  h={h:<5g} r={r:<5g} per-clip {curve['per_clip_rate']:.3f}  " +
                  "  ".join(f"{k} {curve[k]:.2f}" for k in
                            (f"N={n}" for n in BEST_OF_N)))


if __name__ == "__main__":
    main()
