"""EXP-020: does the coherent packet do anything the STEP prompt alone does not?

E1a established that the packet elicits a step-over-like lift in 8/8 clips against 1/8
for a WALK nominal, but places it 1-3 m from the request with a placement gain of about
zero, and that its rotation command is never satisfied at any lag.  That combination has
one obvious reading: the packet is acting through the same semantic route the prompt
already opens, adding elicitation but no spatial control.

This is the matched control for that reading.  For the same eight evaluated seeds, the
same strata and routes, and the same STEP prompt the packet arms used, it generates one
clip conditioned on the route **only** -- no packet, no rotation channel, no root-height
channel -- and scores it on the identical endpoint vector.

Four arms then sit on one table: WALK nominal (no elicitation), STEP text only, STEP +
absolute packet, STEP + residual packet.  If the text-only arm matches the packet arms on
elicitation rate, amplitude, and placement error, the structured channel has collapsed to
the semantics of the text channel and the packet buys nothing measurable.  If the packet
arms separate from it, the representation is doing work even though it cannot place.

Budget: exactly 8 frozen-prior samples.  The nominal and packet arms are read from the
completed exp019 ledger; nothing is regenerated and no gate is re-run.
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
    wilson_interval,
)
from scene2motion.constraints import ConstraintSpec  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.stepover_eval import BoxHeightProbe, foot_kinematics_series  # noqa: E402


SCHEMA_VERSION = "exp020-text-only-control-v1"
FAILURE_SCHEMA_VERSION = "exp020-text-only-control-failure-v1"
E1A_OUT = "outputs/exp019_gait_matched_stepover_v7"


class ControlAbort(RuntimeError):
    """Fail-closed stop after durable evidence has been written."""


def run_control(
    *,
    out: str | Path,
    e1a_out: str | Path = E1A_OUT,
    runner=None,
    body=None,
    code_state_fn=cal._git_state,
    cache_path: str | Path = "outputs/text_cache.npz",
) -> dict:
    output = Path(out)
    if output.exists() and any(output.iterdir()):
        raise ControlAbort(f"refusing nonempty output directory: {output}")
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
        "experiment": "exp020_text_only_control",
        "status": "running", "complete": False, "blocked": False, "stage": stage,
        "design": {
            "question": (
                "does the coherent packet add anything over the STEP prompt alone, on "
                "the same seeds, strata, routes and endpoint vector"
            ),
            "arms_compared": ["nominal (WALK, from e1a)", "text_only (STEP, generated here)",
                              "absolute (from e1a)", "residual (from e1a)"],
            "conditioning": "route root_xz only; no packet, no rotations, no root height",
            "prompt": e19.STEP,
            "budget": "exactly 8 frozen-prior samples",
            "e1a_source": str(e1a_out),
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
            raise ValueError("exp020 requires an exactly clean git worktree")
        receipt["provenance"]["code"] = code
        source = Path(e1a_out)
        e1a = json.loads((source / "receipt.json").read_text())
        if e1a.get("status") != "complete":
            raise ValueError("exp020 requires a completed E1a campaign")
        placement = json.loads((source / "placement_analysis.json").read_text())
        receipt["provenance"]["e1a"] = {
            "path": str(source),
            "receipt_sha256": cal._sha256(source / "receipt.json"),
            "placement_analysis_sha256": cal._sha256(
                source / "placement_analysis.json"),
        }
        per_seed = e1a["placement_selection"]["per_seed"]
        thresholds, dependency = cal._load_physical_threshold_dependency(
            Path("outputs/exp016_threshold_calibration/receipt.json"))
        receipt["provenance"]["physical_threshold_dependency"] = dependency

        runner = runner or ArdyRunner(cache_path=cache_path)
        if int(runner.noise_stream_version) != cal.NOISE_STREAM_VERSION:
            raise ValueError("exp020 requires ARDY noise_stream_version == 2")
        body = body or G1Body(None)
        routes = {label: cal.route_xz_for_speed(speed) for label, speed in cal.SPEEDS}
        seeds = sorted(int(seed) for seed in per_seed)
        specs = [
            ConstraintSpec(root_xz=routes[per_seed[str(seed)]["speed_label"]],
                           heading=None, root_y=None, first_heading=0.0)
            for seed in seeds
        ]

        stage = "generation"
        counters["generate_invocations"] += 1
        counters["samples_launched"] += len(seeds)
        persist()
        returned = runner.generate(
            [e19.STEP] * len(seeds), specs, cal.N_FRAMES, cal.DIFFUSION_STEPS,
            cfg_weight=cal.CFG_WEIGHT, seeds=seeds)
        counters["samples_returned"] += len(returned)
        if len(returned) != len(seeds):
            raise ValueError("runner returned the wrong number of control samples")

        stage = "scoring"
        for seed, sample in zip(seeds, returned):
            info = per_seed[str(seed)]
            label = info["speed_label"]
            route = routes[label]
            obstacle_x = float(info["obstacle_x_m"])
            qpos = np.asarray(runner.to_qpos(sample), dtype=float)
            key = f"s{seed}__text_only"
            qpos_archive[key] = np.asarray(qpos, dtype=np.float32)
            feet = foot_kinematics_series(body, qpos, cal.FPS)
            xs, heights = box_height_profile(qpos, route, e19.OBSTACLE_DEPTH_M)
            lift = lift_location(xs, heights)
            metrics = e17._score(
                body, BoxHeightProbe(obstacle_x, e19.OBSTACLE_DEPTH_M), qpos, route,
                obstacle_x, e19.EXPECTED_SWING_SIDE, cal.FPS, thresholds,
                e19.OBSTACLE_HEIGHT_M)
            row = {
                "seed": seed, "arm": "text_only", "prompt": e19.STEP,
                "speed_label": label, "obstacle_x_m": obstacle_x,
                "sample_sha256": e17._sample_hash(sample),
                "qpos_content_sha256": cal._array_hash({key: qpos_archive[key]}),
                "height_at_obstacle_m": float(metrics["max_box_height_lower_bound_m"]),
                "obstacle_collision_free": bool(metrics["obstacle_collision_free"]),
                "obstacle_min_clearance_m": float(metrics["obstacle_min_clearance_m"]),
                "progress_ratio": float(metrics["progress_ratio"]),
                **lift,
            }
            if lift["lift_x_m"] is not None:
                row["signed_placement_error_m"] = lift["lift_x_m"] - obstacle_x
                row["lift_side"] = lift_side(
                    qpos, feet, lift["lift_x_m"], e19.OBSTACLE_DEPTH_M)
            rows.append(row)
            persist()

        stage = "summary"
        e1a_rows = placement["rows"]

        def summarize(arm: str, source_rows: list[dict]) -> dict:
            selected = [r for r in source_rows if r["arm"] == arm]
            lifts = [r for r in selected if r.get("lift_x_m") is not None]
            errors = np.asarray(
                [abs(r["signed_placement_error_m"]) for r in lifts], dtype=float)
            signed = np.asarray(
                [r["signed_placement_error_m"] for r in lifts], dtype=float)
            heights = np.asarray(
                [r["lift_height_m"] for r in selected], dtype=float)
            low, high = wilson_interval(len(lifts), len(selected))
            return {
                "n": len(selected),
                "n_with_lift": len(lifts),
                "elicitation_rate": len(lifts) / len(selected) if selected else None,
                "elicitation_wilson95": [low, high],
                "mean_lift_height_m": float(heights.mean()) if heights.size else None,
                "max_lift_height_m": float(heights.max()) if heights.size else None,
                "mean_abs_placement_error_m": (
                    float(errors.mean()) if errors.size else None),
                "sd_signed_placement_error_m": (
                    float(signed.std(ddof=1)) if signed.size > 1 else None),
                "n_clearing_obstacle": sum(
                    1 for r in selected if r["height_at_obstacle_m"] > 0),
            }

        summary = {
            arm: summarize(arm, e1a_rows)
            for arm in ("nominal", "absolute", "residual")
        }
        summary["text_only"] = summarize("text_only", rows)
        receipt["summary"] = summary

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
        if isinstance(exc, ControlAbort):
            raise
        raise ControlAbort(str(exc)) from exc


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/exp020_text_only_control")
    parser.add_argument("--e1a-out", default=E1A_OUT)
    parser.add_argument("--cache-path", default="outputs/text_cache.npz")
    args = parser.parse_args(argv)
    receipt = run_control(out=args.out, e1a_out=args.e1a_out,
                          cache_path=args.cache_path)
    summary = receipt["summary"]
    print("=" * 84)
    print("Four arms, same 8 seeds, same routes, same endpoint vector")
    print("=" * 84)
    print(f"{'arm':>10} {'elicits':>9} {'Wilson95':>14} {'mean lift':>10} "
          f"{'max lift':>9} {'|place err|':>12} {'sd signed':>10} {'clears obst':>12}")
    for arm in ("nominal", "text_only", "absolute", "residual"):
        s = summary[arm]
        def show(value, fmt="{:.4f}"):
            return fmt.format(value) if value is not None else "-"
        print(f"{arm:>10} {s['n_with_lift']:>4}/{s['n']:<4} "
              f"[{s['elicitation_wilson95'][0]:.2f}, {s['elicitation_wilson95'][1]:.2f}]"
              f"{'':>2} {show(s['mean_lift_height_m']):>10} "
              f"{show(s['max_lift_height_m']):>9} "
              f"{show(s['mean_abs_placement_error_m'], '{:.2f}'):>12} "
              f"{show(s['sd_signed_placement_error_m'], '{:.2f}'):>10} "
              f"{s['n_clearing_obstacle']:>7}/{s['n']:<4}")


if __name__ == "__main__":
    main()
