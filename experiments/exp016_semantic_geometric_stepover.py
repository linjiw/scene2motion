"""EXP-016: can text name a step while a coherent keyframe puts it at the obstacle?

EXP-015 showed that a step-over prompt can elicit more foot lift than a walk prompt, but it
measured the peak anywhere in a broad time window, used no obstacle, and reused a position
scaffold made from seed 0 for every sample.  It therefore established behaviour elicitation,
not spatially addressable traversal.

This experiment cleanly factors the two native conditioning roles on held-out matched seeds:

    prompt     walk | step-over
    scaffold   none | swing-leg positions | full-body positions | positions+rotations

The scaffold comes from a separate, fixed donor-seed bank generated with the step-over prompt.
Donor selection uses exact qpos foot envelopes, planar velocity, stable contralateral support,
floor penetration, and route progress; ARDY's generated contact feature is not treated as
ground truth.  The selected pose (or short block) is translated so the physical donor foot,
rather than merely the pelvis, is centred on a real floor obstacle.  Evaluation seeds never
participate in donor selection.

The primary readout is exact whole-body box clearance at the obstacle.  Contact preservation,
bilateral flight, progress, spatial phase error, foot penetration, and optional SONIC tracking
keep a jump, stop, or untrackable reference from being called a step-over.  The factorial
interaction answers the sharp question: does semantic text help ARDY preserve a geometrically
placed, coordinated traversal pose beyond what either text or the scaffold achieves alone?
Six default placements stratify two absolute-history levels by phase within ARDY's 52-frame
autoregressive horizon; results are reported by placement as well as pooled by seed cluster.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.constraints import build_conditions, channel_usage  # noqa: E402
from scene2motion.robot import BODY_MARGIN, G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.semantic_scaffold import (  # noqa: E402
    SCAFFOLD_MODES,
    StepEvent,
    build_transplanted_scaffold,
)
from scene2motion.sonic_state_export import (  # noqa: E402
    load_sonic_state_rollouts,
    sonic_state_sample_dt,
)
from scene2motion.sonic_export import write_motion_pkl  # noqa: E402
from scene2motion.stepover_eval import (  # noqa: E402
    BoxHeightProbe,
    StepOverThresholds,
    foot_kinematics_series,
    motion_metrics,
    select_kinematic_step_event,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp011_tracked_addressability import CKPT, run_sonic  # noqa: E402


WALK = "A person walks forward."
STEP = "A person steps over an obstacle."
PROMPTS = {"walk": WALK, "step": STEP}
DEFAULT_TARGET_FRAMES = (56, 78, 100, 108, 130, 152)
LOCAL_STRUCTURE_GATES = (
    "both_feet_cross_before_over_after",
    "lead_trail_order",
    "lead_overlap_has_trailing_support",
    "trail_overlap_has_lead_support",
    "bounded_unsupported_run",
    "lead_landing_dwell",
    "trail_landing_dwell",
    "lead_lands_before_or_during_trailing_overlap",
    "both_feet_finish_beyond",
    "bounded_floor_penetration",
)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state(root: Path) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).splitlines()
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=root,
            stderr=subprocess.DEVNULL)
        return {"commit": commit, "dirty": bool(status), "status": status,
                "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": [],
                "tracked_diff_sha256": None}


def _model_provenance(model_name: str) -> dict:
    """Released ARDY snapshot identity without relying on a live model attribute."""
    hub = (Path.home() / ".cache" / "huggingface" / "hub" /
           f"models--nvidia--{model_name}")
    ref = hub / "refs" / "main"
    revision = ref.read_text().strip() if ref.exists() else None
    snapshot = hub / "snapshots" / revision if revision else None
    checkpoint = snapshot / "denoiser.safetensors" if snapshot else None
    tokenizer = snapshot / "tokenizer.safetensors" if snapshot else None
    return {
        "name": model_name, "hf_repo": f"nvidia/{model_name}", "hf_revision": revision,
        "denoiser_blob": (checkpoint.resolve().name if checkpoint and checkpoint.exists()
                           else None),
        "tokenizer_blob": (tokenizer.resolve().name if tokenizer and tokenizer.exists()
                            else None),
    }


def _cluster_interval(values: list[float], seed: int = 16016,
                      n_boot: int = 5000) -> list[float]:
    """Percentile interval after collapsing repeated placements within each seed."""
    x = np.asarray(values, dtype=float)
    if not len(x):
        return [float("nan"), float("nan")]
    if len(x) == 1:
        return [float(x[0]), float(x[0])]
    rng = np.random.default_rng(seed)
    sampled = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(-1)
    return [float(v) for v in np.quantile(sampled, [0.025, 0.975])]


def _mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact paired discordance table; callers separately report each route position."""
    if len(a) != len(b):
        raise ValueError("McNemar inputs must be paired")
    n01 = sum((not x) and y for x, y in zip(a, b))
    n10 = sum(x and (not y) for x, y in zip(a, b))
    n = n01 + n10
    p = 1.0 if n == 0 else min(
        1.0, 2.0 * sum(math.comb(n, k) for k in range(min(n01, n10) + 1)) / (2 ** n))
    return {"baseline_fail_new_pass": n01, "baseline_pass_new_fail": n10,
            "discordant": n, "exact_two_sided_p": float(p)}


def _paired(rows_by_key: dict, left: tuple[str, str], right: tuple[str, str],
            metric: str) -> dict:
    """Right-minus-left effect, clustered by seed and stratified by target frame."""
    units = sorted({(k[2], k[3]) for k in rows_by_key
                    if (left[0], left[1], k[2], k[3]) in rows_by_key
                    and (right[0], right[1], k[2], k[3]) in rows_by_key})
    diffs = {
        (frame, seed): float(rows_by_key[(right[0], right[1], frame, seed)][metric]) -
        float(rows_by_key[(left[0], left[1], frame, seed)][metric])
        for frame, seed in units
    }
    seeds = sorted({seed for _, seed in units})
    seed_means = [float(np.mean([d for (f, s), d in diffs.items() if s == seed]))
                  for seed in seeds]
    result = {
        "left": f"{left[0]}__{left[1]}", "right": f"{right[0]}__{right[1]}",
        "metric": metric, "direction": "right_minus_left",
        "n_seed_position_pairs": len(units), "n_seed_clusters": len(seeds),
        "mean_difference": float(np.mean(seed_means)) if seed_means else float("nan"),
        "seed_cluster_bootstrap_95": _cluster_interval(seed_means),
        "by_target_frame": {},
    }
    is_binary = bool(units) and all(
        isinstance(rows_by_key[(p, m, f, s)][metric], (bool, np.bool_))
        for p, m in (left, right) for f, s in units)
    for frame in sorted({f for f, _ in units}):
        fs = [s for f, s in units if f == frame]
        lv = [rows_by_key[(left[0], left[1], frame, s)][metric] for s in fs]
        rv = [rows_by_key[(right[0], right[1], frame, s)][metric] for s in fs]
        entry = {"n": len(fs), "mean_difference": float(np.mean(
            [float(y) - float(x) for x, y in zip(lv, rv)]))}
        if is_binary:
            entry["mcnemar"] = _mcnemar([bool(x) for x in lv], [bool(x) for x in rv])
        result["by_target_frame"][str(frame)] = entry
    return result


def aggregate(rows: list[dict]) -> tuple[dict, dict]:
    """Arm summaries and seed-clustered, placement-stratified factorial contrasts."""
    metrics = (
        "obstacle_collision_free", "kinematic_traversal_success",
        "local_step_structure", "kinematic_step_success",
        "max_box_height_lower_bound_m",
        "obstacle_min_clearance_m",
        "progress_ratio", "mean_support_feet", "bilateral_flight_fraction",
        "swing_foot_at_crossing_m", "phase_error_m", "path_error_m",
        "max_foot_floor_penetration_m",
    )
    arms: dict[str, dict] = {}
    for arm in sorted({r["arm"] for r in rows}):
        group = [r for r in rows if r["arm"] == arm]
        arms[arm] = {"n_seed_position_rows": len(group),
                     "n_seeds": len({r["seed"] for r in group}),
                     "n_positions": len({r["target_frame"] for r in group})}
        for metric in metrics:
            arms[arm][metric] = float(np.mean([float(r[metric]) for r in group]))
        # Positions share a generated donor and seeds, so uncertainty is clustered by seed.
        for metric in ("max_box_height_lower_bound_m", "obstacle_min_clearance_m",
                       "kinematic_step_success"):
            seeds = sorted({r["seed"] for r in group})
            collapsed = [float(np.mean([float(r[metric]) for r in group
                                        if r["seed"] == seed])) for seed in seeds]
            arms[arm][f"{metric}__seed_cluster_bootstrap_95"] = _cluster_interval(collapsed)
        by_frame = {}
        for frame in sorted({r["target_frame"] for r in group}):
            frame_rows = [r for r in group if r["target_frame"] == frame]
            by_frame[str(frame)] = {
                "n": len(frame_rows),
                "kinematic_step_success": float(np.mean(
                    [r["kinematic_step_success"] for r in frame_rows])),
                "obstacle_collision_free": float(np.mean(
                    [r["obstacle_collision_free"] for r in frame_rows])),
                "max_box_height_lower_bound_m": float(np.mean(
                    [r["max_box_height_lower_bound_m"] for r in frame_rows])),
                "obstacle_min_clearance_m": float(np.mean(
                    [r["obstacle_min_clearance_m"] for r in frame_rows])),
            }
        arms[arm]["by_target_frame"] = by_frame
        arms[arm]["worst_position"] = {
            metric: float(min(v[metric] for v in by_frame.values()))
            for metric in ("kinematic_step_success", "obstacle_collision_free",
                           "max_box_height_lower_bound_m", "obstacle_min_clearance_m")}

    keyed = {(r["prompt"], r["scaffold"], r["target_frame"], r["seed"]): r for r in rows}
    contrasts: dict[str, dict] = {}
    endpoints = ("max_box_height_lower_bound_m", "obstacle_min_clearance_m",
                 "kinematic_step_success")
    for mode in sorted({r["scaffold"] for r in rows}):
        contrasts[f"text_effect__{mode}"] = {
            metric: _paired(keyed, ("walk", mode), ("step", mode), metric)
            for metric in endpoints}
    for prompt in PROMPTS:
        for mode in sorted({r["scaffold"] for r in rows if r["scaffold"] != "none"}):
            contrasts[f"scaffold_effect__{prompt}__{mode}"] = {
                metric: _paired(keyed, (prompt, "none"), (prompt, mode), metric)
                for metric in endpoints
            }
        contrasts[f"coherence_effect__{prompt}__fullbody_minus_leg"] = {
            metric: _paired(keyed, (prompt, "leg_pos"),
                            (prompt, "fullbody_pos"), metric) for metric in endpoints}
        contrasts[f"rotation_effect__{prompt}__posrot_minus_pos"] = {
            metric: _paired(keyed, (prompt, "fullbody_pos"),
                            (prompt, "fullbody_posrot"), metric) for metric in endpoints}
    # Difference-in-differences: extra text benefit once a scaffold is present, relative to
    # the text effect under root-only conditioning.
    for mode in sorted({r["scaffold"] for r in rows if r["scaffold"] != "none"}):
        units = sorted({(r["target_frame"], r["seed"]) for r in rows
                        if all((p, m, r["target_frame"], r["seed"]) in keyed
                               for p in PROMPTS for m in ("none", mode))})
        metric_results = {}
        for metric in endpoints:
            differences = {(frame, seed):
                (float(keyed[("step", mode, frame, seed)][metric]) -
                 float(keyed[("walk", mode, frame, seed)][metric])) -
                (float(keyed[("step", "none", frame, seed)][metric]) -
                 float(keyed[("walk", "none", frame, seed)][metric]))
                for frame, seed in units}
            seeds = sorted({s for _, s in units})
            seed_means = [float(np.mean([v for (f, s), v in differences.items()
                                         if s == seed])) for seed in seeds]
            metric_results[metric] = {
                "n_seed_position_blocks": len(units), "n_seed_clusters": len(seeds),
                "difference_in_differences": (float(np.mean(seed_means))
                                               if seed_means else float("nan")),
                "seed_cluster_bootstrap_95": _cluster_interval(seed_means),
            }
        contrasts[f"interaction__{mode}"] = metric_results
    return arms, contrasts


def _parse_sonic(log: str) -> dict:
    out = {}
    for line in log.splitlines():
        for prefix, name in (("Success Rate:", "success_rate"),
                             ("Progress Rate:", "progress_rate")):
            if line.startswith(prefix):
                out[name] = float(line.split(":", 1)[1])
        if line.startswith("All:"):
            for token in line.split("\t"):
                bits = token.strip().split(":", 1)
                if len(bits) == 2 and bits[0] in ("mpjpe_l", "accel_dist"):
                    out[bits[0]] = float(bits[1])
    return out


def _resample_plan(root_xz: np.ndarray, n_frames: int) -> np.ndarray:
    """Sample the same geometric route at an achieved roll-out's control rate."""
    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    root_xz = np.asarray(root_xz, dtype=float)
    if n_frames == len(root_xz):
        return root_xz
    source = np.linspace(0.0, 1.0, len(root_xz))
    target = np.linspace(0.0, 1.0, n_frames)
    return np.stack([np.interp(target, source, root_xz[:, axis])
                     for axis in range(2)], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/exp016_semantic_geometric_stepover")
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--n_donors", type=int, default=12)
    ap.add_argument("--seed_start", type=int, default=2400)
    ap.add_argument("--donor_seed_start", type=int, default=2200)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--speed", type=float, default=0.90)
    ap.add_argument("--target_frames", type=int, nargs="+", default=None,
                    help="scaffold frames; default stratifies two AR windows at phases 4/26/48")
    ap.add_argument("--obstacle_x", type=float, default=None,
                    help="single-position exploratory override; mutually exclusive with target_frames")
    ap.add_argument("--obstacle_height", type=float, default=0.08)
    ap.add_argument("--obstacle_depth", type=float, default=0.20)
    ap.add_argument("--diffusion_steps", type=int, default=5)
    ap.add_argument("--block_half_s", type=float, default=0.0,
                    help="0 tests one full-body keyframe; >0 transplants a short block")
    ap.add_argument("--block_stride_s", type=float, default=0.08)
    ap.add_argument("--min_donor_progress", type=float, default=0.75)
    ap.add_argument("--donor_support_window_s", type=float, default=0.24)
    ap.add_argument("--min_donor_stance_support", type=float, default=0.90)
    ap.add_argument("--min_donor_relative_lift_m", type=float, default=0.04)
    ap.add_argument("--support_height_m", type=float, default=0.02)
    ap.add_argument("--support_speed_mps", type=float, default=0.20)
    ap.add_argument("--min_contralateral_support", type=float, default=0.90)
    ap.add_argument("--max_unsupported_run_frames", type=int, default=2)
    ap.add_argument("--landing_dwell_frames", type=int, default=3)
    ap.add_argument("--landing_horizon_s", type=float, default=0.75)
    ap.add_argument("--max_floor_penetration_m", type=float, default=0.02)
    ap.add_argument("--lateral_corridor_half_width_m", type=float, default=0.30)
    ap.add_argument("--corridor_longitudinal_pad_m", type=float, default=0.30)
    ap.add_argument("--confirmatory", action="store_true",
                    help="requires >=24 seeds, the locked six placements, and a threshold receipt")
    ap.add_argument("--threshold_calibration_receipt", type=Path, default=None)
    ap.add_argument("--skip_sonic", action="store_true")
    ap.add_argument("--sonic_arms", nargs="+", default=[
        "walk__none", "step__none", "walk__fullbody_posrot",
        "step__fullbody_posrot"],
                    help="predeclared arms sent to SONIC after the kinematic screen")
    ap.add_argument("--sonic_num_envs", type=int, default=24)
    ap.add_argument("--timeout_s", type=int, default=2400)
    args = ap.parse_args()
    if args.n_seeds < 1 or args.n_donors < 1:
        raise SystemExit("n_seeds and n_donors must be positive")
    if args.obstacle_x is not None and args.target_frames is not None:
        raise SystemExit("pass obstacle_x or target_frames, not both")
    if args.obstacle_height <= 0 or args.obstacle_depth <= 0:
        raise SystemExit("obstacle dimensions must be positive")
    if args.block_half_s < 0 or args.block_stride_s <= 0:
        raise SystemExit("block_half_s must be non-negative and block_stride_s positive")
    if args.donor_support_window_s < 0 or not 0 <= args.min_donor_stance_support <= 1:
        raise SystemExit("invalid donor support-window/fraction threshold")
    if args.min_donor_relative_lift_m < 0:
        raise SystemExit("min_donor_relative_lift_m must be non-negative")
    cli_thresholds = StepOverThresholds(
        support_height_m=args.support_height_m,
        support_speed_mps=args.support_speed_mps,
        min_contralateral_support_fraction=args.min_contralateral_support,
        max_unsupported_run_frames=args.max_unsupported_run_frames,
        landing_dwell_frames=args.landing_dwell_frames,
        landing_horizon_s=args.landing_horizon_s,
        max_floor_penetration_m=args.max_floor_penetration_m,
        lateral_corridor_half_width_m=args.lateral_corridor_half_width_m,
        corridor_longitudinal_pad_m=args.corridor_longitudinal_pad_m,
    )
    cli_thresholds.validate()
    threshold_source = "cli_pilot_defaults"
    thresholds = cli_thresholds
    if args.threshold_calibration_receipt is not None:
        try:
            calibration = json.loads(args.threshold_calibration_receipt.read_text())
            calibrated_values = calibration["stepover_thresholds"]
            thresholds = StepOverThresholds(**calibrated_values)
            thresholds.validate()
            threshold_source = "calibration_receipt"
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid threshold_calibration_receipt: {exc}") from exc
    valid_arms = {f"{prompt}__{mode}" for prompt in PROMPTS for mode in SCAFFOLD_MODES}
    if unknown := sorted(set(args.sonic_arms) - valid_arms):
        raise SystemExit(f"unknown sonic_arms: {unknown}")
    if args.sonic_num_envs < 1:
        raise SystemExit("sonic_num_envs must be positive")
    if args.confirmatory:
        if args.n_seeds < 24:
            raise SystemExit("confirmatory runs require at least 24 evaluation seeds")
        if args.obstacle_x is not None or tuple(args.target_frames or DEFAULT_TARGET_FRAMES) != DEFAULT_TARGET_FRAMES:
            raise SystemExit("confirmatory runs require the locked six target frames")
        if args.threshold_calibration_receipt is None or not args.threshold_calibration_receipt.exists():
            raise SystemExit("confirmatory runs require a threshold_calibration_receipt")
    if not args.skip_sonic and not CKPT.exists():
        raise SystemExit(f"SONIC checkpoint not found at {CKPT}; pass --skip_sonic for kinematics")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eval_seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    donor_seeds = list(range(args.donor_seed_start,
                             args.donor_seed_start + args.n_donors))
    if set(eval_seeds) & set(donor_seeds):
        raise SystemExit("donor and evaluation seed ranges must be disjoint")

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    model_name = runner.model_name
    noise_stream_version = runner.noise_stream_version
    fps = runner.fps
    T = int(args.duration * fps)
    root_xz = np.stack([np.zeros(T), np.linspace(0.0, args.speed * args.duration, T)], -1)
    heading = np.zeros(T)
    root_y = np.full(T, 0.78)
    root_idx = int(runner.skeleton.root_idx)
    horizon = int(runner.model.gen_horizon_len)
    free_body = G1Body(None)

    # ---- Discovery split: choose one coherent donor without seeing evaluation outputs. ----
    donor_base, _ = build_transplanted_scaffold(
        {}, runner.joint_names, root_idx, StepEvent(0, "left", 0, 0, 0), 0,
        root_xz, heading, root_y, "none", first_heading=0.0)
    donor_outs = runner.generate([STEP] * len(donor_seeds), [donor_base] * len(donor_seeds),
                                 T, args.diffusion_steps, seeds=donor_seeds)
    donor_rows, eligible = [], []
    for seed, sample in zip(donor_seeds, donor_outs):
        row = {"seed": seed}
        try:
            q = runner.to_qpos(sample)
            kinematic_event = select_kinematic_step_event(
                free_body, q, fps, (int(0.15 * T), int(0.85 * T)),
                thresholds=thresholds,
                support_window_s=args.donor_support_window_s,
                min_stance_support_fraction=args.min_donor_stance_support,
                min_relative_lift_m=args.min_donor_relative_lift_m)
            event = StepEvent(
                kinematic_event.frame, kinematic_event.side,
                kinematic_event.relative_lift_m, kinematic_event.swing_height_m,
                kinematic_event.stance_height_m)
            progress_ratio = float((q[-1, 0] - q[0, 0]) /
                                   max(root_xz[-1, 1] - root_xz[0, 1], 1e-9))
            row.update(asdict(kinematic_event))
            row["progress_ratio"] = progress_ratio
            if progress_ratio >= args.min_donor_progress:
                eligible.append((event.relative_lift_m, -seed, seed, sample, event,
                                 kinematic_event, q))
        except ValueError as exc:
            row["ineligible_reason"] = str(exc)
        donor_rows.append(row)
    if not eligible:
        (out / "receipt.json").write_text(json.dumps({
            "experiment": "exp016_semantic_geometric_stepover",
            "status": "no_eligible_donor",
            "model": _model_provenance(model_name),
            "noise_stream_version": noise_stream_version,
            "donor_elicitation_ardy_samples": len(donor_seeds),
            "thresholds": asdict(thresholds),
            "threshold_source": threshold_source,
            "donors": donor_rows,
        }, indent=2))
        raise SystemExit("no donor met unilateral-support and progress gates")
    _, _, donor_seed, donor, event, kinematic_event, donor_q = max(eligible)
    np.save(out / "donor_qpos.npy", donor_q.astype(np.float32))

    donor_foot = foot_kinematics_series(free_body, donor_q, fps)[event.side]
    donor_foot_forward_offset = float(
        donor_foot["forward_representative_m"][event.frame] - donor_q[event.frame, 0])
    if args.obstacle_x is not None:
        desired_root_forward = args.obstacle_x - donor_foot_forward_offset
        target_frames = [int(np.argmin(np.abs(root_xz[:, 1] - desired_root_forward)))]
    else:
        target_frames = list(args.target_frames or DEFAULT_TARGET_FRAMES)
    if len(target_frames) != len(set(target_frames)):
        raise SystemExit("target_frames must be unique")
    if any(frame < 1 or frame >= T - 1 for frame in target_frames):
        raise SystemExit(f"target_frames must lie strictly inside [0, {T - 1}]")
    placements = []
    for target_frame in target_frames:
        obstacle_x = (float(args.obstacle_x) if args.obstacle_x is not None else
                      float(root_xz[target_frame, 1] + donor_foot_forward_offset))
        expanded_lo = obstacle_x - args.obstacle_depth / 2 - BODY_MARGIN
        expanded_hi = obstacle_x + args.obstacle_depth / 2 + BODY_MARGIN
        route_lo, route_hi = float(root_xz[:, 1].min()), float(root_xz[:, 1].max())
        if not (route_lo < expanded_lo < expanded_hi < route_hi):
            raise SystemExit(
                f"expanded obstacle [{expanded_lo:.3f}, {expanded_hi:.3f}] at frame "
                f"{target_frame} lies outside planned route [{route_lo:.3f}, {route_hi:.3f}]")
        placements.append({
            "id": f"f{target_frame}", "target_frame": int(target_frame),
            "target_frame_mod_horizon": int(target_frame % horizon),
            "obstacle_x_m": obstacle_x,
            "desired_root_forward_m": float(obstacle_x - donor_foot_forward_offset),
            "expanded_interval_m": [expanded_lo, expanded_hi],
        })
    half_frames = int(round(args.block_half_s * fps))
    stride_frames = max(1, int(round(args.block_stride_s * fps)))

    specs, scaffold_info, usage = {}, {}, {}
    for placement in placements:
        for mode in SCAFFOLD_MODES:
            key = (placement["target_frame"], mode)
            spec, info = build_transplanted_scaffold(
                donor, runner.joint_names, root_idx, event, placement["target_frame"],
                root_xz, heading, root_y, mode,
                half_window_frames=half_frames, stride_frames=stride_frames,
                first_heading=0.0,
            )
            specs[key] = spec
            scaffold_info[f"{placement['id']}__{mode}"] = asdict(info)
            _, mask = build_conditions(runner.model, spec, runner.device)
            usage[f"{placement['id']}__{mode}"] = {
                k: v for k, v in channel_usage(runner.model, mask).items() if v}

    # The exact arrays are part of the audit trail; the JSON records their frame/joint scope.
    np.savez(
        out / "scaffold_targets.npz",
        **{f"{placement['id']}__{mode}_pos_targets":
           specs[(placement["target_frame"], mode)].pos_targets
           for placement in placements for mode in SCAFFOLD_MODES
           if specs[(placement["target_frame"], mode)].pos_targets is not None},
        **{f"{placement['id']}__{mode}_rot_targets":
           specs[(placement["target_frame"], mode)].rot_targets
           for placement in placements for mode in SCAFFOLD_MODES
           if specs[(placement["target_frame"], mode)].rot_targets is not None},
    )

    # ---- Held-out matched factorial. ---------------------------------------------------
    # Root-only output is independent of obstacle position, so generate it once per prompt
    # and score the identical clip at all six positions. Every scaffolded arm is regenerated
    # at every placement. This saves 10*n samples without changing a comparison.
    rows: list[dict] = []
    clips_by_arm: dict[str, dict[str, np.ndarray]] = {
        f"{prompt}__{mode}": {} for prompt in PROMPTS for mode in SCAFFOLD_MODES}
    root_only_qpos: dict[str, dict[int, np.ndarray]] = {}
    n_eval_ardy_samples = 0
    probe_metadata = None
    for placement in placements:
        probe = BoxHeightProbe(placement["obstacle_x_m"], args.obstacle_depth)
        probe_metadata = probe.metadata()
        for prompt_name, prompt in PROMPTS.items():
            for mode in SCAFFOLD_MODES:
                arm = f"{prompt_name}__{mode}"
                if mode == "none" and prompt_name in root_only_qpos:
                    qpos_by_seed = root_only_qpos[prompt_name]
                else:
                    spec = specs[(placement["target_frame"], mode)]
                    generated = runner.generate(
                        [prompt] * len(eval_seeds), [spec] * len(eval_seeds), T,
                        args.diffusion_steps, seeds=eval_seeds)
                    qpos_by_seed = {
                        seed: runner.to_qpos(sample)
                        for seed, sample in zip(eval_seeds, generated)}
                    n_eval_ardy_samples += len(generated)
                    if mode == "none":
                        root_only_qpos[prompt_name] = qpos_by_seed

                placement_rows = []
                for seed in eval_seeds:
                    q = qpos_by_seed[seed]
                    motion_key = (f"{arm}__s{seed}" if mode == "none" else
                                  f"{arm}__{placement['id']}__s{seed}")
                    clips_by_arm[arm].setdefault(motion_key, q)
                    # The probe is mutable; reset the exact fixed-height obstacle before the
                    # primary collision query for every motion. motion_metrics may then sweep
                    # its height to obtain the operational lower bound.
                    obstacle_body = probe.body(args.obstacle_height)
                    m = motion_metrics(
                        free_body, obstacle_body, probe, q, root_xz,
                        placement["obstacle_x_m"], event.side, fps=fps,
                        thresholds=thresholds)
                    gates = m["local_step"]["gates"]
                    m["kinematic_traversal_success"] = bool(
                        gates["whole_body_collision_free"] and gates["root_traversal"] and
                        gates["lateral_corridor"] and m["progress_ratio"] >= 0.8)
                    m["local_step_structure"] = bool(
                        all(gates[name] for name in LOCAL_STRUCTURE_GATES))
                    m["kinematic_step_success"] = bool(
                        m["local_step_success"] and m["progress_ratio"] >= 0.8)
                    m.update({
                        "arm": arm, "prompt": prompt_name, "scaffold": mode,
                        "seed": seed, "motion_key": motion_key,
                        "placement_id": placement["id"],
                        "target_frame": placement["target_frame"],
                        "target_frame_mod_horizon": placement["target_frame_mod_horizon"],
                        "obstacle_x_m": placement["obstacle_x_m"],
                        "sonic_evaluated": False,
                    })
                    rows.append(m)
                    placement_rows.append(m)
                print(
                    f"  {placement['id']:5s} {arm:25s} "
                    f"clear={np.mean([r['obstacle_collision_free'] for r in placement_rows]):.3f} "
                    f"step={np.mean([r['kinematic_step_success'] for r in placement_rows]):.3f} "
                    f"box-lb={100*np.mean([r['max_box_height_lower_bound_m'] for r in placement_rows]):.1f}cm "
                    f"({time.time()-t0:.0f}s)", flush=True)

    arm_summary, contrasts = aggregate(rows)
    pickle_paths: dict[str, Path] = {}
    for arm, clips in clips_by_arm.items():
        pkl = out / f"{arm}.pkl"
        write_motion_pkl(clips, pkl, fps=int(round(fps)), mj_model=free_body.model)
        pickle_paths[arm] = pkl

    # Release ARDY before Isaac/SONIC claims GPU memory.
    del runner, donor_outs, donor
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    sonic: dict[str, dict] = {}
    sonic_failures: list[dict] = []
    achieved: dict[str, dict] = {}
    if not args.skip_sonic:
        for arm in args.sonic_arms:
            pkl = pickle_paths[arm]
            sonic_dir = out / f"sonic_{arm}"
            started = time.time()
            try:
                rc, log = run_sonic(
                    pkl, sonic_dir, min(args.sonic_num_envs, len(clips_by_arm[arm])),
                    args.timeout_s)
            except subprocess.TimeoutExpired as exc:
                rc = -1
                log = ((exc.stdout or "") if isinstance(exc.stdout, str) else "") + "\n" + (
                    (exc.stderr or "") if isinstance(exc.stderr, str) else "")
                log += f"\nScene2Motion: SONIC timed out after {args.timeout_s}s\n"
            (out / f"sonic_{arm}.log").write_text(log)
            parsed = _parse_sonic(log)
            record = {"returncode": rc, **parsed}
            problems = []
            if rc != 0:
                problems.append(f"returncode={rc}")
            for required in ("success_rate", "progress_rate"):
                if required not in parsed:
                    problems.append(f"missing_{required}")
            try:
                archives = ([sonic_dir / "achieved_qpos.npz"]
                            if (sonic_dir / "achieved_qpos.npz").exists()
                            else sorted(sonic_dir.glob("achieved_qpos.rank*.npz")))
                if not archives:
                    raise FileNotFoundError("no achieved-state archive")
                if max(p.stat().st_mtime for p in archives) < started - 1.0:
                    raise RuntimeError("achieved-state archive was not refreshed by this run")
                rollouts = load_sonic_state_rollouts(sonic_dir)
                sample_dt = sonic_state_sample_dt(sonic_dir)
                by_key = {r.motion_key: r for r in rollouts}
                expected = set(clips_by_arm[arm])
                if set(by_key) != expected:
                    raise RuntimeError(
                        f"archive keys differ: missing={sorted(expected-set(by_key))[:5]}, "
                        f"extra={sorted(set(by_key)-expected)[:5]}")
                achieved[arm] = by_key
                record.update({
                    "achieved_state_archive": [str(p) for p in archives],
                    "n_achieved_rollouts": len(rollouts),
                    "sample_dt_s": sample_dt,
                })
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                problems.append(f"achieved_state: {exc}")
            if problems:
                sonic_failures.append({"arm": arm, "problems": problems})
            record["validation_problems"] = problems
            sonic[arm] = record
            print(f"  SONIC {arm:23s} rc={rc} "
                  f"success={record.get('success_rate', float('nan')):.3f} "
                  f"state={'yes' if arm in achieved else 'no'}", flush=True)

        # Replay the exact same geometry/contact gate on states the tracker actually achieved.
        # Isaac did not contain the obstacle in this experiment, so this measures retained
        # clearance and contact topology, not contact-rich obstacle interaction physics.
        for placement in placements:
            probe = BoxHeightProbe(placement["obstacle_x_m"], args.obstacle_depth)
            for row in [r for r in rows if r["target_frame"] == placement["target_frame"]
                        and r["arm"] in achieved]:
                rollout = achieved[row["arm"]][row["motion_key"]]
                row["sonic_evaluated"] = True
                row["sonic_terminated"] = bool(rollout.terminated)
                row["sonic_reported_progress"] = float(rollout.progress)
                row["sonic_valid_length"] = int(rollout.valid_length)
                if rollout.valid_length == 0:
                    row["sonic_achieved_replay_step_success"] = False
                    row["sonic_achieved_metrics"] = None
                    continue
                exec_fps = 1.0 / float(sonic[row["arm"]]["sample_dt_s"])
                obstacle_body = probe.body(args.obstacle_height)
                executed_metrics = motion_metrics(
                    free_body, obstacle_body, probe, rollout.qpos,
                    _resample_plan(root_xz, rollout.valid_length),
                    placement["obstacle_x_m"], event.side, fps=exec_fps,
                    thresholds=thresholds)
                executed_metrics["kinematic_step_success"] = bool(
                    executed_metrics["local_step_success"] and
                    executed_metrics["progress_ratio"] >= 0.8)
                row["sonic_achieved_replay_step_success"] = bool(
                    not rollout.terminated and
                    executed_metrics["kinematic_step_success"])
                row["sonic_achieved_metrics"] = executed_metrics

    execution_summary = {}
    for arm in args.sonic_arms if not args.skip_sonic else ():
        group = [r for r in rows if r["arm"] == arm and r["sonic_evaluated"]]
        execution_summary[arm] = {
            "n_seed_position_rows": len(group),
            "achieved_replay_step_success_rate": (
                float(np.mean([r["sonic_achieved_replay_step_success"] for r in group]))
                if group else None),
            "terminated_rate": (float(np.mean([r["sonic_terminated"] for r in group]))
                                if group else None),
        }

    stage = "confirmatory" if args.confirmatory else "pilot"
    if args.skip_sonic:
        status = f"{stage}_kinematics_complete_sonic_skipped"
    elif sonic_failures:
        status = f"{stage}_sonic_incomplete"
    else:
        status = f"{stage}_complete_achieved_state_replayed"
    repo = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__).resolve(),
        repo / "scene2motion" / "semantic_scaffold.py",
        repo / "scene2motion" / "stepover_eval.py",
        repo / "scene2motion" / "runner.py",
        repo / "scene2motion" / "sonic_state_export.py",
    ]
    receipt = {
        "experiment": "exp016_semantic_geometric_stepover",
        "status": status,
        "design_stage": stage,
        "claim_scope": (
            "single-donor semantic-geometric composition pilot" if not args.confirmatory else
            "locked single-donor semantic-geometric composition confirmation"),
        "model": _model_provenance(model_name),
        "noise_stream_version": noise_stream_version,
        "generation_identity": "noise_stream_v2_advancing_per_sample_autoregressive_rng",
        "fps": fps,
        "generation_horizon_frames": horizon,
        "diffusion_steps": args.diffusion_steps,
        "prompts": PROMPTS,
        "query_accounting": {
            "donor_elicitation_ardy_samples": len(donor_seeds),
            "held_out_evaluation_ardy_samples": n_eval_ardy_samples,
            "held_out_seed_position_rows": len(rows),
            "root_only_clips_reused_across_positions": True,
            "sonic_motion_rollouts_requested": int(sum(
                len(clips_by_arm[a]) for a in args.sonic_arms))
                if not args.skip_sonic else 0,
        },
        "donor_selection": {
            "criterion": (
                "maximum exact physical-foot relative lift among qpos-derived stable unilateral "
                "support events meeting progress and penetration gates; then lower seed"),
            "seed_split": donor_seeds,
            "elicitation_budget_ardy_samples": len(donor_seeds),
            "minimum_progress_ratio": args.min_donor_progress,
            "support_window_s": args.donor_support_window_s,
            "minimum_stance_support_fraction": args.min_donor_stance_support,
            "minimum_relative_lift_m": args.min_donor_relative_lift_m,
            "candidates": donor_rows,
            "selected_seed": donor_seed,
            "selected_event": asdict(kinematic_event),
            "swing_foot_forward_offset_m": donor_foot_forward_offset,
            "conditioning": "all evaluation rows are conditional on this one selected donor",
        },
        "evaluation_seeds": eval_seeds,
        "scene": {
            "obstacle_height_m": args.obstacle_height,
            "obstacle_depth_m": args.obstacle_depth,
            "body_margin_m_already_in_collision_geometry": BODY_MARGIN,
            "placements": placements,
            "placement_design": (
                "six frames: two autoregressive-history levels x just-after/middle/just-before "
                "the 52-frame seam" if len(placements) == 6 else
                "exploratory caller-specified placement design"),
            "box_height_probe": probe_metadata,
        },
        "scaffold": {
            "half_window_s": args.block_half_s,
            "stride_s": args.block_stride_s,
            "info": scaffold_info,
            "channel_usage": usage,
        },
        "success_gate": {
            "kinematic_traversal": (
                "whole-body collision-free + root before/after + lateral corridor + progress"),
            "local_step_structure_gates": list(LOCAL_STRUCTURE_GATES),
            "kinematic_step_success": (
                "every exact local step gate + progress_ratio >= 0.8"),
            "minimum_progress_ratio": 0.8,
            "thresholds": asdict(thresholds),
            "threshold_source": threshold_source,
            "threshold_calibration_receipt": (
                str(args.threshold_calibration_receipt)
                if args.threshold_calibration_receipt else None),
            "threshold_calibration_receipt_sha256": (
                _sha256(args.threshold_calibration_receipt)
                if args.threshold_calibration_receipt else None),
        },
        "arms": arm_summary,
        "paired_contrasts": contrasts,
        "sonic": sonic,
        "sonic_failures": sonic_failures,
        "sonic_execution_summary": execution_summary,
        "sonic_interpretation_guard": (
            "Achieved qpos are replayed against Scene2Motion geometry. The Isaac evaluation "
            "does not yet contain the obstacle, so this tests retained clearance/contact "
            "topology and termination, not physical obstacle interaction."),
        "provenance": {
            "code": _git_state(repo),
            "source_sha256": {
                str(path.relative_to(repo)): _sha256(path) for path in source_paths},
            "sonic_checkpoint": {
                "path": str(CKPT), "size_bytes": CKPT.stat().st_size if CKPT.exists() else None,
                "sha256": _sha256(CKPT) if CKPT.exists() else None,
            },
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    with open(out / "rows.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    with open(out / "receipt.json", "w") as fh:
        json.dump(receipt, fh, indent=2)
    print(f"wrote {len(rows)} held-out rows and {out / 'receipt.json'}", flush=True)
    if sonic_failures:
        raise SystemExit(f"SONIC validation incomplete for {len(sonic_failures)} arm(s)")


if __name__ == "__main__":
    main()
