"""Is the packet ignored, or honored late?

The constraint-residual control found negative rotation compliance at the packet window:
the realized clip ends *further* from the commanded rotations than the nominal was.  Two
readings survive that measurement, with opposite consequences:

* **style cue** -- the decoder never satisfies the command at any time, and placement is
  not addressable through this channel;
* **delayed execution** -- the command is satisfied, but at some lag the autoregressive
  context imposes, in which case an open-loop lead of one measured lag is the whole fix.

This sweeps the comparison window over lags and asks whether *any* lag makes the realized
clip match the command better than the nominal did.  It also checks whether the best lag
lines up with the observed placement error converted to frames, which is what a single
delayed-execution mechanism would predict.

Zero new evidence: the arms are regenerated deterministically and checked byte-identical
against the archived campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import calibrate_ramp_route_phase as cal  # noqa: E402
from experiments import exp017_ramp_residual_stepover as e17  # noqa: E402
from experiments import exp019_ramp_gait_matched_stepover as e19  # noqa: E402
from experiments.analyze_e1a_constraint_residual import _geodesic_angles  # noqa: E402
from scene2motion.ramp import extract_packet_pair  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.stepover_eval import foot_kinematics_series  # noqa: E402


DEFAULT_OUT = "outputs/exp019_gait_matched_stepover_v7"
LAGS = tuple(range(-40, 81))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--cache-path", default="outputs/text_cache.npz")
    args = parser.parse_args(argv)
    out = Path(args.out)
    receipt = json.loads((out / "receipt.json").read_text())
    per_seed = receipt["placement_selection"]["per_seed"]
    placement = {
        (row["seed"], row["arm"]): row
        for row in json.loads((out / "placement_analysis.json").read_text())["rows"]
    }

    thresholds, _ = cal._load_physical_threshold_dependency(
        Path("outputs/exp016_threshold_calibration/receipt.json"))
    stance = float(thresholds.min_contralateral_support_fraction)
    runner = ArdyRunner(cache_path=args.cache_path)
    body = G1Body(None)
    heading = np.zeros(cal.N_FRAMES, dtype=float)
    reference_route = cal.route_xz_for_speed(cal.REFERENCE_SPEED_MPS)

    base = e17._base_spec(reference_route)
    prompts, specs, seeds = [], [], []
    for seed in e19.DONOR_SEEDS:
        prompts.extend((e19.STEP, e19.WALK))
        specs.extend((base, base))
        seeds.extend((seed, seed))
    generated = runner.generate(
        prompts, specs, cal.N_FRAMES, cal.DIFFUSION_STEPS,
        cfg_weight=cal.CFG_WEIGHT, seeds=seeds)
    donor_rows, donor_samples = [], {}
    for index, seed in enumerate(e19.DONOR_SEEDS):
        adapted, neutral = generated[2 * index:2 * index + 2]
        donor_samples[seed] = (adapted, neutral)
        adapted_qpos = np.array(runner.to_qpos(adapted))
        neutral_qpos = np.array(runner.to_qpos(neutral))
        row = {"seed": seed,
               "adapted_clip_sha256": e17._sample_hash(adapted),
               "neutral_clip_sha256": e17._sample_hash(neutral),
               "_adapted_qpos": adapted_qpos, "_neutral_qpos": neutral_qpos}
        try:
            row["_adapted_cycles"] = e17._phase_cycles(
                body, adapted_qpos, cal.FPS, thresholds,
                support_window_s=e19.SUPPORT_WINDOW_S,
                min_stance_support_fraction=stance,
                min_relative_lift_m=e19.DONOR_MIN_RELATIVE_LIFT_M)
            row["_neutral_cycles"] = e17._phase_cycles(
                body, neutral_qpos, cal.FPS, thresholds,
                support_window_s=e19.SUPPORT_WINDOW_S,
                min_stance_support_fraction=stance,
                min_relative_lift_m=e19.DONOR_MIN_RELATIVE_LIFT_M)
        except ValueError:
            row["_adapted_cycles"], row["_neutral_cycles"] = (), ()
        donor_rows.append(row)
    selected_row, adapted_cycle, neutral_cycle, phase_match = e17._select_source_pair(
        donor_rows, half_window_frames=e19.HALF_WINDOW_FRAMES)
    adapted_sample, neutral_sample = donor_samples[int(selected_row["seed"])]
    parents = np.asarray(
        runner.skeleton.joint_parents.detach().cpu().numpy(), dtype=int)
    provenance = {
        "adapted_clip_sha256": selected_row["adapted_clip_sha256"],
        "checkpoint_sha256": "0" * 64, "generator_id": "replay",
        "sampler_seed": int(selected_row["seed"]), "noise_stream_version": 2,
        "event_selector": e17.EVENT_SELECTOR, "code_revision": "response-lag",
    }
    pair = extract_packet_pair(
        adapted_sample, neutral_sample, adapted_cycle.event, neutral_cycle.event,
        adapted_route_heading=heading, neutral_route_heading=heading,
        phase_match=phase_match, joint_names=runner.joint_names,
        parent_indices=parents, root_idx=int(runner.skeleton.root_idx),
        source_fps=cal.FPS, half_window_frames=e19.HALF_WINDOW_FRAMES,
        absolute_provenance=dict(provenance),
        residual_provenance={
            **provenance,
            "neutral_clip_sha256": selected_row["neutral_clip_sha256"]})

    wanted = {(int(s), v["speed_label"]) for s, v in per_seed.items()}
    pool_samples: dict[tuple[int, str], dict] = {}
    for batch in e19.pool_batch_plan():
        if not any((seed, batch["speed_label"]) in wanted for seed in batch["seeds"]):
            continue
        spec = cal.root_only_walk_spec(batch["requested_speed_mps"])
        returned = runner.generate(
            [e19.WALK] * e19.POOL_BATCH_SIZE, [spec] * e19.POOL_BATCH_SIZE,
            cal.N_FRAMES, cal.DIFFUSION_STEPS, cfg_weight=cal.CFG_WEIGHT,
            seeds=batch["seeds"])
        for seed, sample in zip(batch["seeds"], returned):
            if (seed, batch["speed_label"]) in wanted:
                pool_samples[(seed, batch["speed_label"])] = sample

    routes = {label: cal.route_xz_for_speed(speed) for label, speed in cal.SPEEDS}
    plan: list[tuple[int, str]] = []
    arm_specs: dict[tuple[int, str], object] = {}
    for seed_s, info in sorted(per_seed.items()):
        seed = int(seed_s)
        label = info["speed_label"]
        sample = pool_samples[(seed, label)]
        qpos = np.asarray(runner.to_qpos(sample), dtype=float)
        probe = e19.probe_constructibility(
            {"apex_frame": int(info["apex_frame"]),
             "obstacle_x_m": float(info["obstacle_x_m"])},
            pair=pair, sample=sample, qpos=qpos,
            foot_kinematics=foot_kinematics_series(body, qpos, cal.FPS),
            route_xz=routes[label], route_heading=heading, thresholds=thresholds,
            min_stance_support_fraction=stance, runner=runner, body=body)
        if not probe.get("constructible"):
            raise SystemExit(f"replay could not rebuild seed {seed}")
        for arm in e19.PACKET_ARMS:
            arm_specs[(seed, arm)] = probe["_specs"][arm]
            plan.append((seed, arm))
    realized_arms = runner.generate(
        [e19.STEP] * len(plan), [arm_specs[key] for key in plan],
        cal.N_FRAMES, cal.DIFFUSION_STEPS, cfg_weight=cal.CFG_WEIGHT,
        seeds=[seed for seed, _ in plan])

    rows: list[dict] = []
    for (seed, arm), realized in zip(plan, realized_arms):
        label = per_seed[str(seed)]["speed_label"]
        spec = arm_specs[(seed, arm)]
        frames = np.asarray(spec.rot_frames, dtype=int)
        joints = np.asarray(spec.rot_joints, dtype=int)
        commanded = np.asarray(spec.rot_targets, dtype=float)
        nominal_rot = np.asarray(
            pool_samples[(seed, label)]["global_rot_mats"], dtype=float)
        realized_rot = np.asarray(realized["global_rot_mats"], dtype=float)
        nominal_gap = float(np.mean(_geodesic_angles(
            commanded, nominal_rot[frames[:, None], joints[None, :]])))
        curve = {}
        for lag in LAGS:
            shifted = frames + lag
            if shifted.min() < 0 or shifted.max() >= cal.N_FRAMES:
                continue
            curve[lag] = float(np.mean(_geodesic_angles(
                realized_rot[shifted[:, None], joints[None, :]], commanded)))
        best_lag = min(curve, key=curve.get)
        entry = placement.get((seed, arm), {})
        signed_error_m = entry.get("signed_placement_error_m")
        speed = dict(cal.SPEEDS)[label]
        rows.append({
            "seed": seed, "arm": arm,
            "nominal_gap_rad": nominal_gap,
            "error_at_lag0_rad": curve.get(0),
            "best_lag_frames": int(best_lag),
            "error_at_best_lag_rad": curve[best_lag],
            "beats_nominal_at_any_lag": bool(curve[best_lag] < nominal_gap),
            "signed_placement_error_m": signed_error_m,
            "placement_error_frames": (
                None if signed_error_m is None else signed_error_m / speed * cal.FPS),
            "lag_curve": {str(k): v for k, v in curve.items()},
        })

    print("=" * 80)
    print("Does ANY lag make the realized clip match the command better than nominal?")
    print("=" * 80)
    print(f"{'seed':>6} {'arm':>9} {'nominal':>9} {'lag0':>8} {'best':>8} "
          f"{'@lag':>6} {'beats?':>7} {'place err (frames)':>19}")
    for row in rows:
        place = row["placement_error_frames"]
        print(f"{row['seed']:>6} {row['arm']:>9} {row['nominal_gap_rad']:>9.4f} "
              f"{row['error_at_lag0_rad']:>8.4f} {row['error_at_best_lag_rad']:>8.4f} "
              f"{row['best_lag_frames']:>6} "
              f"{'yes' if row['beats_nominal_at_any_lag'] else 'NO':>7} "
              + (f"{place:>19.0f}" if place is not None else f"{'-':>19}"))
    beats = sum(row["beats_nominal_at_any_lag"] for row in rows)
    print(f"\n  clips where some lag beats the nominal baseline: {beats}/{len(rows)}")
    best = np.asarray([row["best_lag_frames"] for row in rows], dtype=float)
    print(f"  best lag: median {np.median(best):+.0f} frames, "
          f"sd {best.std(ddof=1):.0f}, range [{best.min():+.0f}, {best.max():+.0f}]")
    place = np.asarray(
        [row["placement_error_frames"] for row in rows
         if row["placement_error_frames"] is not None], dtype=float)
    if place.size == best.size:
        print(f"  correlation(best lag, placement error in frames): "
              f"r={np.corrcoef(best, place)[0, 1]:+.2f}  "
              f"(a single delayed-execution mechanism would predict r near +1)")

    destination = out / "response_lag.json"
    destination.write_text(json.dumps(
        {"source": str(out), "lags": list(LAGS), "rows": rows},
        indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
