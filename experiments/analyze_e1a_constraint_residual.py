"""Positive control: did the decoder reproduce the packet at its own window?

exp017's Phase-1 finding was that the position channel honors literal requests (ask for
0.35 m, get 0.36 m).  E1a found the packet elicits a step-over but places it 1-3 m away
with a placement gain of about zero.  Two mechanisms explain that, with opposite remedies:

* **honored-and-compensated** -- the commanded rotations and root height are reproduced at
  the packet window, and the lift the probe finds elsewhere is a downstream dynamics
  effect.  Then the packet is entering correctly and placement is a control problem.
* **style-cue** -- the commanded values are not reproduced, and the packet is acting only
  as a soft hint the decoder re-plans around.  Then no amount of placement control on this
  channel will help, and the remedy is a different channel entirely.

The discriminator is **compliance**: how far the realized clip moved from the nominal
substrate toward the commanded value, at exactly the constrained (frame, joint) pairs.

    compliance = 1 - ||realized - commanded|| / ||nominal - commanded||

1.0 means the command was met exactly; 0.0 means the clip stayed where the nominal was;
negative means it moved away from the command.  Reported per arm for the rotation channel
(principal SO(3) geodesic angle) and the root-height channel (metres).

Everything is regenerated deterministically from the frozen identities; sample hashes are
checked against the archived campaign so this is a replay, not new evidence.
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
from experiments import exp018_ramp_route_warped_stepover as e18  # noqa: E402
from experiments import exp019_ramp_gait_matched_stepover as e19  # noqa: E402
from scene2motion.ramp import extract_packet_pair  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.stepover_eval import foot_kinematics_series  # noqa: E402


DEFAULT_OUT = "outputs/exp019_gait_matched_stepover_v7"


def _geodesic_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Principal SO(3) angle between stacks of rotation matrices."""
    relative = a @ np.swapaxes(b, -1, -2)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))


def _compliance(nominal: np.ndarray, commanded: np.ndarray,
                realized: np.ndarray) -> float | None:
    """1 - |realized-commanded| / |nominal-commanded|, guarding a zero request."""
    gap = float(np.linalg.norm(nominal - commanded))
    if gap < 1e-9:
        return None
    return 1.0 - float(np.linalg.norm(realized - commanded)) / gap


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--cache-path", default="outputs/text_cache.npz")
    args = parser.parse_args(argv)
    out = Path(args.out)
    receipt = json.loads((out / "receipt.json").read_text())
    if receipt.get("status") != "complete":
        raise SystemExit("this analysis reads a completed exp019 campaign")
    per_seed = receipt["placement_selection"]["per_seed"]

    thresholds, _ = cal._load_physical_threshold_dependency(
        Path("outputs/exp016_threshold_calibration/receipt.json"))
    stance = float(thresholds.min_contralateral_support_fraction)
    runner = ArdyRunner(cache_path=args.cache_path)
    body = G1Body(None)
    heading = np.zeros(cal.N_FRAMES, dtype=float)
    reference_route = cal.route_xz_for_speed(cal.REFERENCE_SPEED_MPS)

    # ---- donor bank and packet, replayed deterministically ---------------------
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
        row = {"seed": seed,
               "adapted_clip_sha256": e17._sample_hash(adapted),
               "neutral_clip_sha256": e17._sample_hash(neutral)}
        try:
            row.update({
                "_adapted_qpos": np.array(runner.to_qpos(adapted)),
                "_neutral_qpos": np.array(runner.to_qpos(neutral)),
                "_adapted_cycles": e17._phase_cycles(
                    body, np.array(runner.to_qpos(adapted)), cal.FPS, thresholds,
                    support_window_s=e19.SUPPORT_WINDOW_S,
                    min_stance_support_fraction=stance,
                    min_relative_lift_m=e19.DONOR_MIN_RELATIVE_LIFT_M),
                "_neutral_cycles": e17._phase_cycles(
                    body, np.array(runner.to_qpos(neutral)), cal.FPS, thresholds,
                    support_window_s=e19.SUPPORT_WINDOW_S,
                    min_stance_support_fraction=stance,
                    min_relative_lift_m=e19.DONOR_MIN_RELATIVE_LIFT_M),
            })
        except ValueError:
            row.update({"_adapted_qpos": np.array(runner.to_qpos(adapted)),
                        "_neutral_qpos": np.array(runner.to_qpos(neutral)),
                        "_adapted_cycles": (), "_neutral_cycles": ()})
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
        "event_selector": e17.EVENT_SELECTOR, "code_revision": "constraint-residual",
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
            "neutral_clip_sha256": selected_row["neutral_clip_sha256"]},
    )

    # ---- pool clips for the evaluated seeds ------------------------------------
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
            key = (seed, batch["speed_label"])
            if key in wanted:
                pool_samples[key] = sample

    routes = {label: cal.route_xz_for_speed(speed) for label, speed in cal.SPEEDS}
    archived_arms = np.load(out / "arm_qpos.npz")
    rows: list[dict] = []
    plan: list[tuple[int, str]] = []
    arm_specs: dict[tuple[int, str], object] = {}

    for seed_s, info in sorted(per_seed.items()):
        seed = int(seed_s)
        label = info["speed_label"]
        route = routes[label]
        sample = pool_samples[(seed, label)]
        qpos = np.asarray(runner.to_qpos(sample), dtype=float)
        feet = foot_kinematics_series(body, qpos, cal.FPS)
        candidate = {
            "apex_frame": int(info["apex_frame"]),
            "obstacle_x_m": float(info["obstacle_x_m"]),
        }
        probe = e19.probe_constructibility(
            candidate, pair=pair, sample=sample, qpos=qpos, foot_kinematics=feet,
            route_xz=route, route_heading=heading, thresholds=thresholds,
            min_stance_support_fraction=stance, runner=runner, body=body)
        if not probe.get("constructible"):
            raise SystemExit(
                f"replay could not rebuild seed {seed}: {probe['construct_rejection']}")
        for arm in e19.PACKET_ARMS:
            arm_specs[(seed, arm)] = probe["_specs"][arm]
            plan.append((seed, arm))

    returned = runner.generate(
        [e19.STEP] * len(plan), [arm_specs[key] for key in plan],
        cal.N_FRAMES, cal.DIFFUSION_STEPS, cfg_weight=cal.CFG_WEIGHT,
        seeds=[seed for seed, _ in plan])

    for (seed, arm), realized in zip(plan, returned):
        label = per_seed[str(seed)]["speed_label"]
        spec = arm_specs[(seed, arm)]
        nominal_sample = pool_samples[(seed, label)]
        realized_qpos = np.asarray(runner.to_qpos(realized), dtype=np.float32)
        archived_key = f"s{seed}__{arm}"
        matches = bool(np.array_equal(
            realized_qpos, np.asarray(archived_arms[archived_key], dtype=np.float32)))

        frames = np.asarray(spec.rot_frames, dtype=int)
        joints = np.asarray(spec.rot_joints, dtype=int)
        commanded_rot = np.asarray(spec.rot_targets, dtype=float)
        nominal_rot = np.asarray(
            nominal_sample["global_rot_mats"], dtype=float)[frames[:, None],
                                                            joints[None, :]]
        realized_rot = np.asarray(
            realized["global_rot_mats"], dtype=float)[frames[:, None],
                                                      joints[None, :]]
        commanded_gap = _geodesic_angles(commanded_rot, nominal_rot)
        realized_gap = _geodesic_angles(realized_rot, commanded_rot)
        moved = _geodesic_angles(realized_rot, nominal_rot)

        root_y = np.asarray(spec.root_y, dtype=float)
        nominal_root = np.asarray(nominal_sample["smooth_root_pos"], dtype=float)[:, 1]
        realized_root = np.asarray(realized["smooth_root_pos"], dtype=float)[:, 1]
        window = np.zeros(len(root_y), dtype=bool)
        window[frames] = True

        rows.append({
            "seed": seed, "arm": arm, "qpos_matches_archive": matches,
            "n_constrained_frames": int(len(frames)),
            "n_constrained_joints": int(len(joints)),
            "rotation_commanded_change_rad": float(np.mean(commanded_gap)),
            "rotation_realized_error_rad": float(np.mean(realized_gap)),
            "rotation_realized_move_rad": float(np.mean(moved)),
            "rotation_compliance": _compliance(
                nominal_rot.ravel(), commanded_rot.ravel(), realized_rot.ravel()),
            "root_height_commanded_change_m": float(
                np.mean(np.abs(root_y[window] - nominal_root[window]))),
            "root_height_realized_error_m": float(
                np.mean(np.abs(realized_root[window] - root_y[window]))),
            "root_height_compliance": _compliance(
                nominal_root[window], root_y[window], realized_root[window]),
            "root_height_compliance_full_clip": _compliance(
                nominal_root, root_y, realized_root),
        })

    print("=" * 78)
    print("Determinism replay")
    print("=" * 78)
    print(f"  arm clips byte-identical to the archive: "
          f"{sum(r['qpos_matches_archive'] for r in rows)}/{len(rows)}")

    print()
    print("=" * 78)
    print("Packet-window constraint residual (per arm, mean over 8 seeds)")
    print("=" * 78)
    header = (f"{'arm':>9} {'rot cmd Δ':>10} {'rot err':>9} {'rot compl':>10} "
              f"{'y cmd Δ':>9} {'y err':>8} {'y compl':>9}")
    print(header)
    for arm in e19.PACKET_ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        def mean(key):
            values = [r[key] for r in selected if r[key] is not None]
            return float(np.mean(values)) if values else float("nan")
        print(f"{arm:>9} {mean('rotation_commanded_change_rad'):>10.4f} "
              f"{mean('rotation_realized_error_rad'):>9.4f} "
              f"{mean('rotation_compliance'):>10.2f} "
              f"{mean('root_height_commanded_change_m'):>9.4f} "
              f"{mean('root_height_realized_error_m'):>8.4f} "
              f"{mean('root_height_compliance'):>9.2f}")
    print("\n  rot cmd Δ  = how far the packet asked the rotations to move from nominal")
    print("  rot err    = how far the realized clip still is from the command")
    print("  compliance = 1 met exactly, 0 stayed at nominal, <0 moved away")

    print()
    print("=" * 78)
    print("Per-seed compliance")
    print("=" * 78)
    print(f"{'seed':>6} {'arm':>9} {'rot compl':>10} {'y compl':>9} {'y compl (clip)':>15}")
    for row in rows:
        def show(value):
            return f"{value:>+.2f}" if value is not None else "     n/a"
        print(f"{row['seed']:>6} {row['arm']:>9} {show(row['rotation_compliance']):>10} "
              f"{show(row['root_height_compliance']):>9} "
              f"{show(row['root_height_compliance_full_clip']):>15}")

    destination = out / "constraint_residual.json"
    destination.write_text(json.dumps(
        {"source": str(out), "rows": rows}, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
