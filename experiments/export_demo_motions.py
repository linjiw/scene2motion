"""Export a small set of scored clips as browser-playable skeleton animations.

Produces one compact JSON holding world-space joint positions per frame for the clips
that carry the paper's argument, together with the obstacle each was scored against and
the measurements already in the ledgers.  No new generation: every clip is read from a
committed campaign archive.

The clips are chosen to make four points in order:

* the frozen model can step over -- the donor clip clears 0.30 m;
* an ordinary walk clears nothing anywhere, which is the correct floor;
* under the prompt alone the step appears early, and a box placed afterwards at that
  point is cleared while the same motion misses a box at the position a scene specifies;
* asking through the joint-rotation channel keeps the step but displaces and attenuates it.

The two text-only panels are a capability illustration: the box in the first is placed
after watching the clip, so neither panel is a placement or traversal result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402

from experiments import calibrate_ramp_route_phase as cal  # noqa: E402
from experiments import exp019_ramp_gait_matched_stepover as e19  # noqa: E402
from experiments.analyze_e1a_placement import (  # noqa: E402
    box_height_profile,
    lift_location,
)
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.stepover_eval import BoxHeightProbe  # noqa: E402


# A readable subset of the G1 tree: spine, both legs to the foot, both arms to the wrist.
DRAWN_BODIES = (
    "pelvis", "waist_yaw_link", "torso_link",
    "left_hip_pitch_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_knee_link", "right_ankle_roll_link",
    "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_yaw_link",
)
BONES = (
    ("pelvis", "waist_yaw_link"), ("waist_yaw_link", "torso_link"),
    ("pelvis", "left_hip_pitch_link"), ("left_hip_pitch_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_pitch_link"), ("right_hip_pitch_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("torso_link", "left_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
)
FRAME_STRIDE = 2
DECIMALS = 3
# The peak of the clearance profile is what every caption quotes, so the scan has to be
# fine enough to actually land on it; a coarse grid understates the donor by 4 cm.
PROFILE_POINTS = 360


def skeleton_track(body: G1Body, qpos: np.ndarray) -> list[list[float]]:
    """World positions of the drawn bodies, one flat [x,y,z,...] row per frame."""
    model = body.model
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
           for name in DRAWN_BODIES]
    if any(i < 0 for i in ids):
        raise ValueError("a drawn body is missing from the released G1 model")
    frames: list[list[float]] = []
    for index in range(0, len(qpos), FRAME_STRIDE):
        body.fk(np.asarray(qpos[index], dtype=float))
        positions = np.asarray(body.data.xpos, dtype=float)[ids]
        frames.append([round(float(v), DECIMALS) for v in positions.reshape(-1)])
    return frames


def clip_entry(
    body: G1Body, qpos: np.ndarray, route_xz: np.ndarray, *,
    key: str, title: str, caption: str, obstacle_x: float | None,
    obstacle_height_m: float, extra: dict | None = None,
) -> dict:
    xs, heights = box_height_profile(
        qpos, route_xz, e19.OBSTACLE_DEPTH_M, n_points=PROFILE_POINTS)
    lift = lift_location(xs, heights)
    entry = {
        "key": key, "title": title, "caption": caption,
        "fps": cal.FPS / FRAME_STRIDE,
        "route_end_m": float(route_xz[-1, 1]),
        "obstacle_x_m": obstacle_x,
        "obstacle_height_m": obstacle_height_m,
        "obstacle_depth_m": e19.OBSTACLE_DEPTH_M,
        "lift_x_m": lift["lift_x_m"],
        "lift_height_m": round(float(lift["lift_height_m"]), 4),
        "profile_x_m": [round(float(v), 3) for v in xs],
        "profile_height_m": [round(float(v), 4) for v in heights],
        "frames": skeleton_track(body, qpos),
        **(extra or {}),
    }
    if obstacle_x is not None:
        entry["height_at_obstacle_m"] = round(
            float(BoxHeightProbe(obstacle_x, e19.OBSTACLE_DEPTH_M).probe(qpos)), 4)
        if lift["lift_x_m"] is not None:
            entry["placement_error_m"] = round(
                float(lift["lift_x_m"] - obstacle_x), 3)
    return entry


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/demo_motions.json")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    body = G1Body(None)
    reference_route = cal.route_xz_for_speed(cal.REFERENCE_SPEED_MPS)
    clips: list[dict] = []

    # 1. The donor: proof the frozen prior owns a real step-over.
    donor = np.load(repo / "outputs/exp019_gait_matched_stepover_v7/donor_qpos.npz")
    donor_qpos = np.asarray(donor["s2603__adapted"], dtype=float)
    donor_profile_x, donor_profile_h = box_height_profile(
        donor_qpos, reference_route, e19.OBSTACLE_DEPTH_M, n_points=PROFILE_POINTS)
    donor_peak = int(np.argmax(donor_profile_h))
    donor_peak_x = float(donor_profile_x[donor_peak])
    donor_peak_h = float(donor_profile_h[donor_peak])
    clips.append(clip_entry(
        body, donor_qpos, reference_route,
        key="donor",
        title="The model can step over a box",
        caption=(
            "Seed 2603 under the step prompt, scored against a box placed at its own "
            f"crossing: it clears {donor_peak_h:.2f} m. Capability, not placement — the "
            "box was put where the step already was."),
        obstacle_x=donor_peak_x, obstacle_height_m=0.20,
        extra={"badge": "capability", "tone": "good"}))

    # 2. A nominal walk: the correct zero baseline.
    pool = np.load(repo / "outputs/exp019_gait_matched_stepover_v7/pool_qpos.npz")
    receipt = json.loads(
        (repo / "outputs/exp019_gait_matched_stepover_v7/receipt.json").read_text())
    per_seed = receipt["placement_selection"]["per_seed"]
    walk_seed = "4353"
    walk_label = per_seed[walk_seed]["speed_label"]
    walk_route = cal.route_xz_for_speed(dict(cal.SPEEDS)[walk_label])
    clips.append(clip_entry(
        body, np.asarray(pool[f"pilot__{walk_label}__seed{walk_seed}"], dtype=float),
        walk_route,
        key="walk",
        title="An ordinary walk clears nothing",
        caption=(
            "The same route under the walk prompt. The swing foot sweeps through every "
            "position at low height, so the tallest box the whole body clears anywhere on "
            "the route is 5 mm. A zero here is the expected floor, not a broken scene."),
        obstacle_x=float(per_seed[walk_seed]["obstacle_x_m"]), obstacle_height_m=0.08,
        extra={"badge": "baseline", "tone": "neutral"}))

    # 3 and 4. One text-only clip, scored twice: staged, then commanded mid-route.
    distribution = repo / "outputs/exp021_elicited_lift_distribution_v2"
    rows = [json.loads(line) for line in (distribution / "rows.jsonl").open()]
    staged = max(
        (row for row in rows
         if row["lift_x_m"] is not None and 0.9 <= row["lift_x_m"] <= 1.6),
        key=lambda row: row["lift_height_m"])
    text_qpos = np.asarray(
        np.load(distribution / "qpos.npz")[f"s{staged['seed']}"], dtype=float)
    clips.append(clip_entry(
        body, text_qpos, reference_route,
        key="staged",
        title="A box placed where the model stepped",
        caption=(
            f"Seed {staged['seed']} under the prompt alone. The step comes early, as it "
            "usually does, and a box placed at that point is cleared. The box was placed "
            "after watching the clip, so this shows capability, not placement."),
        obstacle_x=float(staged["lift_x_m"]), obstacle_height_m=0.08,
        extra={"badge": "capability, not placement", "tone": "good"}))
    clips.append(clip_entry(
        body, text_qpos, reference_route,
        key="commanded",
        title="The same clip does not clear this box",
        caption=(
            "Identical clip, identical motion — only the box moved, to the mid-route "
            "position a scene would actually specify. The lift is metres away and nothing "
            "clears. Producing a stepping motion does not make it happen at the box."),
        obstacle_x=3.6, obstacle_height_m=0.08,
        extra={"badge": "the specified position", "tone": "bad"}))

    # 5. A packet arm: the behavior survives, displaced and attenuated.
    arms = np.load(repo / "outputs/exp019_gait_matched_stepover_v7/arm_qpos.npz")
    packet_seed = "4353"
    packet_label = per_seed[packet_seed]["speed_label"]
    clips.append(clip_entry(
        body, np.asarray(arms[f"s{packet_seed}__absolute"], dtype=float),
        cal.route_xz_for_speed(dict(cal.SPEEDS)[packet_label]),
        key="packet",
        title="Asking through the joint-rotation channel",
        caption=(
            f"Seed {packet_seed} with a step transported onto its own measured gait "
            "phase at zero frame shift. A step appears — 3.2 m from the box it was "
            "anchored to, at about half the amplitude the prompt alone produces."),
        obstacle_x=float(per_seed[packet_seed]["obstacle_x_m"]), obstacle_height_m=0.08,
        extra={"badge": "displaced and attenuated", "tone": "bad"}))

    payload = {
        "bodies": list(DRAWN_BODIES),
        "bones": [[DRAWN_BODIES.index(a), DRAWN_BODIES.index(b)] for a, b in BONES],
        "frame_stride": FRAME_STRIDE,
        "source_fps": cal.FPS,
        "clips": clips,
    }
    destination = Path(args.out)
    destination.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    size_kb = destination.stat().st_size / 1024
    print(f"wrote {destination} ({size_kb:.0f} KB, {len(clips)} clips)")
    for clip in clips:
        print(f"  {clip['key']:>10}: lift {clip['lift_height_m']:.3f} m at "
              f"x={clip['lift_x_m']}, obstacle {clip['obstacle_x_m']}, "
              f"clears {clip.get('height_at_obstacle_m')}")


if __name__ == "__main__":
    main()
