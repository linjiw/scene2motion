"""Render archived clips as MuJoCo simulation video, robot and obstacle in one scene.

The motion panels elsewhere on the findings page draw a stick figure from forward
kinematics, which is exact but abstract.  This renders the same archived clips through
MuJoCo's own renderer with the obstacle welded into the world, so a reader sees the
released G1 model and the box it is or is not clearing, at the geometry the collision
metric actually uses.

No generation: every clip is read from a committed campaign archive, and the obstacle is
the one that campaign scored the clip against.  Frames are encoded to H.264 with ffmpeg
and left small enough to embed in the page as data URIs.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402

from experiments import calibrate_ramp_route_phase as cal  # noqa: E402
from experiments import exp019_ramp_gait_matched_stepover as e19  # noqa: E402
from scene2motion.robot import ARDY_G1_XML, BODY_MARGIN, build_scene_xml  # noqa: E402
from scene2motion.scenes import Box, Scene  # noqa: E402


WIDTH, HEIGHT = 640, 360
FRAME_STRIDE = 2          # 25 fps source -> 12.5 fps video, half the frames to encode
CRF = 30                  # visually clean at this size, small enough to inline
CAMERA_BACK_M = 2.9       # how far behind the robot the camera sits
CAMERA_HEIGHT_M = 1.25
CAMERA_SIDE_M = 2.6


def obstacle_scene(obstacle_x: float, height_m: float, depth_m: float) -> Scene:
    """The obstacle the campaign scored against, as a welded world box."""
    return Scene(
        scene_id=f"demo_{obstacle_x:.2f}",
        family="demo",
        boxes=[Box(center=(obstacle_x, 0.0, height_m / 2),
                   half=(depth_m / 2, 0.7, height_m / 2), label="obstacle")],
        start=(0.0, 0.0), goal=(7.2, 0.0),
    )


OBSTACLE_RGBA = "0.83 0.42 0.13 0.95"   # the page's clay accent, opaque enough to read


def render_clip(qpos: np.ndarray, obstacle_x: float, height_m: float,
                depth_m: float) -> list[np.ndarray]:
    """One RGB frame per strided sample, camera tracking the robot from the side.

    The box is welded at its *true* size (``body_margin=0``): this model is only ever
    rendered, never used for a collision metric, and showing the margin-inflated box
    would overstate the obstacle a reader is watching the robot clear.
    """
    xml = build_scene_xml(
        obstacle_scene(obstacle_x, height_m, depth_m), ARDY_G1_XML, body_margin=0.0)
    xml = xml.replace('rgba="0.6 0.6 0.65 0.4"', f'rgba="{OBSTACLE_RGBA}"')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    frames: list[np.ndarray] = []
    try:
        for index in range(0, len(qpos), FRAME_STRIDE):
            data.qpos[:] = np.asarray(qpos[index], dtype=float)
            mujoco.mj_forward(model, data)
            # Track the pelvis, but drift toward the obstacle so the crossing stays framed.
            root_x = float(data.qpos[0])
            look = root_x + 0.35 * max(-1.2, min(1.2, obstacle_x - root_x))
            camera.lookat[:] = [look, 0.0, 0.85]
            camera.distance = CAMERA_BACK_M
            camera.azimuth = 90.0
            camera.elevation = -10.0
            renderer.update_scene(data, camera=camera, scene_option=options)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    return frames


def encode(frames: list[np.ndarray], fps: float, destination: Path) -> Path:
    """Write frames to H.264 in an MP4 faststart container."""
    with tempfile.TemporaryDirectory() as workdir:
        for i, frame in enumerate(frames):
            path = Path(workdir) / f"f{i:05d}.png"
            _write_png(frame, path)
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", f"{fps:g}", "-i", str(Path(workdir) / "f%05d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", str(CRF),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(destination),
        ]
        subprocess.run(command, check=True)
    return destination


def _write_png(frame: np.ndarray, path: Path) -> None:
    from PIL import Image
    Image.fromarray(frame).save(path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/demo_videos")
    parser.add_argument("--manifest", default="outputs/demo_videos.json")
    args = parser.parse_args(argv)
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to encode the demo videos")
    repo = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    motions = json.loads((repo / "outputs/demo_motions.json").read_text())
    e1a = repo / "outputs/exp019_gait_matched_stepover_v7"
    receipt = json.loads((e1a / "receipt.json").read_text())
    per_seed = receipt["placement_selection"]["per_seed"]
    donor = np.load(e1a / "donor_qpos.npz")
    pool = np.load(e1a / "pool_qpos.npz")
    arms = np.load(e1a / "arm_qpos.npz")
    distribution = np.load(
        repo / "outputs/exp021_elicited_lift_distribution_v2/qpos.npz")
    staged_rows = [
        json.loads(line) for line in
        (repo / "outputs/exp021_elicited_lift_distribution_v2/rows.jsonl").open()]
    staged = max((r for r in staged_rows
                  if r["lift_x_m"] is not None and 0.9 <= r["lift_x_m"] <= 1.6),
                 key=lambda r: r["lift_height_m"])

    walk_label = per_seed["4353"]["speed_label"]
    packet_label = per_seed["4353"]["speed_label"]
    sources = {
        "donor": np.asarray(donor["s2603__adapted"], dtype=float),
        "walk": np.asarray(pool[f"pilot__{walk_label}__seed4353"], dtype=float),
        "staged": np.asarray(distribution[f"s{staged['seed']}"], dtype=float),
        "commanded": np.asarray(distribution[f"s{staged['seed']}"], dtype=float),
        "packet": np.asarray(arms["s4353__absolute"], dtype=float),
    }

    manifest: dict[str, dict] = {}
    for clip in motions["clips"]:
        key = clip["key"]
        qpos = sources[key]
        obstacle_x = float(clip["obstacle_x_m"])
        height = float(clip["obstacle_height_m"])
        print(f"rendering {key}: {len(qpos)} frames, obstacle at {obstacle_x:.2f} m")
        frames = render_clip(qpos, obstacle_x, height, float(clip["obstacle_depth_m"]))
        destination = out / f"{key}.mp4"
        encode(frames, cal.FPS / FRAME_STRIDE, destination)
        size = destination.stat().st_size
        manifest[key] = {
            "path": str(destination),
            "bytes": size,
            "frames": len(frames),
            "fps": cal.FPS / FRAME_STRIDE,
            "width": WIDTH, "height": HEIGHT,
            "obstacle_x_m": obstacle_x,
            "obstacle_height_m": height,
            "data_uri": "data:video/mp4;base64," + base64.b64encode(
                destination.read_bytes()).decode(),
        }
        print(f"  -> {destination} ({size / 1024:.0f} KB)")

    Path(args.manifest).write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    total = sum(v["bytes"] for v in manifest.values()) / 1024 / 1024
    print(f"\n{len(manifest)} videos, {total:.2f} MB total")


if __name__ == "__main__":
    main()
