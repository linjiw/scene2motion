"""Render the EXP-030 obstacle-present demo: one reference, tracked with and without a box.

Every earlier video in this project replayed achieved states against a box that was never in
the physics scene.  EXP-030 put the box in Isaac.  This script renders the achieved states of
that campaign through MuJoCo with the *same* obstacle welded into the world, so a reader
watches the robot meet the object it actually ran into.

Nothing is generated or re-simulated here.  Every frame is an archived achieved state read
from ``outputs/exp030_obstacle_present/launches/*/attempt-000/eval/achieved_qpos.npz``; every
label is read from that campaign's ``rows.jsonl``.  The reference is chosen by a rule applied
to those rows (``select_reference``), not by eye.

Vocabulary follows ``docs/project-goal-2026-09-02.md``: the obstacle-absent arm is a *control*
and its completion is not a traversal result; a cut-off run is an evaluator stopping rule, not
a fall; "contacted the obstacle" is a real physics contact in this campaign, unlike every
earlier replay-scored collision count in the project.

Usage::

    MUJOCO_GL=glfw $S2M_PY experiments/render_obstacle_demo.py --out docs/media
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from scene2motion.robot import ARDY_G1_XML, BODY_MARGIN, build_scene_xml  # noqa: E402
from scene2motion.sonic_state_export import (  # noqa: E402
    load_sonic_state_rollouts,
    sonic_state_sample_dt,
)
from scene2motion.stepover_eval import step_scene  # noqa: E402

# ---------------------------------------------------------------------------------------
# campaign layout (read-only)
# ---------------------------------------------------------------------------------------

CAMPAIGN = Path("outputs/exp030_obstacle_present")
ARM_CHUNKS = {
    "absent": ("absent_chunk00_seed0", "absent_chunk01_seed0"),
    "present_05": ("present_05_chunk00_seed0", "present_05_chunk01_seed0"),
    "present_20": ("present_20_chunk00_seed0", "present_20_chunk01_seed0"),
}
EVAL_SUBPATH = "attempt-000/eval"

# ---------------------------------------------------------------------------------------
# rendering constants
# ---------------------------------------------------------------------------------------

RENDER_W, RENDER_H = 640, 360
HEADER_H, FOOTER_H = 48, 76
PANEL_W, PANEL_H = RENDER_W, HEADER_H + RENDER_H + FOOTER_H
FRAME_STRIDE = 2            # 50 Hz achieved states -> 25 fps video
HOLD_S = 1.2                # how long a finished panel holds its last archived state
CRF = 30

CAMERA_DISTANCE_M = 3.8
CAMERA_AZIMUTH_DEG = 112.0
CAMERA_ELEVATION_DEG = -15.0
CAMERA_LOOK_Z_M = 0.62
CAMERA_DRIFT = 0.35         # bias the look-at toward the obstacle so the crossing stays framed
CAMERA_DRIFT_CAP_M = 1.2

OBSTACLE_RGBA = "0.83 0.42 0.13 0.97"   # the page's clay accent

INK = (238, 238, 240)
DIM = (150, 156, 166)
CLAY = (211, 107, 33)
GOAL = (110, 160, 200)
PAPER = (17, 19, 24)
RULE = (52, 56, 64)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
ROUTE_MAX_M = 7.6           # route bar spans 0 .. ROUTE_MAX_M metres of world x
BAR_PAD = 22


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / name), size)


# ---------------------------------------------------------------------------------------
# campaign reading and reference selection
# ---------------------------------------------------------------------------------------


def read_rows(repo: Path) -> dict[str, dict[str, dict]]:
    """rows.jsonl indexed as ``rows[motion_key][arm]``."""
    rows: dict[str, dict[str, dict]] = {}
    with (repo / CAMPAIGN / "rows.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            rows.setdefault(row["motion_key"], {})[row["arm"]] = row
    missing = [k for k, arms in rows.items() if set(arms) != set(ARM_CHUNKS)]
    if missing:
        raise ValueError(f"rows.jsonl is missing arms for {missing[:3]}")
    return rows


def select_reference(rows: dict[str, dict[str, dict]], min_gap_m: float = 1.0) -> dict:
    """Pick the most instructive reference *from the rows*, and record why.

    The demo has to show one reference doing two different things, so the rule is:
    among references the 20 cm box stopped (outcome ``collided_obstacle``) and that
    travelled at least ``min_gap_m`` further in the obstacle-absent control arm, take the
    one that travelled furthest without the box.  Ties are impossible on float metres; the
    full ranking is returned so the choice is auditable rather than asserted.
    """
    eligible = []
    for key, arms in rows.items():
        absent, present20 = arms["absent"], arms["present_20"]
        gap = float(absent["max_root_x_m"]) - float(present20["max_root_x_m"])
        if present20["outcome"] != "collided_obstacle" or gap < min_gap_m:
            continue
        eligible.append((float(absent["max_root_x_m"]), key, gap))
    if not eligible:
        raise ValueError("no reference satisfies the selection rule")
    eligible.sort(reverse=True)
    best_x, best_key, best_gap = eligible[0]
    return {
        "motion_key": best_key,
        "rule": (
            "among references whose 20 cm-box rollout ended in obstacle contact and that "
            f"travelled at least {min_gap_m} m further in the obstacle-absent control arm, "
            "the one with the greatest obstacle-absent achieved max root x"
        ),
        "eligible_count": len(eligible),
        "absent_max_root_x_m": best_x,
        "present_20_gap_m": best_gap,
        "ranking_top5": [
            {"motion_key": k, "absent_max_root_x_m": x, "gap_vs_present_20_m": g}
            for x, k, g in eligible[:5]
        ],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Clip:
    arm: str
    motion_key: str
    qpos: np.ndarray            # (valid_length, nq) achieved states, padding removed
    sample_dt_s: float
    archive: Path
    archive_sha256: str
    row: dict
    box_height_m: float | None


def load_clip(repo: Path, motion_key: str, arm: str, row: dict) -> Clip:
    for chunk in ARM_CHUNKS[arm]:
        eval_dir = repo / CAMPAIGN / "launches" / chunk / EVAL_SUBPATH
        rollouts = {r.motion_key: r for r in load_sonic_state_rollouts(eval_dir)}
        if motion_key not in rollouts:
            continue
        rollout = rollouts[motion_key]
        qpos = np.asarray(rollout.qpos[: rollout.valid_length], dtype=float)
        if rollout.valid_length != int(row["valid_frames"]):
            raise ValueError(
                f"{arm}/{motion_key}: archive has {rollout.valid_length} valid samples, "
                f"rows.jsonl says {row['valid_frames']}")
        archive = eval_dir / "achieved_qpos.npz"
        return Clip(
            arm=arm, motion_key=motion_key, qpos=qpos,
            sample_dt_s=sonic_state_sample_dt(eval_dir),
            archive=archive, archive_sha256=sha256_file(archive), row=row,
            box_height_m=row["box_height_m"],
        )
    raise KeyError(f"{motion_key} is not in any {arm} launch archive")


# ---------------------------------------------------------------------------------------
# scene and MuJoCo rendering
# ---------------------------------------------------------------------------------------


def demo_scene(row: dict):
    """The obstacle exactly as the campaign configured it, or None for the control arm."""
    if row["box_height_m"] is None:
        return None
    return step_scene(float(row["obstacle_x_m"]), float(row["box_height_m"]),
                      float(row["obstacle_depth_m"]))


def sample_indices(n_samples: int) -> list[int]:
    """Strided achieved-state indices, always ending on the last archived sample."""
    indices = list(range(0, n_samples, FRAME_STRIDE))
    if indices[-1] != n_samples - 1:
        indices.append(n_samples - 1)
    return indices


def render_states(qpos: np.ndarray, row: dict) -> list[np.ndarray]:
    """One RGB frame per achieved state in ``sample_indices``, camera tracking the pelvis.

    The box is welded at its *true* size (``body_margin=0``).  This model is only rendered,
    never scored; drawing the margin-inflated collision box would show the reader a larger
    obstacle than the campaign put in the physics scene.
    """
    scene = demo_scene(row)
    xml = build_scene_xml(scene, ARDY_G1_XML, body_margin=0.0)
    xml = xml.replace('rgba="0.6 0.6 0.65 0.4"', f'rgba="{OBSTACLE_RGBA}"')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    if model.nq != qpos.shape[1]:
        raise ValueError(f"model nq={model.nq} but achieved states have {qpos.shape[1]}")
    renderer = mujoco.Renderer(model, RENDER_H, RENDER_W)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    obstacle_x = float(row["obstacle_x_m"])
    frames: list[np.ndarray] = []
    try:
        for index in sample_indices(len(qpos)):
            data.qpos[:] = qpos[index]
            mujoco.mj_forward(model, data)
            root_x = float(data.qpos[0])
            drift = max(-CAMERA_DRIFT_CAP_M, min(CAMERA_DRIFT_CAP_M, obstacle_x - root_x))
            camera.lookat[:] = [root_x + CAMERA_DRIFT * drift, 0.0, CAMERA_LOOK_Z_M]
            camera.distance = CAMERA_DISTANCE_M
            camera.azimuth = CAMERA_AZIMUTH_DEG
            camera.elevation = CAMERA_ELEVATION_DEG
            renderer.update_scene(data, camera=camera, scene_option=options)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    return frames


# ---------------------------------------------------------------------------------------
# captions: every string below is derived from rows.jsonl, none is hand-written per clip
# ---------------------------------------------------------------------------------------

OUTCOME_WORDS = {
    "completed": "completed the route",
    "collided_obstacle": "contacted the obstacle",
    "collided_wall": "contacted the corridor wall",
    "cutoff": "evaluator cut the run off",
    "stalled": "stalled short of the obstacle",
    "timeout": "ran out of time",
    "fell": "fell",
    "rejected": "rejected before execution",
}


def captions(clip: Clip) -> dict[str, str]:
    row, traversal = clip.row, clip.row["traversal"]
    if clip.box_height_m is None:
        title = "no box in the physics scene"
        subtitle = "obstacle-absent control arm"
    else:
        title = f"{clip.box_height_m * 100:.0f} cm box in the physics scene"
        subtitle = (f"box at x = {row['obstacle_x_m']:.2f} m, "
                    f"{row['obstacle_depth_m'] * 100:.0f} cm deep, "
                    f"{row['obstacle_width_m']:.1f} m wide")
    parts = [OUTCOME_WORDS[row["outcome"]]]
    contact = traversal.get("obstacle_first_collision_sample")
    if contact is not None:
        parts.append(f"first contact {contact * clip.sample_dt_s:.2f} s")
    if row["tracker_terminated"]:
        parts.append(f"evaluator cutoff at {row['valid_time_s']:.2f} s")
    if row["outcome"] == "completed" and clip.box_height_m is None:
        parts.append("control arm, not a traversal result")
    return {
        "title": title,
        "subtitle": subtitle,
        "status": " · ".join(parts),
        "outcome": row["outcome"],
    }


# ---------------------------------------------------------------------------------------
# panel composition
# ---------------------------------------------------------------------------------------


def compose_panel(frame: np.ndarray, clip: Clip, sample: int, held: bool,
                  caps: dict[str, str]) -> Image.Image:
    """Render + header + route bar for one arm at one archived sample."""
    panel = Image.new("RGB", (PANEL_W, PANEL_H), PAPER)
    panel.paste(Image.fromarray(frame), (0, HEADER_H))
    draw = ImageDraw.Draw(panel)

    f_title = _font("DejaVuSans-Bold.ttf", 17)
    f_small = _font("DejaVuSans.ttf", 12)
    f_mono = _font("DejaVuSansMono.ttf", 12)

    draw.text((BAR_PAD, 8), caps["title"], font=f_title, fill=INK)
    draw.text((BAR_PAD, 29), caps["subtitle"], font=f_small, fill=DIM)
    label = f"{clip.motion_key} · {clip.arm}"
    draw.text((PANEL_W - BAR_PAD - draw.textlength(label, font=f_mono), 30),
              label, font=f_mono, fill=DIM)

    top = HEADER_H + RENDER_H
    x0, x1 = BAR_PAD, PANEL_W - BAR_PAD

    # line 1: where the robot is, and when
    root_x = float(clip.qpos[sample, 0])
    readout = f"x = {root_x:5.2f} m    t = {sample * clip.sample_dt_s:5.2f} s"
    draw.text((x0, top + 5), readout, font=f_mono, fill=INK)

    # line 2: the outcome this run ended in, shrunk to fit rather than clipped
    status = caps["status"] + ("  \u00b7  last archived state held" if held else "")
    size = 12
    while size > 8 and draw.textlength(status, font=_font("DejaVuSans.ttf", size)) > x1 - x0:
        size -= 1
    draw.text((x0, top + 24), status, font=_font("DejaVuSans.ttf", size),
              fill=CLAY if held and caps["outcome"] != "completed" else DIM)

    # line 3: route bar, world x from 0 to ROUTE_MAX_M, same scale in every panel
    bar_y = top + 52

    def to_px(metres: float) -> float:
        return x0 + (x1 - x0) * max(0.0, min(1.0, metres / ROUTE_MAX_M))

    draw.line([(x0, bar_y), (x1, bar_y)], fill=RULE, width=3)
    goal_px = to_px(float(clip.row["traversal"]["planned_distance_m"]))
    draw.line([(goal_px, bar_y - 6), (goal_px, bar_y + 6)], fill=GOAL, width=2)
    draw.text((goal_px - 13, bar_y + 7), "goal", font=f_small, fill=GOAL)
    box_px = to_px(float(clip.row["obstacle_x_m"]))
    boxed = clip.box_height_m is not None
    draw.line([(box_px, bar_y - 6), (box_px, bar_y + 6)],
              fill=CLAY if boxed else DIM, width=3)
    draw.text((box_px - 9, bar_y + 7), "box" if boxed else "no box",
              font=f_small, fill=CLAY if boxed else DIM)

    marker = to_px(root_x)
    draw.line([(x0, bar_y), (marker, bar_y)], fill=INK, width=3)
    draw.ellipse([marker - 4, bar_y - 4, marker + 4, bar_y + 4], fill=INK)
    return panel


def panel_frames(clip: Clip, total_out_frames: int) -> list[Image.Image]:
    """One composed panel per output frame; after the run ends the last state is held."""
    caps = captions(clip)
    samples = sample_indices(len(clip.qpos))
    rendered = render_states(clip.qpos, clip.row)
    frames: list[Image.Image] = []
    for i in range(total_out_frames):
        index = min(i, len(rendered) - 1)
        frames.append(compose_panel(rendered[index], clip, samples[index],
                                    held=index < i, caps=caps))
    return frames


def stack(panels: list[list[Image.Image]], footer: str) -> list[Image.Image]:
    """Lay panels side by side with a shared footer line."""
    n = len(panels)
    strip = 26
    width, height = PANEL_W * n, PANEL_H + strip
    f_small = _font("DejaVuSans.ttf", 12)
    out: list[Image.Image] = []
    for i in range(len(panels[0])):
        canvas = Image.new("RGB", (width, height), PAPER)
        for j, panel in enumerate(panels):
            canvas.paste(panel[i], (j * PANEL_W, 0))
        draw = ImageDraw.Draw(canvas)
        for j in range(1, n):
            draw.line([(j * PANEL_W, 0), (j * PANEL_W, PANEL_H)], fill=RULE, width=2)
        draw.line([(0, PANEL_H), (width, PANEL_H)], fill=RULE, width=1)
        draw.text((BAR_PAD, PANEL_H + 7), footer, font=f_small, fill=DIM)
        out.append(canvas)
    return out


# ---------------------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------------------


def encode(frames: list[Image.Image], fps: float, destination: Path) -> None:
    with tempfile.TemporaryDirectory() as workdir:
        for i, frame in enumerate(frames):
            frame.save(Path(workdir) / f"f{i:05d}.png")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", f"{fps:g}", "-i", str(Path(workdir) / "f%05d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", str(CRF),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(destination),
        ], check=True)


# ---------------------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------------------


def scene_record(clip: Clip) -> dict:
    scene = demo_scene(clip.row)
    if scene is None:
        return {
            "obstacle_drawn": False,
            "why": "the campaign's obstacle-absent control arm had no box in the physics scene",
        }
    box = scene.boxes[0]
    return {
        "obstacle_drawn": True,
        "source": "scene2motion.stepover_eval.step_scene",
        "scene_id": scene.scene_id,
        "center_xyz_m": [float(v) for v in box.center],
        "half_extents_m": [float(v) for v in box.half],
        "obstacle_x_m": float(clip.row["obstacle_x_m"]),
        "height_m": float(clip.row["box_height_m"]),
        "depth_m": float(clip.row["obstacle_depth_m"]),
        "width_m": float(clip.row["obstacle_width_m"]),
        "drawn_body_margin_m": 0.0,
        "note": (
            "drawn at true size; the collision metric inflates the same box by "
            f"BODY_MARGIN = {BODY_MARGIN} m, which is not what the renderer shows"
        ),
    }


def poster_sample_target(clips: dict[str, Clip]) -> int:
    """The one instant every poster shows: first contact with the tallest box.

    A shared instant makes the posters directly comparable — the same reference, the same
    controller, the same moment, with and without the box.
    """
    contacted = [c for c in clips.values()
                 if c.box_height_m is not None
                 and c.row["traversal"].get("obstacle_first_collision_sample") is not None]
    if not contacted:
        return min(len(c.qpos) for c in clips.values()) // 2
    tallest = max(contacted, key=lambda c: c.box_height_m)
    return int(tallest.row["traversal"]["obstacle_first_collision_sample"])


def poster_frame_index(clip: Clip, target: int, n_frames: int) -> int:
    """The output frame whose archived sample is closest to ``target``."""
    samples = sample_indices(len(clip.qpos))
    index = min(range(len(samples)), key=lambda i: abs(samples[i] - target))
    return min(index, n_frames - 1)


def clip_record(clip: Clip, files: dict[str, Path], repo: Path, fps: float,
                frames: int, size: tuple[int, int]) -> dict:
    row, traversal = clip.row, clip.row["traversal"]
    contact = traversal.get("obstacle_first_collision_sample")
    return {
        "motion_key": clip.motion_key,
        "seed": row["seed"],
        "arm": clip.arm,
        "obstacle_in_physics": bool(row["obstacle_in_physics"]),
        "source_archive": str(clip.archive.relative_to(repo)),
        "source_archive_sha256": clip.archive_sha256,
        "sample_dt_s": clip.sample_dt_s,
        "valid_frames": int(row["valid_frames"]),
        "valid_time_s": float(row["valid_time_s"]),
        "outcome": row["outcome"],
        "outcome_in_words": OUTCOME_WORDS[row["outcome"]],
        "tracker_terminated": bool(row["tracker_terminated"]),
        "achieved_max_root_x_m": float(row["max_root_x_m"]),
        "achieved_final_root_x_m": float(row["final_root_x_m"]),
        "min_pelvis_z_m": float(traversal["min_pelvis_z_m"]),
        "final_pelvis_z_m": float(clip.qpos[-1, 2]),
        "obstacle_first_contact_sample": contact,
        "obstacle_first_contact_s": None if contact is None
        else contact * clip.sample_dt_s,
        "obstacle_max_penetration_m": traversal["obstacle_max_penetration_m"],
        "reached_goal": bool(traversal["reached_goal"]),
        "scene_drawn": scene_record(clip),
        "video": str(files["video"].relative_to(repo)),
        "video_bytes": files["video"].stat().st_size,
        "poster": str(files["poster"].relative_to(repo)),
        "poster_bytes": files["poster"].stat().st_size,
        "fps": fps,
        "frames": frames,
        "width": size[0],
        "height": size[1],
    }


# ---------------------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/media")
    parser.add_argument("--motion-key", default=None,
                        help="override the selection rule (the rule is applied and printed "
                             "either way, and a mismatch is recorded in the manifest)")
    args = parser.parse_args(argv)
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to encode the demo videos")

    repo = Path(__file__).resolve().parents[1]
    out = (repo / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_rows(repo)
    selection = select_reference(rows)
    key = args.motion_key or selection["motion_key"]
    selection["rendered_motion_key"] = key
    selection["overridden"] = key != selection["motion_key"]
    print(f"selection rule picked {selection['motion_key']} "
          f"(of {selection['eligible_count']} eligible); rendering {key}")

    clips = {arm: load_clip(repo, key, arm, rows[key][arm]) for arm in ARM_CHUNKS}
    fps = 1.0 / (clips["absent"].sample_dt_s * FRAME_STRIDE)
    hold = int(round(HOLD_S * fps))
    poster_target = poster_sample_target(clips)

    manifest: dict = {
        "schema": "scene2motion-obstacle-demo-manifest-v1",
        "campaign": str(CAMPAIGN),
        "campaign_receipt": str(CAMPAIGN / "receipt.json"),
        "campaign_receipt_sha256": sha256_file(repo / CAMPAIGN / "receipt.json"),
        "campaign_identity_sha256": json.loads(
            (repo / CAMPAIGN / "receipt.json").read_text())["campaign_identity_sha256"],
        "rows": str(CAMPAIGN / "rows.jsonl"),
        "rows_sha256": sha256_file(repo / CAMPAIGN / "rows.jsonl"),
        "physics_seed": clips["absent"].row["physics_seed"],
        "renderer": {
            "engine": f"mujoco {mujoco.__version__}",
            "model": str(ARDY_G1_XML),
            "model_sha256": sha256_file(ARDY_G1_XML),
            "gl": os.environ.get("MUJOCO_GL"),
            "frame_stride": FRAME_STRIDE,
            "source_rate_hz": 1.0 / clips["absent"].sample_dt_s,
            "video_fps": fps,
            "crf": CRF,
            "camera": {
                "azimuth_deg": CAMERA_AZIMUTH_DEG,
                "elevation_deg": CAMERA_ELEVATION_DEG,
                "distance_m": CAMERA_DISTANCE_M,
                "lookat_z_m": CAMERA_LOOK_Z_M,
                "tracks": "pelvis, biased toward the obstacle",
            },
        },
        "selection": selection,
        "poster_sample_target": poster_target,
        "poster_sample_target_why": (
            "the archived sample at which the tallest box was first contacted, so every "
            "poster shows the same instant of the same reference"
        ),
        "clips": {},
        "composites": {},
        "reading_notes": [
            "Every frame is an archived achieved state from EXP-030; nothing was re-simulated.",
            "The obstacle-absent arm is the campaign's control: its completion is not a "
            "local traversal result, because there was no obstacle to traverse.",
            "An evaluator cutoff is a tracking-error stopping rule, not a fall; see "
            "final_pelvis_z_m for the pelvis height at the last archived state of each clip.",
            "Contact with the box in the present arms is a physics contact in the Isaac "
            "scene, not a replay-scored geometry query.",
        ],
    }

    # ---- single-arm clips -------------------------------------------------------------
    panels: dict[str, list[Image.Image]] = {}
    for arm, clip in clips.items():
        samples = sample_indices(len(clip.qpos))
        frames = panel_frames(clip, len(samples) + hold)
        panels[arm] = frames
        stem = f"{key}_{arm}"
        video = out / f"{stem}.mp4"
        encode(frames, fps, video)
        poster_index = poster_frame_index(clip, poster_target, len(frames))
        poster = out / f"{stem}.png"
        frames[poster_index].save(poster, optimize=True)
        manifest["clips"][arm] = clip_record(
            clip, {"video": video, "poster": poster}, repo, fps, len(frames),
            (PANEL_W, PANEL_H))
        manifest["clips"][arm]["poster_sample"] = samples[min(poster_index,
                                                              len(samples) - 1)]
        print(f"  {stem}.mp4 {video.stat().st_size / 1024:6.0f} KB   "
              f"{stem}.png {poster.stat().st_size / 1024:5.0f} KB")

    # ---- composites -------------------------------------------------------------------
    footer = (
        f"EXP-030 · reference {key} · same reference, same controller, same physics seed "
        f"{clips['absent'].row['physics_seed']} · one rollout each · "
        "achieved states replayed in MuJoCo, nothing re-simulated"
    )
    composites = {
        "pair_absent_vs_20cm": ["absent", "present_20"],
        "triptych_absent_5cm_20cm": ["absent", "present_05", "present_20"],
    }
    for name, arms in composites.items():
        length = max(len(panels[a]) for a in arms)
        padded = [panels[a] + [panels[a][-1]] * (length - len(panels[a])) for a in arms]
        frames = stack(padded, footer)
        stem = f"{key}_{name}"
        video = out / f"{stem}.mp4"
        encode(frames, fps, video)
        poster_index = poster_frame_index(clips[arms[-1]], poster_target, len(frames))
        poster = out / f"{stem}.png"
        frames[poster_index].save(poster, optimize=True)
        manifest["composites"][name] = {
            "arms": arms,
            "motion_key": key,
            "video": str(video.relative_to(repo)),
            "video_bytes": video.stat().st_size,
            "poster": str(poster.relative_to(repo)),
            "poster_bytes": poster.stat().st_size,
            "poster_sample": sample_indices(len(clips[arms[-1]].qpos))[poster_index],
            "fps": fps,
            "frames": len(frames),
            "width": PANEL_W * len(arms),
            "height": frames[0].height,
            "footer": footer,
            "panels_hold_last_state": True,
        }
        print(f"  {stem}.mp4 {video.stat().st_size / 1024:6.0f} KB   "
              f"{stem}.png {poster.stat().st_size / 1024:5.0f} KB")

    total = sum(r["video_bytes"] + r["poster_bytes"]
                for r in list(manifest["clips"].values())
                + list(manifest["composites"].values()))
    manifest["total_bytes"] = total
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{len(manifest['clips'])} clips + {len(manifest['composites'])} composites, "
          f"{total / 1024 / 1024:.2f} MB, manifest at {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
