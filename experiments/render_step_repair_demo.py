"""Render the deterministic EXP-031 raw/reference-repair comparison.

This is a reference-level visualisation, not a controller rollout.  The script recomputes the
frozen EXP-031 preparation over all 64 assigned references, checks the admitted candidate hash,
and renders the selected raw and repaired arrays against the true-size 5 cm box.  It writes a
small manifest beside the video so the project page can state exactly what the clip establishes.

Usage::

    MUJOCO_GL=egl $S2M_PY experiments/render_step_repair_demo.py --out docs/media
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp031_prepare_step_repair as prep  # noqa: E402
from experiments.render_demo_videos import (  # noqa: E402
    FRAME_STRIDE,
    encode,
    render_clip,
)


PANEL_W, FRAME_H = 640, 360
HEADER_H, FOOTER_H = 58, 58
PANEL_H = HEADER_H + FRAME_H + FOOTER_H
PAPER = (17, 19, 24)
INK = (238, 238, 240)
DIM = (155, 161, 170)
CLAY = (211, 107, 33)
PINE = (59, 174, 148)
RULE = (52, 56, 64)
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / name), size)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _panel(frame: np.ndarray, qpos: np.ndarray, source_index: int, *, title: str,
           subtitle: str, good: bool, status: str) -> Image.Image:
    panel = Image.new("RGB", (PANEL_W, PANEL_H), PAPER)
    panel.paste(Image.fromarray(frame), (0, HEADER_H))
    draw = ImageDraw.Draw(panel)
    draw.text((20, 8), title, font=_font("DejaVuSans-Bold.ttf", 17), fill=INK)
    draw.text((20, 31), subtitle, font=_font("DejaVuSans.ttf", 12), fill=DIM)
    top = HEADER_H + FRAME_H
    draw.line([(0, top), (PANEL_W, top)], fill=RULE, width=1)
    draw.text((20, top + 8), status, font=_font("DejaVuSans.ttf", 12),
              fill=PINE if good else CLAY)
    readout = f"reference frame {source_index:03d}  ·  root x = {qpos[source_index, 0]:.2f} m"
    draw.text((20, top + 31), readout, font=_font("DejaVuSansMono.ttf", 11), fill=DIM)
    return panel


def _pair(raw: np.ndarray, repaired: np.ndarray, record: dict) -> list[Image.Image]:
    raw_frames = render_clip(raw, prep.OBSTACLE_X_M, prep.OBSTACLE_HEIGHT_M,
                             prep.OBSTACLE_DEPTH_M)
    repaired_frames = render_clip(repaired, prep.OBSTACLE_X_M, prep.OBSTACLE_HEIGHT_M,
                                  prep.OBSTACLE_DEPTH_M)
    if len(raw_frames) != len(repaired_frames):
        raise RuntimeError("raw and repaired references changed duration")
    before = record["before"]
    after = record["after"]
    output: list[Image.Image] = []
    for output_index, (left_frame, right_frame) in enumerate(zip(raw_frames, repaired_frames)):
        source_index = output_index * FRAME_STRIDE
        left = _panel(
            left_frame, raw, source_index,
            title="raw frozen-prior reference",
            subtitle="5 cm box at x = 1.20 m · reference geometry only",
            good=False,
            status=(f"intersects inflated box in {before['collision']['penetration_frames']} "
                    "frames"),
        )
        right = _panel(
            right_frame, repaired, source_index,
            title="obstacle-relative leg repair",
            subtitle="same root, upper body, duration and frame rate",
            good=True,
            status=("collision-free reference · support screen passes · "
                    f"max leg edit {record['deformation']['max_leg_joint_delta_rad']:.3f} rad"),
        )
        canvas = Image.new("RGB", (2 * PANEL_W, PANEL_H + 28), PAPER)
        canvas.paste(left, (0, 0))
        canvas.paste(right, (PANEL_W, 0))
        draw = ImageDraw.Draw(canvas)
        draw.line([(PANEL_W, 0), (PANEL_W, PANEL_H)], fill=RULE, width=2)
        draw.line([(0, PANEL_H), (2 * PANEL_W, PANEL_H)], fill=RULE, width=1)
        draw.text(
            (20, PANEL_H + 7),
            "EXP-031 pre-execution computation · no SONIC rollout · no traversal claim",
            font=_font("DejaVuSans.ttf", 12), fill=DIM,
        )
        output.append(canvas)
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/media")
    parser.add_argument("--motion-key", default="s4434")
    args = parser.parse_args(argv)
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to encode the demo")
    if args.motion_key not in prep.EXPECTED_ACCEPTED_KEYS:
        raise SystemExit(
            f"motion key must be one of the frozen admitted keys {prep.EXPECTED_ACCEPTED_KEYS}"
        )

    repo = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    if not out.is_absolute():
        out = repo / out
    out.mkdir(parents=True, exist_ok=True)

    records, candidates, counts = prep.build_records()
    rows = {record["motion_key"]: record for record in records}
    record = rows[args.motion_key]
    repaired = np.asarray(candidates[args.motion_key], dtype=np.float32)
    with np.load(prep.SOURCE / "qpos.npz", allow_pickle=False) as archive:
        raw = np.asarray(archive[args.motion_key], dtype=np.float32)
    if prep.qpos_sha256(repaired) != prep.EXPECTED_CANDIDATE_ARRAY_SHA256[args.motion_key]:
        raise RuntimeError("repaired candidate no longer matches its frozen content hash")

    frames = _pair(raw, repaired, record)
    stem = f"{args.motion_key}_reference_repair"
    video = out / f"{stem}.mp4"
    poster = out / f"{stem}.png"
    encode([np.asarray(frame) for frame in frames], prep.FPS / FRAME_STRIDE, video)
    poster_source = int(np.argmin(np.abs(raw[:, 0] - prep.OBSTACLE_X_M)))
    poster_index = min(len(frames) - 1, round(poster_source / FRAME_STRIDE))
    frames[poster_index].save(poster, optimize=True)

    history = prep.historical_outcome_disclosure()
    source_qpos = prep.SOURCE / "qpos.npz"
    manifest = {
        "schema": "scene2motion-step-repair-demo-v1",
        "evidence_level": "reference_geometry_only",
        "not_evidence_of": ["SONIC tracking", "physical obstacle traversal", "success rate"],
        "source_pool": {
            "path": str(source_qpos.relative_to(repo)),
            "sha256": prep.sha256_file(source_qpos),
            "n_assigned_references": counts["n_assigned_trials"],
        },
        "preexecution_accounting": counts,
        "selection": {
            "rendered_motion_key": args.motion_key,
            "default_motion_key": "s4434",
            "overridden": args.motion_key != "s4434",
            "rule": ("by default, render the only admitted candidate that historically "
                     "completed the obstacle-absent route; this illustration choice is not "
                     "an admission rule"),
            "historical_outcome_disclosure": history["candidates"][args.motion_key],
        },
        "motion": {
            "raw_array_sha256": prep.qpos_sha256(raw),
            "repaired_array_sha256": prep.qpos_sha256(repaired),
            "record": record,
        },
        "scene": {
            "obstacle_x_m": prep.OBSTACLE_X_M,
            "obstacle_height_m": prep.OBSTACLE_HEIGHT_M,
            "obstacle_depth_m": prep.OBSTACLE_DEPTH_M,
            "drawn_at_true_size": True,
            "body_margin_m_used_for_scoring_not_drawing": record["scene"]["body_margin_m"],
        },
        "renderer": {
            "engine": f"MuJoCo {mujoco.__version__} forward-kinematic replay",
            "gl": os.environ.get("MUJOCO_GL"),
            "model": {"path": str(Path(prep.ARDY_G1_XML)),
                      "sha256": prep.sha256_file(prep.ARDY_G1_XML)},
            "source": {
                "path": str(Path(__file__).resolve().relative_to(repo)),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "source_fps": prep.FPS,
            "frame_stride": FRAME_STRIDE,
            "video_fps": prep.FPS / FRAME_STRIDE,
            "frames": len(frames),
            "width": frames[0].width,
            "height": frames[0].height,
        },
        "files": {
            "video": {"path": str(video.relative_to(repo)), "sha256": _sha256(video),
                      "bytes": video.stat().st_size},
            "poster": {"path": str(poster.relative_to(repo)), "sha256": _sha256(poster),
                       "bytes": poster.stat().st_size, "source_frame": poster_source},
        },
    }
    manifest_path = out / "step_repair_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {video.relative_to(repo)} ({video.stat().st_size / 1024:.0f} KB)")
    print(f"wrote {poster.relative_to(repo)} ({poster.stat().st_size / 1024:.0f} KB)")
    print(f"wrote {manifest_path.relative_to(repo)}")


if __name__ == "__main__":
    main()
