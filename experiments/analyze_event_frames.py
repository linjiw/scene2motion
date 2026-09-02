"""A0 event-frame analyser: when, in each archived clip, does the prompt-elicited step happen?

The plan of record (``docs/plan-2026-09-01-icra2027.md`` §1.1 row 2, §3 row A0, §6) quotes the
timing of the STEP prompt's lift as provisional ("median frame ~34/35, q10-q90 21-55, 43/49 in
the first 50 frames, 4/49 after frame 60, 80-86 % in the first 50 depending on conversion")
because EXP-021 never archived an event frame: its rows carry the lift *position* on the route
(``lift_x_m``) and no committed script converted that position into a time.  This script
derives the event frame of every archived EXP-021 clip (64 STEP clips) and EXP-023 clip (32
clips, four prompt schedules) from the archived qpos under three explicit definitions and
reports the agreement between them, so that every timing number in the paper reproduces from
committed code.

Axis convention (verified in code, not assumed): the route ``cal.route_xz_for_speed`` is
prescribed along its column 1 (ARDY z, 0 -> 7.2 m); ``BoxHeightProbe`` places the box at MuJoCo
world x (``stepover_eval.step_scene``); ``exp017._prescribed_progress_ratio`` and
``exp023.supporting_motion_metrics`` read the realised forward coordinate from ``qpos[:, 0]``.
The forward axis of every archived clip is therefore ``qpos[:, 0]`` (root x ~ 0 at frame 0,
~ 7.18 m at frame 199), and ``stepover_eval._FORWARD`` = +x for the foot envelopes.

Definitions (25 fps; frame k is t = k / 25 s):

  A  root-crossing frame: the first frame at which ``qpos[:, 0]`` reaches ``lift_x_m`` (the
     route position of the clip's tallest whole-body-clearable box; EXP-021's ``lift_x_m``).
  B  nominal-speed conversion: ``lift_x_m / 0.9045 m/s * 25 fps`` -- the (real-valued) frame at
     which the *prescribed* route reaches ``lift_x_m``; reported raw, rounded, floored, ceiled.
  C  foot-based lift frame: the frame at which the lifting foot (EXP-021's ``lift_side``)
     attains its maximum bottom clearance while its forward representative lies inside the
     lift region -- the contiguous positive run of the EXP-021 box-height profile that contains
     ``lift_x_m``, widened by the half-slab (depth/2 + body margin = 0.14 m) that ``lift_side``
     itself uses.  ``C0`` is the same maximum restricted to the +-0.14 m window around
     ``lift_x_m`` (exactly ``lift_side``'s window).  C is also located inside the bilateral
     no-support run that contains it (calibrated support envelope from the exp016 threshold
     receipt, through ``analyze_trackability_contract.support_masks``).

The lift position, height, region count, support and side are recomputed from the archived
qpos with the byte-identical EXP-021 scan (``analyze_e1a_placement.box_height_profile`` at 120
points over the 7.2 m route, ``lift_location``, ``lift_side``) and must reproduce the archived
EXP-021 rows exactly; any mismatch blocks the analysis.  For EXP-023 the archived ``event``
records (96-frame post-onset window, two-foot traversal required) are compared against A and C
rather than replaced.  Everything here is a post hoc CPU analysis of completed archives: no
generator, no tracker, no new samples.

Run:  source env.sh && $S2M_PY experiments/analyze_event_frames.py \
          --out outputs/analysis_event_frames
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments import analyze_exp021_exact_addressability as exact  # noqa: E402
from experiments import calibrate_ramp_route_phase as cal  # noqa: E402
from experiments.analyze_e1a_placement import (  # noqa: E402
    box_height_profile,
    lift_location,
    lift_side,
)
from experiments.analyze_trackability_contract import (  # noqa: E402
    THRESHOLD_RECEIPT,
    THRESHOLD_RECEIPT_SHA256,
    runs_of,
    support_masks,
)


SCHEMA_VERSION = "analysis-event-frames-v1"
FPS = 25.0
N_FRAMES = 200
REFERENCE_SPEED_MPS = float(cal.REFERENCE_SPEED_MPS)  # 7.2 m / 7.96 s = 0.9045 m/s
OBSTACLE_DEPTH_M = 0.20
PROFILE_POINTS = 120  # exp021's SCAN_POINTS
BODY_MARGIN_M = 0.04
HALF_SLAB_M = OBSTACLE_DEPTH_M / 2.0 + BODY_MARGIN_M  # 0.14 m: lift_side's window
FIRST_WINDOW_FRAMES = 50  # "inside the first 50 frames": frame < 50, i.e. t < 2.0 s
LATE_FRAME = 60  # "after frame 60": frame > 60, i.e. t > 2.4 s
HIST_BIN_FRAMES = 10
GATE_RUN_S = 0.2
LIFT_MIN_M = 0.03
LIFT_FIELD_ATOL = 1e-9

DEFAULT_OUT = REPO / "outputs/analysis_event_frames"
EXP021_ARCHIVE = REPO / "outputs/exp021_elicited_lift_distribution_v2"
EXP023_ARCHIVE = REPO / "outputs/exp023_prompt_handoff"

EXP023_ARMS = ("all_walk", "step_0", "step_52", "step_104")
EXP023_ONSETS: Mapping[str, int | None] = {
    "all_walk": None, "step_0": 0, "step_52": 52, "step_104": 104,
}

LOCKED_EXP021_MANIFEST = exact.LOCKED_ARCHIVE_MANIFEST

LOCKED_EXP023_MANIFEST: Mapping[str, Any] = {
    "experiment": "exp023_prompt_handoff",
    "schema": "exp023-prompt-handoff-v1",
    "generation_commit": "d8882254d2d3fc2bf2911c11103381dbf3434c11",
    "analysis_commit": "f7eb604c0ef41e02aced66dd45e673f7b30c33f9",
    "seeds": tuple(range(4500, 4508)),
    "arms": EXP023_ARMS,
    "archived_frames": 208,
    "scored_frames": N_FRAMES,
    "files": {
        "receipt.json": "1e002bd6402daba738fd1557c5dd6fa165adda678e36b25cff815cf440c74fae",
        "rows.jsonl": "9a3596846519c05caf6837a0c41288024308831b40ea0e2fe6d924dfcdb19168",
        "qpos.npz": "14c476e5fece3238146b8d46810cf835521dfd5ad3b8359dece9c7e150ec8dd0",
    },
    "qpos_content_sha256": (
        "84bbbafb76bbf1e03475aef04feac36efdc9f0f27e7fe8889ad510d92d4e8e37"
    ),
}

# Sources whose byte identity defines the event definitions above.  ``stepover_eval`` and
# ``robot`` are the foot envelopes and collision model; ``analyze_e1a_placement`` is the
# EXP-021 scan; ``calibrate_ramp_route_phase`` is the route; ``analyze_trackability_contract``
# is the calibrated support envelope.
LOCKED_SOURCE_FILES: Mapping[str, str] = {
    "scene2motion/stepover_eval.py": (
        "f0e344def2133f3a45935bb1b7ffad8fb2591145557e6b136d50871827cf77b4"
    ),
    "scene2motion/robot.py": (
        "c12b59a999969dd910e0aaef25322695d247bd392899394dd5d35cfc1252fc82"
    ),
    "experiments/analyze_e1a_placement.py": (
        "acef09868fec794ecdbd85f8b297ea40a458c2bd85b66eec31db1d6b9c4d5eaa"
    ),
    "experiments/calibrate_ramp_route_phase.py": (
        "ccbf5a13a78578c31579121650ab9378468c5b73e8c61c758a7c9cc9b503c26a"
    ),
}
# The calibrated support envelope is imported from the (still evolving) trackability
# analysis; bind the two imported functions by their source text rather than the whole file.
LOCKED_FUNCTION_SOURCES: Mapping[str, str] = {
    "analyze_trackability_contract.support_masks": (
        "7aeef6f225e4c73947e32cb6015da614f8de1ce6da84c9ff7856d24e1762db86"
    ),
    "analyze_trackability_contract.runs_of": (
        "a816ddd96103e108528dddac25e7983eaee08ba965895b04b5d8ba5e19ee4a89"
    ),
}
# Recorded for provenance only (the campaigns that produced the archives, and the file the
# locked functions live in); not re-executed and not locked.
RECORDED_SOURCE_FILES = (
    "experiments/exp021_elicited_lift_distribution.py",
    "experiments/exp023_prompt_handoff.py",
    "experiments/analyze_trackability_contract.py",
)
LOCKED_G1_XML_SHA256 = exact.LOCKED_G1_XML_SHA256

PLAN_PROVISIONAL_NUMBERS: Mapping[str, Any] = {
    "source": "docs/plan-2026-09-01-icra2027.md §1.1 row 2 and §6; CLAUDE.md audit table",
    "root_trajectory_median_frame": 35,
    "root_trajectory_q10_q90_frames": [21, 55],
    "inside_first_50_frames": "43/49",
    "after_frame_60": "4/49 (an earlier note said 3/49)",
    "first_50_fraction_by_conversion": "80-86 % depending on the frame conversion",
    "historical_conversion_median_frame": 34,
    "exp023_step_0_latencies_frames": [18, 39, 59, 50, 64, 33],
    "exp023_step_0_median_latency_frames": 44.5,
}


# --------------------------------------------------------------------------------------
# Pure helpers (no MuJoCo): the three definitions, agreement and summaries.
# --------------------------------------------------------------------------------------

def root_crossing_frame(root_forward: Sequence[float], target_x_m: float) -> int | None:
    """Definition A: first frame at which the root forward coordinate reaches ``target_x_m``."""
    x = np.asarray(root_forward, dtype=float)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("root forward coordinate must be a non-empty finite 1-D series")
    target = float(target_x_m)
    if not math.isfinite(target):
        raise ValueError("target x must be finite")
    hits = np.flatnonzero(x >= target)
    return int(hits[0]) if hits.size else None


def nominal_frame(x_m: float, *, speed_mps: float = REFERENCE_SPEED_MPS,
                  fps: float = FPS) -> float:
    """Definition B: the real-valued frame at which the prescribed route reaches ``x_m``."""
    if not math.isfinite(float(x_m)) or speed_mps <= 0.0 or fps <= 0.0:
        raise ValueError("nominal conversion requires finite x and positive speed and fps")
    return float(x_m) / float(speed_mps) * float(fps)


def lift_region_extent(xs: Sequence[float], heights: Sequence[float],
                       lift_x_m: float) -> dict[str, Any]:
    """The contiguous positive run of the box-height profile that contains ``lift_x_m``."""
    x = np.asarray(xs, dtype=float)
    h = np.asarray(heights, dtype=float)
    if x.ndim != 1 or h.shape != x.shape or not x.size:
        raise ValueError("profile arrays must be aligned non-empty 1-D arrays")
    if not np.isfinite(x).all() or not np.isfinite(h).all() or np.any(np.diff(x) <= 0):
        raise ValueError("profile must be finite with strictly increasing positions")
    index = int(np.argmin(np.abs(x - float(lift_x_m))))
    if not np.isclose(x[index], float(lift_x_m), atol=LIFT_FIELD_ATOL, rtol=0.0):
        raise ValueError("lift_x_m is not a point of the profile grid")
    if h[index] <= 0.0:
        raise ValueError("lift_x_m does not sit on a positive profile point")
    positive = h > 0.0
    lo = index
    while lo > 0 and positive[lo - 1]:
        lo -= 1
    hi = index
    while hi + 1 < len(x) and positive[hi + 1]:
        hi += 1
    return {
        "peak_index": index,
        "lo_index": int(lo),
        "hi_index": int(hi),
        "lo_x_m": float(x[lo]),
        "hi_x_m": float(x[hi]),
        "n_points": int(hi - lo + 1),
        "extent_m": float(x[hi] - x[lo]),
    }


def foot_lift_frame(foot: Mapping[str, Sequence[float]], lo_x_m: float, hi_x_m: float,
                    *, half_slab_m: float = HALF_SLAB_M) -> dict[str, Any] | None:
    """Definition C: the frame of maximum bottom clearance while the foot is over the region.

    Frames qualify when the foot's forward representative lies in
    ``[lo_x_m - half_slab_m, hi_x_m + half_slab_m]``; ties resolve to the earliest frame.
    """
    forward = np.asarray(foot["forward_representative_m"], dtype=float)
    clearance = np.asarray(foot["bottom_clearance_m"], dtype=float)
    if forward.ndim != 1 or clearance.shape != forward.shape or not forward.size:
        raise ValueError("foot series must be aligned non-empty 1-D arrays")
    if not np.isfinite(forward).all() or not np.isfinite(clearance).all():
        raise ValueError("foot series must be finite")
    if float(hi_x_m) < float(lo_x_m) or half_slab_m < 0.0:
        raise ValueError("region must be ordered and the half-slab non-negative")
    inside = (forward >= float(lo_x_m) - half_slab_m) & (forward <= float(hi_x_m) + half_slab_m)
    frames = np.flatnonzero(inside)
    if not frames.size:
        return None
    best = int(frames[int(np.argmax(clearance[frames]))])
    return {
        "frame": best,
        "clearance_m": float(clearance[best]),
        "n_frames_inside": int(frames.size),
        "first_frame_inside": int(frames[0]),
        "last_frame_inside": int(frames[-1]),
    }


def containing_run(runs: Sequence[tuple[int, int]], frame: int) -> tuple[int, int] | None:
    """The half-open run ``[start, end)`` that contains ``frame``, if any."""
    for start, end in runs:
        if start <= frame < end:
            return int(start), int(end)
    return None


def max_clearance_frame(feet: Mapping[str, Mapping[str, Sequence[float]]]) -> dict[str, Any]:
    """Frame and side of the highest foot bottom over the whole clip (either foot)."""
    best: dict[str, Any] | None = None
    for side in ("left", "right"):
        clearance = np.asarray(feet[side]["bottom_clearance_m"], dtype=float)
        if clearance.ndim != 1 or not clearance.size or not np.isfinite(clearance).all():
            raise ValueError(f"{side} bottom clearance must be a non-empty finite series")
        frame = int(np.argmax(clearance))
        value = float(clearance[frame])
        if best is None or value > best["clearance_m"]:
            best = {"frame": frame, "side": side, "clearance_m": value}
    assert best is not None
    return best


def function_source_sha256(function: Callable[..., Any]) -> str:
    """SHA-256 of a function's source text (binds an imported definition, not its file)."""
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _wilson(successes: int, total: int) -> list[float] | None:
    if total < 1:
        return None
    low, high = exact.wilson_interval(int(successes), int(total))
    return [float(low), float(high)]


def summarize_frames(values: Sequence[float | None], *, planned_n: int,
                     fps: float = FPS) -> dict[str, Any]:
    """Quantiles, early/late counts with Wilson intervals, and a 10-frame histogram.

    Missing values (``None``) count against the planned denominator as non-events.  Quantiles
    use numpy's default linear interpolation.
    """
    present = np.asarray([float(v) for v in values if v is not None], dtype=float)
    if planned_n < 1 or len(present) > planned_n:
        raise ValueError("planned denominator must be positive and at least the present count")
    n = int(present.size)
    out: dict[str, Any] = {"planned_n": int(planned_n), "n_present": n,
                           "n_missing": int(planned_n - n)}
    if n:
        quantiles = {
            "median": float(np.median(present)),
            "q10": float(np.quantile(present, 0.10)),
            "q90": float(np.quantile(present, 0.90)),
            "min": float(present.min()),
            "max": float(present.max()),
            "mean": float(present.mean()),
        }
        out["frames"] = quantiles
        out["seconds"] = {key: value / fps for key, value in quantiles.items()}
        out["values_frames"] = [float(v) for v in present]
    else:
        out["frames"] = None
        out["seconds"] = None
        out["values_frames"] = []
    first = int((present < FIRST_WINDOW_FRAMES).sum())
    late = int((present > LATE_FRAME).sum())
    for name, count in (("inside_first_50_frames", first), ("after_frame_60", late)):
        out[name] = {
            "count": count,
            "planned_denominator": int(planned_n),
            "fraction_of_planned": count / planned_n,
            "wilson95_of_planned": _wilson(count, planned_n),
            "present_denominator": n,
            "fraction_of_present": (count / n) if n else None,
            "wilson95_of_present": _wilson(count, n),
        }
    out["inside_first_50_frames"]["window_s"] = FIRST_WINDOW_FRAMES / fps
    out["after_frame_60"]["threshold_s"] = LATE_FRAME / fps
    edges = np.arange(0, N_FRAMES + HIST_BIN_FRAMES, HIST_BIN_FRAMES, dtype=float)
    counts, _ = np.histogram(present, bins=edges) if n else (np.zeros(len(edges) - 1, int), None)
    out["histogram_10_frame_bins"] = {
        "edges_frames": [int(e) for e in edges],
        "counts": [int(c) for c in counts],
        "bin_width_s": HIST_BIN_FRAMES / fps,
    }
    return out


def pairwise_agreement(first: Sequence[float | None], second: Sequence[float | None],
                       *, fps: float = FPS) -> dict[str, Any]:
    """Descriptive paired differences ``first - second`` over clips where both exist."""
    if len(first) != len(second):
        raise ValueError("paired sequences must align")
    diffs = np.asarray([float(a) - float(b) for a, b in zip(first, second)
                        if a is not None and b is not None], dtype=float)
    out: dict[str, Any] = {"n_pairs": int(diffs.size)}
    if diffs.size:
        stats = {
            "median": float(np.median(diffs)),
            "mean": float(diffs.mean()),
            "q10": float(np.quantile(diffs, 0.10)),
            "q90": float(np.quantile(diffs, 0.90)),
            "min": float(diffs.min()),
            "max": float(diffs.max()),
            "max_abs": float(np.abs(diffs).max()),
        }
        out["diff_frames"] = stats
        out["diff_s"] = {key: value / fps for key, value in stats.items()}
        out["n_abs_diff_le_2_frames"] = int((np.abs(diffs) <= 2).sum())
        out["n_abs_diff_le_5_frames_0p2s"] = int((np.abs(diffs) <= 5).sum())
        out["n_abs_diff_le_10_frames_0p4s"] = int((np.abs(diffs) <= 10).sum())
    return out


# --------------------------------------------------------------------------------------
# Archive validation and identity.
# --------------------------------------------------------------------------------------

def _load_thresholds() -> dict[str, float]:
    """Load the calibrated support envelope exactly as analyze_trackability_contract does."""
    path = Path(THRESHOLD_RECEIPT)
    if not path.is_file():
        raise ValueError(f"missing threshold receipt: {path}")
    actual = exact.file_sha256(path)
    if actual != THRESHOLD_RECEIPT_SHA256:
        raise ValueError("threshold receipt hash mismatch")
    thresholds = json.loads(path.read_text())["stepover_thresholds"]
    return {
        "path": str(path),
        "sha256": actual,
        "support_height_m": float(thresholds["support_height_m"]),
        "support_speed_mps": float(thresholds["support_speed_mps"]),
        "max_unsupported_run_s": float(thresholds["max_unsupported_run_s"]),
    }


def _locked_source_identity() -> dict[str, Any]:
    import mujoco
    from scene2motion.robot import ARDY_G1_XML, BODY_MARGIN

    source_hashes: dict[str, str] = {}
    for relative, expected in LOCKED_SOURCE_FILES.items():
        actual = exact.file_sha256(REPO / relative)
        if actual != expected:
            raise ValueError(
                f"locked source hash mismatch for {relative}: "
                f"expected={expected}, actual={actual}"
            )
        source_hashes[relative] = actual
    function_hashes: dict[str, str] = {}
    locked_functions = {
        "analyze_trackability_contract.support_masks": support_masks,
        "analyze_trackability_contract.runs_of": runs_of,
    }
    for name, expected in LOCKED_FUNCTION_SOURCES.items():
        actual = function_source_sha256(locked_functions[name])
        if actual != expected:
            raise ValueError(
                f"locked function source hash mismatch for {name}: "
                f"expected={expected}, actual={actual}"
            )
        function_hashes[name] = actual
    xml_sha256 = exact.file_sha256(Path(ARDY_G1_XML))
    if xml_sha256 != LOCKED_G1_XML_SHA256:
        raise ValueError("released G1 XML differs from the model locked for this analysis")
    if float(BODY_MARGIN) != BODY_MARGIN_M:
        raise ValueError("event-frame analysis requires the locked 4 cm G1 body margin")
    return {
        "injected": False,
        "locked_source_sha256": source_hashes,
        "locked_function_source_sha256": function_hashes,
        "recorded_source_sha256": {
            relative: exact.file_sha256(REPO / relative) for relative in RECORDED_SOURCE_FILES
        },
        "g1_xml_path": str(ARDY_G1_XML),
        "g1_xml_sha256": xml_sha256,
        "body_margin_m": float(BODY_MARGIN),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_validated_exp023(
    archive: Path, locked_manifest: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], dict[str, str]]:
    receipt_path = archive / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"missing EXP-023 receipt: {receipt_path}")
    locked_file_hashes: dict[str, str] = {}
    if locked_manifest is not None:
        for relative, expected in locked_manifest["files"].items():
            path = archive / str(relative)
            if not path.is_file():
                raise ValueError(f"locked EXP-023 artifact is missing: {path}")
            actual = exact.file_sha256(path)
            if actual != expected:
                raise ValueError(
                    f"locked EXP-023 artifact hash mismatch for {relative}: "
                    f"expected={expected}, actual={actual}"
                )
            locked_file_hashes[str(relative)] = actual
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "exp023-prompt-handoff-v1":
        raise ValueError("archive is not the EXP-023 prompt-handoff schema")
    if receipt.get("experiment") != "exp023_prompt_handoff":
        raise ValueError("archive is not the EXP-023 prompt-handoff experiment")
    if receipt.get("status") != "complete" or receipt.get("complete") is not True:
        raise ValueError("EXP-023 receipt is not complete")
    if receipt.get("blocked") is not False:
        raise ValueError("EXP-023 receipt is marked blocked or lacks blocked=false")
    qpos_anchor = receipt.get("evidence_anchors", {}).get("qpos", {})
    relative_qpos_path = qpos_anchor.get("path")
    if not isinstance(relative_qpos_path, str) or not relative_qpos_path:
        raise ValueError("EXP-023 receipt lacks evidence_anchors.qpos.path")
    qpos_path = archive / relative_qpos_path
    if not qpos_path.is_file():
        raise ValueError(f"missing EXP-023 qpos archive: {qpos_path}")
    with np.load(qpos_path, allow_pickle=False) as payload:
        arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    if not arrays:
        raise ValueError("EXP-023 qpos archive is empty")
    for key, value in arrays.items():
        if value.ndim != 2 or value.shape[0] < N_FRAMES or not np.isfinite(value).all():
            raise ValueError(f"EXP-023 qpos array {key} must be a finite 2-D array of >= 200 frames")
    actual_hash = exact.array_content_sha256(arrays)
    if actual_hash != qpos_anchor.get("content_sha256"):
        raise ValueError(
            f"EXP-023 qpos content hash mismatch: receipt={qpos_anchor.get('content_sha256')}, "
            f"actual={actual_hash}")
    if qpos_anchor.get("n_arrays") != len(arrays):
        raise ValueError("EXP-023 receipt qpos count does not match the archive")
    rows_path = archive / "rows.jsonl"
    if not rows_path.is_file():
        raise ValueError(f"missing EXP-023 rows: {rows_path}")
    rows = _read_jsonl(rows_path)
    if len(rows) != len(arrays):
        raise ValueError("EXP-023 rows and qpos arrays disagree in count")
    keys = [row.get("archive_key") for row in rows]
    if set(keys) != set(arrays) or len(set(keys)) != len(keys):
        raise ValueError("EXP-023 row archive keys do not match the qpos archive")
    for row in rows:
        arm = row.get("arm")
        if arm not in EXP023_ARMS:
            raise ValueError(f"EXP-023 row {row.get('archive_key')} has unknown arm {arm!r}")
        if not isinstance(row.get("seed"), int) or isinstance(row.get("seed"), bool):
            raise ValueError("EXP-023 row lacks an integer seed")
        if row.get("onset_frame") != EXP023_ONSETS[arm]:
            raise ValueError(f"EXP-023 row {row.get('archive_key')} onset disagrees with its arm")
        if row.get("scored_frames") != N_FRAMES:
            raise ValueError("EXP-023 row was not scored on 200 frames")
        if row.get("archive_key") != f"s{row['seed']}_{arm}":
            raise ValueError("EXP-023 archive key does not follow s<seed>_<arm>")
        event = row.get("event")
        if arm == "all_walk":
            if event is not None:
                raise ValueError("EXP-023 all_walk rows carry no prompt event")
        elif not isinstance(event, dict) or not isinstance(event.get("present"), bool):
            raise ValueError("EXP-023 step rows require an event record with boolean present")
    if locked_manifest is not None:
        if receipt.get("sample_count_exact") is not True:
            raise ValueError("EXP-023 receipt does not assert an exact sample count")
        expected_n = len(locked_manifest["seeds"]) * len(locked_manifest["arms"])
        if receipt.get("actual_ardy_samples") != expected_n or len(arrays) != expected_n:
            raise ValueError(f"EXP-023 sample count is not {expected_n}")
        expected_keys = {
            f"s{seed}_{arm}"
            for seed in locked_manifest["seeds"] for arm in locked_manifest["arms"]
        }
        if set(arrays) != expected_keys:
            raise ValueError("EXP-023 qpos keys differ from the locked seed x arm plan")
        for key, value in arrays.items():
            if value.shape != (locked_manifest["archived_frames"], 36):
                raise ValueError(f"locked EXP-023 qpos {key} has shape {value.shape}")
        if actual_hash != locked_manifest["qpos_content_sha256"]:
            raise ValueError("EXP-023 qpos content differs from the locked archive manifest")
        code = receipt.get("provenance", {}).get("code", {})
        if code.get("commit") != locked_manifest["generation_commit"]:
            raise ValueError("EXP-023 generation commit differs from the locked manifest")
        analysis_commit = (
            receipt.get("provenance", {})
            .get("completion_identity_revalidation", {})
            .get("git", {})
            .get("analysis_commit")
        )
        if analysis_commit != locked_manifest["analysis_commit"]:
            raise ValueError("EXP-023 analysis commit differs from the locked manifest")
    return receipt, rows, arrays, locked_file_hashes


def _exp021_agreement(archived: Mapping[str, Any], recomputed: Mapping[str, Any],
                      side: str | None) -> dict[str, Any]:
    """Field-by-field agreement between the archived EXP-021 row and the recomputed scan."""
    def close(a: Any, b: Any) -> bool:
        if a is None or b is None:
            return a is None and b is None
        return bool(np.isclose(float(a), float(b), atol=LIFT_FIELD_ATOL, rtol=0.0))

    fields = {
        "lift_x_m": close(archived.get("lift_x_m"), recomputed["lift_x_m"]),
        "lift_height_m": close(archived.get("lift_height_m"), recomputed["lift_height_m"]),
        "lift_support_m": close(archived.get("lift_support_m"), recomputed["lift_support_m"]),
        "n_lift_regions": archived.get("n_lift_regions") == recomputed["n_lift_regions"],
        "lift_side": archived.get("lift_side") == side,
    }
    return {"agrees": all(fields.values()), "fields": fields}


# --------------------------------------------------------------------------------------
# Per-clip analysis.
# --------------------------------------------------------------------------------------

def analyze_clip(
    body: Any,
    qpos: np.ndarray,
    route: np.ndarray,
    *,
    support_height_m: float,
    support_speed_mps: float,
    profile_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    support_fn: Callable[..., tuple[Any, np.ndarray, np.ndarray]],
    lift_side_fn: Callable[..., str | None],
    fps: float = FPS,
) -> dict[str, Any]:
    """Recompute the EXP-021 lift and derive definitions A, B and C for one clip."""
    q = np.asarray(qpos, dtype=float)
    if q.ndim != 2 or q.shape[0] != N_FRAMES or not np.isfinite(q).all():
        raise ValueError("clip must be a finite (200, nq) qpos array")
    started = time.monotonic()
    xs, heights = profile_fn(q, route, OBSTACLE_DEPTH_M, n_points=PROFILE_POINTS)
    xs = np.asarray(xs, dtype=float)
    heights = np.asarray(heights, dtype=float)
    profile_s = time.monotonic() - started
    lift = lift_location(xs, heights)
    feet, sup_left, sup_right = support_fn(body, q, support_height_m, support_speed_mps)
    sup_left = np.asarray(sup_left, dtype=bool)
    sup_right = np.asarray(sup_right, dtype=bool)
    if sup_left.shape != (N_FRAMES,) or sup_right.shape != (N_FRAMES,):
        raise ValueError("support masks must cover exactly the 200 scored frames")
    flight = (~sup_left) & (~sup_right)
    runs = runs_of(flight)
    gate_frames = int(math.ceil(GATE_RUN_S * fps))
    longest = max(runs, key=lambda r: r[1] - r[0]) if runs else None
    first_gate = next((r for r in runs if r[1] - r[0] >= gate_frames), None)

    record: dict[str, Any] = {
        "profile": {
            **lift,
            "scan_lo_x_m": float(xs[0]),
            "scan_hi_x_m": float(xs[-1]),
            "scan_points": int(len(xs)),
            "scan_step_m": float(xs[1] - xs[0]) if len(xs) > 1 else None,
            "profile_wall_clock_s": float(profile_s),
        },
        "lifting": lift["lift_x_m"] is not None,
        "lift_ge_3cm": bool(lift["lift_height_m"] >= LIFT_MIN_M),
        "root_x_first_frame_m": float(q[0, 0]),
        "root_x_last_frame_m": float(q[-1, 0]),
        "max_clearance_any_foot": max_clearance_frame(feet),
        "no_support": {
            "n_runs": len(runs),
            "max_unsupported_run_s": float((longest[1] - longest[0]) / fps) if longest else 0.0,
            "first_gate_length_run_onset_frame": int(first_gate[0]) if first_gate else None,
            "first_gate_length_run_onset_s": (
                float(first_gate[0] / fps) if first_gate else None),
        },
        "events": None,
    }
    if lift["lift_x_m"] is None:
        return record

    lift_x = float(lift["lift_x_m"])
    side = lift_side_fn(q, feet, lift_x, OBSTACLE_DEPTH_M)
    record["profile"]["lift_side"] = side
    region = lift_region_extent(xs, heights, lift_x)
    a_frame = root_crossing_frame(q[:, 0], lift_x)
    b_frame = nominal_frame(lift_x, fps=fps)
    c_hit = foot_lift_frame(feet[side], region["lo_x_m"], region["hi_x_m"]) if side else None
    c0_hit = foot_lift_frame(feet[side], lift_x, lift_x) if side else None
    run = containing_run(runs, c_hit["frame"]) if c_hit else None
    c_frame = c_hit["frame"] if c_hit else None
    events: dict[str, Any] = {
        "A_root_crossing_frame": a_frame,
        "A_s": (a_frame / fps) if a_frame is not None else None,
        "B_nominal_frame": b_frame,
        "B_s": b_frame / fps,
        "B_round_frame": int(np.rint(b_frame)),
        "B_floor_frame": int(math.floor(b_frame)),
        "B_ceil_frame": int(math.ceil(b_frame)),
        "A_minus_B_frames": (a_frame - b_frame) if a_frame is not None else None,
        "A_minus_B_s": ((a_frame - b_frame) / fps) if a_frame is not None else None,
        "C_foot_lift_frame": c_frame,
        "C_s": (c_frame / fps) if c_frame is not None else None,
        "C_side": side,
        "C_foot_clearance_m": c_hit["clearance_m"] if c_hit else None,
        "C_region": {
            **region,
            "half_slab_m": HALF_SLAB_M,
            "first_frame_inside": c_hit["first_frame_inside"] if c_hit else None,
            "last_frame_inside": c_hit["last_frame_inside"] if c_hit else None,
            "n_frames_inside": c_hit["n_frames_inside"] if c_hit else None,
        },
        "C0_window_frame": c0_hit["frame"] if c0_hit else None,
        "C0_foot_clearance_m": c0_hit["clearance_m"] if c0_hit else None,
        "C_minus_C0_frames": (
            (c_frame - c0_hit["frame"]) if (c_frame is not None and c0_hit) else None),
        "A_minus_C_frames": (
            (a_frame - c_frame) if (a_frame is not None and c_frame is not None) else None),
        "B_minus_C_frames": (b_frame - c_frame) if c_frame is not None else None,
        "C_inside_nosupport_run": run is not None,
        "C_nosupport_run": None if run is None else {
            "start_frame": run[0],
            "end_frame_exclusive": run[1],
            "onset_s": run[0] / fps,
            "duration_s": (run[1] - run[0]) / fps,
            "at_least_gate_length": (run[1] - run[0]) >= gate_frames,
        },
    }
    record["events"] = events
    return record


# --------------------------------------------------------------------------------------
# Summaries.
# --------------------------------------------------------------------------------------

def _event_values(rows: Sequence[Mapping[str, Any]], name: str) -> list[float | None]:
    values: list[float | None] = []
    for row in rows:
        events = row.get("events")
        if not events:
            values.append(None)
            continue
        if name == "C_nosupport_run_onset_frame":
            run = events.get("C_nosupport_run")
            values.append(None if run is None else run["start_frame"])
        else:
            values.append(events.get(name))
    return values


DEFINITION_FIELDS = (
    ("A_root_crossing", "A_root_crossing_frame"),
    ("B_nominal_speed_real", "B_nominal_frame"),
    ("B_nominal_speed_rounded", "B_round_frame"),
    ("B_nominal_speed_floor", "B_floor_frame"),
    ("B_nominal_speed_ceil", "B_ceil_frame"),
    ("C_foot_lift", "C_foot_lift_frame"),
    ("C0_window_foot_lift", "C0_window_frame"),
    ("C_nosupport_run_onset", "C_nosupport_run_onset_frame"),
)


def _definition_block(rows: Sequence[Mapping[str, Any]], planned_n: int) -> dict[str, Any]:
    block: dict[str, Any] = {"planned_denominator": int(planned_n), "n_rows": len(rows)}
    for label, field in DEFINITION_FIELDS:
        block[label] = summarize_frames(_event_values(rows, field), planned_n=planned_n)
    a = _event_values(rows, "A_root_crossing_frame")
    b = _event_values(rows, "B_nominal_frame")
    b_round = _event_values(rows, "B_round_frame")
    c = _event_values(rows, "C_foot_lift_frame")
    c0 = _event_values(rows, "C0_window_frame")
    onset = _event_values(rows, "C_nosupport_run_onset_frame")
    block["pairwise_agreement"] = {
        "A_minus_B_real": pairwise_agreement(a, b),
        "A_minus_B_rounded": pairwise_agreement(a, b_round),
        "A_minus_C": pairwise_agreement(a, c),
        "B_real_minus_C": pairwise_agreement(b, c),
        "C_minus_C0": pairwise_agreement(c, c0),
        "C_minus_nosupport_run_onset": pairwise_agreement(c, onset),
    }
    inside = [row["events"]["C_inside_nosupport_run"] for row in rows if row.get("events")]
    block["C_inside_bilateral_nosupport_run"] = {
        "count": int(sum(inside)), "of": len(inside),
        "wilson95": _wilson(int(sum(inside)), len(inside)),
        "n_run_at_least_gate_length": int(sum(
            1 for row in rows if row.get("events") and row["events"]["C_nosupport_run"]
            and row["events"]["C_nosupport_run"]["at_least_gate_length"])),
    }
    sides = [row["events"]["C_side"] for row in rows if row.get("events")]
    block["lift_side_counts"] = {
        "left": sides.count("left"), "right": sides.count("right"),
        "undetermined": sides.count(None),
    }
    return block


def _exp021_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lifting = [row for row in rows if row["lifting"]]
    non_lifting = [row for row in rows if not row["lifting"]]
    ge3 = [row for row in lifting if row["lift_ge_3cm"]]
    lifting_block = _definition_block(lifting, planned_n=len(lifting))
    root = lifting_block["A_root_crossing"]
    nominal = lifting_block["B_nominal_speed_real"]
    return {
        "planned_clips": len(rows),
        "lifting_clips_any_positive_profile": len(lifting),
        "lift_ge_3cm_clips": len(ge3),
        "non_lifting_clips": len(non_lifting),
        "lifting_all": lifting_block,
        "lift_ge_3cm": _definition_block(ge3, planned_n=len(ge3)) if ge3 else None,
        "non_lifting_max_clearance_frame": {
            "label": (
                "NOT an event: frame of the highest foot bottom (either foot) in the 15 clips "
                "whose profile is zero everywhere; reported for comparison only"),
            **summarize_frames(
                [row["max_clearance_any_foot"]["frame"] for row in non_lifting],
                planned_n=max(len(non_lifting), 1)),
            "clearance_m_sorted": sorted(
                round(row["max_clearance_any_foot"]["clearance_m"], 4) for row in non_lifting),
        },
        "archived_row_agreement": {
            "n_agreeing": int(sum(row["archived_agreement"]["agrees"] for row in rows)),
            "of": len(rows),
        },
        "provisional_numbers_in_plan": {
            "quoted": dict(PLAN_PROVISIONAL_NUMBERS),
            "computed_A_root_crossing": {
                "median_frame": root["frames"]["median"] if root["frames"] else None,
                "q10_q90_frames": (
                    [root["frames"]["q10"], root["frames"]["q90"]] if root["frames"] else None),
                "inside_first_50_frames": (
                    f"{root['inside_first_50_frames']['count']}/{root['planned_n']}"),
                "after_frame_60": f"{root['after_frame_60']['count']}/{root['planned_n']}",
            },
            "computed_B_nominal_real": {
                "median_frame": nominal["frames"]["median"] if nominal["frames"] else None,
                "q10_q90_frames": (
                    [nominal["frames"]["q10"], nominal["frames"]["q90"]]
                    if nominal["frames"] else None),
                "inside_first_50_frames": (
                    f"{nominal['inside_first_50_frames']['count']}/{nominal['planned_n']}"),
                "after_frame_60": (
                    f"{nominal['after_frame_60']['count']}/{nominal['planned_n']}"),
            },
        },
    }


def _exp023_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in EXP023_ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        planned = len(arm_rows)
        lifting = [row for row in arm_rows if row["lifting"]]
        onset = EXP023_ONSETS[arm]
        block: dict[str, Any] = {
            "planned_clips": planned,
            "onset_frame": onset,
            "lifting_clips_full_clip_profile": len(lifting),
            "lift_ge_3cm_clips_full_clip_profile": sum(row["lift_ge_3cm"] for row in arm_rows),
            "full_clip_definitions": (
                _definition_block(lifting, planned_n=planned) if planned else None),
        }
        if onset is not None:
            c_before = [
                row for row in lifting
                if row["events"]["C_foot_lift_frame"] is not None
                and row["events"]["C_foot_lift_frame"] < onset
            ]
            block["full_clip_lift_before_prompt_onset"] = {
                "count": len(c_before), "of_lifting": len(lifting),
                "keys": [row["key"] for row in c_before],
                "note": "a lift whose C frame precedes the prompt onset happened under WALK",
            }
            present = [row for row in arm_rows if row["archived_event"]["present"]]
            archived_frames = [row["archived_event"]["frame"] for row in present]
            comparisons = [row["archived_event_comparison"] for row in present]
            block["archived_event"] = {
                "planned": planned,
                "present": len(present),
                "wilson95": _wilson(len(present), planned) if planned else None,
                "archived_frame_distribution": summarize_frames(
                    archived_frames, planned_n=max(planned, 1)),
                "archived_latency_frames": [
                    row["archived_event"]["latency_frames"] for row in present],
                "comparison_against_full_clip_definitions": {
                    "per_clip": comparisons,
                    "A_minus_archived": pairwise_agreement(
                        [c["A_root_crossing_frame"] for c in comparisons],
                        [c["archived_frame"] for c in comparisons]),
                    "C_minus_archived": pairwise_agreement(
                        [c["C_foot_lift_frame"] for c in comparisons],
                        [c["archived_frame"] for c in comparisons]),
                    "B_real_minus_archived": pairwise_agreement(
                        [c["B_nominal_frame"] for c in comparisons],
                        [c["archived_frame"] for c in comparisons]),
                    "n_side_agrees": int(sum(
                        1 for c in comparisons if c["side_agrees"] is True)),
                    "n_same_full_clip_lift_region": int(sum(
                        1 for c in comparisons if c["archived_profile_x_inside_C_region"] is True)),
                },
            }
        else:
            control = [row["archived_control_events"] for row in arm_rows]
            block["archived_control_events"] = {
                "n_seeds_with_any_window_event": int(sum(
                    1 for c in control if any(v["present"] for v in c.values()))),
                "of": planned,
            }
        by_arm[arm] = block
    return {"planned_clips": len(rows), "by_arm": by_arm}


# --------------------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------------------

def _definitions(thresholds: Mapping[str, float]) -> dict[str, Any]:
    return {
        "fps": FPS,
        "frame_to_seconds": "t = frame / 25; frame 0 is t = 0 (first generated frame)",
        "forward_axis": (
            "MuJoCo world x = qpos[:, 0]; the route is prescribed along route_xz[:, 1] "
            "(ARDY z) and BoxHeightProbe places the box at world x"),
        "lift_x_m": (
            "EXP-021 definition, recomputed byte-identically: argmax of "
            "box_height_profile(qpos, route, depth=0.20, n_points=120) over the scan "
            "[route start + 0.3 m, route end - 0.3 m] = [0.3, 6.9] m (step 0.0555 m); a clip "
            "'lifts' when any profile point is positive (>= the 5 mm probe resolution)"),
        "A_root_crossing_frame": (
            "first frame at which qpos[:, 0] >= lift_x_m (None if the root never reaches it)"),
        "B_nominal_frame": (
            f"lift_x_m / {REFERENCE_SPEED_MPS:.6f} m/s * 25 fps, the real-valued frame at which "
            "the prescribed route reaches lift_x_m; rounded = numpy rint (half to even), "
            "floor and ceil also reported"),
        "C_foot_lift_frame": (
            "frame of maximum bottom clearance of the lift_side foot among frames whose "
            "forward representative lies in [region_lo - 0.14 m, region_hi + 0.14 m], where "
            "[region_lo, region_hi] is the contiguous positive run of the profile containing "
            "lift_x_m and 0.14 m = depth/2 + BODY_MARGIN is lift_side's own window; ties -> "
            "earliest frame"),
        "C0_window_frame": (
            "same maximum restricted to |forward representative - lift_x_m| <= 0.14 m "
            "(exactly the lift_side window); sensitivity check on the region choice"),
        "C_nosupport_run": (
            "the bilateral no-support run (neither foot inside the calibrated support "
            f"envelope: bottom <= {thresholds['support_height_m']:.4f} m and planar speed <= "
            f"{thresholds['support_speed_mps']:.3f} m/s; analyze_trackability_contract."
            "support_masks) that contains the C frame; onset = its first frame"),
        "inside_first_50_frames": "frame < 50 (t < 2.0 s); real-valued B compared as B < 50",
        "after_frame_60": "frame > 60 (t > 2.4 s); real-valued B compared as B > 60",
        "quantiles": "numpy linear interpolation; median = q50",
        "intervals": "Wilson 95 % on every count, over the planned and the present denominators",
        "planned_denominators": {
            "exp021": "49 lifting clips of 64 (lift_x_m not null); the 44 with lift >= 3 cm "
                      "are a secondary stratum; the 15 non-lifting clips are reported "
                      "separately and are not events",
            "exp023": "8 seeds per arm; step_0 archived events 6/8; A/B/C on every clip whose "
                      "full-clip profile lifts",
        },
        "exp023_archived_event_vs_this_analysis": (
            "EXP-023's event uses a 96-frame post-onset window, rescans that window's route "
            "segment at 120 points (a different grid), requires both physical feet to cross the "
            "slab, selects the tallest traversed profile point (earliest on ties) and takes the "
            "crossing frame with the highest foot bottom; this analysis scans the whole 200-frame "
            "clip on EXP-021's grid, needs no traversal, and takes the peak-clearance frame "
            "inside the region.  They are compared, not equated."),
        "non_lifting_clips": (
            "frame of the highest foot bottom over the whole clip (either foot); labelled as "
            "not an event"),
        "stage": "post hoc CPU analysis of completed archives; no generator; no tracker",
    }


def run_analysis(
    out: str | Path,
    *,
    exp021_archive: str | Path = EXP021_ARCHIVE,
    exp023_archive: str | Path = EXP023_ARCHIVE,
    locked: bool = True,
    body: Any = None,
    profile_fn: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    support_fn: Callable[..., tuple[Any, np.ndarray, np.ndarray]] | None = None,
    lift_side_fn: Callable[..., str | None] | None = None,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
) -> dict[str, Any]:
    """Validate both archives, derive every clip's event frames, and write the bundle."""
    output = Path(out)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    thresholds = _load_thresholds()
    injected = any(fn is not None for fn in (profile_fn, support_fn, lift_side_fn))
    if injected:
        identity: dict[str, Any] = {
            "injected": True,
            "evidentiary_scope": "test/diagnostic only; not the locked scan or foot envelopes",
        }
        body = body if body is not None else object()
    else:
        identity = _locked_source_identity()
        if body is None:
            from scene2motion.robot import G1Body
            body = G1Body(None)
    profile = profile_fn or box_height_profile
    support = support_fn or support_masks
    side_fn = lift_side_fn or lift_side
    route = cal.route_xz_for_speed(REFERENCE_SPEED_MPS)
    if route.shape != (N_FRAMES, 2) or not np.allclose(route[:, 0], 0.0):
        raise ValueError("route must be the 200-frame straight reference-speed route")
    if not np.isclose(route[-1, 1], cal.PILOT_ROUTE_LENGTH_M):
        raise ValueError("route length differs from the 7.2 m pilot route")
    route_speed = float(route[-1, 1] - route[0, 1]) / ((N_FRAMES - 1) / FPS)
    if not np.isclose(route_speed, REFERENCE_SPEED_MPS):
        raise ValueError("route speed differs from the nominal conversion speed")

    exp021_path = Path(exp021_archive)
    exp023_path = Path(exp023_archive)
    receipt021, seeds021, arrays021, hashes021 = exact._load_validated_archive(
        exp021_path, LOCKED_EXP021_MANIFEST if locked else None)
    rows021_by_seed = {row["seed"]: row for row in _read_jsonl(exp021_path / "rows.jsonl")}
    if set(rows021_by_seed) != set(seeds021):
        raise ValueError("EXP-021 rows do not cover the receipt's pool seeds")
    receipt023, rows023, arrays023, hashes023 = _load_validated_exp023(
        exp023_path, LOCKED_EXP023_MANIFEST if locked else None)

    code = dict(code_state_fn(REPO))
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "analysis": "event_frames",
        "status": "running",
        "complete": False,
        "blocked": False,
        "generated_at_unix": time.time(),
        "code": code,
        "script_sha256": exact.file_sha256(Path(__file__)),
        "identity": identity,
        "definitions": _definitions(thresholds),
        "inputs": {
            "threshold_receipt": thresholds,
            "exp021": {
                "archive": str(exp021_path),
                "receipt_status": receipt021["status"],
                "receipt_schema": receipt021["schema"],
                "source_commit": receipt021.get("provenance", {}).get("code", {}).get("commit"),
                "locked_file_sha256": hashes021,
                "qpos_content_sha256": receipt021["evidence_anchors"]["qpos"]["content_sha256"],
                "n_clips": len(arrays021),
                "pool_seeds": list(seeds021),
                "qpos_dtype_archived": str(next(iter(arrays021.values())).dtype),
                "analysed_as": "float64",
                "locked": locked,
            },
            "exp023": {
                "archive": str(exp023_path),
                "receipt_status": receipt023["status"],
                "receipt_schema": receipt023["schema"],
                "generation_commit": receipt023.get("provenance", {}).get("code", {}).get("commit"),
                "locked_file_sha256": hashes023,
                "qpos_content_sha256": receipt023["evidence_anchors"]["qpos"]["content_sha256"],
                "n_clips": len(arrays023),
                "archived_frames": int(next(iter(arrays023.values())).shape[0]),
                "scored_frames": N_FRAMES,
                "qpos_dtype_archived": str(next(iter(arrays023.values())).dtype),
                "analysed_as": "float64",
                "locked": locked,
            },
            "route": {
                "length_m": float(route[-1, 1]),
                "frames": N_FRAMES,
                "reference_speed_mps": REFERENCE_SPEED_MPS,
                "obstacle_depth_m": OBSTACLE_DEPTH_M,
                "profile_points": PROFILE_POINTS,
                "half_slab_m": HALF_SLAB_M,
            },
        },
    }

    def persist() -> None:
        receipt["wall_clock_s"] = float(time.monotonic() - started)
        (output / "receipt.json").write_text(
            json.dumps(receipt, indent=1, sort_keys=True, allow_nan=False) + "\n")

    persist()
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    common = dict(
        support_height_m=thresholds["support_height_m"],
        support_speed_mps=thresholds["support_speed_mps"],
        profile_fn=profile, support_fn=support, lift_side_fn=side_fn,
    )
    with (output / "rows.jsonl").open("w") as handle:
        for seed in seeds021:
            key = f"s{seed}"
            archived = rows021_by_seed[seed]
            clip = analyze_clip(body, arrays021[key][:N_FRAMES], route, **common)
            agreement = _exp021_agreement(archived, clip["profile"],
                                          clip["profile"].get("lift_side"))
            row = {
                "family": "exp021",
                "key": key,
                "seed": int(seed),
                "arm": "step_0",
                "prompt": archived.get("prompt"),
                "onset_frame": 0,
                "archived_frames": int(arrays021[key].shape[0]),
                "scored_frames": N_FRAMES,
                "archived_lift": {
                    name: archived.get(name)
                    for name in ("lift_x_m", "lift_height_m", "lift_side",
                                 "lift_support_m", "n_lift_regions", "clears_height",
                                 "progress_ratio")
                },
                "archived_agreement": agreement,
                **clip,
            }
            if not agreement["agrees"]:
                mismatches.append({"key": key, "fields": agreement["fields"],
                                   "archived": row["archived_lift"],
                                   "recomputed": clip["profile"]})
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
        for archived in rows023:
            key = archived["archive_key"]
            arm = archived["arm"]
            clip = analyze_clip(body, arrays023[key][:N_FRAMES], route, **common)
            row = {
                "family": "exp023",
                "key": key,
                "seed": int(archived["seed"]),
                "arm": arm,
                "prompt_schedule": archived.get("prompt_schedule"),
                "onset_frame": EXP023_ONSETS[arm],
                "archived_frames": int(arrays023[key].shape[0]),
                "scored_frames": N_FRAMES,
                "archived_supporting_motion": archived.get("supporting_motion"),
                **clip,
            }
            if arm == "all_walk":
                row["archived_event"] = None
                row["archived_control_events"] = {
                    str(onset): {"present": bool(value.get("present")),
                                 "max_profile_height_m": value.get("max_profile_height_m")}
                    for onset, value in (archived.get("control_events") or {}).items()
                }
            else:
                event = archived["event"]
                row["archived_event"] = {
                    name: event.get(name)
                    for name in ("present", "frame", "latency_frames", "latency_s", "side",
                                 "profile_x_m", "foot_x_m", "whole_body_clearance_m",
                                 "missing_reason", "max_profile_height_m",
                                 "analysis_window_start_frame", "analysis_window_end_frame")
                }
                row["archived_event_comparison"] = None
                if event.get("present"):
                    events = clip["events"] or {}
                    region = events.get("C_region") or {}
                    px = event.get("profile_x_m")
                    inside_region = (
                        None if (px is None or not region)
                        else bool(region["lo_x_m"] - HALF_SLAB_M <= float(px)
                                  <= region["hi_x_m"] + HALF_SLAB_M))
                    row["archived_event_comparison"] = {
                        "key": key,
                        "archived_frame": int(event["frame"]),
                        "archived_latency_frames": int(event["latency_frames"]),
                        "archived_side": event.get("side"),
                        "archived_profile_x_m": px,
                        "archived_foot_x_m": event.get("foot_x_m"),
                        "full_clip_lift_x_m": clip["profile"]["lift_x_m"],
                        "full_clip_lift_height_m": clip["profile"]["lift_height_m"],
                        "A_root_crossing_frame": events.get("A_root_crossing_frame"),
                        "B_nominal_frame": events.get("B_nominal_frame"),
                        "C_foot_lift_frame": events.get("C_foot_lift_frame"),
                        "C_side": events.get("C_side"),
                        "A_minus_archived_frames": (
                            events["A_root_crossing_frame"] - int(event["frame"])
                            if events.get("A_root_crossing_frame") is not None else None),
                        "C_minus_archived_frames": (
                            events["C_foot_lift_frame"] - int(event["frame"])
                            if events.get("C_foot_lift_frame") is not None else None),
                        "B_real_minus_archived_frames": (
                            events["B_nominal_frame"] - int(event["frame"])
                            if events.get("B_nominal_frame") is not None else None),
                        "profile_x_minus_full_clip_lift_x_m": (
                            float(px) - clip["profile"]["lift_x_m"]
                            if (px is not None and clip["profile"]["lift_x_m"] is not None)
                            else None),
                        "side_agrees": (
                            None if events.get("C_side") is None
                            else events["C_side"] == event.get("side")),
                        "archived_profile_x_inside_C_region": inside_region,
                    }
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()

    receipt["n_rows"] = len(rows)
    if mismatches:
        receipt.update({
            "status": "blocked", "blocked": True,
            "error": "recomputed EXP-021 lift fields do not reproduce the archived rows",
            "mismatches": mismatches,
        })
        persist()
        raise ValueError(
            f"{len(mismatches)} EXP-021 clips do not reproduce their archived lift fields: "
            f"{[m['key'] for m in mismatches]}")

    rows021 = [row for row in rows if row["family"] == "exp021"]
    rows023_out = [row for row in rows if row["family"] == "exp023"]
    receipt["summary"] = {
        "exp021": _exp021_summary(rows021),
        "exp023": _exp023_summary(rows023_out),
        "profile_wall_clock_s_total": float(sum(
            row["profile"]["profile_wall_clock_s"] for row in rows)),
    }
    receipt.update({"status": "complete", "complete": True})
    persist()
    return receipt


def _print_summary(receipt: Mapping[str, Any]) -> None:
    summary = receipt["summary"]
    e21 = summary["exp021"]
    print("=" * 78)
    print(f"EXP-021 event frames: {e21['lifting_clips_any_positive_profile']} lifting of "
          f"{e21['planned_clips']} ({e21['lift_ge_3cm_clips']} >= 3 cm); archived rows "
          f"reproduced {e21['archived_row_agreement']['n_agreeing']}/"
          f"{e21['archived_row_agreement']['of']}")
    print("=" * 78)
    header = f"{'definition':>26} {'n':>3} {'med':>6} {'q10':>6} {'q90':>6} {'min':>5} {'max':>5} " \
             f"{'<50':>6} {'>60':>5}"
    print(header)
    for label, _ in DEFINITION_FIELDS:
        block = e21["lifting_all"][label]
        frames = block["frames"]
        if not frames:
            print(f"{label:>26} {block['n_present']:>3}  (no values)")
            continue
        print(f"{label:>26} {block['n_present']:>3} {frames['median']:>6.1f} {frames['q10']:>6.1f} "
              f"{frames['q90']:>6.1f} {frames['min']:>5.0f} {frames['max']:>5.0f} "
              f"{block['inside_first_50_frames']['count']:>3}/{block['planned_n']:<2} "
              f"{block['after_frame_60']['count']:>3}/{block['planned_n']:<2}")
    agreement = e21["lifting_all"]["pairwise_agreement"]
    for name, item in agreement.items():
        if item.get("diff_frames"):
            print(f"  {name:<28} n={item['n_pairs']:>2} median {item['diff_frames']['median']:+.1f} "
                  f"frames, range [{item['diff_frames']['min']:+.1f}, {item['diff_frames']['max']:+.1f}], "
                  f"|d|<=5: {item['n_abs_diff_le_5_frames_0p2s']}")
    e23 = summary["exp023"]["by_arm"]
    print()
    print("EXP-023 (full-clip profile): " + ", ".join(
        f"{arm} {block['lifting_clips_full_clip_profile']}/{block['planned_clips']} lifting"
        for arm, block in e23.items()))
    step0 = e23["step_0"]["archived_event"]
    print(f"  step_0 archived events {step0['present']}/{step0['planned']}, latencies "
          f"{step0['archived_latency_frames']}")
    comp = step0["comparison_against_full_clip_definitions"]
    for name in ("A_minus_archived", "C_minus_archived", "B_real_minus_archived"):
        item = comp[name]
        if item.get("diff_frames"):
            print(f"  {name:<24} n={item['n_pairs']} median {item['diff_frames']['median']:+.1f}, "
                  f"range [{item['diff_frames']['min']:+.1f}, {item['diff_frames']['max']:+.1f}]")
    print(f"  wall clock {receipt['wall_clock_s']:.0f} s")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--exp021", type=Path, default=EXP021_ARCHIVE)
    parser.add_argument("--exp023", type=Path, default=EXP023_ARCHIVE)
    args = parser.parse_args(argv)
    receipt = run_analysis(args.out, exp021_archive=args.exp021, exp023_archive=args.exp023)
    _print_summary(receipt)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
