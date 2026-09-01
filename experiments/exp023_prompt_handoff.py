"""EXP-023: paired WALK-to-STEP prompt handoff at Horizon52 boundaries.

The campaign uses ARDY's native ``autoregressive_step`` under the released interactive GUI's
default minimum-history policy: an immutable full accepted transcript is retained, while only
its final four-frame token is supplied to each continuation.  Four schedules for each fresh
seed share corresponding-window noise; delayed arms also share byte-identical WALK feature
histories with the all-WALK control up to their fork.  The scientific endpoint is the timing
and position of a whole-body-clearable event in equal post-onset windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import calibrate_ramp_route_phase as cal  # noqa: E402
from experiments import exp017_ramp_residual_stepover as e17  # noqa: E402
from scene2motion.constraints import ConstraintSpec  # noqa: E402
from scene2motion.robot import BODY_MARGIN, G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.stepover_eval import BoxHeightProbe, foot_kinematics_series  # noqa: E402


SCHEMA_VERSION = "exp023-prompt-handoff-v1"
FAILURE_SCHEMA_VERSION = "exp023-prompt-handoff-failure-v1"

WALK = "A person walks forward."
STEP = "A person steps over an obstacle."
FPS = 25.0
N_FRAMES = 200
HORIZON = 52
N_WINDOWS = 4
PADDED_FRAMES = HORIZON * N_WINDOWS
POST_ONSET_FRAMES = 96
FROZEN_LATENCY_FRAMES = 34
DIFFUSION_STEPS = 5
CFG_WEIGHT = (2.0, 2.0)
NOISE_STREAM_VERSION = 2
CHUNK_SEED_COUNT = 2
CHUNK_ROWS = CHUNK_SEED_COUNT * 4
EXPECTED_GLOBAL_HISTORY_START_FRAMES = (0, 48, 100, 152)
EXPECTED_ACCEPTED_TRANSCRIPT_FRAMES_BEFORE = (0, 52, 104, 156)
EXPECTED_INPUT_HISTORY_FRAMES = (0, 4, 4, 4)
EXPECTED_MODEL_NUM_FRAMES = (208, 160, 108, 56)
EXPECTED_TRANSCRIPT_FRAMES = (52, 104, 156, 208)

SEEDS = tuple(range(4500, 4508))
ARMS = ("all_walk", "step_0", "step_52", "step_104")
SCHEDULES: Mapping[str, tuple[str, ...]] = {
    "all_walk": (WALK, WALK, WALK, WALK),
    "step_0": (STEP, STEP, STEP, STEP),
    "step_52": (WALK, STEP, STEP, STEP),
    "step_104": (WALK, WALK, STEP, STEP),
}
ONSETS: Mapping[str, int | None] = {
    "all_walk": None,
    "step_0": 0,
    "step_52": 52,
    "step_104": 104,
}
CONTROL_ONSETS = (0, 52, 104)

OBSTACLE_DEPTH_M = 0.20
GRADED_HEIGHTS_M = (0.03, 0.05, 0.08, 0.12, 0.20, 0.30)
EVENT_MIN_CLEARANCE_M = 0.03
PROFILE_POINTS = 120
PROFILE_MARGIN_M = 0.30

MIN_STEP0_EVENTS = 4
MAX_ALL_WALK_SEEDS_WITH_ANY_EVENT = 1

PINNED_HF_REVISION = "059b8007df0ba194a006a877b59a563955ac7b70"
PINNED_DENOISER_SHA256 = (
    "0c16ac26c1ab75e511cd24bb25bd9ad92078a2460b4ce78529788b5da22647a2"
)
PINNED_ARDY_COMMIT = "693f74d13b3d04a0a22ce127ee79c929dd89756b"
PINNED_G1_XML_SHA256 = (
    "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
)

PROTOCOL_PATH = "docs/ramp-exp023-prompt-handoff-protocol.md"
SOURCE_FILES = (
    PROTOCOL_PATH,
    "experiments/exp023_prompt_handoff.py",
    "experiments/calibrate_ramp_route_phase.py",
    "experiments/exp017_ramp_residual_stepover.py",
    "scene2motion/runner.py",
    "scene2motion/constraints.py",
    "scene2motion/robot.py",
    "scene2motion/stepover_eval.py",
)


class PromptHandoffAbort(RuntimeError):
    """Fail-closed stop after durable campaign evidence has been written."""


class MeasurementRefusal(RuntimeError):
    """A preregistered substrate/specificity gate refused the timing estimate."""


def locked_row_plan() -> list[dict[str, Any]]:
    """Seed-major 32-row plan; its order binds feature and latent-audit rows."""
    pairs = [(seed, arm) for seed in SEEDS for arm in ARMS]
    return [
        {
            "row_index": index,
            "seed": int(seed),
            "arm": arm,
            "prompt_schedule": list(SCHEDULES[arm]),
            "onset_frame": ONSETS[arm],
        }
        for index, (seed, arm) in enumerate(pairs)
    ]


def locked_chunk_plan(plan: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Four seed-paired B=8 chunks; no same-seed causal comparison crosses a call."""
    rows = list(plan if plan is not None else locked_row_plan())
    chunks: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(SEEDS), CHUNK_SEED_COUNT)):
        seeds = tuple(SEEDS[start:start + CHUNK_SEED_COUNT])
        chunk_rows = [dict(row) for row in rows if int(row["seed"]) in seeds]
        if (
            len(seeds) != CHUNK_SEED_COUNT
            or len(chunk_rows) != CHUNK_ROWS
            or {str(row["arm"]) for row in chunk_rows} != set(ARMS)
            or {int(row["seed"]) for row in chunk_rows} != set(seeds)
        ):
            raise ValueError("locked EXP-023 chunk plan no longer forms paired B=8 calls")
        chunks.append({
            "chunk_index": index,
            "name": f"chunk{index:02d}_seeds{seeds[0]}_{seeds[-1]}",
            "seeds": list(seeds),
            "row_indices": [int(row["row_index"]) for row in chunk_rows],
            "rows": chunk_rows,
        })
    if len(chunks) != 4 or sorted(
        row_index for chunk in chunks for row_index in chunk["row_indices"]
    ) != list(range(len(rows))):
        raise ValueError("locked EXP-023 chunks do not partition the 32-row plan")
    return chunks


def route_xz() -> np.ndarray:
    """The exact 200-frame reference-speed route used by exp021."""
    return cal.route_xz_for_speed(cal.REFERENCE_SPEED_MPS)


def root_only_spec(route: np.ndarray) -> ConstraintSpec:
    return ConstraintSpec(
        root_xz=np.asarray(route, dtype=float),
        heading=None,
        root_y=None,
        first_heading=0.0,
    )


def _array_sha256(value: np.ndarray, name: str = "array") -> str:
    return cal._array_hash({name: np.asarray(value)})


def _raw_row_sha256(value: np.ndarray) -> str:
    """Match the runner's byte-level per-row tensor audit."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    """Normalize runner audit evidence without assuming it already uses Python scalars."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _source_hashes(repo: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        digest = cal._sha256(repo / relative)
        if digest is None:
            raise ValueError(f"required EXP-023 source is missing: {relative}")
        hashes[relative] = digest
    return hashes


def _prompt_cache_identity(runner: Any, cache_path: Path) -> dict[str, Any]:
    """Bind both exact cached prompt arrays in memory and on disk."""
    if not cache_path.is_file():
        raise ValueError("EXP-023 prompt cache is missing")
    prompts: dict[str, Any] = {}
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            for prompt in (WALK, STEP):
                key = hashlib.sha1(prompt.encode()).hexdigest()
                if key not in cache.files:
                    raise ValueError(f"cached embedding is missing for {prompt!r}")
                file_value = np.array(cache[key], copy=True)
                memory_value = np.array(runner._text_cache[key], copy=True)
                if (
                    file_value.size == 0
                    or not np.isfinite(file_value).all()
                    or not np.array_equal(file_value, memory_value)
                ):
                    raise ValueError(
                        f"in-memory embedding does not byte-match cache for {prompt!r}"
                    )
                prompts[prompt] = {
                    "cache_key_sha1": key,
                    "content_sha256": _array_sha256(file_value, key),
                    "shape": list(file_value.shape),
                    "dtype": str(file_value.dtype),
                }
    except (AttributeError, KeyError, OSError, ValueError) as exc:
        raise ValueError(f"invalid EXP-023 prompt cache: {exc}") from exc
    fields = {
        "path": str(cache_path),
        "file_sha256": cal._sha256(cache_path),
        "prompts": prompts,
    }
    return cal._identity("exp023-walk-step-prompt-cache-v1", fields)


def _validate_pins(
    generator: Mapping[str, Any],
    runtime: Mapping[str, Any],
    physical_model: Mapping[str, Any],
) -> None:
    checkpoint = generator.get("checkpoint", {})
    if checkpoint.get("hf_revision") != PINNED_HF_REVISION:
        raise ValueError("EXP-023 loaded the wrong ARDY checkpoint revision")
    if checkpoint.get("checkpoint_sha256") != PINNED_DENOISER_SHA256:
        raise ValueError("EXP-023 loaded the wrong ARDY denoiser bytes")
    if checkpoint.get("model_name") != "ARDY-G1-RP-25FPS-Horizon52":
        raise ValueError("EXP-023 requires ARDY-G1-RP-25FPS-Horizon52")
    runtime_fields = runtime.get("fields", {})
    if runtime_fields.get("ardy_git_commit") != PINNED_ARDY_COMMIT:
        raise ValueError("EXP-023 loaded the wrong ARDY runtime commit")
    if runtime_fields.get("ardy_tracked_status") != []:
        raise ValueError("EXP-023 requires a clean tracked ARDY checkout")
    if physical_model.get("fields", {}).get("sha256") != PINNED_G1_XML_SHA256:
        raise ValueError("EXP-023 loaded the wrong released G1 XML")


def _actual_channel_usage(runner: Any, spec: ConstraintSpec) -> dict[str, int]:
    return e17._actual_channel_usage(runner, spec)


def _validate_noise_and_feature_forks(
    features: np.ndarray,
    noise_audit: Sequence[Mapping[str, Any]],
    plan: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Hard causal-design gates before decoding or scientific scoring."""
    exact = np.asarray(features)
    if exact.ndim != 3 or exact.shape[:2] != (len(plan), PADDED_FRAMES):
        raise ValueError(
            f"scheduled features must be ({len(plan)}, {PADDED_FRAMES}, D), "
            f"got {exact.shape}"
        )
    if exact.shape[2] < 1 or not np.isfinite(exact).all():
        raise ValueError("scheduled features are empty or nonfinite")
    if len(noise_audit) != N_WINDOWS:
        raise ValueError(
            f"EXP-023 expected {N_WINDOWS} audited latent draws, got {len(noise_audit)}"
        )

    by_key = {(int(item["seed"]), str(item["arm"])): int(item["row_index"])
              for item in plan}
    planned_seeds = tuple(dict.fromkeys(int(item["seed"]) for item in plan))
    if (
        len(by_key) != len(plan)
        or set(by_key) != {(seed, arm) for seed in planned_seeds for arm in ARMS}
        or sorted(by_key.values()) != list(range(len(plan)))
    ):
        raise ValueError("EXP-023 causal plan is not a complete local seed/arm product")

    def checked_hashes(value: Any, expected: int, label: str) -> list[str]:
        hashes = list(value if isinstance(value, (list, tuple)) else [])
        if len(hashes) != expected or any(
            not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in hashes
        ):
            raise ValueError(f"{label} contains invalid or missing row hashes")
        return hashes

    shapes: list[list[int]] = []
    hashes_by_window: list[list[str]] = []
    history_reconstruction: list[dict[str, Any]] = []
    for window, audit in enumerate(noise_audit):
        if int(audit.get("window_index", -1)) != window:
            raise ValueError(f"history audit window index mismatch at window {window}")
        if int(audit.get("global_history_start_frame", -1)) != (
            EXPECTED_GLOBAL_HISTORY_START_FRAMES[window]
        ):
            raise ValueError(f"global history start mismatch at window {window}")
        if int(audit.get("accepted_transcript_frames_before", -1)) != (
            EXPECTED_ACCEPTED_TRANSCRIPT_FRAMES_BEFORE[window]
        ):
            raise ValueError(f"accepted transcript input length mismatch at window {window}")
        if int(audit.get("input_history_frames", -1)) != EXPECTED_INPUT_HISTORY_FRAMES[window]:
            raise ValueError(f"history input length mismatch at window {window}")
        if int(audit.get("model_num_frames", -1)) != EXPECTED_MODEL_NUM_FRAMES[window]:
            raise ValueError(f"model-visible frame count mismatch at window {window}")
        if int(audit.get("transcript_frames", -1)) != EXPECTED_TRANSCRIPT_FRAMES[window]:
            raise ValueError(f"stable transcript length mismatch at window {window}")
        hashes = checked_hashes(
            audit.get("row_sha256"), len(plan), f"latent audit window {window}")
        shape = list(audit.get("shape", []))
        if (
            len(shape) != 3
            or shape[0] != len(plan)
            or any(dimension <= 0 for dimension in shape)
        ):
            raise ValueError(f"latent audit row/shape mismatch at window {window}")
        stable = checked_hashes(
            audit.get("stable_transcript_row_sha256"), len(plan),
            f"stable transcript audit window {window}")
        expected_stable = [
            _raw_row_sha256(exact[row, :EXPECTED_TRANSCRIPT_FRAMES[window]])
            for row in range(len(plan))
        ]
        if stable != expected_stable:
            raise ValueError(
                f"stable transcript hashes do not match archived features at window {window}"
            )

        expected_history_rows = 0 if window == 0 else len(plan)
        input_hashes = checked_hashes(
            audit.get("input_history_row_sha256"), expected_history_rows,
            f"input history audit window {window}")
        returned_hashes = checked_hashes(
            audit.get("returned_input_history_row_sha256"), expected_history_rows,
            f"returned history audit window {window}")
        expected_input_hashes = [] if window == 0 else [
            _raw_row_sha256(exact[
                row,
                EXPECTED_GLOBAL_HISTORY_START_FRAMES[window]:
                EXPECTED_ACCEPTED_TRANSCRIPT_FRAMES_BEFORE[window],
            ])
            for row in range(len(plan))
        ]
        if input_hashes != expected_input_hashes:
            raise ValueError(
                f"window {window} did not consume the locked visible transcript suffix"
            )
        reconstruction_exact = list(
            audit.get("returned_history_reconstruction_exact", []))
        reconstruction_max_abs = list(
            audit.get("returned_history_reconstruction_max_abs", []))
        if (
            len(reconstruction_exact) != len(plan)
            or any(not isinstance(value, (bool, np.bool_)) for value in reconstruction_exact)
            or len(reconstruction_max_abs) != len(plan)
            or not np.isfinite(np.asarray(reconstruction_max_abs, dtype=float)).all()
            or any(float(value) < 0.0 for value in reconstruction_max_abs)
        ):
            raise ValueError(f"history reconstruction audit is invalid at window {window}")
        if window == 0 and (
            not all(bool(value) for value in reconstruction_exact)
            or any(float(value) != 0.0 for value in reconstruction_max_abs)
        ):
            raise ValueError("first window cannot report a reconstructed input history")
        if window and any(
            bool(is_exact) and float(max_abs) != 0.0
            for is_exact, max_abs in zip(reconstruction_exact, reconstruction_max_abs)
        ):
            raise ValueError(
                f"exact history reconstruction has nonzero error at window {window}"
            )
        shapes.append(shape)
        hashes_by_window.append(hashes)
        history_reconstruction.append({
            "window": window,
            "global_history_start_frame": EXPECTED_GLOBAL_HISTORY_START_FRAMES[window],
            "accepted_transcript_frames_before": (
                EXPECTED_ACCEPTED_TRANSCRIPT_FRAMES_BEFORE[window]
            ),
            "input_history_frames": EXPECTED_INPUT_HISTORY_FRAMES[window],
            "model_num_frames": EXPECTED_MODEL_NUM_FRAMES[window],
            "transcript_frames": EXPECTED_TRANSCRIPT_FRAMES[window],
            "n_returned_prefixes_exact": int(sum(bool(value) for value in reconstruction_exact)),
            "max_returned_prefix_error": float(max(
                (float(value) for value in reconstruction_max_abs), default=0.0)),
            "stable_hashes_match_archived_features": True,
            "input_hashes_match_visible_transcript_suffix": True,
        })
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError("initial-latent shape changed across autoregressive windows")

    paired_noise: list[dict[str, Any]] = []
    for seed in planned_seeds:
        indices = [by_key[(seed, arm)] for arm in ARMS]
        for window, hashes in enumerate(hashes_by_window):
            values = [hashes[index] for index in indices]
            if len(set(values)) != 1:
                raise ValueError(
                    f"same-seed arms received unequal noise at seed {seed}, window {window}"
                )
            paired_noise.append({
                "seed": seed,
                "window": window,
                "sha256": values[0],
                "all_four_arms_equal": True,
            })
        for arm, index in zip(ARMS, indices):
            stream = [hashes[index] for hashes in hashes_by_window]
            if len(set(stream)) != N_WINDOWS:
                raise ValueError(
                    f"latent replay detected for seed {seed}, arm {arm}: {stream}"
                )
    for window, hashes in enumerate(hashes_by_window):
        per_seed = [hashes[by_key[(seed, "all_walk")]] for seed in planned_seeds]
        if len(set(per_seed)) != len(planned_seeds):
            raise ValueError(f"distinct seeds collided in latent audit at window {window}")

    feature_forks: list[dict[str, Any]] = []
    for seed in planned_seeds:
        walk = exact[by_key[(seed, "all_walk")]]
        step52 = exact[by_key[(seed, "step_52")]]
        step104 = exact[by_key[(seed, "step_104")]]
        if not np.array_equal(walk[:52], step52[:52]):
            raise ValueError(f"step_52 feature prefix differs from WALK for seed {seed}")
        if not np.array_equal(walk[:52], step104[:52]):
            raise ValueError(f"step_104 frame-52 prefix differs from WALK for seed {seed}")
        if not np.array_equal(walk[:104], step104[:104]):
            raise ValueError(f"step_104 feature prefix differs from WALK for seed {seed}")
        feature_forks.append({
            "seed": seed,
            "frame_52_sha256": _array_sha256(walk[:52], "prefix"),
            "frame_104_sha256": _array_sha256(walk[:104], "prefix"),
            "step_52_exact_through_51": True,
            "step_104_exact_through_103": True,
        })

    return {
        "n_windows": N_WINDOWS,
        "latent_shapes": shapes,
        "paired_noise": paired_noise,
        "feature_forks": feature_forks,
        "history_reconstruction": history_reconstruction,
        "corresponding_window_noise_equal": True,
        "noise_fresh_across_windows": True,
        "history_inputs_match_visible_transcript_suffixes": True,
        "stable_transcript_hashes_match_archived_features": True,
        "feature_prefixes_exact": True,
    }


def _merge_chunk_audits(
    chunk_results: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
    n_rows: int,
) -> list[dict[str, Any]]:
    """Reconstruct a global row-ordered audit from four independently durable B=8 calls."""
    if len(chunk_results) != len(SEEDS) // CHUNK_SEED_COUNT:
        raise ValueError("EXP-023 does not have all four chunk audits")
    row_fields = (
        "row_sha256",
        "returned_history_reconstruction_exact",
        "returned_history_reconstruction_max_abs",
        "stable_transcript_row_sha256",
    )
    continuation_fields = (
        "input_history_row_sha256",
        "returned_input_history_row_sha256",
    )
    merged: list[dict[str, Any]] = []
    for window in range(N_WINDOWS):
        output: dict[str, Any] = {
            "window_index": window,
            "global_history_start_frame": EXPECTED_GLOBAL_HISTORY_START_FRAMES[window],
            "accepted_transcript_frames_before": (
                EXPECTED_ACCEPTED_TRANSCRIPT_FRAMES_BEFORE[window]
            ),
            "input_history_frames": EXPECTED_INPUT_HISTORY_FRAMES[window],
            "model_num_frames": EXPECTED_MODEL_NUM_FRAMES[window],
            "transcript_frames": EXPECTED_TRANSCRIPT_FRAMES[window],
        }
        tails: list[list[int]] = []
        for field in row_fields + continuation_fields:
            if window == 0 and field in continuation_fields:
                output[field] = []
                continue
            values: list[Any] = [None] * n_rows
            for chunk, audit in chunk_results:
                if len(audit) != N_WINDOWS:
                    raise ValueError(f"{chunk['name']} has the wrong window-audit count")
                local = list(audit[window].get(field, []))
                indices = list(chunk["row_indices"])
                if len(local) != len(indices):
                    raise ValueError(
                        f"{chunk['name']} has the wrong {field} count at window {window}"
                    )
                for global_index, value in zip(indices, local):
                    if values[int(global_index)] is not None:
                        raise ValueError("chunk audits overlap in global row space")
                    values[int(global_index)] = value
            if any(value is None for value in values):
                raise ValueError("chunk audits do not cover every global row")
            output[field] = values
        for chunk, audit in chunk_results:
            entry = audit[window]
            if (
                int(entry.get("window_index", -1)) != window
                or int(entry.get("global_history_start_frame", -1))
                != EXPECTED_GLOBAL_HISTORY_START_FRAMES[window]
                or int(entry.get("accepted_transcript_frames_before", -1))
                != EXPECTED_ACCEPTED_TRANSCRIPT_FRAMES_BEFORE[window]
                or int(entry.get("input_history_frames", -1))
                != EXPECTED_INPUT_HISTORY_FRAMES[window]
                or int(entry.get("model_num_frames", -1))
                != EXPECTED_MODEL_NUM_FRAMES[window]
                or int(entry.get("transcript_frames", -1))
                != EXPECTED_TRANSCRIPT_FRAMES[window]
            ):
                raise ValueError(f"{chunk['name']} changed the locked history policy")
            shape = list(entry.get("shape", []))
            if len(shape) != 3 or shape[0] != len(chunk["row_indices"]):
                raise ValueError(f"{chunk['name']} has an invalid latent shape")
            tails.append(shape[1:])
        if any(tail != tails[0] for tail in tails[1:]):
            raise ValueError("latent shape differs across EXP-023 chunks")
        output["shape"] = [n_rows, *tails[0]]
        merged.append(output)
    return merged


def _serialized_noise_evidence(
    audit: Sequence[Mapping[str, Any]], plan: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Retain every causal hash and reconstruction diagnostic in the final artifact."""
    evidence: list[dict[str, Any]] = []
    for window, item in enumerate(audit):
        rows = []
        for row in plan:
            index = int(row["row_index"])
            rows.append({
                "row_index": index,
                "seed": int(row["seed"]),
                "arm": str(row["arm"]),
                "initial_noise_sha256": str(item["row_sha256"][index]),
                "input_history_sha256": (
                    None if window == 0
                    else str(item["input_history_row_sha256"][index])
                ),
                "returned_input_history_sha256": (
                    None if window == 0
                    else str(item["returned_input_history_row_sha256"][index])
                ),
                "returned_history_reconstruction_exact": bool(
                    item["returned_history_reconstruction_exact"][index]),
                "returned_history_reconstruction_max_abs": float(
                    item["returned_history_reconstruction_max_abs"][index]),
                "stable_transcript_sha256": str(
                    item["stable_transcript_row_sha256"][index]),
            })
        evidence.append({
            "window_index": window,
            "window_start_frame": window * HORIZON,
            "global_history_start_frame": int(item["global_history_start_frame"]),
            "accepted_transcript_frames_before": int(
                item["accepted_transcript_frames_before"]),
            "input_history_frames": int(item["input_history_frames"]),
            "model_num_frames": int(item["model_num_frames"]),
            "transcript_frames": int(item["transcript_frames"]),
            "latent_shape": list(item["shape"]),
            "rows": rows,
        })
    return evidence


def _validate_qpos_forks(
    qpos_by_key: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        walk = np.asarray(qpos_by_key[f"s{seed}_all_walk"])
        step52 = np.asarray(qpos_by_key[f"s{seed}_step_52"])
        step104 = np.asarray(qpos_by_key[f"s{seed}_step_104"])
        if not np.array_equal(walk[:52], step52[:52]):
            raise ValueError(f"step_52 decoded qpos prefix differs for seed {seed}")
        if not np.array_equal(walk[:104], step104[:104]):
            raise ValueError(f"step_104 decoded qpos prefix differs for seed {seed}")
        rows.append({
            "seed": seed,
            "frame_52_sha256": _array_sha256(walk[:52], "qpos_prefix"),
            "frame_104_sha256": _array_sha256(walk[:104], "qpos_prefix"),
            "step_52_exact_through_51": True,
            "step_104_exact_through_103": True,
        })
    return rows


def _ordered_crossing_frames(
    foot: Mapping[str, np.ndarray], slab_low: float, slab_high: float
) -> np.ndarray:
    forward_min = np.asarray(foot["forward_min_m"], dtype=float)
    forward_max = np.asarray(foot["forward_max_m"], dtype=float)
    overlap = (forward_max >= slab_low) & (forward_min <= slab_high)
    before = forward_max < slab_low
    after = forward_min > slab_high
    frames: list[int] = []
    for frame in np.flatnonzero(overlap):
        if np.any(before[:frame]) and np.any(after[frame + 1 :]):
            frames.append(int(frame))
    return np.asarray(frames, dtype=int)


def fixed_box_traversal(
    feet: Mapping[str, Mapping[str, np.ndarray]], obstacle_x_m: float
) -> dict[str, Any]:
    """Require both physical feet to pass the margin-expanded fixed obstacle slab."""
    half = OBSTACLE_DEPTH_M / 2.0 + BODY_MARGIN
    low, high = float(obstacle_x_m) - half, float(obstacle_x_m) + half
    per_side: dict[str, Any] = {}
    traversed = True
    for side in ("left", "right"):
        if side not in feet:
            raise ValueError(f"fixed-box foot series lacks {side}")
        required = ("forward_min_m", "forward_max_m")
        arrays = [np.asarray(feet[side].get(name), dtype=float) for name in required]
        if (
            any(array.ndim != 1 or len(array) == 0 for array in arrays)
            or len(arrays[0]) != len(arrays[1])
            or not all(np.isfinite(array).all() for array in arrays)
        ):
            raise ValueError(f"fixed-box foot series is invalid for {side}")
        frames = _ordered_crossing_frames(feet[side], low, high)
        crossed = bool(len(frames))
        traversed = traversed and crossed
        per_side[side] = {
            "crossed_before_over_after": crossed,
            "overlap_frame_count": int(len(frames)),
            "first_ordered_overlap_frame": int(frames[0]) if len(frames) else None,
            "last_ordered_overlap_frame": int(frames[-1]) if len(frames) else None,
        }
    return {
        "traversed": bool(traversed),
        "criterion": "both physical feet cross before-over-after",
        "margin_expanded_slab_low_m": low,
        "margin_expanded_slab_high_m": high,
        "per_side": per_side,
    }


def select_event_from_profile(
    profile_x_m: np.ndarray,
    profile_height_m: np.ndarray,
    feet: Mapping[str, Mapping[str, np.ndarray]],
    *,
    onset_frame: int,
    min_clearance_m: float = EVENT_MIN_CLEARANCE_M,
    obstacle_depth_m: float = OBSTACLE_DEPTH_M,
) -> dict[str, Any]:
    """Select a deterministic traversed whole-body-clearable event from measured arrays."""
    xs = np.asarray(profile_x_m, dtype=float)
    heights = np.asarray(profile_height_m, dtype=float)
    if (
        xs.ndim != 1
        or heights.shape != xs.shape
        or len(xs) == 0
        or not np.isfinite(xs).all()
        or not np.isfinite(heights).all()
        or np.any(np.diff(xs) <= 0)
    ):
        raise ValueError("event clearance profile is invalid")
    length = None
    for side in ("left", "right"):
        if side not in feet:
            raise ValueError(f"event foot kinematics lack {side}")
        names = (
            "forward_min_m", "forward_max_m", "forward_representative_m",
            "bottom_clearance_m")
        arrays = [np.asarray(feet[side][name], dtype=float) for name in names]
        if any(array.ndim != 1 or not np.isfinite(array).all() for array in arrays):
            raise ValueError(f"event foot series is invalid for {side}")
        sizes = {len(array) for array in arrays}
        if len(sizes) != 1:
            raise ValueError(f"event foot series lengths disagree for {side}")
        side_length = sizes.pop()
        if length is None:
            length = side_length
        elif length != side_length:
            raise ValueError("left/right event foot series lengths disagree")

    global_maximum = float(np.max(heights))
    traversed: list[tuple[int, dict[str, np.ndarray]]] = []
    half = float(obstacle_depth_m) / 2.0 + BODY_MARGIN
    for profile_index in range(len(xs)):
        x = float(xs[profile_index])
        crossings = {
            side: _ordered_crossing_frames(feet[side], x - half, x + half)
            for side in ("left", "right")
        }
        # A clearable box is a traversal event only when both physical feet actually pass it.
        if any(not len(frames) for frames in crossings.values()):
            continue
        traversed.append((profile_index, crossings))

    if not traversed:
        return {
            "present": False,
            "missing_reason": "profile_has_no_two_foot_before_over_after_traversal",
            "max_profile_height_m": None,
            "global_unfiltered_max_profile_height_m": global_maximum,
        }

    traversed_maximum = float(max(heights[index] for index, _ in traversed))
    if traversed_maximum < float(min_clearance_m):
        return {
            "present": False,
            "missing_reason": "whole_body_clearance_below_3cm",
            "max_profile_height_m": traversed_maximum,
            "global_unfiltered_max_profile_height_m": global_maximum,
        }

    candidates: list[tuple[tuple[float, int], dict[str, Any]]] = []
    for profile_index, crossings in traversed:
        if float(heights[profile_index]) < float(min_clearance_m):
            continue
        x = float(xs[profile_index])
        foot_candidates: list[tuple[tuple[float, int, int], str, int]] = []
        for side in ("left", "right"):
            clearance = np.asarray(feet[side]["bottom_clearance_m"], dtype=float)
            for frame in crossings[side]:
                foot_candidates.append((
                    (float(clearance[frame]), -int(frame), int(side == "left")),
                    side,
                    int(frame),
                ))
        _, side, frame = max(foot_candidates, key=lambda item: item[0])
        representative = np.asarray(
            feet[side]["forward_representative_m"], dtype=float)
        clearance = np.asarray(feet[side]["bottom_clearance_m"], dtype=float)
        event = {
            "present": True,
            "missing_reason": None,
            "frame": int(onset_frame + frame),
            "latency_frames": int(frame),
            "latency_s": float(frame / FPS),
            "side": side,
            "profile_x_m": x,
            "foot_x_m": float(representative[frame]),
            "whole_body_clearance_m": float(heights[profile_index]),
            "foot_bottom_clearance_m": float(clearance[frame]),
            "profile_index": int(profile_index),
            "max_profile_height_m": traversed_maximum,
            "global_unfiltered_max_profile_height_m": global_maximum,
        }
        # Maximum whole-body height, then earlier route centre.
        candidates.append(((float(heights[profile_index]), -int(profile_index)), event))
    # The traversed maximum passed the threshold, so at least one candidate must exist.
    if not candidates:  # pragma: no cover - defensive consistency check
        raise ValueError("event traversal/threshold selection became inconsistent")
    return max(candidates, key=lambda item: item[0])[1]


def detect_prompt_event(
    body: Any,
    qpos: np.ndarray,
    route: np.ndarray,
    onset_frame: int,
    *,
    probe_factory: Callable[..., Any] = BoxHeightProbe,
) -> dict[str, Any]:
    """Measure the preregistered 96-frame whole-body event endpoint."""
    exact_qpos = np.asarray(qpos, dtype=float)
    exact_route = np.asarray(route, dtype=float)
    start = int(onset_frame)
    end = start + POST_ONSET_FRAMES
    if (
        exact_qpos.ndim != 2
        or exact_route.shape != (N_FRAMES, 2)
        or start not in CONTROL_ONSETS
        or end > N_FRAMES
        or len(exact_qpos) < N_FRAMES
    ):
        raise ValueError("event detector received an invalid clip, route, or onset")
    segment_qpos = exact_qpos[start:end]
    segment_route = exact_route[start:end]
    low = float(segment_route[0, 1] + PROFILE_MARGIN_M)
    high = float(segment_route[-1, 1] - PROFILE_MARGIN_M)
    if high <= low:
        raise ValueError("event analysis route is too short for the locked scan margin")
    xs = np.linspace(low, high, PROFILE_POINTS, dtype=float)
    heights = np.asarray([
        probe_factory(float(x), OBSTACLE_DEPTH_M).probe(segment_qpos)
        for x in xs
    ], dtype=float)
    feet = foot_kinematics_series(body, segment_qpos, FPS)
    event = select_event_from_profile(
        xs,
        heights,
        feet,
        onset_frame=start,
        min_clearance_m=EVENT_MIN_CLEARANCE_M,
        obstacle_depth_m=OBSTACLE_DEPTH_M,
    )
    return {
        **event,
        "analysis_window_start_frame": start,
        "analysis_window_end_frame": end - 1,
        "analysis_window_frames": POST_ONSET_FRAMES,
        "profile_points": PROFILE_POINTS,
        "profile_margin_m": PROFILE_MARGIN_M,
        "event_min_clearance_m": EVENT_MIN_CLEARANCE_M,
        "obstacle_depth_m": OBSTACLE_DEPTH_M,
    }


def score_exact_boxes(
    body: Any,
    qpos: np.ndarray,
    obstacle_x_m: float,
    *,
    probe_factory: Callable[..., Any] = BoxHeightProbe,
    foot_series_fn: Callable[[Any, np.ndarray, float], Mapping[str, Any]] = (
        foot_kinematics_series
    ),
) -> dict[str, Any]:
    exact = np.asarray(qpos, dtype=float)
    if exact.ndim != 2 or exact.shape[0] != N_FRAMES or not np.isfinite(exact).all():
        raise ValueError("fixed-box scoring requires a finite 200-frame qpos clip")
    probe = probe_factory(float(obstacle_x_m), OBSTACLE_DEPTH_M)
    raw_lower_bound = float(probe.probe(exact))
    traversal = fixed_box_traversal(
        foot_series_fn(body, exact, FPS), float(obstacle_x_m))
    collision_free = {
        f"{height:g}": bool(probe.clears(exact, height))
        for height in GRADED_HEIGHTS_M
    }
    clears = {
        key: bool(traversal["traversed"] and value)
        for key, value in collision_free.items()
    }
    return {
        "obstacle_x_m": float(obstacle_x_m),
        "obstacle_depth_m": OBSTACLE_DEPTH_M,
        "traversal": traversal,
        "max_box_height_collision_free_lower_bound_m": raw_lower_bound,
        "max_box_height_lower_bound_m": (
            raw_lower_bound if traversal["traversed"] else 0.0),
        "collision_free": collision_free,
        "clears": clears,
    }


def supporting_motion_metrics(body: Any, qpos: np.ndarray, route: np.ndarray) -> dict[str, Any]:
    exact = np.asarray(qpos, dtype=float)
    exact_route = np.asarray(route, dtype=float)
    path_world = np.stack([exact[:, 1], exact[:, 0]], axis=-1)
    report = body.trajectory_report(exact)
    return {
        "progress_ratio": float(e17._prescribed_progress_ratio(exact, exact_route)),
        "route_path_mae_m": float(np.linalg.norm(path_world - exact_route, axis=-1).mean()),
        "max_foot_floor_penetration_m": float(report["max_foot_floor_penetration_m"]),
    }


def _validated_event_record(value: Mapping[str, Any], onset: int) -> dict[str, Any]:
    """Validate even dependency-injected detector output before it reaches a gate."""
    event = dict(value)
    present = event.get("present")
    if not isinstance(present, (bool, np.bool_)):
        raise ValueError("event record requires boolean present")
    if present:
        required = (
            "frame", "latency_frames", "latency_s", "side", "profile_x_m",
            "foot_x_m", "whole_body_clearance_m", "foot_bottom_clearance_m",
        )
        if any(name not in event for name in required):
            raise ValueError("present event record is incomplete")
        frame = int(event["frame"])
        latency = int(event["latency_frames"])
        if frame != int(onset) + latency:
            raise ValueError("event frame and prompt-relative latency disagree")
        if not int(onset) <= frame < int(onset) + POST_ONSET_FRAMES:
            raise ValueError("event lies outside the locked 96-frame analysis window")
        if event["side"] not in ("left", "right"):
            raise ValueError("event side must be left or right")
        numeric = [
            event[name] for name in (
                "latency_s", "profile_x_m", "foot_x_m",
                "whole_body_clearance_m", "foot_bottom_clearance_m",
            )
        ]
        if not np.isfinite(np.asarray(numeric, dtype=float)).all():
            raise ValueError("present event contains a non-finite endpoint")
        if float(event["whole_body_clearance_m"]) < EVENT_MIN_CLEARANCE_M:
            raise ValueError("present event does not meet the locked 3 cm endpoint")
    else:
        if not isinstance(event.get("missing_reason"), str) or not event["missing_reason"]:
            raise ValueError("missing event requires a reason")
        maximum = event.get("max_profile_height_m")
        if maximum is not None and not np.isfinite(float(maximum)):
            raise ValueError("missing event maximum must be finite or null")
    return _json_safe(event)


def _validated_exact_box_record(
    value: Mapping[str, Any], expected_x_m: float
) -> dict[str, Any]:
    record = dict(value)
    try:
        x = float(record["obstacle_x_m"])
        depth = float(record["obstacle_depth_m"])
        lower = float(record["max_box_height_lower_bound_m"])
        raw_lower = float(record["max_box_height_collision_free_lower_bound_m"])
        traversal = dict(record["traversal"])
        traversed = traversal["traversed"]
        collision_free = dict(record["collision_free"])
        clears = dict(record["clears"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("fixed-box record is incomplete") from exc
    if not np.isclose(x, float(expected_x_m), atol=1e-12, rtol=0.0):
        raise ValueError("fixed-box scorer changed the frozen obstacle centre")
    if not np.isclose(depth, OBSTACLE_DEPTH_M, atol=0.0, rtol=0.0):
        raise ValueError("fixed-box scorer changed the frozen obstacle depth")
    if (
        not np.isfinite(lower)
        or lower < 0.0
        or not np.isfinite(raw_lower)
        or raw_lower < 0.0
    ):
        raise ValueError("fixed-box lower bound is invalid")
    if not isinstance(traversed, (bool, np.bool_)):
        raise ValueError("fixed-box traversal flag must be boolean")
    if not isinstance(traversal.get("criterion"), str) or not traversal["criterion"]:
        raise ValueError("fixed-box traversal criterion is missing")
    per_side = traversal.get("per_side")
    if not isinstance(per_side, Mapping) or set(per_side) != {"left", "right"}:
        raise ValueError("fixed-box traversal must report both physical feet")
    side_flags: list[bool] = []
    for side in ("left", "right"):
        side_record = per_side[side]
        if not isinstance(side_record, Mapping) or not isinstance(
            side_record.get("crossed_before_over_after"), (bool, np.bool_)
        ):
            raise ValueError(f"fixed-box traversal is invalid for {side}")
        side_flags.append(bool(side_record["crossed_before_over_after"]))
    if bool(traversed) != all(side_flags):
        raise ValueError("fixed-box traversal aggregate disagrees with its feet")
    expected_lower = raw_lower if bool(traversed) else 0.0
    if not np.isclose(lower, expected_lower, atol=0.0, rtol=0.0):
        raise ValueError("fixed-box endpoint did not zero a nonarrival lower bound")
    keys = [f"{height:g}" for height in GRADED_HEIGHTS_M]
    if set(collision_free) != set(keys) or set(clears) != set(keys) or any(
        not isinstance(collision_free[key], (bool, np.bool_))
        or not isinstance(clears[key], (bool, np.bool_))
        for key in keys
    ):
        raise ValueError("fixed-box scorer did not return the locked graded heights")
    expected_clears = {
        key: bool(traversed) and bool(collision_free[key]) for key in keys
    }
    if {key: bool(clears[key]) for key in keys} != expected_clears:
        raise ValueError("fixed-box success is not traversal-gated collision freedom")
    raw_truth = [bool(collision_free[key]) for key in keys]
    truth = [bool(clears[key]) for key in keys]
    if any(
        values[index] and not values[index - 1]
        for values in (raw_truth, truth)
        for index in range(1, len(values))
    ):
        raise ValueError("fixed-box clearances are not monotone by height")
    return {
        **_json_safe(record),
        "obstacle_x_m": x,
        "obstacle_depth_m": depth,
        "traversal": _json_safe(traversal),
        "max_box_height_collision_free_lower_bound_m": raw_lower,
        "max_box_height_lower_bound_m": lower,
        "collision_free": {key: bool(collision_free[key]) for key in keys},
        "clears": {key: bool(clears[key]) for key in keys},
    }


def _validated_motion_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    required = (
        "progress_ratio", "route_path_mae_m", "max_foot_floor_penetration_m")
    if any(name not in record for name in required):
        raise ValueError("supporting-motion record is incomplete")
    if not np.isfinite(np.asarray([record[name] for name in required], dtype=float)).all():
        raise ValueError("supporting-motion record contains a non-finite value")
    return _json_safe(record)


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 2 or len(np.unique(x)) < 2:
        return None
    return float(np.polyfit(x, y, 1)[0])


def summarize_rows(rows: Sequence[Mapping[str, Any]], route: np.ndarray) -> dict[str, Any]:
    by_seed_arm = {
        (int(row["seed"]), str(row["arm"])): row for row in rows
    }
    step_arms = ("step_0", "step_52", "step_104")
    event_rates: dict[str, Any] = {}
    latency: dict[str, Any] = {}
    pooled_onsets: list[float] = []
    pooled_frames: list[float] = []
    pooled_onset_x: list[float] = []
    pooled_event_x: list[float] = []
    for arm in step_arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        events = [row["event"] for row in arm_rows]
        present = [event for event in events if event["present"]]
        event_rates[arm] = {
            "present": len(present),
            "planned": len(events),
            "rate": float(len(present) / len(events)),
            "missing": len(events) - len(present),
        }
        latencies = [float(event["latency_frames"]) for event in present]
        latency[arm] = {
            "values_frames": latencies,
            "median_frames": float(np.median(latencies)) if latencies else None,
        }
        onset = int(ONSETS[arm])
        for event in present:
            pooled_onsets.append(float(onset))
            pooled_frames.append(float(event["frame"]))
            pooled_onset_x.append(float(route[onset, 1]))
            pooled_event_x.append(float(event["profile_x_m"]))

    per_seed_slopes: list[dict[str, Any]] = []
    paired: dict[str, list[dict[str, Any]]] = {"52_minus_0": [], "104_minus_0": []}
    for seed in SEEDS:
        events = {arm: by_seed_arm[(seed, arm)]["event"] for arm in step_arms}
        if all(event["present"] for event in events.values()):
            onset_values = [0.0, 52.0, 104.0]
            frame_values = [float(events[arm]["frame"]) for arm in step_arms]
            onset_x_values = [float(route[int(ONSETS[arm]), 1]) for arm in step_arms]
            event_x_values = [float(events[arm]["profile_x_m"]) for arm in step_arms]
            per_seed_slopes.append({
                "seed": seed,
                "event_frame_on_onset_frame": _slope(onset_values, frame_values),
                "event_x_on_onset_x": _slope(onset_x_values, event_x_values),
            })
        for arm, label in (("step_52", "52_minus_0"), ("step_104", "104_minus_0")):
            first, delayed = events["step_0"], events[arm]
            if first["present"] and delayed["present"]:
                paired[label].append({
                    "seed": seed,
                    "event_frame_difference": int(delayed["frame"] - first["frame"]),
                    "event_x_difference_m": float(
                        delayed["profile_x_m"] - first["profile_x_m"]),
                    "latency_difference_frames": int(
                        delayed["latency_frames"] - first["latency_frames"]),
                })

    walk_rows = [row for row in rows if row["arm"] == "all_walk"]
    walk_window_hits = sum(
        int(event["present"])
        for row in walk_rows
        for event in row["control_events"].values()
    )
    walk_seed_hits = sum(
        any(event["present"] for event in row["control_events"].values())
        for row in walk_rows
    )

    fixed_box_rates: dict[str, Any] = {}
    for arm in step_arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        fixed_box_rates[arm] = {
            f"{height:g}": {
                "clears": int(sum(row["fixed_box"]["clears"][f"{height:g}"]
                                  for row in arm_rows)),
                "planned": len(arm_rows),
                "rate": float(sum(
                    row["fixed_box"]["clears"][f"{height:g}"] for row in arm_rows
                ) / len(arm_rows)),
            }
            for height in GRADED_HEIGHTS_M
        }
    fixed_box_rates["all_walk_matched_windows"] = {
        str(onset): {
            f"{height:g}": {
                "clears": int(sum(
                    row["control_fixed_boxes"][str(onset)]["clears"][f"{height:g}"]
                    for row in walk_rows
                )),
                "planned": len(walk_rows),
                "rate": float(sum(
                    row["control_fixed_boxes"][str(onset)]["clears"][f"{height:g}"]
                    for row in walk_rows
                ) / len(walk_rows)),
            }
            for height in GRADED_HEIGHTS_M
        }
        for onset in CONTROL_ONSETS
    }

    fixed_box_traversal_rates: dict[str, Any] = {
        arm: {
            "traversed": int(sum(
                bool(row["fixed_box"]["traversal"]["traversed"])
                for row in rows if row["arm"] == arm
            )),
            "planned": int(sum(row["arm"] == arm for row in rows)),
        }
        for arm in step_arms
    }
    for arm in step_arms:
        item = fixed_box_traversal_rates[arm]
        item["rate"] = float(item["traversed"] / item["planned"])
    fixed_box_traversal_rates["all_walk_matched_windows"] = {
        str(onset): {
            "traversed": int(sum(
                bool(row["control_fixed_boxes"][str(onset)]["traversal"]["traversed"])
                for row in walk_rows
            )),
            "planned": len(walk_rows),
            "rate": float(sum(
                bool(row["control_fixed_boxes"][str(onset)]["traversal"]["traversed"])
                for row in walk_rows
            ) / len(walk_rows)),
        }
        for onset in CONTROL_ONSETS
    }

    seed_slope_values = [
        float(item["event_frame_on_onset_frame"]) for item in per_seed_slopes
    ]
    return {
        "event_rates_missing_retained": event_rates,
        "prompt_relative_latency": latency,
        "pooled_present_event_slopes_descriptive": {
            "event_frame_on_onset_frame": _slope(pooled_onsets, pooled_frames),
            "event_x_on_onset_x": _slope(pooled_onset_x, pooled_event_x),
            "n_present_events": len(pooled_frames),
            "missing_events_not_imputed": True,
        },
        "per_seed_complete_case_slopes": per_seed_slopes,
        "median_per_seed_event_frame_slope": (
            float(np.median(seed_slope_values)) if seed_slope_values else None
        ),
        "paired_present_event_differences": paired,
        "paired_present_event_counts": {
            label: {"complete_pairs": len(values), "planned_pairs": len(SEEDS)}
            for label, values in paired.items()
        },
        "all_walk_specificity": {
            "seeds_with_any_event": int(walk_seed_hits),
            "planned_seeds": len(walk_rows),
            "window_events": int(walk_window_hits),
            "planned_windows": len(walk_rows) * len(CONTROL_ONSETS),
        },
        "fixed_box_rates": fixed_box_rates,
        "fixed_box_traversal_rates": fixed_box_traversal_rates,
        "timing_interpretation": {
            "primary": "planned-denominator delayed-arm event rates",
            "present_only_slopes_role": "secondary diagnostic only",
            "complete_case_seed_count": len(per_seed_slopes),
            "planned_seed_count": len(SEEDS),
            "binary_timed_prompting_verdict": None,
            "reason": (
                "No binary support verdict is assigned at n=8; present-only slopes cannot "
                "override missing delayed events."
            ),
        },
        "inference": (
            "descriptive paired; planned-denominator rates are primary; present-only slopes "
            "are secondary; no interval or binary timing verdict at n=8"
        ),
    }


def run_campaign(
    *,
    out: str | Path,
    runner: Any | None = None,
    runner_factory: Callable[[], Any] | None = None,
    body: Any | None = None,
    cache_path: str | Path = "outputs/text_cache.npz",
    code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
    source_hashes_fn: Callable[[Path], Mapping[str, str]] = _source_hashes,
    generator_identity_fn: Callable[[Any], Mapping[str, Any]] = cal._generator_identity,
    generator_identity_validator_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] = (
        cal._validated_generator_identity
    ),
    runtime_identity_fn: Callable[[], Mapping[str, Any]] = cal._runtime_identity,
    physical_identity_fn: Callable[[], Mapping[str, Any]] = cal._physical_model_identity,
    prompt_identity_fn: Callable[[Any, Path], Mapping[str, Any]] = _prompt_cache_identity,
    pin_validator_fn: Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], None] = (
        _validate_pins
    ),
    channel_usage_fn: Callable[[Any, ConstraintSpec], Mapping[str, int]] = _actual_channel_usage,
    event_detector_fn: Callable[[Any, np.ndarray, np.ndarray, int], Mapping[str, Any]] = (
        detect_prompt_event
    ),
    exact_box_fn: Callable[[Any, np.ndarray, float], Mapping[str, Any]] = score_exact_boxes,
    motion_metrics_fn: Callable[[Any, np.ndarray, np.ndarray], Mapping[str, Any]] = (
        supporting_motion_metrics
    ),
) -> dict[str, Any]:
    """Run the locked prompt-handoff campaign or raise after durable refusal."""
    output = Path(out)
    if output.exists() and any(output.iterdir()):
        raise PromptHandoffAbort(f"refusing nonempty EXP-023 output directory: {output}")
    repo = Path(__file__).resolve().parents[1]
    code = dict(code_state_fn(repo))
    injected_components = []
    if runner is not None:
        injected_components.append("runner_instance")
    if runner_factory is not None:
        injected_components.append("runner_factory")
    if body is not None:
        injected_components.append("body")
    callbacks = (
        ("code_state_fn", code_state_fn, cal._git_state),
        ("source_hashes_fn", source_hashes_fn, _source_hashes),
        ("generator_identity_fn", generator_identity_fn, cal._generator_identity),
        ("generator_identity_validator_fn", generator_identity_validator_fn,
         cal._validated_generator_identity),
        ("runtime_identity_fn", runtime_identity_fn, cal._runtime_identity),
        ("physical_identity_fn", physical_identity_fn, cal._physical_model_identity),
        ("prompt_identity_fn", prompt_identity_fn, _prompt_cache_identity),
        ("pin_validator_fn", pin_validator_fn, _validate_pins),
        ("channel_usage_fn", channel_usage_fn, _actual_channel_usage),
        ("event_detector_fn", event_detector_fn, detect_prompt_event),
        ("exact_box_fn", exact_box_fn, score_exact_boxes),
        ("motion_metrics_fn", motion_metrics_fn, supporting_motion_metrics),
    )
    injected_components.extend(
        name for name, value, default in callbacks if value is not default
    )
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stage = "preflight"
    sample_count_exact = True
    rows: list[dict[str, Any]] = []
    qpos_archive: dict[str, np.ndarray] = {}
    feature_archive: dict[str, np.ndarray] = {}
    noise_evidence: list[dict[str, Any]] = []
    plan = locked_row_plan()
    chunks = locked_chunk_plan(plan)
    spent_seeds: list[int] = []
    route = route_xz()
    route_digest = _array_sha256(route, "route_xz")
    counters = {
        "schedule_invocations_planned": len(chunks),
        "schedule_invocations_started": 0,
        "schedule_invocations_completed": 0,
        "autoregressive_window_calls_planned": len(chunks) * N_WINDOWS,
        "autoregressive_window_calls_completed": 0,
        "trajectories_planned": len(plan),
        "trajectories_launched": 0,
        "trajectories_returned": 0,
        "trajectories_converted_to_qpos": 0,
        "trajectories_analyzed": 0,
    }
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "experiment": "exp023_prompt_handoff",
        "status": "running",
        "complete": False,
        "blocked": False,
        "stage": stage,
        "sample_count_exact": sample_count_exact,
        "actual_ardy_samples": 0,
        "campaign_design": {
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "schedules": {arm: list(SCHEDULES[arm]) for arm in ARMS},
            "row_plan": plan,
            "row_plan_sha256": cal._json_hash(plan),
            "chunk_plan": [
                {key: value for key, value in chunk.items() if key != "rows"}
                for chunk in chunks
            ],
            "chunk_seed_count": CHUNK_SEED_COUNT,
            "chunk_batch_rows": CHUNK_ROWS,
            "n_schedule_trajectories": len(plan),
            "sample_accounting_definition": (
                "one complete prompt schedule is one frozen-prior sample; four B=8 chunk "
                "calls each make four Horizon52 window calls, counted separately"
            ),
            "history_policy": {
                "name": "ARDY interactive GUI default minimum history",
                "visible_history_frames": list(EXPECTED_INPUT_HISTORY_FRAMES),
                "global_history_start_frames": list(EXPECTED_GLOBAL_HISTORY_START_FRAMES),
                "accepted_transcript_frames_before": list(
                    EXPECTED_ACCEPTED_TRANSCRIPT_FRAMES_BEFORE),
                "accepted_transcript_frames_after": list(EXPECTED_TRANSCRIPT_FRAMES),
                "model_visible_num_frames": list(EXPECTED_MODEL_NUM_FRAMES),
                "full_accepted_transcript_is_immutable": True,
            },
            "fps": FPS,
            "scored_frames": N_FRAMES,
            "archived_frames": PADDED_FRAMES,
            "post_onset_frames": POST_ONSET_FRAMES,
            "frozen_latency_frames": FROZEN_LATENCY_FRAMES,
            "diffusion_steps": DIFFUSION_STEPS,
            "cfg_weight": list(CFG_WEIGHT),
            "route_sha256": route_digest,
            "conditioning": "dense root XZ only for frames 0:200; 200:208 unconditioned",
            "event_min_clearance_m": EVENT_MIN_CLEARANCE_M,
            "event_profile_points": PROFILE_POINTS,
            "graded_heights_m": list(GRADED_HEIGHTS_M),
            "substrate_gate": f"step_0 present >= {MIN_STEP0_EVENTS}/8",
            "specificity_gate": (
                "all_walk seeds with any event in three matched windows "
                f"<= {MAX_ALL_WALK_SEEDS_WITH_ANY_EVENT}/8"
            ),
        },
        "query_accounting": dict(counters),
        "generation_chunks": {
            str(chunk["name"]): {
                "status": "planned",
                "seeds": list(chunk["seeds"]),
                "row_indices": list(chunk["row_indices"]),
            }
            for chunk in chunks
        },
        "provenance": {"code": code},
        "execution_mode": {
            "dependency_injections": injected_components,
            "scientific_evidence_eligible": not injected_components,
            "pre_model_construction_evidence_guaranteed": runner is None,
            "note": (
                "Dependency injection exists for CPU tests; any injected run is explicitly "
                "non-evidentiary. Production constructs the ARDY runner only after the empty "
                "evidence bundle is durable."
            ),
        },
    }

    artifacts_dirty = {"qpos": True, "features": True, "noise": True}
    artifact_content_hash: dict[str, str | None] = {
        "qpos": None,
        "features": None,
        "noise": None,
    }

    def persist() -> None:
        nonlocal artifacts_dirty
        cal._write_jsonl(output / "rows.jsonl", rows)
        if artifacts_dirty["qpos"]:
            cal._persist_qpos(output / "qpos.npz", qpos_archive)
            artifact_content_hash["qpos"] = (
                cal._array_hash(qpos_archive) if qpos_archive else None
            )
            artifacts_dirty["qpos"] = False
        if artifacts_dirty["features"]:
            cal._persist_qpos(output / "features.npz", feature_archive)
            artifact_content_hash["features"] = (
                cal._array_hash(feature_archive) if feature_archive else None
            )
            artifacts_dirty["features"] = False
        if artifacts_dirty["noise"]:
            cal._write_json(output / "noise_audit.json", noise_evidence)
            artifact_content_hash["noise"] = cal._json_hash(noise_evidence)
            artifacts_dirty["noise"] = False
        receipt["stage"] = stage
        receipt["query_accounting"] = dict(counters)
        receipt["sample_count_exact"] = sample_count_exact
        receipt["actual_ardy_samples"] = int(counters["trajectories_returned"])
        receipt["spent_seeds"] = list(spent_seeds)
        receipt["unlaunched_locked_seeds"] = [
            seed for seed in SEEDS if seed not in spent_seeds
        ]
        receipt["seeds_spent_and_must_not_be_reused"] = bool(spent_seeds)
        receipt["evidence_anchors"] = {
            "rows": {
                "path": "rows.jsonl",
                "n_rows": len(rows),
                "logical_sha256": cal._json_hash(rows),
                "file_sha256": cal._sha256(output / "rows.jsonl"),
            },
            "qpos": {
                "path": "qpos.npz",
                "n_arrays": len(qpos_archive),
                "content_sha256": artifact_content_hash["qpos"],
                "file_sha256": cal._sha256(output / "qpos.npz"),
            },
            "features": {
                "path": "features.npz",
                "n_arrays": len(feature_archive),
                "content_sha256": artifact_content_hash["features"],
                "file_sha256": cal._sha256(output / "features.npz"),
            },
            "noise_audit": {
                "path": "noise_audit.json",
                "n_records": len(noise_evidence),
                "logical_sha256": artifact_content_hash["noise"],
                "file_sha256": cal._sha256(output / "noise_audit.json"),
            },
        }
        receipt["wall_clock_s"] = float(time.monotonic() - started)
        cal._write_json(output / "receipt.json", receipt)

    # Required before runner construction or any GPU generation.
    persist()

    try:
        if runner is not None and runner_factory is not None:
            raise ValueError("provide either runner or runner_factory, not both")
        if code.get("dirty") is not False:
            raise ValueError("EXP-023 requires an exactly clean git worktree")
        if not isinstance(code.get("commit"), str) or not code["commit"].strip():
            raise ValueError("EXP-023 requires a concrete git commit")
        if os.environ.get("CHECKPOINTS_DIR"):
            raise ValueError("EXP-023 forbids ambient CHECKPOINTS_DIR")

        source_hashes = dict(source_hashes_fn(repo))
        protocol_sha256 = source_hashes.get(PROTOCOL_PATH)
        if (
            not isinstance(protocol_sha256, str)
            or len(protocol_sha256) != 64
            or any(character not in "0123456789abcdef" for character in protocol_sha256)
        ):
            raise ValueError("EXP-023 protocol content hash is missing or invalid")
        receipt["provenance"]["source_sha256"] = source_hashes
        receipt["provenance"]["protocol"] = {
            "path": PROTOCOL_PATH,
            "sha256": protocol_sha256,
        }
        if runner is None:
            runner = (
                runner_factory() if runner_factory is not None
                else ArdyRunner(cache_path=cache_path)
            )
        if not np.isclose(float(runner.fps), FPS, atol=0.0, rtol=0.0):
            raise ValueError(f"EXP-023 requires runner fps == {FPS:g}")
        if int(runner.noise_stream_version) != NOISE_STREAM_VERSION:
            raise ValueError("EXP-023 requires noise_stream_version == 2")
        if int(runner.model.gen_horizon_len) != HORIZON:
            raise ValueError("EXP-023 requires the Horizon52 checkpoint")
        token = int(runner.model.num_frames_per_token)
        if token <= 0 or HORIZON % token or PADDED_FRAMES % token:
            raise ValueError("EXP-023 checkpoint token size is incompatible with Horizon52")
        if token != EXPECTED_INPUT_HISTORY_FRAMES[1]:
            raise ValueError(
                "EXP-023 locks the released GUI-default four-frame history token"
            )

        generator_identity = dict(generator_identity_validator_fn(
            generator_identity_fn(runner)))
        runtime_identity = dict(runtime_identity_fn())
        physical_identity = dict(physical_identity_fn())
        prompt_identity = dict(prompt_identity_fn(runner, Path(cache_path)))
        pin_validator_fn(generator_identity, runtime_identity, physical_identity)
        receipt["provenance"].update({
            "generator": generator_identity,
            "runtime": runtime_identity,
            "physical_model": physical_identity,
            "walk_step_prompt_cache": prompt_identity,
        })
        body = body or G1Body(None)

        spec = root_only_spec(route)
        channel_usage = {str(name): int(value)
                         for name, value in channel_usage_fn(runner, spec).items()}
        if channel_usage != {"root_pos": 2 * N_FRAMES}:
            raise ValueError(
                "EXP-023 requires exactly dense root-XZ conditioning; observed "
                f"{channel_usage}"
            )
        receipt["campaign_design"]["actual_channel_usage"] = channel_usage

        def revalidate_bound_identities() -> dict[str, Any]:
            current_code = dict(code_state_fn(repo))
            # The output directory is intentionally created after the clean launch-state
            # capture and is normally untracked.  Permit exactly those new evidence files,
            # while rejecting any source/worktree change elsewhere.
            git_check = cal._verify_completion_git_state(
                code, current_code, repo=repo, output=output)
            if dict(source_hashes_fn(repo)) != source_hashes:
                raise ValueError("EXP-023 source content changed during the campaign")
            current_generator = dict(generator_identity_validator_fn(
                generator_identity_fn(runner)))
            current_runtime = dict(runtime_identity_fn())
            current_physical = dict(physical_identity_fn())
            current_prompts = dict(prompt_identity_fn(runner, Path(cache_path)))
            if current_generator != generator_identity:
                raise ValueError("EXP-023 checkpoint identity changed")
            if current_runtime != runtime_identity:
                raise ValueError("EXP-023 ARDY/numerical runtime identity changed")
            if current_physical != physical_identity:
                raise ValueError("EXP-023 G1 physical model identity changed")
            if current_prompts != prompt_identity:
                raise ValueError("EXP-023 cached prompt identity changed")
            pin_validator_fn(current_generator, current_runtime, current_physical)
            if (
                float(runner.fps) != FPS
                or int(runner.noise_stream_version) != NOISE_STREAM_VERSION
                or int(runner.model.gen_horizon_len) != HORIZON
                or int(runner.model.num_frames_per_token)
                != EXPECTED_INPUT_HISTORY_FRAMES[1]
            ):
                raise ValueError("EXP-023 runner contract changed")
            return {
                "git": git_check,
                "sources_unchanged": True,
                "checkpoint_unchanged": True,
                "runtime_unchanged": True,
                "physical_model_unchanged": True,
                "prompt_cache_unchanged": True,
                "runner_contract_unchanged": True,
            }

        chunk_audits: list[
            tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]
        ] = []
        for chunk in chunks:
            chunk_name = str(chunk["name"])
            stage = f"generation_{chunk_name}"
            chunk_rows = list(chunk["rows"])
            local_plan = [
                {
                    **dict(item),
                    "global_row_index": int(item["row_index"]),
                    "row_index": local_index,
                }
                for local_index, item in enumerate(chunk_rows)
            ]
            counters["schedule_invocations_started"] += 1
            counters["trajectories_launched"] += len(chunk_rows)
            spent_seeds.extend(
                seed for seed in chunk["seeds"] if seed not in spent_seeds
            )
            receipt["generation_chunks"][chunk_name].update({
                "status": "running",
                "local_row_plan": local_plan,
            })
            persist()
            try:
                chunk_features, raw_chunk_audit = runner.generate_prompt_schedule(
                    [item["prompt_schedule"] for item in chunk_rows],
                    [spec] * len(chunk_rows),
                    N_FRAMES,
                    DIFFUSION_STEPS,
                    cfg_weight=CFG_WEIGHT,
                    seeds=[int(item["seed"]) for item in chunk_rows],
                    history_frames=token,
                )
            except Exception:
                sample_count_exact = False
                receipt["generation_chunks"][chunk_name]["status"] = (
                    "generation_exception_window_count_unknown"
                )
                persist()
                raise

            exact_chunk = np.asarray(chunk_features)
            counters["schedule_invocations_completed"] += 1
            try:
                observed_window_calls = len(raw_chunk_audit)
            except TypeError:
                observed_window_calls = 0
            counters["autoregressive_window_calls_completed"] += observed_window_calls
            returned_count = int(exact_chunk.shape[0]) if exact_chunk.ndim >= 1 else 0
            counters["trajectories_returned"] += returned_count
            receipt["generation_chunks"][chunk_name].update({
                "status": "returned_unvalidated",
                "trajectories_returned": returned_count,
                "window_calls_returned": observed_window_calls,
            })

            # Archive this chunk before any shape or causal gate. A malformed return still
            # consumes its two fresh seeds and remains durable evidence.
            if exact_chunk.ndim == 0:
                feature_archive[f"{chunk_name}_raw_scalar_return"] = np.asarray(
                    exact_chunk)
            else:
                for local_index in range(returned_count):
                    if local_index < len(chunk_rows):
                        item = chunk_rows[local_index]
                        key = f"s{item['seed']}_{item['arm']}"
                    else:
                        key = f"{chunk_name}_unexpected_returned_row_{local_index}"
                    feature_archive[key] = np.array(exact_chunk[local_index], copy=True)
            artifacts_dirty["features"] = True
            noise_evidence.append({
                "kind": "raw_chunk_runner_audit_before_validation",
                "chunk": chunk_name,
                "seeds": list(chunk["seeds"]),
                "audit": _json_safe(raw_chunk_audit),
            })
            artifacts_dirty["noise"] = True
            persist()

            local_audit = _validate_noise_and_feature_forks(
                exact_chunk, raw_chunk_audit, local_plan)
            if returned_count != len(chunk_rows):
                raise ValueError(f"{chunk_name} returned the wrong trajectory count")
            chunk_audits.append((chunk, raw_chunk_audit))
            receipt["generation_chunks"][chunk_name].update({
                "status": "complete",
                "causal_pairing_audit": local_audit,
                "post_chunk_identity_revalidation": revalidate_bound_identities(),
            })
            persist()

        expected_feature_keys = {
            f"s{item['seed']}_{item['arm']}" for item in plan
        }
        if set(feature_archive) != expected_feature_keys:
            raise ValueError("EXP-023 feature archive does not exactly match its row plan")
        exact_features = np.stack([
            feature_archive[f"s{item['seed']}_{item['arm']}"] for item in plan
        ])
        merged_noise_audit = _merge_chunk_audits(chunk_audits, len(plan))
        fork_audit = _validate_noise_and_feature_forks(
            exact_features, merged_noise_audit, plan)
        receipt["causal_pairing_audit"] = fork_audit
        noise_evidence[:] = _serialized_noise_evidence(merged_noise_audit, plan)
        artifacts_dirty["noise"] = True
        receipt["provenance"]["post_generation_identity_revalidation"] = (
            revalidate_bound_identities()
        )
        persist()

        stage = "decode"
        decoded_count = 0
        for index, sample in enumerate(runner.decode_features(exact_features)):
            if index >= len(plan):
                raise ValueError("EXP-023 decoder returned extra samples")
            item = plan[index]
            key = f"s{item['seed']}_{item['arm']}"
            qpos = np.asarray(runner.to_qpos(sample))
            if (
                qpos.ndim != 2
                or qpos.shape[0] != PADDED_FRAMES
                or not np.isfinite(qpos).all()
            ):
                raise ValueError(f"EXP-023 decoded invalid qpos for {key}: {qpos.shape}")
            # Preserve native decoded precision.  The exact fork gate below must run on the
            # actual decoder result, not a float32 downcast that could erase a small mismatch.
            qpos_archive[key] = np.array(qpos, copy=True)
            artifacts_dirty["qpos"] = True
            counters["trajectories_converted_to_qpos"] += 1
            decoded_count += 1
            persist()
        if decoded_count != len(plan):
            raise ValueError("EXP-023 decoder returned the wrong number of samples")
        receipt["causal_pairing_audit"]["qpos_forks"] = _validate_qpos_forks(
            qpos_archive)
        receipt["causal_pairing_audit"]["qpos_prefixes_exact"] = True
        persist()

        stage = "analysis"
        predicted_centres = {
            onset: float(route[onset + FROZEN_LATENCY_FRAMES, 1])
            for onset in CONTROL_ONSETS
        }
        receipt["campaign_design"]["predicted_box_centres_m"] = {
            str(onset): value for onset, value in predicted_centres.items()
        }
        noise_hash_by_row = {
            int(item["row_index"]): [
                str(merged_noise_audit[window]["row_sha256"][int(item["row_index"])])
                for window in range(N_WINDOWS)
            ]
            for item in plan
        }
        for item in plan:
            seed = int(item["seed"])
            arm = str(item["arm"])
            index = int(item["row_index"])
            key = f"s{seed}_{arm}"
            full_qpos = np.asarray(qpos_archive[key], dtype=float)
            scored_qpos = full_qpos[:N_FRAMES]
            row: dict[str, Any] = {
                **dict(item),
                "archive_key": key,
                "archived_frames": PADDED_FRAMES,
                "scored_frames": N_FRAMES,
                "noise_sha256_by_window": noise_hash_by_row[index],
                "features_sha256": _array_sha256(feature_archive[key], "features"),
                "qpos_sha256": _array_sha256(qpos_archive[key], "qpos"),
                "supporting_motion": _validated_motion_metrics(
                    motion_metrics_fn(body, scored_qpos, route)),
            }
            if arm == "all_walk":
                row["event"] = None
                row["fixed_box"] = None
                row["control_events"] = {
                    str(onset): _validated_event_record(
                        event_detector_fn(body, scored_qpos, route, onset), onset)
                    for onset in CONTROL_ONSETS
                }
                row["control_fixed_boxes"] = {
                    str(onset): _validated_exact_box_record(
                        exact_box_fn(body, scored_qpos, predicted_centres[onset]),
                        predicted_centres[onset])
                    for onset in CONTROL_ONSETS
                }
            else:
                onset = int(ONSETS[arm])
                row["event"] = _validated_event_record(
                    event_detector_fn(body, scored_qpos, route, onset), onset)
                row["fixed_box"] = _validated_exact_box_record(
                    exact_box_fn(body, scored_qpos, predicted_centres[onset]),
                    predicted_centres[onset])
                row["control_events"] = None
                row["control_fixed_boxes"] = None
            rows.append(row)
            counters["trajectories_analyzed"] += 1
            persist()

        if len(rows) != len(plan) or counters["trajectories_analyzed"] != len(plan):
            raise ValueError("EXP-023 analysis did not preserve the planned denominator")
        summary = summarize_rows(rows, route)
        step0_present = int(summary["event_rates_missing_retained"]["step_0"]["present"])
        walk_seed_hits = int(summary["all_walk_specificity"]["seeds_with_any_event"])
        gates = {
            "step0_substrate": {
                "required_min_present": MIN_STEP0_EVENTS,
                "observed_present": step0_present,
                "planned": len(SEEDS),
                "pass": step0_present >= MIN_STEP0_EVENTS,
            },
            "all_walk_specificity": {
                "allowed_max_seeds_with_any_event": MAX_ALL_WALK_SEEDS_WITH_ANY_EVENT,
                "observed_seeds_with_any_event": walk_seed_hits,
                "planned": len(SEEDS),
                "pass": walk_seed_hits <= MAX_ALL_WALK_SEEDS_WITH_ANY_EVENT,
            },
            "delayed_arm_absence_is_an_outcome_not_a_gate": True,
        }
        receipt["summary"] = summary
        receipt["measurement_gates"] = gates
        receipt["provenance"]["post_analysis_identity_revalidation"] = (
            revalidate_bound_identities()
        )
        if not gates["step0_substrate"]["pass"]:
            receipt.update({
                "schema": FAILURE_SCHEMA_VERSION,
                "status": "refused",
                "blocked": True,
                "refusal_reason": "step0_substrate_gate_failed",
                "seeds_spent_and_must_not_be_reused": True,
            })
            persist()
            raise MeasurementRefusal("STEP-from-start substrate gate failed")
        if not gates["all_walk_specificity"]["pass"]:
            receipt.update({
                "schema": FAILURE_SCHEMA_VERSION,
                "status": "refused",
                "blocked": True,
                "refusal_reason": "all_walk_specificity_gate_failed",
                "seeds_spent_and_must_not_be_reused": True,
            })
            persist()
            raise MeasurementRefusal("all-WALK detector-specificity gate failed")

        stage = "complete"
        receipt["provenance"]["completion_identity_revalidation"] = (
            revalidate_bound_identities()
        )
        if any((
            counters["schedule_invocations_completed"] != len(chunks),
            counters["autoregressive_window_calls_completed"] != len(chunks) * N_WINDOWS,
            counters["trajectories_returned"] != len(plan),
            counters["trajectories_converted_to_qpos"] != len(plan),
            counters["trajectories_analyzed"] != len(plan),
        )):
            raise ValueError("EXP-023 completion accounting is not exact")
        receipt.update({
            "status": "complete",
            "complete": True,
            "blocked": False,
            "stage": stage,
            "actual_ardy_samples": len(plan),
        })
        persist()
        return receipt
    except Exception as exc:
        if isinstance(exc, MeasurementRefusal):
            persist()
            raise PromptHandoffAbort(str(exc)) from exc
        receipt.update({
            "schema": FAILURE_SCHEMA_VERSION,
            "status": "blocked",
            "complete": False,
            "blocked": True,
            "failed_stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "actual_ardy_samples": int(counters["trajectories_returned"]),
        })
        persist()
        if isinstance(exc, PromptHandoffAbort):
            raise
        raise PromptHandoffAbort(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/exp023_prompt_handoff")
    parser.add_argument("--cache-path", default="outputs/text_cache.npz")
    args = parser.parse_args(argv)
    receipt = run_campaign(out=args.out, cache_path=args.cache_path)
    summary = receipt["summary"]
    print(json.dumps({
        "status": receipt["status"],
        "actual_ardy_samples": receipt["actual_ardy_samples"],
        "event_rates": summary["event_rates_missing_retained"],
        "slopes": summary["pooled_present_event_slopes_descriptive"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
