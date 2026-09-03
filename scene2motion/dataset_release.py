"""Build and validate the Scene2Motion-DB corpus-pilot release preview.

The pilot is useful only if its labels remain as narrow as the measurements that produced
them.  In particular, overhead target residual and whole-body collision clearance are
different quantities, and the pilot contains reference geometry rather than controller
execution.  This module preserves those distinctions while adding a deterministic split,
content hashes, and a machine-readable schema.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "scene2motion-db-preview-v1"
SOURCE_EXPERIMENT = "corpus_pilot_v2"
OUTCOMES = ("accepted", "accepted_margin", "rejected", "refused")
SPLIT_ORDER = ("train", "validation", "test")


class DatasetValidationError(ValueError):
    """Raised when a source corpus cannot support the advertised preview schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetValidationError(message)


def _finite_number(value: Any, field: str) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)),
             f"{field} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"{field} must be finite")
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetValidationError(
                f"{path}:{line_number} is not valid JSON: {error}"
            ) from error
        _require(isinstance(row, dict), f"{path}:{line_number} must contain an object")
        rows.append(row)
    return rows


def _validate_clip(source: Path, row: dict[str, Any]) -> dict[str, Any]:
    relative = row.get("clip")
    _require(isinstance(relative, str), f"row {row['i']} has no clip path")
    clip_path = Path(relative)
    _require(not clip_path.is_absolute() and ".." not in clip_path.parts,
             f"row {row['i']} clip path escapes the corpus")
    _require(clip_path.parts and clip_path.parts[0] == "clips",
             f"row {row['i']} clip path must live under clips/")
    path = source / clip_path
    _require(path.is_file(), f"row {row['i']} clip is missing: {relative}")

    required_arrays = ("qpos", "route_xy", "overhead", "s_m", "dip_m")
    with np.load(path, allow_pickle=False) as archive:
        _require(set(archive.files) == set(required_arrays),
                 f"row {row['i']} clip arrays are {archive.files}, expected {required_arrays}")
        arrays = {name: np.asarray(archive[name]) for name in required_arrays}

    qpos = arrays["qpos"]
    route_xy = arrays["route_xy"]
    _require(qpos.ndim == 2 and qpos.shape[1] == 36,
             f"row {row['i']} qpos must have shape (T, 36)")
    _require(qpos.shape[0] == int(row.get("n_frames", -1)),
             f"row {row['i']} n_frames does not match qpos")
    _require(route_xy.ndim == 2 and route_xy.shape[1] == 2,
             f"row {row['i']} route_xy must have shape (N, 2)")
    trace_length = arrays["overhead"].shape[0]
    for name in ("overhead", "s_m", "dip_m"):
        _require(arrays[name].ndim == 1 and arrays[name].shape[0] == trace_length,
                 f"row {row['i']} {name} must share the one-dimensional trace length")
    for name, array in arrays.items():
        _require(np.issubdtype(array.dtype, np.floating),
                 f"row {row['i']} {name} must use a floating dtype")
        _require(bool(np.isfinite(array).all()), f"row {row['i']} {name} is not finite")

    return {
        "included": False,
        "source_path": relative,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in arrays.items()
        },
    }


def validate_pilot(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict]]:
    """Validate the committed 300-scene pilot without changing it."""

    source = Path(source)
    receipt_path = source / "receipt.json"
    manifest_path = source / "manifest.jsonl"
    _require(receipt_path.is_file(), f"missing source receipt: {receipt_path}")
    _require(manifest_path.is_file(), f"missing source manifest: {manifest_path}")
    receipt = json.loads(receipt_path.read_text())
    rows = _read_jsonl(manifest_path)

    _require(receipt.get("experiment") == SOURCE_EXPERIMENT,
             f"source experiment must be {SOURCE_EXPERIMENT}")
    _require(receipt.get("n_scenes") == len(rows), "receipt n_scenes does not match rows")
    _require(len(rows) > 0, "source corpus is empty")
    _require([row.get("i") for row in rows] == list(range(len(rows))),
             "row indices must be contiguous and ordered")
    _require(len({row.get("scene_id") for row in rows}) == len(rows),
             "scene_id values must be unique in this pilot")
    _require(len({row.get("seed") for row in rows}) == len(rows),
             "generation seeds must be unique in this pilot")

    observed_counts = Counter(row.get("outcome") for row in rows)
    _require(set(observed_counts) == set(OUTCOMES),
             f"unexpected or missing outcomes: {sorted(observed_counts)}")
    _require(dict(observed_counts) == receipt.get("counts"),
             "receipt outcome counts do not match the manifest")

    clip_metadata: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = row["i"]
        outcome = row["outcome"]
        for key in ("scene_id", "seed", "scene", "preference", "target_m", "proposer",
                    "max_repairs", "noise_stream_version", "cache_version"):
            _require(key in row, f"row {index} is missing {key}")
        _require(row["noise_stream_version"] == receipt.get("noise_stream_version"),
                 f"row {index} noise-stream version differs from the receipt")
        _require(row["cache_version"] == receipt.get("cache_version"),
                 f"row {index} cache version differs from the receipt")
        scene = row["scene"]
        _require(isinstance(scene, dict), f"row {index} scene must be an object")
        for field in ("beam_height", "beam_width", "gap"):
            _require(_finite_number(scene.get(field), f"row {index} scene.{field}") > 0,
                     f"row {index} scene.{field} must be positive")
        _require(isinstance(scene.get("n_beams"), int) and scene["n_beams"] > 0,
                 f"row {index} scene.n_beams must be a positive integer")
        target = _finite_number(row["target_m"], f"row {index} target_m")
        _require(target > 0, f"row {index} target_m must be positive")

        if outcome == "refused":
            refusal = row.get("refusal")
            _require(isinstance(refusal, dict), f"row {index} refusal must be an object")
            _require(refusal.get("functional") in {"overhead_clearance", "lateral_clearance"},
                     f"row {index} has an unknown refusal function")
            _require(_finite_number(refusal.get("deficit_m"),
                                    f"row {index} refusal.deficit_m") > 0,
                     f"row {index} refusal deficit must be positive")
            _require("clip" not in row, f"row {index} refused before a clip but names one")
            continue

        for key in ("collision_free", "min_overhead_m", "min_clearance_m", "n_attempts",
                    "repaired", "final_schedule_hash"):
            _require(key in row, f"row {index} is missing {key}")
        overhead = _finite_number(row["min_overhead_m"], f"row {index} min_overhead_m")
        clearance = _finite_number(row["min_clearance_m"], f"row {index} min_clearance_m")
        if outcome == "accepted":
            _require(row["collision_free"] is True and overhead >= target,
                     f"row {index} accepted semantics are inconsistent")
        elif outcome == "accepted_margin":
            _require(row["collision_free"] is True and 0 <= clearance and overhead < target,
                     f"row {index} accepted_margin semantics are inconsistent")
        else:
            _require(outcome == "rejected" and row["collision_free"] is False and clearance < 0,
                     f"row {index} rejected semantics are inconsistent")

        if outcome in {"accepted", "accepted_margin"}:
            clip_metadata[index] = _validate_clip(source, row)
        else:
            _require("clip" not in row, f"row {index} rejected row must not advertise a clip")

    _require(len(clip_metadata) == observed_counts["accepted"] + observed_counts["accepted_margin"],
             "clip count does not match collision-free records")
    return receipt, rows, clip_metadata


def _split_assignments(rows: Iterable[dict[str, Any]]) -> dict[int, str]:
    """Outcome-stratified, scene-identity-disjoint 70/15/15 development split."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["outcome"]].append(row)
    assignments: dict[int, str] = {}
    for outcome in OUTCOMES:
        group = sorted(
            groups[outcome],
            key=lambda row: (
                hashlib.sha256(row["scene_id"].encode()).hexdigest(), row["scene_id"]
            ),
        )
        n = len(group)
        n_validation = max(1, round(0.15 * n))
        n_test = max(1, round(0.15 * n))
        _require(n_validation + n_test < n,
                 f"outcome {outcome} is too small for the declared three-way split")
        cut_train = n - n_validation - n_test
        cut_validation = n - n_test
        for position, row in enumerate(group):
            split = ("train" if position < cut_train else
                     "validation" if position < cut_validation else "test")
            assignments[row["i"]] = split
    return assignments


def _failure_mode(row: dict[str, Any]) -> str | None:
    outcome = row["outcome"]
    if outcome == "accepted":
        return None
    if outcome == "accepted_margin":
        return "overhead_target_shortfall"
    if outcome == "rejected":
        return "residual_collision_after_repair_budget"
    return f"scene_{row['refusal']['functional']}_unreachable"


def _normalise_record(row: dict[str, Any], split: str,
                      clip: dict[str, Any] | None) -> dict[str, Any]:
    outcome = row["outcome"]
    target = float(row["target_m"])
    if outcome == "refused":
        target_residual = -float(row["refusal"]["deficit_m"])
        contact_clearance = None
        penetration = None
        collision_free = None
        evidence_tier = "scene_feasibility"
    else:
        target_residual = float(row["min_overhead_m"]) - target
        contact_clearance = float(row["min_clearance_m"])
        penetration = max(0.0, -contact_clearance)
        collision_free = bool(row["collision_free"])
        evidence_tier = "reference_geometry"

    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": f"{SOURCE_EXPERIMENT}:{row['i']:04d}",
        "split": split,
        "source": {"experiment": SOURCE_EXPERIMENT, "row_index": row["i"]},
        "identity": {
            "scene_id": row["scene_id"],
            "generation_seed": row["seed"],
            "physics_seed": None,
            "noise_stream_version": row["noise_stream_version"],
            "cache_version": row["cache_version"],
        },
        "scene": row["scene"],
        "request": {
            "preference": row["preference"],
            "proposer": row["proposer"],
            "target_overhead_margin_m": target,
            "max_repairs": row["max_repairs"],
        },
        "label": {
            "evidence_tier": evidence_tier,
            "outcome": outcome,
            "failure_mode": _failure_mode(row),
            "collision_free_reference": collision_free,
            "signed_overhead_target_residual_m": target_residual,
            "signed_whole_body_clearance_m": contact_clearance,
            "whole_body_penetration_depth_m": penetration,
            "controller_execution": "not_measured",
        },
        "repair": {
            "attempts": row.get("n_attempts"),
            "repaired": row.get("repaired"),
            "final_schedule_hash": row.get("final_schedule_hash"),
        },
        "refusal": row.get("refusal"),
        "artifact": clip,
    }


def _schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://scene2motion.local/schema/{SCHEMA_VERSION}.json",
        "title": "Scene2Motion-DB corpus-pilot release-preview record",
        "type": "object",
        "required": [
            "schema_version", "record_id", "split", "source", "identity", "scene",
            "request", "label", "repair", "refusal", "artifact",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "record_id": {"type": "string"},
            "split": {"enum": list(SPLIT_ORDER)},
            "source": {"type": "object"},
            "identity": {"type": "object"},
            "scene": {"type": "object"},
            "request": {"type": "object"},
            "label": {
                "type": "object",
                "required": [
                    "evidence_tier", "outcome", "failure_mode",
                    "collision_free_reference", "signed_overhead_target_residual_m",
                    "signed_whole_body_clearance_m", "whole_body_penetration_depth_m",
                    "controller_execution",
                ],
            },
            "repair": {"type": "object"},
            "refusal": {"type": ["object", "null"]},
            "artifact": {"type": ["object", "null"]},
        },
    }


def _readme(include_clips: bool) -> str:
    payload = "included under `clips/`" if include_clips else "not copied; source paths and hashes only"
    return f"""# Scene2Motion-DB corpus-pilot release preview

This preview contains 300 randomized ducking scenes generated with one heuristic proposer and
at most two reference-space repairs. It packages the committed `corpus_pilot_v2` evidence; it
does not upgrade that evidence.

- Evidence tier: scene feasibility for refusals; reference geometry for generated motions.
- Controller execution: not measured for every row in this preview.
- Motion payload: {payload}.
- Split: deterministic, outcome-stratified 70/15/15 development split with disjoint scene IDs.
  This split is not a geometry-OOD test and cannot support a generalization claim.
- Signed labels: positive `signed_overhead_target_residual_m` meets the requested overhead
  margin; negative values are shortfalls. `signed_whole_body_clearance_m` is a separate contact
  quantity, where negative values denote reference penetration.

The preview is not yet a public redistribution. The repository currently supplies no dataset
license or third-party motion-output redistribution determination. Resolve those terms before
publishing clip payloads. A public benchmark additionally requires an execution-labelled tier,
an OOD scene split fixed before evaluation, and a downstream utility demonstration.
"""


def build_preview(source: Path, output: Path, *, include_clips: bool = False) -> dict[str, Any]:
    """Validate ``source`` and atomically build a release preview at a new path."""

    source = Path(source).resolve()
    output = Path(output).resolve()
    _require(not output.exists(), f"output already exists: {output}")
    receipt, rows, clips = validate_pilot(source)
    assignments = _split_assignments(rows)
    normalised = [
        _normalise_record(row, assignments[row["i"]], clips.get(row["i"])) for row in rows
    ]

    split_counts = {
        split: dict(Counter(record["label"]["outcome"] for record in normalised
                            if record["split"] == split))
        for split in SPLIT_ORDER
    }
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        if include_clips:
            (temp / "clips").mkdir()
            for record in normalised:
                artifact = record["artifact"]
                if artifact is None:
                    continue
                source_clip = source / artifact["source_path"]
                target_clip = temp / artifact["source_path"]
                shutil.copy2(source_clip, target_clip)
                artifact["included"] = True
                artifact["path"] = artifact.pop("source_path")

        records_path = temp / "records.jsonl"
        records_path.write_text("".join(_canonical_json(record) + "\n" for record in normalised))
        (temp / "schema.json").write_text(json.dumps(_schema(), indent=2, sort_keys=True) + "\n")
        splits = {
            "scheme": "outcome_stratified_scene_hash_v1",
            "warning": "development split only; not a geometry-OOD evaluation",
            "scene_identity_disjoint": True,
            "members": {
                split: [record["record_id"] for record in normalised if record["split"] == split]
                for split in SPLIT_ORDER
            },
        }
        (temp / "splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n")
        (temp / "README.md").write_text(_readme(include_clips))

        payload_files = sorted(path for path in temp.rglob("*") if path.is_file())
        output_hashes = {str(path.relative_to(temp)): _sha256(path) for path in payload_files}
        preview_receipt = {
            "schema_version": SCHEMA_VERSION,
            "release_status": "internal_preview_redistribution_review_required",
            "source": {
                "experiment": receipt["experiment"],
                "receipt_sha256": _sha256(source / "receipt.json"),
                "manifest_sha256": _sha256(source / "manifest.jsonl"),
            },
            "n_records": len(normalised),
            "n_motion_payloads": len(clips),
            "motion_payloads_included": include_clips,
            "outcome_counts": dict(Counter(record["label"]["outcome"] for record in normalised)),
            "split_outcome_counts": split_counts,
            "evidence_tier_counts": dict(Counter(
                record["label"]["evidence_tier"] for record in normalised
            )),
            "controller_execution_counts": {"not_measured": len(normalised)},
            "output_sha256": output_hashes,
        }
        (temp / "receipt.json").write_text(
            json.dumps(preview_receipt, indent=2, sort_keys=True) + "\n"
        )
        temp.replace(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return preview_receipt
