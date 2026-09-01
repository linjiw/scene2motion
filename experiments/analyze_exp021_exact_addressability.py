"""Post-hoc exact fixed-obstacle replay of the completed EXP-021 archive.

The original EXP-021 selection analysis answered a tolerance question: whether a clip
cleared *some* box centre within a radius of a target.  That is useful as an addressable-
window diagnostic, but it is not the project's obstacle-centred endpoint.  This script
replays the archived qpos against one box whose centre is fixed at x=1.2 m and reports
exact-height binary outcomes plus the conservative BoxHeightProbe lower-bound diagnostic.

Both the 1.2 m location and every budget derived here are post-hoc and descriptive.  They
must not be presented as a preregistered or fresh-seed validation.  By default the script
is read-only and prints JSON.  It writes only when ``--write`` or ``--out`` is explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

# Allow direct execution from ``experiments/`` as documented by ``--help``.  Pytest imports
# this module from the repository root, which previously hid the missing CLI path setup.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_ARCHIVE = Path("outputs/exp021_elicited_lift_distribution_v2")
DEFAULT_OUTPUT_NAME = "exact_fixed_obstacle_posthoc.json"
SCHEMA_VERSION = "exp021-exact-fixed-obstacle-posthoc-v1"

# The correction is about one already-completed archive, not an arbitrary receipt that
# happens to use the same field names.  Bind the immutable files and historical source
# commit explicitly; the original EXP-021 receipt did not itself bind ARDY/checkpoint/runtime
# identities, so this is a retrospective archive validation rather than a repair of that
# missing provenance.
LOCKED_ARCHIVE_MANIFEST: Mapping[str, Any] = {
    "experiment": "exp021_elicited_lift_distribution",
    "schema": "exp021-elicited-lift-distribution-v1",
    "source_commit": "cde0246fc89d45cb71f23fa6b8b3d8729ec00e6b",
    "seeds": tuple(range(4400, 4464)),
    "files": {
        "receipt.json": "0c53d8c5dc2bdfa587f8c0b35d069fcd677f1cdc30221b5ce1afa70d1a5ccf7e",
        "rows.jsonl": "1d8cc57df2494bd7179940bfe57325ac922f3f41e2581fcc7cb789b5e0c28f71",
        "qpos.npz": "2a4b34479aa24894b854301d91bafe1ad870dc530b70eed5b6703eb02c284687",
    },
    "qpos_content_sha256": (
        "50c27a6b9d61433a1562d1e704d453b2c907ba9d719f7c23d36acb08a8f61ed0"
    ),
}

LOCKED_SCORING_FILES: Mapping[str, str] = {
    "scene2motion/stepover_eval.py": (
        "f0e344def2133f3a45935bb1b7ffad8fb2591145557e6b136d50871827cf77b4"
    ),
    "scene2motion/robot.py": (
        "c12b59a999969dd910e0aaef25322695d247bd392899394dd5d35cfc1252fc82"
    ),
}
LOCKED_G1_XML_SHA256 = (
    "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
)

# Locked to the location selected post hoc in the historical EXP-021 write-up.  Keeping
# these out of the CLI prevents this correction from becoming another target search.
LOCKED_OBSTACLE_X_M = 1.2
LOCKED_OBSTACLE_DEPTH_M = 0.20
GRADED_HEIGHTS_M = (0.03, 0.05, 0.08, 0.12, 0.20, 0.30)
BEST_OF_N = (1, 2, 4, 8, 16, 32)
SEQUENTIAL_BLOCK_SIZE = 8


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Match the content hash used by the EXP-021 receipt."""
    digest = hashlib.sha256()
    if not arrays:
        raise ValueError("array payload is empty")
    for name in sorted(arrays):
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"qpos entry {name!r} is not an ndarray")
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(_canonical_json(list(array.shape)).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial rate."""
    if total < 1:
        raise ValueError("Wilson interval requires a positive total")
    if not 0 <= successes <= total:
        raise ValueError("successes must lie in [0, total]")
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = (phat + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        phat * (1.0 - phat) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def independent_best_of_n(rate: float, n: int) -> float:
    """Plug-in probability of at least one success in ``n`` independent calls."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must lie in [0, 1]")
    if n < 1:
        raise ValueError("n must be positive")
    return 1.0 - (1.0 - rate) ** n


def calls_for_target(rate: float, target: float) -> int | None:
    """Smallest independent call count reaching ``target``; None if rate is zero."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must lie in [0, 1]")
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie strictly between 0 and 1")
    if rate == 0.0:
        return None
    if rate == 1.0:
        return 1
    return int(math.ceil(math.log1p(-target) / math.log1p(-rate)))


def summarize_clearances(
    clearances_m: Sequence[float],
    seeds: Sequence[int],
    *,
    heights_m: Sequence[float] = GRADED_HEIGHTS_M,
    best_of_n: Sequence[int] = BEST_OF_N,
    block_size: int = SEQUENTIAL_BLOCK_SIZE,
    exact_hits_by_height: Mapping[float, Sequence[bool]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Summarize exact fixed-box outcomes and disjoint sequential blocks.

    ``BoxHeightProbe.probe`` is a 5 mm-resolution conservative lower bound.  A binary
    statement such as "clears an 8 cm box" must instead use ``BoxHeightProbe.clears`` at
    exactly 8 cm.  ``exact_hits_by_height`` carries those booleans; omitting it is kept for
    pure helper tests and summarizes the lower bound only.
    """
    clearances = np.asarray(clearances_m, dtype=float)
    seed_values = [int(seed) for seed in seeds]
    if clearances.ndim != 1 or not len(clearances):
        raise ValueError("clearances must be a non-empty 1-D sequence")
    if len(seed_values) != len(clearances):
        raise ValueError("seeds and clearances must align")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be unique")
    if not np.isfinite(clearances).all() or np.any(clearances < 0.0):
        raise ValueError("clearances must be finite and non-negative")
    heights = [float(height) for height in heights_m]
    ns = [int(n) for n in best_of_n]
    if not heights or any(height <= 0.0 for height in heights):
        raise ValueError("graded heights must be positive")
    if len(set(heights)) != len(heights):
        raise ValueError("graded heights must be unique")
    if not ns or any(n < 1 for n in ns) or len(set(ns)) != len(ns):
        raise ValueError("best-of-N values must be unique positive integers")
    if block_size < 1 or len(clearances) % block_size:
        raise ValueError("clearance count must divide exactly into sequential blocks")

    summary: dict[str, dict[str, Any]] = {}
    total = len(clearances)
    for height in heights:
        if exact_hits_by_height is None:
            hits = clearances >= height
            outcome_source = "probe_lower_bound_gte_height"
        else:
            if height not in exact_hits_by_height:
                raise ValueError(f"missing exact hit vector for height {height}")
            hits = np.asarray(exact_hits_by_height[height], dtype=bool)
            if hits.shape != (total,):
                raise ValueError(
                    f"exact hit vector for height {height} has shape {hits.shape}, "
                    f"expected ({total},)"
                )
            outcome_source = "BoxHeightProbe.clears_at_exact_height"
        successes = int(hits.sum())
        rate = successes / total
        low, high = wilson_interval(successes, total)
        blocks = []
        for start in range(0, total, block_size):
            block_hits = hits[start:start + block_size]
            first = np.flatnonzero(block_hits)
            success = bool(first.size)
            first_success_call = int(first[0] + 1) if success else None
            blocks.append({
                "block_index": start // block_size,
                "seeds": seed_values[start:start + block_size],
                "success": success,
                "first_success_call": first_success_call,
                "calls_spent_stop_on_success": first_success_call or block_size,
            })
        block_successes = sum(block["success"] for block in blocks)
        block_low, block_high = wilson_interval(block_successes, len(blocks))
        summary[f"h={height:g}"] = {
            "height_m": height,
            "binary_outcome_source": outcome_source,
            "successes": successes,
            "total": total,
            "per_clip_rate": rate,
            "wilson95": [low, high],
            "independent_plugin_best_of_n": {
                f"N={n}": independent_best_of_n(rate, n) for n in ns
            },
            "independent_plugin_n90": calls_for_target(rate, 0.90),
            "independent_plugin_n95": calls_for_target(rate, 0.95),
            "sequential_n8": {
                "block_size": block_size,
                "n_blocks": len(blocks),
                "successful_blocks": block_successes,
                "empirical_success_rate": block_successes / len(blocks),
                "wilson95": [block_low, block_high],
                "mean_calls_spent_stop_on_success": float(np.mean([
                    block["calls_spent_stop_on_success"] for block in blocks
                ])),
                "blocks": blocks,
            },
        }
    return summary


def _load_validated_archive(
    archive: Path,
    locked_manifest: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[int], dict[str, np.ndarray], dict[str, str]]:
    receipt_path = archive / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"missing EXP-021 receipt: {receipt_path}")
    locked_file_hashes: dict[str, str] = {}
    if locked_manifest is not None:
        files = locked_manifest.get("files")
        if not isinstance(files, Mapping) or not files:
            raise ValueError("locked archive manifest has no file hashes")
        for relative, expected in files.items():
            path = archive / str(relative)
            if not path.is_file():
                raise ValueError(f"locked EXP-021 artifact is missing: {path}")
            actual = file_sha256(path)
            if actual != expected:
                raise ValueError(
                    f"locked EXP-021 artifact hash mismatch for {relative}: "
                    f"expected={expected}, actual={actual}"
                )
            locked_file_hashes[str(relative)] = actual
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "exp021-elicited-lift-distribution-v1":
        raise ValueError("archive is not the EXP-021 v1 distribution schema")
    if receipt.get("experiment") != "exp021_elicited_lift_distribution":
        raise ValueError("archive is not the EXP-021 elicited-lift experiment")
    if receipt.get("status") != "complete" or receipt.get("complete") is not True:
        raise ValueError("EXP-021 receipt is not complete")
    if receipt.get("blocked") is not False:
        raise ValueError("EXP-021 receipt is marked blocked or lacks blocked=false")
    seeds = receipt.get("design", {}).get("pool_seeds")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("receipt lacks an integer design.pool_seeds list")
    if len(set(seeds)) != len(seeds):
        raise ValueError("receipt pool seeds are not unique")
    if locked_manifest is not None:
        if receipt.get("schema") != locked_manifest.get("schema"):
            raise ValueError("EXP-021 schema differs from the locked archive manifest")
        if receipt.get("experiment") != locked_manifest.get("experiment"):
            raise ValueError("EXP-021 experiment differs from the locked archive manifest")
        if tuple(seeds) != tuple(locked_manifest.get("seeds", ())):
            raise ValueError("EXP-021 seed order differs from the locked archive manifest")
        if (
            receipt.get("provenance", {}).get("code", {}).get("commit")
            != locked_manifest.get("source_commit")
        ):
            raise ValueError("EXP-021 source commit differs from the locked archive manifest")

    qpos_anchor = receipt.get("evidence_anchors", {}).get("qpos", {})
    relative_qpos_path = qpos_anchor.get("path")
    if not isinstance(relative_qpos_path, str) or not relative_qpos_path:
        raise ValueError("receipt lacks evidence_anchors.qpos.path")
    qpos_path = archive / relative_qpos_path
    if not qpos_path.is_file():
        raise ValueError(f"missing EXP-021 qpos archive: {qpos_path}")
    with np.load(qpos_path, allow_pickle=False) as payload:
        arrays = {name: np.array(payload[name], copy=True) for name in payload.files}

    expected_keys = [f"s{seed}" for seed in seeds]
    if set(arrays) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(arrays))
        extra = sorted(set(arrays) - set(expected_keys))
        raise ValueError(f"qpos keys do not match pool seeds: missing={missing}, extra={extra}")
    for key, value in arrays.items():
        if value.ndim != 2 or not value.size or not np.isfinite(value).all():
            raise ValueError(f"qpos array {key} must be a non-empty finite 2-D array")
        if locked_manifest is not None and value.shape != (200, 36):
            raise ValueError(
                f"locked EXP-021 qpos {key} has shape {value.shape}, expected (200, 36)"
            )

    expected_count = len(expected_keys)
    count_claims = {
        "actual_ardy_samples": receipt.get("actual_ardy_samples"),
        "summary.n_clips": receipt.get("summary", {}).get("n_clips"),
        "evidence_anchors.qpos.n_arrays": qpos_anchor.get("n_arrays"),
    }
    wrong_counts = {name: value for name, value in count_claims.items()
                    if value != expected_count}
    if wrong_counts:
        raise ValueError(
            f"receipt qpos/sample counts do not equal {expected_count}: {wrong_counts}")
    if locked_manifest is not None:
        accounting = receipt.get("query_accounting", {})
        if accounting != {
            "generate_invocations": 8,
            "samples_launched": 64,
            "samples_returned": 64,
        }:
            raise ValueError("EXP-021 query accounting differs from the locked campaign")
        rows_path = archive / "rows.jsonl"
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
        if (
            len(rows) != 64
            or [row.get("seed") for row in rows] != list(locked_manifest["seeds"])
        ):
            raise ValueError("EXP-021 rows do not match the locked seed order")

    actual_hash = array_content_sha256(arrays)
    expected_hash = qpos_anchor.get("content_sha256")
    if actual_hash != expected_hash:
        raise ValueError(
            f"qpos content hash mismatch: receipt={expected_hash}, actual={actual_hash}")
    if (
        locked_manifest is not None
        and actual_hash != locked_manifest.get("qpos_content_sha256")
    ):
        raise ValueError("qpos content differs from the locked archive manifest")
    return receipt, seeds, arrays, locked_file_hashes


def _locked_scoring_identity() -> dict[str, Any]:
    """Validate the exact collision scorer/model used by this post-hoc correction."""
    import mujoco
    from scene2motion.robot import ARDY_G1_XML, BODY_MARGIN

    repo = Path(__file__).resolve().parents[1]
    source_hashes: dict[str, str] = {}
    for relative, expected in LOCKED_SCORING_FILES.items():
        actual = file_sha256(repo / relative)
        if actual != expected:
            raise ValueError(
                f"locked scoring source hash mismatch for {relative}: "
                f"expected={expected}, actual={actual}"
            )
        source_hashes[relative] = actual
    xml_path = Path(ARDY_G1_XML)
    xml_sha256 = file_sha256(xml_path)
    if xml_sha256 != LOCKED_G1_XML_SHA256:
        raise ValueError(
            "released G1 XML differs from the model locked for the exact replay"
        )
    if float(BODY_MARGIN) != 0.04:
        raise ValueError("exact replay requires the locked 4 cm G1 body margin")
    return {
        "injected_probe": False,
        "source_sha256": source_hashes,
        "g1_xml_path": str(xml_path),
        "g1_xml_sha256": xml_sha256,
        "body_margin_m": float(BODY_MARGIN),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
    }


def analyze_archive(
    archive: str | Path = DEFAULT_ARCHIVE,
    *,
    probe_factory: Callable[[float, float], Any] | None = None,
    locked_manifest: Mapping[str, Any] | None = LOCKED_ARCHIVE_MANIFEST,
) -> dict[str, Any]:
    """Validate and replay one completed EXP-021 archive without generation."""
    archive_path = Path(archive)
    receipt, seeds, arrays, locked_file_hashes = _load_validated_archive(
        archive_path, locked_manifest)
    if probe_factory is None:
        # Lazy import keeps the pure statistical helpers independently testable.
        from scene2motion.stepover_eval import BoxHeightProbe
        probe_factory = BoxHeightProbe
        scoring_identity = _locked_scoring_identity()
    else:
        scoring_identity = {
            "injected_probe": True,
            "evidentiary_scope": "test/diagnostic only; not the locked collision scorer",
        }
    probe = probe_factory(LOCKED_OBSTACLE_X_M, LOCKED_OBSTACLE_DEPTH_M)
    clearances = [float(probe.probe(arrays[f"s{seed}"])) for seed in seeds]
    if not np.isfinite(clearances).all() or any(value < 0.0 for value in clearances):
        raise ValueError("probe returned invalid clearance values")
    if not hasattr(probe, "clears"):
        raise ValueError("fixed-obstacle probe must implement clears(qpos, height)")
    exact_hits = {
        float(height): [
            bool(probe.clears(arrays[f"s{seed}"], float(height))) for seed in seeds
        ]
        for height in GRADED_HEIGHTS_M
    }
    summary = summarize_clearances(
        clearances, seeds, exact_hits_by_height=exact_hits)
    qpos_anchor = receipt["evidence_anchors"]["qpos"]
    return {
        "schema": SCHEMA_VERSION,
        "status": "complete_post_hoc_replay",
        "evidentiary_scope": {
            "stage": "kinematic archive replay; zero new generator samples; no tracker",
            "target_status": (
                "POST HOC: x=1.2 m was selected after inspecting the same EXP-021 archive"
            ),
            "budget_status": (
                "POST HOC plug-in estimates; not preregistered and not fresh-seed validated"
            ),
            "endpoint": (
                "BoxHeightProbe.clears at each exact fixed height for binary outcomes, "
                "plus the conservative probe() lower bound as a continuous diagnostic; "
                "no spatial tolerance or neighboring box-centre search"
            ),
        },
        "locked_obstacle": {
            "x_m": LOCKED_OBSTACLE_X_M,
            "depth_m": LOCKED_OBSTACLE_DEPTH_M,
            "graded_heights_m": list(GRADED_HEIGHTS_M),
        },
        "source": {
            "archive": str(archive_path),
            "experiment": receipt["experiment"],
            "receipt_status": receipt["status"],
            "receipt_schema": receipt["schema"],
            "qpos_path": qpos_anchor["path"],
            "qpos_arrays": len(arrays),
            "qpos_content_sha256": qpos_anchor["content_sha256"],
            "locked_file_sha256": locked_file_hashes,
            "historical_provenance_limit": (
                "EXP-021 did not contemporaneously bind ARDY, checkpoint, prompt-cache, "
                "or numerical-runtime identities; an exact-batch live diagnostic was "
                "performed retrospectively without a committed replay receipt and is "
                "therefore non-citable and does not repair that omission"
            ),
            "verified": {
                "receipt_complete": True,
                "qpos_keys_match_pool_seeds": True,
                "qpos_count_claims_match": True,
                "qpos_content_sha256_matches": True,
                "locked_archive_manifest_matches": locked_manifest is not None,
            },
            "pool_seeds": seeds,
        },
        "scoring_identity": scoring_identity,
        "per_seed_clearance_lower_bound_m": {
            str(seed): clearance for seed, clearance in zip(seeds, clearances)
        },
        "per_seed_exact_clears": {
            str(seed): {
                f"h={height:g}": bool(exact_hits[float(height)][index])
                for height in GRADED_HEIGHTS_M
            }
            for index, seed in enumerate(seeds)
        },
        "summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--write", action="store_true",
        help=f"write to <archive>/{DEFAULT_OUTPUT_NAME}; default is print-only",
    )
    parser.add_argument(
        "--out", type=Path,
        help="explicit output path; implies --write",
    )
    args = parser.parse_args(argv)
    result = analyze_archive(args.archive)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination = args.out
    if args.write and destination is None:
        destination = args.archive / DEFAULT_OUTPUT_NAME
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded)
        print(f"wrote {destination}", file=sys.stderr)
    print(encoded, end="")


if __name__ == "__main__":
    main()
