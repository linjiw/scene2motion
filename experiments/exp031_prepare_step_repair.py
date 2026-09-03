"""Prepare the frozen EXP-031 constructive step-repair candidates.

This CPU stage reads all 64 archived EXP-021 references, applies the frozen support screen and
the obstacle-relative leg IK, and writes one row for every assigned reference.  Only admitted
repaired references are stored in ``qpos.npz`` for the later obstacle-present SONIC driver.

It does not launch a controller.  A dry run performs the deterministic computation and reports
the planned candidate keys without writing.  A real preparation refuses unless the EXP-031
protocol says ``preregistered``, the repository is clean, the source artifacts match their
locked hashes, and the output directory is empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scene2motion.robot import ARDY_G1_XML, G1Body  # noqa: E402
from scene2motion.step_repair import (  # noqa: E402
    FootstepRepairConfig,
    SupportRule,
    repair_step_reference,
)
from scene2motion.stepover_eval import step_scene  # noqa: E402


PROTOCOL = ROOT / "docs/ramp-exp031-constructive-step-repair-protocol.md"
SOURCE = ROOT / "outputs/exp021_elicited_lift_distribution_v2"
THRESHOLD_RECEIPT = ROOT / "outputs/exp016_threshold_calibration/receipt.json"
DEFAULT_OUT = ROOT / "outputs/exp031_constructive_step_repair"

SOURCE_QPOS_SHA256 = "2a4b34479aa24894b854301d91bafe1ad870dc530b70eed5b6703eb02c284687"
SOURCE_ROWS_SHA256 = "1d8cc57df2494bd7179940bfe57325ac922f3f41e2581fcc7cb789b5e0c28f71"
THRESHOLD_RECEIPT_SHA256 = "f6dba8be84a9d5d0b76c8114d4b93b1707bc1bb8a6fec1a26a22aa1780a6e9bf"

FPS = 25.0
OBSTACLE_X_M = 1.2
OBSTACLE_HEIGHT_M = 0.05
OBSTACLE_DEPTH_M = 0.20
POOL_SEEDS = tuple(range(4400, 4464))
EXPECTED_SUPPORT_PASSING_KEYS = (
    "s4408", "s4411", "s4418", "s4419", "s4434", "s4440", "s4452", "s4459",
)
EXPECTED_ACCEPTED_KEYS = ("s4408", "s4411", "s4434", "s4459")
SOURCE_FILES = (
    "experiments/exp031_prepare_step_repair.py",
    "scene2motion/robot.py",
    "scene2motion/step_repair.py",
    "scene2motion/stepover_eval.py",
)


class PreparationRefused(RuntimeError):
    """A fail-closed preparation refusal; existing evidence is never overwritten."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qpos_sha256(qpos: np.ndarray) -> str:
    array = np.ascontiguousarray(qpos)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def git_state(repo: str | Path = ROOT) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def protocol_identity(path: str | Path = PROTOCOL) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    status_line = next((line for line in text.splitlines() if line.startswith("**Status:**")), "")
    status = "preregistered" if "preregistered" in status_line.lower() \
        and "draft" not in status_line.lower() else "draft"
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "status": status}


def support_rule(path: str | Path = THRESHOLD_RECEIPT) -> SupportRule:
    path = Path(path)
    observed = sha256_file(path)
    if observed != THRESHOLD_RECEIPT_SHA256:
        raise PreparationRefused(
            f"threshold receipt hash mismatch: {observed} != {THRESHOLD_RECEIPT_SHA256}"
        )
    values = json.loads(path.read_text())["stepover_thresholds"]
    return SupportRule(
        support_height_m=float(values["support_height_m"]),
        support_speed_mps=float(values["support_speed_mps"]),
        max_unsupported_run_s=float(values["max_unsupported_run_s"]),
    )


def source_identity(source: str | Path = SOURCE) -> dict[str, Any]:
    source = Path(source)
    qpos_path = source / "qpos.npz"
    rows_path = source / "rows.jsonl"
    observed_qpos = sha256_file(qpos_path)
    observed_rows = sha256_file(rows_path)
    if observed_qpos != SOURCE_QPOS_SHA256:
        raise PreparationRefused(
            f"source qpos hash mismatch: {observed_qpos} != {SOURCE_QPOS_SHA256}"
        )
    if observed_rows != SOURCE_ROWS_SHA256:
        raise PreparationRefused(
            f"source rows hash mismatch: {observed_rows} != {SOURCE_ROWS_SHA256}"
        )
    return {
        "directory": str(source.resolve()),
        "qpos": {"path": str(qpos_path.resolve()), "sha256": observed_qpos},
        "rows": {"path": str(rows_path.resolve()), "sha256": observed_rows},
    }


def project_identity(state: Mapping[str, Any]) -> dict[str, Any]:
    files = {relative: sha256_file(ROOT / relative) for relative in SOURCE_FILES}
    return {
        "git": dict(state),
        "source_files": files,
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "g1_xml": {"path": str(ARDY_G1_XML), "sha256": sha256_file(ARDY_G1_XML)},
    }


def build_records(source: str | Path = SOURCE, *,
                  rule: SupportRule | None = None,
                  config: FootstepRepairConfig | None = None,
                  ) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    """Compute all 64 dispositions and the admitted float32 candidate archive."""

    source = Path(source)
    rule = rule or support_rule()
    config = config or FootstepRepairConfig()
    body = G1Body(None)
    obstacle_body = G1Body(step_scene(OBSTACLE_X_M, OBSTACLE_HEIGHT_M, OBSTACLE_DEPTH_M))
    source_rows = {
        f"s{int(row['seed'])}": row
        for row in map(json.loads, (source / "rows.jsonl").read_text().splitlines())
    }
    expected_keys = {f"s{seed}" for seed in POOL_SEEDS}
    if set(source_rows) != expected_keys:
        raise PreparationRefused(
            f"source rows cover {len(source_rows)} keys, expected {len(expected_keys)}"
        )

    records: list[dict[str, Any]] = []
    candidates: dict[str, np.ndarray] = {}
    with np.load(source / "qpos.npz") as archive:
        if set(archive.files) != expected_keys:
            raise PreparationRefused(
                f"source qpos covers {len(archive.files)} keys, expected {len(expected_keys)}"
            )
        for seed in POOL_SEEDS:
            key = f"s{seed}"
            original = np.asarray(archive[key], dtype=float)
            result = repair_step_reference(
                original, fps=FPS, obstacle_x_m=OBSTACLE_X_M,
                obstacle_height_m=OBSTACLE_HEIGHT_M, obstacle_depth_m=OBSTACLE_DEPTH_M,
                support_rule=rule, config=config, body=body, obstacle_body=obstacle_body,
            )
            repaired = np.asarray(result.qpos, dtype=np.float32)
            record = dict(result.record)
            record.update({
                "seed": int(seed),
                "motion_key": key,
                "assigned": True,
                "source_qpos_content_sha256": source_rows[key].get("qpos_content_sha256"),
                "source_array_sha256": qpos_sha256(np.asarray(original, dtype=np.float32)),
                "candidate_stored": bool(record["accepted"]),
                "candidate_array_sha256": qpos_sha256(repaired) if record["accepted"] else None,
            })
            if record["accepted"]:
                # The controller consumes float32.  Refuse a candidate whose admission was a
                # float64-only numerical accident.
                check = repair_step_reference(
                    repaired, fps=FPS, obstacle_x_m=OBSTACLE_X_M,
                    obstacle_height_m=OBSTACLE_HEIGHT_M, obstacle_depth_m=OBSTACLE_DEPTH_M,
                    support_rule=rule, config=config, body=body, obstacle_body=obstacle_body,
                )
                # Re-running the repair is not the desired check because it would edit twice;
                # use its immutable "before" measurements, which score the supplied float32.
                if (not check.record["before"]["collision"]["collision_free"]
                        or not check.record["before"]["support"]["passes"]):
                    raise PreparationRefused(
                        f"{key} loses collision/support admission after float32 conversion"
                    )
                candidates[key] = repaired
            records.append(record)

    support_passing = tuple(
        row["motion_key"] for row in records if row["before"]["support"]["passes"]
    )
    accepted = tuple(row["motion_key"] for row in records if row["accepted"])
    if support_passing != EXPECTED_SUPPORT_PASSING_KEYS:
        raise PreparationRefused(
            f"support-passing keys changed: {support_passing} != {EXPECTED_SUPPORT_PASSING_KEYS}"
        )
    if accepted != EXPECTED_ACCEPTED_KEYS:
        raise PreparationRefused(
            f"accepted keys changed: {accepted} != {EXPECTED_ACCEPTED_KEYS}"
        )

    counts = {
        "n_assigned_trials": len(records),
        "n_input_support_pass": len(support_passing),
        "n_refused_input_support": sum(row["status"] == "refused" for row in records),
        "n_rejected_after_projection": sum(row["status"] == "rejected" for row in records),
        "n_accepted_for_execution": len(accepted),
        "support_passing_keys": list(support_passing),
        "accepted_keys": list(accepted),
        "status_counts": {
            status: sum(row["status"] == status for row in records)
            for status in ("refused", "rejected", "accepted")
        },
    }
    return records, candidates, counts


def dry_run_report(*, source: str | Path = SOURCE,
                   git_state_fn: Callable[[], Mapping[str, Any]] = git_state,
                   protocol_identity_fn: Callable[[], Mapping[str, Any]] = protocol_identity,
                   ) -> dict[str, Any]:
    source_info = source_identity(source)
    protocol = dict(protocol_identity_fn())
    state = dict(git_state_fn())
    records, candidates, counts = build_records(source)
    return {
        "schema": "exp031-step-repair-prepare-v1",
        "status": "dry_run",
        "writes_performed": False,
        "protocol": protocol,
        "project_dirty_observed": bool(state.get("dirty")),
        "source": source_info,
        "support_rule": asdict(support_rule()),
        "config": asdict(FootstepRepairConfig()),
        "summary": counts,
        "candidate_array_hashes": {key: qpos_sha256(value)
                                     for key, value in candidates.items()},
        "n_rows_computed": len(records),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare(*, out: str | Path = DEFAULT_OUT, source: str | Path = SOURCE,
            git_state_fn: Callable[[], Mapping[str, Any]] = git_state,
            protocol_identity_fn: Callable[[], Mapping[str, Any]] = protocol_identity,
            ) -> dict[str, Any]:
    """Write the pre-execution ledger once; never overwrite or loosen a refusal."""

    out = Path(out)
    protocol = dict(protocol_identity_fn())
    if protocol.get("status") != "preregistered":
        raise PreparationRefused(
            f"EXP-031 protocol status is {protocol.get('status')!r}; preregister and commit it first"
        )
    state = dict(git_state_fn())
    if state.get("dirty") is not False:
        raise PreparationRefused("EXP-031 preparation requires an exactly clean worktree")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise PreparationRefused(f"refusing non-empty output directory: {out}")

    source_info = source_identity(source)
    project = project_identity(state)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "schema": "exp031-step-repair-prepare-v1",
        "experiment": "exp031_constructive_step_repair",
        "status": "running",
        "stage": "preparing_references",
        "actual_ardy_samples": 0,
        "n_reused_archived_references": len(POOL_SEEDS),
        "protocol": protocol,
        "source": source_info,
        "project": project,
        "support_rule": asdict(support_rule()),
        "config": asdict(FootstepRepairConfig()),
        "scene": {
            "obstacle_x_m": OBSTACLE_X_M,
            "obstacle_height_m": OBSTACLE_HEIGHT_M,
            "obstacle_depth_m": OBSTACLE_DEPTH_M,
        },
    }
    _write_json(out / "receipt.json", receipt)

    try:
        records, candidates, counts = build_records(source)
        rows_path = out / "rows.jsonl"
        qpos_path = out / "qpos.npz"
        _write_rows(rows_path, records)
        temporary = out / "qpos.tmp.npz"
        np.savez_compressed(temporary, **candidates)
        os.replace(temporary, qpos_path)
        receipt.update({
            "status": "prepared",
            "stage": "references_prepared_and_frozen_before_sonic",
            "wall_clock_s": float(time.monotonic() - started),
            "summary": counts,
            "artifacts": {
                "rows": {"path": str(rows_path.resolve()), "sha256": sha256_file(rows_path)},
                "qpos": {"path": str(qpos_path.resolve()), "sha256": sha256_file(qpos_path),
                          "keys": list(candidates),
                          "content_hashes": {key: qpos_sha256(value)
                                             for key, value in candidates.items()}},
            },
            "sonic_rollouts_requested": 0,
            "sonic_rollouts_returned": 0,
            "claim_scope": (
                "pre-execution reference repair on the archived EXP-021 pool; no controller "
                "rollout and no traversal result"
            ),
        })
        _write_json(out / "receipt.json", receipt)
        return receipt
    except Exception as exc:
        receipt.update({
            "status": "blocked", "stage": "preparation_failed",
            "error_type": type(exc).__name__, "error": str(exc),
            "wall_clock_s": float(time.monotonic() - started),
        })
        _write_json(out / "receipt.json", receipt)
        if isinstance(exc, PreparationRefused):
            raise
        raise PreparationRefused(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = (dry_run_report(source=args.source) if args.dry_run
                  else prepare(out=args.out, source=args.source))
    except PreparationRefused as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
