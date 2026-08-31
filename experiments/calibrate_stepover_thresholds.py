"""Calibrate step-over support/contact thresholds from tracker-successful neutral walks.

EXP-016's kinematic gates read foot support from exact qpos geometry, but its default
support thresholds were conservative guesses: under them a tracked neutral walk can read as
mostly airborne, which nulls ``kinematic_step_success`` in every arm before the factorial
says anything (docs/review-2026-08-30-codex-changeset.md, defect 3).  This tool locks the
thresholds from data the tracker has already certified: the EXP-1C ``ctrl-*`` arms, whose
plain-walk references SONIC tracked with 100 % success, contribute both their 25 fps ARDY
reference clips and their 50 Hz achieved-state rollouts.  Calibrating over both rates at
once yields one threshold set valid wherever EXP-016 applies it.

The final frame of every non-terminated achieved rollout is dropped: SONIC's in-episode
reset overwrites it with a frame-0 teleport pose (same review, defect 1), and its ~300 m/s
finite-difference spike would poison the speed statistics.

The receipt's ``stepover_thresholds`` object is exactly the
:class:`scene2motion.stepover_eval.StepOverThresholds` constructor signature that
``exp016 --threshold_calibration_receipt`` consumes.  Deterministic: statistics are
order-independent and the seed only drives the optional per-arm subsample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.sonic_state_export import (  # noqa: E402
    load_sonic_state_rollouts,
    sonic_state_sample_dt,
)
from scene2motion.stepover_eval import (  # noqa: E402
    StepOverThresholds,
    calibrate_stepover_thresholds,
    foot_kinematics_series,
)

DEFAULT_ARMS = ("ctrl-l05", "ctrl-l08", "ctrl-l12", "ctrl-l16", "ctrl-l20", "ctrl-l28")


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state(root: Path) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).splitlines()
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=root,
            stderr=subprocess.DEVNULL)
        return {"commit": commit, "dirty": bool(status), "status": status,
                "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": [],
                "tracked_diff_sha256": None}


def _failed_keys(arm_dir: Path) -> set[str]:
    metrics = arm_dir / "metrics_eval.json"
    if not metrics.exists():
        return set()
    return set(json.loads(metrics.read_text()).get("failed_keys", []))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/exp016_threshold_calibration")
    ap.add_argument("--exp1c_dir", type=Path, default=Path("outputs/exp1c_stepover"))
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS),
                    help="neutral-walk arms; each needs qpos/<arm>__s*.npy and an "
                         "achieved-state archive under <exp1c_dir>/<arm>/")
    ap.add_argument("--sources", choices=["both", "reference", "achieved"],
                    default="both")
    ap.add_argument("--ref_fps", type=float, default=25.0,
                    help="frame rate of the ARDY reference clips")
    ap.add_argument("--stance_quantile", type=float, default=0.45)
    ap.add_argument("--corpus_quantile", type=float, default=0.95)
    ap.add_argument("--headroom", type=float, default=1.25)
    ap.add_argument("--min_side_support_fraction", type=float, default=0.45)
    ap.add_argument("--max_unsupported_fraction", type=float, default=0.05)
    ap.add_argument("--max_outlier_fraction", type=float, default=0.10)
    ap.add_argument("--max_clips_per_arm", type=int, default=0,
                    help="0 uses every clip; otherwise a seeded subsample per arm/source")
    ap.add_argument("--seed", type=int, default=16016)
    args = ap.parse_args()
    if args.max_clips_per_arm < 0:
        raise SystemExit("max_clips_per_arm must be non-negative")
    if args.ref_fps <= 0:
        raise SystemExit("ref_fps must be positive")
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    body = G1Body(None)

    def subsample(items: list) -> list:
        if not args.max_clips_per_arm or len(items) <= args.max_clips_per_arm:
            return items
        keep = rng.choice(len(items), size=args.max_clips_per_arm, replace=False)
        return [items[i] for i in sorted(keep)]

    kinematics, fps_list, clip_index = [], [], []
    corpus: dict = {"reference": [], "achieved": [], "excluded_achieved": []}
    for arm in args.arms:
        if args.sources in ("both", "reference"):
            paths = sorted((args.exp1c_dir / "qpos").glob(f"{arm}__s*.npy"))
            if not paths:
                raise SystemExit(f"no reference qpos for arm {arm} under "
                                 f"{args.exp1c_dir / 'qpos'}")
            for path in subsample(paths):
                kinematics.append(foot_kinematics_series(
                    body, np.asarray(np.load(path), dtype=float), args.ref_fps))
                fps_list.append(args.ref_fps)
                clip_index.append({"label": path.stem, "source": "reference",
                                   "fps": args.ref_fps})
                corpus["reference"].append(
                    {"path": str(path), "sha256": _sha256(path)})
        if args.sources in ("both", "achieved"):
            arm_dir = args.exp1c_dir / arm
            sample_dt = sonic_state_sample_dt(arm_dir)
            failed = _failed_keys(arm_dir)
            rollouts = []
            for rollout in load_sonic_state_rollouts(arm_dir):
                # Tracker-successful only, and only frames before SONIC's end-of-episode
                # reset teleport (review defect 1): the last stored frame of a
                # non-terminated rollout is a frame-0 pose at the env origin.
                if rollout.terminated or rollout.motion_key in failed:
                    corpus["excluded_achieved"].append(
                        {"motion_key": rollout.motion_key,
                         "reason": "terminated" if rollout.terminated else "failed_key"})
                    continue
                rollouts.append(rollout)
            for rollout in subsample(rollouts):
                kinematics.append(foot_kinematics_series(
                    body, np.asarray(rollout.qpos[:-1], dtype=float), 1.0 / sample_dt))
                fps_list.append(1.0 / sample_dt)
                clip_index.append({"label": f"{rollout.motion_key}.achieved",
                                   "source": "achieved", "fps": 1.0 / sample_dt})
            archive = arm_dir / "achieved_qpos.npz"
            corpus["achieved"].append({
                "arm": arm, "archive": str(archive), "sha256": _sha256(archive),
                "sample_dt_s": sample_dt, "n_successful": len(rollouts),
                "final_frame_dropped": True,
            })
        print(f"  {arm}: corpus at {len(kinematics)} clips "
              f"({time.time() - t0:.0f}s)", flush=True)

    thresholds, diagnostics = calibrate_stepover_thresholds(
        kinematics, fps_list,
        base=StepOverThresholds(),
        stance_quantile=args.stance_quantile,
        corpus_quantile=args.corpus_quantile,
        headroom=args.headroom,
        min_side_support_fraction=args.min_side_support_fraction,
        max_unsupported_fraction=args.max_unsupported_fraction,
        max_outlier_fraction=args.max_outlier_fraction,
    )
    for entry, diag in zip(clip_index, diagnostics["clips"]):
        diag["label"] = entry["label"]
        diag["source"] = entry["source"]

    repo = Path(__file__).resolve().parents[1]
    receipt = {
        "experiment": "stepover_threshold_calibration",
        "status": "calibrated",
        "consumed_by": "exp016 --threshold_calibration_receipt",
        # The schema exp016 reads: StepOverThresholds(**receipt["stepover_thresholds"]).
        "stepover_thresholds": asdict(thresholds),
        "temporal_gate_units": "seconds, converted to frames at each evaluation rate",
        "calibration": diagnostics,
        "corpus": {
            "exp1c_dir": str(args.exp1c_dir),
            "arms": list(args.arms),
            "sources": args.sources,
            "reference_fps": args.ref_fps,
            "selection": ("tracker-successful (non-terminated, not in failed_keys) "
                          "achieved rollouts plus their ARDY reference clips"),
            **corpus,
        },
        "seed": args.seed,
        "max_clips_per_arm": args.max_clips_per_arm,
        "provenance": {
            "code": _git_state(repo),
            "source_sha256": {
                "experiments/calibrate_stepover_thresholds.py": _sha256(
                    Path(__file__).resolve()),
                "scene2motion/stepover_eval.py": _sha256(
                    repo / "scene2motion" / "stepover_eval.py"),
                "scene2motion/sonic_state_export.py": _sha256(
                    repo / "scene2motion" / "sonic_state_export.py"),
            },
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    with open(out / "receipt.json", "w") as fh:
        json.dump(receipt, fh, indent=2)
    print(json.dumps({"stepover_thresholds": receipt["stepover_thresholds"],
                      "n_clips": diagnostics["n_clips"],
                      "n_accepted": diagnostics["n_accepted"],
                      "outliers": diagnostics["outlier_clip_indices"]}, indent=2))
    print(f"wrote {out / 'receipt.json'}", flush=True)


if __name__ == "__main__":
    main()
