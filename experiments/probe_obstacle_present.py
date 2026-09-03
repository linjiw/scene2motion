"""Operational probe: can a real obstacle be put in the SONIC scene, and does the robot feel it?

**Not campaign evidence.** No seeds are spent and no campaign directory is touched. This is the
engineering validation EXP-029 needs before it can measure local traversal with the obstacle
*present* in physics, which the project has never yet done: every executed result so far replays
achieved states against our collision model with the box absent from Isaac.

What the SONIC checkout actually does (read at ``4179cdb``):

* ``manager_env.config.add_table=true`` spawns ``{ENV_REGEX_NS}/Table`` as a ``CuboidCfg`` with
  ``kinematic_enabled``, ``collision_enabled`` and ``activate_contact_sensors``
  (``modular_tracking_env_cfg.py:595-665``).  ``table_size`` is the cuboid's **full x, y, z
  extents** and ``table_position`` its **centre**, so a box of height *h* resting on the floor is
  ``size=[depth_x, width_y, h]`` at ``pos=[x, y, h/2]``.
* **But the spawn pose does not survive a reset.**  Whenever a table exists, ``commands.py:3134``
  rewrites its pose per environment on every reset: from the motion's own ``table_pos`` /
  ``table_quat`` when the motion carries them (**plus that environment's origin**), and otherwise
  from a fallback that puts the table at the object's position with ``z = 0.76`` and no env
  offset (``commands.py:3149-3181``).  Motion pickles written by
  ``scene2motion.sonic_export.write_motion_pkl`` carry no table metadata, so the naive route —
  set ``table_position`` on the command line — would place the obstacle somewhere else entirely.

So this probe tests the route that should work: **per-motion table metadata inside the pickle**,
which makes the cached branch fire and places one obstacle per environment at the intended
position. It runs the same motions twice, with and without the obstacle, and asks whether the
achieved trajectories differ. Identical trajectories mean the obstacle had no physical effect;
different ones mean the robot actually felt it.

Run:  $S2M_PY experiments/probe_obstacle_present.py --box-height 0.30
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments import exp1b_execution_clearance as exp1b  # noqa: E402
from experiments import exp028_termination_free_rollouts as e28  # noqa: E402
from experiments import probe_sonic_vram as pv  # noqa: E402
from scene2motion import host_gate  # noqa: E402
from scene2motion.sonic_export import write_motion_pkl  # noqa: E402
from scene2motion.sonic_state_export import load_sonic_state_rollouts  # noqa: E402

RUN_DIR = REPO / "run/probe_obstacle_present"
OUT_DIR = REPO / "outputs/probe_obstacle_present"
#: Two archived EXP-021 references that reach the obstacle position under the release evaluator.
DEFAULT_MOTION_KEYS = ("s4401", "s4409")
#: Trajectories are float32 world states; anything above this is a real physical difference.
SAME_TRAJECTORY_TOL_M = 1e-4


def build_pkl(keys: list[str], path: Path, table: dict[str, Any] | None) -> Path:
    """Write the motion pickle, optionally giving every motion the same table pose.

    ``commands.py`` reads ``table_pos`` / ``table_quat`` off each motion entry and adds the
    environment origin, so the obstacle lands at the same scene-relative place in every
    environment.  Without them the reset-time fallback would move it.
    """
    from scene2motion.robot import G1Body
    with np.load(pv.EXP021_QPOS, allow_pickle=False) as archive:
        missing = [key for key in keys if key not in archive.files]
        if missing:
            raise pv.ProbeRefusal(f"archive lacks {missing}")
        clips = {key: np.array(archive[key], copy=True) for key in keys}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_motion_pkl(clips, path, fps=25, mj_model=G1Body(None).model)
    if table is None:
        return path
    with open(path, "rb") as handle:
        motions = pickle.load(handle)
    for entry in motions.values():
        entry["table_pos"] = list(table["pos"])
        entry["table_quat"] = list(table["quat"])
    with open(path, "wb") as handle:
        pickle.dump(motions, handle, protocol=4)
    return path


def launch(pkl: Path, eval_dir: Path, overrides: list[str], *, num_envs: int, timeout_s: int,
           abort_free_mib: int, abort_free_ram_mib: int) -> dict[str, Any]:
    """One monitored SONIC launch; the VRAM probe's safety monitor is reused unchanged."""
    command = e28.build_sonic_command(pkl, eval_dir, num_envs, 0, overrides)
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=exp1b.SONIC, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, env=exp1b.sonic_env(),
                               stdin=subprocess.DEVNULL, start_new_session=True)
    monitor = pv.Monitor(process.pid, abort_free_mib, abort_free_ram_mib)
    monitor.start()
    try:
        log = process.communicate(timeout=timeout_s)[0]
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        log = process.communicate()[0]
    finally:
        monitor.stop_event.set()
        monitor.join(timeout=5)
    eval_dir.parent.mkdir(parents=True, exist_ok=True)
    (eval_dir.parent / "sonic.log").write_text(log or "")
    return {"command": command, "returncode": process.returncode,
            "elapsed_s": round(time.monotonic() - started, 1),
            "peak_launch_mib": monitor.peak("launch_mib"),
            "min_free_mib": monitor.minimum("free_mib"),
            "min_available_ram_mib": monitor.minimum("available_ram_mib"),
            "aborted_reason": monitor.aborted_reason,
            "log_tail": (log or "").strip().splitlines()[-6:]}


def compare(no_box: dict[str, Any], box: dict[str, Any]) -> dict[str, Any]:
    """Per motion: did the obstacle change the achieved trajectory at all?"""
    rows: list[dict[str, Any]] = []
    for key in sorted(set(no_box) & set(box)):
        a, b = no_box[key], box[key]
        n = int(min(a.valid_length, b.valid_length))
        delta = (float(np.abs(a.qpos[:n] - b.qpos[:n]).max()) if n > 0 else None)
        rows.append({
            "motion_key": key,
            "valid_length": {"no_box": int(a.valid_length), "box": int(b.valid_length)},
            "terminated": {"no_box": bool(a.terminated), "box": bool(b.terminated)},
            "max_root_x_m": {"no_box": float(a.qpos[:a.valid_length, 0].max()),
                             "box": float(b.qpos[:b.valid_length, 0].max())},
            "min_root_z_m": {"no_box": float(a.qpos[:a.valid_length, 2].min()),
                             "box": float(b.qpos[:b.valid_length, 2].min())},
            "max_abs_qpos_difference_over_common_frames": delta,
            "differs": bool(delta is not None and delta > SAME_TRAJECTORY_TOL_M),
        })
    differing = [r for r in rows if r["differs"]]
    return {
        "per_motion": rows,
        "n_motions": len(rows),
        "n_differing": len(differing),
        "obstacle_has_physical_effect": bool(differing),
        "tolerance_m": SAME_TRAJECTORY_TOL_M,
        "reading": ("the obstacle changed the achieved motion, so it is present in physics and "
                    "the robot interacted with it" if differing else
                    "the achieved motions are identical: the obstacle was either not spawned, "
                    "not where it was asked for, or not in the robot's path"),
    }


def probe(*, motion_keys: list[str], box_x_m: float, box_height_m: float, box_depth_m: float,
          box_width_m: float, env_spacing_m: float, episode_length_s: float, num_envs: int,
          timeout_s: int, safety_floor_mib: int, abort_free_mib: int,
          safety_floor_ram_mib: int = 10000, abort_free_ram_mib: int = 1500,
          extra_box_overrides: list[str] | None = None,
          run_dir: Path = RUN_DIR, dry_run: bool = False) -> dict[str, Any]:
    run_dir = Path(run_dir)
    before = pv.gpu_sample()
    table = {"pos": [float(box_x_m), 0.0, float(box_height_m) / 2.0],
             "quat": [1.0, 0.0, 0.0, 0.0],
             "size_xyz": [float(box_depth_m), float(box_width_m), float(box_height_m)]}
    shared = [f"++manager_env.config.env_spacing={float(env_spacing_m)}",
              f"++manager_env.config.episode_length_s={float(episode_length_s)}"]
    box_overrides = shared + [
        "++manager_env.config.add_table=true",
        "++manager_env.config.table_size=" + json.dumps(table["size_xyz"]),
        "++manager_env.config.table_position=" + json.dumps(table["pos"]),
    ] + list(extra_box_overrides or [])
    report: dict[str, Any] = {
        "probe": "obstacle_present", "campaign_evidence": False,
        "motion_keys": list(motion_keys), "num_envs": int(num_envs),
        "obstacle": {**table, "convention": (
            "CuboidCfg size is the full x/y/z extents and the position is the box centre; a box "
            "of height h resting on the floor is size=[depth_x, width_y, h] at pos=[x, y, h/2]")},
        "shared_overrides": shared,
        "box_overrides": box_overrides,
        "table_metadata_in_pickle": True,
        "why_metadata": ("commands.py rewrites the table pose per environment on every reset; "
                         "without per-motion table_pos/table_quat it falls back to an "
                         "object-derived pose at z=0.76 and ignores the spawn position"),
        "before": {"vram": {k: before.get(k) for k in ("used_mib", "free_mib", "total_mib")},
                   "ram": dict(host_gate.query_available_ram_mib()),
                   "concurrent_isaac_processes": len(host_gate.concurrent_isaac_processes())},
    }
    if dry_run:
        report["status"] = "dry_run"
        return report
    free = before.get("free_mib")
    if free is None or free < int(safety_floor_mib):
        raise pv.ProbeRefusal(f"free VRAM {free} MiB is below the safety floor {safety_floor_mib}")
    available_ram = report["before"]["ram"].get("available_mib")
    if available_ram is None or available_ram < int(safety_floor_ram_mib):
        raise pv.ProbeRefusal(
            f"available host RAM {available_ram} MiB is below the safety floor "
            f"{safety_floor_ram_mib} MiB; a launch consumes about 6.8 GiB and the kernel "
            "OOM-killer takes whatever is largest")

    arms: dict[str, Any] = {}
    rollouts: dict[str, dict[str, Any]] = {}
    for arm, overrides, table_meta in (("no_box", shared, None), ("box", box_overrides, table)):
        arm_dir = run_dir / arm
        pkl = build_pkl(list(motion_keys), arm_dir / "motions.pkl", table_meta)
        arms[arm] = launch(pkl, arm_dir / "eval", overrides, num_envs=num_envs,
                           timeout_s=timeout_s, abort_free_mib=abort_free_mib,
                           abort_free_ram_mib=abort_free_ram_mib)
        if arms[arm]["returncode"] == 0:
            try:
                rollouts[arm] = {r.motion_key: r
                                 for r in load_sonic_state_rollouts(arm_dir / "eval")}
            except (OSError, ValueError, FileNotFoundError) as exc:
                arms[arm]["archive_error"] = str(exc)
    report["arms"] = arms
    if len(rollouts) == 2:
        report["comparison"] = compare(rollouts["no_box"], rollouts["box"])
        report["status"] = "complete"
    else:
        report["status"] = "failed"
        report["reading"] = "at least one launch did not produce an achieved-state archive"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion-keys", nargs="+", default=list(DEFAULT_MOTION_KEYS))
    parser.add_argument("--box-x-m", type=float, default=1.2)
    parser.add_argument("--box-height-m", type=float, default=0.30)
    parser.add_argument("--box-depth-m", type=float, default=0.20)
    parser.add_argument("--box-width-m", type=float, default=2.0)
    parser.add_argument("--env-spacing-m", type=float, default=12.0,
                        help="must exceed the 7.2 m route; the checkout default is 2.5 m")
    parser.add_argument("--episode-length-s", type=float, default=20.0,
                        help="the checkout default is 10.0 s against ~8.3 s references")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--safety-floor-mib", type=int, default=4500)
    parser.add_argument("--abort-free-mib", type=int, default=1200)
    parser.add_argument("--safety-floor-ram-mib", type=int, default=10000,
                        help="refuse to launch unless this much host RAM is available")
    parser.add_argument("--abort-free-ram-mib", type=int, default=1500,
                        help="kill the launch if available host RAM falls below this")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--run-dir", default=str(RUN_DIR))
    parser.add_argument("--extra-box-override", action="append", default=[],
                        help="extra Hydra override for the obstacle arm only (repeatable)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = probe(motion_keys=list(args.motion_keys), box_x_m=args.box_x_m,
                       box_height_m=args.box_height_m, box_depth_m=args.box_depth_m,
                       box_width_m=args.box_width_m, env_spacing_m=args.env_spacing_m,
                       episode_length_s=args.episode_length_s, num_envs=args.num_envs,
                       timeout_s=args.timeout_s, safety_floor_mib=args.safety_floor_mib,
                       abort_free_mib=args.abort_free_mib,
                       safety_floor_ram_mib=args.safety_floor_ram_mib,
                       abort_free_ram_mib=args.abort_free_ram_mib,
                       extra_box_overrides=list(args.extra_box_override),
                       run_dir=Path(args.run_dir), dry_run=args.dry_run)
    except pv.ProbeRefusal as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, indent=2))
        return 2
    if not args.dry_run:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        tag = "_".join(o.split("=")[0].split(".")[-1] for o in args.extra_box_override)
        (out / f"report_h{args.box_height_m:g}{('_' + tag) if tag else ''}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
    printable = {k: v for k, v in report.items() if k not in {"arms", "box_overrides"}}
    printable["arms"] = {k: {kk: vv for kk, vv in v.items() if kk != "command"}
                         for k, v in report.get("arms", {}).items()}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0 if report.get("status") in {"complete", "dry_run"} else 1


if __name__ == "__main__":
    sys.exit(main())
