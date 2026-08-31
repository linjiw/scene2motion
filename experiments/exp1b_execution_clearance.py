"""EXP-1B: execution clearance — reconstructed, resumable driver.

The original driver (`exp1b_full.py`) lived in /tmp and was lost to a reboot mid-campaign.
This file re-derives its exact behaviour from the artifacts it left behind — `run/receipt.json`
(selection parameters and launch plan), `run/join.jsonl` (the 859 selected rows), and five
intact launches — and is asserted against them: the selection below must reproduce
`run/join.jsonl` byte-for-value, and the chunking must reproduce the receipt's launch plan,
or the run refuses to continue.

Pipeline: select verified rows from the Phase-4E ledger (methods heuristic/heuristic+1/+2,
preference shortest, outcome accepted or accepted_margin) -> chunk into launches of <= 36
clips -> per launch, write a SONIC motion pkl and run one SONIC rollout subprocess with the
achieved-state export callback -> join achieved qpos back to the exact scene geometry into
`run/rows.jsonl` -> fit the one-sided conformal execution gate tau(d) of
docs/exec-gate-audit.md with the scene-level calibration split the RAMP guidance requires.

What changed relative to the lost run
(`docs/exp1b-frame-diagnosis-2026-08-31.md` has the numbers):

* The frame red-flag is NOT an env-origin offset — achieved and reference roots agree to
  ~2.6 cm at frame 0 and the error grows monotonically to metres because the tracker stalls.
  No offset subtraction is applied, deliberately.
* The one real archive defect was the post-reset teleport frame of non-terminated rollouts
  (review defect #1).  New archives (schema v2) exclude it at write time; old intact v1
  archives have it dropped at load time by `load_sonic_state_rollouts`.  Each row records
  `archive_schema_version` and `teleport_frame_handling` so provenance stays explicit.
* `executed_success` now additionally requires `passed_last_obstacle`: a rollout that stalls
  collision-free 2 m into a 15 m route is not a success (the v1-suspect rows, kept as
  `run/rows.v1-suspect.jsonl`, counted 28 such rows as successes).
* `loss_m` (ref minus exec minimum overhead clearance, margin applied once by G1Body) is an
  eligible gate observation (`loss_valid`) only when the rollout passed the last obstacle:
  termination or a stall before the obstacle is an execution failure, not a censored
  clearance observation (docs/exec-gate-audit.md).

Resume is idempotent: launches whose directory holds a complete, key-matching achieved-state
archive plus metrics and a clean rc are skipped; everything else (including heuristic_05,
whose raw artifacts a reboot zeroed, and heuristic_06's 0-byte pkl) is repaired aside and
relaunched.  `--dry-run` prints the launch plan and per-launch status without touching SONIC.

    source env.sh && $S2M_PY experiments/exp1b_execution_clearance.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.demo.cache import CACHE_VERSION, ClipCache  # noqa: E402
from scene2motion.demo.scene_builder import BeamParams, beam_footprints, build  # noqa: E402
from scene2motion.robot import CLEARANCE_MARGIN, G1Body  # noqa: E402
from scene2motion.sonic_export import write_motion_pkl  # noqa: E402
from scene2motion.sonic_state_export import (  # noqa: E402
    load_sonic_state_rollouts,
    sonic_state_archive_schema,
    sonic_state_hydra_overrides,
    sonic_state_sample_dt,
    sonic_state_subprocess_env,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "outputs/phase4e_architecture_v2_s8/experiment.json"
CACHE_DIR = ROOT / "scene2motion/demo_outputs/clips"

METHODS = ("heuristic", "heuristic+1", "heuristic+2")
OK_OUTCOMES = ("accepted", "accepted_margin")  # collision-free; 'rejected' rows are excluded
PREFERENCE = "shortest"
TARGET_M = 0.18
PROGRESS_OK = 0.95
NOISE_STREAM_VERSION = 2
CHUNK_SIZE = 36
PHYSICS_SEEDS = (0,)

# Same checkout facts as exp011 (see its comments for why -m, absolute paths, and the EULA
# variable are each load-bearing).
SONIC = Path("/home/linjiw/lucid/GR00T-WholeBodyControl")
SONIC_PY = Path("/home/linjiw/isaaclab-install/env_isaaclab/bin/python")
CKPT = SONIC / "sonic_release" / "last.pt"

GATE_ALPHA = 0.10
GATE_MIN_OBS = 8


# ---------------------------------------------------------------------------- selection

def scene_for(ledger_row: dict):
    """Rebuild the exact demo scene a ledger row was planned in, with an identity check."""
    scene = build(BeamParams(beam_height=ledger_row["beam_h"], beam_width=ledger_row["beam_w"],
                             n_beams=ledger_row["n_beams"], gap=ledger_row["gap"]))
    if scene.scene_id != ledger_row["scene_id"]:
        raise ValueError(f"rebuilt {scene.scene_id!r} != ledger {ledger_row['scene_id']!r}")
    return scene


def motion_key_for(ledger_row: dict) -> str:
    return f"{ledger_row['scene_id']}__{ledger_row['method']}__s{ledger_row['seed']}"


def verify_clip(cache: ClipCache, ledger_row: dict) -> tuple[dict | None, str | None]:
    """(clip meta, None) when the cached clip is the row's verified artifact, else (None, why)."""
    got = cache.get(ledger_row["clip_key"])
    if got is None:
        return None, "clip missing from cache"
    qpos, meta = got
    checks = [
        (meta.get("cache_version") == CACHE_VERSION, f"cache_version {meta.get('cache_version')}"),
        (meta.get("noise_stream_version") == NOISE_STREAM_VERSION,
         f"noise_stream_version {meta.get('noise_stream_version')}"),
        (meta.get("scene_id") == ledger_row["scene_id"], "scene_id mismatch"),
        (meta.get("preference") == PREFERENCE, f"preference {meta.get('preference')}"),
        (meta.get("schedule_hash") == ledger_row["final_schedule_hash"],
         "schedule hash is not the final schedule"),
        (qpos.ndim == 2 and qpos.shape == (int(meta.get("n_frames", -1)), 36),
         f"qpos shape {qpos.shape}"),
        (bool(np.all(np.isfinite(qpos))), "non-finite qpos"),
    ]
    for ok, why in checks:
        if not ok:
            return None, why
    return meta, None


def join_row_from_ledger(ledger_row: dict, cache: ClipCache) -> tuple[dict | None, dict | None]:
    """One selected+verified row in run/join.jsonl's exact shape, or a skip record."""
    meta, why = verify_clip(cache, ledger_row)
    if meta is None:
        return None, {"motion_key": motion_key_for(ledger_row),
                      "clip_key": ledger_row["clip_key"], "reason": why}
    scene = scene_for(ledger_row)
    route_len = float(np.hypot(scene.goal[0] - scene.start[0], scene.goal[1] - scene.start[1]))
    return {
        "motion_key": motion_key_for(ledger_row),
        "scene_id": ledger_row["scene_id"],
        "method": ledger_row["method"],
        "seed": ledger_row["seed"],
        "outcome": ledger_row["outcome"],
        "clip_key": ledger_row["clip_key"],
        "beam_h": ledger_row["beam_h"],
        "beam_w": ledger_row["beam_w"],
        "n_beams": ledger_row["n_beams"],
        "gap": ledger_row["gap"],
        "ref_min_overhead_m": ledger_row["min_overhead_m"],
        "ref_min_clearance_m": ledger_row["min_clearance_m"],
        "ref_goal_error_m": ledger_row["goal_error_m"],
        "peak_dip_m": ledger_row["peak_dip_m"],
        "peak_dip_initial_m": ledger_row["peak_dip_initial_m"],
        "n_repairs": ledger_row["n_repairs"],
        "route_len_m_rebuilt": round(route_len, 3),
        "initial_schedule_hash": ledger_row["initial_schedule_hash"],
        "final_schedule_hash": ledger_row["final_schedule_hash"],
        "clip_n_frames": meta["n_frames"],
        "clip_fps": meta["fps"],
        "noise_stream_version": meta["noise_stream_version"],
    }, None


def derive_selection(ledger: dict, cache: ClipCache) -> tuple[list[dict], list[dict]]:
    """The receipt's selection, method-grouped in ledger order — deterministic by construction."""
    if ledger.get("preference") != PREFERENCE or ledger.get("target_m") != TARGET_M:
        raise ValueError("ledger preference/target does not match the exp1b receipt")
    selected: list[dict] = []
    skipped: list[dict] = []
    for method in METHODS:
        for ledger_row in ledger["rows"]:
            if ledger_row["method"] != method or ledger_row["outcome"] not in OK_OUTCOMES:
                continue
            row, skip = join_row_from_ledger(ledger_row, cache)
            (selected if row is not None else skipped).append(row if row is not None else skip)
    return selected, skipped


# ------------------------------------------------------------------------------- launches

@dataclass
class LaunchSpec:
    method: str
    chunk: int
    physics_seed: int
    rows: list[dict] = field(repr=False)

    @property
    def name(self) -> str:
        return f"{self.method}_{self.chunk:02d}_seed{self.physics_seed}"

    @property
    def motion_keys(self) -> list[str]:
        return [r["motion_key"] for r in self.rows]


def plan_launches(selected: list[dict], chunk_size: int = CHUNK_SIZE,
                  physics_seeds: tuple[int, ...] = PHYSICS_SEEDS) -> list[LaunchSpec]:
    plan: list[LaunchSpec] = []
    for method in METHODS:
        rows = [r for r in selected if r["method"] == method]
        for seed in physics_seeds:
            for chunk, lo in enumerate(range(0, len(rows), chunk_size)):
                plan.append(LaunchSpec(method, chunk, seed, rows[lo:lo + chunk_size]))
    return plan


def launch_status(run_dir: Path, spec: LaunchSpec, launch_records: dict[str, dict]
                  ) -> tuple[str, str]:
    """('complete'|'incomplete'|'pending', reason).  Artifacts are the ground truth; the
    launches.jsonl rc only demotes (a clean archive with rc!=0 is still a failed run)."""
    d = run_dir / "launches" / spec.name
    if not d.exists() or not any(d.iterdir()):
        return "pending", "no artifacts"
    eval_dir = d / "eval"
    archive = eval_dir / "achieved_qpos.npz"
    log = d / "sonic.log"
    metrics = eval_dir / "metrics_eval.json"
    if not archive.exists() or archive.stat().st_size == 0:
        return "incomplete", "achieved archive missing or empty"
    try:
        rollouts = load_sonic_state_rollouts(eval_dir)
    except Exception as exc:
        return "incomplete", f"archive unreadable: {exc}"
    if sorted(r.motion_key for r in rollouts) != sorted(spec.motion_keys):
        return "incomplete", "archive motion keys do not match the chunk"
    if not metrics.exists() or metrics.stat().st_size == 0:
        return "incomplete", "metrics_eval.json missing or empty"
    if not log.exists() or log.stat().st_size == 0:
        return "incomplete", "sonic.log missing or empty"
    record = launch_records.get(spec.name)
    if record is not None and record.get("rc") != 0:
        return "incomplete", f"recorded rc={record.get('rc')}"
    return "complete", f"{len(rollouts)} rollouts" + ("" if record else " (no launch record)")


# ---------------------------------------------------------------------------- run/ repair

def repair_launches_jsonl(path: Path) -> dict:
    """Truncate the NUL tail a reboot left mid-write and drop unparseable lines."""
    if not path.exists():
        return {"existed": False, "kept": 0, "dropped_bytes": 0}
    raw = path.read_bytes()
    cut = raw.find(b"\x00")
    head = raw if cut < 0 else raw[:cut]
    kept, dropped_lines = [], 0
    for line in head.split(b"\n"):
        if not line.strip():
            continue
        try:
            json.loads(line)
            kept.append(line)
        except ValueError:
            dropped_lines += 1
    repaired = b"".join(k + b"\n" for k in kept)
    if repaired != raw:
        path.write_bytes(repaired)
    return {"existed": True, "kept": len(kept), "dropped_bytes": len(raw) - len(repaired),
            "dropped_lines": dropped_lines}


def repair_run_state(run_dir: Path, specs: list[LaunchSpec]) -> list[str]:
    """Make the artifact tree truthful: NUL tails gone, 0-byte files not mistaken for data."""
    actions: list[str] = []
    summary = repair_launches_jsonl(run_dir / "launches.jsonl")
    if summary["existed"] and summary["dropped_bytes"]:
        actions.append(f"launches.jsonl: kept {summary['kept']} lines, "
                       f"truncated {summary['dropped_bytes']} bytes of NUL/garbage tail")

    for spec in specs:
        d = run_dir / "launches" / spec.name
        if not d.exists():
            continue
        pkl = d / f"{spec.name}.pkl"
        if pkl.exists() and pkl.stat().st_size == 0:
            pkl.unlink()  # zero bytes carry no data worth keeping
            actions.append(f"{spec.name}: removed 0-byte pkl")
        for rel in ("sonic.log", "eval/achieved_qpos.npz", "eval/metrics_eval.json"):
            f = d / rel
            if f.exists() and f.stat().st_size == 0:
                aside = d / "zeroed" / f.name
                aside.parent.mkdir(exist_ok=True)
                f.rename(aside)
                actions.append(f"{spec.name}: moved zeroed {rel} -> zeroed/{f.name}")

    rows = run_dir / "rows.jsonl"
    suspect = run_dir / "rows.v1-suspect.jsonl"
    if rows.exists() and not suspect.exists():
        rows.rename(suspect)
        actions.append("rows.jsonl -> rows.v1-suspect.jsonl (pre-diagnosis join, superseded)")
    return actions


# ------------------------------------------------------------------------------ SONIC run

def sonic_env() -> dict:
    e = dict(os.environ)
    e["OMNI_KIT_ACCEPT_EULA"] = "YES"  # recorded acceptance, see exp011
    e["ISAACLAB_PATH"] = "/home/linjiw/isaaclab-install/IsaacLab"
    e.setdefault("PYTHONUNBUFFERED", "1")
    return sonic_state_subprocess_env(e)


def run_sonic(pkl: Path, out_dir: Path, num_envs: int, physics_seed: int,
              timeout_s: int) -> tuple[int, str]:
    pkl, out_dir = pkl.resolve(), out_dir.resolve()
    cmd = [str(SONIC_PY), "-u", "-m", "gear_sonic.eval_agent_trl",
           f"+checkpoint={CKPT}", "+headless=True", "++eval_callbacks=im_eval",
           "++run_eval_loop=False", f"++num_envs={num_envs}",
           f"++eval_output_dir={out_dir}",
           f"++seed={physics_seed}",
           "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False",
           "+manager_env/terminations=tracking/eval",
           f"+manager_env.commands.motion.motion_lib_cfg.motion_file={pkl}",
           f"+log_keys={pkl.stem}",
           *sonic_state_hydra_overrides()]
    print("  " + " ".join(cmd[:4]) + " ...", flush=True)
    p = subprocess.run(cmd, cwd=SONIC, capture_output=True, text=True, timeout=timeout_s,
                       env=sonic_env(), stdin=subprocess.DEVNULL)
    return p.returncode, (p.stdout + "\n" + p.stderr)


def ensure_launch_pkl(spec: LaunchSpec, cache: ClipCache, mj_model, run_dir: Path) -> Path:
    d = run_dir / "launches" / spec.name
    pkl = d / f"{spec.name}.pkl"
    if pkl.exists() and pkl.stat().st_size > 0:
        try:
            with open(pkl, "rb") as fh:
                existing = pickle.load(fh)
            if (sorted(existing) == sorted(spec.motion_keys)
                    and all(len(existing[r["motion_key"]]["root_trans_offset"])
                            == r["clip_n_frames"] for r in spec.rows)):
                return pkl
        except Exception:
            pass
        pkl.unlink()  # wrong or unreadable content: rebuild deterministically from the cache
    clips = {}
    for row in spec.rows:
        got = cache.get(row["clip_key"])
        if got is None:
            raise FileNotFoundError(f"{row['clip_key']} vanished from the clip cache")
        clips[row["motion_key"]] = got[0]
    fps = {int(round(r["clip_fps"])) for r in spec.rows}
    if len(fps) != 1:
        raise ValueError(f"{spec.name} mixes clip fps values {sorted(fps)}")
    return write_motion_pkl(clips, pkl, fps=fps.pop(), mj_model=mj_model)


# ----------------------------------------------------------------------------------- join

def replay_exec(body: G1Body, scene, traj: np.ndarray) -> dict:
    """Exact scene-geometry replay of one achieved trajectory (teleport already excluded)."""
    beams = beam_footprints(scene)
    min_clear, min_over = CLEARANCE_MARGIN, CLEARANCE_MARGIN
    coll_over = coll_lat = False
    culprits: set[str] = set()
    floor_pen = 0.0
    for t in range(len(traj)):
        scene_c, floor_c = body.frame_contacts(traj[t], t)
        if floor_c:
            floor_pen = max(floor_pen, max(0.0, -min(c.dist for c in floor_c)))
        for c in scene_c:
            min_clear = min(min_clear, c.dist)
            if c.overhead:
                min_over = min(min_over, c.dist)
            if c.dist < 0:
                culprits.add(c.robot_geom)
                if c.overhead:
                    coll_over = True
                else:
                    coll_lat = True
    max_x = float(traj[:, 0].max()) if len(traj) else float("nan")
    last = traj[-1] if len(traj) else np.full(36, np.nan)
    goal_err = float(np.hypot(last[0] - scene.goal[0], last[1] - scene.goal[1]))
    return {
        "passed_last_obstacle": bool(len(traj) and max_x >= beams[-1]["x_hi"]),
        "reached_first_obstacle": bool(len(traj) and max_x >= beams[0]["x_lo"]),
        "exec_min_overhead_m": round(float(min_over), 5),
        "exec_min_clearance_m": round(float(min_clear), 5),
        "exec_collision_any": bool(coll_over or coll_lat),
        "exec_collision_overhead": coll_over,
        "exec_collision_lateral": coll_lat,
        "exec_goal_error_m": round(goal_err, 4),
        "exec_max_x_m": round(max_x, 4),
        "max_foot_floor_penetration_m": round(float(floor_pen), 5),
        "culprits": sorted(culprits),
    }


def join_rollout(join_row: dict, rollout, scene, body: G1Body, *, launch: str,
                 physics_seed: int, schema_version: int, sample_dt_s: float,
                 progress_ok: float = PROGRESS_OK) -> dict:
    exec_part = replay_exec(body, scene, rollout.qpos)
    execution_failure = bool(rollout.terminated or rollout.progress < progress_ok)
    loss = None
    if not execution_failure:
        loss = round(join_row["ref_min_overhead_m"] - exec_part["exec_min_overhead_m"], 5)
    return {
        "motion_key": join_row["motion_key"],
        "scene_id": join_row["scene_id"],
        "method": join_row["method"],
        "seed": join_row["seed"],
        "clip_key": join_row["clip_key"],
        "outcome": join_row["outcome"],
        "ref_min_overhead_m": join_row["ref_min_overhead_m"],
        "peak_dip_m": join_row["peak_dip_m"],
        "route_len_m": join_row["route_len_m_rebuilt"],
        "physics_seed": physics_seed,
        "launch": launch,
        "archive_schema_version": schema_version,
        "teleport_frame_handling": "dropped_at_load" if schema_version < 2
                                   else "dropped_at_write",
        "sample_dt_s": sample_dt_s,
        "terminated": bool(rollout.terminated),
        "progress": round(float(rollout.progress), 4),
        "valid_frames": int(rollout.valid_length),
        **exec_part,
        "execution_failure": execution_failure,
        "executed_success": bool(not execution_failure
                                 and not exec_part["exec_collision_any"]
                                 and exec_part["passed_last_obstacle"]),
        "loss_m": loss,
        # Only a rollout that actually cleared the last obstacle yields a clearance-loss
        # observation; anything earlier is failure, not censoring (docs/exec-gate-audit.md).
        "loss_valid": bool(not execution_failure and exec_part["passed_last_obstacle"]),
    }


def rebuild_rows(run_dir: Path, specs: list[LaunchSpec], statuses: dict[str, str],
                 join_rows: list[dict]) -> list[dict]:
    by_key = {r["motion_key"]: r for r in join_rows}
    bodies: dict[str, tuple] = {}
    rows: list[dict] = []
    for spec in specs:
        if statuses[spec.name] != "complete":
            continue
        eval_dir = run_dir / "launches" / spec.name / "eval"
        schema = sonic_state_archive_schema(eval_dir)
        sample_dt = sonic_state_sample_dt(eval_dir)
        rollouts = {r.motion_key: r for r in load_sonic_state_rollouts(eval_dir)}
        t0 = time.time()
        for key in spec.motion_keys:
            join_row = by_key[key]
            if join_row["scene_id"] not in bodies:
                scene = scene_for(join_row)
                bodies[join_row["scene_id"]] = (scene, G1Body(scene))
            scene, body = bodies[join_row["scene_id"]]
            rows.append(join_rollout(join_row, rollouts[key], scene, body, launch=spec.name,
                                     physics_seed=spec.physics_seed, schema_version=schema,
                                     sample_dt_s=sample_dt))
        print(f"  joined {spec.name}: {len(spec.motion_keys)} rollouts, schema v{schema} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return rows


# ------------------------------------------------------------------------------- gate fit

def scene_split(scene_ids: set[str]) -> tuple[list[str], list[str]]:
    """Deterministic scene-level calibration/holdout split (RAMP: scene is the inference
    unit; conformal calibration must be split by scene, never by duplicated rows)."""
    calib, holdout = [], []
    for sid in sorted(scene_ids):
        digest = int(hashlib.sha1(sid.encode()).hexdigest(), 16)
        (calib if digest % 2 == 0 else holdout).append(sid)
    return calib, holdout


def fit_execution_gate(rows: list[dict], alpha: float = GATE_ALPHA,
                       target_m: float = TARGET_M) -> dict:
    """One-sided conformal upper bound on clearance loss, tau_alpha(d) per the audit doc."""
    calib_scenes, holdout_scenes = scene_split({r["scene_id"] for r in rows})
    obs = [r for r in rows if r["loss_valid"] and r["scene_id"] in set(calib_scenes)]
    out = {
        "alpha": alpha,
        "target_m": target_m,
        "dip_feature": "peak_dip_m",
        "calibration_scenes": calib_scenes,
        "holdout_scenes": holdout_scenes,
        "n_rows": len(rows),
        "n_calibration_obs": len(obs),
        "n_execution_failures": sum(r["execution_failure"] for r in rows),
    }
    if len(obs) < GATE_MIN_OBS or not holdout_scenes:
        out["status"] = "insufficient_data"
        return out

    d = np.asarray([r["peak_dip_m"] for r in obs], float)
    loss = np.asarray([r["loss_m"] for r in obs], float)
    if float(np.ptp(d)) > 1e-9:
        beta1, beta0 = np.polyfit(d, loss, 1)
    else:
        beta1, beta0 = 0.0, float(loss.mean())
    if beta1 < 0:  # the audit allows the monotone restriction only when the data support it
        beta1, beta0 = 0.0, float(loss.mean())
    residuals = loss - (beta0 + beta1 * d)
    # Finite-sample one-sided conformal quantile: ceil((n+1)(1-alpha))/n.
    n = len(residuals)
    rank = min(n, int(np.ceil((n + 1) * (1.0 - alpha))))
    q = float(np.sort(residuals)[rank - 1])

    def tau(dip: float) -> float:
        return max(0.0, beta0 + beta1 * dip + q)

    held = [r for r in rows if r["scene_id"] in set(holdout_scenes)]
    passed = [r for r in held if r["ref_min_overhead_m"] >= tau(r["peak_dip_m"])]
    passed_margin = [r for r in held
                     if r["ref_min_overhead_m"] >= target_m + tau(r["peak_dip_m"])]

    def false_safe(sub: list[dict]) -> float | None:
        if not sub:
            return None
        bad = sum(1 for r in sub if r["execution_failure"] or r["exec_collision_any"]
                  or not r["passed_last_obstacle"])
        return round(bad / len(sub), 4)

    out.update({
        "status": "fitted",
        "beta0": round(float(beta0), 5),
        "beta1": round(float(beta1), 5),
        "conformal_quantile": round(q, 5),
        "tau_at_dip": {f"{dip:.2f}": round(tau(dip), 5)
                       for dip in sorted({round(float(x), 2) for x in d})},
        "holdout": {
            "n_rows": len(held),
            "coverage_collision_free_gate": round(len(passed) / len(held), 4) if held else None,
            "false_safe_collision_free_gate": false_safe(passed),
            "coverage_margin_gate": round(len(passed_margin) / len(held), 4) if held else None,
            "false_safe_margin_gate": false_safe(passed_margin),
        },
    })
    return out


# ----------------------------------------------------------------------------------- main

def load_launch_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    records[rec["launch"]] = rec
                except (ValueError, KeyError):
                    continue
    return records


def assert_selection_matches(selected: list[dict], skipped: list[dict], join_path: Path,
                             receipt: dict | None) -> None:
    if receipt is not None:
        for key, want in [("methods", list(METHODS)), ("preference", PREFERENCE),
                          ("target_m", TARGET_M), ("progress_ok", PROGRESS_OK),
                          ("n_selected", len(selected)), ("n_skipped", len(skipped)),
                          ("n_unique_clips", len({r["clip_key"] for r in selected}))]:
            if receipt.get(key) != want:
                raise SystemExit(f"selection drift: receipt {key}={receipt.get(key)!r} "
                                 f"but this run derives {want!r}")
    if join_path.exists():
        existing = [json.loads(line) for line in join_path.read_text().splitlines()
                    if line.strip()]
        if len(existing) != len(selected):
            raise SystemExit(f"join.jsonl has {len(existing)} rows; derived {len(selected)}")
        for i, (a, b) in enumerate(zip(existing, selected)):
            if a != b:
                diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
                raise SystemExit(f"join.jsonl row {i} ({a.get('motion_key')}) differs from "
                                 f"the re-derived selection in fields {sorted(diff)}")
    else:
        with open(join_path, "w") as fh:
            for row in selected:
                fh.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="run")
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--dry-run", action="store_true",
                    help="repair run/ state, verify selection, print the plan; no SONIC")
    ap.add_argument("--timeout_s", type=int, default=2400)
    ap.add_argument("--alpha", type=float, default=GATE_ALPHA)
    args = ap.parse_args()
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ledger = json.load(open(args.ledger))
    cache = ClipCache(CACHE_DIR)
    selected, skipped = derive_selection(ledger, cache)
    specs = plan_launches(selected)
    receipt_path = run_dir / "receipt.json"
    receipt = json.load(open(receipt_path)) if receipt_path.exists() else None
    if receipt is not None:
        plan_want = [{"method": s.method, "chunk": s.chunk, "n_rows": len(s.rows)}
                     for s in specs]
        if receipt.get("launch_plan") != plan_want:
            raise SystemExit("launch plan drift vs run/receipt.json; refusing to resume")

    actions = repair_run_state(run_dir, specs)
    assert_selection_matches(selected, skipped, run_dir / "join.jsonl", receipt)
    with open(run_dir / "skipped.jsonl", "w") as fh:
        for row in skipped:
            fh.write(json.dumps(row) + "\n")

    records = load_launch_records(run_dir / "launches.jsonl")
    statuses, reasons = {}, {}
    for spec in specs:
        statuses[spec.name], reasons[spec.name] = launch_status(run_dir, spec, records)

    print(f"exp1b_execution_clearance: {len(selected)} rows / "
          f"{len({r['clip_key'] for r in selected})} unique clips / "
          f"{len({r['scene_id'] for r in selected})} scenes; {len(skipped)} skipped; "
          f"{len(specs)} launches (selection matches run/join.jsonl)")
    for a in actions:
        print(f"  repair: {a}")
    print(f"\n  {'launch':26s} {'rows':>4s}  {'status':10s}  detail")
    for spec in specs:
        print(f"  {spec.name:26s} {len(spec.rows):4d}  {statuses[spec.name]:10s}  "
              f"{reasons[spec.name]}")
    n_done = sum(1 for s in statuses.values() if s == "complete")
    print(f"\n  {n_done}/{len(specs)} launches complete; "
          f"{len(specs) - n_done} to (re)launch")

    if args.dry_run:
        print(f"\n--dry-run: no SONIC launched, no join rebuilt ({time.time()-t0:.1f}s)")
        return

    if not CKPT.exists():
        raise SystemExit(f"SONIC checkpoint not found at {CKPT}")
    mj_model = G1Body(None).model
    for spec in specs:
        if statuses[spec.name] == "complete":
            continue
        d = run_dir / "launches" / spec.name
        d.mkdir(parents=True, exist_ok=True)
        pkl = ensure_launch_pkl(spec, cache, mj_model, run_dir)
        started = time.time()
        rc, log = run_sonic(pkl, d / "eval", len(spec.rows), spec.physics_seed,
                            args.timeout_s)
        (d / "sonic.log").write_text(log)
        record = {"launch": spec.name, "method": spec.method, "chunk": spec.chunk,
                  "physics_seed": spec.physics_seed, "n_clips": len(spec.rows), "rc": rc,
                  "elapsed_s": round(time.time() - started, 1)}
        try:
            rollouts = load_sonic_state_rollouts(d / "eval")
            record["n_rollouts"] = len(rollouts)
            record["sample_dt_s"] = sonic_state_sample_dt(d / "eval")
        except Exception as exc:
            record["archive_error"] = str(exc)
        with open(run_dir / "launches.jsonl", "a") as fh:
            fh.write(json.dumps(record) + "\n")
        statuses[spec.name], reasons[spec.name] = launch_status(
            run_dir, spec, {spec.name: record})
        print(f"  {spec.name}: rc {rc} -> {statuses[spec.name]} "
              f"({record['elapsed_s']:.0f}s)", flush=True)

    rows = rebuild_rows(run_dir, specs, statuses, selected)
    tmp = run_dir / "rows.jsonl.tmp"
    with open(tmp, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    tmp.replace(run_dir / "rows.jsonl")
    print(f"wrote {len(rows)} rows -> {run_dir/'rows.jsonl'}")

    gate = fit_execution_gate(rows, alpha=args.alpha)
    gate["partial_campaign"] = any(s != "complete" for s in statuses.values())
    json.dump(gate, open(run_dir / "gate.json", "w"), indent=1)
    print(f"gate fit: {gate.get('status')} "
          f"({gate['n_calibration_obs']} calibration observations, "
          f"{gate['n_execution_failures']} execution failures) -> {run_dir/'gate.json'}")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
