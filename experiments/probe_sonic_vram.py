"""Operational probe: how much VRAM does one SONIC eval launch actually need?

**Not campaign evidence.** No seeds are spent, no campaign directory is touched and nothing here
may be cited as a result. Its only purpose is to replace the inherited launch thresholds in
``scene2motion.host_gate.SONIC_LAUNCH_GATE`` (12 GiB free VRAM, 18 GiB RAM, no Isaac co-tenant)
with a *measured* preset, the same way ``ARDY_GENERATION_GATE`` was measured on 2026-09-02.

It is written to be safe on a shared GPU:

* it refuses to launch unless free VRAM is above ``--safety-floor-mib``, so it never starts a
  launch that would immediately squeeze a co-tenant;
* it samples ``nvidia-smi`` once a second while SONIC runs and **kills its own launch** if free
  VRAM falls below ``--abort-free-mib``, so a co-tenant training job keeps its headroom;
* it launches into ``run/`` (machine-local, git-ignored) and writes only a small JSON report
  under ``outputs/probe_sonic_vram/``.

The motions are archived EXP-021 references, reused read-only. ``--num-envs`` is the quantity
under test: EXP-022A and the pending campaigns launch 32 environments at a time, so a preset may
only be lowered for the campaigns once 32 has been measured — a 2-environment measurement bounds
the Isaac baseline, not the campaign.

Run:  $S2M_PY experiments/probe_sonic_vram.py --num-envs 2
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments import exp1b_execution_clearance as exp1b  # noqa: E402
from experiments import exp028_termination_free_rollouts as e28  # noqa: E402
from scene2motion import host_gate  # noqa: E402
from scene2motion.sonic_export import write_motion_pkl  # noqa: E402

EXP021_QPOS = REPO / "outputs/exp021_elicited_lift_distribution_v2/qpos.npz"
RUN_DIR = REPO / "run/probe_sonic_vram"
OUT_DIR = REPO / "outputs/probe_sonic_vram"
POLL_S = 1.0


class ProbeRefusal(RuntimeError):
    """The host did not offer enough headroom to probe safely."""


def gpu_sample() -> dict[str, Any]:
    """One ``nvidia-smi`` sample: totals plus per-process usage."""
    out: dict[str, Any] = {"t": time.time()}
    totals = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20, check=False)
    if totals.returncode == 0 and totals.stdout.strip():
        used, free, total = (int(v.strip()) for v in totals.stdout.strip().splitlines()[0].split(","))
        out.update(used_mib=used, free_mib=free, total_mib=total)
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20, check=False)
    procs: dict[int, int] = {}
    if apps.returncode == 0:
        for line in apps.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                procs[int(parts[0])] = int(parts[1])
    out["processes"] = procs
    return out


def descendants(pid: int) -> set[int]:
    """``pid`` and every process below it, so the launch's own VRAM is attributed correctly."""
    found = {pid}
    try:
        listing = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True,
                                 timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return found
    children: dict[int, list[int]] = {}
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    stack = [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


class Monitor(threading.Thread):
    """Sample the GPU while the launch runs; abort the launch if the host gets tight."""

    def __init__(self, pid: int, abort_free_mib: int):
        super().__init__(daemon=True)
        self.pid = pid
        self.abort_free_mib = int(abort_free_mib)
        self.samples: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.aborted_reason: str | None = None

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                sample = gpu_sample()
            except (OSError, subprocess.SubprocessError, ValueError):
                self.stop_event.wait(POLL_S)
                continue
            ours = descendants(self.pid)
            sample["launch_mib"] = sum(v for p, v in sample.get("processes", {}).items()
                                       if p in ours)
            sample["other_mib"] = sum(v for p, v in sample.get("processes", {}).items()
                                      if p not in ours)
            self.samples.append(sample)
            free = sample.get("free_mib")
            if free is not None and free < self.abort_free_mib and self.aborted_reason is None:
                self.aborted_reason = (f"free VRAM {free} MiB fell below the abort floor "
                                       f"{self.abort_free_mib} MiB")
                try:
                    os.killpg(os.getpgid(self.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            self.stop_event.wait(POLL_S)

    def peak(self, key: str) -> int | None:
        values = [s[key] for s in self.samples if s.get(key) is not None]
        return int(max(values)) if values else None

    def minimum(self, key: str) -> int | None:
        values = [s[key] for s in self.samples if s.get(key) is not None]
        return int(min(values)) if values else None


def build_probe_pkl(num_motions: int, run_dir: Path) -> tuple[Path, list[str]]:
    """A motion pickle of ``num_motions`` archived EXP-021 references (read-only reuse)."""
    from scene2motion.robot import G1Body
    with np.load(EXP021_QPOS, allow_pickle=False) as archive:
        keys = sorted(archive.files)[:int(num_motions)]
        clips = {key: np.array(archive[key], copy=True) for key in keys}
    if len(clips) != int(num_motions):
        raise ProbeRefusal(f"archive has {len(clips)} clips, wanted {num_motions}")
    run_dir.mkdir(parents=True, exist_ok=True)
    pkl = write_motion_pkl(clips, run_dir / "probe_motions.pkl", fps=25,
                           mj_model=G1Body(None).model)
    return pkl, keys


def probe(*, num_envs: int, safety_floor_mib: int, abort_free_mib: int, timeout_s: int,
          run_dir: Path = RUN_DIR, dry_run: bool = False) -> dict[str, Any]:
    run_dir = Path(run_dir) / f"envs{int(num_envs)}"
    before = gpu_sample()
    ram = host_gate.query_available_ram_mib()
    isaac = host_gate.concurrent_isaac_processes()
    plan: dict[str, Any] = {
        "probe": "sonic_vram", "campaign_evidence": False,
        "num_envs": int(num_envs), "num_motions": int(num_envs),
        "safety_floor_mib": int(safety_floor_mib), "abort_free_mib": int(abort_free_mib),
        "timeout_s": int(timeout_s),
        "before": {"vram": {k: before.get(k) for k in ("used_mib", "free_mib", "total_mib")},
                   "ram": dict(ram),
                   "concurrent_isaac_processes": len(isaac)},
    }
    if dry_run:
        plan["status"] = "dry_run"
        return plan
    free = before.get("free_mib")
    if free is None or free < int(safety_floor_mib):
        raise ProbeRefusal(
            f"free VRAM {free} MiB is below the probe's safety floor {safety_floor_mib} MiB; "
            "refusing to launch beside a co-tenant")
    pkl, keys = build_probe_pkl(num_envs, run_dir)
    plan["motion_keys"] = keys
    eval_dir = run_dir / "eval"
    command = e28.build_sonic_command(pkl, eval_dir, int(num_envs), 0)
    plan["command"] = command
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=exp1b.SONIC, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, env=exp1b.sonic_env(),
                               stdin=subprocess.DEVNULL, start_new_session=True)
    monitor = Monitor(process.pid, abort_free_mib)
    monitor.start()
    timed_out = False
    try:
        log = process.communicate(timeout=int(timeout_s))[0]
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        log = process.communicate()[0]
    finally:
        monitor.stop_event.set()
        monitor.join(timeout=5)
    elapsed = time.monotonic() - started
    (run_dir / "sonic.log").write_text(log or "")
    plan.update({
        "status": ("aborted_to_protect_the_host" if monitor.aborted_reason else
                   "timeout" if timed_out else "complete" if process.returncode == 0 else "failed"),
        "returncode": process.returncode,
        "aborted_reason": monitor.aborted_reason,
        "elapsed_s": round(elapsed, 1),
        "n_samples": len(monitor.samples),
        "peak_launch_mib": monitor.peak("launch_mib"),
        "peak_total_used_mib": monitor.peak("used_mib"),
        "min_free_mib": monitor.minimum("free_mib"),
        "peak_other_process_mib": monitor.peak("other_mib"),
        "after": {k: v for k, v in gpu_sample().items() if k != "processes"},
        "log_tail": (log or "").strip().splitlines()[-12:],
        "log_path": str(run_dir / "sonic.log"),
    })
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--safety-floor-mib", type=int, default=4500,
                        help="refuse to launch unless at least this much VRAM is free")
    parser.add_argument("--abort-free-mib", type=int, default=700,
                        help="kill the launch if free VRAM falls below this while it runs")
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--run-dir", default=str(RUN_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = probe(num_envs=args.num_envs, safety_floor_mib=args.safety_floor_mib,
                       abort_free_mib=args.abort_free_mib, timeout_s=args.timeout_s,
                       run_dir=Path(args.run_dir), dry_run=args.dry_run)
    except ProbeRefusal as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, indent=2))
        return 2
    if not args.dry_run:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"report_envs{args.num_envs}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "command"}, indent=2,
                     sort_keys=True))
    return 0 if report.get("status") in {"complete", "dry_run"} else 1


if __name__ == "__main__":
    sys.exit(main())
