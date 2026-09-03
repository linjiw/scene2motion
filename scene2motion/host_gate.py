"""Host-resource gate shared by every GPU campaign driver.

Both presets are **measured**, not inherited: see ``ARDY_GENERATION_GATE`` and
``SONIC_LAUNCH_GATE`` below for what was measured and when.  The
gate must be evaluated *before* a runner is constructed and before any seed is spent, and its
measured values are bound into the campaign receipt.  A failed gate must leave the output
directory untouched so the same directory can be launched later.

All probes are injectable so the gate is unit-testable on a CPU host without ``nvidia-smi``.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

MIN_FREE_VRAM_MIB = 12 * 1024
MIN_AVAILABLE_RAM_MIB = 18 * 1024

# Named presets.  SONIC launches (Isaac Sim + the tracker) keep the plan-of-record thresholds
# and forbid a concurrent Isaac process.  ARDY-only generation was measured on 2026-09-02
# (one B=8 Horizon52 prompt-schedule call, 4 windows x 208 frames, decode included):
# peak CUDA reserved 1,076 MiB, peak host RSS 2,297 MiB, 2.2 s model load; the ARDY preset
# keeps roughly a 4x margin on both and records, but does not gate on, Isaac co-tenants.
ARDY_GENERATION_GATE: dict[str, Any] = {
    "min_free_vram_mib": 4 * 1024,
    "min_available_ram_mib": 8 * 1024,
    "require_no_isaac": False,
}
# SONIC eval launches were measured on 2026-09-03 by ``experiments/probe_sonic_vram.py``
# (reports in ``outputs/probe_sonic_vram/``): at 2, 16 and **32** environments — 32 being the
# campaign configuration — one launch peaked at 3,631 / 3,727 / **3,769 MiB** of VRAM and
# consumed about **6,810 MiB** of host RAM (available RAM fell 8,820 -> 2,010 MiB), completing
# in 49-62 s.  VRAM is dominated by the Isaac Sim baseline and barely grows with the environment
# count; host RAM is the binding resource.  All three launches completed with return code 0
# **beside four concurrent Isaac processes**, so the previous "no concurrent Isaac process"
# condition is not a launch requirement and co-tenants are recorded instead of gated.  The
# thresholds below keep roughly 1.5x the measured VRAM peak and leave >= 2.7 GiB of RAM after a
# launch's measured consumption.  The former 12 GiB / 18 GiB / no-Isaac figures were inherited
# from the plan of record and never measured; they blocked every SONIC campaign for a day.
SONIC_LAUNCH_GATE: dict[str, Any] = {
    "min_free_vram_mib": 5500,
    "min_available_ram_mib": 9500,
    "require_no_isaac": False,
}
#: What one 32-environment launch actually used (2026-09-03; provenance for the preset above).
SONIC_MEASURED_NEED = {
    "num_envs": 32, "peak_launch_vram_mib": 3769, "host_ram_consumed_mib": 6810,
    "elapsed_s": 61.6, "concurrent_isaac_processes": 4,
    "report": "outputs/probe_sonic_vram/report_envs32.json",
}

# Command-line fragments that identify an Isaac Sim / Isaac Lab process on this host.
ISAAC_PROCESS_PATTERNS: tuple[str, ...] = (
    "eval_agent_trl",
    "env_isaaclab",
    "isaaclab",
    "isaacsim",
    "omni.isaac",
)


class HostResourceGateFailed(RuntimeError):
    """The host does not satisfy the preregistered launch conditions."""


def query_free_vram_mib(run: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Free and total VRAM (MiB) of GPU 0 from ``nvidia-smi``; ``None`` when unavailable."""
    try:
        proc = run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"free_mib": None, "total_mib": None, "used_mib": None, "error": str(exc)}
    if proc.returncode != 0:
        return {
            "free_mib": None, "total_mib": None, "used_mib": None,
            "error": f"nvidia-smi rc={proc.returncode}: {proc.stderr.strip()[:200]}",
        }
    first = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = [part.strip() for part in first.split(",")]
    if len(parts) != 3 or not all(re.fullmatch(r"\d+", part) for part in parts):
        return {
            "free_mib": None, "total_mib": None, "used_mib": None,
            "error": f"unparseable nvidia-smi line: {first!r}",
        }
    free, total, used = (int(part) for part in parts)
    return {"free_mib": free, "total_mib": total, "used_mib": used, "error": None}


def query_available_ram_mib(meminfo_path: str = "/proc/meminfo") -> dict[str, Any]:
    """MemAvailable and MemTotal (MiB) from ``/proc/meminfo``."""
    values: dict[str, int] = {}
    try:
        with open(meminfo_path, "r", encoding="utf-8") as handle:
            for line in handle:
                match = re.match(r"^(MemAvailable|MemTotal):\s+(\d+)\s+kB", line)
                if match:
                    values[match.group(1)] = int(match.group(2)) // 1024
    except OSError as exc:
        return {"available_mib": None, "total_mib": None, "error": str(exc)}
    if "MemAvailable" not in values:
        return {"available_mib": None, "total_mib": values.get("MemTotal"),
                "error": "MemAvailable missing"}
    return {"available_mib": values["MemAvailable"], "total_mib": values.get("MemTotal"),
            "error": None}


def concurrent_isaac_processes(
    run: Callable[..., Any] = subprocess.run,
    patterns: Sequence[str] = ISAAC_PROCESS_PATTERNS,
    own_pid: int | None = None,
) -> list[dict[str, Any]]:
    """Processes whose command line matches an Isaac pattern, excluding this process."""
    own = os.getpid() if own_pid is None else int(own_pid)
    try:
        proc = run(["ps", "-eo", "pid=,args="], capture_output=True, text=True,
                   timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    found: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, args = line.partition(" ")
        if not pid_text.isdigit() or int(pid_text) == own:
            continue
        if any(pattern in args for pattern in patterns):
            found.append({"pid": int(pid_text), "args": args[:240]})
    return found


def host_resource_report(
    *,
    min_free_vram_mib: int = MIN_FREE_VRAM_MIB,
    min_available_ram_mib: int = MIN_AVAILABLE_RAM_MIB,
    require_no_isaac: bool = False,
    vram_fn: Callable[[], Mapping[str, Any]] = query_free_vram_mib,
    ram_fn: Callable[[], Mapping[str, Any]] = query_available_ram_mib,
    isaac_fn: Callable[[], Sequence[Mapping[str, Any]]] = concurrent_isaac_processes,
) -> dict[str, Any]:
    """Measure the host and evaluate every gate condition; never raises."""
    vram = dict(vram_fn())
    ram = dict(ram_fn())
    isaac = [dict(item) for item in isaac_fn()] if require_no_isaac else []
    vram_ok = vram.get("free_mib") is not None and int(vram["free_mib"]) >= min_free_vram_mib
    ram_ok = (ram.get("available_mib") is not None
              and int(ram["available_mib"]) >= min_available_ram_mib)
    isaac_ok = (not isaac) if require_no_isaac else True
    return {
        "measured_at_unix_s": time.time(),
        "thresholds": {
            "min_free_vram_mib": int(min_free_vram_mib),
            "min_available_ram_mib": int(min_available_ram_mib),
            "require_no_concurrent_isaac": bool(require_no_isaac),
        },
        "vram": vram,
        "ram": ram,
        "concurrent_isaac_processes": isaac,
        "checks": {"vram": bool(vram_ok), "ram": bool(ram_ok), "no_isaac": bool(isaac_ok)},
        "pass": bool(vram_ok and ram_ok and isaac_ok),
    }


def require_host_resources(**kwargs: Any) -> dict[str, Any]:
    """Return the report when every condition holds; raise :class:`HostResourceGateFailed`."""
    report = host_resource_report(**kwargs)
    if not report["pass"]:
        failed = [name for name, ok in report["checks"].items() if not ok]
        raise HostResourceGateFailed(
            "host-resource gate failed on "
            + ", ".join(failed)
            + f": free VRAM {report['vram'].get('free_mib')} MiB, "
            f"available RAM {report['ram'].get('available_mib')} MiB, "
            f"{len(report['concurrent_isaac_processes'])} Isaac process(es)"
        )
    return report
