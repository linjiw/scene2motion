"""CPU tests for the shared host-resource gate."""

from __future__ import annotations

import subprocess

import pytest

from scene2motion import host_gate as hg


class _Proc:
    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "") -> None:
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def test_query_free_vram_parses_nvidia_smi_csv() -> None:
    run = lambda *a, **k: _Proc("4436, 16303, 11867\n")  # noqa: E731
    got = hg.query_free_vram_mib(run=run)
    assert got == {"free_mib": 4436, "total_mib": 16303, "used_mib": 11867, "error": None}


def test_query_free_vram_reports_missing_tool_as_none() -> None:
    def run(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    got = hg.query_free_vram_mib(run=run)
    assert got["free_mib"] is None and "nvidia-smi" in got["error"]


def test_query_free_vram_rejects_unparseable_output() -> None:
    got = hg.query_free_vram_mib(run=lambda *a, **k: _Proc("N/A, N/A, N/A\n"))
    assert got["free_mib"] is None and "unparseable" in got["error"]


def test_query_available_ram_reads_meminfo(tmp_path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       30000000 kB\nMemFree: 1 kB\nMemAvailable:   20480000 kB\n")
    got = hg.query_available_ram_mib(str(meminfo))
    assert got == {"available_mib": 20000, "total_mib": 29296, "error": None}


def test_concurrent_isaac_processes_excludes_self_and_matches_patterns() -> None:
    ps = (
        "  100 /usr/bin/bash\n"
        "  200 /home/x/isaaclab-install/env_isaaclab/bin/python train.py\n"
        "  300 python -m gear_sonic.eval_agent_trl +checkpoint=a\n"
        "  400 python experiments/exp023b.py\n"
    )
    found = hg.concurrent_isaac_processes(run=lambda *a, **k: _Proc(ps), own_pid=300)
    assert [item["pid"] for item in found] == [200]


def test_report_passes_when_every_threshold_holds() -> None:
    report = hg.host_resource_report(
        vram_fn=lambda: {"free_mib": 15000, "total_mib": 16303, "used_mib": 1303, "error": None},
        ram_fn=lambda: {"available_mib": 20000, "total_mib": 29296, "error": None},
        isaac_fn=lambda: [],
        require_no_isaac=True,
    )
    assert report["pass"] and report["checks"] == {"vram": True, "ram": True, "no_isaac": True}
    assert report["thresholds"]["min_free_vram_mib"] == 12 * 1024
    assert report["thresholds"]["min_available_ram_mib"] == 18 * 1024


def test_report_fails_on_vram_ram_or_isaac() -> None:
    base = dict(
        vram_fn=lambda: {"free_mib": 15000, "total_mib": 16303, "used_mib": 0, "error": None},
        ram_fn=lambda: {"available_mib": 20000, "total_mib": 29296, "error": None},
        isaac_fn=lambda: [{"pid": 1, "args": "env_isaaclab"}],
    )
    assert hg.host_resource_report(**base, require_no_isaac=False)["pass"]
    assert not hg.host_resource_report(**base, require_no_isaac=True)["checks"]["no_isaac"]
    low_vram = dict(base, vram_fn=lambda: {"free_mib": 4436, "total_mib": 16303,
                                            "used_mib": 11867, "error": None})
    assert not hg.host_resource_report(**low_vram)["checks"]["vram"]
    low_ram = dict(base, ram_fn=lambda: {"available_mib": 8000, "total_mib": 29296, "error": None})
    assert not hg.host_resource_report(**low_ram)["checks"]["ram"]
    unknown = dict(base, vram_fn=lambda: {"free_mib": None, "total_mib": None, "used_mib": None,
                                           "error": "no nvidia-smi"})
    assert not hg.host_resource_report(**unknown)["pass"]


def test_require_host_resources_raises_with_measured_values() -> None:
    with pytest.raises(hg.HostResourceGateFailed, match="vram, ram.*4436 MiB.*8000 MiB"):
        hg.require_host_resources(
            vram_fn=lambda: {"free_mib": 4436, "total_mib": 16303, "used_mib": 11867,
                             "error": None},
            ram_fn=lambda: {"available_mib": 8000, "total_mib": 29296, "error": None},
            isaac_fn=lambda: [],
        )


def test_live_probes_do_not_raise() -> None:
    # The live host may or may not have a GPU; the probes must degrade to None, never raise.
    hg.query_free_vram_mib()
    hg.query_available_ram_mib()
    hg.concurrent_isaac_processes()
    assert isinstance(hg.host_resource_report(require_no_isaac=True)["pass"], bool)
    assert subprocess is not None
