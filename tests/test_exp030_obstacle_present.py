"""CPU tests for the EXP-030 obstacle-present tracking driver.

Everything here is self-contained: the archived EXP-021 pool, the EXP-022A comparison rows, the
tracker identity, the host gate and SONIC itself are injected as fakes, so the suite runs in a
bare checkout (``outputs/**/motions.pkl`` is gitignored and must never be a test dependency).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from experiments import calibrate_ramp_route_phase as cal
from experiments import exp022_exact_tracking_bridge as exp022
from experiments import exp028_termination_free_rollouts as e28
from experiments import exp030_obstacle_present as x
from scene2motion import host_gate as hg
from scene2motion import traversal_eval as te
from scene2motion.sonic_state_export import SonicRollout, write_sonic_state_archive

FULL_LEN = 397  # EXP-022A survivors: ref_len - 1 archived samples
CUT_LEN = 60


# ------------------------------------------------------------------------------ fakes

def _clean_code_state(_repo: Path) -> dict:
    return {"commit": "test-commit", "dirty": False, "status": [], "tracked_diff_sha256": "0" * 64}


def _dirty_code_state(_repo: Path) -> dict:
    return {"commit": "dirty-test", "dirty": True, "status": ["?? note"],
            "tracked_diff_sha256": "0" * 64}


def _fix_report(present: bool = True) -> dict:
    return {"file": x.ADD_TABLE_FIX_FILE, "sha256": "f" * 64, "fix_present": present,
            "problems": [] if present else ["the sensor assignment is not inside the branch"],
            "sensor_line": 758, "sensor_indent": 16, "add_object_branch_line": 717,
            "add_object_branch_indent": 12}


FAKE_CHECKPOINT = "/fake/legacy/sonic_release/last.pt"


def _tracker_identity(fix_present: bool = True):
    def identity(sonic_root=None) -> dict:
        return {
            "root": str(sonic_root or x.SONIC_EXP029_ROOT),
            "branch": x.SONIC_EXP029_BRANCH,
            "expected_branch": x.SONIC_EXP029_BRANCH,
            "git": {"commit": x.ADD_TABLE_FIX_COMMIT, "dirty": False, "tracked_dirty": False,
                    "status": []},
            "contains_add_table_fix_commit": {"commit": x.ADD_TABLE_FIX_COMMIT,
                                              "contained": True},
            "dirty_paths": [],
            "guarded_dirty_paths": [],
            "core_source_sha256": {"eval.py": "a" * 64},
            "core_source_manifest_sha256": "b" * 64,
            "exp022a_core_source_manifest_sha256": x.EXP022A_CORE_MANIFEST_SHA256,
            "core_source_manifest_matches_exp022a": False,
            "legacy_checkout": {"root": str(x.LEGACY_SONIC_ROOT), "core_source_sha256": {},
                                "files_differing_from_execution_root": [x.ADD_TABLE_FIX_FILE]},
            "evaluator_source_sha256": {"terminations.py": "c" * 64},
            "add_table_fix": _fix_report(fix_present),
            "release_bundle": {"root": "/fake/legacy", "source": "legacy_checkout_release_bundle",
                               "checkpoint_path": FAKE_CHECKPOINT,
                               "config_path": "/fake/legacy/sonic_release/config.yaml"},
            "checkpoint": {"path": FAKE_CHECKPOINT, "sha256": x.EXPECTED_CHECKPOINT_SHA256,
                           "source": "legacy_checkout_release_bundle"},
            "python_runtime": {"packages": {"torch": "test"}},
            "isaaclab": {"git": {"commit": "isaac-test"}},
        }
    return identity


def _protocol(status: str = "preregistered"):
    def identity() -> dict:
        return {"path": "/fake/protocol.md", "sha256": "d" * 64, "status": status}
    return identity


def _gate_pass(**_kwargs) -> dict:
    return {"pass": True, "checks": {"vram": True, "ram": True, "no_isaac": True},
            "vram": {"free_mib": 15000}, "ram": {"available_mib": 24000},
            "concurrent_isaac_processes": []}


def _gate_fail(**_kwargs) -> dict:
    raise hg.HostResourceGateFailed("host-resource gate failed on vram: free VRAM 4436 MiB, "
                                    "available RAM 8000 MiB, 2 Isaac process(es)")


def _gate_after(n_passes: int):
    state = {"calls": 0}

    def gate(**kwargs):
        state["calls"] += 1
        if state["calls"] > n_passes:
            return _gate_fail(**kwargs)
        return _gate_pass(**kwargs)
    return gate


def _isaac_none(**_kwargs) -> list:
    return [{"pid": 4242, "args": "python -m gear_sonic.eval_agent_trl"}]


def _walk_qpos(n: int, x_end: float) -> np.ndarray:
    q = np.zeros((n, 36), dtype=np.float32)
    q[:, 0] = np.linspace(0.0, x_end, n)
    q[:, 2] = 0.78
    q[:, 3] = 1.0
    return q


def _fake_clips() -> dict[str, np.ndarray]:
    return {f"s{seed}": _walk_qpos(x.N_FRAMES, 2.0) for seed in x.POOL_SEEDS}


def _fake_source(_source_dir) -> dict:
    return {"identity": {"path": "/fake/exp021", "qpos_content_sha256": "e" * 64,
                         "pool_seeds": list(x.POOL_SEEDS)},
            "rows": [], "clips": _fake_clips()}


def _fake_exp022a_identity(_directory) -> dict:
    return {"path": "/fake/exp022a", "receipt_sha256": "1" * 64, "physics_seed": 0}


def _absent_terminated(seed_index: int) -> bool:
    """The fake launcher's rule for the obstacle-absent arm, per chunk-local motion id."""
    return (seed_index % exp022.CHUNK_SIZE) % 8 == 0


#: EXP-022A's fake rows disagree with the `absent` arm on exactly these four references.
FLIPPED = {1, 2, 3, 4}


def _fake_exp022a_rows(_directory) -> dict:
    rows = {}
    for index, seed in enumerate(x.POOL_SEEDS):
        terminated = _absent_terminated(index)
        if index in FLIPPED:
            terminated = not terminated
        rows[f"s{seed}"] = {"motion_key": f"s{seed}", "tracker_terminated": terminated,
                            "valid_frames": CUT_LEN if terminated else FULL_LEN}
    return {"rows": rows, "path": "/fake/exp022a/achieved_rows.jsonl", "sha256": "2" * 64}


def _fake_worktree(out: Path) -> Path:
    """A stand-in for the patched worktree: any directory that is not the legacy checkout."""
    root = Path(out).parent / "GR00T-WBC-exp029"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fake_export(clips, path, fps=25, mj_model=None):
    payload = {key: {"root_trans_offset": np.asarray(q[:, :3], dtype=np.float32),
                     "test_qpos": np.asarray(q, dtype=np.float32), "fps": fps}
               for key, q in clips.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def _termination_table(names_time_out: dict[str, bool]) -> str:
    rows = "\n".join(f"|   {i}   | {name:<15} |  {str(flag):<6}  |"
                     for i, (name, flag) in enumerate(names_time_out.items()))
    return (f"[INFO] Termination Manager:  <TerminationManager> contains {len(names_time_out)} "
            "active terms.\n+------------------------------------+\n"
            "|      Active Termination Terms      |\n+-------+-----------------+----------+\n"
            "| Index | Name            | Time Out |\n+-------+-----------------+----------+\n"
            f"{rows}\n+-------+-----------------+----------+\n")


RELEASE_TABLE = {"time_out": True, "anchor_pos": False, "anchor_ori_full": False,
                 "ee_body_pos": False, "foot_pos_xyz": False}


def _fake_sonic_artifacts(eval_dir: Path, records: list[SonicRollout],
                          table: dict[str, bool] | None = None) -> str:
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_sonic_state_archive(records, eval_dir / "achieved_qpos.npz", sample_dt_s=0.02)
    success_rate = float(np.mean([not r.terminated for r in records]))
    progress_rate = float(np.mean([r.progress for r in records]))
    cal._write_json(eval_dir / "metrics_eval.json", {
        "eval/all_metrics_dict": {"motion_keys": [r.motion_key for r in records],
                                  "terminated": [r.terminated for r in records],
                                  "progress": [r.progress for r in records]},
        "failed_keys": [r.motion_key for r in records if r.terminated],
        "eval/success/success_rate": success_rate,
        "eval/success/progress_rate": progress_rate,
    })
    header = _termination_table(dict(table or RELEASE_TABLE))
    return f"{header}\nSuccess Rate:{success_rate:.10f}\nProgress Rate:{progress_rate:.10f}\n"


def _make_launcher(calls: list, *, table: dict[str, bool] | None = None):
    """Obstacle-absent rollouts walk the route; obstacle-present rollouts stop at the box."""
    def launch(pkl, eval_dir, num_envs, physics_seed, timeout_s, extra_overrides):
        with Path(pkl).open("rb") as handle:
            motions = pickle.load(handle)
        present = any("add_table=true" in item for item in extra_overrides)
        calls.append({"keys": list(motions), "num_envs": num_envs, "physics_seed": physics_seed,
                      "extra_overrides": list(extra_overrides), "eval_dir": str(eval_dir),
                      "table_pos": next(iter(motions.values())).get("table_pos"),
                      "present": present})
        records = []
        for motion_id, key in enumerate(motions):
            if present:
                terminated, valid, x_end, progress = True, 120, 1.0, 0.15
            elif _absent_terminated(motion_id):
                terminated, valid, x_end, progress = True, CUT_LEN, 0.9, 0.3
            else:
                terminated, valid, x_end, progress = False, FULL_LEN, 7.2, 1.0
            records.append(SonicRollout(key, _walk_qpos(valid, x_end), valid, terminated,
                                        progress, motion_id))
        return 0, _fake_sonic_artifacts(Path(eval_dir), records, table)
    return launch


def _fake_collision(scene, qpos) -> dict:
    """A trajectory that runs past 3 m is treated as having hit the box; nothing else collides."""
    q = np.asarray(qpos, dtype=float)
    hit = bool(len(scene.boxes)) and float(q[:, 0].max()) >= 3.0
    return {"collision_free": not hit, "penetration_frames": 3 if hit else 0,
            "max_penetration_m": 0.02 if hit else 0.0,
            "min_clearance_m": -0.02 if hit else 0.31,
            "worst": {"frame": 7, "depth_m": 0.02 if hit else 0.0},
            "first": {"frame": 5, "depth_m": 0.01 if hit else 0.0}}


def _campaign_kwargs(out: Path, calls: list, **overrides):
    kwargs = dict(
        out=out, sonic_root=_fake_worktree(out), launch_fn=_make_launcher(calls),
        export_fn=_fake_export,
        host_gate_fn=_gate_pass, host_report_fn=_gate_pass, isaac_fn=_isaac_none,
        code_state_fn=_clean_code_state, tracker_identity_fn=_tracker_identity(),
        protocol_identity_fn=_protocol(), source_fn=_fake_source,
        exp022a_identity_fn=_fake_exp022a_identity, exp022a_rows_fn=_fake_exp022a_rows,
        collision_fn=_fake_collision, mj_model=object(),
    )
    kwargs.update(overrides)
    return kwargs


# ------------------------------------------------------------------------------ launch plan

def test_launch_plan_is_six_launches_over_exp022a_chunks():
    plan = x.launch_plan()
    chunks = exp022.chunk_plan()
    assert [spec["name"] for spec in plan] == [
        "absent_chunk00_seed0", "absent_chunk01_seed0",
        "present_05_chunk00_seed0", "present_05_chunk01_seed0",
        "present_20_chunk00_seed0", "present_20_chunk01_seed0"]
    assert [spec["arm"] for spec in plan] == ["absent", "absent", "present_05", "present_05",
                                              "present_20", "present_20"]
    for spec in plan:
        chunk = chunks[spec["chunk"]]
        assert spec["motion_keys"] == chunk["motion_keys"]
        assert spec["seeds"] == chunk["seeds"]
        assert spec["n_motions"] == 32 and spec["physics_seed"] == 0
    covered = [key for spec in plan if spec["arm"] == "absent" for key in spec["motion_keys"]]
    assert covered == [f"s{seed}" for seed in x.POOL_SEEDS]
    assert [spec["box_height_m"] for spec in plan] == [None, None, 0.05, 0.05, 0.20, 0.20]
    assert [spec["obstacle_in_physics"] for spec in plan] == [False, False, True, True, True, True]


def test_arm_overrides_are_the_probe_proven_list():
    shared = ["++manager_env.config.env_spacing=12.0",
              "++manager_env.config.episode_length_s=20.0"]
    assert x.arm_overrides(None) == shared
    assert x.arm_overrides(0.05) == shared + [
        "++manager_env.config.add_table=true",
        "++manager_env.config.table_size=[0.2, 2.8, 0.05]",
        "++manager_env.config.table_position=[1.2, 0.0, 0.025]"]
    assert x.arm_overrides(0.20) == shared + [
        "++manager_env.config.add_table=true",
        "++manager_env.config.table_size=[0.2, 2.8, 0.2]",
        "++manager_env.config.table_position=[1.2, 0.0, 0.1]"]
    # The physics box is the geometry the collision model scores: step_scene's full width.
    assert x.OBSTACLE_WIDTH_M == pytest.approx(2 * 1.4)
    assert x.OBSTACLE_DEPTH_M == 0.20 and x.OBSTACLE_X_M == 1.2


def test_command_line_carries_the_arm_overrides_and_the_explicit_checkpoint():
    from experiments import exp1b_execution_clearance as exp1b
    pkl, eval_dir = Path("/tmp/m.pkl"), Path("/tmp/eval")
    absent = x.build_sonic_command(pkl, eval_dir, 32, 0, x.arm_overrides(None),
                                   checkpoint=FAKE_CHECKPOINT)
    present = x.build_sonic_command(pkl, eval_dir, 32, 0, x.arm_overrides(0.05),
                                    checkpoint=FAKE_CHECKPOINT)
    assert absent[:4] == [str(exp1b.SONIC_PY), "-u", "-m", "gear_sonic.eval_agent_trl"]
    assert absent[4] == f"+checkpoint={FAKE_CHECKPOINT}"
    assert "+manager_env/terminations=tracking/eval" in absent and "++seed=0" in absent
    assert "++num_envs=32" in absent
    assert present[:len(absent)] == absent
    assert present[len(absent):] == ["++manager_env.config.add_table=true",
                                     "++manager_env.config.table_size=[0.2, 2.8, 0.05]",
                                     "++manager_env.config.table_position=[1.2, 0.0, 0.025]"]


# ------------------------------------------------------------------------ the two checkouts

def test_execution_root_refuses_the_legacy_checkout(tmp_path):
    with pytest.raises(x.CampaignAbort, match="refuses the legacy tracker checkout"):
        x.require_execution_root(x.LEGACY_SONIC_ROOT)
    with pytest.raises(x.CampaignAbort, match="worktree is missing"):
        x.require_execution_root(tmp_path / "not-a-checkout")
    worktree = tmp_path / "GR00T-WBC-exp029"
    worktree.mkdir()
    assert x.require_execution_root(worktree) == worktree.resolve()
    assert x.LEGACY_SONIC_ROOT != x.SONIC_EXP029_ROOT


def test_campaign_refuses_the_legacy_checkout_as_its_execution_root(tmp_path):
    calls: list = []
    out = tmp_path / "campaign"
    with pytest.raises(x.CampaignAbort, match="refuses the legacy tracker checkout"):
        x.run_campaign(stage="launch", **_campaign_kwargs(
            out, calls, sonic_root=x.LEGACY_SONIC_ROOT))
    assert not out.exists() and calls == []


def _git(repo: Path, *args: str) -> str:
    import subprocess
    return subprocess.check_output(["git", *args], cwd=repo, text=True,
                                   stderr=subprocess.DEVNULL).strip()


def test_commit_contains_reads_real_git_ancestry(tmp_path):
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("one\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "first")
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "a.txt").write_text("two\n")
    _git(repo, "commit", "-qam", "second")
    assert x._commit_contains(repo, first) is True
    assert x._commit_contains(repo, _git(repo, "rev-parse", "HEAD")) is True
    _git(repo, "checkout", "-q", "-b", "side", first)
    (repo / "b.txt").write_text("side\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-qm", "side")
    side = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    assert x._commit_contains(repo, side) is False, "a divergent commit is not contained"
    assert x._git_branch(repo) == "main"
    with pytest.raises(ValueError, match="could not resolve"):
        x._commit_contains(repo, "0" * 40)
    subprocess.run(["true"], check=False)


def test_release_bundle_is_resolved_from_the_worktree_or_the_legacy_checkout(tmp_path, monkeypatch):
    worktree = tmp_path / "GR00T-WBC-exp029"
    legacy = tmp_path / "GR00T-WholeBodyControl"
    (worktree / "gear_sonic").mkdir(parents=True)
    (legacy / "sonic_release").mkdir(parents=True)
    (legacy / "sonic_release/last.pt").write_bytes(b"ckpt")
    (legacy / "sonic_release/config.yaml").write_text("terrain_type: trimesh\n")
    monkeypatch.setattr(x, "LEGACY_SONIC_ROOT", legacy)

    # The worktree does not carry the gitignored bundle: it comes from the legacy checkout.
    bundle = x.resolve_release_bundle(worktree)
    assert bundle["source"] == "legacy_checkout_release_bundle"
    assert bundle["checkpoint_path"] == str((legacy / "sonic_release/last.pt").resolve())

    # Once the bundle is present in the execution root, that copy wins.
    (worktree / "sonic_release").mkdir()
    (worktree / "sonic_release/last.pt").write_bytes(b"ckpt")
    (worktree / "sonic_release/config.yaml").write_text("terrain_type: trimesh\n")
    assert x.resolve_release_bundle(worktree)["source"] == "execution_root"

    monkeypatch.setattr(x, "LEGACY_SONIC_ROOT", tmp_path / "nowhere")
    with pytest.raises(ValueError, match="no SONIC release bundle"):
        x.resolve_release_bundle(tmp_path / "empty")


def test_core_source_hashes_take_each_file_from_its_own_root(tmp_path):
    worktree = tmp_path / "wt"
    legacy = tmp_path / "legacy"
    for name in x.CORE_SONIC_FILES:
        base = legacy if name.startswith(x.RELEASE_BUNDLE_PREFIX) else worktree
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"content of {name}\n")
    bundle = {"root": str(legacy)}
    hashes, provenance = x.core_source_hashes(worktree, bundle)
    assert set(hashes) == set(x.CORE_SONIC_FILES)
    assert provenance["sonic_release/config.yaml"].startswith(str(legacy))
    assert provenance[x.ADD_TABLE_FIX_FILE].startswith(str(worktree))
    (worktree / x.ADD_TABLE_FIX_FILE).unlink()
    with pytest.raises(ValueError, match="required tracker source is missing"):
        x.core_source_hashes(worktree, bundle)


def test_subprocess_env_puts_the_execution_checkout_first(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("PYTHONPATH", "/some/other/path")
    env = x.sonic_subprocess_env(tmp_path)
    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(tmp_path.resolve())
    assert "/some/other/path" in entries


# ------------------------------------------------------------------------------ table metadata

def test_table_metadata_is_written_into_every_motion(tmp_path):
    clips = {key: _walk_qpos(x.N_FRAMES, 2.0) for key in ("s4400", "s4401")}
    for height in (0.05, 0.20):
        table = x.table_spec(height)
        assert table["pos"] == [1.2, 0.0, height / 2.0]  # the CENTRE, so z = h/2
        assert table["quat"] == [1.0, 0.0, 0.0, 0.0]     # identity, w-first
        assert table["size_xyz"] == [0.2, 2.8, height]   # full x/y/z extents
        path = x.write_arm_motion_pkl(clips, tmp_path / f"h{height}.pkl", table,
                                      export_fn=_fake_export)
        with path.open("rb") as handle:
            motions = pickle.load(handle)
        assert set(motions) == set(clips)
        for entry in motions.values():
            assert entry["table_pos"] == [1.2, 0.0, height / 2.0]
            assert entry["table_quat"] == [1.0, 0.0, 0.0, 0.0]
        x.validate_motion_pkl(path, list(clips), table)

    assert x.table_spec(None) is None
    absent = x.write_arm_motion_pkl(clips, tmp_path / "absent.pkl", None, export_fn=_fake_export)
    with absent.open("rb") as handle:
        motions = pickle.load(handle)
    assert all("table_pos" not in entry and "table_quat" not in entry
               for entry in motions.values())
    x.validate_motion_pkl(absent, list(clips), None)

    with pytest.raises(ValueError, match="carries table metadata"):
        x.validate_motion_pkl(tmp_path / "h0.05.pkl", list(clips), None)
    with pytest.raises(ValueError, match="table_pos"):
        x.validate_motion_pkl(absent, list(clips), x.table_spec(0.05))
    moved = x.table_spec(0.05)
    moved["pos"] = [3.6, 0.0, 0.025]
    with pytest.raises(ValueError, match="table_pos"):
        x.validate_motion_pkl(tmp_path / "h0.05.pkl", list(clips), moved)


def test_ensure_motion_pkl_is_deterministic_and_arm_specific(tmp_path):
    clips = _fake_clips()
    plan = {spec["arm"]: spec for spec in x.launch_plan() if spec["chunk"] == 0}
    paths = {}
    for arm, spec in plan.items():
        paths[arm] = x.ensure_motion_pkl(spec, clips, tmp_path, export_fn=_fake_export)
        # A second call must adopt the identical bytes rather than rewrite them.
        again = x.ensure_motion_pkl(spec, clips, tmp_path, export_fn=_fake_export)
        assert again == paths[arm]
    shas = {arm: x._sha256(path) for arm, path in paths.items()}
    assert len(set(shas.values())) == 3, "each arm's pickle carries its own table pose"


# ------------------------------------------------------------------------------ tracker fix

_UNFIXED_SOURCE = '''
class MySceneCfg:
    def make(self, config):
        if config.get("add_table", False):
            self.table = CuboidCfg()
            if config.get("add_object", False):
                right_hand_wrist_links = [
                    "{ENV_REGEX_NS}/Robot/right_wrist_roll_link",
                ]
                self.object_to_robot_contact_sensor = ContactSensorCfg(
                    filter_prim_paths_expr=right_hand_wrist_links,
                )

            # Table-to-robot contact sensor
            self.table_to_robot_contact_sensor = ContactSensorCfg(
                filter_prim_paths_expr=right_hand_wrist_links,
            )
'''

_FIXED_SOURCE = '''
class MySceneCfg:
    def make(self, config):
        if config.get("add_table", False):
            self.table = CuboidCfg()
            if config.get("add_object", False):
                right_hand_wrist_links = [
                    "{ENV_REGEX_NS}/Robot/right_wrist_roll_link",
                ]
                self.object_to_robot_contact_sensor = ContactSensorCfg(
                    filter_prim_paths_expr=right_hand_wrist_links,
                )

                # Kept inside the add_object branch, where the filter list is defined.
                self.table_to_robot_contact_sensor = ContactSensorCfg(
                    filter_prim_paths_expr=right_hand_wrist_links,
                )
'''


def test_tracker_fix_detection_accepts_the_fix_and_refuses_the_unfixed_source():
    fixed = x.tracker_fix_report(_FIXED_SOURCE, path="cfg.py", sha256="a" * 64)
    assert fixed["fix_present"] is True and fixed["problems"] == []
    assert fixed["sensor_inside_add_object_branch"] is True
    assert fixed["unbound_local_risk"] is False
    assert fixed["sensor_indent"] > fixed["add_object_branch_indent"]
    assert fixed["binding_line"] < fixed["sensor_line"]
    assert x.require_tracker_fix(fixed) is fixed

    unfixed = x.tracker_fix_report(_UNFIXED_SOURCE, path="cfg.py")
    assert unfixed["fix_present"] is False and unfixed["unbound_local_risk"] is True
    assert unfixed["sensor_inside_add_object_branch"] is False
    assert unfixed["sensor_indent"] == unfixed["add_object_branch_indent"]
    with pytest.raises(x.CampaignAbort, match="add_table work without add_object"):
        x.require_tracker_fix(unfixed)

    missing_binding = _FIXED_SOURCE.replace("right_hand_wrist_links = [", "other_links = [")
    report = x.tracker_fix_report(missing_binding)
    assert report["fix_present"] is False
    assert "never bound" in report["problems"][0]

    duplicated = _FIXED_SOURCE + _FIXED_SOURCE
    assert x.tracker_fix_report(duplicated)["fix_present"] is False


def test_campaign_refuses_to_launch_without_the_tracker_fix(tmp_path):
    calls: list = []
    out = tmp_path / "campaign"
    with pytest.raises(x.CampaignAbort, match="add_table work without add_object"):
        x.run_campaign(stage="launch", **_campaign_kwargs(
            out, calls, tracker_identity_fn=_tracker_identity(fix_present=False)))
    assert not out.exists() and calls == []
    # The dry run still reports, so the operator can see what is missing before committing.
    report = x.run_campaign(stage="all", dry_run=True, **_campaign_kwargs(
        out, calls, tracker_identity_fn=_tracker_identity(fix_present=False)))
    assert report["tracker_add_table_fix"]["fix_present"] is False
    assert not out.exists()


# ------------------------------------------------------------------------------ scene

def test_scene_matches_the_collision_model_geometry():
    scene = x.scene_for(0.05)
    assert scene.start == (0.0, 0.0) and scene.goal == (7.2, 0.0)
    assert scene.meta["corridor_half"] == pytest.approx(1.4)
    box = scene.boxes[0]
    assert box.center == pytest.approx((1.2, 0.0, 0.025))
    assert box.half == pytest.approx((0.1, 1.4, 0.025))
    assert x.scene_for(None).boxes == []
    assert x.criteria().time_limit_s is None  # no deadline was preregistered


# ------------------------------------------------------------------------------ statistics

def _record(outcome: str, key: str, **extra) -> dict:
    return {"outcome": outcome, "motion_key": key, "executed": True, **extra}


def test_outcome_breakdown_and_completion_arithmetic():
    records = ([_record("completed", f"c{i}") for i in range(4)]
               + [_record("collided_obstacle", f"o{i}") for i in range(10)]
               + [_record("cutoff", f"t{i}") for i in range(48)]
               + [_record("fell", "f0", collided_obstacle=True)]
               + [_record("stalled", "s0")])
    summary = x.summarise_arm(records, arm="present_05", box_height_m=0.05)
    assert summary["n_assigned_trials"] == 64 and summary["n_executed"] == 64
    assert summary["outcomes"]["completed"] == 4
    assert summary["outcomes"]["collided_obstacle"] == 10
    assert summary["outcomes"]["cutoff"] == 48 and summary["outcomes"]["fell"] == 1
    assert sum(summary["outcomes"].values()) == 64
    completion = summary["local_traversal_completion"]
    assert completion == {**completion, "completed": 4, "n_assigned_trials": 64}
    assert completion["rate"] == pytest.approx(4 / 64)
    low, high = completion["wilson95"]
    assert 0.0 < low < 4 / 64 < high < 1.0
    assert completion["completing_motion_keys"] == ["c0", "c1", "c2", "c3"]
    # The event count keeps the collision the exclusive `fell` label would have hidden.
    assert summary["event_counts"]["collided_obstacle"] == 11
    assert summary["collision_rate"] == pytest.approx(11 / 64)
    assert summary["completion_rate"] == pytest.approx(4 / 64)
    assert summary["n_timeout_assessed"] == 0  # no deadline: zero means "not assessed"


def test_cohens_kappa_on_a_known_confusion_matrix():
    pairs = ([("completed", "completed")] * 45 + [("completed", "fell")] * 15
             + [("fell", "completed")] * 10 + [("fell", "fell")] * 30)
    kappa = x.cohens_kappa(pairs)
    assert kappa["n"] == 100
    assert kappa["observed_agreement"] == pytest.approx(0.75)
    assert kappa["expected_agreement"] == pytest.approx(0.51)
    assert kappa["kappa"] == pytest.approx(0.24 / 0.49)
    assert kappa["degenerate"] is False

    matrix = x.confusion_matrix(pairs)
    assert matrix["labels"] == ["fell", "completed"]  # te.OUTCOMES precedence order
    assert matrix["matrix"]["completed"]["fell"] == 15
    assert matrix["matrix"]["fell"]["completed"] == 10
    assert sum(sum(row.values()) for row in matrix["matrix"].values()) == 100

    perfect = [("cutoff", "cutoff")] * 64
    degenerate = x.cohens_kappa(perfect)
    assert degenerate["kappa"] is None and degenerate["degenerate"] is True
    assert degenerate["observed_agreement"] == 1.0
    assert x.bootstrap_kappa(perfect, n_resamples=50)["ci95"] is None
    assert x.bootstrap_kappa(perfect, n_resamples=50)["n_degenerate_excluded"] == 50

    independent = [("completed", "fell")] * 32 + [("fell", "completed")] * 32
    assert x.cohens_kappa(independent)["kappa"] == pytest.approx(-1.0)

    boot = x.bootstrap_kappa(pairs, n_resamples=200, seed=x.KAPPA_BOOTSTRAP_SEED)
    assert boot["n_finite"] + boot["n_degenerate_excluded"] == 200
    low, high = boot["ci95"]
    assert low < kappa["kappa"] < high
    assert x.bootstrap_kappa(pairs, n_resamples=200)["ci95"] == boot["ci95"], "seeded"


def _proxy_rows(*, agreeing: int, present_completions: int = 0) -> list[dict]:
    """Rows for `disagreeing` references where the replay proxy and physics disagree."""
    rows = []
    for index, seed in enumerate(x.POOL_SEEDS):
        key = f"s{seed}"
        measured = "completed" if index < present_completions else "cutoff"
        inferred = measured if index < agreeing else "collided_obstacle"
        terminated = _absent_terminated(index)
        rows.append({"arm": "absent", "motion_key": key, "outcome": "cutoff",
                     "max_root_x_m": 7.2,
                     "valid_frames": CUT_LEN if terminated else FULL_LEN,
                     "tracker_terminated": terminated,
                     "replay_inferred": {"outcome": inferred},
                     "traversal": _record("cutoff", key)})
        rows.append({"arm": "present_05", "motion_key": key, "outcome": measured,
                     "max_root_x_m": 1.0, "valid_frames": 120, "tracker_terminated": True,
                     "traversal": _record(measured, key)})
        rows.append({"arm": "present_20", "motion_key": key, "outcome": "cutoff",
                     "max_root_x_m": 1.0, "valid_frames": 120, "tracker_terminated": True,
                     "traversal": _record("cutoff", key)})
    return rows


def test_proxy_check_and_paired_progress_are_paired_per_reference():
    rows = _proxy_rows(agreeing=52)
    proxy = x.proxy_check(rows)
    assert proxy["n"] == 64 and proxy["n_agreeing"] == 52
    assert proxy["agreement_fraction"] == pytest.approx(52 / 64)
    assert proxy["confusion"]["matrix"]["collided_obstacle"]["cutoff"] == 12
    assert proxy["confusion"]["matrix"]["cutoff"]["cutoff"] == 52
    assert len(proxy["disagreeing_references"]) == 12
    assert proxy["per_class_agreement"]["cutoff"]["n_physics_measured"] == 64

    progress = x.paired_progress_change(rows)
    assert progress["n"] == 64 and progress["median_m"] == pytest.approx(6.2)
    assert progress["iqr_m"] == [pytest.approx(6.2), pytest.approx(6.2)]
    assert progress["n_falling_more_than_threshold"] == 64

    rows[0]["max_root_x_m"] = 1.02  # the absent arm barely reached the box
    progress = x.paired_progress_change(rows)
    assert progress["n_falling_more_than_threshold"] == 63


def test_p1_p2_p3_rules_evaluate_pass_and_fail():
    rows = _proxy_rows(agreeing=64)
    arms = {arm: x.summarise_arm([row["traversal"] for row in rows if row["arm"] == arm],
                                 arm=arm, box_height_m=None if arm == "absent" else 0.05)
            for arm in x.ARM_NAMES}
    agreeing_rows = _fake_exp022a_rows(None)["rows"]
    p1 = x.exp022a_agreement(rows, agreeing_rows)
    # The synthetic absent rows use the launcher's rule, so only FLIPPED disagree.
    assert p1["termination_flag"]["n_agreeing"] == 64 - len(FLIPPED)
    assert p1["valid_length"]["n_agreeing"] == 64 - len(FLIPPED)
    assert len(p1["disagreeing_references"]) == len(FLIPPED)

    proxy = x.proxy_check(rows)
    held = x.evaluate_predictions(arms=arms, p1=p1, proxy=proxy)
    assert held["P1"]["threshold"] == 58 and held["P1"]["n_agreeing"] == 60
    assert held["P1"]["prediction_held"] is True
    assert held["P2"]["completions"] == {"present_05": 0, "present_20": 0}
    assert held["P2"]["prediction_held"] is True
    # Perfect agreement on a constant label: kappa is undefined, so the rule is not evaluable.
    assert held["P3"]["agreement_fraction"] == 1.0 and held["P3"]["agreement_ok"] is True
    assert held["P3"]["kappa"] is None and held["P3"]["kappa_degenerate"] is True
    assert held["P3"]["evaluable"] is False and held["P3"]["prediction_held"] is None

    broken_rows = _proxy_rows(agreeing=30, present_completions=2)
    broken_arms = {
        arm: x.summarise_arm([row["traversal"] for row in broken_rows if row["arm"] == arm],
                             arm=arm, box_height_m=None if arm == "absent" else 0.05)
        for arm in x.ARM_NAMES}
    broken_exp022a = {key: {**row, "tracker_terminated": not row["tracker_terminated"]}
                      for key, row in agreeing_rows.items()}
    broken_p1 = x.exp022a_agreement(broken_rows, broken_exp022a)
    broken = x.evaluate_predictions(arms=broken_arms, p1=broken_p1,
                                    proxy=x.proxy_check(broken_rows))
    assert broken_p1["termination_flag"]["n_agreeing"] == len(FLIPPED)
    assert broken["P1"]["prediction_held"] is False
    assert broken["P2"]["completions"]["present_05"] == 2
    assert broken["P2"]["completing_motion_keys"]["present_05"] == ["s4400", "s4401"]
    assert broken["P2"]["prediction_held"] is False
    assert broken["P3"]["agreement_fraction"] == pytest.approx(30 / 64)
    assert broken["P3"]["evaluable"] is True and broken["P3"]["prediction_held"] is False


# ------------------------------------------------------------------------------ campaign

def test_dry_run_writes_nothing_and_prints_six_commands(tmp_path):
    out = tmp_path / "must-not-exist"
    calls: list = []
    result = x.run_campaign(stage="all", dry_run=True, **_campaign_kwargs(
        out, calls, code_state_fn=_dirty_code_state, protocol_identity_fn=_protocol("draft")))
    assert result["status"] == "dry_run" and result["writes_performed"] is False
    assert result["project_dirty_observed"] is True
    assert len(result["launch_plan"]) == 6 and len(result["commands"]) == 6
    absent = result["commands"]["absent_chunk00_seed0"]
    present = result["commands"]["present_20_chunk01_seed0"]
    assert "add_table" not in " ".join(absent)
    assert "++manager_env.config.table_size=[0.2, 2.8, 0.2]" in present
    assert all("++seed=0" in command for command in result["commands"].values())
    assert result["host_resource_gate"]["pass"] is True
    assert len(result["concurrent_isaac_processes"]) == 1
    assert result["tracker_add_table_fix"]["fix_present"] is True
    # Every command runs the patched worktree's checkout with the release bundle's checkpoint.
    assert all(f"+checkpoint={FAKE_CHECKPOINT}" in command
               for command in result["commands"].values())
    execution = result["execution"]
    assert execution["sonic_root"] == str(_fake_worktree(out).resolve())
    assert execution["branch"] == x.SONIC_EXP029_BRANCH
    assert execution["legacy_root_refused"] == str(x.LEGACY_SONIC_ROOT)
    assert execution["files_differing_from_legacy_checkout"] == [x.ADD_TABLE_FIX_FILE]
    assert not out.exists() and calls == []


def test_production_refuses_a_draft_protocol_and_a_dirty_tree(tmp_path):
    calls: list = []
    with pytest.raises(x.CampaignAbort, match="preregistered"):
        x.run_campaign(stage="launch", **_campaign_kwargs(
            tmp_path / "a", calls, protocol_identity_fn=_protocol("draft")))
    with pytest.raises(x.CampaignAbort, match="clean"):
        x.run_campaign(stage="launch", **_campaign_kwargs(
            tmp_path / "b", calls, code_state_fn=_dirty_code_state))
    assert calls == [] and not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_analysis_refuses_before_the_launches(tmp_path):
    calls: list = []
    out = tmp_path / "campaign"
    with pytest.raises(x.CampaignAbort, match="requires the six completed launches"):
        x.run_campaign(stage="analyze", **_campaign_kwargs(out, calls))
    assert calls == []
    assert (out / "receipt.json").is_file()  # the ledger is persisted before any launch


def test_full_injected_campaign_and_idempotent_resume(tmp_path):
    calls: list = []
    out = tmp_path / "campaign"
    receipt = x.run_campaign(stage="all", **_campaign_kwargs(out, calls))
    assert receipt["status"] == "complete" and receipt["actual_ardy_samples"] == 0
    assert receipt["sonic_rollouts_requested"] == 192
    assert receipt["sonic_rollouts_returned"] == 192
    assert len(calls) == 6
    assert [call["num_envs"] for call in calls] == [32] * 6
    assert [call["physics_seed"] for call in calls] == [0] * 6
    assert [call["present"] for call in calls] == [False, False, True, True, True, True]
    assert [call["table_pos"] for call in calls] == [
        None, None, [1.2, 0.0, 0.025], [1.2, 0.0, 0.025], [1.2, 0.0, 0.1], [1.2, 0.0, 0.1]]
    assert calls[0]["keys"] == exp022.chunk_plan()[0]["motion_keys"]
    assert calls[5]["keys"] == exp022.chunk_plan()[1]["motion_keys"]
    execution = receipt["design"]["execution_root"]
    assert execution["sonic_root"] == str(_fake_worktree(out).resolve())
    assert execution["checkpoint"] == FAKE_CHECKPOINT
    assert execution["legacy_root_refused"] == str(x.LEGACY_SONIC_ROOT)
    for name, record in receipt["launches"].items():
        command = json.loads((Path(record["attempt"]) / "command.json").read_text())
        assert command["cwd"] == str(_fake_worktree(out).resolve())
        assert command["checkpoint"] == FAKE_CHECKPOINT
        assert f"+checkpoint={FAKE_CHECKPOINT}" in command["command"]
        assert record["status"] == "complete" and record["returncode"] == 0
        assert record["host_resource_gate"]["gate"]["pass"] is True
        assert record["host_resource_gate"]["n_concurrent_isaac_processes"] == 1
        assert sorted(record["log_termination_terms"]) == list(x.RELEASE_TERMINATION_TERMS)
        assert record["rollout_check"]["n_rollouts"] == 32

    rows = x._read_jsonl(out / "rows.jsonl")
    assert len(rows) == 192
    assert {row["arm"] for row in rows} == set(x.ARM_NAMES)
    absent = [row for row in rows if row["arm"] == "absent"]
    assert all("replay_inferred" in row for row in absent)
    assert not any("replay_inferred" in row for row in rows if row["arm"] != "absent")

    summary = json.loads((out / "summary.json").read_text())
    # 8 obstacle-absent rollouts are cut off; the rest walk the whole route to the goal.
    assert summary["arms"]["absent"]["outcomes"]["completed"] == 56
    assert summary["arms"]["absent"]["outcomes"]["cutoff"] == 8
    assert summary["arms"]["present_05"]["outcomes"]["cutoff"] == 64
    assert summary["arms"]["present_05"]["local_traversal_completion"]["completed"] == 0
    assert summary["arms"]["present_20"]["local_traversal_completion"]["wilson95"][0] == 0.0
    proxy = summary["q1_proxy_check"]
    assert proxy["n_agreeing"] == 8 and proxy["agreement_fraction"] == pytest.approx(0.125)
    assert proxy["cohens_kappa"]["kappa"] == pytest.approx(0.0)
    assert proxy["confusion"]["matrix"]["collided_obstacle"]["cutoff"] == 56
    progress = summary["paired_progress_change"]
    assert progress["median_m"] == pytest.approx(6.2)
    assert progress["n_falling_more_than_threshold"] == 56
    predictions = summary["predictions"]
    assert predictions["P1"]["n_agreeing"] == 60 and predictions["P1"]["prediction_held"] is True
    assert predictions["P2"]["prediction_held"] is True
    assert predictions["P3"]["prediction_held"] is False
    assert summary["arms"]["present_05"]["n_assigned_trials"] == 64

    again = x.run_campaign(stage="all", resume=True, **_campaign_kwargs(out, calls))
    assert again["status"] == "complete" and len(calls) == 6, "a complete campaign never relaunches"
    with pytest.raises(x.CampaignAbort, match="non-empty"):
        x.run_campaign(stage="all", **_campaign_kwargs(out, calls))


def test_resume_skips_completed_launches_and_supersedes_an_interrupted_attempt(tmp_path):
    calls: list = []
    out = tmp_path / "campaign"
    # An interrupted third launch: a pre-launch receipt with no artifacts must not be adopted.
    with pytest.raises(x.CampaignPaused, match="blocked_host_gate"):
        x.run_campaign(stage="launch", **_campaign_kwargs(out, calls,
                                                          host_gate_fn=_gate_after(3)))
    assert len(calls) == 2
    spec = x.launch_plan()[2]
    interrupted = out / "launches" / spec["name"] / "attempt-000"
    interrupted.mkdir(parents=True)
    pkl = out / "launches" / spec["name"] / "motions.pkl"
    cal._write_json(interrupted / "receipt.json",
                    {"status": "running", **x._attempt_expectations(spec, x._sha256(pkl))})

    x.run_campaign(stage="launch", resume=True, **_campaign_kwargs(out, calls))
    assert len(calls) == 6, "the two completed launches are skipped, the other four run"
    assert calls[2]["eval_dir"].endswith(f"{spec['name']}/attempt-001/eval")
    receipt = x.run_campaign(stage="analyze", resume=True, **_campaign_kwargs(out, calls))
    assert receipt["status"] == "complete" and len(calls) == 6

    record = receipt["launches"]["present_20_chunk01_seed0"]
    altered = json.loads((Path(record["attempt"]) / "receipt.json").read_text())
    altered["manual_change"] = True
    cal._write_json(Path(record["attempt"]) / "receipt.json", altered)
    with pytest.raises(x.CampaignAbort, match="artifacts changed"):
        x.run_campaign(stage="analyze", resume=True, **_campaign_kwargs(out, calls))


def test_host_gate_refusal_leaves_the_output_untouched(tmp_path):
    calls: list = []
    out = tmp_path / "campaign"
    with pytest.raises(x.CampaignAbort, match="host-resource gate failed before"):
        x.run_campaign(stage="launch", **_campaign_kwargs(out, calls, host_gate_fn=_gate_fail))
    assert not out.exists() and calls == []

    # A gate that fails at the second launch pauses the campaign without touching evidence.
    with pytest.raises(x.CampaignPaused, match="blocked_host_gate"):
        x.run_campaign(stage="launch", **_campaign_kwargs(out, calls,
                                                          host_gate_fn=_gate_after(2)))
    assert len(calls) == 1
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "running" and receipt["blocked"] is False
    assert receipt["host_gate_blocks"][0]["note"] == "blocked_host_gate"
    assert receipt["host_gate_blocks"][0]["launch"] == "absent_chunk01_seed0"
    assert not (out / "launches" / "absent_chunk01_seed0" / "attempt-000").exists()
    assert len(receipt["launches"]) == 1

    x.run_campaign(stage="launch", resume=True, **_campaign_kwargs(out, calls))
    assert len(calls) == 6
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["stages_complete"]["launch"] is True


def test_zero_length_archive_is_labelled_not_dropped():
    rollout = SonicRollout("s4400", np.zeros((0, 36), dtype=np.float32), 0, True, 0.0, 0)
    record = x.score_rollout(rollout, x.scene_for(0.05))
    assert record["outcome"] == "cutoff" and record["zero_length_archive"] is True
    assert record["executed"] is True and record["motion_key"] == "s4400"
    assert te.summarise([record])["n_assigned_trials"] == 1


def test_release_evaluator_terms_are_checked_against_the_log():
    log = _termination_table(RELEASE_TABLE)
    assert sorted(x._check_log_terminations(log)) == list(x.RELEASE_TERMINATION_TERMS)
    with pytest.raises(ValueError, match="differ from the release evaluator"):
        x._check_log_terminations(_termination_table(
            {"time_out": True, "anchor_pos": False}))


def test_locked_design_constants_match_the_protocol():
    assert x.PHYSICS_SEED == 0 and x.SAMPLE_DT_S == 0.02
    assert x.P1_MIN_TERMINATION_AGREEMENT == 58
    assert x.P2_PREDICTED_COMPLETIONS == 0
    assert (x.P3_MIN_AGREEMENT, x.P3_MIN_KAPPA) == (0.80, 0.6)
    assert x.ARM_NAMES == ("absent", "present_05", "present_20")
    assert x.EXPECTED_CHECKPOINT_SHA256.startswith("e6bdab3f")
    # The two-checkout ruling: the patched worktree, pinned at the fix commit, for every arm.
    assert x.SONIC_EXP029_ROOT == Path("/home/linjiw/lucid/GR00T-WBC-exp029")
    assert x.SONIC_EXP029_BRANCH == "exp029-obstacle-present"
    assert x.ADD_TABLE_FIX_COMMIT.startswith("7c63c53")
    assert x.LEGACY_SONIC_ROOT == Path("/home/linjiw/lucid/GR00T-WholeBodyControl")
    # EXP-030 declares its own tracker baseline: EXP-022A's manifest is recorded, never asserted.
    assert x.EXP022A_CORE_MANIFEST_SHA256 == e28.EXPECTED_CORE_MANIFEST_SHA256
    assert "EXPECTED_CORE_MANIFEST_SHA256" not in dir(x)
    protocol = x.protocol_identity()
    assert protocol["status"] == "preregistered" and len(protocol["sha256"]) == 64


def test_evaluator_version_is_recorded_not_assumed(monkeypatch):
    """The launch stage must not depend on a marker a later evaluator revision introduced."""
    import scene2motion.traversal_eval as te
    monkeypatch.delattr(te, "EVALUATOR_VERSION", raising=False)
    assert x.evaluator_version() == 1
    monkeypatch.setattr(te, "EVALUATOR_VERSION", 2, raising=False)
    assert x.evaluator_version() == 2
