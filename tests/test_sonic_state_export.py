from __future__ import annotations

import os

import numpy as np
import pytest

from scene2motion.sonic_export import G1_29DOF
from scene2motion.sonic_state_export import (
    ARCHIVE_NAME,
    CALLBACK_TARGET,
    FRAME_CONVENTION,
    QPOS_WIDTH,
    SAMPLE_OFFSET_STEPS,
    SonicRollout,
    SonicStateExportCallback,
    achieved_qpos_by_key,
    isaac_state_to_mujoco_qpos,
    load_sonic_state_rollouts,
    sonic_state_archive_schema,
    sonic_state_hydra_overrides,
    sonic_state_sample_dt,
    sonic_state_subprocess_env,
    write_sonic_state_archive,
)


def _qpos(length: int, marker: float) -> np.ndarray:
    qpos = np.zeros((length, QPOS_WIDTH), dtype=np.float32)
    qpos[:, 0] = marker + np.arange(length, dtype=np.float32)
    qpos[:, 2] = 0.8
    qpos[:, 3] = 1.0
    return qpos


def test_isaac_state_conversion_is_local_wxyz_and_name_ordered():
    joint_names = ["left_finger_joint", *reversed(G1_29DOF)]
    joint_pos = np.stack(
        [np.arange(len(joint_names)), 100 + np.arange(len(joint_names))]
    ).astype(np.float32)
    roots = np.asarray([[11.0, 22.0, 0.8], [-2.0, 7.0, 0.9]], dtype=np.float32)
    origins = np.asarray([[10.0, 20.0, 0.0], [-5.0, 5.0, 0.0]], dtype=np.float32)
    # Deliberately non-unit to exercise defensive normalization.  Input and output are wxyz.
    quats = np.asarray([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]], dtype=np.float32)

    qpos = isaac_state_to_mujoco_qpos(
        roots, quats, joint_pos, joint_names, origins
    )

    assert qpos.shape == (2, 36)
    np.testing.assert_allclose(qpos[:, :3], roots - origins)
    np.testing.assert_allclose(qpos[:, 3:7], [[1, 0, 0, 0], [0, 0, 0, 1]])
    expected_order = [joint_names.index(name) for name in G1_29DOF]
    np.testing.assert_array_equal(qpos[:, 7:], joint_pos[:, expected_order])


def test_isaac_state_conversion_rejects_missing_or_ambiguous_joints():
    root = np.zeros((1, 3), dtype=np.float32)
    quat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    joints = np.zeros((1, len(G1_29DOF)), dtype=np.float32)
    with pytest.raises(ValueError, match="missing"):
        isaac_state_to_mujoco_qpos(root, quat, joints[:, :-1], G1_29DOF[:-1])
    with pytest.raises(ValueError, match="duplicates"):
        isaac_state_to_mujoco_qpos(root, quat, joints, [G1_29DOF[0]] * len(G1_29DOF))


def test_archive_loader_trims_padding_and_preserves_outcomes(tmp_path):
    records = [
        SonicRollout("late", _qpos(3, 20), 3, False, 1.0, 2),
        SonicRollout("early-stop", _qpos(1, 10), 1, True, 0.25, 0),
    ]
    archive = write_sonic_state_archive(records, tmp_path / ARCHIVE_NAME, sample_dt_s=0.02)

    with np.load(archive, allow_pickle=False) as raw:
        assert raw["qpos"].shape == (2, 3, 36)
        assert np.isnan(raw["qpos"][0, 1:]).all()  # sorted motion id 0 is padded
        assert raw["root_quaternion_order"].item() == "wxyz"

    loaded = load_sonic_state_rollouts(tmp_path)
    assert sonic_state_sample_dt(tmp_path) == pytest.approx(0.02)
    assert [r.motion_key for r in loaded] == ["early-stop", "late"]
    assert [r.valid_length for r in loaded] == [1, 3]
    assert [r.terminated for r in loaded] == [True, False]
    assert loaded[0].progress == pytest.approx(0.25)
    np.testing.assert_array_equal(loaded[0].qpos, records[1].qpos)
    keyed = achieved_qpos_by_key(archive)
    assert set(keyed) == {"early-stop", "late"}
    assert keyed["early-stop"].shape == (1, 36)


def write_v1_archive(path, records, sample_dt_s=0.02):
    """The pre-fix on-disk layout: schema 1, teleport frame kept for non-terminated envs."""
    max_len = max(r.valid_length for r in records)
    padded = np.full((len(records), max_len, QPOS_WIDTH), np.nan, dtype=np.float32)
    for i, r in enumerate(records):
        padded[i, : r.valid_length] = r.qpos
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int16),
        qpos=padded,
        valid_lengths=np.asarray([r.valid_length for r in records], dtype=np.int32),
        terminated=np.asarray([r.terminated for r in records], dtype=np.bool_),
        progress=np.asarray([r.progress for r in records], dtype=np.float32),
        motion_keys=np.asarray([r.motion_key for r in records], dtype=np.str_),
        motion_ids=np.asarray([r.motion_id for r in records], dtype=np.int64),
        joint_names=np.asarray(G1_29DOF, dtype=np.str_),
        root_frame=np.asarray("isaac_env_local", dtype=np.str_),
        root_quaternion_order=np.asarray("wxyz", dtype=np.str_),
        sample_dt_s=np.asarray(sample_dt_s, dtype=np.float64),
    )


def test_v2_archive_encodes_frame_rate_convention():
    # Pins review defect #2: the achieved[i] <-> reference frame mapping must live in the
    # archive, not in a consumer's head.
    assert SAMPLE_OFFSET_STEPS == 1
    assert "no frame-0 sample" in FRAME_CONVENTION


def test_v2_archive_carries_convention_fields(tmp_path):
    archive = write_sonic_state_archive(
        [SonicRollout("only", _qpos(2, 0), 2, False, 1.0, 0)],
        tmp_path / ARCHIVE_NAME,
        sample_dt_s=0.02,
    )
    with np.load(archive, allow_pickle=False) as raw:
        assert int(raw["schema_version"]) == 2
        assert int(raw["sample_offset_steps"]) == 1
        assert bool(raw["timeout_frame_dropped"])
        assert str(raw["frame_convention"].item()) == FRAME_CONVENTION
    assert sonic_state_archive_schema(archive) == 2
    # A v2 loader must not trim again: the writer's convention already excludes the teleport.
    assert load_sonic_state_rollouts(archive)[0].valid_length == 2


def test_v1_archive_loader_drops_the_nonterminated_teleport_frame(tmp_path):
    # Measured on heuristic_00..04: every non-terminated rollout's final v1 frame is the
    # post-reset frame-0 pose at the env origin (|a[-1] - ref[0]| = 0.000 m across 43 envs).
    timed_out = _qpos(4, 50)
    timed_out[-1] = timed_out[0]  # the teleport: back to the start pose
    early_stop = _qpos(3, 10)
    path = tmp_path / ARCHIVE_NAME
    write_v1_archive(
        path,
        [
            SonicRollout("timed-out", timed_out, 4, False, 1.0, 0),
            SonicRollout("early-stop", early_stop, 3, True, 0.5, 1),
        ],
    )
    assert sonic_state_archive_schema(path) == 1
    loaded = {r.motion_key: r for r in load_sonic_state_rollouts(path)}
    assert loaded["timed-out"].valid_length == 3  # teleport frame gone
    np.testing.assert_array_equal(loaded["timed-out"].qpos, timed_out[:3])
    assert loaded["early-stop"].valid_length == 3  # terminated rollouts are untouched
    np.testing.assert_array_equal(loaded["early-stop"].qpos, early_stop)


def test_callback_batch_trim_excludes_timeout_teleport_frame():
    # Pins review defect #1 at the source, without an Isaac environment: a non-terminated env
    # must archive ref_len - 1 samples because the snapshot at index ref_len - 1 is the
    # post-reset pose Isaac produced inside motion_time_out's own step().
    from types import SimpleNamespace

    cb = object.__new__(SonicStateExportCallback)
    cb._achieved_records = []
    steps = 6
    cb._batch_qpos = [np.full((2, QPOS_WIDTH), float(t), dtype=np.float32) for t in range(steps)]
    cb._batch_motion_ids = np.asarray([0, 1], dtype=np.int64)
    cb._batch_reference_lengths = np.asarray([steps, steps], dtype=np.int64)
    cb.terminate_memory = [np.asarray([False, True])]
    cb.progress_memory = [np.asarray([1.0, 0.5])]
    cb.env = SimpleNamespace(
        _motion_lib=SimpleNamespace(_num_unique_motions=2, _motion_data_keys=["walk", "duck"])
    )
    cb._finish_achieved_batch()

    by_key = {r.motion_key: r for r in cb._achieved_records}
    assert by_key["walk"].valid_length == steps - 1
    assert by_key["walk"].progress == 1.0
    assert not by_key["walk"].terminated
    np.testing.assert_array_equal(by_key["walk"].qpos[:, 0], np.arange(steps - 1))
    assert by_key["duck"].valid_length == 3  # rint(0.5 * 6): terminated trim is unchanged
    assert by_key["duck"].terminated


def test_directory_loader_merges_rank_shards_by_global_motion_id(tmp_path):
    write_sonic_state_archive(
        [SonicRollout("one", _qpos(2, 1), 2, False, 1.0, 1)],
        tmp_path / "achieved_qpos.rank1.npz",
        sample_dt_s=0.02,
    )
    write_sonic_state_archive(
        [SonicRollout("zero", _qpos(2, 0), 2, False, 1.0, 0)],
        tmp_path / "achieved_qpos.rank0.npz",
        sample_dt_s=0.02,
    )
    assert [r.motion_key for r in load_sonic_state_rollouts(tmp_path)] == ["zero", "one"]
    assert sonic_state_sample_dt(tmp_path) == pytest.approx(0.02)


def test_sample_dt_rejects_invalid_or_disagreeing_shards(tmp_path):
    write_sonic_state_archive(
        [SonicRollout("zero", _qpos(2, 0), 2, False, 1.0, 0)],
        tmp_path / "achieved_qpos.rank0.npz", sample_dt_s=0.02)
    write_sonic_state_archive(
        [SonicRollout("one", _qpos(2, 1), 2, False, 1.0, 1)],
        tmp_path / "achieved_qpos.rank1.npz", sample_dt_s=0.04)
    with pytest.raises(ValueError, match="disagree"):
        sonic_state_sample_dt(tmp_path)


def test_hydra_invocation_helper_exposes_local_callback(tmp_path):
    overrides = sonic_state_hydra_overrides()
    assert f"++callbacks.im_eval._target_={CALLBACK_TARGET}" in overrides
    assert f"++callbacks.im_eval.state_filename={ARCHIVE_NAME}" in overrides
    with pytest.raises(ValueError, match="simple filename"):
        sonic_state_hydra_overrides("nested/achieved.npz")

    env = sonic_state_subprocess_env(
        {"PYTHONPATH": os.pathsep.join(["/old/one", "/old/two"]), "KEEP": "yes"},
        project_root=tmp_path,
    )
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(tmp_path.resolve()),
        "/old/one",
        "/old/two",
    ]
    assert env["KEEP"] == "yes"
