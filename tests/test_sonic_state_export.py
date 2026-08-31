from __future__ import annotations

import os

import numpy as np
import pytest

from scene2motion.sonic_export import G1_29DOF
from scene2motion.sonic_state_export import (
    ARCHIVE_NAME,
    CALLBACK_TARGET,
    QPOS_WIDTH,
    SonicRollout,
    achieved_qpos_by_key,
    isaac_state_to_mujoco_qpos,
    load_sonic_state_rollouts,
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
