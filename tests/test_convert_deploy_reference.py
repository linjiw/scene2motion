"""CPU tests for the deployment-reference -> motion-pickle converter.

The converter's job is to make SONIC's shipped one-leg-jump reference usable as EXP-024b's
known-trackable control.  Two mistakes would not crash anything and would invalidate that
control, so both are pinned here on synthetic references built by forward kinematics:

  * a wrong joint permutation -- the tracker would follow a plausible but different motion;
  * a misdeclared frame rate -- the motion library treats the pickle's ``fps`` as the source
    rate when it resamples, so the same frames declared at half the rate play at half speed.

The synthetic references are self-consistent by construction: joint angles are drawn in MuJoCo
order, MuJoCo forward kinematics produces the tracked-body world positions, and the CSVs are
written in the vendor's own layout and float format.  Writing the joint columns in IsaacLab
order gives a reference the converter must accept; writing them already in MuJoCo order gives
one whose ordering check must fire.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from experiments import convert_deploy_reference as cdr
from scene2motion.sonic_export import G1_29DOF, NUM_DOF, write_motion_pkl

TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
SYNTHETIC_FPS = 50


# ---------------------------------------------------------------- fixtures and builders

#: The vendor tables live in the SONIC checkout, which is not on every host that runs the CPU
#: suite (docs/SETUP.md installs this repository without it).  Without a guard the fixture
#: ERRORS rather than skips, which would break "CPU tests; no GPU needed" on a fresh machine.
requires_sonic = pytest.mark.skipif(
    not (cdr.SONIC_ROOT / "gear_sonic").is_dir(),
    reason="the SONIC checkout is not present on this host")


@pytest.fixture(scope="module")
def tables() -> cdr.VendorTables:
    if not (cdr.SONIC_ROOT / "gear_sonic").is_dir():
        pytest.skip("the SONIC checkout is not present on this host")
    return cdr.read_vendor_tables()


@pytest.fixture(scope="module")
def mj_model():
    import mujoco
    return mujoco.MjModel.from_xml_path(str(cdr.ARDY_G1_XML))


def _synthetic_qpos(n_frames: int) -> np.ndarray:
    """Joint angles in MuJoCo order plus a moving root, deterministic and well conditioned."""
    t = np.arange(n_frames, dtype=np.float64)[:, None]
    phase = np.arange(NUM_DOF, dtype=np.float64)[None, :]
    dof_mj = 0.35 * np.sin(0.21 * t + 0.7 * phase)
    root_pos = np.stack([0.02 * t[:, 0], 0.005 * t[:, 0], 0.78 + 0.03 * np.sin(0.3 * t[:, 0])], 1)
    rotvec = np.stack([0.05 * np.sin(0.11 * t[:, 0]),
                       0.04 * np.cos(0.13 * t[:, 0]),
                       0.20 * np.sin(0.07 * t[:, 0])], 1)
    angle = np.linalg.norm(rotvec, axis=1, keepdims=True)
    axis = np.divide(rotvec, np.where(angle > 0, angle, 1.0))
    root_quat_wxyz = np.concatenate([np.cos(angle / 2), axis * np.sin(angle / 2)], axis=1)
    return root_pos, root_quat_wxyz, dof_mj


def _central_difference(values: np.ndarray, fps: int) -> np.ndarray:
    out = np.zeros_like(values)
    if len(values) >= 3:
        out[1:-1] = (values[2:] - values[:-2]) * (fps / 2.0)
    return out


def _write_csv(path: Path, header: list[str], data: np.ndarray) -> None:
    lines = [",".join(header)]
    lines += [",".join(f"{v:.6f}" for v in row) for row in np.asarray(data).reshape(len(data), -1)]
    path.write_text("\n".join(lines) + "\n")


def build_synthetic_reference(directory: Path, mj_model, tables: cdr.VendorTables, *,
                              n_frames: int = 12, joint_order: str = "isaaclab",
                              name: str | None = None) -> dict:
    """Write one deployment reference directory whose CSVs are consistent with its own angles.

    ``joint_order="isaaclab"`` writes what the real deployment format contains, so the converter
    must accept it.  ``joint_order="mujoco"`` writes the columns already permuted, which is the
    exact mistake the ordering check exists to catch.
    """
    import mujoco

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(mj_model)
    root_pos, root_quat, dof_mj = _synthetic_qpos(n_frames)
    body_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in TRACKED_BODIES]

    body_pos = np.zeros((n_frames, len(TRACKED_BODIES), 3))
    body_quat = np.zeros((n_frames, len(TRACKED_BODIES), 4))
    for i in range(n_frames):
        data.qpos[:3] = root_pos[i]
        data.qpos[3:7] = root_quat[i]
        data.qpos[7:7 + NUM_DOF] = dof_mj[i]
        mujoco.mj_kinematics(mj_model, data)
        body_pos[i] = np.asarray(data.xpos)[body_ids]
        body_quat[i] = np.asarray(data.xquat)[body_ids]
    body_pos[:, 0] = root_pos                       # the pelvis row is the root by definition
    body_quat[:, 0] = root_quat

    if joint_order == "isaaclab":
        joint_columns = np.empty_like(dof_mj)
        joint_columns[:, tables.isaaclab_to_mujoco_dof] = dof_mj
    elif joint_order == "mujoco":
        joint_columns = dof_mj.copy()
    else:
        raise ValueError(joint_order)

    _write_csv(directory / "joint_pos.csv",
               [f"joint_{i}" for i in range(NUM_DOF)], joint_columns)
    _write_csv(directory / "joint_vel.csv",
               [f"joint_vel_{i}" for i in range(NUM_DOF)],
               _central_difference(joint_columns, SYNTHETIC_FPS))
    _write_csv(directory / "body_pos.csv",
               [f"body_{i // 3}_{'xyz'[i % 3]}" for i in range(3 * len(TRACKED_BODIES))],
               body_pos.reshape(n_frames, -1))
    _write_csv(directory / "body_quat.csv",
               [f"body_{i // 4}_{'wxyz'[i % 4]}" for i in range(4 * len(TRACKED_BODIES))],
               body_quat.reshape(n_frames, -1))
    _write_csv(directory / "body_lin_vel.csv",
               [f"body_{i // 3}_vel_{'xyz'[i % 3]}" for i in range(3 * len(TRACKED_BODIES))],
               _central_difference(body_pos, SYNTHETIC_FPS).reshape(n_frames, -1))
    indexes = [tables.isaaclab_body_names.index(n) for n in TRACKED_BODIES]
    (directory / "metadata.txt").write_text(
        f"Metadata for: {name or directory.name}\n\n"
        f"Body part indexes:\n[{'  '.join(str(i) for i in indexes)}]\n\n"
        f"Total timesteps: {n_frames}\n")
    return {"root_pos": root_pos, "root_quat_wxyz": root_quat, "dof_mj": dof_mj,
            "body_pos": body_pos, "n_frames": n_frames}


@pytest.fixture
def reference_dir(tmp_path, mj_model, tables):
    truth = build_synthetic_reference(tmp_path / "synthetic_motion", mj_model, tables)
    return tmp_path / "synthetic_motion", truth


# ---------------------------------------------------------------- vendor tables

def test_the_three_vendor_sources_agree_on_one_permutation(tables):
    perm = tables.isaaclab_to_mujoco_dof
    assert perm.shape == (NUM_DOF,)
    assert sorted(perm.tolist()) == list(range(NUM_DOF))
    assert set(tables.sources) >= {rel for rel, _ in cdr.PERMUTATION_SOURCES}


def test_vendor_name_lists_have_the_expected_lengths(tables):
    assert len(tables.isaaclab_joint_names) == NUM_DOF
    assert len(tables.isaaclab_body_names) == NUM_DOF + 1


def test_disagreeing_vendor_sources_are_refused(tmp_path, monkeypatch):
    root = tmp_path / "sonic"
    for rel, symbol in cdr.PERMUTATION_SOURCES:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        order = list(range(NUM_DOF))
        if symbol == "isaaclab_to_mujoco":
            order[0], order[1] = order[1], order[0]
        path.write_text(f"{symbol} = {order}\n")
    with pytest.raises(cdr.ConversionError, match="not consistent across its own sources"):
        cdr.read_vendor_tables(root)


# ---------------------------------------------------------------- joint order, by name

def test_the_permutation_maps_isaaclab_names_onto_the_motion_pickle_order(tables):
    report = cdr.check_joint_order_by_name(tables)
    assert report["passed"] is True
    assert report["mujoco_order_from_isaaclab_names"] == list(G1_29DOF)


def test_the_name_check_fires_on_a_swapped_permutation(tables):
    swapped = tables.isaaclab_to_mujoco_dof.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    broken = cdr.VendorTables(swapped, tables.isaaclab_joint_names,
                              tables.isaaclab_body_names, tables.sources)
    with pytest.raises(cdr.ConversionError, match="name check failed at MuJoCo index 0"):
        cdr.check_joint_order_by_name(broken)


# ---------------------------------------------------------------- the deployment layout

def test_the_loader_reads_the_synthetic_layout(reference_dir, tables):
    directory, truth = reference_dir
    reference = cdr.load_deploy_reference(directory, tables)
    assert reference.n_frames == truth["n_frames"]
    assert reference.joint_pos_il.shape == (truth["n_frames"], NUM_DOF)
    assert reference.body_pos_w.shape == (truth["n_frames"], len(TRACKED_BODIES), 3)
    assert reference.body_quat_w.shape == (truth["n_frames"], len(TRACKED_BODIES), 4)
    assert reference.tracked_body_names == TRACKED_BODIES
    assert set(reference.files) >= {"joint_pos.csv", "body_pos.csv", "body_quat.csv",
                                    "metadata.txt"}


def test_a_renamed_column_is_refused(reference_dir, tables):
    directory, _ = reference_dir
    path = directory / "body_quat.csv"
    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace("body_0_w", "body_0_q", 1)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(cdr.ConversionError, match="not the deployment reference layout"):
        cdr.load_deploy_reference(directory, tables)


def test_csvs_that_disagree_on_the_frame_count_are_refused(reference_dir, tables):
    directory, _ = reference_dir
    path = directory / "body_pos.csv"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(cdr.ConversionError, match="disagree on the frame count"):
        cdr.load_deploy_reference(directory, tables)


def test_non_unit_quaternions_are_refused(reference_dir, tables):
    directory, _ = reference_dir
    path = directory / "body_quat.csv"
    lines = path.read_text().splitlines()
    values = [float(v) for v in lines[1].split(",")]
    values[0] += 0.5
    lines[1] = ",".join(f"{v:.6f}" for v in values)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(cdr.ConversionError, match="not unit quaternions"):
        cdr.load_deploy_reference(directory, tables)


def test_metadata_that_contradicts_the_csvs_is_refused(reference_dir, tables):
    directory, truth = reference_dir
    path = directory / "metadata.txt"
    path.write_text(path.read_text().replace(f"Total timesteps: {truth['n_frames']}",
                                             "Total timesteps: 999"))
    with pytest.raises(cdr.ConversionError, match="declares 999 timesteps"):
        cdr.load_deploy_reference(directory, tables)


# ---------------------------------------------------------------- joint order, by geometry

def test_the_geometry_check_passes_on_a_consistent_reference(reference_dir, tables, mj_model):
    directory, _ = reference_dir
    reference = cdr.load_deploy_reference(directory, tables)
    report = cdr.check_joint_order_by_geometry(reference, tables, mj_model=mj_model)
    assert report["passed"] is True
    assert report["worst_shared_frame_error_m"] < cdr.EXACT_FRAME_TOL_M
    assert report["discrimination_ratio_vs_best_alternative"] >= cdr.MIN_DISCRIMINATION_RATIO
    assert set(report["candidate_scores"]) == {"vendor_isaaclab_to_mujoco",
                                               "identity_no_reorder",
                                               "inverse_mujoco_to_isaaclab"}
    best = min(report["candidate_scores"], key=lambda k: report["candidate_scores"][k]["max_error_m"])
    assert best == "vendor_isaaclab_to_mujoco"


def test_the_geometry_check_fires_when_the_joint_columns_are_already_in_mujoco_order(
        tmp_path, tables, mj_model):
    directory = tmp_path / "wrong_order"
    build_synthetic_reference(directory, mj_model, tables, joint_order="mujoco")
    reference = cdr.load_deploy_reference(directory, tables)
    with pytest.raises(cdr.ConversionError, match="geometry check failed"):
        cdr.check_joint_order_by_geometry(reference, tables, mj_model=mj_model)


def test_a_wrongly_ordered_reference_writes_nothing(tmp_path, tables, mj_model):
    directory = tmp_path / "wrong_order"
    build_synthetic_reference(directory, mj_model, tables, joint_order="mujoco")
    out = tmp_path / "out" / "motion.pkl"
    with pytest.raises(cdr.ConversionError, match="geometry check failed"):
        cdr.convert_deploy_reference(directory, out, fps_source=SYNTHETIC_FPS,
                                     mj_model=mj_model)
    assert not out.exists()
    assert not Path(str(out) + ".provenance.json").exists()


def test_the_geometry_check_needs_a_body_whose_frame_is_shared(reference_dir, tables, mj_model,
                                                               monkeypatch):
    directory, _ = reference_dir
    reference = cdr.load_deploy_reference(directory, tables)
    monkeypatch.setattr(cdr, "EXACT_FRAME_BODIES", ("no_such_link",))
    with pytest.raises(cdr.ConversionError, match="no exact anchor"):
        cdr.check_joint_order_by_geometry(reference, tables, mj_model=mj_model)


# ---------------------------------------------------------------- duration

def test_duration_is_preserved_when_nothing_is_resampled():
    report = cdr.assert_duration_preserved(500, 50, 500, 50)
    assert report["preserved"] is True
    assert report["source_frames_over_fps_s"] == report["output_frames_over_fps_s"] == 10.0
    assert report["source_span_frames_minus_one_over_fps_s"] == pytest.approx(9.98)


def test_the_duration_assertion_fires_when_the_rate_is_misdeclared():
    # 500 frames written as 25 fps would play the ten-second jump over twenty seconds.
    with pytest.raises(cdr.ConversionError, match="duration is not preserved"):
        cdr.assert_duration_preserved(500, 50, 500, 25)


def test_the_duration_assertion_fires_when_a_frame_is_dropped():
    with pytest.raises(cdr.ConversionError, match="duration is not preserved"):
        cdr.assert_duration_preserved(500, 50, 499, 50)


def test_an_exact_stride_resample_costs_at_most_one_source_frame_of_span():
    report = cdr.assert_duration_preserved(500, 50, 250, 25)
    assert report["source_frames_over_fps_s"] == report["output_frames_over_fps_s"] == 10.0
    span = (report["source_span_frames_minus_one_over_fps_s"]
            - report["output_span_frames_minus_one_over_fps_s"])
    assert 0 < span <= 1.0 / 50 + 1e-9


def test_a_non_divisor_output_rate_is_refused(reference_dir, mj_model):
    directory, _ = reference_dir
    with pytest.raises(cdr.ConversionError, match="not an exact divisor"):
        cdr.convert_deploy_reference(directory, directory.parent / "out.pkl",
                                     fps_source=SYNTHETIC_FPS, fps_out=30, mj_model=mj_model)


def test_a_stride_resample_keeps_the_duration(tmp_path, tables, mj_model):
    directory = tmp_path / "even_length"
    build_synthetic_reference(directory, mj_model, tables, n_frames=12)
    sidecar = cdr.convert_deploy_reference(directory, tmp_path / "half.pkl",
                                           fps_source=SYNTHETIC_FPS, fps_out=SYNTHETIC_FPS // 2,
                                           mj_model=mj_model)
    assert sidecar["output"]["n_frames"] == 6
    assert sidecar["output"]["fps"] == SYNTHETIC_FPS // 2
    assert (sidecar["duration"]["output_frames_over_fps_s"]
            == sidecar["duration"]["source_frames_over_fps_s"])


# ---------------------------------------------------------------- the written pickle

def test_the_entry_has_the_same_schema_write_motion_pkl_produces(reference_dir, mj_model,
                                                                 tmp_path):
    directory, truth = reference_dir
    out = tmp_path / "converted.pkl"
    cdr.convert_deploy_reference(directory, out, fps_source=SYNTHETIC_FPS, mj_model=mj_model)
    converted = pickle.load(out.open("rb"))[directory.name]

    qpos = np.zeros((truth["n_frames"], 7 + NUM_DOF), np.float32)
    qpos[:, 0:3] = truth["root_pos"]
    qpos[:, 3:7] = truth["root_quat_wxyz"]
    qpos[:, 7:] = truth["dof_mj"]
    ardy = tmp_path / "ardy.pkl"
    write_motion_pkl({"ardy": qpos}, ardy, fps=25, mj_model=mj_model)
    reference_entry = pickle.load(ardy.open("rb"))["ardy"]

    assert set(converted) == set(reference_entry)
    for key, value in reference_entry.items():
        if isinstance(value, np.ndarray):
            assert converted[key].dtype == value.dtype
            assert converted[key].shape[1:] == value.shape[1:]
        else:
            assert isinstance(converted[key], type(value))


def test_the_written_dof_columns_are_in_mujoco_order(reference_dir, mj_model, tmp_path):
    directory, truth = reference_dir
    out = tmp_path / "converted.pkl"
    cdr.convert_deploy_reference(directory, out, fps_source=SYNTHETIC_FPS, mj_model=mj_model)
    dof = pickle.load(out.open("rb"))[directory.name]["dof"]
    assert np.allclose(dof, truth["dof_mj"], atol=1e-5)


# ---------------------------------------------------------------- the provenance sidecar

@pytest.fixture
def converted(reference_dir, mj_model, tmp_path):
    directory, truth = reference_dir
    out = tmp_path / "assets" / "converted.pkl"
    sidecar = cdr.convert_deploy_reference(directory, out, fps_source=SYNTHETIC_FPS,
                                           mj_model=mj_model)
    return directory, truth, out, sidecar


def test_the_sidecar_is_written_next_to_the_pickle(converted):
    _, _, out, sidecar = converted
    path = Path(str(out) + ".provenance.json")
    assert path.is_file()
    assert json.loads(path.read_text())["output"]["sha256"] == sidecar["output"]["sha256"]


def test_the_sidecar_records_the_source_path_and_hashes(converted):
    directory, _, _, sidecar = converted
    source = sidecar["source"]
    assert source["path"] == str(directory.resolve())
    for name, digest in source["sha256_of_files"].items():
        assert digest == hashlib.sha256((directory / name).read_bytes()).hexdigest()


def test_the_sidecar_records_the_source_rate_and_frame_count(converted):
    _, truth, _, sidecar = converted
    assert sidecar["source"]["fps"] == SYNTHETIC_FPS
    assert sidecar["source"]["n_frames"] == truth["n_frames"]
    assert sidecar["source"]["fps_is_declared_by_the_format"] is False
    assert sidecar["source"]["fps_evidence"]
    assert sidecar["output"]["n_frames"] == truth["n_frames"]
    assert sidecar["duration"]["preserved"] is True


def test_the_sidecar_output_hash_is_the_hash_of_the_pickle(converted):
    _, _, out, sidecar = converted
    assert sidecar["output"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert sidecar["output"]["path"] == str(out.resolve())
    assert sidecar["output"]["written_by"] == "scene2motion.sonic_export.write_motion_pkl"


def test_the_sidecar_records_both_joint_order_checks(converted):
    _, _, _, sidecar = converted
    check = sidecar["joint_order_check"]
    assert check["by_name"]["passed"] is True
    assert check["by_geometry"]["passed"] is True
    assert check["by_geometry"]["worst_shared_frame_error_m"] < cdr.EXACT_FRAME_TOL_M
    assert len(check["permutation_isaaclab_to_mujoco"]) == NUM_DOF
    assert len(check["vendor_sources_agree"]) == len(cdr.PERMUTATION_SOURCES)


def test_the_sidecar_binds_the_checkouts_it_read(converted):
    _, _, _, sidecar = converted
    provenance = sidecar["provenance"]
    assert provenance["mjcf_sha256"] == hashlib.sha256(
        Path(cdr.ARDY_G1_XML).read_bytes()).hexdigest()
    assert set(provenance["sonic_sources"]) >= {rel for rel, _ in cdr.PERMUTATION_SOURCES}
    assert provenance["sonic_export_sha256"]


def test_the_sidecar_is_json_serialisable_and_claims_no_tracking_result(converted):
    _, _, _, sidecar = converted
    json.dumps(sidecar)
    assert "nothing here establishes that this reference tracks" in sidecar["scope"]
    # The asset is a candidate, never an established control: the purpose line must not assert
    # what the scope line disclaims, and the directory must not read as campaign evidence.
    assert sidecar["campaign_evidence"] is False
    assert sidecar["purpose"].startswith("candidate control reference")
    assert "INHERITED" in sidecar["purpose"]
    assert "known-trackable control reference" not in sidecar["purpose"]


def test_the_sidecar_screens_the_reference_without_claiming_an_outcome(converted):
    _, _, _, sidecar = converted
    screen = sidecar["reference_screen"]
    # Same thresholds the reference screen uses everywhere else, bound by the exp016 receipt.
    assert screen["thresholds"]["receipt_sha256"]
    assert screen["measured_at_fps"] == SYNTHETIC_FPS
    assert screen["flagged_by_calibrated_gate"] == (
        screen["max_unsupported_run_s"] > screen["thresholds"]["max_unsupported_run_s"])
    assert "never an executed outcome" in screen["interpretation"]


def test_the_sidecar_hashes_every_file_in_the_directory_not_only_the_ones_it_reads(
        converted, tmp_path):
    directory, _, _, sidecar = converted
    source = sidecar["source"]
    on_disk = {p.name for p in Path(directory).iterdir() if p.is_file()}
    assert set(source["sha256_of_files"]) == on_disk
    assert set(source["files_read_by_this_converter"]) == set(cdr.REQUIRED_FILES)
    assert set(source["files_present_but_unread"]) == on_disk - set(cdr.REQUIRED_FILES)


def test_an_edit_to_an_unread_file_changes_the_source_binding(reference_dir, mj_model, tmp_path):
    directory, _ = reference_dir
    out = tmp_path / "a.pkl"
    first = cdr.convert_deploy_reference(directory, out, fps_source=SYNTHETIC_FPS,
                                         mj_model=mj_model)
    (Path(directory) / "notes_the_loader_never_reads.txt").write_text("hello\n")
    second = cdr.convert_deploy_reference(directory, tmp_path / "b.pkl",
                                          fps_source=SYNTHETIC_FPS, mj_model=mj_model)
    assert second["output"]["sha256"] == first["output"]["sha256"]
    assert (second["source"]["sha256_of_source_bytes"]
            != first["source"]["sha256_of_source_bytes"])


def test_the_sidecar_records_the_working_tree_state_not_just_a_commit(converted):
    _, _, _, sidecar = converted
    for key in ("scene2motion", "sonic"):
        state = sidecar["provenance"][key]
        assert set(state) == {"commit", "dirty", "dirty_paths"}
        # A commit alone does not reproduce a dirty tree; the flag has to travel with it.
        assert state["dirty"] is None or isinstance(state["dirty"], bool)
    assert sidecar["provenance"]["scene2motion_commit"] == (
        sidecar["provenance"]["scene2motion"]["commit"])


def test_the_sidecar_records_the_vendor_quaternion_conflict_and_how_it_was_settled(converted):
    _, _, _, sidecar = converted
    order = sidecar["source"]["root_quaternion_component_order"]
    assert order["used"] == "wxyz"
    assert "visualize_motion.py" in order["vendor_conflict"]
    # The header and the vendor's own player disagree; only the geometry separates them.
    assert (order["shared_frame_error_m_reading_xyzw"]
            > 1000 * order["shared_frame_error_m_reading_wxyz"])


# ---------------------------------------------------------------- survey

def test_the_survey_reports_every_reference_and_writes_no_pickle(tmp_path, tables, mj_model):
    root = tmp_path / "example"
    root.mkdir()
    build_synthetic_reference(root / "asset_a", mj_model, tables, n_frames=12)
    build_synthetic_reference(root / "asset_b", mj_model, tables, n_frames=8)
    (root / "not_a_reference").mkdir()
    out = tmp_path / "survey.json"
    result = cdr.survey_deploy_references(root, out, fps_source=SYNTHETIC_FPS,
                                          mj_model=mj_model)
    assert result["n_directories"] == 2 and result["n_converted"] == 2
    assert [row["name"] for row in result["assets"]] == ["asset_a", "asset_b"]
    assert result["assets"][0]["n_frames"] == 12
    assert result["assets"][0]["duration_frames_over_fps_s"] == 12 / SYNTHETIC_FPS
    assert all("reference_screen" in row for row in result["assets"])
    assert len(result["summary"]["max_unsupported_run_s_sorted"]) == 2
    # A survey is a measurement, not a prepared campaign input: no pickle, and it says so.
    assert list(tmp_path.glob("**/*.pkl")) == []
    assert result["campaign_evidence"] is False
    assert json.loads(out.read_text())["schema"] == cdr.SURVEY_SCHEMA_VERSION


def test_the_survey_records_a_refusal_instead_of_aborting(tmp_path, tables, mj_model):
    root = tmp_path / "example"
    root.mkdir()
    build_synthetic_reference(root / "good", mj_model, tables, n_frames=10)
    build_synthetic_reference(root / "wrong_order", mj_model, tables, n_frames=10,
                              joint_order="mujoco")
    result = cdr.survey_deploy_references(root, tmp_path / "survey.json",
                                          fps_source=SYNTHETIC_FPS, mj_model=mj_model)
    rows = {row["name"]: row for row in result["assets"]}
    assert rows["good"]["converted"] is True
    assert rows["wrong_order"]["converted"] is False
    assert "geometry check failed" in rows["wrong_order"]["refused"]
    assert result["n_converted"] == 1 and result["n_directories"] == 2


def test_the_survey_refuses_a_root_with_no_references(tmp_path, tables, mj_model):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(cdr.ConversionError, match="no deployment reference directories"):
        cdr.survey_deploy_references(empty, tmp_path / "survey.json", mj_model=mj_model)


# ---------------------------------------------------------------- idempotence

def test_reconversion_produces_the_same_bytes(converted, mj_model):
    directory, _, out, sidecar = converted
    again = cdr.convert_deploy_reference(directory, out, fps_source=SYNTHETIC_FPS,
                                         mj_model=mj_model)
    assert again["output"]["sha256"] == sidecar["output"]["sha256"]


def test_an_existing_pickle_with_different_bytes_is_not_replaced(converted, mj_model):
    directory, _, out, _ = converted
    out.write_bytes(b"not a motion pickle")
    with pytest.raises(cdr.ConversionError, match="exists with different bytes"):
        cdr.convert_deploy_reference(directory, out, fps_source=SYNTHETIC_FPS, mj_model=mj_model)
    assert out.read_bytes() == b"not a motion pickle"
    cdr.convert_deploy_reference(directory, out, fps_source=SYNTHETIC_FPS, mj_model=mj_model,
                                 force=True)
    assert out.read_bytes() != b"not a motion pickle"
