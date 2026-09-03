"""SONIC *deployment* reference directory -> the motion pickle our eval bridge consumes.

Why this exists
---------------
EXP-024b needs a control the tracker itself ships.  The checkout carries thirteen deployment
references that the vendor distributes as examples of motions this controller is deployed to
execute, among them

    gear_sonic_deploy/reference/example/tired_one_leg_jumping_R_001__A359

-- but in the **deployment** format (a directory of CSVs written by
``gear_sonic_deploy/reference/convert_motions.py``), not in the motion-library pickle that
``scene2motion.sonic_export.write_motion_pkl`` produces and that every SONIC launch in this
repository consumes.  This module converts one deployment directory into that pickle.

What the converted asset can and cannot support
-----------------------------------------------
It is a real jump -- the feet leave the floor and the pelvis rises 0.31 m over the clip -- but
its bilateral no-support runs are **short**.  Measured through this project's own screen
(``experiments/analyze_trackability_contract.features`` with the hash-locked exp016 support
thresholds) at the declared 50 fps source rate, the longest run is 0.200 s and the flight
fraction is 0.046, so the asset **passes** the calibrated gate (which flags
``max_unsupported_run_s > 0.20 s``) -- by a single frame.  The shipped walk's longest run is
0.000 s.  For contrast, the twelve EXP-021 references that clear the staged 5 cm box run
0.44-3.12 s with none inside the gate
(``outputs/analysis_trackability_contract/receipt.json`` ->
``summary.exact_clear_5cm_at_x1.2.max_unsupported_run_s_sorted`` and
``n_within_calibrated_gate``).

This asset is therefore **not** a "long no-support run, yet trackable" control -- it does not
sit on the far side of the screen at all.  Nor does any of its siblings: over all thirteen
shipped references the longest no-support run is **0.24 s** (both macarena variants, the only
two the gate flags; seven assets are at 0.000 s), against 0.44-3.12 s for the EXP-021
references that clear the box.  The claim this set can support is a different one: *motions the
vendor ships as deployment examples have flight runs at or just past the gate, so the screen is
not a blanket penalty on dynamic motion.*  Even that needs the tracking run, because nothing here establishes that this
reference executes under our evaluation configuration -- it ships as a **deployment** example
and its outcome under the release **eval** configuration the bridge uses is unrecorded.
Whoever preregisters EXP-024b must write the motivation from this, not from the older
"known-trackable, long-run" premise.  Every number above is in
``outputs/probe_deploy_reference_conversion/survey.json`` (``campaign_evidence: false``).

The two things that would silently corrupt the control, and how each is checked
--------------------------------------------------------------------------------
1. **Joint order.**  The deployment CSV's 29 joint columns are in *IsaacLab* order; the motion
   pickle needs *MuJoCo/MJCF actuator* order.  A wrong permutation crashes nothing: it produces
   a plausible-looking but different motion, and every EXP-024b number would be wrong.  The
   permutation is therefore never trusted:

     * it is read from three independent vendor sources and required to agree
       (``MJ_TO_IL`` in ``gear_sonic/data_process/convert_soma_csv_to_motion_lib.py``,
       ``isaaclab_to_mujoco`` in ``gear_sonic_deploy/visualize_motion.py`` -- the vendor's own
       player for this exact directory layout -- and ``G1_ISAACLAB_TO_MUJOCO_DOF`` in
       ``gear_sonic/envs/manager_env/robots/g1.py``);
     * it is checked **by name**, against the vendor's IsaacLab joint-name list, entry by entry;
     * it is checked **by geometry** on the asset's own data: the CSVs carry world-frame
       positions of 14 tracked bodies, so MuJoCo forward kinematics on the reordered angles
       must reproduce those positions.  Candidate orderings (identity, the permutation, its
       inverse) are scored and the permutation must be the unique best by a wide margin.  On
       the shipped jump this reproduces the shared-frame bodies (pelvis and both leg chains)
       to 1.6e-6 m, against 0.711 m for the identity ordering and 0.702 m for the inverse --
       a ratio of 64x on the whole-body maximum (all three figures are in the sidecar's
       ``candidate_scores``, not retyped here).
     * the **root quaternion convention** is scored the same way, and it needs to be: the
       file's own header declares ``body_0_w,body_0_x,body_0_y,body_0_z`` (wxyz), but the
       vendor's own player for this layout reads the same four columns at
       ``gear_sonic_deploy/visualize_motion.py:88`` under the comment ``# [x, y, z, w]``.  The
       two cannot both be right and nothing in the vendor tree settles it, so the asset's
       geometry does: wxyz reproduces the shared-frame bodies to 1.6e-6 m and xyzw to 1.296 m.
       The conflict and both residuals go into the sidecar so a reader who opens the player is
       not left with an unresolved contradiction.

2. **Physical duration.**  The deployment format records **no** frame rate -- ``convert_motions.py``
   writes no fps field -- while the motion pickle's ``fps`` is exactly what SONIC's motion
   library uses as the *source* rate when it resamples to its own ``target_fps``
   (``sonic_release/config.yaml: target_fps: 50``;
   ``gear_sonic/utils/motion_lib/motion_lib_base.py:1875-1879`` calls
   ``fk_batch(..., fps=curr_file["fps"], target_fps=self.target_fps, ...)``, and ``:1334``
   computes the clip length as ``1.0 / motion_fps * (num_frames - 1)``).  Declaring 500 frames at 25 fps instead
   of 50 would play the jump at half speed and change every no-support run length.  The source
   rate is therefore an explicit argument (default 50, evidence below) and the converter asserts
   that the written clip lasts the same number of seconds as the source: exactly under the
   frame-count convention ``T / fps``, and to within one source frame period under the motion
   library's own span convention ``(T - 1) / fps``, which is what a legitimate stride resample
   costs at the end of a clip.  Both numbers are recorded.

   Evidence for 50 fps: the vendor's player for this directory layout hardcodes ``fps = 50``
   (``gear_sonic_deploy/visualize_motion.py``), and the release config's motion library uses
   ``target_fps: 50``.  A consistency check on the asset itself -- fitting dt to the CSV's own
   ``body_lin_vel``/``joint_vel`` channels by finite differences -- gives 46.9 fps and
   52.5 fps on the shipped jump, which bracket 50 but are not precise enough to determine
   it; the sidecar records both.

Everything else is deferred rather than reinvented: the conversion math follows the vendor's
``convert_sequence`` (root translation = tracked body 0 = pelvis, root rotation = its quaternion
wxyz -> xyzw, ``pose_aa`` = ``DOF_AXIS * dof`` with the root rotvec at index 0), and it is
executed by building an ARDY-shaped ``qpos`` array and handing it to ``write_motion_pkl``, so the
output is produced by the *same* code path as every archived reference in this project rather
than by a second implementation that could drift from it.

CPU only.  Launches nothing.  Writes a provenance sidecar next to the pickle.

Run:
    $S2M_PY experiments/convert_deploy_reference.py \\
        --reference $SONIC_ROOT/gear_sonic_deploy/reference/example/tired_one_leg_jumping_R_001__A359 \\
        --out outputs/probe_deploy_reference_conversion/tired_one_leg_jumping_R_001__A359.pkl

    $S2M_PY experiments/convert_deploy_reference.py \\
        --survey $SONIC_ROOT/gear_sonic_deploy/reference/example \\
        --out outputs/probe_deploy_reference_conversion/survey.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene2motion.robot import ARDY_G1_XML                                   # noqa: E402
from scene2motion.sonic_export import (                                      # noqa: E402
    G1_29DOF, NUM_DOF, SONIC_ROOT, check_joint_order, write_motion_pkl,
)

# ---------------------------------------------------------------- constants

#: The deployment format carries no frame rate; this is the vendor's own playback rate for this
#: directory layout (``gear_sonic_deploy/visualize_motion.py``) and the release motion library's
#: ``target_fps``.  It is an argument, not a silent default, and it is recorded in the sidecar.
DEPLOY_FPS_DEFAULT = 50

#: Files the loader requires and reads.  The asset is the whole directory, though: everything
#: present is hashed, and the sidecar keeps the read/unread split explicit.
REQUIRED_FILES = ("joint_pos.csv", "joint_vel.csv", "body_pos.csv", "body_quat.csv",
                  "body_lin_vel.csv", "metadata.txt")

SURVEY_SCHEMA_VERSION = "deploy-reference-survey-v1"

#: Vendor files that define the IsaacLab <-> MuJoCo joint permutation, and the symbol in each.
#: Three independent copies; all three must agree.
PERMUTATION_SOURCES = (
    ("gear_sonic/data_process/convert_soma_csv_to_motion_lib.py", "MJ_TO_IL"),
    ("gear_sonic_deploy/visualize_motion.py", "isaaclab_to_mujoco"),
    ("gear_sonic/envs/manager_env/robots/g1.py", "G1_ISAACLAB_TO_MUJOCO_DOF"),
)
#: Vendor list of the 29 joint names in IsaacLab order.
ISAACLAB_JOINT_NAMES_SOURCE = ("gear_sonic/envs/env_utils/joint_utils.py", "G1_ISAACLab_ORDER")
#: Vendor list of the 30 body names in IsaacLab order (indexed by the asset's ``_body_indexes``).
ISAACLAB_BODY_NAMES_SOURCE = ("gear_sonic/envs/manager_env/robots/g1.py", "G1_ISAACLAB_JOINTS")

#: Bodies whose link frames are identical between the MJCF and IsaacLab's USD, so forward
#: kinematics must reproduce their world positions to numerical precision.  Measured on the
#: shipped jump: max 1.6e-6 m.  The upper body carries a constant MJCF/USD frame-origin offset
#: (torso 11.18 mm, arms 5.30 mm, standard deviation over the sampled frames <= 1.18 mm), which
#: is a link-frame definition difference, not an ordering error -- it is identical whether the
#: check runs against ARDY's ``g1.xml`` or SONIC's own ``g1_29dof_rev_1_0.xml``.  The sidecar
#: carries these as ``joint_order_check.by_geometry.candidate_scores
#: .vendor_isaaclab_to_mujoco.per_body_max_error_m`` / ``per_body_std_error_m``.
EXACT_FRAME_BODIES = (
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
)
EXACT_FRAME_TOL_M = 1e-4      # 60x headroom over the observed 1.6e-6 m
ALL_BODY_TOL_M = 0.02         # covers the constant upper-body frame offset (max 11.18 mm)
MIN_DISCRIMINATION_RATIO = 5.0  # the runner-up ordering must be this much worse
MAX_FK_FRAMES = 50            # frames sampled for the geometry check


class ConversionError(RuntimeError):
    """Raised when a check fails.  Nothing is written when this is raised."""


# ---------------------------------------------------------------- vendor tables

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _int_list_from_source(source: str, symbol: str, origin: str) -> list[int]:
    """Read ``symbol = [...]`` or ``symbol = np.array([...], ...)`` out of a python source file.

    Parsed, never imported: these vendor modules pull in torch / Isaac at module scope.  Parsing
    keeps the table bound to the vendor file so it cannot drift from a private copy here.
    """
    tree = ast.parse(source)
    found: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    found.append(node.value)
    if not found:
        raise ConversionError(f"{origin} does not define {symbol}")
    value = found[0]
    if isinstance(value, ast.Call):                       # np.array([...], dtype=...)
        func = value.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "array" or not value.args:
            raise ConversionError(f"{origin}:{symbol} is a call this reader does not understand")
        value = value.args[0]
    try:
        literal = ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise ConversionError(f"{origin}:{symbol} is not a literal: {exc}") from exc
    if not isinstance(literal, (list, tuple)) or not all(isinstance(v, int) for v in literal):
        raise ConversionError(f"{origin}:{symbol} is not a list of ints")
    return list(literal)


def _str_list_from_source(source: str, symbol: str, origin: str) -> list[str]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    literal = ast.literal_eval(node.value)
                    if not isinstance(literal, (list, tuple)) or not all(
                            isinstance(v, str) for v in literal):
                        raise ConversionError(f"{origin}:{symbol} is not a list of strings")
                    return list(literal)
    raise ConversionError(f"{origin} does not define {symbol}")


@dataclass(frozen=True)
class VendorTables:
    """The permutation and the name lists, read from the SONIC checkout."""

    isaaclab_to_mujoco_dof: np.ndarray          # (29,) mj index -> IsaacLab index
    isaaclab_joint_names: tuple[str, ...]       # 29, IsaacLab order
    isaaclab_body_names: tuple[str, ...]        # 30, IsaacLab order
    sources: dict[str, dict[str, str]]          # relative path -> {sha256, symbol(s)}


def read_vendor_tables(sonic_root: Path | str = SONIC_ROOT) -> VendorTables:
    sonic_root = Path(sonic_root)
    sources: dict[str, dict[str, str]] = {}
    permutations: dict[str, list[int]] = {}
    for rel, symbol in PERMUTATION_SOURCES:
        path = sonic_root / rel
        if not path.is_file():
            raise ConversionError(f"vendor source not found: {path}")
        permutations[f"{rel}:{symbol}"] = _int_list_from_source(
            path.read_text(), symbol, rel)
        sources.setdefault(rel, {"sha256": _sha256(path), "symbols": ""})
        sources[rel]["symbols"] = ",".join(
            filter(None, [sources[rel]["symbols"], symbol]))

    distinct = {tuple(v) for v in permutations.values()}
    if len(distinct) != 1:
        raise ConversionError(
            "the vendor's IsaacLab->MuJoCo joint permutation is not consistent across its own "
            f"sources: {json.dumps(permutations, indent=2)}")
    perm = np.asarray(next(iter(distinct)), dtype=np.int64)
    if perm.shape != (NUM_DOF,) or sorted(perm.tolist()) != list(range(NUM_DOF)):
        raise ConversionError(
            f"vendor permutation is not a permutation of 0..{NUM_DOF - 1}: {perm.tolist()}")

    names: list[tuple[str, str, str]] = []
    for rel, symbol in (ISAACLAB_JOINT_NAMES_SOURCE, ISAACLAB_BODY_NAMES_SOURCE):
        path = sonic_root / rel
        if not path.is_file():
            raise ConversionError(f"vendor source not found: {path}")
        names.append((rel, symbol, path.read_text()))
        entry = sources.setdefault(rel, {"sha256": _sha256(path), "symbols": ""})
        entry["symbols"] = ",".join(filter(None, [entry["symbols"], symbol]))

    joint_names = _str_list_from_source(names[0][2], names[0][1], names[0][0])
    body_names = _str_list_from_source(names[1][2], names[1][1], names[1][0])
    if len(joint_names) != NUM_DOF:
        raise ConversionError(
            f"{names[0][0]}:{names[0][1]} has {len(joint_names)} names, expected {NUM_DOF}")
    if len(body_names) != NUM_DOF + 1:
        raise ConversionError(
            f"{names[1][0]}:{names[1][1]} has {len(body_names)} names, expected {NUM_DOF + 1}")
    return VendorTables(perm, tuple(joint_names), tuple(body_names), sources)


# ---------------------------------------------------------------- the deployment format

@dataclass(frozen=True)
class DeployReference:
    """One deployment reference directory, as it is actually laid out on disk."""

    directory: Path
    name: str
    joint_pos_il: np.ndarray        # (T, 29) IsaacLab DOF order, radians
    joint_vel_il: np.ndarray        # (T, 29) IsaacLab DOF order, rad/s
    body_pos_w: np.ndarray          # (T, 14, 3) world frame, metres
    body_quat_w: np.ndarray         # (T, 14, 4) world frame, wxyz (declared by the CSV header)
    body_lin_vel_w: np.ndarray      # (T, 14, 3) world frame, m/s
    tracked_body_indexes: tuple[int, ...]
    tracked_body_names: tuple[str, ...]
    files: dict[str, str]           # every file in the directory -> sha256
    files_read: tuple[str, ...]     # the subset this converter actually reads

    @property
    def n_frames(self) -> int:
        return int(self.joint_pos_il.shape[0])


def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ConversionError(f"{path.name} has no data rows")
    header = [c.strip() for c in lines[0].split(",")]
    data = np.array([[float(v) for v in line.split(",")] for line in lines[1:] if line.strip()],
                    dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(header):
        raise ConversionError(
            f"{path.name}: {data.shape[1] if data.ndim == 2 else '?'} columns of data against "
            f"{len(header)} header fields")
    return header, data


def _expect_header(header: Sequence[str], expected: Sequence[str], path: Path) -> None:
    if list(header) != list(expected):
        bad = next((i for i, (a, b) in enumerate(zip(header, expected)) if a != b), None)
        detail = (f"first difference at column {bad}: file has {header[bad]!r}, "
                  f"expected {expected[bad]!r}") if bad is not None else \
                 f"file has {len(header)} columns, expected {len(expected)}"
        raise ConversionError(f"{path.name} is not the deployment reference layout; {detail}")


def _parse_metadata(path: Path) -> tuple[tuple[int, ...], int | None]:
    text = path.read_text()
    match = re.search(r"Body part indexes:\s*\[([^\]]*)\]", text)
    if not match:
        raise ConversionError(f"{path.name} has no 'Body part indexes:' block")
    indexes = tuple(int(v) for v in match.group(1).split())
    steps = re.search(r"Total timesteps:\s*(\d+)", text)
    return indexes, (int(steps.group(1)) if steps else None)


def load_deploy_reference(directory: Path | str,
                          tables: VendorTables | None = None) -> DeployReference:
    """Read one deployment reference directory, validating its declared column layout.

    The layout is not assumed: every header is checked against what
    ``gear_sonic_deploy/reference/convert_motions.py`` writes, which is also where the
    quaternion's wxyz component order is declared (``body_0_w,body_0_x,body_0_y,body_0_z``).
    """
    directory = Path(directory)
    tables = tables or read_vendor_tables()
    if not directory.is_dir():
        raise ConversionError(f"not a directory: {directory}")

    files = REQUIRED_FILES
    for name in files:
        if not (directory / name).is_file():
            raise ConversionError(f"{directory} is missing {name}")

    jp_header, jp = _read_csv(directory / "joint_pos.csv")
    n_dof = len(jp_header)
    if n_dof != NUM_DOF:
        raise ConversionError(
            f"joint_pos.csv has {n_dof} joint columns, this converter handles {NUM_DOF} "
            f"(G1 29-DOF)")
    _expect_header(jp_header, [f"joint_{i}" for i in range(n_dof)],
                   directory / "joint_pos.csv")

    jv_header, jv = _read_csv(directory / "joint_vel.csv")
    _expect_header(jv_header, [f"joint_vel_{i}" for i in range(n_dof)],
                   directory / "joint_vel.csv")

    bp_header, bp = _read_csv(directory / "body_pos.csv")
    if len(bp_header) % 3:
        raise ConversionError("body_pos.csv column count is not a multiple of 3")
    n_bodies = len(bp_header) // 3
    _expect_header(bp_header,
                   [f"body_{i // 3}_{'xyz'[i % 3]}" for i in range(3 * n_bodies)],
                   directory / "body_pos.csv")

    bq_header, bq = _read_csv(directory / "body_quat.csv")
    _expect_header(bq_header,
                   [f"body_{i // 4}_{'wxyz'[i % 4]}" for i in range(4 * n_bodies)],
                   directory / "body_quat.csv")

    bv_header, bv = _read_csv(directory / "body_lin_vel.csv")
    _expect_header(bv_header,
                   [f"body_{i // 3}_vel_{'xyz'[i % 3]}" for i in range(3 * n_bodies)],
                   directory / "body_lin_vel.csv")

    lengths = {"joint_pos": len(jp), "joint_vel": len(jv), "body_pos": len(bp),
               "body_quat": len(bq), "body_lin_vel": len(bv)}
    if len(set(lengths.values())) != 1:
        raise ConversionError(f"the CSVs disagree on the frame count: {lengths}")
    n_frames = len(jp)

    indexes, declared_steps = _parse_metadata(directory / "metadata.txt")
    if len(indexes) != n_bodies:
        raise ConversionError(
            f"metadata.txt lists {len(indexes)} tracked bodies against {n_bodies} in the CSVs")
    if declared_steps is not None and declared_steps != n_frames:
        raise ConversionError(
            f"metadata.txt declares {declared_steps} timesteps against {n_frames} CSV rows")
    if max(indexes) >= len(tables.isaaclab_body_names):
        raise ConversionError(
            f"metadata.txt body index {max(indexes)} is outside the vendor's "
            f"{len(tables.isaaclab_body_names)}-body IsaacLab order")
    body_names = tuple(tables.isaaclab_body_names[i] for i in indexes)

    quat = bq.reshape(n_frames, n_bodies, 4)
    norms = np.linalg.norm(quat, axis=-1)
    if not np.all(np.abs(norms - 1.0) < 1e-3):
        raise ConversionError(
            f"body_quat.csv rows are not unit quaternions (norm range "
            f"{norms.min():.6f}..{norms.max():.6f}); the wxyz reading may be wrong")

    return DeployReference(
        directory=directory,
        name=directory.name,
        joint_pos_il=jp.astype(np.float64),
        joint_vel_il=jv.astype(np.float64),
        body_pos_w=bp.reshape(n_frames, n_bodies, 3),
        body_quat_w=quat,
        body_lin_vel_w=bv.reshape(n_frames, n_bodies, 3),
        tracked_body_indexes=indexes,
        tracked_body_names=body_names,
        # Hash EVERY file in the directory, not only the ones the loader reads: the asset is
        # the whole directory, and an edit to body_ang_vel.csv or info.txt must not be able to
        # slip past the source binding just because this converter has no use for it.
        files={path.name: _sha256(path)
               for path in sorted(directory.iterdir()) if path.is_file()},
        files_read=files,
    )


# ---------------------------------------------------------------- joint-order checks

def check_joint_order_by_name(tables: VendorTables) -> dict[str, Any]:
    """Entry-by-entry: applying the permutation to the IsaacLab names must give MuJoCo order.

    Static, and independent of the asset: it compares the vendor's own name list against the
    order ``sonic_export`` requires.  It cannot catch a file whose columns are not in the order
    its own vendor claims -- that is what the geometry check is for.
    """
    permuted = [tables.isaaclab_joint_names[i] for i in tables.isaaclab_to_mujoco_dof.tolist()]
    if permuted != list(G1_29DOF):
        bad = next(i for i, (a, b) in enumerate(zip(permuted, G1_29DOF)) if a != b)
        raise ConversionError(
            f"joint-order name check failed at MuJoCo index {bad}: the vendor permutation puts "
            f"{permuted[bad]!r} where the motion pickle needs {G1_29DOF[bad]!r}")
    return {"passed": True, "n_joints": len(permuted),
            "mujoco_order_from_isaaclab_names": permuted}


def _fk_body_positions(mj_model, mj_data, body_ids: Sequence[int],
                       root_pos: np.ndarray, root_quat_wxyz: np.ndarray,
                       dof_mj: np.ndarray) -> np.ndarray:
    import mujoco
    mj_data.qpos[:3] = root_pos
    mj_data.qpos[3:7] = root_quat_wxyz
    mj_data.qpos[7:7 + NUM_DOF] = dof_mj
    mujoco.mj_kinematics(mj_model, mj_data)
    return np.asarray(mj_data.xpos)[list(body_ids)].copy()


def check_root_quaternion_order(reference: DeployReference, tables: VendorTables,
                                mj_model=None, *,
                                mjcf_path: Path | str = ARDY_G1_XML) -> dict[str, float]:
    """Score both readings of the root quaternion columns against the asset's own geometry.

    ``body_quat.csv``'s header declares ``body_0_w,body_0_x,body_0_y,body_0_z`` (wxyz), but the
    vendor's own player for this layout (``gear_sonic_deploy/visualize_motion.py:88``) slices
    the same four columns under the comment ``# [x, y, z, w]``.  The two cannot both be right,
    and nothing in the vendor tree settles it, so it is settled here on the asset's data: a
    wrong root rotation moves every tracked body, so forward kinematics on the shared-frame
    bodies separates the two readings by five orders of magnitude.  Recorded in the sidecar so
    a reader who opens the player is not left with an unresolved contradiction.
    """
    import mujoco

    if mj_model is None:
        mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    mj_data = mujoco.MjData(mj_model)
    body_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in reference.tracked_body_names]
    if any(bid < 0 for bid in body_ids):
        raise ConversionError("a tracked body is missing from the MJCF")
    n_frames = reference.n_frames
    frames = np.unique(np.linspace(0, n_frames - 1, min(n_frames, MAX_FK_FRAMES)).astype(int))
    shared = [i for i, name in enumerate(reference.tracked_body_names)
              if name in EXACT_FRAME_BODIES]
    out: dict[str, float] = {}
    for label, take in (("wxyz_m", (0, 1, 2, 3)), ("xyzw_m", (3, 0, 1, 2))):
        errors = np.array([
            np.linalg.norm(
                _fk_body_positions(mj_model, mj_data, body_ids,
                                   reference.body_pos_w[t, 0],
                                   reference.body_quat_w[t, 0][list(take)],
                                   reference.joint_pos_il[t][tables.isaaclab_to_mujoco_dof])
                - reference.body_pos_w[t], axis=1)
            for t in frames])
        out[label] = float(errors[:, shared].max())
    return out


def check_joint_order_by_geometry(reference: DeployReference, tables: VendorTables,
                                  mj_model=None, *,
                                  mjcf_path: Path | str = ARDY_G1_XML) -> dict[str, Any]:
    """Score candidate orderings against the asset's own tracked-body world positions.

    The deployment CSVs carry, for the same frames, both the joint angles and the world-frame
    positions of 14 tracked bodies.  Forward kinematics from the root pose and the reordered
    angles must reproduce those positions.  Three candidate orderings are scored -- identity,
    the vendor permutation, and its inverse (the realistic confusion) -- and the vendor
    permutation must be the unique best by a wide margin.
    """
    import mujoco

    if mj_model is None:
        mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    check_joint_order(mj_model)          # the MJCF's hinge order is the motion library's order
    mj_data = mujoco.MjData(mj_model)

    body_ids = []
    for name in reference.tracked_body_names:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ConversionError(
                f"tracked body {name!r} is not a body in {mjcf_path}; the asset's tracked-body "
                f"list cannot be checked against this model")
        body_ids.append(bid)

    n_frames = reference.n_frames
    frames = np.unique(np.linspace(0, n_frames - 1, min(n_frames, MAX_FK_FRAMES)).astype(int))
    perm = tables.isaaclab_to_mujoco_dof
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(NUM_DOF)
    candidates = {
        "vendor_isaaclab_to_mujoco": perm,
        "identity_no_reorder": np.arange(NUM_DOF),
        "inverse_mujoco_to_isaaclab": inverse,
    }

    scores: dict[str, dict[str, Any]] = {}
    for label, candidate in candidates.items():
        errors = np.array([
            np.linalg.norm(
                _fk_body_positions(mj_model, mj_data, body_ids,
                                   reference.body_pos_w[t, 0],
                                   reference.body_quat_w[t, 0],
                                   reference.joint_pos_il[t][candidate])
                - reference.body_pos_w[t], axis=1)
            for t in frames])
        scores[label] = {
            "max_error_m": float(errors.max()),
            "mean_error_m": float(errors.mean()),
            "per_body_max_error_m": {name: float(col.max()) for name, col
                                     in zip(reference.tracked_body_names, errors.T)},
            "per_body_std_error_m": {name: float(col.std()) for name, col
                                     in zip(reference.tracked_body_names, errors.T)},
        }

    chosen = scores["vendor_isaaclab_to_mujoco"]
    others = {k: v["max_error_m"] for k, v in scores.items() if k != "vendor_isaaclab_to_mujoco"}
    runner_up = min(others.values())

    exact = [n for n in reference.tracked_body_names if n in EXACT_FRAME_BODIES]
    if not exact:
        raise ConversionError(
            "none of the asset's tracked bodies is one whose link frame is shared between the "
            "MJCF and IsaacLab, so the geometry check has no exact anchor")
    worst_exact = max(chosen["per_body_max_error_m"][n] for n in exact)
    worst_all = chosen["max_error_m"]
    ratio = float(runner_up / worst_all) if worst_all > 0 else float("inf")

    failures = []
    if worst_exact > EXACT_FRAME_TOL_M:
        failures.append(
            f"shared-frame bodies {exact} reproduce to {worst_exact:.6f} m, over the "
            f"{EXACT_FRAME_TOL_M} m tolerance")
    if worst_all > ALL_BODY_TOL_M:
        failures.append(
            f"worst tracked body reproduces to {worst_all:.6f} m, over the {ALL_BODY_TOL_M} m "
            f"tolerance")
    if ratio < MIN_DISCRIMINATION_RATIO:
        failures.append(
            f"the vendor permutation is only {ratio:.2f}x better than the best alternative "
            f"({min(others, key=others.get)}), under the required {MIN_DISCRIMINATION_RATIO}x; "
            f"either the ordering is wrong or this clip is too close to the rest pose to "
            f"identify it")
    if failures:
        raise ConversionError(
            "joint-order geometry check failed on " + reference.name + ": "
            + "; ".join(failures)
            + f". Scores: {json.dumps({k: v['max_error_m'] for k, v in scores.items()})}")

    return {
        "passed": True,
        "method": ("MuJoCo forward kinematics from the reordered joint angles and the root pose, "
                   "compared against the asset's own tracked-body world positions"),
        "mjcf": str(mjcf_path),
        "mjcf_sha256": _sha256(mjcf_path),
        "n_frames_checked": int(len(frames)),
        "tracked_bodies": list(reference.tracked_body_names),
        "shared_frame_bodies": exact,
        "worst_shared_frame_error_m": float(worst_exact),
        "worst_tracked_body_error_m": float(worst_all),
        "discrimination_ratio_vs_best_alternative": ratio,
        "tolerances": {"shared_frame_m": EXACT_FRAME_TOL_M, "all_bodies_m": ALL_BODY_TOL_M,
                       "min_discrimination_ratio": MIN_DISCRIMINATION_RATIO},
        "candidate_scores": scores,
    }


# ---------------------------------------------------------------- fps / duration

def implied_fps_from_velocity_channels(reference: DeployReference) -> dict[str, float]:
    """Least-squares dt from the asset's own velocity channels: a consistency check, not a rate.

    Central differences give ``x[t+1] - x[t-1] = 2 dt v[t]``; the least-squares dt over every
    body/joint and frame is reported.  Finite-difference bias makes this a bracket (46.9 and
    52.5 fps on the shipped jump), never a determination of the source rate.
    """
    out: dict[str, float] = {}
    for label, pos, vel in (("body_lin_vel", reference.body_pos_w, reference.body_lin_vel_w),
                            ("joint_vel", reference.joint_pos_il, reference.joint_vel_il)):
        if len(pos) < 3:
            continue
        delta = (pos[2:] - pos[:-2]).ravel()
        speed = (2.0 * vel[1:-1]).ravel()
        denominator = float(speed @ speed)
        if denominator <= 0:
            continue
        dt = float(speed @ delta) / denominator
        if dt > 0:
            out[f"{label}_implied_fps"] = 1.0 / dt
    return out


def _stride_resample(qpos: np.ndarray, fps_source: int, fps_out: int) -> np.ndarray:
    """The vendor's ``downsample_sequence`` rule, refused unless it preserves duration."""
    if fps_out == fps_source:
        return qpos
    if fps_out <= 0 or fps_source % fps_out:
        raise ConversionError(
            f"output rate {fps_out} is not an exact divisor of the source rate {fps_source}; "
            f"resampling to it would not preserve the clip's duration")
    stride = fps_source // fps_out
    if len(qpos) % stride:
        raise ConversionError(
            f"cannot stride-resample {len(qpos)} frames by {stride} without changing the "
            f"duration; convert at the source rate instead")
    return qpos[::stride]


def assert_duration_preserved(n_frames_source: int, fps_source: int,
                              n_frames_out: int, fps_out: int) -> dict[str, Any]:
    """The clip must last the same number of seconds after conversion.

    The hazard this exists for is declaring the wrong rate rather than dropping frames: the
    motion pickle's ``fps`` is what SONIC's motion library treats as the source rate when it
    resamples to its own ``target_fps``, so writing 500 frames as 25 fps would play the jump at
    half speed and double every no-support run length -- silently, with no shape change and no
    error anywhere downstream.

    Two conventions are reported.  ``T / fps`` must match exactly.  The motion library's own
    span, ``(T - 1) / fps``, may differ by at most one source frame period, which is what a
    legitimate stride resample costs at the end of the clip.
    """
    source_duration = n_frames_source / fps_source
    out_duration = n_frames_out / fps_out
    source_span = (n_frames_source - 1) / fps_source
    out_span = (n_frames_out - 1) / fps_out
    span_tolerance = 1.0 / fps_source + 1e-9
    if abs(out_duration - source_duration) > 1e-9 or abs(out_span - source_span) > span_tolerance:
        raise ConversionError(
            f"duration is not preserved: source {n_frames_source} frames @ {fps_source} fps "
            f"= {source_duration:.6f} s (span {source_span:.6f} s), output {n_frames_out} "
            f"frames @ {fps_out} fps = {out_duration:.6f} s (span {out_span:.6f} s)")
    return {
        "source_frames_over_fps_s": source_duration,
        "output_frames_over_fps_s": out_duration,
        "source_span_frames_minus_one_over_fps_s": source_span,
        "output_span_frames_minus_one_over_fps_s": out_span,
        "span_tolerance_s": span_tolerance,
        "preserved": True,
        "motion_library_convention": "(num_frames - 1) / fps, motion_lib_base.py",
    }


# ---------------------------------------------------------------- conversion

def deploy_reference_to_qpos(reference: DeployReference, tables: VendorTables) -> np.ndarray:
    """(T, 36) ARDY-shaped qpos: root xyz, root quaternion wxyz, 29 DOF in MuJoCo order.

    Follows the vendor's ``convert_sequence``: root translation is tracked body 0 (the pelvis,
    checked against the asset's own body-index metadata) and root rotation is its quaternion.
    Feeding this to ``write_motion_pkl`` reproduces the vendor's ``pose_aa``/``root_rot``
    construction through the same code path every archived reference in this project used.
    """
    if reference.tracked_body_names[0] != "pelvis":
        raise ConversionError(
            f"tracked body 0 is {reference.tracked_body_names[0]!r}, not the pelvis; the root "
            f"pose cannot be read from it")
    n_frames = reference.n_frames
    qpos = np.zeros((n_frames, 7 + NUM_DOF), dtype=np.float32)
    qpos[:, 0:3] = reference.body_pos_w[:, 0, :]
    qpos[:, 3:7] = reference.body_quat_w[:, 0, :]                 # wxyz, as the header declares
    qpos[:, 7:] = reference.joint_pos_il[:, tables.isaaclab_to_mujoco_dof]
    return qpos


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def measure_reference_screen(qpos: np.ndarray, fps: float, *, body=None) -> dict[str, Any]:
    """Run this project's calibrated support screen over a converted reference.

    Imported, never re-implemented: ``experiments.analyze_trackability_contract`` owns both the
    feature definitions and the hash-locked exp016 thresholds, so this row is the same
    measurement the reference screen makes everywhere else and can be compared with
    ``outputs/analysis_trackability_contract/receipt.json`` directly.

    This is a *kinematic* measurement of the reference.  It says nothing about whether the
    tracker executes it -- passing the gate is a prediction that the evaluator will not cut the
    episode, not an outcome.
    """
    from experiments import analyze_trackability_contract as tc
    from scene2motion.robot import G1Body

    thresholds = tc.load_support_thresholds()
    if body is None:
        body = G1Body()
    feats = tc.features(body, np.asarray(qpos, dtype=float),
                        thresholds["support_height_m"], thresholds["support_speed_mps"],
                        fps=float(fps))
    gate_s = float(thresholds["max_unsupported_run_s"])
    longest = float(feats["max_unsupported_run_s"])
    return {
        "measured_at_fps": float(fps),
        "thresholds": thresholds,
        "max_unsupported_run_s": longest,
        # None when the clip never leaves the support envelope -- a real answer, not a zero.
        "first_nosupport_run_s": _opt_float(feats["first_nosupport_run_s"]),
        "first_nosupport_onset_s": _opt_float(feats["first_nosupport_onset_s"]),
        "bilateral_flight_frac": _opt_float(feats["bilateral_flight_frac"]),
        # None where the clip has no no-support run to describe.
        "ballistic_ratio": _opt_float(feats["ballistic_ratio"]),
        "longest_run_pelvis_rise_m": _opt_float(feats["longest_run_pelvis_rise_m"]),
        "root_z_range_m": _opt_float(feats["root_z_range"]),
        "min_foot_bottom_m": _opt_float(feats["min_foot_bottom_m"]),
        "max_foot_clearance_m": _opt_float(feats["max_foot_clearance_m"]),
        "flagged_by_calibrated_gate": bool(longest > gate_s),
        "gate_rule": f"flagged when max_unsupported_run_s > {gate_s} s",
        "interpretation": (
            "kinematic screen of the reference only. Passing is a prediction that the "
            "evaluator will not cut the episode, never an executed outcome"),
    }


def _git_state(path: Path) -> dict[str, Any]:
    """Commit AND working-tree state: a commit alone does not reproduce a dirty tree.

    ``git status --porcelain`` pads the status to two columns, so an unstaged modification's
    line starts with a space; the output must not be stripped before the path is sliced off.
    """
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                                  text=True, check=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    head = run("rev-parse", "HEAD")
    porcelain = run("status", "--porcelain")
    if head is None:
        return {"commit": None, "dirty": None, "dirty_paths": None}
    paths = None
    if porcelain is not None:
        seen = set()
        for line in porcelain.splitlines():
            if not line.strip():
                continue
            entry = line[3:] if len(line) > 3 else line.strip()
            seen.add(entry.split(" -> ")[-1].strip().strip('"'))
        paths = sorted(seen)
    return {"commit": head.strip(),
            "dirty": None if porcelain is None else bool(porcelain.strip()),
            "dirty_paths": paths}


def convert_deploy_reference(
    directory: Path | str,
    out_pkl: Path | str,
    *,
    fps_source: int = DEPLOY_FPS_DEFAULT,
    fps_out: int | None = None,
    motion_name: str | None = None,
    sonic_root: Path | str = SONIC_ROOT,
    mjcf_path: Path | str = ARDY_G1_XML,
    mj_model=None,
    force: bool = False,
) -> dict[str, Any]:
    """Convert one deployment reference directory; write the pickle and its provenance sidecar.

    Returns the sidecar contents.  Raises ``ConversionError`` -- writing nothing -- if any check
    fails.  Idempotent: re-running over an existing pickle recomputes it and refuses to replace
    it with different bytes unless ``force``.
    """
    import mujoco

    fps_out = int(fps_source if fps_out is None else fps_out)
    fps_source = int(fps_source)
    if fps_source <= 0:
        raise ConversionError("--fps-source must be positive; the deployment format records none")

    tables = read_vendor_tables(sonic_root)
    reference = load_deploy_reference(directory, tables)
    name = motion_name or reference.name

    if mj_model is None:
        mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    name_check = check_joint_order_by_name(tables)
    geometry_check = check_joint_order_by_geometry(
        reference, tables, mj_model=mj_model, mjcf_path=mjcf_path)
    quaternion_order_check = check_root_quaternion_order(
        reference, tables, mj_model=mj_model, mjcf_path=mjcf_path)

    qpos = deploy_reference_to_qpos(reference, tables)
    qpos_out = _stride_resample(qpos, fps_source, fps_out)
    screen = measure_reference_screen(qpos_out, fps_out)
    sonic_git = _git_state(Path(sonic_root))
    repo_git = _git_state(ROOT)

    duration = assert_duration_preserved(reference.n_frames, fps_source, len(qpos_out), fps_out)

    out_pkl = Path(out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_pkl.with_suffix(out_pkl.suffix + ".tmp")
    write_motion_pkl({name: qpos_out}, temporary, fps=fps_out, mj_model=mj_model)
    new_sha = _sha256(temporary)
    if out_pkl.exists() and not force:
        if _sha256(out_pkl) != new_sha:
            temporary.unlink()
            raise ConversionError(
                f"{out_pkl} exists with different bytes; inspect it, then pass force=True to "
                f"replace it")
        temporary.unlink()
    else:
        temporary.replace(out_pkl)

    with out_pkl.open("rb") as handle:
        entry = pickle.load(handle)[name]

    sidecar = {
        "converter": "experiments/convert_deploy_reference.py",
        "converter_sha256": _sha256(Path(__file__)),
        "campaign_evidence": False,
        "purpose": ("candidate control reference for EXP-024b, converted from the tracker's "
                    "deployment reference format into the motion pickle the evaluation bridge "
                    "consumes. 'Known-trackable' is INHERITED from the asset shipping as a "
                    "deployment example and has not been measured under the release eval "
                    "configuration; and its longest bilateral no-support run is short (see "
                    "reference_screen below), so it is not a control that sits on the far side "
                    "of the calibrated gate"),
        "source": {
            "path": str(Path(directory).resolve()),
            "name": reference.name,
            "sha256_of_files": reference.files,
            "files_read_by_this_converter": list(reference.files_read),
            "files_present_but_unread": [name for name in reference.files
                                         if name not in reference.files_read],
            "sha256_of_source_bytes": hashlib.sha256(
                b"".join(sorted(f"{k}:{v}".encode() for k, v in reference.files.items()))
            ).hexdigest(),
            "n_frames": reference.n_frames,
            "n_joint_columns": int(reference.joint_pos_il.shape[1]),
            "joint_column_order": "isaaclab",
            "n_tracked_bodies": len(reference.tracked_body_names),
            "tracked_body_indexes_isaaclab": list(reference.tracked_body_indexes),
            "tracked_body_names": list(reference.tracked_body_names),
            "root_quaternion_component_order": {
                "used": "wxyz",
                "declared_by": ("the body_quat.csv header itself: "
                                "body_0_w,body_0_x,body_0_y,body_0_z"),
                "vendor_conflict": (
                    "gear_sonic_deploy/visualize_motion.py:88 reads the same four columns as "
                    "[0, 1, 2, 3] under the comment '# [x, y, z, w]', which contradicts the "
                    "header this converter follows"),
                "settled_by": (
                    "the geometry check on the asset's own tracked-body world positions: "
                    "reading the columns as wxyz reproduces the leg chain to "
                    "worst_shared_frame_error_m below, reading them as xyzw gives 1.296 m"),
                "shared_frame_error_m_reading_wxyz": float(
                    geometry_check["worst_shared_frame_error_m"]),
                "shared_frame_error_m_reading_xyzw": float(quaternion_order_check["xyzw_m"]),
            },
            "fps": fps_source,
            "fps_is_declared_by_the_format": False,
            "fps_evidence": [
                "gear_sonic_deploy/visualize_motion.py plays this directory layout at fps = 50",
                "sonic_release/config.yaml motion_lib_cfg.target_fps: 50",
            ],
            "fps_consistency_check_from_velocity_channels":
                implied_fps_from_velocity_channels(reference),
        },
        "output": {
            "path": str(out_pkl.resolve()),
            "sha256": _sha256(out_pkl),
            "motion_key": name,
            "fps": fps_out,
            "n_frames": int(len(qpos_out)),
            "entry_keys": sorted(entry),
            "entry_shapes": {k: list(np.shape(v)) for k, v in entry.items()
                             if hasattr(v, "shape")},
            "written_by": "scene2motion.sonic_export.write_motion_pkl",
        },
        "duration": duration,
        "reference_screen": screen,
        "joint_order_check": {
            "by_name": name_check,
            "by_geometry": geometry_check,
            "permutation_isaaclab_to_mujoco": tables.isaaclab_to_mujoco_dof.tolist(),
            "vendor_sources_agree": [f"{rel}:{sym}" for rel, sym in PERMUTATION_SOURCES],
        },
        "provenance": {
            "sonic_root": str(Path(sonic_root).resolve()),
            "sonic": sonic_git,
            "sonic_commit": sonic_git["commit"],
            "sonic_sources": tables.sources,
            "mjcf": str(mjcf_path),
            "mjcf_sha256": _sha256(mjcf_path),
            "scene2motion": repo_git,
            "scene2motion_commit": repo_git["commit"],
            "sonic_export_sha256": _sha256(ROOT / "scene2motion/sonic_export.py"),
        },
        "scope": ("format conversion and kinematic screening only; nothing here establishes "
                  "that this reference tracks under our evaluation configuration -- that is "
                  "what EXP-024b would measure. Not campaign evidence: EXP-024b has not run"),
    }
    sidecar_path = Path(str(out_pkl) + ".provenance.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=False) + "\n")
    sidecar["sidecar_path"] = str(sidecar_path.resolve())
    return sidecar


# ---------------------------------------------------------------- survey

def is_deploy_reference_dir(path: Path) -> bool:
    """A directory laid out like a deployment reference (the loader's required files)."""
    path = Path(path)
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_FILES)


def survey_deploy_references(
    root: Path | str,
    out_json: Path | str,
    *,
    fps_source: int = DEPLOY_FPS_DEFAULT,
    sonic_root: Path | str = SONIC_ROOT,
    mjcf_path: Path | str = ARDY_G1_XML,
    mj_model=None,
) -> dict[str, Any]:
    """Check and screen every shipped deployment reference under ``root``; write one artifact.

    No pickle is written: each asset is converted in memory, so the survey costs nothing but
    CPU and leaves nothing that could later be mistaken for a prepared campaign input.  One row
    per asset carries the frame count, the duration at the declared source rate, both
    joint-order check reports and the reference-screen row -- which is the only way a statement
    like "no shipped asset has a long no-support run" can be cited instead of asserted.

    An asset that fails a check does not abort the survey: its row records the refusal, because
    "which of the shipped assets are convertible" is part of what the survey answers.
    """
    import mujoco

    root = Path(root)
    if not root.is_dir():
        raise ConversionError(f"not a directory: {root}")
    tables = read_vendor_tables(sonic_root)
    if mj_model is None:
        mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    from scene2motion.robot import G1Body
    body = G1Body()

    directories = sorted(d for d in root.iterdir() if is_deploy_reference_dir(d))
    if not directories:
        raise ConversionError(f"no deployment reference directories under {root}")

    rows: list[dict[str, Any]] = []
    for directory in directories:
        row: dict[str, Any] = {"name": directory.name, "path": str(directory.resolve())}
        try:
            reference = load_deploy_reference(directory, tables)
            geometry_check = check_joint_order_by_geometry(
                reference, tables, mj_model=mj_model, mjcf_path=mjcf_path)
            quaternion_check = check_root_quaternion_order(
                reference, tables, mj_model=mj_model, mjcf_path=mjcf_path)
            qpos = deploy_reference_to_qpos(reference, tables)
            row.update({
                "converted": True,
                "n_frames": reference.n_frames,
                "declared_source_fps": int(fps_source),
                "duration_frames_over_fps_s": reference.n_frames / float(fps_source),
                "duration_span_s": (reference.n_frames - 1) / float(fps_source),
                "sha256_of_source_bytes": hashlib.sha256(
                    b"".join(sorted(f"{k}:{v}".encode()
                                    for k, v in reference.files.items()))).hexdigest(),
                "joint_order_by_geometry": {
                    "worst_shared_frame_error_m":
                        geometry_check["worst_shared_frame_error_m"],
                    "worst_tracked_body_error_m":
                        geometry_check["worst_tracked_body_error_m"],
                    "discrimination_ratio_vs_best_alternative":
                        geometry_check["discrimination_ratio_vs_best_alternative"],
                },
                "root_quaternion_order_check_m": quaternion_check,
                "implied_fps_from_velocity_channels":
                    implied_fps_from_velocity_channels(reference),
                "reference_screen": measure_reference_screen(qpos, fps_source, body=body),
            })
        except ConversionError as exc:
            row.update({"converted": False, "refused": str(exc)})
        rows.append(row)

    screened = [r for r in rows if r.get("converted")]
    runs = sorted(r["reference_screen"]["max_unsupported_run_s"] for r in screened)
    flagged = [r["name"] for r in screened if r["reference_screen"]["flagged_by_calibrated_gate"]]
    result = {
        "schema": SURVEY_SCHEMA_VERSION,
        "analysis": "deploy_reference_survey",
        "campaign_evidence": False,
        "status": "complete",
        "interpretation": (
            "Every deployment reference the tracker checkout ships, converted in memory and "
            "measured with this project's calibrated support screen. The screen is kinematic: "
            "a row that passes the gate is a PREDICTION that the evaluator would not cut the "
            "episode, never a tracking outcome, and none of these assets has been run under "
            "our evaluation configuration. Not campaign evidence: EXP-024b has not run."
        ),
        "root": str(root.resolve()),
        "n_directories": len(directories),
        "n_converted": len(screened),
        "declared_source_fps": int(fps_source),
        "fps_caveat": ("the deployment format records no frame rate; every duration and every "
                       "run length here is at the declared source rate"),
        "summary": {
            "max_unsupported_run_s_sorted": runs,
            "longest_over_all_assets_s": (runs[-1] if runs else None),
            "gate_rule": (f"flagged when max_unsupported_run_s > "
                          f"{screened[0]['reference_screen']['thresholds']['max_unsupported_run_s']} s"
                          if screened else None),
            "n_flagged_by_calibrated_gate": len(flagged),
            "flagged_assets": flagged,
            "reading": (
                "Compare longest_over_all_assets_s against "
                "outputs/analysis_trackability_contract/receipt.json -> "
                "summary.exact_clear_5cm_at_x1.2.max_unsupported_run_s_sorted (0.44-3.12 s, "
                "n_within_calibrated_gate 0). A shipped asset is only the 'long no-support "
                "run, yet trackable' control EXP-024b was first motivated by if its run "
                "reaches that band; sitting a frame or two past the gate is not the same "
                "claim. Read n_flagged_by_calibrated_gate and the sorted runs together, and "
                "remember these are kinematic predictions -- no asset here has been executed "
                "under our evaluation configuration."
            ),
        },
        "provenance": {
            "converter": "experiments/convert_deploy_reference.py",
            "converter_sha256": _sha256(Path(__file__)),
            "sonic_root": str(Path(sonic_root).resolve()),
            "sonic": _git_state(Path(sonic_root)),
            "sonic_sources": tables.sources,
            "mjcf": str(mjcf_path),
            "mjcf_sha256": _sha256(mjcf_path),
            "scene2motion": _git_state(ROOT),
        },
        "assets": rows,
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")
    return result


# ---------------------------------------------------------------- CLI

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a SONIC deployment reference directory to a motion pickle.")
    parser.add_argument("--reference", default=None,
                        help="deployment reference directory (joint_pos.csv, body_pos.csv, ...)")
    parser.add_argument("--survey", default=None,
                        help=("instead of converting one reference, check and screen every "
                              "deployment reference directory under this root and write the "
                              "survey JSON to --out (nothing else is written)"))
    parser.add_argument("--out", required=True,
                        help="output .pkl path, or the survey JSON path with --survey")
    parser.add_argument("--name", default=None,
                        help="motion key inside the pickle (default: the directory name)")
    parser.add_argument("--fps-source", type=int, default=DEPLOY_FPS_DEFAULT,
                        help=(f"source frame rate; the deployment format records none "
                              f"(default {DEPLOY_FPS_DEFAULT}, the vendor's playback rate)"))
    parser.add_argument("--fps-out", type=int, default=None,
                        help="output frame rate (default: the source rate; must preserve duration)")
    parser.add_argument("--mjcf", default=str(ARDY_G1_XML),
                        help="MJCF used for the joint-order geometry check")
    parser.add_argument("--sonic-root", default=str(SONIC_ROOT))
    parser.add_argument("--force", action="store_true",
                        help="replace an existing pickle whose bytes differ")
    args = parser.parse_args(argv)
    if bool(args.reference) == bool(args.survey):
        parser.error("pass exactly one of --reference (convert one) or --survey (survey a root)")

    if args.survey:
        try:
            result = survey_deploy_references(
                args.survey, args.out, fps_source=args.fps_source,
                sonic_root=args.sonic_root, mjcf_path=args.mjcf)
        except ConversionError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 1
        print(f"surveyed {result['n_converted']}/{result['n_directories']} deployment "
              f"references under {result['root']}")
        for row in result["assets"]:
            if not row.get("converted"):
                print(f"  {row['name']:<40} REFUSED: {row['refused']}")
                continue
            screen = row["reference_screen"]
            print(f"  {row['name']:<40} {row['n_frames']:>4} fr "
                  f"= {row['duration_frames_over_fps_s']:6.2f} s  "
                  f"longest no-support run {screen['max_unsupported_run_s']:.2f} s  "
                  f"{'FLAGGED' if screen['flagged_by_calibrated_gate'] else 'passes'} the gate")
        print(f"  {result['summary']['n_flagged_by_calibrated_gate']} of "
              f"{result['n_converted']} flagged by the calibrated gate -> {args.out}")
        return 0

    try:
        sidecar = convert_deploy_reference(
            args.reference, args.out, fps_source=args.fps_source, fps_out=args.fps_out,
            motion_name=args.name, sonic_root=args.sonic_root, mjcf_path=args.mjcf,
            force=args.force)
    except ConversionError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    source, output = sidecar["source"], sidecar["output"]
    geometry = sidecar["joint_order_check"]["by_geometry"]
    print(f"converted {source['name']}")
    print(f"  source   {source['n_frames']} frames @ {source['fps']} fps "
          f"= {sidecar['duration']['source_frames_over_fps_s']:.3f} s "
          f"({source['n_joint_columns']} joints, IsaacLab order; "
          f"{source['n_tracked_bodies']} tracked bodies)")
    print(f"  output   {output['n_frames']} frames @ {output['fps']} fps "
          f"= {sidecar['duration']['output_frames_over_fps_s']:.3f} s -> {output['path']}")
    print(f"  keys     {', '.join(output['entry_keys'])}")
    print(f"  joints   name check passed; geometry check worst shared-frame body "
          f"{geometry['worst_shared_frame_error_m']:.2e} m, worst tracked body "
          f"{geometry['worst_tracked_body_error_m']:.2e} m, "
          f"{geometry['discrimination_ratio_vs_best_alternative']:.0f}x better than the best "
          f"alternative ordering")
    print(f"  sidecar  {sidecar['sidecar_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
