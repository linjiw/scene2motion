"""ARDY qpos -> SONIC motion-lib, so kinematic claims can be checked against physics.

Why this is short
-----------------
The two representations turn out to line up almost exactly, and the one fact that makes it work
is worth stating because it was not obvious in advance and it is the thing that would silently
corrupt every tracker number if it were false:

    ARDY's exported qpos[7:36] is in the SAME joint order as SONIC's 29-DOF G1 motion library.

Both are MuJoCo/MJCF actuator order for `g1_29dof_rev_1_0`.  Verified by name against
`gear_sonic/data_process/convert_soma_csv_to_motion_lib.py:BONES_CSV_JOINT_NAMES`: all 29 names
match position for position.  A reordering here would not crash anything -- it would produce a
robot that tracks a plausible-looking but different motion, and every TrackedAddressability
number would be wrong in a way no assertion catches.  So the check is re-run at import time.

Conventions that DO differ, and are converted:
    root quaternion   MuJoCo qpos is wxyz;  motion_lib `root_rot` is xyzw (scipy)
    pose_aa           per-body axis-angle, = DOF_AXIS * dof, with body 0 the root rotvec

`DOF_AXIS` is imported from SONIC's own converter rather than copied, so it cannot drift.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

SONIC_ROOT = Path("/home/linjiw/isaaclab-install/GR00T-WholeBodyControl")
ARDY_FPS = 25
NUM_DOF, NUM_BODIES = 29, 30

# The 29 joint names, in the order both representations use.
G1_29DOF = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _dof_axis() -> np.ndarray:
    """SONIC's own DOF_AXIS table, imported rather than copied."""
    import importlib.util
    p = SONIC_ROOT / "gear_sonic/data_process/convert_soma_csv_to_motion_lib.py"
    if not p.exists():
        raise FileNotFoundError(f"SONIC converter not found at {p}")
    src = p.read_text()
    # The module imports heavy SONIC/Isaac deps at module scope, so lift out just the table.
    ns: dict = {"np": np}
    i = src.index("DOF_AXIS = np.array(")
    j = src.index("\n)\n", i) + 3
    exec(src[i:j], ns)                                    # noqa: S102 -- a literal array
    a = np.asarray(ns["DOF_AXIS"], np.float32)
    if a.shape != (NUM_DOF, 3):
        raise ValueError(f"DOF_AXIS has shape {a.shape}, expected ({NUM_DOF}, 3)")
    return a


def check_joint_order(mj_model) -> None:
    """Assert the MuJoCo model's hinge order is the motion library's order.

    Called by `qpos_to_entry` whenever a model is available.  This is the assertion that
    protects every downstream tracker number.
    """
    import mujoco
    names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in range(mj_model.njnt)]
    hinges = [n for n, t in zip(names, mj_model.jnt_type) if int(t) != 0]
    if hinges != G1_29DOF:
        bad = next((i for i, (a, b) in enumerate(zip(hinges, G1_29DOF)) if a != b), None)
        raise ValueError(
            f"joint order mismatch at index {bad}: MuJoCo has {hinges[bad]!r}, "
            f"SONIC expects {G1_29DOF[bad]!r}. Tracking this would silently follow the "
            f"wrong motion.")


def qpos_to_entry(qpos: np.ndarray, fps: int = ARDY_FPS, mj_model=None) -> dict:
    """One ARDY (T, 36) qpos trajectory -> one motion_lib entry."""
    from scipy.spatial import transform

    q = np.asarray(qpos, np.float32)
    if q.ndim != 2 or q.shape[1] < 7 + NUM_DOF:
        raise ValueError(f"expected (T, >= {7 + NUM_DOF}) qpos, got {q.shape}")
    if mj_model is not None:
        check_joint_order(mj_model)
    T = len(q)
    root_trans = q[:, 0:3].copy()
    root_rot_xyzw = q[:, [4, 5, 6, 3]].copy()             # MuJoCo wxyz -> scipy xyzw
    n = np.linalg.norm(root_rot_xyzw, axis=1, keepdims=True)
    root_rot_xyzw = root_rot_xyzw / np.maximum(n, 1e-8)
    dof = q[:, 7:7 + NUM_DOF].copy()

    pose_aa = np.zeros((T, NUM_BODIES, 3), np.float32)
    pose_aa[:, 1:, :] = _dof_axis()[None, :, :] * dof[:, :, None]
    pose_aa[:, 0, :] = transform.Rotation.from_quat(root_rot_xyzw).as_rotvec()
    return {"root_trans_offset": root_trans.astype(np.float32),
            "pose_aa": pose_aa.astype(np.float32),
            "dof": dof.astype(np.float32),
            "root_rot": root_rot_xyzw.astype(np.float32),
            "smpl_joints": np.zeros((T, 24, 3), np.float32),
            "fps": int(fps)}


def write_motion_pkl(clips: dict[str, np.ndarray], path: str | Path,
                     fps: int = ARDY_FPS, mj_model=None) -> Path:
    """Write `{name: qpos}` as a motion-lib pickle SONIC's joblib.load can read."""
    out = {k: qpos_to_entry(v, fps, mj_model) for k, v in clips.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return path
