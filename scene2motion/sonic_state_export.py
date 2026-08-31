"""Export the states SONIC actually achieved during ``im_eval``.

The stock SONIC callback reports aggregate tracking errors, but it does not persist the robot
states needed to re-run Scene2Motion's geometry checks.  This module supplies a Hydra-compatible
subclass without changing the SONIC checkout.  It deliberately keeps the archive and conversion
code independent of Isaac Lab so saved rollouts can be inspected in the ordinary CPU test
environment.

Archive convention
------------------
``qpos`` has MuJoCo's free-joint layout: local root xyz, root quaternion wxyz, then the 29 G1
joints in :data:`scene2motion.sonic_export.G1_29DOF` order.  Isaac Lab stores root positions in
world coordinates and replicates environments at different origins, so the callback subtracts
``scene.env_origins``.  Isaac Lab's ``root_quat_w`` is already wxyz.  Joint positions are mapped
by name instead of trusting Isaac Lab's interleaved storage order.

Terminated trajectories are trimmed to the last step SONIC counted as alive.  The padded on-disk
array uses NaNs after ``valid_lengths`` so an accidental unsliced consumer fails loudly rather
than treating reset-state padding as achieved motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from scene2motion.sonic_export import G1_29DOF

ARCHIVE_NAME = "achieved_qpos.npz"
ARCHIVE_SCHEMA_VERSION = 1
CALLBACK_TARGET = "scene2motion.sonic_state_export.SonicStateExportCallback"
QPOS_WIDTH = 7 + len(G1_29DOF)


try:  # Available in the Isaac/SONIC subprocess, intentionally absent from CPU-only tests.
    from gear_sonic.trl.callbacks.im_eval_callback import ImEvalCallback as _ImEvalCallback
except ImportError as _sonic_import_error:  # pragma: no cover - exercised only outside SONIC
    _SONIC_IMPORT_ERROR: ImportError | None = _sonic_import_error

    class _ImEvalCallback:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "SonicStateExportCallback must be instantiated inside the SONIC environment"
            ) from _SONIC_IMPORT_ERROR

else:  # pragma: no cover - the real base class requires Isaac/GPU dependencies
    _SONIC_IMPORT_ERROR = None


@dataclass(frozen=True)
class SonicRollout:
    """One achieved trajectory after removing archive padding."""

    motion_key: str
    qpos: np.ndarray
    valid_length: int
    terminated: bool
    progress: float
    motion_id: int


def isaac_state_to_mujoco_qpos(
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_pos: np.ndarray,
    joint_names: Sequence[str],
    env_origins: np.ndarray | None = None,
) -> np.ndarray:
    """Convert batched achieved Isaac states into MuJoCo-compatible G1 qpos.

    Arrays may have any common leading shape.  Only their final dimensions are interpreted.
    Quaternions are normalized defensively; a zero or non-finite quaternion is rejected.
    """

    root = np.asarray(root_pos_w, dtype=np.float32)
    quat = np.asarray(root_quat_wxyz, dtype=np.float32)
    joints = np.asarray(joint_pos, dtype=np.float32)
    names = [str(name) for name in joint_names]

    if root.shape[-1:] != (3,):
        raise ValueError(f"root_pos_w must end in 3 coordinates, got {root.shape}")
    if quat.shape[-1:] != (4,):
        raise ValueError(f"root_quat_wxyz must end in 4 coordinates, got {quat.shape}")
    if joints.ndim == 0 or joints.shape[-1] != len(names):
        raise ValueError(
            f"joint_pos final dimension {joints.shape[-1:] or joints.shape} does not match "
            f"{len(names)} joint names"
        )
    if root.shape[:-1] != quat.shape[:-1] or root.shape[:-1] != joints.shape[:-1]:
        raise ValueError(
            "root_pos_w, root_quat_wxyz, and joint_pos must have identical leading shapes"
        )
    if len(set(names)) != len(names):
        raise ValueError("joint_names contains duplicates; name-based reordering is ambiguous")

    missing = [name for name in G1_29DOF if name not in names]
    if missing:
        raise ValueError(f"Isaac robot is missing {len(missing)} G1 joints: {missing}")
    order = np.asarray([names.index(name) for name in G1_29DOF], dtype=np.int64)

    if env_origins is None:
        local_root = root.copy()
    else:
        origins = np.asarray(env_origins, dtype=np.float32)
        if origins.shape != root.shape:
            try:
                origins = np.broadcast_to(origins, root.shape)
            except ValueError as exc:
                raise ValueError(
                    f"env_origins shape {origins.shape} cannot broadcast to roots {root.shape}"
                ) from exc
        local_root = root - origins

    quat_norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if not np.all(np.isfinite(quat_norm)) or np.any(quat_norm <= 1e-8):
        raise ValueError("root quaternion contains a zero or non-finite norm")
    quat = quat / quat_norm
    return np.concatenate([local_root, quat, joints[..., order]], axis=-1).astype(
        np.float32, copy=False
    )


def _coerce_rollout(rollout: SonicRollout) -> SonicRollout:
    qpos = np.asarray(rollout.qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != QPOS_WIDTH:
        raise ValueError(
            f"rollout {rollout.motion_key!r} qpos must be (T, {QPOS_WIDTH}), got {qpos.shape}"
        )
    valid = int(rollout.valid_length)
    if valid != len(qpos):
        raise ValueError(
            f"rollout {rollout.motion_key!r} valid_length={valid} but has {len(qpos)} rows"
        )
    if not np.all(np.isfinite(qpos)):
        raise ValueError(f"rollout {rollout.motion_key!r} contains non-finite valid qpos")
    return SonicRollout(
        motion_key=str(rollout.motion_key),
        qpos=qpos,
        valid_length=valid,
        terminated=bool(rollout.terminated),
        progress=float(rollout.progress),
        motion_id=int(rollout.motion_id),
    )


def write_sonic_state_archive(
    rollouts: Iterable[SonicRollout],
    path: str | Path,
    *,
    sample_dt_s: float = float("nan"),
) -> Path:
    """Write achieved rollouts in a non-pickle, shape-checked archive."""

    records = sorted((_coerce_rollout(r) for r in rollouts), key=lambda r: r.motion_id)
    keys = [r.motion_key for r in records]
    ids = [r.motion_id for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("motion keys must be unique within an achieved-state archive")
    if len(ids) != len(set(ids)):
        raise ValueError("motion ids must be unique within an achieved-state archive")

    max_len = max((r.valid_length for r in records), default=0)
    padded = np.full((len(records), max_len, QPOS_WIDTH), np.nan, dtype=np.float32)
    for i, record in enumerate(records):
        padded[i, : record.valid_length] = record.qpos

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(ARCHIVE_SCHEMA_VERSION, dtype=np.int16),
        qpos=padded,
        valid_lengths=np.asarray([r.valid_length for r in records], dtype=np.int32),
        terminated=np.asarray([r.terminated for r in records], dtype=np.bool_),
        progress=np.asarray([r.progress for r in records], dtype=np.float32),
        motion_keys=np.asarray(keys, dtype=np.str_),
        motion_ids=np.asarray(ids, dtype=np.int64),
        joint_names=np.asarray(G1_29DOF, dtype=np.str_),
        root_frame=np.asarray("isaac_env_local", dtype=np.str_),
        root_quaternion_order=np.asarray("wxyz", dtype=np.str_),
        sample_dt_s=np.asarray(float(sample_dt_s), dtype=np.float64),
    )
    return path


def _load_archive(path: Path) -> tuple[list[SonicRollout], float]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "qpos",
            "valid_lengths",
            "terminated",
            "progress",
            "motion_keys",
            "motion_ids",
            "joint_names",
            "root_frame",
            "root_quaternion_order",
            "sample_dt_s",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path} is missing achieved-state fields: {missing}")
        version = int(np.asarray(data["schema_version"]).item())
        if version != ARCHIVE_SCHEMA_VERSION:
            raise ValueError(
                f"{path} has schema version {version}; expected {ARCHIVE_SCHEMA_VERSION}"
            )
        if str(np.asarray(data["root_frame"]).item()) != "isaac_env_local":
            raise ValueError(f"{path} does not contain environment-local root positions")
        if str(np.asarray(data["root_quaternion_order"]).item()) != "wxyz":
            raise ValueError(f"{path} does not contain wxyz root quaternions")
        if [str(x) for x in data["joint_names"].tolist()] != G1_29DOF:
            raise ValueError(f"{path} joint order does not match G1_29DOF")

        qpos = np.asarray(data["qpos"], dtype=np.float32)
        lengths = np.asarray(data["valid_lengths"], dtype=np.int64)
        terminated = np.asarray(data["terminated"], dtype=np.bool_)
        progress = np.asarray(data["progress"], dtype=np.float32)
        keys = np.asarray(data["motion_keys"]).astype(str)
        ids = np.asarray(data["motion_ids"], dtype=np.int64)
        sample_dt_s = float(np.asarray(data["sample_dt_s"]).item())

    if qpos.ndim != 3 or qpos.shape[2] != QPOS_WIDTH:
        raise ValueError(f"{path} qpos must be (N, T, {QPOS_WIDTH}), got {qpos.shape}")
    n = len(qpos)
    if any(len(field) != n for field in (lengths, terminated, progress, keys, ids)):
        raise ValueError(f"{path} metadata lengths do not match its {n} qpos trajectories")
    if np.any(lengths < 0) or np.any(lengths > qpos.shape[1]):
        raise ValueError(f"{path} has valid_lengths outside [0, {qpos.shape[1]}]")

    result = []
    for i in range(n):
        valid = int(lengths[i])
        achieved = qpos[i, :valid].copy()
        if not np.all(np.isfinite(achieved)):
            raise ValueError(f"{path} motion {keys[i]!r} contains non-finite valid qpos")
        result.append(
            SonicRollout(
                motion_key=str(keys[i]),
                qpos=achieved,
                valid_length=valid,
                terminated=bool(terminated[i]),
                progress=float(progress[i]),
                motion_id=int(ids[i]),
            )
        )
    return result, sample_dt_s


def load_sonic_state_rollouts(path: str | Path) -> list[SonicRollout]:
    """Load one archive or merge rank shards from an evaluation output directory.

    Passing a directory prefers ``achieved_qpos.npz``.  If it is absent, all
    ``achieved_qpos.rank*.npz`` shards are merged and sorted by global motion id.
    """

    source = Path(path)
    if source.is_dir():
        canonical = source / ARCHIVE_NAME
        paths = [canonical] if canonical.exists() else sorted(source.glob("achieved_qpos.rank*.npz"))
        if not paths:
            raise FileNotFoundError(f"no achieved-state archive found in {source}")
    else:
        paths = [source]
    records: list[SonicRollout] = []
    sample_dts: list[float] = []
    for archive in paths:
        loaded, sample_dt_s = _load_archive(archive)
        records.extend(loaded)
        if np.isfinite(sample_dt_s):
            sample_dts.append(sample_dt_s)
    if sample_dts and not np.allclose(sample_dts, sample_dts[0], rtol=0.0, atol=1e-12):
        raise ValueError(f"rank shards disagree on sample_dt_s: {sample_dts}")
    keys = [r.motion_key for r in records]
    ids = [r.motion_id for r in records]
    if len(keys) != len(set(keys)) or len(ids) != len(set(ids)):
        raise ValueError("merged achieved-state archives contain duplicate motion keys or ids")
    return sorted(records, key=lambda r: r.motion_id)


def sonic_state_sample_dt(path: str | Path) -> float:
    """Return the achieved-state sampling period, requiring shard agreement.

    Geometry alone is frame-rate agnostic, but contact support uses physical foot speed.  A
    caller must therefore not silently reuse the reference-motion FPS when SONIC exported
    achieved states at the policy control rate.
    """
    source = Path(path)
    if source.is_dir():
        canonical = source / ARCHIVE_NAME
        paths = [canonical] if canonical.exists() else sorted(source.glob("achieved_qpos.rank*.npz"))
        if not paths:
            raise FileNotFoundError(f"no achieved-state archive found in {source}")
    else:
        paths = [source]
    values = []
    for archive in paths:
        with np.load(archive, allow_pickle=False) as data:
            if "sample_dt_s" not in data.files:
                raise ValueError(f"{archive} is missing sample_dt_s")
            values.append(float(np.asarray(data["sample_dt_s"]).item()))
    if any(not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f"achieved-state archive has invalid sample_dt_s: {values}")
    if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
        raise ValueError(f"rank shards disagree on sample_dt_s: {values}")
    return values[0]


def achieved_qpos_by_key(path: str | Path) -> dict[str, np.ndarray]:
    """Post-process an archive into trimmed ``{motion_key: qpos}`` trajectories."""

    return {record.motion_key: record.qpos for record in load_sonic_state_rollouts(path)}


def sonic_state_hydra_overrides(state_filename: str = ARCHIVE_NAME) -> list[str]:
    """Hydra arguments that replace SONIC's ``im_eval`` target with this subclass."""

    filename = Path(state_filename)
    if (
        filename.name != str(filename)
        or filename.name in {"", ".", ".."}
        or filename.suffix != ".npz"
    ):
        raise ValueError("state_filename must be a simple filename ending in .npz")
    return [
        f"++callbacks.im_eval._target_={CALLBACK_TARGET}",
        f"++callbacks.im_eval.state_filename={filename.name}",
    ]


def sonic_state_subprocess_env(
    env: Mapping[str, str] | None = None,
    *,
    project_root: str | Path | None = None,
) -> dict[str, str]:
    """Return an environment in which SONIC can import the local callback target."""

    result = dict(os.environ if env is None else env)
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    existing = [p for p in result.get("PYTHONPATH", "").split(os.pathsep) if p]
    root_s = str(root)
    result["PYTHONPATH"] = os.pathsep.join([root_s] + [p for p in existing if p != root_s])
    return result


class SonicStateExportCallback(_ImEvalCallback):
    """SONIC ``ImEvalCallback`` that also saves achieved MuJoCo-compatible qpos."""

    def __init__(self, *args, state_filename: str = ARCHIVE_NAME, **kwargs):
        super().__init__(*args, **kwargs)
        name = Path(state_filename)
        if name.name != str(name) or name.name in {"", ".", ".."} or name.suffix != ".npz":
            raise ValueError("state_filename must be a simple filename ending in .npz")
        self.state_filename = name.name
        self._achieved_records: list[SonicRollout] = []
        self._batch_qpos: list[np.ndarray] = []
        self._batch_motion_ids: np.ndarray | None = None
        self._batch_reference_lengths: np.ndarray | None = None

    def _pre_evaluate_policy(self, reset_env=True):
        super()._pre_evaluate_policy(reset_env=reset_env)
        self._achieved_records = []
        self._reset_achieved_batch()

    def _reset_achieved_batch(self) -> None:
        self._batch_qpos = []
        self._batch_motion_ids = None
        self._batch_reference_lengths = None

    @staticmethod
    def _as_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value)

    def _snapshot_achieved_qpos(self) -> np.ndarray:
        manager_env = getattr(self.env, "env", self.env)
        scene = manager_env.scene
        robot = scene["robot"]
        return isaac_state_to_mujoco_qpos(
            self._as_numpy(robot.data.root_pos_w),
            self._as_numpy(robot.data.root_quat_w),
            self._as_numpy(robot.data.joint_pos),
            robot.joint_names,
            self._as_numpy(scene.env_origins),
        )

    def _begin_achieved_batch(self) -> None:
        local_ids = self._as_numpy(self.env.motion_ids).astype(np.int64, copy=False)
        self._batch_motion_ids = int(self.env.start_idx) + local_ids
        lengths = self.env._motion_lib.get_motion_num_steps(self.env.motion_ids)
        self._batch_reference_lengths = self._as_numpy(lengths).astype(np.int64, copy=False)

    def _finish_achieved_batch(self) -> None:
        if self._batch_motion_ids is None or self._batch_reference_lengths is None:
            raise RuntimeError("achieved-state batch ended before its motion ids were captured")
        if not self._batch_qpos:
            raise RuntimeError("achieved-state batch ended without a robot-state sample")
        states = np.stack(self._batch_qpos, axis=0)  # (steps, envs, 36)
        terminated = np.asarray(self.terminate_memory[-1], dtype=np.bool_)
        progress_raw = np.asarray(self.progress_memory[-1], dtype=np.float64)
        if states.shape[1] != len(self._batch_motion_ids):
            raise RuntimeError(
                f"captured {states.shape[1]} environments for {len(self._batch_motion_ids)} ids"
            )

        num_unique = int(self.env._motion_lib._num_unique_motions)
        keys = self.env._motion_lib._motion_data_keys
        for env_idx, motion_id_raw in enumerate(self._batch_motion_ids):
            motion_id = int(motion_id_raw)
            if motion_id < 0 or motion_id >= num_unique:
                continue  # padded environments in the final batch
            ref_len = int(self._batch_reference_lengths[env_idx])
            if terminated[env_idx]:
                valid = int(np.rint(progress_raw[env_idx] * ref_len))
                progress = float(np.clip(progress_raw[env_idx], 0.0, 1.0))
            else:
                valid = ref_len
                progress = 1.0
            valid = min(max(valid, 0), states.shape[0], ref_len)
            key = keys[motion_id]
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            self._achieved_records.append(
                SonicRollout(
                    motion_key=str(key),
                    qpos=states[:valid, env_idx].copy(),
                    valid_length=valid,
                    terminated=bool(terminated[env_idx]),
                    progress=progress,
                    motion_id=motion_id,
                )
            )
        self._reset_achieved_batch()

    def _post_eval_env_step(self, actor_state):
        if self._batch_motion_ids is None:
            self._begin_achieved_batch()
        # Save the state immediately after this physics step.  If this step terminates an env,
        # the parent's progress counter excludes it and _finish_achieved_batch trims it away.
        self._batch_qpos.append(self._snapshot_achieved_qpos())
        previous_loop = int(self.env_eval_loop_idx)
        result = super()._post_eval_env_step(actor_state)
        if int(self.env_eval_loop_idx) != previous_loop:
            self._finish_achieved_batch()
        return result

    def _post_evaluate_policy(self, eval_res):
        metrics = super()._post_evaluate_policy(eval_res)
        if self.output_dir is None:
            raise RuntimeError("SonicStateExportCallback requires eval_output_dir/output_dir")
        num_processes = int(getattr(self.accelerator, "num_processes", 1))
        process_index = int(getattr(self.accelerator, "process_index", 0))
        if num_processes == 1:
            filename = self.state_filename
        else:
            stem = Path(self.state_filename).stem
            filename = f"{stem}.rank{process_index}.npz"
        manager_env = getattr(self.env, "env", self.env)
        sample_dt_s = float(getattr(manager_env, "step_dt", float("nan")))
        destination = Path(self.output_dir) / filename
        write_sonic_state_archive(
            self._achieved_records, destination, sample_dt_s=sample_dt_s
        )
        print(f"Saved achieved qpos to {destination}")
        return metrics
