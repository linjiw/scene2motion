"""EXP-030: the archived stepping pool tracked with the obstacle actually in the scene.

Every executed result in this project so far -- EXP-1B, EXP-1C, EXP-011/012/014, EXP-022A and
the pending EXP-024/EXP-028 -- tracked references with the **obstacle absent from the physics
scene** and then replayed the achieved states against our MuJoCo collision model.  EXP-030 puts
the box in Isaac and measures what changes (protocol:
``docs/ramp-exp030-obstacle-present-stepping-protocol.md``, preregistered 2026-09-03):

* **Q1, the proxy question.** Does the obstacle-absent replay predict the obstacle-present
  outcome, per reference?
* **Q2, the endpoint the project has never measured.** With the obstacle present, does any
  prompt-elicited reference complete **local traversal** -- through the corridor, past the box,
  to the goal, upright?

Three paired arms over the same 64 archived EXP-021 references, in EXP-022A's chunk order, at
physics seed 0 under the release evaluator: ``absent`` (no box), ``present_05`` (5 cm box) and
``present_20`` (20 cm box).  Six launches of 32 environments.  No new ARDY samples.

The obstacle plumbing is the one ``experiments/probe_obstacle_present.py`` proved (REPORT §47):
``add_table=true`` spawns a collidable cuboid whose ``table_size`` is the **full x/y/z extents**
and whose position is its **centre**, but the pose is rewritten per environment on every reset,
so the box has to be carried as per-motion ``table_pos`` / ``table_quat`` inside the motion
pickle.  The width comes from ``stepover_eval.step_scene`` so the physics box is the geometry
our collision model has always scored.

**Tracker baseline, and which checkout.** The ``add_table`` fix touches a file inside the
core-source manifest that EXP-022A/EXP-024/EXP-028 bind equal to ``44e98c45...``, so it was
reverted on the legacy checkout (``350cae1``) and lives on a dedicated worktree instead
(CLAUDE.md, "Two tracker checkouts"; protocol section 2's 2026-09-03 amendment).  EXP-030 runs
**every** arm -- the ``absent`` control included -- on ``/home/linjiw/lucid/GR00T-WBC-exp029``
(branch ``exp029-obstacle-present``, pinned at ``7c63c53``), never on the legacy checkout, which
it refuses outright.  It also requires the checkout's HEAD to contain the fix commit and the
source itself to carry the fix (:func:`tracker_fix_report` reads the file and checks that the
``table_to_robot_contact_sensor`` assignment lies inside the ``add_object`` branch that binds
``right_hand_wrist_links``, so a mislabelled commit cannot pass).  Unlike EXP-028 it does **not**
assert equality with EXP-022A's core manifest: it records its own manifest, the per-file diff
against the legacy checkout and both values, and the ``absent`` arm measures whether the
difference is inert.  The released checkpoint bundle is gitignored and therefore absent from the
worktree; it is passed as an absolute ``+checkpoint=`` path and content-hashed.

Stages (``--stage``): ``launch`` (the six SONIC launches), ``analyze`` (CPU only; scores every
rollout with :mod:`scene2motion.traversal_eval` and evaluates P1/P2/P3), ``all``.  Every stage is
resumable and fails closed; ``--dry-run`` prints the plan, the six command lines and the host
gate and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import analyze_exp021_exact_addressability as addr  # noqa: E402
from experiments import calibrate_ramp_route_phase as cal  # noqa: E402
from experiments import exp022_exact_tracking_bridge as exp022  # noqa: E402
from experiments import exp028_termination_free_rollouts as e28  # noqa: E402
from experiments import exp1b_execution_clearance as exp1b  # noqa: E402
from scene2motion import host_gate  # noqa: E402
from scene2motion import traversal_eval as te  # noqa: E402
from scene2motion.robot import ARDY_G1_XML, BODY_MARGIN  # noqa: E402
from scene2motion.sonic_export import SONIC_ROOT as EXPORT_SONIC_ROOT  # noqa: E402
from scene2motion.sonic_export import write_motion_pkl  # noqa: E402
from scene2motion.sonic_state_export import (  # noqa: E402
    ARCHIVE_SCHEMA_VERSION,
    sonic_state_hydra_overrides,
)
from scene2motion.stepover_eval import step_scene  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE_OUT = exp022.SOURCE_OUT
EXP022A_OUT = e28.EXP022A_OUT
DEFAULT_OUT = ROOT / "outputs/exp030_obstacle_present"
PROTOCOL_PATH = ROOT / "docs/ramp-exp030-obstacle-present-stepping-protocol.md"

SCHEMA_VERSION = "exp030-obstacle-present-v1"
FAILURE_SCHEMA_VERSION = "exp030-obstacle-present-failure-v1"
PROCESS_RESULT_SCHEMA = "exp030-sonic-process-result-v1"
STAGES = ("launch", "analyze", "all")


def evaluator_version() -> int:
    """Which ``traversal_eval`` scored this run, recorded rather than assumed.

    The outcome evaluator gained a version marker only in a later revision.  The **launch** stage
    is independent of it — the six SONIC rollouts are the same whichever evaluator later reads
    them — so the driver must not refuse to launch merely because the marker is absent.  The
    receipt therefore records the version actually used, defaulting to 1, and the analysis stage
    is free to be re-run under a newer evaluator as a separate, versioned analysis.
    """
    return int(getattr(te, "EVALUATOR_VERSION", 1))

# ------------------------------------------------------------------ locked external identities
EXPECTED_CHECKPOINT_SHA256 = e28.EXPECTED_CHECKPOINT_SHA256
EXPECTED_G1_XML_SHA256 = e28.EXPECTED_G1_XML_SHA256
#: EXP-022A's core-source manifest.  **Recorded, never asserted**: EXP-030 needs the fork fix and
#: therefore declares its own baseline; the receipt shows both so the diff is visible.
EXP022A_CORE_MANIFEST_SHA256 = e28.EXPECTED_CORE_MANIFEST_SHA256
CORE_SONIC_FILES = exp022.CORE_SONIC_FILES
EVALUATOR_SONIC_FILES = e28.EVALUATOR_SONIC_FILES
#: The file whose ``add_table``-without-``add_object`` fix this campaign requires.
ADD_TABLE_FIX_FILE = "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py"

# --------------------------------------------------------------------- the two tracker checkouts
#
# CLAUDE.md, "Two tracker checkouts", and protocol §2's 2026-09-03 amendment: the fix touches a
# file inside the core-source manifest that EXP-022A/024/028 bind equal to ``44e98c45…``, so it
# was reverted on the legacy checkout and lives on a dedicated worktree instead.  EXP-030 runs
# **every** arm, the ``absent`` control included, on the patched worktree.
SONIC_EXP029_ROOT = Path("/home/linjiw/lucid/GR00T-WBC-exp029")
SONIC_EXP029_BRANCH = "exp029-obstacle-present"
#: SONIC ``7c63c53``: "fix(manager_env): let add_table work without add_object".  HEAD must
#: contain it, and the source check below must independently confirm the fix is really there.
ADD_TABLE_FIX_COMMIT = "7c63c539a17008f5efb1e768034c0fb434ae1f65"
#: The unpatched checkout EXP-022A/EXP-024/EXP-028 keep.  Refused as an EXP-030 execution root.
LEGACY_SONIC_ROOT = Path(exp1b.SONIC)
#: The released checkpoint bundle (``last.pt`` + its ``config.yaml``) is a downloaded artifact,
#: gitignored in both checkouts, so a worktree does not carry it.  It is resolved by absolute
#: path and its source is recorded; the file content is what the receipt binds either way.
RELEASE_BUNDLE_PREFIX = "sonic_release/"
RELEASE_CHECKPOINT_RELATIVE = "sonic_release/last.pt"

SOURCE_FILES = (
    "env.sh",
    "experiments/calibrate_ramp_route_phase.py",
    "experiments/exp022_exact_tracking_bridge.py",
    "experiments/exp028_termination_free_rollouts.py",
    "experiments/exp030_obstacle_present.py",
    "experiments/exp1b_execution_clearance.py",
    "scene2motion/host_gate.py",
    "scene2motion/robot.py",
    "scene2motion/scenes.py",
    "scene2motion/sonic_export.py",
    "scene2motion/sonic_state_export.py",
    "scene2motion/stepover_eval.py",
    "scene2motion/traversal_eval.py",
)

# ------------------------------------------------------------------------------ locked design
POOL_SEEDS = exp022.POOL_SEEDS
N_FRAMES = exp022.N_FRAMES
FPS = exp022.FPS
PHYSICS_SEED = 0
SAMPLE_DT_S = exp022.EXPECTED_SAMPLE_DT_S

OBSTACLE_X_M = 1.2
OBSTACLE_DEPTH_M = exp022.OBSTACLE_DEPTH_M
#: ``step_scene`` makes the box corridor-spanning; the physics cuboid uses the same width so the
#: spawned obstacle is the geometry the collision model has always scored.
OBSTACLE_HALF_WIDTH_M = exp022.OBSTACLE_HALF_WIDTH_M
OBSTACLE_WIDTH_M = 2.0 * OBSTACLE_HALF_WIDTH_M
IDENTITY_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)

ROUTE_START_XY = (0.0, 0.0)
ROUTE_GOAL_XY = (float(cal.PILOT_ROUTE_LENGTH_M), 0.0)
GOAL_TOLERANCE_M = 0.5
#: The route is 7.2 m and the checkout default env spacing is 2.5 m; 12 m keeps neighbours apart.
ENV_SPACING_M = 12.0
#: The checkout default is 10 s against ~8.3 s references.
EPISODE_LENGTH_S = 20.0

ARMS: tuple[dict[str, Any], ...] = (
    {"arm": "absent", "box_height_m": None,
     "why": "replicates EXP-022A's configuration on this tracker build; the control for Q1 "
            "and the check that the fork fix is inert"},
    {"arm": "present_05", "box_height_m": 0.05,
     "why": "the project's staged endpoint height; the primary arm"},
    {"arm": "present_20", "box_height_m": 0.20,
     "why": "a graded, harder obstacle; separates 'the box is too small to matter' from "
            "'the robot cannot pass any box'"},
)
ARM_NAMES = tuple(arm["arm"] for arm in ARMS)
PRESENT_ARMS = tuple(arm["arm"] for arm in ARMS if arm["box_height_m"] is not None)
PROXY_ARM = "present_05"
PROXY_HEIGHT_M = 0.05

# --------------------------------------------------------------- preregistered predictions (§5)
P1_RULE = ("P1 (control): the `absent` arm reproduces EXP-022A -- >= 58/64 references agree on "
           "the termination flag")
P1_MIN_TERMINATION_AGREEMENT = 58
P2_RULE = ("P2 (completion): local traversal completion is 0/64 in both present arms; any "
           "completion is the project's first measured local traversal and is reported "
           "prominently, with its reference identified")
P2_PREDICTED_COMPLETIONS = 0
P3_RULE = ("P3 (the proxy): the replay-inferred and physics-measured classes for present_05 "
           "agree on >= 80 % of references (kappa >= 0.6)")
P3_MIN_AGREEMENT = 0.80
P3_MIN_KAPPA = 0.6
KAPPA_BOOTSTRAP_RESAMPLES = 2000
KAPPA_BOOTSTRAP_SEED = 30

RELEASE_TERMINATION_TERMS = tuple(sorted([*e28.EXPECTED_TRACKING_TERMS, e28.TIME_OUT_TERM]))


class CampaignAbort(exp022.BridgeAbort):
    """Fail-closed stop after any available evidence has been made durable."""


class CampaignPaused(CampaignAbort):
    """A refusal that leaves the campaign resumable: host gate or stage lock, never evidence."""


_sha256 = exp022._sha256
_read_jsonl = exp022._read_jsonl
_git_state = exp022._git_state
_file_hashes = exp022._file_hashes
validate_project_recheck = exp022.validate_project_recheck
hydra_overrides_of = e28.hydra_overrides_of
parse_log_termination_terms = e28.parse_log_termination_terms


def build_sonic_command(pkl: Path, eval_dir: Path, num_envs: int, physics_seed: int,
                        extra_overrides: Sequence[str] = (), *,
                        checkpoint: str | Path) -> list[str]:
    """EXP-022A's launcher (``exp1b_execution_clearance.run_sonic``) with an explicit checkpoint.

    Identical to EXP-028's command construction except that the checkpoint is passed in rather
    than taken from the legacy checkout: EXP-030 executes on the patched worktree, which does
    not carry the gitignored release bundle.
    """
    pkl, eval_dir = Path(pkl).resolve(), Path(eval_dir).resolve()
    return [str(exp1b.SONIC_PY), "-u", "-m", "gear_sonic.eval_agent_trl",
            f"+checkpoint={Path(checkpoint).resolve()}", "+headless=True",
            "++eval_callbacks=im_eval", "++run_eval_loop=False", f"++num_envs={int(num_envs)}",
            f"++eval_output_dir={eval_dir}",
            f"++seed={int(physics_seed)}",
            "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False",
            "+manager_env/terminations=tracking/eval",
            f"+manager_env.commands.motion.motion_lib_cfg.motion_file={pkl}",
            f"+log_keys={pkl.stem}",
            *sonic_state_hydra_overrides(),
            *[str(item) for item in extra_overrides]]


def sonic_subprocess_env(sonic_root: str | Path) -> dict[str, str]:
    """EXP-1B's SONIC environment with the execution checkout first on ``PYTHONPATH``.

    ``python -m`` already puts the working directory first, so the worktree's ``gear_sonic``
    wins; this makes that explicit and survivable if the launcher's cwd handling ever changes.
    """
    env = dict(exp1b.sonic_env())
    root = str(Path(sonic_root).resolve())
    existing = [item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item]
    env["PYTHONPATH"] = os.pathsep.join([root] + [item for item in existing if item != root])
    return env


def sonic_launcher(sonic_root: str | Path, checkpoint: str | Path
                   ) -> Callable[..., tuple[int, str]]:
    """A launcher bound to one checkout: the subprocess runs with that checkout as its cwd."""
    root = Path(sonic_root).resolve()

    def launch(pkl: Path, eval_dir: Path, num_envs: int, physics_seed: int, timeout_s: int,
               extra_overrides: Sequence[str] = ()) -> tuple[int, str]:
        command = build_sonic_command(pkl, eval_dir, num_envs, physics_seed, extra_overrides,
                                      checkpoint=checkpoint)
        print("  " + " ".join(command[:4]) + f" ... (cwd={root})", flush=True)
        proc = subprocess.run(command, cwd=root, capture_output=True, text=True,
                              timeout=timeout_s, env=sonic_subprocess_env(root),
                              stdin=subprocess.DEVNULL)
        return proc.returncode, (proc.stdout + "\n" + proc.stderr)

    return launch


def _jsonable(value: Any) -> Any:
    """JSON-safe copy: numpy scalars become Python numbers, non-finite floats become ``None``.

    ``cal._write_json`` writes with ``allow_nan=False``; an empty collision report legitimately
    carries ``inf`` minimum clearance (no obstacle in the scene), which must not abort a ledger
    write nor become the string ``Infinity`` in an artifact.
    """
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


# ------------------------------------------------------------------------------ tracker fix

_ADD_OBJECT_IF = re.compile(r"^(\s*)if\s+config\.get\(\s*[\"']add_object[\"']")
_WRIST_BIND = re.compile(r"^(\s*)right_hand_wrist_links\s*=")
_TABLE_SENSOR = re.compile(r"^(\s*)self\.table_to_robot_contact_sensor\s*=")

FIX_DETECTION = (
    "the `self.table_to_robot_contact_sensor` assignment must be indented deeper than the "
    "`if config.get(\"add_object\", ...)` line that binds `right_hand_wrist_links`, with the "
    "binding before it and no line between them dedenting out of that branch; otherwise "
    "`add_table=true` without `add_object=true` raises UnboundLocalError while the env cfg is "
    "built, before the simulation starts (SONIC 7c63c53, reverted in 350cae1)"
)


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def tracker_fix_report(source: str, *, path: str | None = None,
                       sha256: str | None = None) -> dict[str, Any]:
    """Static check that ``add_table`` works without ``add_object`` in this tracker source."""
    lines = source.splitlines()
    sensors = [(i, _indent_of(line)) for i, line in enumerate(lines) if _TABLE_SENSOR.match(line)]
    binds = [(i, _indent_of(line)) for i, line in enumerate(lines) if _WRIST_BIND.match(line)]
    branches = [(i, _indent_of(line)) for i, line in enumerate(lines) if _ADD_OBJECT_IF.match(line)]
    report: dict[str, Any] = {
        "file": path, "sha256": sha256, "detection": FIX_DETECTION,
        "n_table_sensor_assignments": len(sensors),
        "n_right_hand_wrist_links_bindings": len(binds),
        "n_add_object_branches": len(branches),
        "sensor_line": None, "sensor_indent": None,
        "binding_line": None, "binding_indent": None,
        "add_object_branch_line": None, "add_object_branch_indent": None,
        "binding_precedes_use": False, "sensor_inside_add_object_branch": False,
        "no_dedent_between_binding_and_use": False,
        "unbound_local_risk": True, "fix_present": False, "problems": [],
    }
    problems: list[str] = report["problems"]
    if len(sensors) != 1:
        problems.append(f"expected exactly one table_to_robot_contact_sensor assignment, "
                        f"found {len(sensors)}")
        return report
    sensor_index, sensor_indent = sensors[0]
    report.update({"sensor_line": sensor_index + 1, "sensor_indent": sensor_indent})
    earlier_binds = [item for item in binds if item[0] < sensor_index]
    if not earlier_binds:
        problems.append("right_hand_wrist_links is never bound before the sensor uses it")
        return report
    bind_index, bind_indent = earlier_binds[-1]
    report.update({"binding_line": bind_index + 1, "binding_indent": bind_indent,
                   "binding_precedes_use": True})
    earlier_branches = [item for item in branches if item[0] < bind_index]
    if not earlier_branches:
        problems.append("the binding is not inside an `if config.get(\"add_object\"...)` branch")
        return report
    branch_index, branch_indent = earlier_branches[-1]
    report.update({"add_object_branch_line": branch_index + 1,
                   "add_object_branch_indent": branch_indent})
    inside = bool(sensor_indent > branch_indent and sensor_indent >= bind_indent)
    dedents = [i + 1 for i, line in enumerate(lines[bind_index + 1:sensor_index],
                                              start=bind_index + 1)
               if line.strip() and not line.lstrip().startswith("#")
               and _indent_of(line) <= branch_indent]
    report.update({
        "sensor_inside_add_object_branch": inside,
        "no_dedent_between_binding_and_use": not dedents,
        "dedented_lines_between_binding_and_use": dedents,
    })
    if not inside:
        problems.append(
            f"the sensor assignment at line {sensor_index + 1} (indent {sensor_indent}) is not "
            f"inside the add_object branch at line {branch_index + 1} (indent {branch_indent}) "
            f"that binds right_hand_wrist_links at line {bind_index + 1}")
    if dedents:
        problems.append(f"lines {dedents} leave the add_object branch between the binding and "
                        "the sensor assignment")
    report["unbound_local_risk"] = bool(problems)
    report["fix_present"] = not problems
    return report


def read_tracker_fix(sonic_root: str | Path = SONIC_EXP029_ROOT,
                     relative_path: str = ADD_TABLE_FIX_FILE) -> dict[str, Any]:
    path = Path(sonic_root) / relative_path
    if not path.is_file():
        raise ValueError(f"tracker source is missing: {path}")
    return tracker_fix_report(path.read_text(), path=relative_path, sha256=_sha256(path))


def require_tracker_fix(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """EXP-030 refuses to launch unless the ``add_table``-without-``add_object`` fix is present."""
    if not report.get("fix_present"):
        raise CampaignAbort(
            "EXP-030 requires the tracker fix that lets add_table work without add_object "
            f"({report.get('file')}): " + "; ".join(report.get("problems") or ["not detected"])
            + f". Run against the patched worktree {SONIC_EXP029_ROOT} (branch "
            f"{SONIC_EXP029_BRANCH}, pinned at {ADD_TABLE_FIX_COMMIT[:7]}).")
    return report


def require_execution_root(sonic_root: str | Path) -> Path:
    """The obstacle-present campaign must never run on the legacy, unpatched checkout.

    The legacy checkout reverted the fix (``350cae1``) so EXP-022A/024/028 keep their pinned
    manifest; launching EXP-030 there raises ``UnboundLocalError`` before the simulation starts.
    """
    root = Path(sonic_root).resolve()
    if root == LEGACY_SONIC_ROOT.resolve():
        raise CampaignAbort(
            f"EXP-030 refuses the legacy tracker checkout {root}: its add_table fix is reverted "
            f"(350cae1) so EXP-022A/EXP-024/EXP-028 keep manifest {EXP022A_CORE_MANIFEST_SHA256} "
            f"-- run every arm on the patched worktree {SONIC_EXP029_ROOT} (branch "
            f"{SONIC_EXP029_BRANCH})")
    if not root.is_dir():
        raise CampaignAbort(f"the EXP-030 tracker worktree is missing: {root}")
    return root


def _commit_contains(root: Path, commit: str) -> bool:
    """Is ``commit`` an ancestor of (or equal to) this checkout's HEAD?"""
    try:
        completed = subprocess.run(["git", "merge-base", "--is-ancestor", str(commit), "HEAD"],
                                   cwd=root, capture_output=True, text=True, timeout=60,
                                   stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"could not test the tracker checkout for {commit}: {exc}") from exc
    if completed.returncode not in (0, 1):
        raise ValueError(f"git could not resolve {commit} in {root}: "
                         f"{completed.stderr.strip()[:200]}")
    return completed.returncode == 0


def _git_branch(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_release_bundle(sonic_root: str | Path) -> dict[str, Any]:
    """Locate the released checkpoint bundle: in the execution root, else the legacy checkout.

    ``sonic_release/`` is gitignored, so the worktree does not carry it.  The bundle is the
    vendor's downloaded artifact rather than checkout code, so resolving it by absolute path
    changes nothing about which tracker sources run -- and both files are content-hashed here.
    """
    for candidate, source in ((Path(sonic_root), "execution_root"),
                              (LEGACY_SONIC_ROOT, "legacy_checkout_release_bundle")):
        checkpoint = Path(candidate) / RELEASE_CHECKPOINT_RELATIVE
        config = Path(candidate) / "sonic_release/config.yaml"
        if checkpoint.is_file() and config.is_file():
            return {
                "root": str(Path(candidate).resolve()),
                "source": source,
                "checkpoint_path": str(checkpoint.resolve()),
                "config_path": str(config.resolve()),
                "note": ("the release bundle is gitignored in both checkouts; it is passed to "
                         "SONIC as an absolute +checkpoint= path and its config.yaml is the "
                         "one the eval loads from the checkpoint's own directory"),
            }
    raise ValueError(f"no SONIC release bundle (sonic_release/last.pt + config.yaml) under "
                     f"{Path(sonic_root)} or {LEGACY_SONIC_ROOT}")


def core_source_hashes(sonic_root: Path, bundle: Mapping[str, Any]) -> tuple[dict[str, str],
                                                                            dict[str, str]]:
    """Hash the core manifest, taking the release-bundle entries from where the bundle lives."""
    hashes: dict[str, str] = {}
    provenance: dict[str, str] = {}
    for name in CORE_SONIC_FILES:
        base = Path(bundle["root"]) if name.startswith(RELEASE_BUNDLE_PREFIX) else Path(sonic_root)
        digest = _sha256(base / name)
        if digest is None:
            raise ValueError(f"required tracker source is missing: {base / name}")
        hashes[name] = digest
        provenance[name] = str((base / name).resolve())
    return hashes, provenance


# ------------------------------------------------------------------------------ identities

def protocol_identity(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    digest = _sha256(path)
    if digest is None:
        raise ValueError(f"EXP-030 protocol is missing: {path}")
    match = re.search(r"^\*\*Status:\s*(\w+)", path.read_text(), flags=re.MULTILINE)
    return {"path": str(path.resolve()), "sha256": digest,
            "status": match.group(1).lower() if match else None}


def project_identity(repo: Path = ROOT, *,
                     code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
                     ) -> dict[str, Any]:
    model_path = Path(ARDY_G1_XML)
    model_hash = _sha256(model_path)
    if model_hash != EXPECTED_G1_XML_SHA256:
        raise ValueError(f"g1.xml sha256 {model_hash} differs from the pinned "
                         f"{EXPECTED_G1_XML_SHA256}")
    return {
        "git": dict(code_state_fn(repo)),
        "source_sha256": _file_hashes(repo, SOURCE_FILES),
        "runtime": cal._runtime_identity(),
        "physical_model": {"path": str(model_path.resolve()), "sha256": model_hash,
                           "body_margin_m": BODY_MARGIN},
    }


def tracker_identity(sonic_root: str | Path = SONIC_EXP029_ROOT) -> dict[str, Any]:
    """EXP-030's **own** tracker baseline, taken from the patched worktree.

    Three refusals, in order: the execution root must not be the legacy checkout, its HEAD must
    contain the ``add_table`` fix commit, and the source itself must actually carry the fix
    (:func:`tracker_fix_report`, which does not trust the commit graph).  The core-source
    manifest is recorded beside EXP-022A's value rather than asserted equal to it -- this
    campaign needs a checkout that differs by exactly that fix, and the ``absent`` arm is what
    measures whether the difference is inert.  Dirty paths are recorded and every guarded file
    is content-hashed, so provenance is bound by content even where it is not bound by a commit.
    """
    root = require_execution_root(sonic_root)
    git = _git_state(root)
    branch = _git_branch(root)
    contains_fix = _commit_contains(root, ADD_TABLE_FIX_COMMIT)
    if not contains_fix:
        raise CampaignAbort(
            f"the EXP-030 tracker checkout {root} (HEAD {git['commit']}, branch {branch}) does "
            f"not contain the add_table fix commit {ADD_TABLE_FIX_COMMIT}")
    fix = read_tracker_fix(root)
    dirty_paths = [line[3:] for line in git["status"] if len(line) >= 4]
    guarded = set(CORE_SONIC_FILES) | set(EVALUATOR_SONIC_FILES) | {ADD_TABLE_FIX_FILE}
    bundle = resolve_release_bundle(root)
    checkpoint = Path(bundle["checkpoint_path"])
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"SONIC checkpoint sha256 {checkpoint_sha} differs from the pinned "
                         f"{EXPECTED_CHECKPOINT_SHA256}")
    if not Path(exp1b.SONIC_PY).is_file():
        raise ValueError(f"SONIC Python is missing: {exp1b.SONIC_PY}")
    if not e28.ISAACLAB_ROOT.is_dir():
        raise ValueError(f"Isaac Lab checkout is missing: {e28.ISAACLAB_ROOT}")
    isaac_git = _git_state(e28.ISAACLAB_ROOT)
    if isaac_git["dirty"]:
        raise ValueError("Isaac Lab checkout must be exactly clean for an execution campaign")
    core, core_paths = core_source_hashes(root, bundle)
    manifest = cal._json_hash(core)
    # The exported motion pickle's joint table is read through ``sonic_export.SONIC_ROOT``; it
    # must be the same converter this checkout ships, or the pickle is not attributable to it.
    export_root = Path(EXPORT_SONIC_ROOT).resolve()
    converter = "gear_sonic/data_process/convert_soma_csv_to_motion_lib.py"
    export_converter_sha = _sha256(export_root / converter)
    if export_converter_sha != core[converter]:
        raise ValueError(f"the motion-pickle converter read from {export_root} differs from the "
                         f"execution checkout's copy; set SONIC_ROOT={root} and re-run")
    legacy_core = (_file_hashes(LEGACY_SONIC_ROOT, [name for name in CORE_SONIC_FILES
                                                    if not name.startswith(RELEASE_BUNDLE_PREFIX)])
                   if LEGACY_SONIC_ROOT.is_dir() else {})
    differs_from_legacy = sorted(name for name, digest in legacy_core.items()
                                 if core.get(name) != digest)
    return {
        "root": str(root),
        "role": ("the dedicated patched worktree for obstacle-present campaigns; every arm, "
                 "including the absent control, runs here (CLAUDE.md: two tracker checkouts)"),
        "branch": branch,
        "expected_branch": SONIC_EXP029_BRANCH,
        "git": git,
        "contains_add_table_fix_commit": {"commit": ADD_TABLE_FIX_COMMIT, "contained": True},
        "dirty_paths": dirty_paths,
        "guarded_dirty_paths": sorted(path for path in dirty_paths if path in guarded),
        "core_source_sha256": core,
        "core_source_paths": core_paths,
        "core_source_manifest_sha256": manifest,
        "exp022a_core_source_manifest_sha256": EXP022A_CORE_MANIFEST_SHA256,
        "core_source_manifest_matches_exp022a": manifest == EXP022A_CORE_MANIFEST_SHA256,
        "legacy_checkout": {
            "root": str(LEGACY_SONIC_ROOT),
            "core_source_sha256": legacy_core,
            "files_differing_from_execution_root": differs_from_legacy,
        },
        "manifest_policy": ("EXP-030 declares its own tracker baseline: it requires the "
                            "add_table-without-add_object fix and therefore records both "
                            "manifests instead of asserting EXP-022A's; the per-file diff "
                            "against the legacy checkout is recorded above"),
        "evaluator_source_sha256": _file_hashes(root, EVALUATOR_SONIC_FILES),
        "add_table_fix": fix,
        "release_bundle": bundle,
        "checkpoint": {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size,
                       "sha256": checkpoint_sha, "source": bundle["source"]},
        "motion_export_root": {"path": str(export_root), "converter": converter,
                               "converter_sha256": export_converter_sha,
                               "matches_execution_checkout": True},
        "python": str(Path(exp1b.SONIC_PY).resolve()),
        "python_runtime": exp022._sonic_python_runtime(Path(exp1b.SONIC_PY)),
        "isaaclab": {"root": str(e28.ISAACLAB_ROOT.resolve()), "git": isaac_git},
        "physics_seed": PHYSICS_SEED,
        "expected_achieved_sample_dt_s": SAMPLE_DT_S,
        "callback_schema_version": ARCHIVE_SCHEMA_VERSION,
    }


def bound_tracker_identity(tracker: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of the tracker identity that must not change during the campaign."""
    fix = tracker.get("add_table_fix", {})
    return {
        "root": tracker["root"],
        "branch": tracker.get("branch"),
        "commit": tracker["git"]["commit"],
        "checkpoint_path": tracker["checkpoint"].get("path"),
        "core_source_manifest_sha256": tracker["core_source_manifest_sha256"],
        "evaluator_source_sha256": tracker.get("evaluator_source_sha256"),
        "checkpoint_sha256": tracker["checkpoint"]["sha256"],
        "add_table_fix_sha256": fix.get("sha256"),
        "add_table_fix_present": fix.get("fix_present"),
        "python_runtime": tracker.get("python_runtime"),
        "isaaclab_commit": tracker.get("isaaclab", {}).get("git", {}).get("commit"),
    }


def exp022a_achieved_rows(exp022a_dir: str | Path = EXP022A_OUT) -> dict[str, dict[str, Any]]:
    """EXP-022A's per-reference achieved rows (staged obstacle), for the P1 control check."""
    path = Path(exp022a_dir) / "achieved_rows.jsonl"
    rows = [row for row in _read_jsonl(path) if row.get("obstacle_label") == "staged"]
    by_key = {str(row["motion_key"]): row for row in rows}
    if len(by_key) != len(POOL_SEEDS):
        raise ValueError(f"EXP-022A achieved rows cover {len(by_key)} references, expected "
                         f"{len(POOL_SEEDS)}")
    return {"rows": by_key, "path": str(path.resolve()), "sha256": _sha256(path)}


# ------------------------------------------------------------------------------ launch plan

def table_spec(box_height_m: float | None) -> dict[str, Any] | None:
    """The Isaac cuboid: full x/y/z extents, and the **centre** (so z = height / 2)."""
    if box_height_m is None:
        return None
    height = float(box_height_m)
    if height <= 0:
        raise ValueError("box height must be positive")
    return {
        "pos": [float(OBSTACLE_X_M), 0.0, height / 2.0],
        "quat": list(IDENTITY_QUAT_WXYZ),
        "size_xyz": [float(OBSTACLE_DEPTH_M), float(OBSTACLE_WIDTH_M), height],
        "height_m": height,
        "convention": ("CuboidCfg size is the full x/y/z extents and the position is the box "
                       "centre; a box of height h resting on the floor is "
                       "size=[depth_x, width_y, h] at pos=[x, y, h/2]"),
        "width_source": "scene2motion.stepover_eval.step_scene (half-width "
                        f"{OBSTACLE_HALF_WIDTH_M} m)",
    }


def shared_overrides() -> list[str]:
    return [f"++manager_env.config.env_spacing={float(ENV_SPACING_M)}",
            f"++manager_env.config.episode_length_s={float(EPISODE_LENGTH_S)}"]


def arm_overrides(box_height_m: float | None) -> list[str]:
    """The probe-proven override list: shared env geometry, plus the table for a present arm."""
    overrides = shared_overrides()
    table = table_spec(box_height_m)
    if table is None:
        return overrides
    return overrides + [
        "++manager_env.config.add_table=true",
        "++manager_env.config.table_size=" + json.dumps(table["size_xyz"]),
        "++manager_env.config.table_position=" + json.dumps(table["pos"]),
    ]


def launch_plan() -> list[dict[str, Any]]:
    """Six launches: three arms x EXP-022A's two 32-motion chunks, in its exact key order."""
    chunks = exp022.chunk_plan()
    plan: list[dict[str, Any]] = []
    for arm in ARMS:
        for chunk in chunks:
            plan.append({
                "name": f"{arm['arm']}_chunk{chunk['chunk']:02d}_seed{PHYSICS_SEED}",
                "arm": arm["arm"],
                "chunk": chunk["chunk"],
                "physics_seed": PHYSICS_SEED,
                "seeds": list(chunk["seeds"]),
                "motion_keys": list(chunk["motion_keys"]),
                "n_motions": chunk["n_motions"],
                "box_height_m": arm["box_height_m"],
                "obstacle_in_physics": arm["box_height_m"] is not None,
                "table": table_spec(arm["box_height_m"]),
                "extra_overrides": arm_overrides(arm["box_height_m"]),
            })
    if len(plan) != 6:
        raise RuntimeError("EXP-030 must contain exactly six launches")
    return plan


def launches_for_arm(arm: str) -> list[dict[str, Any]]:
    return [spec for spec in launch_plan() if spec["arm"] == arm]


# ------------------------------------------------------------------------------ motion pickles

def write_arm_motion_pkl(clips: Mapping[str, np.ndarray], path: Path,
                         table: Mapping[str, Any] | None, *,
                         export_fn: Callable[..., Path] = write_motion_pkl,
                         mj_model: Any = None, fps: int = FPS) -> Path:
    """Export the chunk, then give every motion the arm's table pose (REPORT §47).

    ``commands.py`` rewrites the table pose per environment on every reset from the motion's own
    ``table_pos`` / ``table_quat`` plus that environment's origin; without them a reset-time
    fallback moves the obstacle somewhere else entirely.
    """
    path = Path(path)
    export_fn(dict(clips), path, fps=fps, mj_model=mj_model)
    if table is None:
        return path
    with path.open("rb") as handle:
        motions = pickle.load(handle)
    if set(motions) != set(clips):
        raise ValueError("exported motion pickle keys do not match the chunk")
    for entry in motions.values():
        entry["table_pos"] = [float(value) for value in table["pos"]]
        entry["table_quat"] = [float(value) for value in table["quat"]]
    with path.open("wb") as handle:
        pickle.dump(motions, handle, protocol=4)
    return path


def validate_motion_pkl(path: Path, expected_keys: Sequence[str],
                        table: Mapping[str, Any] | None) -> None:
    exp022._validate_motion_pkl(Path(path), expected_keys)
    with Path(path).open("rb") as handle:
        motions = pickle.load(handle)
    for key in expected_keys:
        entry = motions[key]
        if table is None:
            if "table_pos" in entry or "table_quat" in entry:
                raise ValueError(f"obstacle-absent motion {key} carries table metadata")
            continue
        pos = [float(value) for value in entry.get("table_pos", [])]
        quat = [float(value) for value in entry.get("table_quat", [])]
        if pos != [float(value) for value in table["pos"]]:
            raise ValueError(f"motion {key} has table_pos {pos}, expected {table['pos']}")
        if quat != [float(value) for value in table["quat"]]:
            raise ValueError(f"motion {key} has table_quat {quat}, expected {table['quat']}")


def ensure_motion_pkl(spec: Mapping[str, Any], clips: Mapping[str, np.ndarray], output: Path,
                      *, export_fn: Callable[..., Path] = write_motion_pkl,
                      mj_model: Any = None) -> Path:
    """EXP-022A's deterministic-export discipline, plus this arm's per-motion table metadata."""
    launch_dir = Path(output) / "launches" / str(spec["name"])
    launch_dir.mkdir(parents=True, exist_ok=True)
    path = launch_dir / "motions.pkl"
    temporary = launch_dir / "motions.expected.tmp.pkl"
    if temporary.exists():
        raise CampaignAbort(f"stale temporary motion pickle requires inspection: {temporary}")
    selected = {key: clips[key] for key in spec["motion_keys"]}
    write_arm_motion_pkl(selected, temporary, spec["table"], export_fn=export_fn,
                         mj_model=mj_model)
    validate_motion_pkl(temporary, spec["motion_keys"], spec["table"])
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    if path.exists():
        validate_motion_pkl(path, spec["motion_keys"], spec["table"])
        if _sha256(path) != _sha256(temporary):
            raise CampaignAbort(f"existing motion pickle differs from deterministic export: {path}")
        temporary.unlink()
        return path
    os.replace(temporary, path)
    directory_fd = os.open(launch_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path


# ------------------------------------------------------------------------------ attempts

def _overrides_sha256(overrides: Sequence[str]) -> str:
    return cal._json_hash([str(item) for item in overrides])


def _attempt_expectations(spec: Mapping[str, Any], pkl_sha256: str) -> dict[str, Any]:
    return {
        "launch": spec["name"], "arm": spec["arm"], "physics_seed": int(spec["physics_seed"]),
        "motion_keys": list(spec["motion_keys"]), "motion_pkl_sha256": pkl_sha256,
        "extra_overrides": list(spec["extra_overrides"]),
        "box_height_m": spec["box_height_m"], "table": spec["table"],
    }


def _write_process_result(attempt: Path, *, returncode: int, log_path: Path,
                          spec: Mapping[str, Any], motion_pkl_sha256: str) -> dict[str, Any]:
    payload = {
        "schema": PROCESS_RESULT_SCHEMA,
        "returncode": int(returncode),
        "returncode_observed": True,
        "sonic_log_sha256": _sha256(log_path),
        "launch": str(spec["name"]),
        "arm": str(spec["arm"]),
        "physics_seed": int(spec["physics_seed"]),
        "motion_keys": list(spec["motion_keys"]),
        "motion_pkl_sha256": motion_pkl_sha256,
        "extra_overrides": list(spec["extra_overrides"]),
        "extra_overrides_sha256": _overrides_sha256(spec["extra_overrides"]),
    }
    cal._write_json(attempt / "process_result.json", payload)
    return {**payload, "file_sha256": _sha256(attempt / "process_result.json")}


def _validate_process_result(attempt: Path, *, spec: Mapping[str, Any],
                             motion_pkl_sha256: str) -> dict[str, Any]:
    path = attempt / "process_result.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CampaignAbort(f"valid SONIC artifacts lack durable return-code evidence: "
                            f"{attempt}") from exc
    if not isinstance(payload, dict):
        raise CampaignAbort(f"SONIC process-result receipt is not an object: {attempt}")
    expected = {
        "schema": PROCESS_RESULT_SCHEMA,
        "returncode_observed": True,
        "sonic_log_sha256": _sha256(attempt / "sonic.log"),
        "launch": str(spec["name"]),
        "arm": str(spec["arm"]),
        "physics_seed": int(spec["physics_seed"]),
        "motion_keys": list(spec["motion_keys"]),
        "motion_pkl_sha256": motion_pkl_sha256,
        "extra_overrides": list(spec["extra_overrides"]),
        "extra_overrides_sha256": _overrides_sha256(spec["extra_overrides"]),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise CampaignAbort(f"SONIC process-result receipt has mismatched {field}: {attempt}")
    returncode = payload.get("returncode")
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise CampaignAbort(f"SONIC process-result return code is invalid: {attempt}")
    if returncode != 0:
        raise CampaignAbort(f"SONIC process returned {returncode}: {attempt}")
    return {**payload, "file_sha256": _sha256(path)}


def _check_log_terminations(log: str) -> dict[str, bool]:
    """Every arm runs the unchanged release evaluator; the log's own table must say so."""
    terms = parse_log_termination_terms(log)
    if tuple(sorted(terms)) != RELEASE_TERMINATION_TERMS:
        raise ValueError(f"SONIC log termination terms {sorted(terms)} differ from the release "
                         f"evaluator's {list(RELEASE_TERMINATION_TERMS)}")
    if terms.get(e28.TIME_OUT_TERM) is not True:
        raise ValueError("motion_time_out is not the log's time-out term")
    return terms


def _rollout_check(spec: Mapping[str, Any], rollouts: Sequence[Any]) -> dict[str, Any]:
    if len(rollouts) != int(spec["n_motions"]):
        raise ValueError(f"{spec['name']}: archived {len(rollouts)} rollouts, expected "
                         f"{spec['n_motions']}")
    return {
        "n_rollouts": len(rollouts),
        "terminated_keys": sorted(r.motion_key for r in rollouts if r.terminated),
        "valid_lengths": {r.motion_key: int(r.valid_length) for r in rollouts},
    }


def run_or_resume_launch(
    spec: Mapping[str, Any], pkl: Path, output: Path, *,
    launch_fn: Callable[..., tuple[int, str]], timeout_s: int,
    sonic_root: str | Path = SONIC_EXP029_ROOT,
    checkpoint: str | Path | None = None,
    host_gate_fn: Callable[..., Mapping[str, Any]] = host_gate.require_host_resources,
    isaac_fn: Callable[..., Sequence[Mapping[str, Any]]] = host_gate.concurrent_isaac_processes,
    rewrite_receipt: bool = True,
) -> tuple[dict[str, Any], list[Any]]:
    """EXP-028's attempt discipline, with this campaign's arm/table expectations bound in.

    ``rewrite_receipt=False`` adopts a completed attempt read-only, so the analysis stage never
    touches launch evidence.
    """
    launch_dir = Path(output) / "launches" / str(spec["name"])
    attempts = sorted(path for path in launch_dir.glob("attempt-*") if path.is_dir())
    pkl_sha256 = _sha256(Path(pkl))
    if pkl_sha256 is None:
        raise CampaignAbort(f"SONIC motion pickle disappeared: {pkl}")
    expectations = _attempt_expectations(spec, pkl_sha256)

    attempt_infos: list[tuple[Path, dict[str, Any] | None]] = []
    for attempt in attempts:
        receipt_path = attempt / "receipt.json"
        attempt_receipt = None
        if receipt_path.is_file():
            try:
                attempt_receipt = json.loads(receipt_path.read_text())
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise CampaignAbort(f"unreadable attempt receipt requires inspection: "
                                    f"{attempt}") from exc
            if not isinstance(attempt_receipt, dict):
                raise CampaignAbort(f"attempt receipt is not an object: {attempt}")
        elif any(attempt.iterdir()):
            raise CampaignAbort(f"SONIC attempt has artifacts but no pre-launch receipt: {attempt}")
        if attempt_receipt and attempt_receipt.get("status") == "failed":
            raise CampaignAbort(f"{spec['name']} has a recorded failed SONIC attempt; "
                                "preserve it and use a fresh campaign output")
        if attempt_receipt is not None:
            if attempt_receipt.get("status") not in {"running", "complete"}:
                raise CampaignAbort(f"attempt has an unknown status: {attempt}")
            for field, expected in expectations.items():
                if attempt_receipt.get(field) != expected:
                    raise CampaignAbort(f"attempt {attempt} has mismatched {field}; "
                                        "refusing provenance relabel")
        attempt_infos.append((attempt, attempt_receipt))

    valid: list[tuple[Path, dict[str, Any], dict[str, Any], list[Any]]] = []
    for attempt, attempt_receipt in attempt_infos:
        if attempt_receipt is None:
            continue
        process_result = None
        if (attempt / "process_result.json").is_file():
            process_result = _validate_process_result(attempt, spec=spec,
                                                      motion_pkl_sha256=pkl_sha256)
        try:
            record, rollouts = exp022.validate_attempt(attempt, spec["motion_keys"])
        except (OSError, ValueError):
            if attempt_receipt.get("status") == "complete" or process_result is not None:
                raise CampaignAbort(f"completed attempt is now invalid: {attempt}")
            continue
        if process_result is None:
            raise CampaignAbort(f"valid SONIC artifacts lack durable return-code evidence: "
                                f"{attempt}")
        if attempt_receipt.get("status") == "complete":
            if (attempt_receipt.get("returncode_observed") is not True
                    or attempt_receipt.get("returncode") != 0):
                raise CampaignAbort(f"completed attempt has invalid return-code evidence: {attempt}")
            if attempt_receipt.get("process_result") != process_result:
                raise CampaignAbort(f"completed attempt changed its process result: {attempt}")
            for field in ("artifacts", "archive_schema_version", "sample_dt_s",
                          "motion_id_key_map_sha256", "n_rollouts"):
                if attempt_receipt.get(field) != record.get(field):
                    raise CampaignAbort(f"completed attempt changed its {field}: {attempt}")
        valid.append((attempt, record, process_result, rollouts))

    if len(valid) > 1:
        raise CampaignAbort(f"{spec['name']} has multiple complete SONIC attempts; "
                            "refusing ambiguous evidence")
    if valid:
        attempt, record, process_result, rollouts = valid[0]
        try:
            log_terms = _check_log_terminations((attempt / "sonic.log").read_text())
            check = _rollout_check(spec, rollouts)
        except (OSError, ValueError) as exc:
            raise CampaignAbort(f"{spec['name']}: completed attempt fails revalidation: "
                                f"{exc}") from exc
        gate_path = attempt / "host_resource_gate.json"
        gate = json.loads(gate_path.read_text()) if gate_path.is_file() else None
        record["recovered_or_resumed"] = True
        record.update({"status": "complete", **expectations, "returncode": 0,
                       "returncode_observed": True, "process_result": process_result,
                       "log_termination_terms": log_terms, "rollout_check": check,
                       "host_resource_gate": gate})
        if rewrite_receipt:
            cal._write_json(attempt / "receipt.json", _jsonable(record))
        record["attempt_receipt_sha256"] = _sha256(attempt / "receipt.json")
        return record, rollouts

    # The host gate runs before the attempt directory exists: a failed gate leaves no attempt
    # behind, so the same campaign directory can be resumed once the host is free.
    attempt = launch_dir / f"attempt-{len(attempts):03d}"
    gate = dict(host_gate_fn(**host_gate.SONIC_LAUNCH_GATE))
    isaac = [dict(item) for item in isaac_fn()]
    host = {"gate": gate, "concurrent_isaac_processes": isaac,
            "n_concurrent_isaac_processes": len(isaac),
            "note": ("the SONIC preset does not gate on Isaac co-tenants (see host_gate); they "
                     "are recorded per launch as the protocol requires")}
    command = build_sonic_command(
        pkl, attempt / "eval", spec["n_motions"], spec["physics_seed"], spec["extra_overrides"],
        checkpoint=checkpoint or (Path(sonic_root) / RELEASE_CHECKPOINT_RELATIVE))
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cal._write_json(attempt / "receipt.json", _jsonable({"status": "running", **expectations}))
    try:
        cal._write_json(attempt / "host_resource_gate.json", _jsonable(host))
        cal._write_json(attempt / "command.json", {
            "command": command, "cwd": str(Path(sonic_root).resolve()),
            "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
            "note": ("the subprocess runs with the patched worktree as its working directory, "
                     "so `python -m gear_sonic...` resolves that checkout")})
        rc, log = launch_fn(pkl, attempt / "eval", int(spec["n_motions"]),
                            int(spec["physics_seed"]), timeout_s, list(spec["extra_overrides"]))
        exp022._atomic_text(attempt / "sonic.log", log)
        process_result = _write_process_result(attempt, returncode=rc,
                                               log_path=attempt / "sonic.log", spec=spec,
                                               motion_pkl_sha256=pkl_sha256)
        if rc != 0:
            raise RuntimeError(f"SONIC returned {rc}")
        record, rollouts = exp022.validate_attempt(attempt, spec["motion_keys"])
        log_terms = _check_log_terminations(log)
        check = _rollout_check(spec, rollouts)
        record.update({"status": "complete", **expectations, "returncode": rc,
                       "elapsed_s": float(time.monotonic() - started),
                       "recovered_or_resumed": False, "returncode_observed": True,
                       "process_result": process_result, "log_termination_terms": log_terms,
                       "rollout_check": check, "host_resource_gate": host,
                       "command_sha256": _sha256(attempt / "command.json")})
        cal._write_json(attempt / "receipt.json", _jsonable(record))
        record["attempt_receipt_sha256"] = _sha256(attempt / "receipt.json")
        return record, rollouts
    except Exception as exc:
        cal._write_json(attempt / "receipt.json", _jsonable({
            "status": "failed", **expectations, "error_type": type(exc).__name__,
            "error": str(exc), "elapsed_s": float(time.monotonic() - started)}))
        raise CampaignAbort(f"SONIC launch {spec['name']} failed: {exc}") from exc


def _forbid_launch(*_args: Any, **_kwargs: Any) -> tuple[int, str]:
    raise CampaignAbort("the analysis stage must never launch SONIC; a launch is incomplete")


def load_completed_launch(spec: Mapping[str, Any], output: Path) -> tuple[dict[str, Any], list[Any]]:
    """Adopt a completed launch's archive through the full attempt audit, never launching."""
    return run_or_resume_launch(
        spec, Path(output) / "launches" / str(spec["name"]) / "motions.pkl", output,
        launch_fn=_forbid_launch, timeout_s=0, host_gate_fn=_forbid_launch,
        isaac_fn=_forbid_launch, rewrite_receipt=False)


# ------------------------------------------------------------------------------ scoring

def scene_for(box_height_m: float | None):
    """The traversal problem: start at the route origin, goal at its end, the arm's box between."""
    base = step_scene(OBSTACLE_X_M, float(box_height_m or PROXY_HEIGHT_M), OBSTACLE_DEPTH_M)
    boxes = list(base.boxes) if box_height_m is not None else []
    scene_id = ("route_no_obstacle" if box_height_m is None
                else f"route_box_h{float(box_height_m):.3f}")
    return replace(base, scene_id=scene_id, boxes=boxes, start=ROUTE_START_XY, goal=ROUTE_GOAL_XY,
                   meta={"corridor_half": OBSTACLE_HALF_WIDTH_M})


def criteria() -> te.TraversalCriteria:
    return te.TraversalCriteria(goal_tolerance_m=GOAL_TOLERANCE_M,
                                corridor_half_width_m=OBSTACLE_HALF_WIDTH_M,
                                time_limit_s=None)


class CollisionCache:
    """One MuJoCo body per distinct box geometry; the default evaluator rebuilds it per call."""

    def __init__(self) -> None:
        self._bodies: dict[str, Any] = {}

    @staticmethod
    def key(scene: Any) -> str:
        return cal._json_hash([[list(box.center), list(box.half), str(box.label)]
                               for box in scene.boxes])

    def __call__(self, scene: Any, qpos: np.ndarray) -> Mapping[str, Any]:
        key = self.key(scene)
        body = self._bodies.get(key)
        if body is None:
            from scene2motion.robot import G1Body
            body = G1Body(scene=scene)
            self._bodies[key] = body
        return body.trajectory_report(np.asarray(qpos, dtype=float))


def _zero_length_record(rollout: Any, scene: Any) -> dict[str, Any]:
    """An archived rollout with no alive sample cannot be scored geometrically.

    It still occupies an assigned trial, so it is labelled by the tracker's own flag -- the
    evaluator cutoff if it fired, otherwise ``stalled`` -- and marked, never dropped.
    """
    return {
        "outcome": "cutoff" if rollout.terminated else "stalled",
        "evaluator_version": evaluator_version(), "executed": True,
        "zero_length_archive": True, "scene_id": scene.scene_id, "samples": 0,
        "duration_s": 0.0, "tracker_terminated": bool(rollout.terminated),
        "collided_obstacle": False, "collided_wall": False, "fell": False, "timed_out": False,
        "timeout_assessed": False, "passed_obstacle": False, "passed_within_corridor": False,
        "reached_goal": False, "clearance_ok": False,
        "note": "the achieved archive holds no alive sample for this reference",
    }


def score_rollout(rollout: Any, scene: Any, *,
                  collision_fn: Callable[..., Mapping[str, Any]] | None = None,
                  ) -> dict[str, Any]:
    qpos = np.asarray(rollout.qpos, dtype=float)
    if len(qpos) == 0:
        record = _zero_length_record(rollout, scene)
    else:
        record = te.evaluate_traversal(
            qpos, scene, terminated=bool(rollout.terminated), sample_dt_s=SAMPLE_DT_S,
            criteria=criteria(),
            **({"collision_fn": collision_fn} if collision_fn is not None else {}))
    record["motion_key"] = rollout.motion_key
    return record


def build_rows(rollouts_by_arm: Mapping[str, Mapping[str, Any]], *,
               collision_fn: Callable[..., Mapping[str, Any]] | None = None,
               ) -> list[dict[str, Any]]:
    """One row per (arm, reference): the traversal record, plus the absent arm's replay proxy."""
    scenes = {arm["arm"]: scene_for(arm["box_height_m"]) for arm in ARMS}
    proxy_scene = scene_for(PROXY_HEIGHT_M)
    heights = {arm["arm"]: arm["box_height_m"] for arm in ARMS}
    rows: list[dict[str, Any]] = []
    for arm in ARM_NAMES:
        rollouts = rollouts_by_arm[arm]
        for seed in POOL_SEEDS:
            key = f"s{seed}"
            rollout = rollouts[key]
            qpos = np.asarray(rollout.qpos, dtype=float)
            record = score_rollout(rollout, scenes[arm], collision_fn=collision_fn)
            row = {
                "arm": arm, "seed": int(seed), "motion_key": key,
                "motion_id": int(rollout.motion_id), "physics_seed": PHYSICS_SEED,
                "obstacle_in_physics": heights[arm] is not None,
                "box_height_m": heights[arm],
                "obstacle_x_m": OBSTACLE_X_M, "obstacle_depth_m": OBSTACLE_DEPTH_M,
                "obstacle_width_m": OBSTACLE_WIDTH_M,
                "valid_frames": int(rollout.valid_length),
                "valid_time_s": float(rollout.valid_length * SAMPLE_DT_S),
                "tracker_terminated": bool(rollout.terminated),
                "tracker_reported_progress": float(rollout.progress),
                "max_root_x_m": float(qpos[:, 0].max()) if len(qpos) else None,
                "final_root_x_m": float(qpos[-1, 0]) if len(qpos) else None,
                "outcome": record["outcome"],
                "traversal": record,
            }
            if arm == "absent":
                proxy = score_rollout(rollout, proxy_scene, collision_fn=collision_fn)
                row["replay_inferred"] = {
                    "box_height_m": PROXY_HEIGHT_M, "outcome": proxy["outcome"],
                    "definition": ("the obstacle-absent achieved states scored against the same "
                                   "5 cm box: what the project's earlier replay endpoint would "
                                   "have concluded"),
                    "traversal": proxy,
                }
            rows.append(row)
    return rows


# ------------------------------------------------------------------------------ statistics

def wilson(k: int, n: int) -> list[float] | None:
    return None if n < 1 else list(addr.wilson_interval(int(k), int(n)))


def summarise_arm(records: Sequence[Mapping[str, Any]], *, arm: str,
                  box_height_m: float | None) -> dict[str, Any]:
    """The full outcome breakdown over all 64 assigned trials, plus completion with a Wilson CI."""
    summary = te.summarise(records)
    total = summary["n_assigned_trials"]
    completed = summary["outcomes"]["completed"]
    summary.update({
        "arm": arm, "box_height_m": box_height_m,
        "obstacle_in_physics": box_height_m is not None,
        "local_traversal_completion": {
            "completed": completed, "n_assigned_trials": total,
            "rate": (completed / total) if total else 0.0,
            "wilson95": wilson(completed, total),
            "completing_motion_keys": sorted(r["motion_key"] for r in records
                                             if r.get("outcome") == "completed"),
            "definition": ("local traversal: passed the obstacle inside the corridor "
                           f"(|y| <= {OBSTACLE_HALF_WIDTH_M} m), reached within "
                           f"{GOAL_TOLERANCE_M} m of the goal at {ROUTE_GOAL_XY[0]} m after "
                           "passing, collision-free and upright; walking around does not count "
                           "and this is not navigation"),
        },
        "outcome_precedence": list(te.OUTCOMES),
    })
    return summary


def confusion_matrix(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Rows = replay-inferred class, columns = physics-measured class."""
    labels = [name for name in te.OUTCOMES
              if any(name in pair for pair in pairs)]
    matrix = {row: {column: 0 for column in labels} for row in labels}
    for inferred, measured in pairs:
        matrix[inferred][measured] += 1
    return {"labels": labels, "matrix": matrix,
            "row_definition": "replay-inferred class (absent arm scored against the 5 cm box)",
            "column_definition": "physics-measured class (present_05 arm)"}


def cohens_kappa(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Cohen's kappa for two labellings of the same references.

    ``kappa = (po - pe) / (1 - pe)`` with ``po`` the observed agreement and ``pe`` the agreement
    expected from the two marginal label distributions.  When both labellings are constant and
    identical, ``pe == 1`` and kappa is undefined -- reported as ``None`` with ``degenerate``
    set, never silently as 1.0 or 0.0.
    """
    n = len(pairs)
    if n < 1:
        raise ValueError("Cohen's kappa needs at least one paired observation")
    labels = sorted({label for pair in pairs for label in pair})
    observed = sum(1 for a, b in pairs if a == b) / n
    marginal_a = {label: sum(1 for a, _ in pairs if a == label) / n for label in labels}
    marginal_b = {label: sum(1 for _, b in pairs if b == label) / n for label in labels}
    expected = sum(marginal_a[label] * marginal_b[label] for label in labels)
    degenerate = bool(1.0 - expected <= 1e-12)
    return {
        "n": n,
        "observed_agreement": float(observed),
        "expected_agreement": float(expected),
        "kappa": None if degenerate else float((observed - expected) / (1.0 - expected)),
        "degenerate": degenerate,
        "degenerate_note": ("both labellings are (almost) constant, so chance agreement is 1 "
                            "and kappa is undefined; the agreement fraction still applies"),
    }


def bootstrap_kappa(pairs: Sequence[tuple[str, str]], *,
                    n_resamples: int = KAPPA_BOOTSTRAP_RESAMPLES,
                    seed: int = KAPPA_BOOTSTRAP_SEED) -> dict[str, Any]:
    """Percentile bootstrap over references; degenerate resamples are excluded and counted."""
    pairs = list(pairs)
    n = len(pairs)
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    degenerate = 0
    for _ in range(int(n_resamples)):
        index = rng.integers(0, n, size=n)
        sample = [pairs[int(i)] for i in index]
        kappa = cohens_kappa(sample)["kappa"]
        if kappa is None:
            degenerate += 1
        else:
            values.append(float(kappa))
    ci = ([float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
          if len(values) >= 100 else None)
    return {
        "n_resamples": int(n_resamples), "seed": int(seed),
        "n_finite": len(values), "n_degenerate_excluded": degenerate,
        "ci95": ci,
        "method": ("percentile bootstrap over the 64 references; resamples whose chance "
                   "agreement is 1 have no defined kappa and are excluded and counted; the "
                   "interval is reported only when at least 100 finite resamples remain"),
    }


def proxy_check(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Q1: does the obstacle-absent replay predict the obstacle-present class, per reference?"""
    inferred = {row["motion_key"]: row["replay_inferred"]["outcome"]
                for row in rows if row["arm"] == "absent"}
    measured = {row["motion_key"]: row["outcome"] for row in rows if row["arm"] == PROXY_ARM}
    keys = sorted(set(inferred) & set(measured))
    if len(keys) != len(POOL_SEEDS):
        raise ValueError(f"the proxy check covers {len(keys)} references, expected "
                         f"{len(POOL_SEEDS)}")
    pairs = [(inferred[key], measured[key]) for key in keys]
    agree = sum(1 for a, b in pairs if a == b)
    kappa = cohens_kappa(pairs)
    per_class = {}
    for label in sorted({a for a, _ in pairs} | {b for _, b in pairs}):
        n_measured = sum(1 for _, b in pairs if b == label)
        n_inferred = sum(1 for a, _ in pairs if a == label)
        both = sum(1 for a, b in pairs if a == b == label)
        per_class[label] = {
            "n_replay_inferred": n_inferred, "n_physics_measured": n_measured,
            "n_agreeing": both,
            "agreement_of_measured": (both / n_measured) if n_measured else None,
        }
    return {
        "question": ("Q1: does the obstacle-absent replay predict the obstacle-present outcome, "
                     "per reference?"),
        "box_height_m": PROXY_HEIGHT_M,
        "n": len(keys),
        "n_agreeing": agree,
        "agreement_fraction": agree / len(keys),
        "agreement_wilson95": wilson(agree, len(keys)),
        "confusion": confusion_matrix(pairs),
        "cohens_kappa": kappa,
        "kappa_bootstrap": bootstrap_kappa(pairs),
        "per_class_agreement": per_class,
        "disagreeing_references": [
            {"motion_key": key, "replay_inferred": inferred[key], "physics_measured": measured[key]}
            for key in keys if inferred[key] != measured[key]],
        "scope": ("'collided_obstacle' means measured contact in the present arm and a replay "
                  "intersection in the inferred column; that distinction is the point of the "
                  "comparison"),
    }


def paired_progress_change(rows: Sequence[Mapping[str, Any]],
                           arm: str = PROXY_ARM,
                           threshold_m: float = 0.05) -> dict[str, Any]:
    """How much the obstacle changed the rollout: paired absent - present maximum root x."""
    absent = {row["motion_key"]: row["max_root_x_m"] for row in rows if row["arm"] == "absent"}
    present = {row["motion_key"]: row["max_root_x_m"] for row in rows if row["arm"] == arm}
    paired = sorted(set(absent) & set(present))
    keys = [key for key in paired if absent[key] is not None and present[key] is not None]
    if not keys:
        raise ValueError("no reference has a measurable maximum root x in both arms")
    deltas = np.array([float(absent[key]) - float(present[key]) for key in keys], dtype=float)
    fell = [key for key, delta in zip(keys, deltas) if delta > threshold_m]
    q25, q75 = (float(np.percentile(deltas, 25)), float(np.percentile(deltas, 75)))
    return {
        "arm": arm, "n": len(keys),
        "n_excluded_without_a_measurable_maximum": len(paired) - len(keys),
        "definition": f"absent max root x - {arm} max root x, per reference (metres)",
        "median_m": float(np.median(deltas)),
        "iqr_m": [q25, q75],
        "iqr_width_m": q75 - q25,
        "min_m": float(deltas.min()), "max_m": float(deltas.max()),
        "threshold_m": float(threshold_m),
        "n_falling_more_than_threshold": len(fell),
        "references_falling_more_than_threshold": fell,
        "note": ("a reference that never reaches the box must show about 0; a large positive "
                 "difference means the box stopped the robot short"),
    }


def exp022a_agreement(rows: Sequence[Mapping[str, Any]],
                      exp022a_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """P1: does the `absent` arm reproduce EXP-022A on this tracker build?"""
    absent = {row["motion_key"]: row for row in rows if row["arm"] == "absent"}
    keys = sorted(set(absent) & set(exp022a_rows))
    if len(keys) != len(POOL_SEEDS):
        raise ValueError(f"the EXP-022A comparison covers {len(keys)} references, expected "
                         f"{len(POOL_SEEDS)}")
    termination_agree, length_agree, disagreements = 0, 0, []
    for key in keys:
        mine, theirs = absent[key], exp022a_rows[key]
        same_termination = bool(mine["tracker_terminated"]) == bool(theirs["tracker_terminated"])
        same_length = int(mine["valid_frames"]) == int(theirs["valid_frames"])
        termination_agree += int(same_termination)
        length_agree += int(same_length)
        if not (same_termination and same_length):
            disagreements.append({
                "motion_key": key,
                "terminated": {"exp030_absent": bool(mine["tracker_terminated"]),
                               "exp022a": bool(theirs["tracker_terminated"])},
                "valid_frames": {"exp030_absent": int(mine["valid_frames"]),
                                 "exp022a": int(theirs["valid_frames"])}})
    return {
        "n": len(keys),
        "termination_flag": {"n_agreeing": termination_agree, "n": len(keys),
                             "fraction": termination_agree / len(keys),
                             "wilson95": wilson(termination_agree, len(keys))},
        "valid_length": {"n_agreeing": length_agree, "n": len(keys),
                         "fraction": length_agree / len(keys),
                         "wilson95": wilson(length_agree, len(keys)),
                         "role": "reported alongside; the preregistered rule is the flag"},
        "terminated_counts": {
            "exp030_absent": sum(bool(absent[key]["tracker_terminated"]) for key in keys),
            "exp022a": sum(bool(exp022a_rows[key]["tracker_terminated"]) for key in keys)},
        "disagreeing_references": disagreements,
    }


def evaluate_predictions(*, arms: Mapping[str, Mapping[str, Any]],
                         p1: Mapping[str, Any],
                         proxy: Mapping[str, Any]) -> dict[str, Any]:
    """Mechanical evaluation of the protocol's P1/P2/P3 against their preregistered thresholds."""
    agree = int(p1["termination_flag"]["n_agreeing"])
    completions = {arm: int(arms[arm]["local_traversal_completion"]["completed"])
                   for arm in PRESENT_ARMS}
    completing = {arm: list(arms[arm]["local_traversal_completion"]["completing_motion_keys"])
                  for arm in PRESENT_ARMS}
    agreement = float(proxy["agreement_fraction"])
    kappa = proxy["cohens_kappa"]["kappa"]
    agreement_ok = bool(agreement >= P3_MIN_AGREEMENT)
    kappa_ok = bool(kappa is not None and kappa >= P3_MIN_KAPPA)
    evaluable = kappa is not None
    return {
        "P1": {
            "rule": P1_RULE, "threshold": P1_MIN_TERMINATION_AGREEMENT,
            "n_agreeing": agree, "n": int(p1["n"]),
            "prediction_held": bool(agree >= P1_MIN_TERMINATION_AGREEMENT),
            "consequence_if_failed": ("the fork fix or the run conditions are not inert: every "
                                      "cross-campaign comparison is re-scoped, Q1's answer is "
                                      "scoped to this tracker build, and the disagreement is "
                                      "the headline rather than the proxy result"),
        },
        "P2": {
            "rule": P2_RULE, "predicted_completions": P2_PREDICTED_COMPLETIONS,
            "completions": completions, "completing_motion_keys": completing,
            "prediction_held": bool(all(value == P2_PREDICTED_COMPLETIONS
                                        for value in completions.values())),
            "consequence_if_failed": ("any completion is the project's first measured local "
                                      "traversal and is reported prominently, with its "
                                      "reference identified"),
        },
        "P3": {
            "rule": P3_RULE,
            "min_agreement": P3_MIN_AGREEMENT, "min_kappa": P3_MIN_KAPPA,
            "agreement_fraction": agreement, "agreement_ok": agreement_ok,
            "kappa": kappa, "kappa_ok": kappa_ok,
            "kappa_ci95": proxy["kappa_bootstrap"]["ci95"],
            "kappa_degenerate": bool(proxy["cohens_kappa"]["degenerate"]),
            "evaluable": evaluable,
            "prediction_held": (bool(agreement_ok and kappa_ok) if evaluable else None),
            "note": (None if evaluable else
                     "kappa is undefined because chance agreement is 1 (both labellings are "
                     "constant); the rule as preregistered cannot be evaluated, and only the "
                     "agreement fraction is reported"),
            "consequence_if_failed": ("the paper states that obstacle-absent replay is a biased "
                                      "proxy and names the direction of the bias"),
        },
    }


def summarise(rows: Sequence[Mapping[str, Any]],
              exp022a_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    heights = {arm["arm"]: arm["box_height_m"] for arm in ARMS}
    arms = {}
    for arm in ARM_NAMES:
        records = [row["traversal"] for row in rows if row["arm"] == arm]
        arms[arm] = summarise_arm(records, arm=arm, box_height_m=heights[arm])
    proxy = proxy_check(rows)
    p1 = exp022a_agreement(rows, exp022a_rows)
    return {
        "status": "complete",
        "n_assigned_trials_per_arm": len(POOL_SEEDS),
        "arms": arms,
        "q1_proxy_check": proxy,
        "paired_progress_change": paired_progress_change(rows),
        "p1_absent_vs_exp022a": p1,
        "predictions": evaluate_predictions(arms=arms, p1=p1, proxy=proxy),
        "scene": {
            "start_xy_m": list(ROUTE_START_XY), "goal_xy_m": list(ROUTE_GOAL_XY),
            "goal_tolerance_m": GOAL_TOLERANCE_M,
            "corridor_half_width_m": OBSTACLE_HALF_WIDTH_M,
            "obstacle_x_m": OBSTACLE_X_M, "obstacle_depth_m": OBSTACLE_DEPTH_M,
            "obstacle_width_m": OBSTACLE_WIDTH_M,
            "sample_dt_s": SAMPLE_DT_S,
            "time_limit_s": None,
            "timeout_note": ("no wall-clock deadline was preregistered, so the timeout class is "
                             "not assessed and its count of zero means 'not assessed'"),
            "evaluator_version": evaluator_version(),
        },
        "interpretation_guard": (
            "The obstacle was present in the Isaac scene, so 'collided_obstacle' here means the "
            "robot actually contacted the box -- unlike every earlier campaign, where it meant "
            "the recorded motion intersected the box's volume in replay. Local traversal is not "
            "navigation: passing outside the corridor is a failure here. One route, one scene, "
            "one obstacle position, physics seed 0, one rollout per reference; rates are over "
            "all 64 assigned trials and are not generalised beyond this scene."),
    }


# ------------------------------------------------------------------------------ campaign

def _persist(output: Path, receipt: dict[str, Any], *, started: float,
             rows: Sequence[Mapping[str, Any]] | None = None,
             summary: Mapping[str, Any] | None = None) -> None:
    anchors = receipt.setdefault("evidence_anchors", {})
    if rows is not None:
        payload = [_jsonable(row) for row in rows]
        cal._write_jsonl(output / "rows.jsonl", payload)
        anchors["rows"] = {"n_rows": len(payload), "logical_sha256": cal._json_hash(payload),
                           "file_sha256": _sha256(output / "rows.jsonl")}
    if summary is not None:
        payload_summary = _jsonable(summary)
        cal._write_json(output / "summary.json", payload_summary)
        anchors["summary"] = {"logical_sha256": cal._json_hash(payload_summary),
                              "file_sha256": _sha256(output / "summary.json")}
    receipt["wall_clock_s"] = float(time.monotonic() - started)
    cal._write_json(output / "receipt.json", _jsonable(receipt))


def validate_completed_output(output: Path, receipt: Mapping[str, Any]) -> None:
    """Revalidate a completed campaign's evidence before an idempotent resume returns it."""
    anchors = receipt.get("evidence_anchors", {})
    rows = _read_jsonl(output / "rows.jsonl")
    anchor = anchors.get("rows", {})
    expected_rows = len(ARM_NAMES) * len(POOL_SEEDS)
    if (len(rows) != expected_rows or anchor.get("n_rows") != expected_rows
            or cal._json_hash(rows) != anchor.get("logical_sha256")
            or _sha256(output / "rows.jsonl") != anchor.get("file_sha256")):
        raise CampaignAbort("completed rows no longer match their evidence anchor")
    try:
        summary = json.loads((output / "summary.json").read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CampaignAbort(f"completed summary is unreadable: {exc}") from exc
    anchor = anchors.get("summary", {})
    if (cal._json_hash(summary) != anchor.get("logical_sha256")
            or _sha256(output / "summary.json") != anchor.get("file_sha256")
            or receipt.get("summary") != summary):
        raise CampaignAbort("completed summary no longer matches its evidence anchor")
    for spec in launch_plan():
        record = receipt.get("launches", {}).get(spec["name"])
        if not isinstance(record, dict) or record.get("status") != "complete":
            raise CampaignAbort(f"completed receipt lacks launch {spec['name']}")
        attempt = Path(record.get("attempt", ""))
        if (_sha256(attempt / "receipt.json") != record.get("attempt_receipt_sha256")
                or _sha256(attempt / "process_result.json")
                != record.get("process_result", {}).get("file_sha256")):
            raise CampaignAbort(f"completed launch artifacts changed for {spec['name']}")


def _campaign_identity(project: Mapping[str, Any], tracker: Mapping[str, Any],
                       source: Mapping[str, Any], exp022a: Mapping[str, Any],
                       protocol: Mapping[str, Any]) -> str:
    return cal._json_hash({
        "schema": SCHEMA_VERSION,
        "project_source_sha256": project["source_sha256"],
        "project_commit": project["git"].get("commit"),
        "physical_model": project["physical_model"],
        "tracker": bound_tracker_identity(tracker),
        "source_exp021": source, "exp022a": exp022a,
        "protocol_sha256": protocol["sha256"],
        "plan": launch_plan(),
        "execution_root": tracker.get("root"),
        "add_table_fix_commit": ADD_TABLE_FIX_COMMIT,
        "scene": {"start": list(ROUTE_START_XY), "goal": list(ROUTE_GOAL_XY),
                  "goal_tolerance_m": GOAL_TOLERANCE_M,
                  "corridor_half_width_m": OBSTACLE_HALF_WIDTH_M,
                  "obstacle_x_m": OBSTACLE_X_M, "obstacle_depth_m": OBSTACLE_DEPTH_M,
                  "obstacle_width_m": OBSTACLE_WIDTH_M},
        "predictions": {"P1": P1_MIN_TERMINATION_AGREEMENT, "P2": P2_PREDICTED_COMPLETIONS,
                        "P3": [P3_MIN_AGREEMENT, P3_MIN_KAPPA]},
    })


def run_campaign(
    *,
    stage: str = "all",
    out: str | Path = DEFAULT_OUT,
    resume: bool = False,
    dry_run: bool = False,
    source_dir: str | Path = SOURCE_OUT,
    exp022a_dir: str | Path = EXP022A_OUT,
    timeout_s: int = 2400,
    sonic_root: str | Path = SONIC_EXP029_ROOT,
    launch_fn: Callable[..., tuple[int, str]] | None = None,
    export_fn: Callable[..., Path] = write_motion_pkl,
    host_gate_fn: Callable[..., Mapping[str, Any]] = host_gate.require_host_resources,
    host_report_fn: Callable[..., Mapping[str, Any]] = host_gate.host_resource_report,
    isaac_fn: Callable[..., Sequence[Mapping[str, Any]]] = host_gate.concurrent_isaac_processes,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
    tracker_identity_fn: Callable[..., Mapping[str, Any]] = tracker_identity,
    protocol_identity_fn: Callable[[], Mapping[str, Any]] = protocol_identity,
    source_fn: Callable[[Any], Mapping[str, Any]] = exp022.load_source_bundle,
    exp022a_identity_fn: Callable[[Any], Mapping[str, Any]] = e28.exp022a_identity,
    exp022a_rows_fn: Callable[[Any], Mapping[str, Any]] = exp022a_achieved_rows,
    collision_fn: Callable[..., Mapping[str, Any]] | None = None,
    mj_model: Any = None,
    require_preregistered: bool = True,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    output = Path(out)
    source = source_fn(source_dir)
    exp022a = dict(exp022a_identity_fn(exp022a_dir))
    exp022a_rows = dict(exp022a_rows_fn(exp022a_dir))
    protocol = dict(protocol_identity_fn())
    current_project = project_identity(code_state_fn=code_state_fn)
    tracker = dict(tracker_identity_fn(sonic_root))
    checkpoint = tracker.get("checkpoint", {}).get("path") or str(
        Path(sonic_root) / RELEASE_CHECKPOINT_RELATIVE)
    if launch_fn is None:
        launch_fn = sonic_launcher(sonic_root, checkpoint)
    plan = launch_plan()

    if dry_run:
        report = dict(host_report_fn(**host_gate.SONIC_LAUNCH_GATE))
        isaac = [dict(item) for item in isaac_fn()]
        commands = {spec["name"]: build_sonic_command(
            output / "launches" / spec["name"] / "motions.pkl",
            output / "launches" / spec["name"] / "attempt-000/eval",
            spec["n_motions"], spec["physics_seed"], spec["extra_overrides"],
            checkpoint=checkpoint) for spec in plan}
        return {
            "schema": SCHEMA_VERSION, "experiment": "exp030_obstacle_present",
            "status": "dry_run", "writes_performed": False,
            "project_dirty_observed": current_project["git"].get("dirty"),
            "protocol": protocol, "tracker": bound_tracker_identity(tracker),
            "tracker_add_table_fix": tracker.get("add_table_fix"),
            "tracker_dirty_paths": tracker.get("dirty_paths"),
            "source": source["identity"], "exp022a": exp022a,
            "execution": {"sonic_root": str(Path(sonic_root).resolve()),
                          "branch": tracker.get("branch"),
                          "expected_branch": SONIC_EXP029_BRANCH,
                          "legacy_root_refused": str(LEGACY_SONIC_ROOT),
                          "checkpoint": checkpoint,
                          "release_bundle": tracker.get("release_bundle"),
                          "core_source_manifest_sha256":
                              tracker.get("core_source_manifest_sha256"),
                          "files_differing_from_legacy_checkout":
                              tracker.get("legacy_checkout", {})
                              .get("files_differing_from_execution_root")},
            "launch_plan": plan, "commands": commands,
            "host_resource_gate": report, "concurrent_isaac_processes": isaac,
            "campaign_identity_sha256": _campaign_identity(
                current_project, tracker, source["identity"], exp022a, protocol),
        }

    require_execution_root(sonic_root)
    require_tracker_fix(tracker.get("add_table_fix", {}))
    if require_preregistered and protocol.get("status") != "preregistered":
        raise CampaignAbort(f"EXP-030 protocol status is {protocol.get('status')!r}; commit it "
                            "as 'preregistered' before the first launch")
    existing_receipt = output / "receipt.json"
    old: dict[str, Any] | None = None
    resume_project_check: dict[str, Any] | None = None
    if resume:
        if not existing_receipt.is_file():
            raise CampaignAbort(f"--resume requires an existing EXP-030 receipt in {output}")
        try:
            old = json.loads(existing_receipt.read_text())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CampaignAbort(f"existing campaign receipt is unreadable: {exc}") from exc
        if not isinstance(old, dict):
            raise CampaignAbort("existing campaign receipt is not an object")
        if old.get("status") == "blocked" or old.get("schema") == FAILURE_SCHEMA_VERSION:
            raise CampaignAbort("existing EXP-030 campaign is blocked; preserve it and use a "
                                "fresh output directory")
        if old.get("schema") != SCHEMA_VERSION:
            raise CampaignAbort("existing output is not a resumable EXP-030 campaign")
        pinned_project = old.get("provenance", {}).get("project")
        if (not isinstance(pinned_project, dict)
                or pinned_project.get("git", {}).get("dirty") is not False):
            raise CampaignAbort("existing receipt lacks a clean pinned Scene2Motion identity")
        try:
            resume_project_check = validate_project_recheck(pinned_project, current_project, output)
        except ValueError as exc:
            raise CampaignAbort(str(exc)) from exc
        project = dict(pinned_project)
        if old.get("provenance", {}).get("protocol", {}).get("sha256") != protocol["sha256"]:
            raise CampaignAbort("EXP-030 protocol changed since the campaign was created")
    else:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise CampaignAbort(f"refusing non-empty output for a fresh campaign: {output} "
                                "(pass --resume to continue an EXP-030 campaign)")
        if current_project["git"].get("dirty") is not False:
            raise CampaignAbort("EXP-030 requires an exactly clean Scene2Motion worktree")
        project = current_project

    campaign_identity = _campaign_identity(project, tracker, source["identity"], exp022a, protocol)
    if old is not None and old.get("campaign_identity_sha256") != campaign_identity:
        raise CampaignAbort("existing EXP-030 output has a different campaign identity")
    if old is not None and old.get("status") == "complete":
        validate_completed_output(output, old)
        return old

    started = time.monotonic()
    if old is None:
        # Host gate before the output directory exists: a failed gate leaves nothing behind.
        try:
            initial_gate = dict(host_gate_fn(**host_gate.SONIC_LAUNCH_GATE))
        except host_gate.HostResourceGateFailed as exc:
            raise CampaignAbort(f"host-resource gate failed before the campaign was created: "
                                f"{exc}") from exc
        # An EMPTY directory may already exist: a refused gate or a crash before the ledger was
        # written leaves one behind, and refusing it would strand the campaign on a directory
        # containing nothing.  The non-empty refusal above is what protects real evidence.
        output.mkdir(parents=True, exist_ok=True)
        receipt: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "experiment": "exp030_obstacle_present",
            "status": "running", "complete": False, "blocked": False, "stage": "preflight",
            "campaign_identity_sha256": campaign_identity,
            "resume_supported": True,
            "claim_scope": ("the archived EXP-021 pool replayed through SONIC with the obstacle "
                            "PRESENT in the Isaac scene (per-motion table pose) beside an "
                            "obstacle-absent control arm; release evaluator, physics seed 0, "
                            "one rollout per reference, one scene"),
            "obstacle_present_in_physics": True,
            "actual_ardy_samples": 0, "reused_archived_ardy_samples": len(POOL_SEEDS),
            "design": {
                "launch_plan": plan,
                "arms": list(ARMS),
                "execution_root": {
                    "sonic_root": str(Path(sonic_root).resolve()),
                    "branch": tracker.get("branch"),
                    "checkpoint": checkpoint,
                    "legacy_root_refused": str(LEGACY_SONIC_ROOT),
                    "why": ("CLAUDE.md 'Two tracker checkouts' and protocol section 2's "
                            "amendment: the add_table fix is reverted on the legacy checkout so "
                            "EXP-022A/024/028 keep manifest "
                            f"{EXP022A_CORE_MANIFEST_SHA256}; every EXP-030 arm, including the "
                            "absent control, runs on the patched worktree"),
                },
                "shared_overrides": shared_overrides(),
                "obstacle": {"x_m": OBSTACLE_X_M, "depth_m": OBSTACLE_DEPTH_M,
                             "width_m": OBSTACLE_WIDTH_M,
                             "half_width_m": OBSTACLE_HALF_WIDTH_M,
                             "width_source": "scene2motion.stepover_eval.step_scene",
                             "carried_as": ("per-motion table_pos / table_quat inside the motion "
                                            "pickle, because commands.py rewrites the table pose "
                                            "per environment on every reset (REPORT §47)")},
                "scene": {"start_xy_m": list(ROUTE_START_XY), "goal_xy_m": list(ROUTE_GOAL_XY),
                          "goal_tolerance_m": GOAL_TOLERANCE_M,
                          "corridor_half_width_m": OBSTACLE_HALF_WIDTH_M,
                          "sample_dt_s": SAMPLE_DT_S, "time_limit_s": None,
                          "evaluator_version": evaluator_version(),
                          "outcome_precedence": list(te.OUTCOMES)},
                "env_spacing_m": ENV_SPACING_M, "episode_length_s": EPISODE_LENGTH_S,
                "num_envs": exp022.CHUNK_SIZE, "physics_seed": PHYSICS_SEED,
                "release_termination_terms": list(RELEASE_TERMINATION_TERMS),
                "predictions": {"P1": P1_RULE, "P2": P2_RULE, "P3": P3_RULE},
            },
            "provenance": {
                "project": project, "protocol": protocol, "source_exp021": source["identity"],
                "exp022a": exp022a,
                "exp022a_achieved_rows": {"path": exp022a_rows.get("path"),
                                          "sha256": exp022a_rows.get("sha256")},
                "tracker": tracker, "initial_host_resource_gate": initial_gate,
            },
            "stages_complete": {"launch": False, "analysis": False},
            "resume_project_check": resume_project_check,
            "post_launch_revalidation": {}, "launches": {}, "host_gate_blocks": [],
        }
        _persist(output, receipt, started=started)
    else:
        receipt = old
        receipt["resume_project_check"] = resume_project_check
        receipt["provenance"]["tracker_at_resume"] = tracker
        receipt.setdefault("host_gate_blocks", [])

    model_cache: dict[str, Any] = {}

    def model_for_export() -> Any:
        if mj_model is not None:
            return mj_model
        if "model" not in model_cache:
            model_cache["model"] = _default_mj_model()
        return model_cache["model"]

    def revalidate(name: str) -> None:
        if source_fn(source_dir)["identity"] != source["identity"]:
            raise ValueError("EXP-021 source artifacts changed during SONIC execution")
        if dict(exp022a_identity_fn(exp022a_dir)) != exp022a:
            raise ValueError("EXP-022A artifacts changed during SONIC execution")
        current_tracker = dict(tracker_identity_fn(sonic_root))
        if bound_tracker_identity(current_tracker) != bound_tracker_identity(tracker):
            raise ValueError("SONIC checkout/checkpoint/config changed during execution")
        git_check = validate_project_recheck(
            project, project_identity(code_state_fn=code_state_fn), output)
        receipt["post_launch_revalidation"][name] = {
            "source_exp021_unchanged": True, "exp022a_unchanged": True,
            "tracker_bound_identity_unchanged": True,
            "tracker_dirty_paths_now": current_tracker.get("dirty_paths"), "project": git_check}

    def run_launches() -> None:
        for spec in plan:
            receipt["stage"] = f"launching_{spec['name']}"
            _persist(output, receipt, started=started)
            pkl = ensure_motion_pkl(spec, source["clips"], output, export_fn=export_fn,
                                    mj_model=model_for_export())
            if spec["arm"] == "absent" and export_fn is write_motion_pkl:
                expected = e28.EXP022A_MOTION_PKL_SHA256[f"chunk{spec['chunk']:02d}_seed0"]
                observed = _sha256(pkl)
                if observed != expected:
                    raise ValueError(f"{spec['name']} motion pickle {observed} differs from "
                                     f"EXP-022A's {expected}")
            try:
                record, _ = run_or_resume_launch(spec, pkl, output, launch_fn=launch_fn,
                                                 timeout_s=timeout_s, sonic_root=sonic_root,
                                                 checkpoint=checkpoint,
                                                 host_gate_fn=host_gate_fn, isaac_fn=isaac_fn)
            except host_gate.HostResourceGateFailed as exc:
                receipt["host_gate_blocks"].append({
                    "launch": spec["name"], "note": "blocked_host_gate", "error": str(exc),
                    "at_unix_s": time.time()})
                receipt["stage"] = f"blocked_host_gate_{spec['name']}"
                _persist(output, receipt, started=started)
                raise CampaignPaused(f"blocked_host_gate: {exc}") from exc
            receipt["launches"][spec["name"]] = record
            if not record.get("recovered_or_resumed"):
                revalidate(spec["name"])
            _persist(output, receipt, started=started)

    try:
        stages = [stage] if stage != "all" else ["launch", "analyze"]
        for current in stages:
            if current == "launch":
                run_launches()
                receipt["stages_complete"]["launch"] = True
                _persist(output, receipt, started=started)
            elif current == "analyze":
                if not receipt["stages_complete"]["launch"]:
                    raise CampaignPaused("analysis requires the six completed launches "
                                         "(run --stage launch first)")
                receipt["stage"] = "analysis"
                _persist(output, receipt, started=started)
                rollouts_by_arm: dict[str, dict[str, Any]] = {}
                for spec in plan:
                    _, rollouts = load_completed_launch(spec, output)
                    rollouts_by_arm.setdefault(spec["arm"], {}).update(
                        {r.motion_key: r for r in rollouts})
                expected_keys = {f"s{seed}" for seed in POOL_SEEDS}
                if (set(rollouts_by_arm) != set(ARM_NAMES)
                        or any(set(value) != expected_keys for value in rollouts_by_arm.values())):
                    raise ValueError("launch archives do not cover all 64 motions in every arm")
                scorer = collision_fn if collision_fn is not None else CollisionCache()
                rows = build_rows(rollouts_by_arm, collision_fn=scorer)
                _persist(output, receipt, started=started, rows=rows)
                summary = summarise(rows, exp022a_rows["rows"])
                receipt["stages_complete"]["analysis"] = True
                receipt.update({
                    "status": "complete", "complete": True, "stage": "complete",
                    "sonic_rollouts_requested": len(plan) * exp022.CHUNK_SIZE,
                    "sonic_rollouts_returned": sum(int(item.get("n_rollouts", 0))
                                                   for item in receipt["launches"].values()),
                    "summary": summary})
                _persist(output, receipt, started=started, rows=rows, summary=summary)
        if receipt["status"] != "complete":
            receipt["stage"] = f"{stage}_complete"
        _persist(output, receipt, started=started)
        return receipt
    except Exception as exc:
        if isinstance(exc, CampaignPaused):
            raise
        receipt.update({
            "schema": FAILURE_SCHEMA_VERSION, "status": "blocked", "complete": False,
            "blocked": True, "failed_stage": receipt.get("stage"),
            "error_type": type(exc).__name__, "error": str(exc)})
        _persist(output, receipt, started=started)
        if isinstance(exc, CampaignAbort):
            raise
        raise CampaignAbort(str(exc)) from exc


def _default_mj_model() -> Any:
    from scene2motion.robot import G1Body
    return G1Body(None).model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--resume", action="store_true",
                        help="continue an existing EXP-030 campaign directory")
    parser.add_argument("--source", default=str(SOURCE_OUT))
    parser.add_argument("--exp022a", default=str(EXP022A_OUT))
    parser.add_argument("--timeout-s", type=int, default=2400)
    parser.add_argument("--sonic-root", default=str(SONIC_EXP029_ROOT),
                        help="the patched tracker worktree every arm runs on; the legacy "
                             f"checkout {LEGACY_SONIC_ROOT} is refused")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the launch plan, the six command lines and the host gate; "
                             "write nothing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = run_campaign(stage=args.stage, out=args.out, resume=args.resume,
                               dry_run=args.dry_run, source_dir=args.source,
                               exp022a_dir=args.exp022a, timeout_s=args.timeout_s,
                               sonic_root=args.sonic_root)
    except (CampaignAbort, host_gate.HostResourceGateFailed) as exc:
        print(json.dumps({"status": "aborted", "error_type": type(exc).__name__,
                          "error": str(exc)}, indent=2))
        return 2
    if args.dry_run:
        print(json.dumps({
            "status": receipt["status"], "writes_performed": False,
            "protocol": receipt["protocol"],
            "tracker": receipt["tracker"],
            "execution": receipt["execution"],
            "tracker_add_table_fix": {
                key: receipt["tracker_add_table_fix"].get(key)
                for key in ("file", "sha256", "fix_present", "sensor_line", "sensor_indent",
                            "add_object_branch_line", "add_object_branch_indent", "problems")},
            "tracker_dirty_paths": receipt["tracker_dirty_paths"],
            "host_resource_gate": receipt["host_resource_gate"],
            "concurrent_isaac_processes": len(receipt["concurrent_isaac_processes"]),
            "launch_plan": [{key: value for key, value in spec.items()
                             if key not in ("seeds", "motion_keys", "table")}
                            | {"first_key": spec["motion_keys"][0],
                               "last_key": spec["motion_keys"][-1],
                               "table_pos": None if spec["table"] is None else spec["table"]["pos"],
                               "table_size": None if spec["table"] is None
                               else spec["table"]["size_xyz"]}
                            for spec in receipt["launch_plan"]],
            "commands": {name: shlex.join(command)
                         for name, command in receipt["commands"].items()},
        }, indent=2))
        return 0
    print(json.dumps({"status": receipt["status"], "stage": receipt.get("stage"),
                      "stages_complete": receipt.get("stages_complete"),
                      "launches": len(receipt.get("launches", {}))}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
