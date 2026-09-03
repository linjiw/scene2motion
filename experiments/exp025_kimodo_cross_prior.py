"""EXP-025 Part A: cross-prior timing and reference screen on the Kimodo-G1 prior.

Protocol: ``docs/ramp-exp025-kimodo-cross-prior-protocol.md`` (**preregistered 2026-09-03**,
including its 2026-09-03 amendment; its sha256 is bound into the receipt before the first
sample).  Part B (the 84-clip reduced capability audit) is explicitly out of scope.

The question.  exp021/exp023 measured, on the autoregressive ARDY-G1 prior, *when* the STEP
prompt's lift lands on a fixed route and whether it sits inside a bilateral no-support run long
enough for the calibrated reference screen to flag it.  Kimodo-G1-RP-v1 is a released
**offline (non-autoregressive)** diffusion prior for the same skeleton and the same released
LLM2Vec text embedding.  Running the byte-identical measurement on it separates "a property of
released text-conditioned G1 priors" from "a property of autoregressive rollout context".
Either outcome is reportable; the protocol predicts neither.

Design (locked before the first sample):

* 64 seeds 4700-4763, **both arms on the same seeds** -- ``step`` ("A person steps over an
  obstacle.") and ``walk`` ("A person walks forward at a steady pace.", the free nominal arm
  and elicitation floor, house rule 9).  128 samples exactly.
* 16 generation calls of B=8, each holding 4 seeds x 2 arms, so every same-seed step/walk pair
  shares one call and (under per-sample noise v2) one latent.  The protocol fixes "batches of
  8" and leaves the composition open; pairing inside the call is the exp024 precedent and is
  what makes the free-nominal comparison a paired one.
* 30 fps, 240 frames, a straight 7.2 m route, 100 DDIM steps, cfg (2.0, 2.0), first heading 0,
  post-processing bypassed (the recovered runner calls ``_generate`` + ``motion_rep.inverse``
  directly and never reaches ``Kimodo.__call__``'s post-processing).
* Constraint: the ``smooth_root_2d`` dense path only; root height and heading free -- the
  exp021 ``free`` contract.

The amendment that changes a number.  ``smooth_root_2d`` constrains the ADMM-**smoothed** root,
not the raw pelvis (smoother margin 0.06 m), so **route error is measured against the sample's
own ``smooth_root_pos``**, never against ``qpos``.  Measuring it against the pelvis reads about
6 cm high by construction and would bias the cross-prior comparison against Kimodo.  The raw
pelvis deviation is archived and reported *beside* it, labelled as the smoother's sway, so the
difference is visible rather than hidden.  ``smooth_root_pos`` is therefore archived per clip
next to the qpos.

Stages (each resumable; each a separate process, because the two halves of this campaign need
different interpreters -- generation needs ``kimodo`` (the Kimodo venv, which has no mujoco) and
scoring needs ``mujoco`` (``$S2M_PY``, which has no kimodo))::

    --stage generate   GPU, Kimodo venv: 16 B=8 calls; persists the empty ledger first
    --stage score      CPU, $S2M_PY: reference endpoints for all 128 clips
    --stage analyze    CPU, either: planned-denominator summary + the two decision rules
    --stage all        the three in order, each re-invoked under the interpreter that can run it

Three things this driver refuses to do, each of which would flatter the result:

* **Credit a robot that never arrived.**  ``BoxHeightProbe`` reports "collision-free" -- and
  its 0.40 m cap -- for a box the body never reaches, and Kimodo's ``smooth_root_2d`` following
  is looser than ARDY's by design (the protocol's Risks section: ~0.09 m against 1.4 cm).  So
  every clip's whole-body forward extent is measured first, unswept scan points are excluded
  from the elicitation argmax, and an obstacle the body never swept gets a null outcome, never
  a pass (``COVERAGE_RULE``).
* **Read the timing rule on the elicitation denominator.**  The ARDY comparator the rule is
  calibrated against ("80-86 % inside 2.0 s" = 40/49 and 42/49) is over clips with *any*
  positive lift, so that is rule 1's denominator; rule 2 keeps the >= 0.03 m elicited set it
  names in so many words.  A clip whose root never reaches its own lift position counts as
  not-early rather than being dropped, because dropping it can only raise the fraction
  (``TIMING_DENOMINATOR_RULE``, ``MISSING_EVENT_TIME_RULE``).
* **Brick the archive on a scoring bug.**  Only ``generate`` spends seeds, so only ``generate``
  can block a campaign.  A ``score``/``analyze`` failure is recorded and left resumable, so the
  fix re-scores the byte-identical archives instead of regenerating them.

The two stages also run under different interpreters by design, so the Kimodo runtime identity
is split: its ``checkout`` half (git state + source manifest) is revalidated at every stage,
while its ``interpreter`` half (``sys.version``, numpy, torch) is recorded per stage and never
compared across them.

Nothing here launches SONIC: this campaign is kinematic only, and no statement it produces is a
tracking outcome.  The 0.20 s rule is the *reference screen for predicted tracking cutoffs*
calibrated on ARDY-family clips; applying it to Kimodo references measures whether the same
kinematic signature is present, not whether SONIC would cut those rollouts off.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Both of these import only numpy/torch/stdlib, so this module imports cleanly under the Kimodo
# venv (no mujoco) *and* under $S2M_PY (no kimodo).  Everything mujoco-flavoured is imported
# lazily inside the score stage, and ``kimodo`` itself only inside the production runner factory.
from experiments.kimodo_recovered import kimodo_runner as kr  # noqa: E402
from scene2motion.constraints import ConstraintSpec  # noqa: E402
from scene2motion.host_gate import (  # noqa: E402
    ARDY_GENERATION_GATE,
    HostResourceGateFailed,
    host_resource_report,
    require_host_resources,
)


# --------------------------------------------------------------------------- locked design

SCHEMA_VERSION = "exp025-kimodo-cross-prior-v1"
FAILURE_SCHEMA_VERSION = "exp025-kimodo-cross-prior-failure-v1"
EXPERIMENT = "exp025_kimodo_cross_prior"
PROTOCOL_PATH = "docs/ramp-exp025-kimodo-cross-prior-protocol.md"
DEFAULT_OUT = ROOT / "outputs/exp025_kimodo_cross_prior"

MODEL_NAME = "Kimodo-G1-RP-v1"
DEVICE = "cuda:0"

FPS = 30.0
N_FRAMES = 240
ROUTE_LENGTH_M = 7.2
#: ARDY's frame-span convention (``cal.route_xz_for_speed``): the route spans N frames, so the
#: prescribed speed is length / ((N - 1) / fps).  The protocol's round "0.9 m/s over 8.0 s" is
#: the design intent and is bound beside it; both agree to 4 mm/s, and the route *array* -- a
#: straight 7.2 m line -- is geometrically identical to the ARDY family's, which is what makes
#: the scan window, the two obstacle centres and the box-height profile comparable.
DURATION_S = (N_FRAMES - 1) / FPS
NOMINAL_SPEED_MPS = ROUTE_LENGTH_M / DURATION_S
PROTOCOL_SPEED_MPS = 0.9
PROTOCOL_DURATION_S = N_FRAMES / FPS

DIFFUSION_STEPS = 100
CFG_WEIGHT = (2.0, 2.0)
CFG_TYPE = None                      # keep the checkpoint default ("separated")
FIRST_HEADING = 0.0
NOISE_STREAM_VERSION = 2

SEEDS = tuple(range(4700, 4764))
ARMS = ("step", "walk")
STEP = "A person steps over an obstacle."
WALK = "A person walks forward at a steady pace."
ARM_PROMPTS: Mapping[str, str] = {"step": STEP, "walk": WALK}

BATCH_SIZE = 8
CHUNK_SEED_COUNT = BATCH_SIZE // len(ARMS)          # 4 seeds x 2 arms per B=8 call
CHUNK_ROWS = CHUNK_SEED_COUNT * len(ARMS)
N_CHUNKS = len(SEEDS) // CHUNK_SEED_COUNT
N_ROWS = len(SEEDS) * len(ARMS)

#: The single native contract: the dense ``smooth_root_2d`` ground path, height and heading free.
CONSTRAINT_CHANNEL = "smooth_root_2d"
CONSTRAINT_CONTRACT = "free"
#: What the adapter writes (frame-index counts per named channel), computable without a model.
EXPECTED_ADAPTER_CHANNELS: Mapping[str, int] = {CONSTRAINT_CHANNEL: N_FRAMES}
#: What the model's mask must then carry: ``smooth_root_2d`` fills columns 0 and 2 of the
#: ``smooth_root_pos`` block at every frame and nothing else (kimodo_motionrep.py:242-251).
EXPECTED_CHANNEL_USAGE: Mapping[str, int] = {"smooth_root_pos": 2 * N_FRAMES}

#: Endpoint constants, held byte-equal to the ARDY family (asserted in the tests against
#: ``exp022_exact_tracking_bridge`` / ``analyze_e1a_placement``, which need mujoco to import).
OBSTACLE_DEPTH_M = 0.20
OBSTACLES: tuple[tuple[str, float], ...] = (("staged", 1.2), ("unstaged", 3.6))
GRADED_HEIGHTS_M = (0.03, 0.05, 0.08, 0.12, 0.20, 0.30)
SCAN_POINTS = 120
ELICITATION_MIN_M = 0.03
#: The forward axis of the MuJoCo export (kimodo z -> mujoco x), asserted per clip by
#: ``route_fidelity``'s ``forward_axis_dominant`` and globally by the score stage.
FORWARD_AXIS = (1.0, 0.0, 0.0)
#: Coverage guard.  ``BoxHeightProbe`` reports "collision-free" -- and its 0.40 m cap -- for a
#: box the robot simply never reaches, so a clip that stalls short would otherwise be scored as
#: maximally elicited and as clearing every graded height at a position it never visited.  The
#: protocol's own Risks section names the condition that makes this live here and not in the
#: ARDY family: Kimodo's ``smooth_root_2d`` following is ~0.09 m on indoor_nav against ARDY's
#: 1.4 cm.  So every clearance claim is confined to the ground the body actually swept.
COVERAGE_RULE = (
    "a box centred at x with depth d counts as swept only if [x - d/2, x + d/2] lies inside "
    "the forward interval the whole-body collision envelope (body margin included) covers over "
    "all frames; clearance at an unswept centre is recorded as 'not reached' (null), never as "
    "a success, and unswept scan points are excluded from the elicitation argmax")
#: The calibrated reference screen (exp016 receipt, frozen before exp021) and the post hoc cut
#: reported beside it.  Both flag ``max_unsupported_run_s > threshold``.  The thresholds are in
#: SECONDS, so they are fps-free and transfer unchanged from 25 fps to Kimodo's 30 fps.
PRIMARY_GATE_S = 0.2
SECONDARY_GATE_S = 0.28
THRESHOLD_RECEIPT_PATH = ROOT / "outputs/exp016_threshold_calibration/receipt.json"
THRESHOLD_RECEIPT_SHA256 = "f6dba8be84a9d5d0b76c8114d4b93b1707bc1bb8a6fec1a26a22aa1780a6e9bf"

#: The ARDY comparison window: "inside the first 50 frames" at 25 fps is t < 2.0 s, and the
#: committed analyser reports 40/49 (root crossing) and 42/49 (nominal) inside it.
EARLY_WINDOW_S = 2.0
ARDY_REFERENCE_EARLY_FRACTION = {
    "root_crossing": [40, 49],
    "nominal_speed": [42, 49],
    "source": "outputs/analysis_event_frames (experiments/analyze_event_frames.py)",
    "denominator": "the 49 exp021 clips with any positive whole-body-clearable lift",
}

#: Why the two preregistered rules count over different sets, spelled out in every receipt.
TIMING_DENOMINATOR_RULE = (
    "the lift-time distribution is reported over the clips that HAVE a lift position -- any "
    "positive whole-body-clearable lift -- because that is the ARDY comparator this campaign "
    "is calibrated against (40/49 root crossing, 42/49 nominal: the '80-86 % inside 2.0 s' the "
    "protocol quotes, over the 49 exp021 clips with any positive lift).  The >= 0.03 m "
    "elicited fraction and the rate over all assigned trials are reported beside it.  The "
    "screen rule keeps the elicited denominator, which the protocol names explicitly ('>= 80 % "
    "of Kimodo's elicited clips')")
#: A clip whose root never reaches its own lift position is the latest event there is, so it is
#: counted as not-early rather than dropped: dropping it can only raise the fraction.
MISSING_EVENT_TIME_RULE = (
    "a clip with a lift position but no defined event time counts as NOT within the first "
    "2.0 s; it is never excluded from the denominator, because a root that never reaches the "
    "lift position certainly did not reach it inside the window")


#: Decision rules, quoted from the protocol and evaluated mechanically into the receipt.
TIMING_GENERALISES_MIN_FRACTION = 0.7
TIMING_ROLLOUT_MAX_FRACTION = 0.4
SCREEN_GENERALISES_MIN_FRACTION = 0.8

STAGES = ("generate", "score", "analyze")

# External pins (CLAUDE.md "External pins"; re-derived and revalidated in every receipt).
KIMODO_ROOT = Path(os.environ.get("KIMODO_ROOT", "/home/linjiw/kimodo"))
ARDY_ROOT = Path(os.environ.get("ARDY_ROOT", "/home/linjiw/ardy"))
PINNED_KIMODO_COMMIT = "1aece8c124d73d255ceff5086d983b844c9f4e94"
PINNED_KIMODO_HF_REVISION = "3020ad8c419c244e0429d360163730c63c4ed011"
PINNED_KIMODO_CHECKPOINT_SHA256 = (
    "e18c1de73e2ce17a107b06d85155fbbc5debe68eb35455aa5b033e6ddbe056a5"
)
#: Byte-identical in the ARDY and Kimodo checkouts, so ``G1Body`` scores both families through
#: the same released MJCF (the protocol's prerequisite 3).
PINNED_G1_XML_SHA256 = "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
KIMODO_G1_XML = KIMODO_ROOT / "kimodo/assets/skeletons/g1skel34/xml/g1.xml"
ARDY_G1_XML_PATH = ARDY_ROOT / "ardy/assets/skeletons/g1skel34/xml/g1.xml"
KIMODO_SANITIZE_PATH = KIMODO_ROOT / "kimodo/sanitize.py"
KIMODO_TEXT_CACHE = KIMODO_ROOT / "data/indoor_nav_1k/text_cache.npz"
ARDY_TEXT_CACHE = ROOT / "outputs/text_cache.npz"
CAMPAIGN_TEXT_CACHE_NAME = "kimodo_text_cache.npz"

#: The two LLM2Vec wrapper sources whose equality licenses copying ARDY's cached STEP vector.
ARDY_LLM2VEC_WRAPPER = ARDY_ROOT / "ardy/model/llm2vec/llm2vec_wrapper.py"
KIMODO_LLM2VEC_WRAPPER = KIMODO_ROOT / "kimodo/model/llm2vec/llm2vec_wrapper.py"
ARDY_LOAD_MODEL = ARDY_ROOT / "ardy/model/load_model.py"
KIMODO_LOAD_MODEL = KIMODO_ROOT / "kimodo/model/load_model.py"
EXPECTED_LLM2VEC_KWARGS: Mapping[str, Any] = {
    "base_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    "peft_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    "dtype": "bfloat16",
    "llm_dim": 4096,
    "device": "auto",
}

#: Repository sources whose content is bound at generation and revalidated at every later stage.
SOURCE_FILES = (
    PROTOCOL_PATH,
    "env.sh",
    "experiments/exp025_kimodo_cross_prior.py",
    "experiments/kimodo_recovered/kimodo_runner.py",
    "experiments/analyze_trackability_contract.py",
    "experiments/analyze_e1a_placement.py",
    "experiments/analyze_event_frames.py",
    "experiments/calibrate_ramp_route_phase.py",
    "scene2motion/constraints.py",
    "scene2motion/host_gate.py",
    "scene2motion/robot.py",
    "scene2motion/stepover_eval.py",
)
#: External (non-repository) sources bound by absolute path.
EXTERNAL_SOURCE_FILES: tuple[Path, ...] = (
    KIMODO_SANITIZE_PATH,
    KIMODO_LLM2VEC_WRAPPER,
    ARDY_LLM2VEC_WRAPPER,
    KIMODO_LOAD_MODEL,
    ARDY_LOAD_MODEL,
    KIMODO_ROOT / "kimodo/model/kimodo_model.py",
    KIMODO_ROOT / "kimodo/model/diffusion.py",
    KIMODO_ROOT / "kimodo/motion_rep/reps/kimodo_motionrep.py",
    KIMODO_ROOT / "kimodo/motion_rep/smooth_root.py",
    KIMODO_ROOT / "kimodo/exports/mujoco.py",
    KIMODO_G1_XML,
    ARDY_G1_XML_PATH,
)

# Interpreters: generation needs ``kimodo``, scoring needs ``mujoco``, and no local interpreter
# has both.  ``--stage all`` re-invokes itself under whichever one a stage requires.
KIMODO_PY = Path(os.environ.get("KIMODO_PY", str(KIMODO_ROOT / ".venv/bin/python")))
SCORING_PY = Path(os.environ.get("S2M_PY", str(ARDY_ROOT / ".venv/bin/python")))
STAGE_REQUIREMENTS: Mapping[str, str | None] = {
    "generate": "kimodo", "score": "mujoco", "analyze": None}
STAGE_INTERPRETERS: Mapping[str, Path] = {"generate": KIMODO_PY, "score": SCORING_PY}

if len(SEEDS) != 64 or N_ROWS != 128 or N_CHUNKS != 16 or CHUNK_ROWS != BATCH_SIZE:
    raise RuntimeError("EXP-025 batch plan drifted from 64 seeds x 2 arms in 16 B=8 calls")
if abs(NOMINAL_SPEED_MPS - PROTOCOL_SPEED_MPS) > 0.01:
    raise RuntimeError("EXP-025 route speed drifted from the protocol's 0.9 m/s")
if abs(PROTOCOL_SPEED_MPS * PROTOCOL_DURATION_S - ROUTE_LENGTH_M) > 1e-9:
    raise RuntimeError("EXP-025 route length is not 0.9 m/s x 8.0 s")
if int(kr.NOISE_STREAM_VERSION) != NOISE_STREAM_VERSION:
    raise RuntimeError("the recovered Kimodo runner is not on noise stream v2")


class CampaignAbort(RuntimeError):
    """Fail-closed stop after every available piece of evidence has been made durable."""


# --------------------------------------------------------- hashing and durable-write helpers
#
# These are byte-for-byte the algorithms of ``calibrate_ramp_route_phase`` (``_canonical_json``,
# ``_json_hash``, ``_sha256``, ``_array_hash``, ``_identity``, ``_git_state``, ``_atomic_write``,
# ``_write_json``, ``_write_jsonl``, ``_persist_qpos``).  They are re-implemented here, and only
# here, because ``calibrate_ramp_route_phase`` imports ``scene2motion.robot`` and therefore
# mujoco, which the Kimodo venv does not have -- while the generation stage must run under it.
# ``tests/test_exp025_kimodo_cross_prior.py`` asserts each one agrees with ``cal``'s on sample
# inputs, so drift is a test failure, not a silent divergence.


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _identity(schema: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"schema": str(schema), "fields": dict(fields)}
    normalized = json.loads(_canonical_json(_json_safe(payload)))
    return {**normalized, "sha256": _json_hash(normalized)}


def _array_hash(arrays: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    found = False
    for name in sorted(arrays):
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            continue
        found = True
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(_canonical_json(list(array.shape)).encode())
        digest.update(array.tobytes(order="C"))
    if not found:
        raise ValueError("array payload contains no ndarray values")
    return digest.hexdigest()


def _sample_hash(sample: Mapping[str, Any]) -> str:
    return _array_hash({str(name): value for name, value in sample.items()})


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _atomic_write(path: Path, writer: Callable[[Any], None]) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False)
               + "\n").encode()
    _atomic_write(Path(path), lambda handle: handle.write(payload))


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(_canonical_json(_json_safe(dict(value))) + "\n"
                      for value in values).encode()
    _atomic_write(Path(path), lambda handle: handle.write(payload))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise CampaignAbort(f"invalid JSONL artifact {path}: {exc}") from exc


def _persist_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    _atomic_write(Path(path), lambda handle: np.savez(handle, **normalized))


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise CampaignAbort(f"invalid array archive {path}: {exc}") from exc


def _git_state(repo: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True,
                                         stderr=subprocess.DEVNULL).splitlines()
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo,
                                       stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("could not identify git state") from exc
    return {"commit": commit, "dirty": bool(status), "status": status,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}


# --------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    """Wilson 95 % score interval -- byte-identical to ``analyze_trackability_contract.wilson``."""
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def rate(k: int, n: int) -> dict[str, Any]:
    """A rate with its planned denominator and Wilson 95 % interval; ``n=0`` is never a pass."""
    k, n = int(k), int(n)
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"invalid rate {k}/{n}")
    lo, hi = wilson(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": (k / n if n else None),
            "wilson95": [lo, hi] if n else [None, None]}


def quantiles(values: Sequence[float],
              qs: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> dict[str, Any]:
    array = np.asarray([v for v in values if v is not None], dtype=float)
    if array.size == 0:
        return {"n": 0, "mean": None, "sd": None,
                **{f"q{q:g}": None for q in qs}, "min": None, "max": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else None,
        **{f"q{q:g}": float(np.quantile(array, float(q))) for q in qs},
        "min": float(array.min()), "max": float(array.max()),
    }


# ------------------------------------------------------------------------- locked plans


def locked_row_plan() -> list[dict[str, Any]]:
    """Chunk-major 128-row plan: chunk c holds seeds 4700+4c..4703+4c x (step, walk)."""
    rows: list[dict[str, Any]] = []
    for chunk in range(N_CHUNKS):
        seeds = SEEDS[chunk * CHUNK_SEED_COUNT:(chunk + 1) * CHUNK_SEED_COUNT]
        for local, (seed, arm) in enumerate((s, a) for s in seeds for a in ARMS):
            rows.append({
                "row_index": len(rows),
                "chunk": chunk,
                "chunk_name": f"chunk{chunk:02d}",
                "batch_position": local,
                "seed": int(seed),
                "arm": arm,
                "prompt": ARM_PROMPTS[arm],
                "archive_key": f"s{seed}_{arm}",
            })
    if len(rows) != N_ROWS:
        raise RuntimeError("locked row plan does not hold exactly 128 rows")
    return rows


def locked_chunk_plan(plan: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Sixteen B=8 generation calls; no same-seed step/walk pair crosses a call."""
    rows = list(plan if plan is not None else locked_row_plan())
    chunks: list[dict[str, Any]] = []
    for chunk in range(N_CHUNKS):
        members = [row for row in rows if int(row["chunk"]) == chunk]
        seeds = list(dict.fromkeys(int(row["seed"]) for row in members))
        if len(members) != CHUNK_ROWS or len(seeds) != CHUNK_SEED_COUNT:
            raise RuntimeError(f"chunk {chunk} is not 4 seeds x 2 arms")
        if [row["arm"] for row in members] != list(ARMS) * CHUNK_SEED_COUNT:
            raise RuntimeError(f"chunk {chunk} arm order drifted")
        chunks.append({
            "chunk": chunk,
            "name": f"chunk{chunk:02d}",
            "seeds": seeds,
            "row_indices": [int(row["row_index"]) for row in members],
            "archive_keys": [str(row["archive_key"]) for row in members],
            "prompts": [str(row["prompt"]) for row in members],
            "batch_size": len(members),
            "rows": members,
        })
    return chunks


def route_xz() -> np.ndarray:
    """The straight 7.2 m route, ARDY-frame (column 0 lateral, column 1 forward)."""
    return np.stack([np.zeros(N_FRAMES, dtype=float),
                     np.linspace(0.0, ROUTE_LENGTH_M, N_FRAMES, dtype=float)], axis=-1)


def campaign_spec(route: np.ndarray | None = None) -> ConstraintSpec:
    """The one locked contract: dense ``smooth_root_2d`` only, height and heading free."""
    route = route_xz() if route is None else np.asarray(route, dtype=float)
    return ConstraintSpec(root_xz=route, heading=None, root_y=None,
                          first_heading=FIRST_HEADING)


def spec_sha256(spec: ConstraintSpec) -> str:
    return _json_hash({
        "root_xz": _array_hash({"root_xz": np.asarray(spec.root_xz, dtype=float)}),
        "heading": None if spec.heading is None else _array_hash(
            {"heading": np.asarray(spec.heading, dtype=float)}),
        "root_y": None if spec.root_y is None else _array_hash(
            {"root_y": np.asarray(spec.root_y, dtype=float)}),
        "first_heading": spec.first_heading,
        "n_frames": int(spec.T),
    })


def static_channel_usage(spec: ConstraintSpec) -> dict[str, int]:
    """Which Kimodo filler keys the adapter writes, counted without a model (CPU, no runner).

    This is where ``smooth_root_2d`` is pinned: ``KimodoConstraintSet`` renames ARDY's
    ``root_2d`` to it, and a campaign that silently wrote ``root_2d`` would constrain nothing.
    """
    keys = ("smooth_root_2d", "global_root_heading", "root_y_pos",
            "global_joints_rots", "global_joints_positions")
    data: dict[str, list] = {key: [] for key in keys}
    index: dict[str, list] = {key: [] for key in keys}
    kr.KimodoConstraintSet(spec, root_idx=0, device="cpu").update_constraints(data, index)
    return {key: int(sum(len(entry) for entry in index[key]))
            for key in keys if data[key]}


def _actual_channel_usage(runner: Any, spec: ConstraintSpec) -> dict[str, int]:
    """The mask the model itself would see, counted per feature block (needs a loaded model)."""
    _observed, mask = kr.build_conditions(runner.model, spec, runner.device)
    return {str(key): int(value)
            for key, value in kr.channel_usage(runner.model, mask).items() if value}


# ---------------------------------------------------- prompt embeddings and their provenance


def load_sanitize_text(path: Path = KIMODO_SANITIZE_PATH) -> Callable[..., str]:
    """Kimodo's own ``sanitize_text``, loaded from its file.

    ``kimodo/sanitize.py`` imports nothing, so the vendor function is usable byte-exactly from
    an interpreter that has no ``kimodo`` package -- which is what lets the CPU tests and the
    scoring interpreter check the cache-key scheme without the Kimodo venv.
    """
    path = Path(path)
    spec = importlib.util.spec_from_file_location("exp025_kimodo_sanitize", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load Kimodo's sanitize_text from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sanitize_text


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = list(node.body)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _drop_method(tree: ast.AST, class_name: str, method_name: str) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            node.body = [item for item in node.body
                         if not (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                                 and item.name == method_name)] or [ast.Pass()]
    return tree


def _normalized_wrapper_dump(path: Path) -> str:
    """The wrapper's AST with docstrings and ``LLM2VecEncoder.to`` removed.

    The protocol claims the two wrappers "differ only in docstring and device helper".  This
    normalisation is exactly that claim made checkable: everything that touches the embedding
    -- the preset plumbing, ``__call__``, the batch_size=1 repeatability comment's code, the
    pooling and the dtype -- must survive it identically.
    """
    tree = ast.parse(Path(path).read_text())
    return ast.dump(_strip_docstrings(_drop_method(tree, "LLM2VecEncoder", "to")),
                    annotate_fields=True, include_attributes=False)


def _text_encoder_preset(path: Path) -> dict[str, Any]:
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "TEXT_ENCODER_PRESETS"
                   for t in node.targets):
            continue
        try:
            presets = ast.literal_eval(node.value)
            return dict(presets["llm2vec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"unreadable TEXT_ENCODER_PRESETS in {path}: {exc}") from exc
    raise ValueError(f"no TEXT_ENCODER_PRESETS in {path}")


def encoder_equivalence_report(
    *,
    ardy_wrapper: Path = ARDY_LLM2VEC_WRAPPER,
    kimodo_wrapper: Path = KIMODO_LLM2VEC_WRAPPER,
    ardy_load_model: Path = ARDY_LOAD_MODEL,
    kimodo_load_model: Path = KIMODO_LOAD_MODEL,
) -> dict[str, Any]:
    """Assert the protocol's licence for copying ARDY's cached STEP embedding into Kimodo.

    Two facts have to hold, and both are checked rather than asserted in prose: the two
    ``LLM2VecEncoder`` wrappers are the same code apart from the module docstring and the
    ``to()`` device helper, and both models instantiate the *same* LLM2Vec preset (the same
    base + PEFT checkpoints, bfloat16, llm_dim 4096).  If either fails, the two caches hold
    different quantities and the campaign must not run.
    """
    ardy_dump = _normalized_wrapper_dump(ardy_wrapper)
    kimodo_dump = _normalized_wrapper_dump(kimodo_wrapper)
    wrappers_equal = ardy_dump == kimodo_dump
    ardy_preset = _text_encoder_preset(ardy_load_model)
    kimodo_preset = _text_encoder_preset(kimodo_load_model)
    ardy_kwargs = dict(ardy_preset.get("kwargs", {}))
    kimodo_kwargs = dict(kimodo_preset.get("kwargs", {}))
    kwargs_equal = ardy_kwargs == kimodo_kwargs == dict(EXPECTED_LLM2VEC_KWARGS)
    report = {
        "schema": "exp025-llm2vec-encoder-equivalence-v1",
        "wrapper_sources": {
            "ardy": {"path": str(ardy_wrapper), "sha256": _sha256(ardy_wrapper)},
            "kimodo": {"path": str(kimodo_wrapper), "sha256": _sha256(kimodo_wrapper)},
        },
        "normalisation": ("module docstring and LLM2VecEncoder.to removed; compared as an AST "
                          "dump with docstrings stripped"),
        "normalized_ast_sha256": {
            "ardy": hashlib.sha256(ardy_dump.encode()).hexdigest(),
            "kimodo": hashlib.sha256(kimodo_dump.encode()).hexdigest(),
        },
        "wrappers_equal_after_normalisation": bool(wrappers_equal),
        "preset_kwargs": {"ardy": ardy_kwargs, "kimodo": kimodo_kwargs,
                          "expected": dict(EXPECTED_LLM2VEC_KWARGS)},
        "preset_kwargs_equal": bool(kwargs_equal),
        "expected_difference": {
            "target": {"ardy": ardy_preset.get("target"), "kimodo": kimodo_preset.get("target")},
            "note": "only the import path of the wrapper class differs, as expected",
        },
        "load_model_sources": {
            "ardy": {"path": str(ardy_load_model), "sha256": _sha256(ardy_load_model)},
            "kimodo": {"path": str(kimodo_load_model), "sha256": _sha256(kimodo_load_model)},
        },
    }
    report["equivalent"] = bool(wrappers_equal and kwargs_equal)
    if not report["equivalent"]:
        raise ValueError(
            "ARDY and Kimodo LLM2Vec encoders are not equivalent "
            f"(wrappers_equal={wrappers_equal}, preset_kwargs_equal={kwargs_equal}); "
            "copying the cached STEP embedding is not licensed")
    return report


def _cache_entry(cache_path: Path, key: str, prompt: str) -> np.ndarray:
    path = Path(cache_path)
    if not path.is_file():
        raise ValueError(f"prompt cache is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as cache:
            if key not in cache.files:
                raise ValueError(f"cached embedding is missing for {prompt!r} in {path}")
            value = np.array(cache[key], copy=True)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid prompt cache {path}: {exc}") from exc
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[-1] != EXPECTED_LLM2VEC_KWARGS["llm_dim"]:
        raise ValueError(f"cached embedding for {prompt!r} has shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"cached embedding for {prompt!r} is not finite")
    return array


def build_campaign_text_cache(
    out_path: Path,
    *,
    ardy_cache: Path = ARDY_TEXT_CACHE,
    kimodo_cache: Path = KIMODO_TEXT_CACHE,
    sanitize_fn: Callable[..., str] | None = None,
    encoder_equivalence_fn: Callable[[], Mapping[str, Any]] = encoder_equivalence_report,
) -> dict[str, Any]:
    """Write the two-prompt campaign-local Kimodo cache and bind both content hashes.

    STEP is *copied* out of ARDY's cache: Kimodo keys entries by ``sha1(sanitize_text(prompt))``
    and, for this prompt, ``sanitize_text`` is the identity, so ARDY's raw-text key and Kimodo's
    canonical key are the same string.  That coincidence is checked, not assumed -- if
    ``sanitize_text`` ever changed the prompt, the copy would land under a key Kimodo never
    looks up and generation would silently run on the wrong embedding.  WALK comes from
    Kimodo's own 300-prompt indoor_nav cache and is never copied.
    """
    sanitize = sanitize_fn or load_sanitize_text()
    equivalence = dict(encoder_equivalence_fn())
    entries: dict[str, np.ndarray] = {}
    prompts: dict[str, Any] = {}
    for arm, prompt, source in (("step", STEP, Path(ardy_cache)),
                                ("walk", WALK, Path(kimodo_cache))):
        sanitized = sanitize(prompt)
        canonical_key = kr._raw_key(sanitized)
        raw_key = kr._raw_key(prompt)
        if arm == "step":
            if sanitized != prompt or canonical_key != raw_key:
                raise ValueError(
                    "Kimodo's canonical cache key no longer coincides with ARDY's raw-text key "
                    f"for {prompt!r}; copying the ARDY embedding is not valid")
            value = _cache_entry(source, raw_key, prompt)
        else:
            value = _cache_entry(source, canonical_key, prompt)
        entries[canonical_key] = value
        prompts[arm] = {
            "prompt": prompt,
            "sanitized": sanitized,
            "sanitize_is_identity": bool(sanitized == prompt),
            "canonical_key_sha1": canonical_key,
            "raw_key_sha1": raw_key,
            "keys_coincide": bool(canonical_key == raw_key),
            "source_path": str(source),
            "source_file_sha256": _sha256(source),
            "copied_from_ardy_cache": bool(arm == "step"),
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "content_sha256": _array_hash({canonical_key: value}),
        }
    if len(entries) != 2:
        raise ValueError("campaign text cache did not resolve two distinct prompt keys")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _persist_arrays(out_path, entries)
    return _identity("exp025-kimodo-prompt-cache-v1", {
        "path": str(out_path),
        "file_sha256": _sha256(out_path),
        "n_entries": len(entries),
        "content_sha256": _array_hash(entries),
        "key_scheme": "sha1(sanitize_text(prompt)); sha1(raw) accepted as a fallback",
        "sanitize_source": {"path": str(KIMODO_SANITIZE_PATH),
                            "sha256": _sha256(KIMODO_SANITIZE_PATH)},
        "prompts": prompts,
        "encoder_equivalence": equivalence,
        "encoder_loaded": False,
        "note": ("no text encoder runs in this campaign; the cache is authoritative and "
                 "KimodoRunner is constructed with text_encoder=False"),
    })


def verify_runner_text_cache(runner: Any, identity: Mapping[str, Any]) -> dict[str, Any]:
    """The embeddings the runner actually holds must byte-match the ones just bound."""
    prompts = dict(identity.get("fields", {}).get("prompts", {}))
    if set(prompts) != set(ARMS):
        raise ValueError("prompt-cache identity does not cover both arms")
    checked: dict[str, Any] = {}
    for arm, record in prompts.items():
        key = str(record["canonical_key_sha1"])
        try:
            resolved = runner._cache_key(str(record["prompt"]))
            memory = np.asarray(runner._text_cache[key], dtype=np.float32)
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError(f"runner does not hold the cached {arm} embedding: {exc}") from exc
        if resolved != key:
            raise ValueError(
                f"runner resolves the {arm} prompt to {resolved!r}, expected {key!r}")
        content = _array_hash({key: memory})
        if content != record["content_sha256"]:
            raise ValueError(f"in-memory {arm} embedding does not byte-match the campaign cache")
        checked[arm] = {"cache_key_sha1": key, "content_sha256": content,
                        "shape": [int(d) for d in memory.shape]}
    return {"prompts": checked, "runner_memory_byte_matches_cache": True}


# ----------------------------------------------------------------- provenance and identities


def _source_hashes(repo: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        digest = _sha256(Path(repo) / relative)
        if digest is None:
            raise ValueError(f"required EXP-025 source is missing: {relative}")
        hashes[relative] = digest
    return hashes


def external_source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in EXTERNAL_SOURCE_FILES:
        digest = _sha256(path)
        if digest is None:
            raise ValueError(f"required external EXP-025 source is missing: {path}")
        hashes[str(path)] = digest
    return hashes


def kimodo_checkout_identity() -> dict[str, Any]:
    """The **stage-invariant** half of the Kimodo runtime: checkout and Python sources.

    Nothing in here depends on which interpreter is running, which is what makes it
    revalidatable across stages.  Deliberately does not import ``kimodo`` (or mujoco): the
    scoring stage runs under an interpreter that cannot import Kimodo.
    """
    package_root = KIMODO_ROOT / "kimodo"
    source_files = sorted(package_root.rglob("*.py"))
    if not source_files:
        raise ValueError("could not enumerate Kimodo Python runtime sources")
    sources: dict[str, str] = {}
    for path in source_files:
        digest = _sha256(path)
        if digest is None:
            raise ValueError(f"Kimodo runtime source disappeared: {path}")
        sources[path.relative_to(package_root).as_posix()] = digest
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=KIMODO_ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        tracked_status = subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"], cwd=KIMODO_ROOT, text=True,
            stderr=subprocess.DEVNULL).splitlines()
        tracked_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"],
                                               cwd=KIMODO_ROOT, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("could not bind the external Kimodo git identity") from exc
    return _identity("exp025-kimodo-checkout-identity-v1", {
        "kimodo_package_root": str(package_root),
        "kimodo_git_commit": commit,
        "kimodo_tracked_status": tracked_status,
        "kimodo_tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "kimodo_python_source_count": len(sources),
        "kimodo_python_source_manifest_sha256": _json_hash(sources),
        "kimodo_python_source_sha256": sources,
    })


def interpreter_runtime_identity() -> dict[str, Any]:
    """The **per-stage** half: which interpreter and numerical stack ran this stage.

    This campaign runs its stages under *different* interpreters by design -- generation needs
    the Kimodo venv (which has no mujoco) and scoring needs ``$S2M_PY`` (which has no kimodo)
    -- so ``sys.version``/``numpy``/``torch`` differ between them as a matter of course.  They
    are therefore recorded additively per stage and **never** compared across stages: an
    identity that folded them in with the checkout could not match after generation and would
    abort every scoring run once the 64 reserved seeds had already been spent.
    """
    try:
        import torch

        torch_version: str | None = str(torch.__version__)
        torch_cuda_version: str | None = torch.version.cuda
    except ImportError:              # an analyze-only interpreter needs neither torch nor mujoco
        torch_version = None
        torch_cuda_version = None
    return _identity("exp025-interpreter-runtime-identity-v1", {
        "executable": sys.executable,
        "python": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
    })


def kimodo_runtime_identity() -> dict[str, Any]:
    """Bind the Kimodo checkout and the interpreter that is running, kept separable.

    ``checkout`` is stage-invariant and is revalidated byte-for-byte at every later stage;
    ``interpreter`` is stage-local evidence and is only ever recorded.
    """
    return {
        "schema": "exp025-kimodo-runtime-identity-v2",
        "checkout": kimodo_checkout_identity(),
        "interpreter": interpreter_runtime_identity(),
        "note": ("only 'checkout' is revalidated across stages; 'interpreter' differs between "
                 "the generation (kimodo venv) and scoring ($S2M_PY) interpreters by design"),
    }


def _checkout_part(identity: Mapping[str, Any] | None) -> Any:
    """The stage-invariant part of a runtime identity, tolerant of injected fakes."""
    return dict(identity or {}).get("checkout")


def _interpreter_part(identity: Mapping[str, Any] | None) -> Any:
    return dict(identity or {}).get("interpreter")


def physical_model_identity() -> dict[str, Any]:
    """The released G1 MJCF that ``G1Body`` loads, and the byte-identical copy Kimodo ships."""
    ardy_sha = _sha256(ARDY_G1_XML_PATH)
    kimodo_sha = _sha256(KIMODO_G1_XML)
    if ardy_sha is None or kimodo_sha is None:
        raise ValueError("released G1 MJCF is missing from one of the two checkouts")
    if ardy_sha != kimodo_sha:
        raise ValueError("the ARDY and Kimodo G1 MJCFs are no longer byte-identical")
    return _identity("exp025-physical-model-identity-v1", {
        "path": str(ARDY_G1_XML_PATH),
        "sha256": ardy_sha,
        "kimodo_copy_path": str(KIMODO_G1_XML),
        "kimodo_copy_sha256": kimodo_sha,
        "role": "physical-foot-geometries-and-forward-kinematics",
    })


def kimodo_generator_identity(runner: Any) -> dict[str, Any]:
    """Snapshot identity derived from the *loaded* model's own statistics folder."""
    from huggingface_hub.constants import HF_HUB_CACHE

    model_name = str(runner.model_name)
    try:
        stats_folder = Path(runner.model.motion_rep.body_stats.folder)
    except (AttributeError, TypeError) as exc:
        raise ValueError("runner does not expose its loaded motion-statistics path") from exc
    if stats_folder.name != "body" or stats_folder.parent.name != "motion":
        raise ValueError("loaded Kimodo motion-statistics path has unexpected structure")
    snapshot = stats_folder.parent.parent.parent
    try:
        snapshot.resolve().relative_to(Path(HF_HUB_CACHE).resolve())
    except ValueError as exc:
        raise ValueError("loaded Kimodo model is not inside the Hugging Face cache") from exc
    if (snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name != f"models--nvidia--{model_name}"):
        raise ValueError("loaded Kimodo snapshot path does not match the resolved model name")
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    manifest: dict[str, str] = {}
    for path in files:
        digest = _sha256(path)
        if digest is None:
            raise ValueError(f"snapshot file disappeared while hashing: {path}")
        manifest[path.relative_to(snapshot).as_posix()] = digest
    required = {"config.yaml", "model.safetensors",
                "stats/motion/body/mean.npy", "stats/motion/body/std.npy",
                "stats/motion/global_root/mean.npy", "stats/motion/global_root/std.npy",
                "stats/motion/local_root/mean.npy", "stats/motion/local_root/std.npy"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError("released Kimodo snapshot lacks model-consumed files: "
                         + ", ".join(missing))
    return {
        "checkpoint": {
            "generator_id": f"nvidia/{model_name}@{snapshot.name}",
            "model_name": model_name,
            "hf_revision": snapshot.name,
            "snapshot_path": str(snapshot),
            "hf_hub_cache": str(Path(HF_HUB_CACHE)),
            "snapshot_file_count": len(manifest),
            "snapshot_file_sha256": manifest,
            "snapshot_manifest_sha256": _json_hash(manifest),
            "required_model_files": sorted(required),
            "checkpoint_sha256": manifest["model.safetensors"],
        },
        "runner_class": type(runner).__name__,
        "runner_fps": float(runner.fps),
        "noise_stream_version": int(runner.noise_stream_version),
        "sampler": ("Kimodo offline DDIM, eta=0, non-autoregressive single pass "
                    "(kimodo/model/diffusion.py); post-processing bypassed"),
        "diffusion_steps": DIFFUSION_STEPS,
    }


def validated_generator_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized = json.loads(_canonical_json(_json_safe(dict(value))))
        checkpoint = normalized["checkpoint"]
        manifest = checkpoint["snapshot_file_sha256"]
        if (not _is_sha256(checkpoint["checkpoint_sha256"])
                or not _is_sha256(checkpoint["snapshot_manifest_sha256"])
                or not isinstance(manifest, dict) or not manifest
                or any(not _is_sha256(digest) for digest in manifest.values())
                or checkpoint["snapshot_manifest_sha256"] != _json_hash(manifest)
                or any(name not in manifest for name in checkpoint["required_model_files"])):
            raise ValueError("Kimodo snapshot manifest is invalid")
        if float(normalized["runner_fps"]) != FPS:
            raise ValueError("runner fps is not 30")
        if int(normalized["noise_stream_version"]) != NOISE_STREAM_VERSION:
            raise ValueError("runner is not on noise stream v2")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid Kimodo generator identity: {exc}") from exc
    return normalized


def validate_pins(generator: Mapping[str, Any], runtime: Mapping[str, Any],
                  physical_model: Mapping[str, Any]) -> None:
    """The three external pins CLAUDE.md names for this prior, re-checked at generation."""
    checkpoint = generator.get("checkpoint", {})
    if checkpoint.get("model_name") != MODEL_NAME:
        raise ValueError(f"EXP-025 requires {MODEL_NAME}")
    if checkpoint.get("hf_revision") != PINNED_KIMODO_HF_REVISION:
        raise ValueError("EXP-025 loaded the wrong Kimodo snapshot revision")
    if checkpoint.get("checkpoint_sha256") != PINNED_KIMODO_CHECKPOINT_SHA256:
        raise ValueError("EXP-025 loaded the wrong Kimodo checkpoint bytes")
    fields = dict(_checkout_part(runtime) or {}).get("fields", {})
    if fields.get("kimodo_git_commit") != PINNED_KIMODO_COMMIT:
        raise ValueError("EXP-025 loaded the wrong Kimodo runtime commit")
    if fields.get("kimodo_tracked_status") != []:
        raise ValueError("EXP-025 requires a clean tracked Kimodo checkout")
    if physical_model.get("fields", {}).get("sha256") != PINNED_G1_XML_SHA256:
        raise ValueError("EXP-025 loaded the wrong released G1 XML")


def _verify_stage_git_state(pinned: Mapping[str, Any], current: Mapping[str, Any],
                            *, repo: Path, output: Path) -> dict[str, Any]:
    """Later stages may run after new commits; require an unchanged tracked diff and no
    worktree change outside the campaign output."""
    if current.get("tracked_diff_sha256") != pinned.get("tracked_diff_sha256"):
        raise ValueError("tracked git diff changed since the campaign was pinned")
    try:
        relative_output = Path(output).resolve().relative_to(Path(repo).resolve()).as_posix()
    except ValueError:
        relative_output = None
    unexpected: list[str] = []
    allowed: list[str] = []
    for line in current.get("status", []):
        path = line[3:] if len(line) >= 4 else ""
        if relative_output is not None and (path == relative_output
                                            or path.startswith(relative_output + "/")):
            allowed.append(line)
        else:
            unexpected.append(line)
    if unexpected:
        raise ValueError("worktree changed outside the campaign output: " + "; ".join(unexpected))
    return {
        "pinned_commit": pinned.get("commit"), "current_commit": current.get("commit"),
        "commit_changed": current.get("commit") != pinned.get("commit"),
        "tracked_diff_unchanged": True,
        "allowed_output_status": allowed, "unexpected_status": unexpected,
    }


# ------------------------------------------------------------------------ reference scoring


class ScoringContext:
    """Everything the CPU scorer needs, built once for all 128 clips."""

    def __init__(self, route: np.ndarray, support: Mapping[str, float], modules: Any,
                 body: Any, probes: Mapping[str, Any]):
        self.route = np.asarray(route, dtype=float)
        self.support = dict(support)
        self.modules = modules
        self.body = body
        self.probes = dict(probes)


def build_scoring_context(route: np.ndarray, support: Mapping[str, float]) -> ScoringContext:
    from types import SimpleNamespace

    from experiments import analyze_event_frames as aef
    from experiments import analyze_trackability_contract as atc
    from experiments.analyze_e1a_placement import box_height_profile, lift_location, lift_side
    from scene2motion.robot import G1Body
    from scene2motion.stepover_eval import BoxHeightProbe, foot_kinematics_series

    modules = SimpleNamespace(
        atc=atc, aef=aef, box_height_profile=box_height_profile, lift_location=lift_location,
        lift_side=lift_side, foot_kinematics_series=foot_kinematics_series)
    probes = {label: BoxHeightProbe(float(x), OBSTACLE_DEPTH_M) for label, x in OBSTACLES}
    return ScoringContext(route, support, modules, G1Body(None), probes)


def load_support_thresholds() -> dict[str, float]:
    """The calibrated support envelope, through the hash-locked exp016 receipt loader.

    The thresholds are stated in SECONDS (``max_unsupported_run_s``) and metres/(m/s), so they
    carry from ARDY's 25 fps to Kimodo's 30 fps unchanged; nothing here rescales them.
    """
    from experiments import analyze_trackability_contract as atc

    support = dict(atc.load_support_thresholds(THRESHOLD_RECEIPT_PATH, THRESHOLD_RECEIPT_SHA256))
    if float(support["max_unsupported_run_s"]) != PRIMARY_GATE_S:
        raise ValueError("the calibrated screen is no longer the 0.20 s rule")
    if float(atc.PRIMARY_GATE_S) != PRIMARY_GATE_S or float(atc.SECONDARY_GATE_S) != SECONDARY_GATE_S:
        raise ValueError("EXP-025 screen thresholds drifted from the contract analyser")
    return support


def route_fidelity(smooth_root_pos: np.ndarray, qpos: np.ndarray,
                   route: np.ndarray) -> dict[str, Any]:
    """Route error against the channel that was actually constrained.

    ``smooth_root_2d`` writes columns 0 and 2 of Kimodo's ``smooth_root_pos`` block -- the
    ADMM-smoothed root's ground plane -- and leaves the pelvis free to sway around it by roughly
    the smoother's 0.06 m margin.  So the route error is ``smooth_root_pos[:, [0, 2]]`` minus the
    requested ``root_xz``, in the model's own frame, with no axis conversion.  The pelvis
    deviation is computed too and reported beside it, explicitly *not* as the route error.
    """
    smooth = np.asarray(smooth_root_pos, dtype=float)
    exact = np.asarray(qpos, dtype=float)
    target = np.asarray(route, dtype=float)
    if smooth.shape != (N_FRAMES, 3) or not np.isfinite(smooth).all():
        raise ValueError(f"smooth_root_pos must be a finite ({N_FRAMES}, 3) array, "
                         f"got {smooth.shape}")
    ground = smooth[:, [0, 2]]                       # (kimodo x, kimodo z) == route columns
    error = ground - target
    distance = np.linalg.norm(error, axis=-1)
    # MuJoCo qpos is Z-up with kimodo z -> mujoco x and kimodo x -> mujoco y
    # (kimodo/exports/mujoco.py: kimodo_to_mujoco_matrix), matching ARDY's convention.
    pelvis = np.stack([exact[:, 1], exact[:, 0]], axis=-1)
    pelvis_distance = np.linalg.norm(pelvis - target, axis=-1)
    planned = float(target[-1, 1] - target[0, 1])
    return {
        "constrained_channel": CONSTRAINT_CHANNEL,
        "measured_against": "smooth_root_pos",
        "smooth_root_path_mae_m": float(distance.mean()),
        "smooth_root_path_max_m": float(distance.max()),
        "smooth_root_forward_mae_m": float(np.abs(error[:, 1]).mean()),
        "smooth_root_lateral_mae_m": float(np.abs(error[:, 0]).mean()),
        "smooth_root_progress_ratio": float((ground[-1, 1] - ground[0, 1]) / planned),
        "pelvis_path_mae_m": float(pelvis_distance.mean()),
        "pelvis_path_max_m": float(pelvis_distance.max()),
        "pelvis_minus_smooth_root_mae_m": float(pelvis_distance.mean() - distance.mean()),
        "pelvis_progress_ratio": float((exact[-1, 0] - exact[0, 0]) / planned),
        "pelvis_note": ("the pelvis is NOT the constrained quantity; the smoother allows it to "
                        "sway around the smooth root (0.06 m margin)"),
        "forward_axis_dominant": bool(
            (exact[:, 0].max() - exact[:, 0].min()) > (exact[:, 1].max() - exact[:, 1].min())),
    }


def body_forward_extent(body: Any, qpos: np.ndarray) -> tuple[float, float]:
    """(min, max) forward reach of the whole-body collision envelope over the whole clip.

    This is the same envelope ``BoxHeightProbe`` collides against -- every named robot
    primitive, expanded by ``G1Body.body_margin`` -- projected onto the exported forward axis
    and minimised/maximised over all frames.  It answers one question: which stretch of the
    route did this robot's body actually occupy?
    """
    normal = np.asarray(FORWARD_AXIS, dtype=float)
    exact = np.asarray(qpos, dtype=float)
    if exact.ndim != 2 or not len(exact):
        raise ValueError("forward extent needs a non-empty (T, nq) clip")
    low, high = math.inf, -math.inf
    for frame in exact:
        body.fk(frame)
        for geom in body.robot_geoms:
            lo, hi = body.geom_extent(geom, normal, extra_margin=body.body_margin)
            low = lo if lo < low else low
            high = hi if hi > high else high
    if not (math.isfinite(low) and math.isfinite(high)) or high < low:
        raise ValueError("could not measure the clip's forward extent")
    return float(low), float(high)


def swept_box_centres(extent: tuple[float, float], centres: Sequence[float],
                      depth_m: float = OBSTACLE_DEPTH_M) -> np.ndarray:
    """Which box centres the body swept end to end (``COVERAGE_RULE``), as a boolean mask."""
    low, high = float(extent[0]), float(extent[1])
    half = 0.5 * float(depth_m)
    xs = np.asarray(centres, dtype=float)
    return (xs - half >= low) & (xs + half <= high)


def lift_timing(qpos: np.ndarray, lift_x_m: float | None, ctx: ScoringContext) -> dict[str, Any]:
    """When the clip's tallest clearable box is reached, in SECONDS (never frames).

    ARDY runs at 25 fps and Kimodo at 30, so every timing endpoint is converted to seconds
    before it leaves this function; the frame indices are kept only as provenance.  Both
    committed ARDY definitions are computed: the root-crossing frame (definition A, the
    paper's "1.4 s") is primary and the nominal-speed conversion (definition B) is secondary.
    """
    exact = np.asarray(qpos, dtype=float)
    record: dict[str, Any] = {
        "fps": FPS, "early_window_s": EARLY_WINDOW_S,
        "definition_primary": "root_crossing", "definition_secondary": "nominal_speed",
        "nominal_speed_mps": NOMINAL_SPEED_MPS,
        "root_crossing_frame": None, "lift_time_root_crossing_s": None,
        "within_first_2s_root_crossing": None,
        "nominal_frame": None, "lift_time_nominal_s": None, "within_first_2s_nominal": None,
    }
    if lift_x_m is None:
        return record
    frame = ctx.modules.aef.root_crossing_frame(exact[:, 0], float(lift_x_m))
    if frame is not None:
        record["root_crossing_frame"] = int(frame)
        record["lift_time_root_crossing_s"] = float(frame) / FPS
        record["within_first_2s_root_crossing"] = bool(float(frame) / FPS < EARLY_WINDOW_S)
    nominal = ctx.modules.aef.nominal_frame(float(lift_x_m), speed_mps=NOMINAL_SPEED_MPS,
                                            fps=FPS)
    record["nominal_frame"] = float(nominal)
    record["lift_time_nominal_s"] = float(nominal) / FPS
    record["within_first_2s_nominal"] = bool(float(nominal) / FPS < EARLY_WINDOW_S)
    return record


def score_reference_clip(qpos: np.ndarray, smooth_root_pos: np.ndarray, arm: str,
                         ctx: ScoringContext) -> dict[str, Any]:
    """The protocol's per-clip endpoints, in its order, for one 240-frame Kimodo clip."""
    exact = np.asarray(qpos, dtype=float)
    if exact.shape != (N_FRAMES, 36) or not np.isfinite(exact).all():
        raise ValueError(f"reference scoring requires a finite ({N_FRAMES}, 36) clip")
    route = ctx.route
    # 0. coverage: the stretch of route this body actually occupied.  Everything scored below
    #    is confined to it, because a box the robot never reaches is trivially "collision-free"
    #    and would otherwise be credited as a clearance (COVERAGE_RULE).
    extent_min, extent_max = body_forward_extent(ctx.body, exact)
    # 1. elicitation: the exp021 whole-body box-height profile, unchanged, read only where the
    #    body swept
    xs, heights = ctx.modules.box_height_profile(exact, route, OBSTACLE_DEPTH_M,
                                                 n_points=SCAN_POINTS)
    swept_mask = swept_box_centres((extent_min, extent_max), xs, OBSTACLE_DEPTH_M)
    swept_xs, swept_heights = np.asarray(xs)[swept_mask], np.asarray(heights)[swept_mask]
    excluded = np.asarray(heights)[~swept_mask]
    lift = ctx.modules.lift_location(swept_xs, swept_heights)
    side = None
    if lift["lift_x_m"] is not None:
        side = ctx.modules.lift_side(
            exact, ctx.modules.foot_kinematics_series(ctx.body, exact, FPS),
            float(lift["lift_x_m"]), OBSTACLE_DEPTH_M)
    coverage = {
        "forward_extent_min_m": float(extent_min),
        "forward_extent_max_m": float(extent_max),
        "forward_axis": list(FORWARD_AXIS),
        "envelope": "every robot collision primitive, expanded by G1Body.body_margin",
        "obstacle_depth_m": OBSTACLE_DEPTH_M,
        "scan_points": SCAN_POINTS,
        "scan_points_swept": int(swept_mask.sum()),
        "scan_window_m": [float(xs[0]), float(xs[-1])] if len(xs) else [None, None],
        "swept_scan_range_m": ([float(swept_xs[0]), float(swept_xs[-1])]
                               if swept_xs.size else [None, None]),
        "excluded_max_box_height_m": (float(excluded.max()) if excluded.size else None),
        "obstacles_swept": {label: bool(swept_box_centres((extent_min, extent_max), [x],
                                                          OBSTACLE_DEPTH_M)[0])
                            for label, x in OBSTACLES},
        "rule": COVERAGE_RULE,
    }
    coverage["covers_all_obstacles"] = bool(all(coverage["obstacles_swept"].values()))
    elicitation = {
        **lift,
        "lift_side": side,
        "elicited": bool(float(lift["lift_height_m"]) >= ELICITATION_MIN_M),
        "any_lift": bool(float(lift["lift_height_m"]) > 0.0),
        "min_clearance_m": ELICITATION_MIN_M,
        "scan_points": SCAN_POINTS,
        "scan_points_swept": int(swept_mask.sum()),
        "clears_height_anywhere": {f"{h:g}": bool(float(lift["lift_height_m"]) >= h)
                                   for h in GRADED_HEIGHTS_M},
        "denominator_note": ("scored over the swept scan points only; the excluded points are "
                             "positions this body never occupied"),
    }
    # 2. lift time, in seconds
    timing = lift_timing(exact, lift["lift_x_m"], ctx)
    # 3. exact fixed-centre clearance at 1.2 m and 3.6 m, graded heights (never +/- r), and
    #    only where the body swept the box: a non-arrival is "not reached" (null), not a pass
    exact_boxes: dict[str, Any] = {}
    for label, x in OBSTACLES:
        probe = ctx.probes[label]
        reached = bool(coverage["obstacles_swept"][label])
        exact_boxes[label] = {
            "obstacle_x_m": float(x),
            "obstacle_depth_m": OBSTACLE_DEPTH_M,
            "body_swept": reached,
            "max_box_height_lower_bound_m": (float(probe.probe(exact)) if reached else None),
            "exact_clears": ({f"{h:g}": bool(probe.clears(exact, float(h)))
                              for h in GRADED_HEIGHTS_M} if reached
                             else {f"{h:g}": None for h in GRADED_HEIGHTS_M}),
            "probe": {**probe.metadata(), "evaluated": reached},
            "not_reached_note": (None if reached else
                                 "the body never swept this box; the probe was not evaluated "
                                 "and no clearance is credited"),
        }
    # 4. the reference-screen features, at 30 fps with fps-free support thresholds
    features = ctx.modules.atc.features(ctx.body, exact, ctx.support["support_height_m"],
                                        ctx.support["support_speed_mps"], FPS)
    predictions = ctx.modules.atc.gate_predictions(features, PRIMARY_GATE_S, SECONDARY_GATE_S)
    # 5. route fidelity, against the smoothed root the channel actually constrains
    fidelity = route_fidelity(smooth_root_pos, exact, route)
    return {
        "arm": arm,
        "coverage": coverage,
        "elicitation": elicitation,
        "timing": timing,
        "exact_boxes": exact_boxes,
        "contract_features": features,
        "screen_predictions": predictions,
        "route_fidelity": fidelity,
    }


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number}")
    return number


def validated_reference_score(value: Mapping[str, Any]) -> dict[str, Any]:
    """Planned-denominator guard: every endpoint present, well typed and JSON-safe."""
    try:
        score = json.loads(_canonical_json(_json_safe(dict(value))))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reference score is not JSON-serialisable: {exc}") from exc
    coverage = score["coverage"]
    extent_min = _finite_number(coverage["forward_extent_min_m"], "forward_extent_min_m")
    extent_max = _finite_number(coverage["forward_extent_max_m"], "forward_extent_max_m")
    if extent_max < extent_min:
        raise ValueError("forward extent is inverted")
    if set(coverage["obstacles_swept"]) != {label for label, _ in OBSTACLES}:
        raise ValueError("coverage does not cover both obstacle centres")
    if any(not isinstance(flag, bool) for flag in coverage["obstacles_swept"].values()):
        raise ValueError("obstacles_swept must be boolean")
    elicitation = score["elicitation"]
    _finite_number(elicitation["lift_height_m"], "lift_height_m")
    if elicitation["lift_x_m"] is not None:
        lift_x = _finite_number(elicitation["lift_x_m"], "lift_x_m")
        low, high = coverage["swept_scan_range_m"]
        if low is None or high is None or not (low - 1e-9 <= lift_x <= high + 1e-9):
            raise ValueError("the lift position lies outside the swept scan window; a lift may "
                             "never be credited to route the body did not occupy")
    elif int(coverage["scan_points_swept"]) and float(elicitation["lift_height_m"]) > 0.0:
        raise ValueError("a positive lift height must carry a lift position")
    for name in ("elicited", "any_lift"):
        if not isinstance(elicitation[name], bool):
            raise ValueError(f"{name} must be boolean")
    if set(elicitation["clears_height_anywhere"]) != {f"{h:g}" for h in GRADED_HEIGHTS_M}:
        raise ValueError("clears_height_anywhere does not cover the graded heights")
    timing = score["timing"]
    if timing["fps"] != FPS or timing["early_window_s"] != EARLY_WINDOW_S:
        raise ValueError("timing record does not carry the locked fps / window")
    for name in ("lift_time_root_crossing_s", "lift_time_nominal_s"):
        if timing[name] is not None:
            _finite_number(timing[name], name)
    if (elicitation["lift_x_m"] is None) != (timing["lift_time_nominal_s"] is None):
        raise ValueError("timing presence disagrees with the lift position")
    for label, x in OBSTACLES:
        box = score["exact_boxes"][label]
        if box["obstacle_x_m"] != float(x):
            raise ValueError(f"exact box {label} is not centred at {x}")
        swept = box["body_swept"]
        if not isinstance(swept, bool):
            raise ValueError(f"body_swept at {label} must be boolean")
        if swept != bool(coverage["obstacles_swept"][label]):
            raise ValueError(f"body_swept at {label} disagrees with the coverage record")
        clears = box["exact_clears"]
        if set(clears) != {f"{h:g}" for h in GRADED_HEIGHTS_M}:
            raise ValueError(f"exact clears at {label} do not cover the graded heights")
        if swept:
            if any(not isinstance(flag, bool) for flag in clears.values()):
                raise ValueError(f"exact clears at {label} must be boolean")
            _finite_number(box["max_box_height_lower_bound_m"], "max_box_height_lower_bound_m")
        else:
            # A box the body never swept has no clearance outcome at all -- not True, and not
            # the probe's 0.40 m cap either.
            if any(flag is not None for flag in clears.values()):
                raise ValueError(f"exact clears at {label} must be null where the body never "
                                 "swept the box")
            if box["max_box_height_lower_bound_m"] is not None:
                raise ValueError(f"the {label} box height bound must be null where the body "
                                 "never swept the box")
    features = score["contract_features"]
    _finite_number(features["max_unsupported_run_s"], "max_unsupported_run_s")
    predictions = score["screen_predictions"]
    if (predictions["primary_threshold_s"] != PRIMARY_GATE_S
            or predictions["secondary_threshold_s"] != SECONDARY_GATE_S
            or not isinstance(predictions["primary_flag"], bool)
            or not isinstance(predictions["secondary_flag"], bool)):
        raise ValueError("screen predictions do not carry the locked rules")
    if predictions["max_unsupported_run_s"] != features["max_unsupported_run_s"]:
        raise ValueError("screen prediction feature disagrees with the contract features")
    fidelity = score["route_fidelity"]
    if fidelity["measured_against"] != "smooth_root_pos":
        raise ValueError("route fidelity must be measured against smooth_root_pos")
    if fidelity["constrained_channel"] != CONSTRAINT_CHANNEL:
        raise ValueError(f"route fidelity must name the {CONSTRAINT_CHANNEL} channel")
    for name in ("smooth_root_path_mae_m", "smooth_root_path_max_m", "smooth_root_progress_ratio",
                 "pelvis_path_mae_m", "pelvis_progress_ratio"):
        _finite_number(fidelity[name], name)
    return score


# --------------------------------------------------------------------------------- analysis


def clip_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One flat analysis record per clip; the planned denominator is 128 (64 per arm)."""
    records: list[dict[str, Any]] = []
    for row in rows:
        reference = row.get("reference")
        if reference is None:
            raise ValueError(f"row {row.get('archive_key')} has no reference score")
        elicitation = reference["elicitation"]
        timing = reference["timing"]
        run = float(reference["contract_features"]["max_unsupported_run_s"])
        records.append({
            "seed": int(row["seed"]), "arm": str(row["arm"]), "key": str(row["archive_key"]),
            "prompt": str(row["prompt"]),
            "elicited": bool(elicitation["elicited"]),
            "any_lift": bool(elicitation["any_lift"]),
            "lift_height_m": float(elicitation["lift_height_m"]),
            "lift_x_m": elicitation["lift_x_m"],
            "lift_side": elicitation["lift_side"],
            "lift_time_root_crossing_s": timing["lift_time_root_crossing_s"],
            "lift_time_nominal_s": timing["lift_time_nominal_s"],
            "within_first_2s_root_crossing": timing["within_first_2s_root_crossing"],
            "within_first_2s_nominal": timing["within_first_2s_nominal"],
            "exact_clears": {label: dict(reference["exact_boxes"][label]["exact_clears"])
                             for label, _ in OBSTACLES},
            "body_swept": {label: bool(reference["exact_boxes"][label]["body_swept"])
                           for label, _ in OBSTACLES},
            "max_box_height_lower_bound_m": {
                label: (None if reference["exact_boxes"][label][
                    "max_box_height_lower_bound_m"] is None
                    else float(reference["exact_boxes"][label][
                        "max_box_height_lower_bound_m"]))
                for label, _ in OBSTACLES},
            "forward_extent_min_m": float(reference["coverage"]["forward_extent_min_m"]),
            "forward_extent_max_m": float(reference["coverage"]["forward_extent_max_m"]),
            "covers_all_obstacles": bool(reference["coverage"]["covers_all_obstacles"]),
            "scan_points_swept": int(reference["coverage"]["scan_points_swept"]),
            "max_unsupported_run_s": run,
            "primary_flag": bool(reference["screen_predictions"]["primary_flag"]),
            "secondary_flag": bool(reference["screen_predictions"]["secondary_flag"]),
            "ballistic_ratio": reference["contract_features"].get("ballistic_ratio"),
            "root_z_max": float(reference["contract_features"]["root_z_max"]),
            "smooth_root_path_mae_m": float(
                reference["route_fidelity"]["smooth_root_path_mae_m"]),
            "pelvis_path_mae_m": float(reference["route_fidelity"]["pelvis_path_mae_m"]),
            "smooth_root_progress_ratio": float(
                reference["route_fidelity"]["smooth_root_progress_ratio"]),
        })
    return records


def _timing_block(members: Sequence[Mapping[str, Any]], lifting: Sequence[Mapping[str, Any]],
                  elicited: Sequence[Mapping[str, Any]], time_field: str,
                  flag_field: str) -> dict[str, Any]:
    """One event-time definition, on the protocol's denominator and its two companions."""
    n = len(members)
    timed = [r for r in lifting if r[time_field] is not None]
    k_lifting = sum(1 for r in lifting if bool(r[flag_field]))
    k_elicited = sum(1 for r in elicited if bool(r[flag_field]))
    k_timed = sum(1 for r in timed if bool(r[flag_field]))
    return {
        **quantiles([r[time_field] for r in timed]),
        "n_with_lift_position": len(lifting),
        "n_elicited": len(elicited),
        "n_timed": len(timed),
        "n_with_lift_position_without_event_time": len(lifting) - len(timed),
        "n_elicited_without_event_time": sum(1 for r in elicited if r[time_field] is None),
        # primary: the protocol's own denominator, missing event times counted as not-early
        "within_first_2s": rate(k_lifting, len(lifting)),
        "within_first_2s_over_all_assigned": rate(k_lifting, n),
        # companion: the >= 0.03 m elicited set, same missing-event-time rule
        "within_first_2s_elicited": rate(k_elicited, len(elicited)),
        "within_first_2s_elicited_over_all_assigned": rate(k_elicited, n),
        # secondary, clearly labelled: the timed-only fraction, which is biased upward
        "within_first_2s_timed_only": rate(k_timed, len(timed)),
        "missing_event_time_rule": MISSING_EVENT_TIME_RULE,
    }


def _arm_summary(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(members)
    elicited = [r for r in members if r["elicited"]]
    lifting = [r for r in members if r["any_lift"]]
    summary: dict[str, Any] = {
        "n_assigned": n,
        "elicitation": rate(sum(1 for r in members if r["elicited"]), n),
        "any_lift": rate(sum(1 for r in members if r["any_lift"]), n),
        "lift_height_m": quantiles([r["lift_height_m"] for r in members]),
        "lift_position_m": quantiles([r["lift_x_m"] for r in elicited]),
        "lift_position_any_lift_m": quantiles([r["lift_x_m"] for r in lifting]),
        "lift_time_s": {
            "primary_denominator": "clips with a lift position (any positive whole-body-"
                                   "clearable lift), matching the ARDY comparator",
            "companion_denominator": "elicited clips (whole-body-clearable lift >= 0.03 m)",
            "denominator_note": TIMING_DENOMINATOR_RULE,
            "n_with_lift_position": len(lifting),
            "n_elicited": len(elicited),
            "root_crossing": _timing_block(members, lifting, elicited,
                                           "lift_time_root_crossing_s",
                                           "within_first_2s_root_crossing"),
            "nominal_speed": _timing_block(members, lifting, elicited, "lift_time_nominal_s",
                                           "within_first_2s_nominal"),
        },
        "exact_clearance": {
            label: {f"{h:g}": rate(
                sum(1 for r in members if r["exact_clears"][label][f"{h:g}"] is True), n)
                for h in GRADED_HEIGHTS_M}
            for label, _ in OBSTACLES},
        "coverage": {
            "denominator": "all assigned trials",
            "rule": COVERAGE_RULE,
            "obstacles": {
                label: {
                    "body_swept": rate(sum(1 for r in members if r["body_swept"][label]), n),
                    "not_reached": n - sum(1 for r in members if r["body_swept"][label]),
                }
                for label, _ in OBSTACLES},
            "covers_all_obstacles": rate(
                sum(1 for r in members if r["covers_all_obstacles"]), n),
            "forward_extent_max_m": quantiles([r["forward_extent_max_m"] for r in members]),
            "scan_points_swept": quantiles([r["scan_points_swept"] for r in members]),
            "note": ("exact clearance is counted over ALL assigned trials; a clip that never "
                     "swept a box has a null outcome there and is counted as not clearing it, "
                     "never as a pass"),
        },
        "max_box_height_lower_bound_m": {
            label: quantiles([r["max_box_height_lower_bound_m"][label] for r in members])
            for label, _ in OBSTACLES},
        "screen": {
            "denominator": "elicited clips",
            "n_elicited": len(elicited),
            "float_primary_0p20s": rate(sum(1 for r in elicited if r["primary_flag"]),
                                        len(elicited)),
            "float_secondary_0p28s": rate(sum(1 for r in elicited if r["secondary_flag"]),
                                          len(elicited)),
            "float_primary_over_all_assigned": rate(
                sum(1 for r in members if r["primary_flag"]), n),
            "float_primary_elicited_over_all_assigned": rate(
                sum(1 for r in members if r["elicited"] and r["primary_flag"]), n),
            "max_unsupported_run_s": quantiles([r["max_unsupported_run_s"] for r in members]),
            "max_unsupported_run_s_elicited": quantiles(
                [r["max_unsupported_run_s"] for r in elicited]),
            "note": ("the 0.20 s rule is the calibrated reference screen for predicted "
                     "tracking cutoffs; no rollout was executed in this campaign"),
        },
        "route_fidelity": {
            "smooth_root_path_mae_m": quantiles([r["smooth_root_path_mae_m"] for r in members]),
            "pelvis_path_mae_m": quantiles([r["pelvis_path_mae_m"] for r in members]),
            "smooth_root_progress_ratio": quantiles(
                [r["smooth_root_progress_ratio"] for r in members]),
        },
        "root_z_max_m": quantiles([r["root_z_max"] for r in members]),
    }
    return summary


def paired_differences(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Same-seed step-minus-walk differences, descriptive only.

    One scene, so the seed is the resampling budget and not an inference unit: counts and
    medians are reported and no interval is claimed on any difference (house rule 7).
    """
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        by_seed.setdefault(int(record["seed"]), {})[str(record["arm"])] = record
    pairs = [(v["step"], v["walk"]) for v in by_seed.values()
             if set(v) == set(ARMS)]
    if len(pairs) != len(SEEDS):
        raise ValueError(f"expected {len(SEEDS)} same-seed pairs, got {len(pairs)}")
    step_only = sum(1 for s, w in pairs if s["elicited"] and not w["elicited"])
    walk_only = sum(1 for s, w in pairs if w["elicited"] and not s["elicited"])
    return {
        "n_pairs": len(pairs),
        "inference_unit": "one scene; seeds are the resampling budget within it",
        "interval_claimed_on_difference": False,
        "elicitation_discordant_pairs": {"step_only": step_only, "walk_only": walk_only,
                                         "concordant": len(pairs) - step_only - walk_only},
        "elicitation_difference": (
            sum(1 for s, _ in pairs if s["elicited"]) / len(pairs)
            - sum(1 for _, w in pairs if w["elicited"]) / len(pairs)),
        "median_lift_height_difference_m": float(np.median(
            [s["lift_height_m"] - w["lift_height_m"] for s, w in pairs])),
        "median_max_unsupported_run_difference_s": float(np.median(
            [s["max_unsupported_run_s"] - w["max_unsupported_run_s"] for s, w in pairs])),
        "median_smooth_root_path_mae_difference_m": float(np.median(
            [s["smooth_root_path_mae_m"] - w["smooth_root_path_mae_m"] for s, w in pairs])),
    }


def evaluate_decisions(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The protocol's two decision rules, evaluated mechanically.

    Rule 1 (timing).  "'early and rollout-tied' is claimed only for ARDY unless Kimodo's
    first-2 s fraction is also >= 0.7; if Kimodo's fraction is <= 0.4 the paper attributes the
    ARDY window to autoregressive rollout context; in between, the paper reports both
    distributions without a mechanism claim."

    Rule 2 (screen).  "The contract statement generalises to 'released G1 priors' only if
    >= 80 % of Kimodo's elicited clips are also floats by the calibrated 0.20 s gate;
    otherwise it stays ARDY-scoped."

    Both are evaluated on the STEP arm -- the arm the ARDY statements were made about -- but
    **not on the same denominator**, because the protocol does not state the same one twice.
    Rule 1's endpoint is the lift-time distribution, and the ARDY comparator it is calibrated
    against ("80-86 % inside 2.0 s" = 40/49 and 42/49) is over the exp021 clips with *any*
    positive whole-body-clearable lift; so rule 1 counts over the clips that have a lift
    position at all, with the >= 0.03 m fraction reported beside it.  Rule 2 names its
    denominator in so many words -- ">= 80 % of Kimodo's **elicited** clips" -- and keeps it.
    Mixing them would let one clip population decide a question asked about another.

    A clip with a lift position but no defined event time (its root never reaches the lift
    position) counts as **not** within the first 2.0 s rather than being dropped: it is the
    latest event there is, and dropping it can only push the fraction up.  An empty denominator
    is ``indeterminate``, never a pass.
    """
    step = [r for r in records if r["arm"] == "step"]
    if len(step) != len(SEEDS):
        raise ValueError(f"decision rules need {len(SEEDS)} step clips, got {len(step)}")
    elicited = [r for r in step if r["elicited"]]
    lifting = [r for r in step if r["any_lift"]]

    def timing_branch(fraction: float | None) -> str:
        if fraction is None:
            return "indeterminate_no_clips_with_a_lift_position"
        if fraction >= TIMING_GENERALISES_MIN_FRACTION:
            return "timing_generalises_to_released_g1_priors"
        if fraction <= TIMING_ROLLOUT_MAX_FRACTION:
            return "ardy_window_attributed_to_autoregressive_rollout_context"
        return "report_both_distributions_without_mechanism_claim"

    timing_rules: dict[str, Any] = {}
    for name, time_field, flag_field in (
            ("root_crossing", "lift_time_root_crossing_s", "within_first_2s_root_crossing"),
            ("nominal_speed", "lift_time_nominal_s", "within_first_2s_nominal")):
        timed = [r for r in lifting if r[time_field] is not None]
        # No event time means not-early, not excluded (MISSING_EVENT_TIME_RULE).
        k = sum(1 for r in lifting if bool(r[flag_field]))
        k_elicited = sum(1 for r in elicited if bool(r[flag_field]))
        k_timed = sum(1 for r in timed if bool(r[flag_field]))
        fraction = (k / len(lifting)) if lifting else None
        elicited_fraction = (k_elicited / len(elicited)) if elicited else None
        timing_rules[name] = {
            "first_2s": rate(k, len(lifting)),
            "fraction": fraction,
            "n_with_lift_position": len(lifting),
            "n_elicited": len(elicited),
            "n_timed": len(timed),
            "n_with_lift_position_without_event_time": len(lifting) - len(timed),
            "n_elicited_without_event_time": sum(1 for r in elicited
                                                 if r[time_field] is None),
            "over_all_assigned_step_trials": rate(
                sum(1 for r in step if r["any_lift"] and bool(r[flag_field])), len(step)),
            "first_2s_elicited": rate(k_elicited, len(elicited)),
            "fraction_elicited": elicited_fraction,
            "elicited_over_all_assigned_step_trials": rate(
                sum(1 for r in step if r["elicited"] and bool(r[flag_field])), len(step)),
            "first_2s_timed_only": rate(k_timed, len(timed)),
            "fraction_timed_only": (k_timed / len(timed)) if timed else None,
            "median_lift_time_s": (float(np.median([r[time_field] for r in timed]))
                                   if timed else None),
            "q10_q90_lift_time_s": ([float(np.quantile([r[time_field] for r in timed], 0.1)),
                                     float(np.quantile([r[time_field] for r in timed], 0.9))]
                                    if timed else None),
            "outcome": timing_branch(fraction),
            "outcome_on_elicited_denominator": timing_branch(elicited_fraction),
        }
    primary_timing = timing_rules["root_crossing"]
    # Fail closed where the preregistered branch would depend on which definition is read: the
    # protocol names one first-2 s fraction, so two definitions disagreeing while some clip has
    # no event time is an ambiguity to record, not a branch to pick.
    definitions_agree = bool(
        timing_rules["root_crossing"]["outcome"] == timing_rules["nominal_speed"]["outcome"])
    missing_event_times = int(primary_timing["n_with_lift_position_without_event_time"])
    timing_refusal = None
    if missing_event_times and not definitions_agree:
        timing_refusal = {
            "reason": "definitions_disagree_with_missing_event_times",
            "n_with_lift_position_without_event_time": missing_event_times,
            "root_crossing_outcome": timing_rules["root_crossing"]["outcome"],
            "nominal_speed_outcome": timing_rules["nominal_speed"]["outcome"],
            "note": ("the preregistered timing branch is not determined: the two committed "
                     "event-time definitions land on different branches and some clip with a "
                     "lift position has no root-crossing time.  Record the refusal and amend "
                     "the protocol; do not pick the definition that reads better"),
        }

    n_elicited = len(elicited)
    k_primary = sum(1 for r in elicited if r["primary_flag"])
    k_secondary = sum(1 for r in elicited if r["secondary_flag"])
    screen_fraction = (k_primary / n_elicited) if n_elicited else None
    if screen_fraction is None:
        screen_outcome = "indeterminate_no_elicited_clips"
    elif screen_fraction >= SCREEN_GENERALISES_MIN_FRACTION:
        screen_outcome = "screen_generalises_to_released_g1_priors"
    else:
        screen_outcome = "screen_stays_ardy_scoped"

    return {
        "arm_used": "step",
        "denominator_rule": TIMING_DENOMINATOR_RULE,
        "denominators": {
            "timing_rule": "STEP clips with a lift position (any positive whole-body-clearable "
                           "lift), the ARDY comparator's denominator",
            "screen_rule": "elicited STEP clips (whole-body-clearable lift >= 0.03 m), the "
                           "denominator the protocol names for this rule",
            "reported_beside_both": "rates over all 64 assigned STEP trials",
            "why_they_differ": TIMING_DENOMINATOR_RULE,
        },
        "timing_rule": {
            "thresholds": {"generalises_if_fraction_at_least": TIMING_GENERALISES_MIN_FRACTION,
                           "rollout_context_if_fraction_at_most": TIMING_ROLLOUT_MAX_FRACTION},
            "window_s": EARLY_WINDOW_S,
            "primary_definition": "root_crossing",
            "denominator": "STEP clips with a lift position (any positive lift)",
            "companion_denominator": "elicited STEP clips (lift >= 0.03 m)",
            "missing_event_time_rule": MISSING_EVENT_TIME_RULE,
            "n_with_lift_position": len(lifting),
            "n_elicited": len(elicited),
            "definitions": timing_rules,
            "outcome": primary_timing["outcome"],
            "outcome_on_elicited_denominator": primary_timing[
                "outcome_on_elicited_denominator"],
            "definitions_agree": definitions_agree,
            "refusal": timing_refusal,
            "ardy_reference": dict(ARDY_REFERENCE_EARLY_FRACTION),
        },
        "screen_rule": {
            "threshold_fraction": SCREEN_GENERALISES_MIN_FRACTION,
            "primary_s": PRIMARY_GATE_S, "secondary_s": SECONDARY_GATE_S,
            "n_elicited": n_elicited,
            "float_primary_0p20s": rate(k_primary, n_elicited),
            "float_secondary_0p28s": rate(k_secondary, n_elicited),
            "fraction": screen_fraction,
            "denominator": "elicited STEP clips (lift >= 0.03 m), as the protocol names",
            "over_all_assigned_step_trials": rate(
                sum(1 for r in step if r["primary_flag"]), len(step)),
            "over_all_assigned_step_trials_note": (
                "numerator = every assigned STEP clip above the screen, elicited or not; "
                "'elicited_float_over_all_assigned_step_trials' is the headline's own "
                "numerator over the same 64 assigned trials"),
            "elicited_float_over_all_assigned_step_trials": rate(
                sum(1 for r in step if r["elicited"] and r["primary_flag"]), len(step)),
            "outcome": screen_outcome,
            "ardy_reference": {"lifting_clips_above_0p20s": [44, 44],
                               "source": "outputs/analysis_trackability_contract"},
        },
        "no_arm_expansion_after_outcomes": {
            "arms": list(ARMS), "seeds": [int(SEEDS[0]), int(SEEDS[-1])],
            "held": True,
            "note": "the protocol forbids adding an arm after seeing outcomes",
        },
        "scope": ("kinematic only; no SONIC rollout was executed, so nothing here is a tracking "
                  "outcome and the 0.20 s rule is a reference screen, not a physical verdict"),
    }


def build_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm = {arm: [r for r in records if r["arm"] == arm] for arm in ARMS}
    for arm, members in by_arm.items():
        if len(members) != len(SEEDS):
            raise ValueError(f"summary needs {len(SEEDS)} records for {arm}, got {len(members)}")
    if len(records) != N_ROWS:
        raise ValueError(f"summary needs the planned {N_ROWS} records, got {len(records)}")
    return {
        "schema": SCHEMA_VERSION,
        "n_clips": len(records),
        "planned_n_per_arm": len(SEEDS),
        "prior": {"model": MODEL_NAME, "fps": FPS, "family": "offline (non-autoregressive)"},
        "tiers": {
            "generated": len(records),
            "kinematically_scored": len(records),
            "sonic_executed": 0,
            "note": ("house rule 10: every count names its tier.  Nothing in this campaign was "
                     "executed by a tracker, so no number here is an executed-clearance rate"),
        },
        "arms": {arm: _arm_summary(members) for arm, members in by_arm.items()},
        "paired_step_minus_walk": paired_differences(records),
        "decisions": evaluate_decisions(records),
    }


# ----------------------------------------------------------------------------------- ledger


def _read_receipt(output: Path) -> dict[str, Any]:
    path = Path(output) / "receipt.json"
    if not path.is_file():
        raise CampaignAbort(f"no EXP-025 receipt at {path}; run --stage generate first")
    try:
        receipt = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise CampaignAbort(f"unreadable EXP-025 receipt: {exc}") from exc
    if not isinstance(receipt, dict) or receipt.get("experiment") != EXPERIMENT:
        raise CampaignAbort("existing output is not an EXP-025 campaign")
    if receipt.get("blocked") is True:
        raise CampaignAbort(
            "existing EXP-025 campaign is blocked; preserve it, record the refusal and resize "
            "on FRESH seeds in a fresh output directory")
    return receipt


class Ledger:
    """Receipt + rows + evidence anchors for one campaign directory."""

    def __init__(self, output: Path, receipt: dict[str, Any], rows: list[dict[str, Any]],
                 started: float | None = None):
        self.output = Path(output)
        self.receipt = receipt
        self.rows = rows
        self.started = time.monotonic() if started is None else started

    @classmethod
    def load(cls, output: Path) -> "Ledger":
        output = Path(output)
        receipt = _read_receipt(output)
        rows = _read_jsonl(output / "rows.jsonl")
        anchors = receipt.get("evidence_anchors", {}).get("rows", {})
        if (anchors.get("n_rows") != len(rows)
                or anchors.get("logical_sha256") != _json_hash(_json_safe(rows))):
            raise CampaignAbort("rows.jsonl no longer matches its evidence anchor")
        if anchors.get("file_sha256") != _sha256(output / "rows.jsonl"):
            raise CampaignAbort("rows.jsonl file hash no longer matches its evidence anchor")
        return cls(output, receipt, rows,
                   started=time.monotonic() - float(receipt.get("wall_clock_s", 0.0)))

    def stage(self, name: str) -> dict[str, Any]:
        return self.receipt.setdefault("stages", {}).setdefault(name, {"status": "planned"})

    def require_stage_complete(self, name: str) -> None:
        if self.stage(name).get("status") != "complete":
            raise CampaignAbort(f"EXP-025 stage {name!r} is not complete in {self.output}")

    def anchor_file(self, name: str, path: Path, **extra: Any) -> None:
        self.receipt.setdefault("evidence_anchors", {})[name] = {
            "path": Path(path).name, "file_sha256": _sha256(path), **extra}

    def persist(self, *, stage_label: str | None = None) -> None:
        _write_jsonl(self.output / "rows.jsonl", self.rows)
        self.receipt.setdefault("evidence_anchors", {})["rows"] = {
            "path": "rows.jsonl", "n_rows": len(self.rows),
            "logical_sha256": _json_hash(_json_safe(self.rows)),
            "file_sha256": _sha256(self.output / "rows.jsonl"),
        }
        if stage_label is not None:
            self.receipt["stage"] = stage_label
        self.receipt["wall_clock_s"] = float(time.monotonic() - self.started)
        _write_json(self.output / "receipt.json", self.receipt)

    def fail(self, stage_name: str, exc: BaseException, stage_label: str,
             *, blocking: bool = True) -> None:
        """Record a stage failure durably.  ``blocked`` is reserved for the generate stage.

        Only ``generate`` spends seeds, so only ``generate`` can leave a campaign that must be
        preserved and resized on fresh seeds.  A failure in the CPU-only ``score`` or
        ``analyze`` stage leaves the 128 archived clips byte-identical: blocking the receipt
        there would make ``Ledger.load`` refuse the directory for ever and force a
        regeneration, which is exactly what CLAUDE.md forbids ("finish it with a resume script
        that re-scores the archives through byte-identical sources ... never by regenerating").
        So a post-generation failure is recorded, kept in an append-only history, and left
        resumable.
        """
        record = {"stage": stage_name, "failed_at": stage_label,
                  "error_type": type(exc).__name__, "error": str(exc)}
        self.stage(stage_name).update({"status": "failed", "failure": dict(record)})
        self.receipt.setdefault("stage_failures", []).append(dict(record))
        if blocking:
            self.receipt.update({
                "schema": FAILURE_SCHEMA_VERSION, "status": "blocked", "complete": False,
                "blocked": True, "failed_stage": stage_label,
                "error_type": type(exc).__name__, "error": str(exc)})
        else:
            self.receipt.update({
                "status": "running", "complete": False, "blocked": False,
                "last_stage_failure": dict(record), "resumable": True,
                "resume_note": ("post-generation failure: the archived clips are unchanged, so "
                                "this campaign is re-scored in place and never regenerated"),
            })
        self.persist(stage_label=stage_label)


def _validate_generation_archive(ledger: Ledger) -> tuple[dict[str, np.ndarray],
                                                          dict[str, np.ndarray]]:
    """Revalidate both archives against the receipt before any later stage reads them."""
    ledger.require_stage_complete("generate")
    plan = locked_row_plan()
    if len(ledger.rows) != N_ROWS:
        raise CampaignAbort("ledger does not hold the planned 128 rows")
    clips = _load_arrays(ledger.output / "qpos.npz")
    smooth = _load_arrays(ledger.output / "smooth_root.npz")
    anchors = ledger.receipt.get("evidence_anchors", {})
    keys = {row["archive_key"] for row in plan}
    if set(clips) != keys or anchors.get("qpos", {}).get("n_arrays") != N_ROWS:
        raise CampaignAbort("qpos archive keys do not match the locked row plan")
    if set(smooth) != keys or anchors.get("smooth_root", {}).get("n_arrays") != N_ROWS:
        raise CampaignAbort("smooth_root archive keys do not match the locked row plan")
    if _array_hash(clips) != anchors.get("qpos", {}).get("content_sha256"):
        raise CampaignAbort("qpos archive content hash no longer matches its evidence anchor")
    if _array_hash(smooth) != anchors.get("smooth_root", {}).get("content_sha256"):
        raise CampaignAbort("smooth_root content hash no longer matches its evidence anchor")
    for row, item in zip(ledger.rows, plan):
        for field in ("row_index", "seed", "arm", "archive_key", "chunk", "prompt"):
            if row.get(field) != item[field]:
                raise CampaignAbort(f"ledger row {row.get('row_index')} drifted from the plan")
        key = row["archive_key"]
        qpos, root = clips[key], smooth[key]
        if qpos.shape != (N_FRAMES, 36) or not np.isfinite(qpos).all():
            raise CampaignAbort(f"archived {key} is not a finite {N_FRAMES}x36 clip")
        if root.shape != (N_FRAMES, 3) or not np.isfinite(root).all():
            raise CampaignAbort(f"archived smooth root {key} is not a finite {N_FRAMES}x3 array")
        if _array_hash({key: qpos}) != row.get("qpos_content_sha256"):
            raise CampaignAbort(f"archived {key} does not match its row hash")
        if _array_hash({key: root}) != row.get("smooth_root_content_sha256"):
            raise CampaignAbort(f"archived smooth root {key} does not match its row hash")
    return clips, smooth


def _stage_provenance_check(ledger: Ledger, *, code_state_fn, source_hashes_fn,
                            external_hashes_fn, runtime_identity_fn,
                            physical_identity_fn) -> dict[str, Any]:
    provenance = ledger.receipt.get("provenance", {})
    current_code = dict(code_state_fn(ROOT))
    git = _verify_stage_git_state(provenance.get("code", {}), current_code,
                                  repo=ROOT, output=ledger.output)
    current_sources = dict(source_hashes_fn(ROOT))
    pinned_sources = dict(provenance.get("source_sha256") or {})
    changed = sorted(name for name in set(current_sources) | set(pinned_sources)
                     if current_sources.get(name) != pinned_sources.get(name))
    if changed:
        raise ValueError("EXP-025 source content changed since generation: " + ", ".join(changed))
    current_external = dict(external_hashes_fn())
    if current_external != dict(provenance.get("external_source_sha256") or {}):
        raise ValueError("EXP-025 external (Kimodo/ARDY) source content changed since generation")
    # Only the *checkout* half of the Kimodo runtime identity is stage-invariant.  The
    # interpreter half (sys.version, numpy, torch) differs between the generation venv and the
    # scoring venv by design, so it is recorded for this stage and never compared: comparing it
    # would abort every post-generation stage after the reserved seeds had been spent.
    current_runtime = dict(runtime_identity_fn())
    pinned_runtime = provenance.get("kimodo_runtime")
    if _checkout_part(current_runtime) != _checkout_part(pinned_runtime):
        raise ValueError("EXP-025 Kimodo checkout identity changed since generation")
    if dict(physical_identity_fn()) != provenance.get("physical_model"):
        raise ValueError("EXP-025 physical model identity changed since generation")
    return {"git": git, "sources_unchanged": True, "external_sources_unchanged": True,
            "kimodo_checkout_unchanged": True, "physical_model_unchanged": True,
            "generation_interpreter_runtime": _interpreter_part(pinned_runtime),
            "stage_interpreter_runtime": _interpreter_part(current_runtime),
            "interpreter_runtime_compared_across_stages": False,
            "interpreter_note": ("generation runs under the kimodo venv and scoring under "
                                 "$S2M_PY; the two interpreters are recorded, not equated"),
            "current_code": current_code}


# ------------------------------------------------------------------------------ noise audit


@contextmanager
def latent_row_audit() -> Iterator[dict[str, Any]]:
    """Observe the per-sample latent rows the runner draws during one ``generate`` call.

    Kimodo is non-autoregressive: the only stochastic input is a single
    ``torch.randn((B, T, motion_rep_dim))`` (kimodo_model.py:610), and the recovered runner's
    ``_per_sample_noise`` replaces it with one draw per batch row from a generator seeded by
    that row's seed alone.  Wrapping ``torch.randn`` around the call therefore records exactly
    one hash per row without touching the runner, and turns the campaign's pairing claim --
    same-seed step/walk rows share a latent -- into evidence.
    """
    import torch

    real = torch.randn
    draws: list[dict[str, Any]] = []
    generator_order: dict[int, int] = {}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        generator = kwargs.get("generator")
        if generator is not None:
            identity = id(generator)
            if identity not in generator_order:
                generator_order[identity] = len(generator_order)
            draws.append({
                "row": generator_order[identity],
                "shape": [int(dim) for dim in result.shape],
                "sha256": hashlib.sha256(
                    result.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
            })
        return result

    audit: dict[str, Any] = {"draws": draws}
    torch.randn = wrapper
    try:
        yield audit
    finally:
        torch.randn = real


def summarize_latent_audit(audit: Mapping[str, Any],
                           chunk_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Turn the observed draws into pairing evidence, and enforce the pairing contract.

    Raises when rows that share a seed drew different latents (the paired step/walk comparison
    would not be paired) or when rows with different seeds coincide.  An absent audit is
    recorded as ``unavailable``, never as a pass.
    """
    draws = list(audit.get("draws", []))
    n_rows = len(chunk_rows)
    if not draws:
        return {"status": "unavailable", "n_rows": n_rows, "draws_per_row": 0, "rows": [],
                "pairing_verified": False,
                "note": "no generator-tagged torch.randn draws were observed"}
    per_row: dict[int, list[dict[str, Any]]] = {}
    for draw in draws:
        per_row.setdefault(int(draw["row"]), []).append(draw)
    observed_rows = sorted(per_row)
    if observed_rows[:n_rows] != list(range(n_rows)):
        raise ValueError(
            f"latent audit observed rows {observed_rows}, expected the first {n_rows}")
    extra = [row for row in observed_rows if row >= n_rows]
    counts = {row: len(per_row[row]) for row in range(n_rows)}
    if len(set(counts.values())) != 1 or next(iter(counts.values())) < 1:
        raise ValueError(f"latent audit rows drew unequal draw counts: {counts}")
    per_row_draws = next(iter(counts.values()))
    rows_out = []
    for local, item in enumerate(chunk_rows):
        hashes = [draw["sha256"] for draw in per_row[local]]
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"latent stream repeated a draw for row {local}")
        rows_out.append({
            "batch_position": local, "seed": int(item["seed"]), "arm": str(item["arm"]),
            "row_sha256_by_draw": hashes,
            "latent_shape": per_row[local][0]["shape"],
        })
    by_seed: dict[int, list[list[str]]] = {}
    for entry in rows_out:
        by_seed.setdefault(entry["seed"], []).append(entry["row_sha256_by_draw"])
    for seed, lists in by_seed.items():
        if any(item != lists[0] for item in lists[1:]):
            raise ValueError(f"same-seed rows drew different latents for seed {seed}")
    seeds = list(by_seed)
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            if any(x == y for x, y in zip(by_seed[a][0], by_seed[b][0])):
                raise ValueError(f"different seeds {a} and {b} share a latent draw")
    return {
        "status": "verified", "n_rows": n_rows, "draws_per_row": per_row_draws,
        "pairing_verified": True,
        "same_seed_rows_identical": True,
        "different_seed_rows_differ": True,
        "extra_generator_draw_rows": extra,
        "rows": rows_out,
    }


# --------------------------------------------------------------------------------- generate


def default_runner_factory(cache_path: Path) -> Callable[[], Any]:
    def factory() -> Any:
        from experiments.kimodo_recovered.kimodo_runner import KimodoRunner

        return KimodoRunner(model_name=MODEL_NAME, device=DEVICE,
                            cache_path=str(cache_path), text_encoder=False)
    return factory


def run_generate(
    *,
    out: str | Path,
    runner_factory: Callable[[], Any] | None = None,
    ardy_cache: str | Path = ARDY_TEXT_CACHE,
    kimodo_cache: str | Path = KIMODO_TEXT_CACHE,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = _git_state,
    source_hashes_fn: Callable[[Path], Mapping[str, str]] = _source_hashes,
    external_hashes_fn: Callable[[], Mapping[str, str]] = external_source_hashes,
    text_cache_fn: Callable[..., Mapping[str, Any]] = build_campaign_text_cache,
    generator_identity_fn: Callable[[Any], Mapping[str, Any]] = kimodo_generator_identity,
    generator_identity_validator_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] = (
        validated_generator_identity),
    runtime_identity_fn: Callable[[], Mapping[str, Any]] = kimodo_runtime_identity,
    physical_identity_fn: Callable[[], Mapping[str, Any]] = physical_model_identity,
    pin_validator_fn: Callable[..., None] = validate_pins,
    channel_usage_fn: Callable[[Any, ConstraintSpec], Mapping[str, int]] = _actual_channel_usage,
    host_gate_fn: Callable[..., Mapping[str, Any]] = require_host_resources,
    latent_audit_factory: Callable[[], Any] = latent_row_audit,
) -> dict[str, Any]:
    """Stage 1: sixteen locked B=8 calls into an empty directory; the ledger exists first.

    ``runner_factory`` is the only way a model reaches this function, and it is called *after*
    the empty evidence bundle and the campaign prompt cache are durable.  Production passes
    ``None`` and gets the real ``KimodoRunner``; the CPU tests inject a fake and every injection
    is recorded, which marks the run non-evidentiary.
    """
    output = Path(out)
    if output.exists() and any(output.iterdir()):
        raise CampaignAbort(f"refusing nonempty EXP-025 output directory: {output}")
    # The host gate runs before anything is created or any seed is spent, so a refusal leaves
    # --out untouched and the same directory can be launched later.
    try:
        gate_report = dict(host_gate_fn(**ARDY_GENERATION_GATE))
    except HostResourceGateFailed as exc:
        raise CampaignAbort(f"generation refused by the host-resource gate: {exc}") from exc
    code = dict(code_state_fn(ROOT))
    if code.get("dirty") is not False:
        raise CampaignAbort("EXP-025 requires an exactly clean git worktree")
    if not isinstance(code.get("commit"), str) or not code["commit"].strip():
        raise CampaignAbort("EXP-025 requires a concrete git commit")
    if os.environ.get("CHECKPOINTS_DIR"):
        raise CampaignAbort("EXP-025 forbids ambient CHECKPOINTS_DIR")
    source_hashes = dict(source_hashes_fn(ROOT))
    protocol_sha = source_hashes.get(PROTOCOL_PATH)
    if not _is_sha256(protocol_sha):
        raise CampaignAbort("EXP-025 protocol content hash is missing or invalid")
    external_hashes = dict(external_hashes_fn())

    injected = [name for name, value, default in (
        ("runner_factory", runner_factory, None),
        ("code_state_fn", code_state_fn, _git_state),
        ("source_hashes_fn", source_hashes_fn, _source_hashes),
        ("external_hashes_fn", external_hashes_fn, external_source_hashes),
        ("text_cache_fn", text_cache_fn, build_campaign_text_cache),
        ("generator_identity_fn", generator_identity_fn, kimodo_generator_identity),
        ("generator_identity_validator_fn", generator_identity_validator_fn,
         validated_generator_identity),
        ("runtime_identity_fn", runtime_identity_fn, kimodo_runtime_identity),
        ("physical_identity_fn", physical_identity_fn, physical_model_identity),
        ("pin_validator_fn", pin_validator_fn, validate_pins),
        ("channel_usage_fn", channel_usage_fn, _actual_channel_usage),
        ("host_gate_fn", host_gate_fn, require_host_resources),
        ("latent_audit_factory", latent_audit_factory, latent_row_audit),
    ) if value is not default]

    plan = locked_row_plan()
    chunks = locked_chunk_plan(plan)
    route = route_xz()
    spec = campaign_spec(route)
    spec_hash = spec_sha256(spec)
    adapter_channels = static_channel_usage(spec)
    if adapter_channels != dict(EXPECTED_ADAPTER_CHANNELS):
        raise CampaignAbort(
            f"the constraint adapter writes {adapter_channels}, expected "
            f"{dict(EXPECTED_ADAPTER_CHANNELS)} -- the {CONSTRAINT_CHANNEL} channel is the "
            "whole conditioning design")
    # Hash the plan both before and after the spec hash is stamped in, so the receipt's
    # ``row_plan_sha256`` is the same number ``--dry-run`` prints and the tests pin.
    plan_sha256 = _json_hash(plan)
    for row in plan:
        row["spec_sha256"] = spec_hash
    bound_plan_sha256 = _json_hash(plan)

    counters = {
        "generate_invocations_planned": len(chunks),
        "generate_invocations_started": 0,
        "generate_invocations_completed": 0,
        "samples_planned": N_ROWS,
        "samples_launched": 0,
        "samples_returned": 0,
        "samples_converted_to_qpos": 0,
    }
    spent_seeds: list[int] = []
    qpos_archive: dict[str, np.ndarray] = {}
    smooth_archive: dict[str, np.ndarray] = {}
    noise_evidence: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "running", "complete": False, "blocked": False, "stage": "preflight",
        "stages": {name: {"status": "planned"} for name in STAGES},
        "sample_count_exact": True,
        "actual_kimodo_samples": 0,
        "campaign_design": {
            "question": ("does the ARDY STEP-prompt timing and reference-screen signature "
                         "reproduce on a released offline (non-autoregressive) G1 prior"),
            "prior": {"model": MODEL_NAME, "device": DEVICE, "fps": FPS,
                      "family": "offline diffusion; one denoising pass, no history window"},
            "arms": list(ARMS),
            "arm_prompts": dict(ARM_PROMPTS),
            "control_arm": {"arm": "walk",
                            "role": "free nominal arm / elicitation floor (house rule 9)"},
            "seeds": list(SEEDS),
            "row_plan": plan,
            "row_plan_sha256": plan_sha256,
            "row_plan_with_spec_sha256": bound_plan_sha256,
            "chunk_plan": [{k: v for k, v in chunk.items() if k != "rows"} for chunk in chunks],
            "batch_size": BATCH_SIZE,
            "chunk_seed_count": CHUNK_SEED_COUNT,
            "batch_composition_rationale": (
                "each B=8 call holds 4 seeds x (step, walk), so every same-seed pair shares one "
                "call and one per-sample latent; the protocol fixes the batch size, not its "
                "composition"),
            "constraint": {
                "channel": CONSTRAINT_CHANNEL,
                "contract": CONSTRAINT_CONTRACT,
                "root_height": "free", "heading": "free",
                "dense": True, "spec_sha256": spec_hash,
                "adapter_channels_written": adapter_channels,
                "expected_model_channel_usage": dict(EXPECTED_CHANNEL_USAGE),
                "note": ("smooth_root_2d constrains the ADMM-smoothed root (0.06 m smoother "
                         "margin), not the raw pelvis; route error is measured against "
                         "smooth_root_pos"),
            },
            "route": {"length_m": ROUTE_LENGTH_M, "n_frames": N_FRAMES, "fps": FPS,
                      "nominal_speed_mps": NOMINAL_SPEED_MPS,
                      "frame_span_duration_s": DURATION_S,
                      "protocol_speed_mps": PROTOCOL_SPEED_MPS,
                      "protocol_duration_s": PROTOCOL_DURATION_S,
                      "sha256": _array_hash({"route_xz": route})},
            "diffusion_steps": DIFFUSION_STEPS, "cfg_weight": list(CFG_WEIGHT),
            "cfg_type": CFG_TYPE, "first_heading": FIRST_HEADING,
            "post_processing": ("bypassed: the runner calls Kimodo._generate + "
                                "motion_rep.inverse and never reaches Kimodo.__call__"),
            "noise_stream_version": NOISE_STREAM_VERSION,
            "obstacles": [{"label": label, "x_m": x} for label, x in OBSTACLES],
            "obstacle_depth_m": OBSTACLE_DEPTH_M,
            "graded_heights_m": list(GRADED_HEIGHTS_M),
            "elicitation": {"scan_points": SCAN_POINTS, "min_clearance_m": ELICITATION_MIN_M,
                            "definition": "exp021 box_height_profile / lift_location",
                            "scored_only_where_the_body_swept": True},
            "coverage_guard": {"rule": COVERAGE_RULE, "forward_axis": list(FORWARD_AXIS),
                               "envelope": ("every robot collision primitive, expanded by "
                                            "G1Body.body_margin")},
            "denominators": {"timing_rule": TIMING_DENOMINATOR_RULE,
                             "missing_event_time": MISSING_EVENT_TIME_RULE,
                             "screen_rule": ("elicited clips (lift >= 0.03 m), the "
                                             "denominator the protocol names for this rule")},
            "screen": {"primary_s": PRIMARY_GATE_S, "secondary_s": SECONDARY_GATE_S,
                       "flag_rule": "max_unsupported_run_s > threshold",
                       "note": ("thresholds are in seconds and therefore fps-free; the "
                                "0.20 s rule is the calibrated reference screen for predicted "
                                "tracking cutoffs, not a physical verdict")},
            "decision_rules": {
                "timing": ("first-2 s fraction >= 0.7 -> generalises; <= 0.4 -> the ARDY window "
                           "is attributed to autoregressive rollout context; otherwise report "
                           "both distributions with no mechanism claim"),
                "screen": (">= 80 % of elicited Kimodo clips above the 0.20 s screen -> the "
                           "statement generalises to released G1 priors; otherwise ARDY-scoped"),
                "no_arm_expansion_after_outcomes": True,
            },
            "stage_scope": "kinematic only; no SONIC in this campaign",
            "part_b_reduced_audit": "out of scope for ICRA 2027 (protocol section 'Part B')",
        },
        "query_accounting": dict(counters),
        "generation_chunks": {chunk["name"]: {"status": "planned", "seeds": list(chunk["seeds"]),
                                              "row_indices": list(chunk["row_indices"])}
                              for chunk in chunks},
        "host_resource_gate": {"generate": gate_report},
        "interpreters": {"generate": str(KIMODO_PY), "score": str(SCORING_PY),
                         "running": sys.executable,
                         "note": ("generation needs the kimodo package, scoring needs mujoco, "
                                  "and no local interpreter has both")},
        "provenance": {
            "code": code,
            "source_sha256": source_hashes,
            "external_source_sha256": external_hashes,
            "protocol": {"path": PROTOCOL_PATH, "sha256": protocol_sha,
                         "status": "preregistered 2026-09-03, before the first sample"},
        },
        "execution_mode": {
            "dependency_injections": injected,
            "scientific_evidence_eligible": not injected,
            "pre_model_construction_evidence_guaranteed": True,
            "note": ("Dependency injection exists for CPU tests; any injected run is "
                     "non-evidentiary. Production constructs the Kimodo runner only after the "
                     "empty evidence bundle and the campaign prompt cache are durable."),
        },
        "spent_seeds": [],
        "seeds_spent_and_must_not_be_reused": False,
    }
    ledger = Ledger(output, receipt, [])
    stage_record = ledger.stage("generate")
    stage_record["status"] = "running"

    def persist(stage_label: str) -> None:
        _persist_arrays(output / "qpos.npz", qpos_archive)
        _persist_arrays(output / "smooth_root.npz", smooth_archive)
        _write_json(output / "noise_audit.json", noise_evidence)
        receipt["query_accounting"] = dict(counters)
        receipt["actual_kimodo_samples"] = int(counters["samples_returned"])
        receipt["spent_seeds"] = list(spent_seeds)
        receipt["unlaunched_locked_seeds"] = [s for s in SEEDS if s not in spent_seeds]
        receipt["seeds_spent_and_must_not_be_reused"] = bool(spent_seeds)
        anchors = receipt.setdefault("evidence_anchors", {})
        anchors["qpos"] = {
            "path": "qpos.npz", "n_arrays": len(qpos_archive),
            "content_sha256": _array_hash(qpos_archive) if qpos_archive else None,
            "file_sha256": _sha256(output / "qpos.npz")}
        anchors["smooth_root"] = {
            "path": "smooth_root.npz", "n_arrays": len(smooth_archive),
            "content_sha256": _array_hash(smooth_archive) if smooth_archive else None,
            "file_sha256": _sha256(output / "smooth_root.npz")}
        anchors["noise_audit"] = {
            "path": "noise_audit.json", "n_records": len(noise_evidence),
            "logical_sha256": _json_hash(_json_safe(noise_evidence)),
            "file_sha256": _sha256(output / "noise_audit.json")}
        ledger.persist(stage_label=stage_label)

    # The empty ledger is durable before anything else exists.
    persist("preflight")
    stage_label = "preflight"
    runner: Any | None = None
    try:
        cache_identity = dict(text_cache_fn(
            output / CAMPAIGN_TEXT_CACHE_NAME,
            ardy_cache=Path(ardy_cache), kimodo_cache=Path(kimodo_cache)))
        receipt["provenance"]["prompt_cache"] = cache_identity
        ledger.anchor_file("prompt_cache", output / CAMPAIGN_TEXT_CACHE_NAME,
                           identity_sha256=cache_identity.get("sha256"))
        persist("prompt_cache_bound")

        stage_label = "runner_construction"
        factory = runner_factory or default_runner_factory(output / CAMPAIGN_TEXT_CACHE_NAME)
        runner = factory()
        if float(runner.fps) != FPS:
            raise ValueError(f"EXP-025 requires runner fps == {FPS:g}, got {runner.fps}")
        if int(runner.noise_stream_version) != NOISE_STREAM_VERSION:
            raise ValueError("EXP-025 requires noise_stream_version == 2")

        stage_label = "identities"
        generator_identity = dict(generator_identity_validator_fn(generator_identity_fn(runner)))
        runtime_identity = dict(runtime_identity_fn())
        physical_identity = dict(physical_identity_fn())
        pin_validator_fn(generator_identity, runtime_identity, physical_identity)
        receipt["provenance"].update({
            "generator": generator_identity,
            "kimodo_runtime": runtime_identity,
            "physical_model": physical_identity,
            "runner_prompt_cache_check": verify_runner_text_cache(runner, cache_identity),
        })
        observed_usage = {str(k): int(v) for k, v in channel_usage_fn(runner, spec).items()}
        if observed_usage != dict(EXPECTED_CHANNEL_USAGE):
            raise ValueError(f"the campaign conditions the wrong channels: {observed_usage} != "
                             f"{dict(EXPECTED_CHANNEL_USAGE)}")
        receipt["campaign_design"]["actual_channel_usage"] = observed_usage
        persist("identities_bound")

        def revalidate() -> dict[str, Any]:
            current_code = dict(code_state_fn(ROOT))
            git_check = _verify_stage_git_state(code, current_code, repo=ROOT, output=output)
            if current_code.get("commit") != code.get("commit"):
                raise ValueError("git commit changed during EXP-025 generation")
            if dict(source_hashes_fn(ROOT)) != source_hashes:
                raise ValueError("EXP-025 source content changed during generation")
            if dict(external_hashes_fn()) != external_hashes:
                raise ValueError("EXP-025 external source content changed during generation")
            current = dict(generator_identity_validator_fn(generator_identity_fn(runner)))
            if current != generator_identity:
                raise ValueError("EXP-025 checkpoint identity changed during generation")
            current_runtime = dict(runtime_identity_fn())
            if _checkout_part(current_runtime) != _checkout_part(runtime_identity):
                raise ValueError("EXP-025 Kimodo checkout identity changed during generation")
            if dict(physical_identity_fn()) != physical_identity:
                raise ValueError("EXP-025 G1 physical model identity changed")
            pin_validator_fn(current, runtime_identity, physical_identity)
            verify_runner_text_cache(runner, cache_identity)
            if float(runner.fps) != FPS or int(runner.noise_stream_version) != NOISE_STREAM_VERSION:
                raise ValueError("EXP-025 runner contract changed during generation")
            return {"git": git_check, "sources_unchanged": True,
                    "external_sources_unchanged": True, "checkpoint_unchanged": True,
                    "kimodo_checkout_unchanged": True, "physical_model_unchanged": True,
                    "generation_interpreter_runtime": _interpreter_part(current_runtime),
                    "prompt_cache_unchanged": True, "runner_contract_unchanged": True}

        for chunk in chunks:
            name = str(chunk["name"])
            stage_label = f"generation_{name}"
            chunk_rows = list(chunk["rows"])
            counters["generate_invocations_started"] += 1
            counters["samples_launched"] += len(chunk_rows)
            spent_seeds.extend(s for s in chunk["seeds"] if s not in spent_seeds)
            receipt["generation_chunks"][name]["status"] = "running"
            persist(stage_label)
            prompts = [str(row["prompt"]) for row in chunk_rows]
            seeds = [int(row["seed"]) for row in chunk_rows]
            try:
                with latent_audit_factory() as audit:
                    returned = runner.generate(
                        prompts, [spec] * len(chunk_rows), N_FRAMES, DIFFUSION_STEPS,
                        cfg_weight=CFG_WEIGHT, seeds=seeds, cfg_type=CFG_TYPE)
            except Exception:
                receipt["sample_count_exact"] = False
                receipt["generation_chunks"][name]["status"] = "generation_exception"
                persist(stage_label)
                raise
            returned = list(returned)
            counters["generate_invocations_completed"] += 1
            counters["samples_returned"] += len(returned)
            receipt["generation_chunks"][name].update({
                "status": "returned_unvalidated", "samples_returned": len(returned)})
            audit_summary = summarize_latent_audit(audit, chunk_rows)
            noise_evidence.append({"chunk": name, "seeds": list(chunk["seeds"]),
                                   "batch_seed_order": seeds, "batch_prompts": prompts,
                                   **audit_summary})
            if len(returned) != len(chunk_rows):
                persist(stage_label)
                raise ValueError(
                    f"{name} returned {len(returned)} samples, expected {len(chunk_rows)}")
            for row, sample in zip(chunk_rows, returned):
                key = str(row["archive_key"])
                qpos = np.asarray(runner.to_qpos(sample))
                root = np.asarray(sample["smooth_root_pos"])
                if qpos.shape != (N_FRAMES, 36) or not np.isfinite(qpos).all():
                    qpos_archive[f"{key}_invalid_return"] = np.array(qpos, copy=True)
                    persist(stage_label)
                    raise ValueError(f"{key} decoded to an invalid qpos of shape {qpos.shape}")
                if root.shape != (N_FRAMES, 3) or not np.isfinite(root).all():
                    smooth_archive[f"{key}_invalid_return"] = np.array(root, copy=True)
                    persist(stage_label)
                    raise ValueError(
                        f"{key} returned an invalid smooth_root_pos of shape {root.shape}")
                qpos_archive[key] = np.array(qpos, dtype=np.float32, copy=True)
                smooth_archive[key] = np.array(root, dtype=np.float32, copy=True)
                counters["samples_converted_to_qpos"] += 1
                ledger.rows.append({
                    **dict(row),
                    "sample_sha256": _sample_hash(sample),
                    "qpos_content_sha256": _array_hash({key: qpos_archive[key]}),
                    "smooth_root_content_sha256": _array_hash({key: smooth_archive[key]}),
                    "qpos_dtype": str(qpos_archive[key].dtype),
                    "latent_row_sha256": (
                        audit_summary["rows"][int(row["batch_position"])]["row_sha256_by_draw"]
                        if audit_summary["status"] == "verified" else None),
                    "generated": True,
                })
            receipt["generation_chunks"][name].update({
                "status": "complete", "latent_audit_status": audit_summary["status"]})
            persist(stage_label)

        if len(ledger.rows) != N_ROWS or counters["samples_returned"] != N_ROWS:
            raise ValueError("EXP-025 generation accounting is not exactly 128/128")
        if [row["archive_key"] for row in ledger.rows] != [row["archive_key"] for row in plan]:
            raise ValueError("EXP-025 rows are not in locked plan order")
        stage_label = "post_generation_revalidation"
        receipt["provenance"]["post_generation_identity_revalidation"] = revalidate()
        stage_record.update({
            "status": "complete", "samples": N_ROWS,
            "latent_audit": {
                "verified_chunks": sum(1 for e in noise_evidence if e["status"] == "verified"),
                "unavailable_chunks": sum(
                    1 for e in noise_evidence if e["status"] == "unavailable"),
                "pairing_verified_every_chunk": all(
                    e["status"] == "verified" for e in noise_evidence),
            },
        })
        receipt["actual_kimodo_samples"] = N_ROWS
        persist("generated")
        return receipt
    except Exception as exc:
        receipt["stage"] = stage_label
        ledger.fail("generate", exc, stage_label)
        try:
            persist(stage_label)
        except Exception:  # pragma: no cover - the failure receipt above is already durable
            pass
        if isinstance(exc, CampaignAbort):
            raise
        raise CampaignAbort(str(exc)) from exc


# ------------------------------------------------------------------------------------ score


def run_score(
    *,
    out: str | Path,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = _git_state,
    source_hashes_fn: Callable[[Path], Mapping[str, str]] = _source_hashes,
    external_hashes_fn: Callable[[], Mapping[str, str]] = external_source_hashes,
    runtime_identity_fn: Callable[[], Mapping[str, Any]] = kimodo_runtime_identity,
    physical_identity_fn: Callable[[], Mapping[str, Any]] = physical_model_identity,
    support_thresholds_fn: Callable[[], Mapping[str, float]] = load_support_thresholds,
    scoring_context_fn: Callable[..., Any] = build_scoring_context,
    reference_scorer_fn: Callable[..., Mapping[str, Any]] = score_reference_clip,
) -> dict[str, Any]:
    """Stage 2: CPU reference endpoints for all 128 clips; resumes per clip."""
    output = Path(out)
    ledger = Ledger.load(output)
    if ledger.stage("score").get("status") == "complete":
        _validate_generation_archive(ledger)
        return ledger.receipt
    stage_label = "score_preflight"
    try:
        clips, smooth = _validate_generation_archive(ledger)
        check = _stage_provenance_check(
            ledger, code_state_fn=code_state_fn, source_hashes_fn=source_hashes_fn,
            external_hashes_fn=external_hashes_fn, runtime_identity_fn=runtime_identity_fn,
            physical_identity_fn=physical_identity_fn)
        support = dict(support_thresholds_fn())
        stage = ledger.stage("score")
        stage.pop("failure", None)          # the history lives in receipt["stage_failures"]
        stage.update({"status": "running", "provenance_check": check,
                      "support_thresholds": support, "scored": 0, "planned": N_ROWS,
                      "interpreter": sys.executable})
        ledger.persist(stage_label="scoring")
        route = route_xz()
        if _array_hash({"route_xz": route}) != ledger.receipt["campaign_design"]["route"]["sha256"]:
            raise ValueError("route drifted from the generation stage")
        ctx = scoring_context_fn(route, support)
        scored_now = 0
        for index, row in enumerate(ledger.rows):
            stage_label = f"score_{row['archive_key']}"
            if row.get("reference") is not None:
                continue
            started = time.monotonic()
            score = validated_reference_score(reference_scorer_fn(
                clips[row["archive_key"]], smooth[row["archive_key"]], str(row["arm"]), ctx))
            row["reference"] = score
            row["reference_scoring_wall_clock_s"] = float(time.monotonic() - started)
            scored_now += 1
            stage["scored"] = index + 1
            if (index + 1) % 8 == 0 or index + 1 == N_ROWS:
                ledger.persist(stage_label="scoring")
        if sum(1 for row in ledger.rows if row.get("reference") is not None) != N_ROWS:
            raise ValueError("reference scoring did not preserve the planned denominator")
        stage_label = "score_axis_check"
        # One global guard on the export convention: Kimodo's MuJoCo export maps its forward
        # axis (kimodo z) to MuJoCo x, exactly as ARDY's does.  Every box probe and every
        # root-crossing time depends on that; if it ever changed, this stage must fail closed
        # rather than silently score the lateral axis.
        dominant = sum(1 for row in ledger.rows
                       if row["reference"]["route_fidelity"]["forward_axis_dominant"])
        if dominant != N_ROWS:
            raise ValueError(
                f"the exported forward axis is qpos[:, 0] in only {dominant}/{N_ROWS} clips; "
                "the Kimodo -> MuJoCo axis convention is not what the scorer assumes")
        stage_label = "score_summary"
        records = clip_records(ledger.rows)
        stage.update({
            "status": "complete", "scored": N_ROWS, "scored_this_invocation": scored_now,
            "forward_axis_dominant_clips": dominant,
            "reference_summary_per_arm": {
                arm: {
                    "n": len(members),
                    "elicitation": rate(sum(1 for r in members if r["elicited"]), len(members)),
                    "any_lift": rate(sum(1 for r in members if r["any_lift"]), len(members)),
                    "float_primary_0p20s": rate(
                        sum(1 for r in members if r["primary_flag"]), len(members)),
                    "median_smooth_root_path_mae_m": float(np.median(
                        [r["smooth_root_path_mae_m"] for r in members])),
                    "median_pelvis_path_mae_m": float(np.median(
                        [r["pelvis_path_mae_m"] for r in members])),
                }
                for arm, members in (
                    (arm, [r for r in records if r["arm"] == arm]) for arm in ARMS)},
            "post_score_provenance_check": _stage_provenance_check(
                ledger, code_state_fn=code_state_fn, source_hashes_fn=source_hashes_fn,
                external_hashes_fn=external_hashes_fn, runtime_identity_fn=runtime_identity_fn,
                physical_identity_fn=physical_identity_fn),
        })
        ledger.persist(stage_label="scored")
        return ledger.receipt
    except Exception as exc:
        # Not blocking: scoring spends no seeds, and the archives it reads are byte-identical.
        ledger.fail("score", exc, stage_label, blocking=False)
        if isinstance(exc, CampaignAbort):
            raise
        raise CampaignAbort(str(exc)) from exc


# ---------------------------------------------------------------------------------- analyze


def run_analyze(
    *,
    out: str | Path,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = _git_state,
    source_hashes_fn: Callable[[Path], Mapping[str, str]] = _source_hashes,
    external_hashes_fn: Callable[[], Mapping[str, str]] = external_source_hashes,
    runtime_identity_fn: Callable[[], Mapping[str, Any]] = kimodo_runtime_identity,
    physical_identity_fn: Callable[[], Mapping[str, Any]] = physical_model_identity,
) -> dict[str, Any]:
    """Stage 3: the planned-denominator summary and the two preregistered decision rules."""
    output = Path(out)
    ledger = Ledger.load(output)
    stage = ledger.stage("analyze")
    if stage.get("status") == "complete":
        return ledger.receipt
    try:
        _validate_generation_archive(ledger)
        ledger.require_stage_complete("score")
        check = _stage_provenance_check(
            ledger, code_state_fn=code_state_fn, source_hashes_fn=source_hashes_fn,
            external_hashes_fn=external_hashes_fn, runtime_identity_fn=runtime_identity_fn,
            physical_identity_fn=physical_identity_fn)
        stage.pop("failure", None)          # the history lives in receipt["stage_failures"]
        records = clip_records(ledger.rows)
        summary = build_summary(records)
        refusal = summary["decisions"]["timing_rule"].get("refusal")
        summary["status"] = "refused" if refusal else "complete"
        # Every measurement is made durable first, refusal or not: a refusal withholds the
        # preregistered branch, never the numbers it was computed from.
        _write_json(output / "summary.json", summary)
        _write_jsonl(output / "clip_records.jsonl", records)
        ledger.anchor_file("summary", output / "summary.json",
                           logical_sha256=_json_hash(_json_safe(summary)))
        ledger.anchor_file("clip_records", output / "clip_records.jsonl", n_rows=len(records),
                           logical_sha256=_json_hash(_json_safe(records)))
        ledger.receipt.update({"summary": summary, "decisions": summary["decisions"]})
        if refusal:
            stage.update({"provenance_check": check, "refusal": refusal})
            ledger.persist(stage_label="analyze_refusal")
            raise CampaignAbort(
                "EXP-025 timing rule refused: " + str(refusal["note"])
                + f" (root_crossing -> {refusal['root_crossing_outcome']}, nominal_speed -> "
                  f"{refusal['nominal_speed_outcome']}, "
                  f"{refusal['n_with_lift_position_without_event_time']} clips with a lift "
                  "position have no event time)")
        stage.update({"status": "complete", "provenance_check": check})
        ledger.receipt.update({"status": "complete", "complete": True, "blocked": False,
                               "stage": "complete"})
        ledger.persist(stage_label="complete")
        return ledger.receipt
    except Exception as exc:
        # Not blocking: analysis reads the same byte-identical archives (see Ledger.fail).
        ledger.fail("analyze", exc, "analyze", blocking=False)
        if isinstance(exc, CampaignAbort):
            raise
        raise CampaignAbort(str(exc)) from exc


# ----------------------------------------------------------------------------------- dry run


def dry_run_report() -> dict[str, Any]:
    """Batch plan, arm specs and the live host gate; touches no disk, no GPU, no model."""
    route = route_xz()
    spec = campaign_spec(route)
    return {
        "schema": SCHEMA_VERSION, "experiment": EXPERIMENT, "status": "dry_run",
        "writes_performed": False,
        "protocol": {"path": PROTOCOL_PATH, "sha256": _sha256(ROOT / PROTOCOL_PATH)},
        "samples_planned": N_ROWS,
        "batch_plan": {
            "batch_size": BATCH_SIZE, "n_calls": N_CHUNKS,
            "row_plan_sha256": _json_hash(locked_row_plan()),
            "chunks": [{k: v for k, v in chunk.items() if k != "rows"}
                       for chunk in locked_chunk_plan()],
        },
        "arms": {arm: {
            "prompt": ARM_PROMPTS[arm],
            "role": ("free nominal arm / elicitation floor" if arm == "walk"
                     else "prompt under test"),
            "constraint_channel": CONSTRAINT_CHANNEL,
            "contract": CONSTRAINT_CONTRACT,
            "spec_sha256": spec_sha256(spec),
            "adapter_channels_written": static_channel_usage(spec),
            "expected_model_channel_usage": dict(EXPECTED_CHANNEL_USAGE),
        } for arm in ARMS},
        "route": {"length_m": ROUTE_LENGTH_M, "n_frames": N_FRAMES, "fps": FPS,
                  "nominal_speed_mps": NOMINAL_SPEED_MPS,
                  "protocol_speed_mps": PROTOCOL_SPEED_MPS,
                  "protocol_duration_s": PROTOCOL_DURATION_S,
                  "sha256": _array_hash({"route_xz": route})},
        "generation": {"diffusion_steps": DIFFUSION_STEPS, "cfg_weight": list(CFG_WEIGHT),
                       "first_heading": FIRST_HEADING,
                       "noise_stream_version": NOISE_STREAM_VERSION,
                       "post_processing": "bypassed"},
        "endpoints": {"obstacles": [{"label": label, "x_m": x} for label, x in OBSTACLES],
                      "graded_heights_m": list(GRADED_HEIGHTS_M),
                      "elicitation_min_m": ELICITATION_MIN_M, "scan_points": SCAN_POINTS,
                      "screen_primary_s": PRIMARY_GATE_S,
                      "screen_secondary_s": SECONDARY_GATE_S,
                      "early_window_s": EARLY_WINDOW_S,
                      "route_error_measured_against": "smooth_root_pos",
                      "coverage_rule": COVERAGE_RULE,
                      "timing_denominator_rule": TIMING_DENOMINATOR_RULE,
                      "missing_event_time_rule": MISSING_EVENT_TIME_RULE},
        "host_resource_gate": {"generate": host_resource_report(**ARDY_GENERATION_GATE)},
        "interpreters": {
            **{stage: str(STAGE_INTERPRETERS[stage]) for stage in STAGE_INTERPRETERS},
            "analyze": "any interpreter (needs neither kimodo nor mujoco)",
            "requirements": {stage: STAGE_REQUIREMENTS[stage] for stage in STAGES},
        },
        "sonic": "none; this campaign is kinematic only",
    }


# --------------------------------------------------------------------------------------- CLI


def _module_available(name: str | None) -> bool:
    if name is None:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _stage_in_subprocess(stage: str, out: str | Path,
                         extra: Sequence[str] = ()) -> None:
    interpreter = STAGE_INTERPRETERS.get(stage)
    if interpreter is None or not Path(interpreter).exists():
        raise CampaignAbort(
            f"stage {stage!r} needs the {STAGE_REQUIREMENTS[stage]!r} module, which this "
            f"interpreter lacks, and no interpreter is available at {interpreter}")
    # ``--no-dispatch`` in the child makes a mis-configured interpreter fail loudly instead of
    # handing the stage back and recursing.
    cmd = [str(interpreter), str(Path(__file__).resolve()), "--stage", stage, "--out", str(out),
           "--no-dispatch", *extra]
    print("  " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise CampaignAbort(f"EXP-025 {stage} subprocess returned {completed.returncode}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--ardy-cache", default=str(ARDY_TEXT_CACHE))
    parser.add_argument("--kimodo-cache", default=str(KIMODO_TEXT_CACHE))
    parser.add_argument("--resume", action="store_true",
                        help="continue an existing output directory's later stages")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the batch plan, the arm specs and the host-gate report; "
                             "writes nothing")
    parser.add_argument("--no-dispatch", action="store_true",
                        help="refuse a stage this interpreter cannot run instead of re-invoking "
                             "it under the interpreter that can (set on dispatched children)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps(_json_safe(dry_run_report()), indent=2, sort_keys=True))
        return
    stages = list(STAGES) if args.stage == "all" else [args.stage]

    def show(payload: Mapping[str, Any]) -> None:
        print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))

    try:
        for stage in stages:
            if stage == "generate" and args.stage == "all" and (
                    Path(args.out) / "receipt.json").is_file():
                if not args.resume:
                    raise CampaignAbort(
                        f"{args.out} already holds a campaign; pass --resume to continue its "
                        "later stages or choose a fresh --out")
                continue
            if not _module_available(STAGE_REQUIREMENTS[stage]):
                if args.no_dispatch:
                    raise CampaignAbort(
                        f"stage {stage!r} needs the {STAGE_REQUIREMENTS[stage]!r} module and "
                        f"{sys.executable} does not have it")
                _stage_in_subprocess(
                    stage, args.out,
                    ["--ardy-cache", args.ardy_cache, "--kimodo-cache", args.kimodo_cache]
                    if stage == "generate" else [])
                continue
            if stage == "generate":
                receipt = run_generate(out=args.out, ardy_cache=args.ardy_cache,
                                       kimodo_cache=args.kimodo_cache)
                show({"stage": "generate", "status": receipt["stages"]["generate"]["status"],
                      "actual_kimodo_samples": receipt["actual_kimodo_samples"]})
            elif stage == "score":
                score_stage = run_score(out=args.out)["stages"]["score"]
                show({"stage": "score", "scored": score_stage["scored"],
                      "per_arm": score_stage["reference_summary_per_arm"]})
            elif stage == "analyze":
                summary = run_analyze(out=args.out)["summary"]
                show({"stage": "analyze",
                      "elicitation": {arm: summary["arms"][arm]["elicitation"] for arm in ARMS},
                      "decisions": summary["decisions"]})
    except (CampaignAbort, HostResourceGateFailed) as exc:
        raise SystemExit(f"EXP-025 {args.stage}: {exc}")


if __name__ == "__main__":
    main()
