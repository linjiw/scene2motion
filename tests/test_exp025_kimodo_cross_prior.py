"""CPU tests for the EXP-025 Part A driver (no GPU, no Kimodo checkpoint, no Isaac).

Everything here runs with fakes.  Nothing loads Kimodo-G1-RP-v1, imports the ``kimodo``
package, touches CUDA, or reads a gitignored artifact: the two prompt caches, the LLM2Vec
wrapper sources and the runner itself are all synthesised in ``tmp_path``.  The handful of
checks that must look at the real external checkouts (``/home/linjiw/kimodo``,
``/home/linjiw/ardy``) are guarded by ``requires_external``.

The load-bearing test in this file is
``test_route_fidelity_is_measured_against_smooth_root_not_the_pelvis``: the protocol's
2026-09-03 amendment says the ``smooth_root_2d`` channel constrains the ADMM-smoothed root, so
route error measured against the raw pelvis reads about 6 cm high by construction and biases
the whole cross-prior comparison.  That test fails if anyone switches the measurement back.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import calibrate_ramp_route_phase as cal
from experiments import exp025_kimodo_cross_prior as exp
from experiments.kimodo_recovered import kimodo_runner as kr
from scene2motion.constraints import ArdyConstraintSet
from scene2motion.host_gate import HostResourceGateFailed

ROOT = Path(__file__).resolve().parents[1]
LLM_DIM = int(exp.EXPECTED_LLM2VEC_KWARGS["llm_dim"])

_EXTERNAL_PATHS = (exp.KIMODO_SANITIZE_PATH, exp.ARDY_LLM2VEC_WRAPPER,
                   exp.KIMODO_LLM2VEC_WRAPPER, exp.ARDY_LOAD_MODEL, exp.KIMODO_LOAD_MODEL)
requires_external = pytest.mark.skipif(
    not all(path.is_file() for path in _EXTERNAL_PATHS),
    reason="the ARDY and Kimodo checkouts are not present on this host")


# ------------------------------------------------------------------------------- fakes


def identity_sanitize(text: str, paragraph: bool = True) -> str:
    """Kimodo's ``sanitize_text`` is the identity on both campaign prompts (asserted below)."""
    return text


def stub_equivalence() -> dict[str, object]:
    return {"schema": "stub", "equivalent": True, "wrappers_equal_after_normalisation": True,
            "preset_kwargs_equal": True}


def clean_code_state(_repo):
    return {"commit": "a" * 40, "dirty": False, "status": [], "tracked_diff_sha256": "b" * 64}


def fake_source_hashes(_repo):
    return {exp.PROTOCOL_PATH: "d" * 64,
            "experiments/exp025_kimodo_cross_prior.py": "c" * 64}


def fake_external_hashes():
    return {str(exp.KIMODO_SANITIZE_PATH): "e" * 64}


def fake_runtime_identity(interpreter: str = "kimodo-venv", commit: str = "1" * 40):
    """A runtime identity shaped like the real one: a stage-invariant checkout half and a
    per-stage interpreter half.

    The campaign runs generation under the Kimodo venv and scoring under ``$S2M_PY`` by design,
    so the fakes below deliberately differ in the interpreter half and agree in the checkout
    half -- the shape that must revalidate.
    """
    return {
        "schema": "exp025-kimodo-runtime-identity-v2",
        "checkout": {"schema": "fake-checkout", "fields": {"kimodo_git_commit": commit},
                     "sha256": "a" * 64},
        "interpreter": {"schema": "fake-interpreter",
                        "fields": {"python": f"3.11.16 ({interpreter})",
                                   "numpy_version": "2.4.6" if "kimodo" in interpreter
                                   else "1.26.4"},
                        "sha256": "b" * 64},
    }


def passing_gate(**_kwargs):
    return {"pass": True, "checks": {"vram": True, "ram": True, "no_isaac": True},
            "vram": {"free_mib": 15000}, "ram": {"available_mib": 20000},
            "concurrent_isaac_processes": []}


def failing_gate(**_kwargs):
    raise HostResourceGateFailed("host-resource gate failed on vram: free VRAM 1001 MiB")


def write_prompt_caches(tmp_path: Path) -> tuple[Path, Path]:
    """An ARDY-shaped cache holding STEP and a Kimodo-shaped cache holding WALK."""
    rng = np.random.default_rng(2025)
    ardy = tmp_path / "ardy_text_cache.npz"
    kimodo = tmp_path / "kimodo_indoor_nav_text_cache.npz"
    np.savez(ardy, **{
        kr._raw_key(exp.STEP): rng.normal(size=(1, LLM_DIM)).astype(np.float32),
        kr._raw_key("A person walks forward."): rng.normal(size=(1, LLM_DIM)).astype(np.float32),
    })
    np.savez(kimodo, **{
        kr._raw_key(exp.WALK): rng.normal(size=(1, LLM_DIM)).astype(np.float32),
        kr._raw_key("A person opens a door."): rng.normal(size=(1, LLM_DIM)).astype(np.float32),
    })
    return ardy, kimodo


def cache_factory(ardy: Path, kimodo: Path):
    def cache_fn(out_path, *, ardy_cache, kimodo_cache):
        assert Path(ardy_cache) == ardy and Path(kimodo_cache) == kimodo
        return exp.build_campaign_text_cache(
            out_path, ardy_cache=ardy_cache, kimodo_cache=kimodo_cache,
            sanitize_fn=identity_sanitize, encoder_equivalence_fn=stub_equivalence)
    return cache_fn


def synthetic_qpos(seed: int, arm: str) -> np.ndarray:
    route = exp.route_xz()
    qpos = np.zeros((exp.N_FRAMES, 36), dtype=np.float32)
    qpos[:, 0] = route[:, 1]                       # forward axis (kimodo z -> mujoco x)
    qpos[:, 2] = 0.78
    qpos[:, 3] = 1.0
    qpos[:, 7] = seed / 1e4 + (0.0 if arm == "step" else 0.5)
    return qpos


def synthetic_smooth_root(seed: int, arm: str) -> np.ndarray:
    """Kimodo frame ``(x, y, z)`` = (lateral, height, forward); the ground plane is (0, 2)."""
    route = exp.route_xz()
    smooth = np.zeros((exp.N_FRAMES, 3), dtype=np.float32)
    smooth[:, 0] = route[:, 0]
    smooth[:, 1] = 0.78
    smooth[:, 2] = route[:, 1]
    return smooth


class FakeRunner:
    """Stands in for ``KimodoRunner`` and asserts the evidence bundle already exists."""

    fps = exp.FPS
    noise_stream_version = exp.NOISE_STREAM_VERSION
    model_name = exp.MODEL_NAME
    device = "cpu"

    def __init__(self, output: Path):
        for name in ("receipt.json", "rows.jsonl", "qpos.npz", "smooth_root.npz",
                     "noise_audit.json", exp.CAMPAIGN_TEXT_CACHE_NAME):
            assert (output / name).exists(), name
        receipt = json.loads((output / "receipt.json").read_text())
        assert receipt["status"] == "running"
        assert receipt["stages"]["generate"]["status"] == "running"
        assert (output / "rows.jsonl").read_text() == ""
        with np.load(output / exp.CAMPAIGN_TEXT_CACHE_NAME) as cache:
            self._text_cache = {key: np.array(cache[key], copy=True) for key in cache.files}
        self.output = output
        self.calls = 0
        self.seen: list[list[int]] = []

    def _cache_key(self, text: str) -> str | None:
        for key in (kr._raw_key(identity_sanitize(text)), kr._raw_key(text)):
            if key in self._text_cache:
                return key
        return None

    def _draw(self, seeds):
        # The generators stay alive for the whole call, exactly as ``_per_sample_noise`` keeps
        # them in a dict: the audit identifies rows by generator object.
        generators = [torch.Generator().manual_seed(int(seed)) for seed in seeds]
        for generator in generators:
            torch.randn((4, 3), generator=generator)

    def generate(self, prompts, specs, num_frames, diffusion_steps, cfg_weight, seeds,
                 cfg_type=None):
        assert num_frames == exp.N_FRAMES and diffusion_steps == exp.DIFFUSION_STEPS
        assert tuple(cfg_weight) == exp.CFG_WEIGHT and cfg_type is exp.CFG_TYPE
        assert len(prompts) == len(specs) == len(seeds) == exp.CHUNK_ROWS
        expected = exp.SEEDS[self.calls * exp.CHUNK_SEED_COUNT:
                             (self.calls + 1) * exp.CHUNK_SEED_COUNT]
        assert tuple(dict.fromkeys(seeds)) == expected
        assert prompts == [exp.ARM_PROMPTS[a] for _ in expected for a in exp.ARMS]
        assert all(self._cache_key(prompt) is not None for prompt in prompts)
        assert all(spec.heading is None and spec.root_y is None for spec in specs)
        self.calls += 1
        self.seen.append(list(seeds))
        self._draw(seeds)
        samples = []
        for seed, prompt in zip(seeds, prompts):
            arm = "step" if prompt == exp.STEP else "walk"
            samples.append({"qpos": synthetic_qpos(seed, arm),
                            "smooth_root_pos": synthetic_smooth_root(seed, arm)})
        return samples

    @staticmethod
    def to_qpos(sample):
        return sample["qpos"]


class BrokenPairingRunner(FakeRunner):
    def _draw(self, seeds):
        generators = [torch.Generator().manual_seed(int(seed) * 7 + index)
                      for index, seed in enumerate(seeds)]
        for generator in generators:
            torch.randn((4, 3), generator=generator)


class SecondChunkFailureRunner(FakeRunner):
    def generate(self, *args, **kwargs):
        if self.calls == 1:
            self.calls += 1
            raise RuntimeError("synthetic second-chunk failure")
        return super().generate(*args, **kwargs)


def fake_coverage(extent=(-0.25, 7.45)):
    """A coverage record built through the production sweep rule, not hand-written."""
    xs = np.linspace(0.30, 6.90, exp.SCAN_POINTS)
    mask = exp.swept_box_centres(extent, xs, exp.OBSTACLE_DEPTH_M)
    swept_xs = xs[mask]
    obstacles = {label: bool(exp.swept_box_centres(extent, [x], exp.OBSTACLE_DEPTH_M)[0])
                 for label, x in exp.OBSTACLES}
    return {
        "forward_extent_min_m": float(extent[0]), "forward_extent_max_m": float(extent[1]),
        "forward_axis": list(exp.FORWARD_AXIS),
        "envelope": "fake", "obstacle_depth_m": exp.OBSTACLE_DEPTH_M,
        "scan_points": exp.SCAN_POINTS, "scan_points_swept": int(mask.sum()),
        "scan_window_m": [float(xs[0]), float(xs[-1])],
        "swept_scan_range_m": ([float(swept_xs[0]), float(swept_xs[-1])]
                               if swept_xs.size else [None, None]),
        "excluded_max_box_height_m": None,
        "obstacles_swept": obstacles,
        "covers_all_obstacles": bool(all(obstacles.values())),
        "rule": exp.COVERAGE_RULE,
    }


def fake_reference_scorer(qpos, smooth_root_pos, arm, ctx, extent=(-0.25, 7.45)):
    """Deterministic stand-in for the mujoco scorer, shaped exactly like the real one."""
    seed = int(round((float(qpos[0, 7]) - (0.0 if arm == "step" else 0.5)) * 1e4))
    elicited = arm == "step" and seed % 4 != 0
    lift_height = 0.06 if elicited else (0.01 if seed % 8 == 1 else 0.0)
    lift_x = 1.2 if lift_height > 0 else None
    coverage = fake_coverage(extent)
    run = 0.5 if elicited else 0.1
    exact = {f"{h:g}": bool(elicited and h <= 0.05) for h in exp.GRADED_HEIGHTS_M}
    features = {"max_unsupported_run_s": run, "root_z_max": 0.97 if elicited else 0.78,
                "bilateral_flight_frac": 0.1, "ballistic_ratio": 2.0}
    timing = exp.lift_timing(qpos, lift_x, ctx)
    return {
        "arm": arm,
        "coverage": coverage,
        "elicitation": {
            "lift_x_m": lift_x, "lift_height_m": lift_height,
            "n_lift_regions": int(lift_height > 0), "lift_support_m": 0.3 * (lift_height > 0),
            "lift_side": "left" if lift_height > 0 else None,
            "elicited": bool(elicited), "any_lift": bool(lift_height > 0),
            "min_clearance_m": exp.ELICITATION_MIN_M, "scan_points": exp.SCAN_POINTS,
            "clears_height_anywhere": {f"{h:g}": bool(lift_height >= h)
                                       for h in exp.GRADED_HEIGHTS_M},
        },
        "timing": timing,
        "exact_boxes": {
            label: {"obstacle_x_m": float(x), "obstacle_depth_m": exp.OBSTACLE_DEPTH_M,
                    "body_swept": bool(coverage["obstacles_swept"][label]),
                    "max_box_height_lower_bound_m": (
                        (0.05 if (elicited and label == "staged") else 0.0)
                        if coverage["obstacles_swept"][label] else None),
                    "exact_clears": (
                        (exact if label == "staged" else {k: False for k in exact})
                        if coverage["obstacles_swept"][label]
                        else {k: None for k in exact}),
                    "probe": {"quantity": "max_box_height_lower_bound_m",
                              "evaluated": bool(coverage["obstacles_swept"][label])},
                    "not_reached_note": (None if coverage["obstacles_swept"][label]
                                         else "the body never swept this box")}
            for label, x in exp.OBSTACLES},
        "contract_features": features,
        "screen_predictions": {
            "max_unsupported_run_s": run,
            "primary_threshold_s": exp.PRIMARY_GATE_S, "primary_flag": bool(run > exp.PRIMARY_GATE_S),
            "secondary_threshold_s": exp.SECONDARY_GATE_S,
            "secondary_flag": bool(run > exp.SECONDARY_GATE_S)},
        "route_fidelity": exp.route_fidelity(smooth_root_pos, qpos, ctx.route),
    }


def generate_kwargs(output: Path, tmp_path: Path, runner_cls=FakeRunner, **overrides):
    ardy, kimodo = write_prompt_caches(tmp_path)
    kwargs = {
        "out": output,
        "runner_factory": lambda: runner_cls(Path(output)),
        "ardy_cache": ardy,
        "kimodo_cache": kimodo,
        "code_state_fn": clean_code_state,
        "source_hashes_fn": fake_source_hashes,
        "external_hashes_fn": fake_external_hashes,
        "text_cache_fn": cache_factory(ardy, kimodo),
        "generator_identity_fn": lambda _runner: {"generator": "fake"},
        "generator_identity_validator_fn": lambda value: dict(value),
        # generation binds the Kimodo venv's interpreter half ...
        "runtime_identity_fn": lambda: fake_runtime_identity("kimodo-venv"),
        "physical_identity_fn": lambda: {"physical": "fake"},
        "pin_validator_fn": lambda _g, _r, _p: None,
        "channel_usage_fn": lambda _runner, _spec: dict(exp.EXPECTED_CHANNEL_USAGE),
        "host_gate_fn": passing_gate,
    }
    kwargs.update(overrides)
    return kwargs


def stage_kwargs():
    # ... and every later stage runs under the *scoring* interpreter, whose sys.version and
    # numpy differ.  Only the checkout half is revalidated, so these must still pass.
    return {"code_state_fn": clean_code_state, "source_hashes_fn": fake_source_hashes,
            "external_hashes_fn": fake_external_hashes,
            "runtime_identity_fn": lambda: fake_runtime_identity("scoring-venv"),
            "physical_identity_fn": lambda: {"physical": "fake"}}


def score_kwargs(output: Path, scorer=fake_reference_scorer):
    return {
        "out": output, **stage_kwargs(),
        "support_thresholds_fn": lambda: {"support_height_m": 0.02, "support_speed_mps": 0.3,
                                          "max_unsupported_run_s": exp.PRIMARY_GATE_S},
        "scoring_context_fn": lambda route, support: SimpleNamespace(
            route=np.asarray(route, dtype=float), support=dict(support),
            modules=SimpleNamespace(aef=__import__(
                "experiments.analyze_event_frames", fromlist=["x"]))),
        "reference_scorer_fn": scorer,
    }


def run_through_score(output: Path, tmp_path: Path):
    exp.run_generate(**generate_kwargs(output, tmp_path))
    return exp.run_score(**score_kwargs(output))


# ------------------------------------------------------------------------- locked plans


def test_locked_plan_is_128_rows_of_four_seeds_by_two_arms_per_call():
    plan = exp.locked_row_plan()
    chunks = exp.locked_chunk_plan(plan)
    assert exp.SEEDS == tuple(range(4700, 4764))
    assert len(plan) == exp.N_ROWS == 128 and len(chunks) == 16
    assert [row["row_index"] for row in plan] == list(range(128))
    assert len({row["archive_key"] for row in plan}) == 128
    assert {row["seed"] for row in plan} == set(range(4700, 4764))
    assert [chunk["seeds"] for chunk in chunks] == [
        [4700 + 4 * c + i for i in range(4)] for c in range(16)]
    for chunk in chunks:
        assert chunk["batch_size"] == exp.BATCH_SIZE == 8
        assert [row["arm"] for row in chunk["rows"]] == list(exp.ARMS) * 4
        # every same-seed step/walk pair sits inside one B=8 call
        for seed in chunk["seeds"]:
            assert {f"s{seed}_{arm}" for arm in exp.ARMS} <= set(chunk["archive_keys"])
    assert [row["prompt"] for row in plan[:2]] == [exp.STEP, exp.WALK]


def test_batch_plan_hash_is_stable():
    plan = exp.locked_row_plan()
    assert exp._json_hash(plan) == exp._json_hash(exp.locked_row_plan())
    # Pinned: any drift in seeds, arms, prompts, order or keys changes this hash.
    assert exp._json_hash(plan) == (
        "d9fd522b0c0a3a663c15b86294757146de3f2f1f45184ab4584e2f8f316eb87c")


def test_planned_denominator_is_128_samples_on_64_seeds_in_two_arms():
    assert len(exp.SEEDS) == 64 and len(exp.ARMS) == 2
    assert exp.N_ROWS == len(exp.SEEDS) * len(exp.ARMS) == 128
    assert exp.N_CHUNKS * exp.BATCH_SIZE == exp.N_ROWS
    plan = exp.locked_row_plan()
    for arm in exp.ARMS:
        assert sum(1 for row in plan if row["arm"] == arm) == 64
        assert {row["seed"] for row in plan if row["arm"] == arm} == set(exp.SEEDS)


def test_route_is_the_locked_seven_point_two_metre_line_at_thirty_fps():
    route = exp.route_xz()
    assert route.shape == (240, 2)
    assert route[0].tolist() == [0.0, 0.0]
    assert route[-1].tolist() == [0.0, 7.2]
    assert np.allclose(route[:, 0], 0.0)
    assert exp.FPS == 30.0 and exp.N_FRAMES == 240
    assert math.isclose(exp.PROTOCOL_DURATION_S, 8.0)
    assert abs(exp.NOMINAL_SPEED_MPS - 0.9) < 0.01


# --------------------------------------------------------------- arm / constraint construction


def test_constraint_channel_is_smooth_root_2d_and_not_root_2d():
    spec = exp.campaign_spec()
    usage = exp.static_channel_usage(spec)
    assert usage == {"smooth_root_2d": exp.N_FRAMES} == dict(exp.EXPECTED_ADAPTER_CHANNELS)
    assert "root_2d" not in usage
    # The ARDY adapter on the same spec writes the *other* name; the rename is the whole point.
    data = {k: [] for k in ("root_2d", "global_root_heading", "root_y_pos",
                            "global_joints_rots", "global_joints_positions")}
    index = {k: [] for k in data}
    ArdyConstraintSet(spec, root_idx=0, device="cpu").update_constraints(data, index)
    assert [k for k, v in data.items() if v] == ["root_2d"]


def test_campaign_spec_is_the_free_contract():
    spec = exp.campaign_spec()
    assert spec.heading is None and spec.root_y is None
    assert spec.first_heading == exp.FIRST_HEADING == 0.0
    assert spec.T == exp.N_FRAMES
    assert np.allclose(spec.root_xz, exp.route_xz())
    assert exp.CONSTRAINT_CONTRACT == "free"
    assert exp.spec_sha256(spec) == exp.spec_sha256(exp.campaign_spec())


class FakeMotionRep:
    """Mirrors kimodo_motionrep.py:242-251 (the ``smooth_root_2d`` filler) and :33-41."""

    def __init__(self):
        self.slice_dict = {"smooth_root_pos": slice(0, 3), "global_root_heading": slice(3, 5),
                           "local_joints_positions": slice(5, 8), "global_rot_data": slice(8, 14),
                           "velocities": slice(14, 17), "foot_contacts": slice(17, 21)}
        self.dim = 21

    def create_conditions_from_constraints_batched(self, constraints, lengths, to_normalize,
                                                   device):
        length = int(lengths.max())
        observed = torch.zeros(length, self.dim)
        mask = torch.zeros(length, self.dim, dtype=bool)
        data: dict = {k: [] for k in ("smooth_root_2d", "global_root_heading", "root_y_pos",
                                      "global_joints_rots", "global_joints_positions")}
        index: dict = {k: [] for k in data}
        for item in constraints:
            item.update_constraints(data, index)
        if index["smooth_root_2d"]:
            frames = torch.cat(index["smooth_root_2d"])
            values = torch.cat(data["smooth_root_2d"])
            block = observed[:, self.slice_dict["smooth_root_pos"]]
            block[frames, 0] = values[:, 0]
            block[frames, 2] = values[:, 1]
            block_mask = mask[:, self.slice_dict["smooth_root_pos"]]
            block_mask[frames, 0] = True
            block_mask[frames, 2] = True
        if index["root_y_pos"]:
            mask[:, self.slice_dict["smooth_root_pos"]][torch.cat(index["root_y_pos"]), 1] = True
        return observed[None], mask[None]


def test_actual_channel_usage_marks_only_the_smooth_root_ground_plane():
    runner = SimpleNamespace(
        model=SimpleNamespace(motion_rep=FakeMotionRep(),
                              skeleton=SimpleNamespace(root_idx=0)),
        device="cpu")
    usage = exp._actual_channel_usage(runner, exp.campaign_spec())
    assert usage == {"smooth_root_pos": 2 * exp.N_FRAMES} == dict(exp.EXPECTED_CHANNEL_USAGE)


def test_generate_refuses_a_model_that_conditions_the_wrong_channels(tmp_path):
    out = tmp_path / "campaign"
    with pytest.raises(exp.CampaignAbort, match="conditions the wrong channels"):
        exp.run_generate(**generate_kwargs(
            out, tmp_path, channel_usage_fn=lambda _r, _s: {"smooth_root_pos": 3 * exp.N_FRAMES}))
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["blocked"] is True and receipt["actual_kimodo_samples"] == 0


# ------------------------------------------------------- prompt embeddings and provenance


def test_campaign_cache_copies_the_ardy_step_vector_under_kimodos_canonical_key(tmp_path):
    ardy, kimodo = write_prompt_caches(tmp_path)
    identity = exp.build_campaign_text_cache(
        tmp_path / "cache.npz", ardy_cache=ardy, kimodo_cache=kimodo,
        sanitize_fn=identity_sanitize, encoder_equivalence_fn=stub_equivalence)
    fields = identity["fields"]
    step, walk = fields["prompts"]["step"], fields["prompts"]["walk"]
    assert step["prompt"] == exp.STEP and walk["prompt"] == exp.WALK
    assert step["copied_from_ardy_cache"] is True
    assert walk["copied_from_ardy_cache"] is False
    assert step["canonical_key_sha1"] == step["raw_key_sha1"] == kr._raw_key(exp.STEP)
    assert step["keys_coincide"] is True and step["sanitize_is_identity"] is True
    assert walk["canonical_key_sha1"] == kr._raw_key(exp.WALK)
    assert step["shape"] == walk["shape"] == [1, LLM_DIM]
    with np.load(tmp_path / "cache.npz") as written:
        assert set(written.files) == {step["canonical_key_sha1"], walk["canonical_key_sha1"]}
        with np.load(ardy) as source:
            assert np.array_equal(written[step["canonical_key_sha1"]],
                                  source[kr._raw_key(exp.STEP)])
        with np.load(kimodo) as source:
            assert np.array_equal(written[walk["canonical_key_sha1"]],
                                  source[kr._raw_key(exp.WALK)])
    assert exp._is_sha256(step["content_sha256"]) and exp._is_sha256(walk["content_sha256"])
    assert step["content_sha256"] != walk["content_sha256"]
    assert fields["encoder_loaded"] is False
    assert identity["sha256"] == exp._identity(identity["schema"], fields)["sha256"]


def test_campaign_cache_refuses_when_sanitize_moves_the_step_key(tmp_path):
    ardy, kimodo = write_prompt_caches(tmp_path)
    with pytest.raises(ValueError, match="canonical cache key no longer coincides"):
        exp.build_campaign_text_cache(
            tmp_path / "cache.npz", ardy_cache=ardy, kimodo_cache=kimodo,
            sanitize_fn=lambda text, paragraph=True: text.lower(),
            encoder_equivalence_fn=stub_equivalence)
    assert not (tmp_path / "cache.npz").exists()


def test_campaign_cache_refuses_a_missing_or_malformed_embedding(tmp_path):
    ardy, kimodo = write_prompt_caches(tmp_path)
    empty = tmp_path / "empty.npz"
    np.savez(empty, **{"deadbeef": np.zeros((1, LLM_DIM), dtype=np.float32)})
    with pytest.raises(ValueError, match="cached embedding is missing"):
        exp.build_campaign_text_cache(tmp_path / "a.npz", ardy_cache=empty, kimodo_cache=kimodo,
                                      sanitize_fn=identity_sanitize,
                                      encoder_equivalence_fn=stub_equivalence)
    wrong = tmp_path / "wrong.npz"
    np.savez(wrong, **{kr._raw_key(exp.WALK): np.zeros((1, 7), dtype=np.float32)})
    with pytest.raises(ValueError, match="has shape"):
        exp.build_campaign_text_cache(tmp_path / "b.npz", ardy_cache=ardy, kimodo_cache=wrong,
                                      sanitize_fn=identity_sanitize,
                                      encoder_equivalence_fn=stub_equivalence)


def test_verify_runner_text_cache_catches_a_swapped_embedding(tmp_path):
    ardy, kimodo = write_prompt_caches(tmp_path)
    identity = exp.build_campaign_text_cache(
        tmp_path / "cache.npz", ardy_cache=ardy, kimodo_cache=kimodo,
        sanitize_fn=identity_sanitize, encoder_equivalence_fn=stub_equivalence)
    with np.load(tmp_path / "cache.npz") as cache:
        entries = {key: np.array(cache[key], copy=True) for key in cache.files}

    class Holder:
        def __init__(self, values):
            self._text_cache = values

        def _cache_key(self, text):
            key = kr._raw_key(identity_sanitize(text))
            return key if key in self._text_cache else None

    assert exp.verify_runner_text_cache(
        Holder(entries), identity)["runner_memory_byte_matches_cache"] is True
    swapped = dict(entries)
    step_key = identity["fields"]["prompts"]["step"]["canonical_key_sha1"]
    swapped[step_key] = swapped[step_key] + 1.0
    with pytest.raises(ValueError, match="does not byte-match"):
        exp.verify_runner_text_cache(Holder(swapped), identity)
    with pytest.raises(ValueError, match="does not hold the cached"):
        exp.verify_runner_text_cache(Holder({k: v for k, v in entries.items()
                                             if k != step_key}), identity)


def _write_wrapper(path: Path, docstring: str, to_body: str, call_body: str) -> Path:
    path.write_text(
        f'"""{docstring}"""\n\n\nclass LLM2VecEncoder:\n'
        '    """LLM2Vec text embeddings."""\n\n'
        f'    def to(self, device):\n        {to_body}\n\n'
        f'    def __call__(self, text):\n        {call_body}\n')
    return path


def _write_load_model(path: Path, target: str, kwargs: dict) -> Path:
    path.write_text("TEXT_ENCODER_PRESETS = {\n"
                    f"    'llm2vec': {{'target': {target!r}, 'kwargs': {kwargs!r}}}\n}}\n")
    return path


def test_encoder_equivalence_accepts_docstring_and_device_helper_differences(tmp_path):
    a = _write_wrapper(tmp_path / "a.py", "ARDY wrapper", "return self.model.to(device)",
                       "return self.model.encode(text, batch_size=1)")
    b = _write_wrapper(tmp_path / "b.py", "Kimodo wrapper", "self.model = self.model.to(device)",
                       "return self.model.encode(text, batch_size=1)")
    la = _write_load_model(tmp_path / "la.py", "ardy.model.LLM2VecEncoder",
                           dict(exp.EXPECTED_LLM2VEC_KWARGS))
    lb = _write_load_model(tmp_path / "lb.py", "kimodo.model.LLM2VecEncoder",
                           dict(exp.EXPECTED_LLM2VEC_KWARGS))
    report = exp.encoder_equivalence_report(ardy_wrapper=a, kimodo_wrapper=b,
                                            ardy_load_model=la, kimodo_load_model=lb)
    assert report["equivalent"] is True
    assert report["wrappers_equal_after_normalisation"] is True
    assert report["preset_kwargs_equal"] is True
    assert report["expected_difference"]["target"] == {
        "ardy": "ardy.model.LLM2VecEncoder", "kimodo": "kimodo.model.LLM2VecEncoder"}
    assert report["normalized_ast_sha256"]["ardy"] == report["normalized_ast_sha256"]["kimodo"]


def test_encoder_equivalence_rejects_a_changed_encoding_path(tmp_path):
    a = _write_wrapper(tmp_path / "a.py", "ARDY wrapper", "return self",
                       "return self.model.encode(text, batch_size=1)")
    b = _write_wrapper(tmp_path / "b.py", "Kimodo wrapper", "return self",
                       "return self.model.encode(text, batch_size=8)")
    la = _write_load_model(tmp_path / "la.py", "a", dict(exp.EXPECTED_LLM2VEC_KWARGS))
    lb = _write_load_model(tmp_path / "lb.py", "b", dict(exp.EXPECTED_LLM2VEC_KWARGS))
    with pytest.raises(ValueError, match="not equivalent"):
        exp.encoder_equivalence_report(ardy_wrapper=a, kimodo_wrapper=b,
                                       ardy_load_model=la, kimodo_load_model=lb)


def test_encoder_equivalence_rejects_a_different_llm2vec_preset(tmp_path):
    a = _write_wrapper(tmp_path / "a.py", "ARDY wrapper", "return self", "return 1")
    b = _write_wrapper(tmp_path / "b.py", "Kimodo wrapper", "return self", "return 1")
    la = _write_load_model(tmp_path / "la.py", "a", dict(exp.EXPECTED_LLM2VEC_KWARGS))
    lb = _write_load_model(tmp_path / "lb.py", "b",
                           {**exp.EXPECTED_LLM2VEC_KWARGS, "llm_dim": 2048})
    with pytest.raises(ValueError, match="not equivalent"):
        exp.encoder_equivalence_report(ardy_wrapper=a, kimodo_wrapper=b,
                                       ardy_load_model=la, kimodo_load_model=lb)


@requires_external
def test_released_ardy_and_kimodo_encoders_are_equivalent():
    report = exp.encoder_equivalence_report()
    assert report["equivalent"] is True
    assert report["preset_kwargs"]["ardy"] == dict(exp.EXPECTED_LLM2VEC_KWARGS)
    assert report["preset_kwargs"]["kimodo"] == dict(exp.EXPECTED_LLM2VEC_KWARGS)
    assert (report["wrapper_sources"]["ardy"]["sha256"]
            != report["wrapper_sources"]["kimodo"]["sha256"])


@requires_external
def test_kimodo_sanitize_is_the_identity_on_both_campaign_prompts():
    sanitize = exp.load_sanitize_text()
    assert sanitize(exp.STEP) == exp.STEP
    assert sanitize(exp.WALK) == exp.WALK
    assert kr._raw_key(sanitize(exp.STEP)) == kr._raw_key(exp.STEP)


@requires_external
def test_the_two_checkouts_ship_the_same_released_g1_xml():
    identity = exp.physical_model_identity()
    assert identity["fields"]["sha256"] == exp.PINNED_G1_XML_SHA256
    assert identity["fields"]["kimodo_copy_sha256"] == exp.PINNED_G1_XML_SHA256


# ----------------------------------------------------------------------- route fidelity


def test_route_fidelity_is_measured_against_smooth_root_not_the_pelvis():
    """The amendment, made mechanical.

    ``smooth_root_2d`` constrains the smoothed root, so a clip whose *smooth root* sits exactly
    on the requested path has zero route error even when the pelvis sways 6 cm around it.
    Measuring against ``qpos`` instead would report 0.06 m here and bias the cross-prior
    comparison against Kimodo -- so this test fails the moment anyone switches it.
    """
    route = exp.route_xz()
    smooth = np.zeros((exp.N_FRAMES, 3))
    smooth[:, 0] = route[:, 0]
    smooth[:, 1] = 0.78
    smooth[:, 2] = route[:, 1]
    qpos = np.zeros((exp.N_FRAMES, 36))
    qpos[:, 0] = route[:, 1]
    qpos[:, 1] = 0.06                              # pelvis sway, lateral (mujoco y)
    qpos[:, 2] = 0.78
    qpos[:, 3] = 1.0
    fidelity = exp.route_fidelity(smooth, qpos, route)
    assert fidelity["measured_against"] == "smooth_root_pos"
    assert fidelity["constrained_channel"] == "smooth_root_2d"
    assert fidelity["smooth_root_path_mae_m"] == pytest.approx(0.0, abs=1e-12)
    assert fidelity["smooth_root_path_max_m"] == pytest.approx(0.0, abs=1e-12)
    assert fidelity["pelvis_path_mae_m"] == pytest.approx(0.06, abs=1e-12)
    assert fidelity["pelvis_minus_smooth_root_mae_m"] == pytest.approx(0.06, abs=1e-12)
    assert fidelity["smooth_root_progress_ratio"] == pytest.approx(1.0)
    assert fidelity["forward_axis_dominant"] is True


def test_route_fidelity_reports_a_smooth_root_error_the_pelvis_does_not_show():
    """The converse: a clip whose pelvis is perfect but whose smooth root is off by 5 cm."""
    route = exp.route_xz()
    smooth = np.zeros((exp.N_FRAMES, 3))
    smooth[:, 0] = route[:, 0] + 0.05
    smooth[:, 2] = route[:, 1]
    qpos = np.zeros((exp.N_FRAMES, 36))
    qpos[:, 0] = route[:, 1]
    qpos[:, 3] = 1.0
    fidelity = exp.route_fidelity(smooth, qpos, route)
    assert fidelity["smooth_root_path_mae_m"] == pytest.approx(0.05)
    assert fidelity["smooth_root_lateral_mae_m"] == pytest.approx(0.05)
    assert fidelity["smooth_root_forward_mae_m"] == pytest.approx(0.0, abs=1e-12)
    assert fidelity["pelvis_path_mae_m"] == pytest.approx(0.0, abs=1e-12)


def test_route_fidelity_rejects_a_malformed_smooth_root():
    route = exp.route_xz()
    qpos = np.zeros((exp.N_FRAMES, 36))
    qpos[:, 0] = route[:, 1]
    with pytest.raises(ValueError, match="smooth_root_pos must be"):
        exp.route_fidelity(np.zeros((exp.N_FRAMES, 2)), qpos, route)
    bad = np.zeros((exp.N_FRAMES, 3))
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="smooth_root_pos must be"):
        exp.route_fidelity(bad, qpos, route)


def test_validated_reference_score_refuses_a_pelvis_measured_route_error():
    route = exp.route_xz()
    smooth = np.zeros((exp.N_FRAMES, 3))
    smooth[:, 2] = route[:, 1]
    qpos = synthetic_qpos(4700, "step")
    ctx = SimpleNamespace(route=route)
    score = fake_reference_scorer(qpos, smooth, "step", ctx)
    exp.validated_reference_score(score)                       # the honest record validates
    score["route_fidelity"] = {**score["route_fidelity"], "measured_against": "qpos"}
    with pytest.raises(ValueError, match="measured against smooth_root_pos"):
        exp.validated_reference_score(score)


# ------------------------------------------------------------------- coverage guard


def test_swept_box_centres_requires_the_whole_box_inside_the_swept_extent():
    mask = exp.swept_box_centres((0.0, 2.0), [0.05, 0.10, 1.00, 1.90, 1.95], 0.20)
    assert list(mask) == [False, True, True, True, False]
    assert list(exp.swept_box_centres((0.0, 0.1), [1.2, 3.6], 0.20)) == [False, False]


def test_body_forward_extent_measures_the_whole_envelope_not_the_pelvis():
    from scene2motion.robot import G1Body

    body = G1Body(None)
    qpos = synthetic_qpos(4700, "step")          # pelvis runs 0.0 -> 7.2 m
    low, high = exp.body_forward_extent(body, qpos)
    assert low < 0.0 and high > 7.2              # feet and margin reach past the pelvis
    # the same measurement, done frame by frame through the public geom API
    normal = np.asarray(exp.FORWARD_AXIS, dtype=float)
    head, tail = qpos[:3], qpos[-3:]
    reference_low, reference_high = math.inf, -math.inf
    for frame in np.concatenate([head, tail]):
        body.fk(frame)
        for geom in body.robot_geoms:
            lo, hi = body.geom_extent(geom, normal, extra_margin=body.body_margin)
            reference_low, reference_high = min(reference_low, lo), max(reference_high, hi)
    assert exp.body_forward_extent(body, head)[0] == pytest.approx(reference_low)
    assert exp.body_forward_extent(body, tail)[1] == pytest.approx(reference_high)


def test_a_clip_that_stalls_short_is_never_credited_with_clearance(tmp_path, monkeypatch):
    """The real scorer, on a clip whose root advances only 0 -> 2.0 m.

    ``BoxHeightProbe`` returns its 0.40 m cap and "collision-free" for a box the body never
    reaches, so without the coverage guard this clip would score as maximally elicited at a
    position it never visited and as clearing every graded height at x = 3.6 m.  Both are
    asserted here: the vacuous probe still says yes, and the scored record says "not reached".
    """
    monkeypatch.setattr(exp, "SCAN_POINTS", 8)
    support = {"support_height_m": 0.02, "support_speed_mps": 0.3,
               "max_unsupported_run_s": exp.PRIMARY_GATE_S}
    ctx = exp.build_scoring_context(exp.route_xz(), support)
    stalled = synthetic_qpos(4700, "step")
    stalled[:, 0] = np.linspace(0.0, 2.0, exp.N_FRAMES)
    smooth = synthetic_smooth_root(4700, "step")
    smooth[:, 2] = stalled[:, 0]

    # what the unguarded endpoints would have said: a 0.40 m "lift" at 3.13 m, and a clean
    # sweep of every graded height at the 3.6 m box the robot never came near
    assert ctx.probes["unstaged"].clears(stalled, 0.30) is True
    assert ctx.probes["unstaged"].probe(stalled) == pytest.approx(0.40)
    xs, heights = ctx.modules.box_height_profile(stalled, exp.route_xz(),
                                                 exp.OBSTACLE_DEPTH_M,
                                                 n_points=exp.SCAN_POINTS)
    unguarded = ctx.modules.lift_location(xs, heights)
    assert unguarded["lift_x_m"] == pytest.approx(3.13, abs=0.01)
    assert unguarded["lift_height_m"] == pytest.approx(0.40)

    score = exp.validated_reference_score(
        exp.score_reference_clip(stalled, smooth, "step", ctx))
    coverage = score["coverage"]
    assert coverage["obstacles_swept"] == {"staged": True, "unstaged": False}
    assert coverage["covers_all_obstacles"] is False
    assert coverage["forward_extent_max_m"] < 3.5
    assert 0 < coverage["scan_points_swept"] < exp.SCAN_POINTS
    unstaged = score["exact_boxes"]["unstaged"]
    assert unstaged["body_swept"] is False
    assert unstaged["max_box_height_lower_bound_m"] is None
    assert set(unstaged["exact_clears"].values()) == {None}
    assert unstaged["probe"]["evaluated"] is False
    # ... and the guarded elicitation drops the 0.40 m phantom entirely
    assert coverage["excluded_max_box_height_m"] == pytest.approx(0.40)
    assert score["elicitation"]["lift_x_m"] is None
    assert score["elicitation"]["lift_height_m"] == 0.0
    assert score["elicitation"]["elicited"] is False and score["elicitation"]["any_lift"] is False
    assert set(score["elicitation"]["clears_height_anywhere"].values()) == {False}
    assert score["elicitation"]["scan_points_swept"] == coverage["scan_points_swept"]


def test_the_same_clip_scores_every_scan_point_when_it_traverses_the_route(tmp_path,
                                                                          monkeypatch):
    """The guard must not bite a clip that actually walks the route."""
    monkeypatch.setattr(exp, "SCAN_POINTS", 8)
    support = {"support_height_m": 0.02, "support_speed_mps": 0.3,
               "max_unsupported_run_s": exp.PRIMARY_GATE_S}
    ctx = exp.build_scoring_context(exp.route_xz(), support)
    score = exp.validated_reference_score(exp.score_reference_clip(
        synthetic_qpos(4700, "step"), synthetic_smooth_root(4700, "step"), "step", ctx))
    assert score["coverage"]["scan_points_swept"] == exp.SCAN_POINTS
    assert score["coverage"]["covers_all_obstacles"] is True
    assert all(box["body_swept"] for box in score["exact_boxes"].values())
    assert all(isinstance(flag, bool)
               for box in score["exact_boxes"].values()
               for flag in box["exact_clears"].values())


def test_validated_reference_score_refuses_a_clearance_at_an_unswept_box():
    """A "not reached" box may be null, never ``True`` and never the probe's cap."""
    support = {"support_height_m": 0.02, "support_speed_mps": 0.3,
               "max_unsupported_run_s": exp.PRIMARY_GATE_S}
    ctx = SimpleNamespace(route=exp.route_xz(), support=support,
                          modules=SimpleNamespace(aef=__import__(
                              "experiments.analyze_event_frames", fromlist=["x"])))
    score = fake_reference_scorer(synthetic_qpos(4701, "step"),
                                  synthetic_smooth_root(4701, "step"), "step", ctx,
                                  extent=(-0.25, 2.3))
    validated = exp.validated_reference_score(score)
    assert validated["exact_boxes"]["unstaged"]["body_swept"] is False
    poisoned = json.loads(json.dumps(score))
    poisoned["exact_boxes"]["unstaged"]["exact_clears"]["0.03"] = True
    with pytest.raises(ValueError, match="must be null where the body never swept"):
        exp.validated_reference_score(poisoned)
    capped = json.loads(json.dumps(score))
    capped["exact_boxes"]["unstaged"]["max_box_height_lower_bound_m"] = 0.40
    with pytest.raises(ValueError, match="height bound must be null"):
        exp.validated_reference_score(capped)
    displaced = json.loads(json.dumps(score))
    displaced["elicitation"]["lift_x_m"] = 3.13
    with pytest.raises(ValueError, match="outside the swept scan window"):
        exp.validated_reference_score(displaced)


def test_a_clip_that_never_reached_the_box_counts_as_not_clearing_over_all_trials():
    records = []
    for index, seed in enumerate(exp.SEEDS):
        clears = index < 20
        records.append(_record(seed, "step", elicited=clears, lift=0.06 if clears else 0.0,
                               t=1.0 if clears else None, run=0.5 if clears else 0.05,
                               clears5=clears, swept=index >= 12))
        records.append(_record(seed, "walk"))
    step = exp.build_summary(records)["arms"]["step"]
    # 20 clips would have cleared the staged 5 cm box; the 12 that never swept it do not count
    assert step["exact_clearance"]["staged"]["0.05"] == {
        "k": 8, "n": 64, "rate": pytest.approx(8 / 64),
        "wilson95": pytest.approx(exp.wilson(8, 64))}
    coverage = step["coverage"]
    assert coverage["obstacles"]["staged"] == {
        "body_swept": {"k": 52, "n": 64, "rate": pytest.approx(52 / 64),
                       "wilson95": pytest.approx(exp.wilson(52, 64))},
        "not_reached": 12}
    assert coverage["covers_all_obstacles"]["k"] == 52
    assert coverage["denominator"] == "all assigned trials"
    assert step["max_box_height_lower_bound_m"]["staged"]["n"] == 52


# --------------------------------------------------------------------- timing endpoints


def test_lift_timing_is_reported_in_seconds_on_the_thirty_fps_grid():
    ctx = SimpleNamespace(modules=SimpleNamespace(
        aef=__import__("experiments.analyze_event_frames", fromlist=["x"])))
    qpos = synthetic_qpos(4700, "step")
    timing = exp.lift_timing(qpos, 1.2, ctx)
    assert timing["fps"] == 30.0 and timing["early_window_s"] == 2.0
    frame = timing["root_crossing_frame"]
    assert timing["lift_time_root_crossing_s"] == pytest.approx(frame / 30.0)
    assert timing["lift_time_root_crossing_s"] == pytest.approx(1.2 / exp.NOMINAL_SPEED_MPS,
                                                                abs=1.0 / 30.0)
    assert timing["within_first_2s_root_crossing"] is True
    assert timing["lift_time_nominal_s"] == pytest.approx(1.2 / exp.NOMINAL_SPEED_MPS)
    late = exp.lift_timing(qpos, 6.0, ctx)
    assert late["within_first_2s_root_crossing"] is False
    assert late["within_first_2s_nominal"] is False
    absent = exp.lift_timing(qpos, None, ctx)
    assert absent["lift_time_root_crossing_s"] is None
    assert absent["lift_time_nominal_s"] is None


def test_support_thresholds_are_seconds_and_so_survive_the_fps_change():
    """The screen's thresholds are durations, so only the conversion changes with fps."""
    from experiments import analyze_trackability_contract as atc
    from scene2motion.robot import G1Body

    body = G1Body(None)
    qpos = np.zeros((60, 36))
    qpos[:, 0] = np.linspace(0.0, 1.5, 60)
    qpos[:, 2] = 0.78
    qpos[:, 3] = 1.0
    at25 = atc.features(body, qpos, 0.02, 0.3, 25.0)
    at30 = atc.features(body, qpos, 0.02, 0.3, 30.0)
    assert at25["bilateral_flight_frac"] == pytest.approx(at30["bilateral_flight_frac"])
    # the same frame count, converted by the respective fps
    assert at25["max_unsupported_run_s"] * 25.0 == pytest.approx(
        at30["max_unsupported_run_s"] * 30.0)
    assert exp.PRIMARY_GATE_S == atc.PRIMARY_GATE_S
    assert exp.SECONDARY_GATE_S == atc.SECONDARY_GATE_S


# --------------------------------------------------------------- summary arithmetic / Wilson


def test_wilson_matches_the_contract_analyser():
    from experiments import analyze_trackability_contract as atc

    for n in (1, 5, 8, 11, 44, 49, 64, 128):
        for k in (0, 1, n // 2, n):
            assert exp.wilson(k, n) == pytest.approx(atc.wilson(k, n))
    assert all(math.isnan(value) for value in exp.wilson(0, 0))


def test_rate_carries_its_planned_denominator_and_never_passes_on_zero():
    assert exp.rate(44, 64) == {"k": 44, "n": 64, "rate": pytest.approx(44 / 64),
                                "wilson95": pytest.approx(exp.wilson(44, 64))}
    empty = exp.rate(0, 0)
    assert empty == {"k": 0, "n": 0, "rate": None, "wilson95": [None, None]}
    with pytest.raises(ValueError):
        exp.rate(5, 4)


def _record(seed, arm, *, elicited=False, lift=0.0, t=None, run=0.0, clears5=False,
            lift_x=None, mae=0.02, swept=True, t_nominal="same", extent=(-0.25, 7.45)):
    """One analysis record.  ``t`` is the root-crossing time; ``t_nominal`` defaults to it.

    ``swept=False`` is the clip that stalled short: its obstacle outcomes are null (not
    reached), never ``False`` and never ``True``.
    """
    nominal = t if t_nominal == "same" else t_nominal
    return {
        "seed": seed, "arm": arm, "key": f"s{seed}_{arm}", "prompt": exp.ARM_PROMPTS[arm],
        "elicited": bool(elicited), "any_lift": bool(lift > 0.0),
        "lift_height_m": float(lift),
        "lift_x_m": (lift_x if lift_x is not None else (1.2 if lift > 0 else None)),
        "lift_side": "left" if lift > 0 else None,
        "lift_time_root_crossing_s": t, "lift_time_nominal_s": nominal,
        "within_first_2s_root_crossing": (None if t is None else bool(t < 2.0)),
        "within_first_2s_nominal": (None if nominal is None else bool(nominal < 2.0)),
        "exact_clears": {label: {f"{h:g}": (bool(clears5 and h <= 0.05 and label == "staged")
                                            if swept else None)
                                 for h in exp.GRADED_HEIGHTS_M}
                         for label, _ in exp.OBSTACLES},
        "body_swept": {label: bool(swept) for label, _ in exp.OBSTACLES},
        "max_box_height_lower_bound_m": {label: ((0.05 if clears5 else 0.0) if swept else None)
                                         for label, _ in exp.OBSTACLES},
        "forward_extent_min_m": float(extent[0]),
        "forward_extent_max_m": float(extent[1] if swept else 2.3),
        "covers_all_obstacles": bool(swept),
        "scan_points_swept": int(exp.SCAN_POINTS if swept else 30),
        "max_unsupported_run_s": float(run),
        "primary_flag": bool(run > exp.PRIMARY_GATE_S),
        "secondary_flag": bool(run > exp.SECONDARY_GATE_S),
        "ballistic_ratio": 2.0, "root_z_max": 0.9,
        "smooth_root_path_mae_m": float(mae), "pelvis_path_mae_m": float(mae) + 0.06,
        "smooth_root_progress_ratio": 1.0,
    }


def synthetic_records(*, n_elicited_step=44, step_time=1.0, step_run=0.5, n_clears5=12,
                      walk_elicited=0):
    records = []
    for index, seed in enumerate(exp.SEEDS):
        elicited = index < n_elicited_step
        records.append(_record(seed, "step", elicited=elicited,
                               lift=0.06 if elicited else 0.0,
                               t=step_time if elicited else None,
                               run=step_run if elicited else 0.05,
                               clears5=index < n_clears5))
        walk = index < walk_elicited
        records.append(_record(seed, "walk", elicited=walk, lift=0.04 if walk else 0.0,
                               t=1.0 if walk else None, run=0.05))
    return records


def test_arm_summary_arithmetic_and_denominators():
    records = synthetic_records(n_elicited_step=44, n_clears5=12)
    summary = exp.build_summary(records)
    step = summary["arms"]["step"]
    assert step["n_assigned"] == 64
    assert step["elicitation"]["k"] == 44 and step["elicitation"]["n"] == 64
    assert step["elicitation"]["rate"] == pytest.approx(44 / 64)
    assert step["elicitation"]["wilson95"] == pytest.approx(exp.wilson(44, 64))
    assert step["exact_clearance"]["staged"]["0.05"]["k"] == 12
    assert step["exact_clearance"]["staged"]["0.08"]["k"] == 0
    assert step["exact_clearance"]["unstaged"]["0.05"]["k"] == 0
    assert step["screen"]["n_elicited"] == 44
    assert step["screen"]["float_primary_0p20s"] == {
        "k": 44, "n": 44, "rate": 1.0, "wilson95": pytest.approx(exp.wilson(44, 44))}
    assert step["screen"]["float_primary_over_all_assigned"]["n"] == 64
    assert step["lift_time_s"]["root_crossing"]["n_timed"] == 44
    assert step["lift_time_s"]["root_crossing"]["within_first_2s"]["k"] == 44
    assert step["lift_time_s"]["root_crossing"]["q0.5"] == pytest.approx(1.0)
    assert summary["arms"]["walk"]["elicitation"]["k"] == 0
    assert summary["arms"]["walk"]["screen"]["float_primary_0p20s"]["n"] == 0
    assert summary["arms"]["walk"]["screen"]["float_primary_0p20s"]["rate"] is None
    assert summary["n_clips"] == 128 and summary["planned_n_per_arm"] == 64
    paired = summary["paired_step_minus_walk"]
    assert paired["n_pairs"] == 64
    assert paired["elicitation_discordant_pairs"] == {"step_only": 44, "walk_only": 0,
                                                      "concordant": 20}
    assert paired["interval_claimed_on_difference"] is False
    assert paired["median_smooth_root_path_mae_difference_m"] == pytest.approx(0.0)


def test_build_summary_requires_the_planned_denominator():
    records = synthetic_records()
    with pytest.raises(ValueError, match="64 records"):
        exp.build_summary(records[:-1])


def test_route_fidelity_summary_reports_both_quantities():
    summary = exp.build_summary(synthetic_records())
    fidelity = summary["arms"]["step"]["route_fidelity"]
    assert fidelity["smooth_root_path_mae_m"]["q0.5"] == pytest.approx(0.02)
    assert fidelity["pelvis_path_mae_m"]["q0.5"] == pytest.approx(0.08)


# ------------------------------------------------------------------------- decision rules


@pytest.mark.parametrize("fraction, outcome", [
    (1.0, "timing_generalises_to_released_g1_priors"),
    (0.75, "timing_generalises_to_released_g1_priors"),
    (0.5, "report_both_distributions_without_mechanism_claim"),
    (0.25, "ardy_window_attributed_to_autoregressive_rollout_context"),
    (0.0, "ardy_window_attributed_to_autoregressive_rollout_context"),
])
def test_timing_decision_rule_branches(fraction, outcome):
    n_elicited = 40
    early = int(round(fraction * n_elicited))
    records = []
    for index, seed in enumerate(exp.SEEDS):
        elicited = index < n_elicited
        records.append(_record(seed, "step", elicited=elicited, lift=0.06 if elicited else 0.0,
                               t=(1.0 if index < early else 3.0) if elicited else None,
                               run=0.5 if elicited else 0.05))
        records.append(_record(seed, "walk"))
    decisions = exp.evaluate_decisions(records)
    rule = decisions["timing_rule"]
    assert rule["definitions"]["root_crossing"]["fraction"] == pytest.approx(fraction)
    assert rule["outcome"] == outcome
    assert rule["definitions_agree"] is True
    assert rule["thresholds"] == {"generalises_if_fraction_at_least": 0.7,
                                  "rollout_context_if_fraction_at_most": 0.4}
    assert rule["definitions"]["root_crossing"]["over_all_assigned_step_trials"]["n"] == 64


@pytest.mark.parametrize("fraction, boundary", [(0.7, "timing_generalises_to_released_g1_priors"),
                                                (0.4, "ardy_window_attributed_to_"
                                                      "autoregressive_rollout_context")])
def test_timing_decision_rule_is_inclusive_at_both_boundaries(fraction, boundary):
    n_elicited = 40
    early = int(round(fraction * n_elicited))
    records = []
    for index, seed in enumerate(exp.SEEDS):
        elicited = index < n_elicited
        records.append(_record(seed, "step", elicited=elicited, lift=0.06 if elicited else 0.0,
                               t=(1.0 if index < early else 3.0) if elicited else None,
                               run=0.5 if elicited else 0.05))
        records.append(_record(seed, "walk"))
    assert exp.evaluate_decisions(records)["timing_rule"]["outcome"] == boundary


def test_timing_decision_rule_is_indeterminate_without_elicited_clips():
    records = []
    for seed in exp.SEEDS:
        records.append(_record(seed, "step"))
        records.append(_record(seed, "walk"))
    decisions = exp.evaluate_decisions(records)
    assert decisions["timing_rule"]["outcome"] == "indeterminate_no_clips_with_a_lift_position"
    assert decisions["screen_rule"]["outcome"] == "indeterminate_no_elicited_clips"
    assert decisions["screen_rule"]["fraction"] is None
    assert decisions["screen_rule"]["float_primary_0p20s"]["rate"] is None


def test_timing_rule_counts_a_clip_without_an_event_time_as_not_early():
    """A root that never reaches the lift position did not reach it inside 2.0 s either.

    Dropping those clips from the denominator can only raise the fraction, and they are
    exactly the never/latest events -- the exclusion is correlated with the outcome being
    measured.  Here ten clips lift, five cross at 1.0 s and five never arrive: the rule reads
    5/10, not the flattering 5/5.
    """
    records = []
    for index, seed in enumerate(exp.SEEDS):
        elicited = index < 10
        records.append(_record(seed, "step", elicited=elicited, lift=0.06 if elicited else 0.0,
                               t=(1.0 if index < 5 else None) if elicited else None,
                               t_nominal=1.0 if elicited else None,
                               run=0.5 if elicited else 0.05))
        records.append(_record(seed, "walk"))
    rule = exp.evaluate_decisions(records)["timing_rule"]["definitions"]["root_crossing"]
    assert rule["n_with_lift_position"] == 10 and rule["n_elicited"] == 10
    assert rule["n_timed"] == 5
    assert rule["n_with_lift_position_without_event_time"] == 5
    assert rule["n_elicited_without_event_time"] == 5
    assert rule["first_2s"] == {"k": 5, "n": 10, "rate": 0.5,
                                "wilson95": pytest.approx(exp.wilson(5, 10))}
    assert rule["fraction"] == pytest.approx(0.5)
    assert rule["outcome"] == "report_both_distributions_without_mechanism_claim"
    # the timed-only fraction is still reported, but only as a labelled secondary
    assert rule["first_2s_timed_only"] == {"k": 5, "n": 5, "rate": 1.0,
                                           "wilson95": pytest.approx(exp.wilson(5, 5))}
    assert rule["fraction_timed_only"] == pytest.approx(1.0)
    assert rule["over_all_assigned_step_trials"] == {
        "k": 5, "n": 64, "rate": pytest.approx(5 / 64),
        "wilson95": pytest.approx(exp.wilson(5, 64))}
    assert exp.evaluate_decisions(records)["timing_rule"]["missing_event_time_rule"] == (
        exp.MISSING_EVENT_TIME_RULE)
    # the per-arm summary uses the same rule, so the two never disagree
    block = exp.build_summary(records)["arms"]["step"]["lift_time_s"]["root_crossing"]
    assert block["within_first_2s"] == {"k": 5, "n": 10, "rate": 0.5,
                                        "wilson95": pytest.approx(exp.wilson(5, 10))}
    assert block["within_first_2s_timed_only"]["n"] == 5
    assert block["n_with_lift_position_without_event_time"] == 5
    assert block["within_first_2s_over_all_assigned"] == {
        "k": 5, "n": 64, "rate": pytest.approx(5 / 64),
        "wilson95": pytest.approx(exp.wilson(5, 64))}


def test_timing_rule_refuses_when_the_two_definitions_disagree_with_missing_event_times():
    """Ten lifting clips: root crossing 4/10 (rollout context), nominal 10/10 (generalises).

    The preregistered branch would then depend on which committed definition is read, so the
    campaign fails closed with the numbers preserved instead of picking one.
    """
    records = []
    for index, seed in enumerate(exp.SEEDS):
        elicited = index < 10
        records.append(_record(seed, "step", elicited=elicited, lift=0.06 if elicited else 0.0,
                               t=(1.0 if index < 4 else None) if elicited else None,
                               t_nominal=1.0 if elicited else None,
                               run=0.5 if elicited else 0.05))
        records.append(_record(seed, "walk"))
    rule = exp.evaluate_decisions(records)["timing_rule"]
    assert rule["definitions"]["root_crossing"]["outcome"] == (
        "ardy_window_attributed_to_autoregressive_rollout_context")
    assert rule["definitions"]["nominal_speed"]["outcome"] == (
        "timing_generalises_to_released_g1_priors")
    assert rule["definitions_agree"] is False
    assert rule["refusal"]["reason"] == "definitions_disagree_with_missing_event_times"
    assert rule["refusal"]["n_with_lift_position_without_event_time"] == 6


def test_timing_rule_is_read_on_the_any_lift_denominator_the_protocol_names():
    """The ARDY comparator (40/49, 42/49) is over clips with ANY positive lift.

    Fifty clips lift; the twenty >= 3 cm ones are early and the thirty sub-3 cm ones are late.
    On the elicited denominator that reads 15/20 = 0.75 -> "generalises"; on the protocol's own
    denominator it is 20/50 = 0.40 -> "attributed to autoregressive rollout context".  The
    campaign must take the second branch and report the first beside it.
    """
    records = []
    for index, seed in enumerate(exp.SEEDS):
        elicited = index < 20
        sub_threshold = 20 <= index < 50
        lift = 0.06 if elicited else (0.01 if sub_threshold else 0.0)
        # 15 of the 20 elicited clips are early; 5 of the 30 sub-3 cm lifts are
        early = index < 15 or (20 <= index < 25)
        t = (1.0 if early else 3.0) if lift > 0 else None
        records.append(_record(seed, "step", elicited=elicited, lift=lift, t=t,
                               run=0.5 if elicited else 0.05))
        records.append(_record(seed, "walk"))
    rule = exp.evaluate_decisions(records)["timing_rule"]
    primary = rule["definitions"]["root_crossing"]
    assert primary["n_with_lift_position"] == 50 and primary["n_elicited"] == 20
    assert primary["first_2s"] == {"k": 20, "n": 50, "rate": pytest.approx(0.4),
                                   "wilson95": pytest.approx(exp.wilson(20, 50))}
    assert primary["fraction"] == pytest.approx(0.4)
    assert rule["outcome"] == "ardy_window_attributed_to_autoregressive_rollout_context"
    # the elicited fraction is reported beside it, and is the opposite branch
    assert primary["first_2s_elicited"]["k"] == 15 and primary["first_2s_elicited"]["n"] == 20
    assert primary["fraction_elicited"] == pytest.approx(0.75)
    assert rule["outcome_on_elicited_denominator"] == (
        "timing_generalises_to_released_g1_priors")
    assert rule["denominator"].startswith("STEP clips with a lift position")
    # ... while the screen rule keeps the elicited denominator the protocol names for it
    screen = exp.evaluate_decisions(records)["screen_rule"]
    assert screen["n_elicited"] == 20 and screen["float_primary_0p20s"]["n"] == 20
    assert "elicited" in screen["denominator"]


def test_over_all_assigned_timing_rates_count_the_same_event_as_their_headline():
    """Four late elicited clips beside forty early 1 mm lifts.

    The elicited headline is 0/4; the number printed beside it must not be 40/64, which would
    state that 62.5 % of assigned STEP trials lift within the first 2 s while no elicited clip
    does.
    """
    records = []
    for index, seed in enumerate(exp.SEEDS):
        elicited = index < 4
        tiny = 4 <= index < 44
        lift = 0.06 if elicited else (0.001 if tiny else 0.0)
        t = (3.0 if elicited else 1.0) if lift > 0 else None
        records.append(_record(seed, "step", elicited=elicited, lift=lift, t=t,
                               run=0.5 if elicited else 0.05))
        records.append(_record(seed, "walk"))
    primary = exp.evaluate_decisions(records)["timing_rule"]["definitions"]["root_crossing"]
    assert primary["first_2s_elicited"]["k"] == 0
    assert primary["elicited_over_all_assigned_step_trials"] == {
        "k": 0, "n": 64, "rate": 0.0, "wilson95": pytest.approx(exp.wilson(0, 64))}
    # the any-lift headline is 40/44, and its own over-all-assigned rate is 40/64
    assert primary["first_2s"]["k"] == 40 and primary["first_2s"]["n"] == 44
    assert primary["over_all_assigned_step_trials"]["k"] == 40
    step = exp.build_summary(records)["arms"]["step"]["lift_time_s"]["root_crossing"]
    assert step["within_first_2s_elicited"] == {"k": 0, "n": 4, "rate": 0.0,
                                                "wilson95": pytest.approx(exp.wilson(0, 4))}
    assert step["within_first_2s_elicited_over_all_assigned"]["k"] == 0
    assert step["within_first_2s"]["k"] == 40 and step["within_first_2s"]["n"] == 44
    assert step["within_first_2s_over_all_assigned"]["k"] == 40


@pytest.mark.parametrize("n_float, outcome", [
    (40, "screen_generalises_to_released_g1_priors"),      # 40/40 = 1.00
    (32, "screen_generalises_to_released_g1_priors"),      # 32/40 = 0.80, inclusive
    (31, "screen_stays_ardy_scoped"),                      # 31/40 = 0.775
    (0, "screen_stays_ardy_scoped"),
])
def test_screen_decision_rule_branches(n_float, outcome):
    n_elicited = 40
    records = []
    for index, seed in enumerate(exp.SEEDS):
        elicited = index < n_elicited
        records.append(_record(seed, "step", elicited=elicited, lift=0.06 if elicited else 0.0,
                               t=1.0 if elicited else None,
                               run=(0.5 if index < n_float else 0.1) if elicited else 0.05))
        records.append(_record(seed, "walk"))
    rule = exp.evaluate_decisions(records)["screen_rule"]
    assert rule["n_elicited"] == n_elicited
    assert rule["float_primary_0p20s"]["k"] == n_float
    assert rule["fraction"] == pytest.approx(n_float / n_elicited)
    assert rule["outcome"] == outcome
    assert rule["threshold_fraction"] == 0.8
    assert rule["primary_s"] == 0.2 and rule["secondary_s"] == 0.28
    assert rule["over_all_assigned_step_trials"]["n"] == 64


def test_screen_rule_reports_the_secondary_cut_beside_the_calibrated_one():
    records = []
    for index, seed in enumerate(exp.SEEDS):
        elicited = index < 20
        # 0.25 s is above the 0.20 s screen but below the 0.28 s post hoc cut
        records.append(_record(seed, "step", elicited=elicited, lift=0.06 if elicited else 0.0,
                               t=1.0 if elicited else None, run=0.25 if elicited else 0.05))
        records.append(_record(seed, "walk"))
    rule = exp.evaluate_decisions(records)["screen_rule"]
    assert rule["float_primary_0p20s"]["k"] == 20
    assert rule["float_secondary_0p28s"]["k"] == 0
    assert rule["outcome"] == "screen_generalises_to_released_g1_priors"


def test_decision_rules_use_the_step_arm_and_lock_the_arms():
    decisions = exp.evaluate_decisions(synthetic_records())
    assert decisions["arm_used"] == "step"
    assert decisions["no_arm_expansion_after_outcomes"]["arms"] == ["step", "walk"]
    assert decisions["no_arm_expansion_after_outcomes"]["seeds"] == [4700, 4763]
    assert "kinematic only" in decisions["scope"]
    assert decisions["timing_rule"]["ardy_reference"]["root_crossing"] == [40, 49]


# ------------------------------------------------------------------------ generate stage


def test_generate_writes_the_planned_128_with_exact_accounting(tmp_path):
    out = tmp_path / "campaign"
    receipt = exp.run_generate(**generate_kwargs(out, tmp_path))
    assert receipt["stages"]["generate"]["status"] == "complete"
    assert receipt["actual_kimodo_samples"] == 128
    accounting = receipt["query_accounting"]
    assert accounting["samples_planned"] == 128
    assert accounting["samples_launched"] == accounting["samples_returned"] == 128
    assert accounting["samples_converted_to_qpos"] == 128
    assert accounting["generate_invocations_planned"] == 16
    assert accounting["generate_invocations_completed"] == 16
    assert receipt["sample_count_exact"] is True
    assert receipt["spent_seeds"] == list(exp.SEEDS)
    assert receipt["unlaunched_locked_seeds"] == []
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 128
    assert [row["archive_key"] for row in rows] == [
        row["archive_key"] for row in exp.locked_row_plan()]
    with np.load(out / "qpos.npz") as clips, np.load(out / "smooth_root.npz") as smooth:
        assert len(clips.files) == len(smooth.files) == 128
        assert clips[rows[0]["archive_key"]].shape == (240, 36)
        assert smooth[rows[0]["archive_key"]].shape == (240, 3)
    anchors = receipt["evidence_anchors"]
    assert anchors["qpos"]["n_arrays"] == anchors["smooth_root"]["n_arrays"] == 128
    assert exp._is_sha256(anchors["smooth_root"]["content_sha256"])
    design = receipt["campaign_design"]
    assert design["row_plan_sha256"] == exp._json_hash(exp.locked_row_plan())
    assert design["row_plan_with_spec_sha256"] != design["row_plan_sha256"]
    assert {row["spec_sha256"] for row in rows} == {exp.spec_sha256(exp.campaign_spec())}
    assert receipt["campaign_design"]["constraint"]["channel"] == "smooth_root_2d"
    assert receipt["campaign_design"]["actual_channel_usage"] == dict(exp.EXPECTED_CHANNEL_USAGE)
    assert receipt["execution_mode"]["scientific_evidence_eligible"] is False
    assert "runner_factory" in receipt["execution_mode"]["dependency_injections"]
    assert receipt["provenance"]["runner_prompt_cache_check"][
        "runner_memory_byte_matches_cache"] is True


def test_generate_pairs_same_seed_step_and_walk_on_one_latent(tmp_path):
    out = tmp_path / "campaign"
    receipt = exp.run_generate(**generate_kwargs(out, tmp_path))
    assert receipt["stages"]["generate"]["latent_audit"]["pairing_verified_every_chunk"] is True
    audit = json.loads((out / "noise_audit.json").read_text())
    assert len(audit) == 16
    first = audit[0]
    assert first["status"] == "verified" and first["pairing_verified"] is True
    by_seed: dict[int, list[list[str]]] = {}
    for row in first["rows"]:
        by_seed.setdefault(row["seed"], []).append(row["row_sha256_by_draw"])
    assert all(len(v) == 2 and v[0] == v[1] for v in by_seed.values())
    assert len({tuple(v[0]) for v in by_seed.values()}) == 4


def test_generate_refuses_a_runner_whose_same_seed_rows_diverge(tmp_path):
    out = tmp_path / "campaign"
    with pytest.raises(exp.CampaignAbort, match="same-seed rows drew different latents"):
        exp.run_generate(**generate_kwargs(out, tmp_path, runner_cls=BrokenPairingRunner))
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["blocked"] is True and receipt["status"] == "blocked"


def test_generate_fails_closed_mid_campaign_and_keeps_the_partial_evidence(tmp_path):
    out = tmp_path / "campaign"
    with pytest.raises(exp.CampaignAbort, match="synthetic second-chunk failure"):
        exp.run_generate(**generate_kwargs(out, tmp_path, runner_cls=SecondChunkFailureRunner))
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["blocked"] is True
    assert receipt["sample_count_exact"] is False
    assert receipt["actual_kimodo_samples"] == 8
    assert receipt["seeds_spent_and_must_not_be_reused"] is True
    assert receipt["spent_seeds"] == list(exp.SEEDS[:8])
    assert receipt["unlaunched_locked_seeds"] == list(exp.SEEDS[8:])
    # a blocked campaign is never resumed in place
    with pytest.raises(exp.CampaignAbort, match="blocked"):
        exp.run_score(**score_kwargs(out))


def test_generate_refuses_a_nonempty_output_directory(tmp_path):
    out = tmp_path / "campaign"
    out.mkdir()
    (out / "stray.txt").write_text("x")
    with pytest.raises(exp.CampaignAbort, match="nonempty"):
        exp.run_generate(**generate_kwargs(out, tmp_path))


def test_generate_requires_a_clean_worktree(tmp_path):
    out = tmp_path / "campaign"
    dirty = {"commit": "a" * 40, "dirty": True, "status": [" M x.py"],
             "tracked_diff_sha256": "b" * 64}
    with pytest.raises(exp.CampaignAbort, match="clean git worktree"):
        exp.run_generate(**generate_kwargs(out, tmp_path, code_state_fn=lambda _r: dirty))
    assert not (out / "receipt.json").exists()


# ------------------------------------------------------------------------------ host gate


def test_host_gate_refusal_leaves_out_untouched(tmp_path):
    out = tmp_path / "never_created"
    with pytest.raises(exp.CampaignAbort, match="host-resource gate"):
        exp.run_generate(**generate_kwargs(out, tmp_path, host_gate_fn=failing_gate))
    assert not out.exists()

    existing = tmp_path / "already_there"
    existing.mkdir()
    with pytest.raises(exp.CampaignAbort, match="host-resource gate"):
        exp.run_generate(**generate_kwargs(existing, tmp_path, host_gate_fn=failing_gate))
    assert list(existing.iterdir()) == []


def test_generation_uses_the_ardy_generation_gate_preset(tmp_path):
    seen = {}

    def recording_gate(**kwargs):
        seen.update(kwargs)
        return passing_gate()

    out = tmp_path / "campaign"
    receipt = exp.run_generate(**generate_kwargs(out, tmp_path, host_gate_fn=recording_gate))
    from scene2motion.host_gate import ARDY_GENERATION_GATE

    assert seen == dict(ARDY_GENERATION_GATE)
    assert receipt["host_resource_gate"]["generate"]["pass"] is True


# --------------------------------------------------------------------------- score stage


def test_score_writes_one_validated_reference_per_clip(tmp_path):
    out = tmp_path / "campaign"
    receipt = run_through_score(out, tmp_path)
    stage = receipt["stages"]["score"]
    assert stage["status"] == "complete" and stage["scored"] == 128
    assert stage["scored_this_invocation"] == 128
    assert stage["forward_axis_dominant_clips"] == 128
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    assert all(row["reference"] is not None for row in rows)
    step_rows = [row for row in rows if row["arm"] == "step"]
    assert stage["reference_summary_per_arm"]["step"]["elicitation"]["n"] == 64
    assert stage["reference_summary_per_arm"]["walk"]["elicitation"]["k"] == 0
    reference = step_rows[0]["reference"]
    assert reference["route_fidelity"]["measured_against"] == "smooth_root_pos"
    assert set(reference["exact_boxes"]) == {"staged", "unstaged"}
    assert reference["timing"]["fps"] == 30.0


def test_score_resumes_after_an_interrupted_run(tmp_path):
    out = tmp_path / "campaign"
    exp.run_generate(**generate_kwargs(out, tmp_path))

    calls: list[str] = []

    def interrupting(qpos, smooth, arm, ctx):
        if len(calls) == 40:
            raise KeyboardInterrupt("operator stopped the scoring stage")
        calls.append(arm)
        return fake_reference_scorer(qpos, smooth, arm, ctx)

    with pytest.raises(KeyboardInterrupt):
        exp.run_score(**score_kwargs(out, scorer=interrupting))
    partial = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    scored_before = [row["archive_key"] for row in partial if row.get("reference") is not None]
    assert 0 < len(scored_before) < 128
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["blocked"] is False and receipt["stages"]["score"]["status"] == "running"

    resumed: list[str] = []

    def counting(qpos, smooth, arm, ctx):
        resumed.append(arm)
        return fake_reference_scorer(qpos, smooth, arm, ctx)

    stage = exp.run_score(**score_kwargs(out, scorer=counting))["stages"]["score"]
    assert stage["status"] == "complete" and stage["scored"] == 128
    assert stage["scored_this_invocation"] == 128 - len(scored_before)
    assert len(resumed) == 128 - len(scored_before)
    rows = {row["archive_key"]: row for row in
            (json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines())}
    assert len(rows) == 128 and all(row["reference"] is not None for row in rows.values())
    for key in scored_before:
        assert rows[key]["reference"] is not None


def test_score_is_idempotent_once_complete(tmp_path):
    out = tmp_path / "campaign"
    run_through_score(out, tmp_path)
    digest = exp._sha256(out / "rows.jsonl")

    def poisoned(*_args, **_kwargs):
        raise AssertionError("a completed score stage must not re-score")

    receipt = exp.run_score(**score_kwargs(out, scorer=poisoned))
    assert receipt["stages"]["score"]["status"] == "complete"
    assert exp._sha256(out / "rows.jsonl") == digest


def test_score_refuses_when_a_bound_source_changed(tmp_path):
    out = tmp_path / "campaign"
    exp.run_generate(**generate_kwargs(out, tmp_path))
    kwargs = score_kwargs(out)
    kwargs["source_hashes_fn"] = lambda _repo: {**fake_source_hashes(_repo),
                                                exp.PROTOCOL_PATH: "9" * 64}
    with pytest.raises(exp.CampaignAbort, match="source content changed"):
        exp.run_score(**kwargs)


def test_score_refuses_a_tampered_archive(tmp_path):
    out = tmp_path / "campaign"
    exp.run_generate(**generate_kwargs(out, tmp_path))
    with np.load(out / "smooth_root.npz") as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    key = next(iter(arrays))
    arrays[key] = arrays[key] + 1.0
    np.savez(out / "smooth_root.npz", **arrays)
    with pytest.raises(exp.CampaignAbort, match="smooth_root content hash"):
        exp.run_score(**score_kwargs(out))


def test_score_fails_closed_if_the_forward_axis_convention_changes(tmp_path):
    out = tmp_path / "campaign"
    exp.run_generate(**generate_kwargs(out, tmp_path))

    def lateral_scorer(qpos, smooth, arm, ctx):
        score = fake_reference_scorer(qpos, smooth, arm, ctx)
        score["route_fidelity"] = {**score["route_fidelity"], "forward_axis_dominant": False}
        return score

    with pytest.raises(exp.CampaignAbort, match="axis convention"):
        exp.run_score(**score_kwargs(out, scorer=lateral_scorer))


def test_score_revalidates_the_checkout_across_two_interpreters(tmp_path):
    """Generation runs under the Kimodo venv and scoring under ``$S2M_PY`` -- by design.

    Only the checkout half of the Kimodo runtime identity is stage-invariant.  If the
    interpreter half (``sys.version``, numpy, torch) were compared across stages, every score
    and analyze stage would abort *after* the 64 reserved seeds had already been spent.
    """
    out = tmp_path / "campaign"
    exp.run_generate(**generate_kwargs(out, tmp_path))
    bound = json.loads((out / "receipt.json").read_text())["provenance"]["kimodo_runtime"]
    scoring = fake_runtime_identity("scoring-venv")
    assert bound["checkout"] == scoring["checkout"]
    assert bound["interpreter"] != scoring["interpreter"]

    stage = exp.run_score(**score_kwargs(out))["stages"]["score"]
    assert stage["status"] == "complete"
    check = stage["provenance_check"]
    assert check["kimodo_checkout_unchanged"] is True
    assert check["interpreter_runtime_compared_across_stages"] is False
    assert check["generation_interpreter_runtime"] == bound["interpreter"]
    assert check["stage_interpreter_runtime"] == scoring["interpreter"]
    assert exp.run_analyze(out=out, **stage_kwargs())["complete"] is True


def test_score_refuses_when_the_kimodo_checkout_identity_changes(tmp_path):
    out = tmp_path / "campaign"
    exp.run_generate(**generate_kwargs(out, tmp_path))
    kwargs = score_kwargs(out)
    kwargs["runtime_identity_fn"] = lambda: fake_runtime_identity("scoring-venv",
                                                                  commit="9" * 40)
    with pytest.raises(exp.CampaignAbort, match="checkout identity changed"):
        exp.run_score(**kwargs)


def test_validate_pins_reads_the_checkout_half_of_the_runtime_identity():
    generator = {"checkpoint": {"model_name": exp.MODEL_NAME,
                                "hf_revision": exp.PINNED_KIMODO_HF_REVISION,
                                "checkpoint_sha256": exp.PINNED_KIMODO_CHECKPOINT_SHA256}}
    physical = {"fields": {"sha256": exp.PINNED_G1_XML_SHA256}}

    def runtime(commit):
        return {"checkout": {"fields": {"kimodo_git_commit": commit,
                                        "kimodo_tracked_status": []}},
                "interpreter": {"fields": {"python": "whatever"}}}

    exp.validate_pins(generator, runtime(exp.PINNED_KIMODO_COMMIT), physical)
    with pytest.raises(ValueError, match="runtime commit"):
        exp.validate_pins(generator, runtime("0" * 40), physical)


@requires_external
def test_kimodo_runtime_identity_keeps_the_interpreter_out_of_the_checkout():
    identity = exp.kimodo_runtime_identity()
    assert set(identity) >= {"checkout", "interpreter"}
    assert identity["checkout"] == exp.kimodo_checkout_identity()
    checkout = json.dumps(identity["checkout"], sort_keys=True)
    for interpreter_specific in (sys.version, np.__version__):
        assert interpreter_specific not in checkout
    fields = identity["interpreter"]["fields"]
    assert fields["python"] == sys.version and fields["numpy_version"] == np.__version__
    assert fields["executable"] == sys.executable


def test_a_scoring_error_leaves_the_campaign_resumable_and_not_blocked(tmp_path):
    """Only ``generate`` spends seeds, so only ``generate`` may block a campaign.

    A scorer that raises an ordinary ``Exception`` (not the ``KeyboardInterrupt`` the resume
    test uses, which escapes ``except Exception`` entirely) must leave the 128 archived clips
    reachable: CLAUDE.md requires a killed campaign to be finished by re-scoring byte-identical
    archives, never by regenerating on fresh seeds.
    """
    out = tmp_path / "campaign"
    exp.run_generate(**generate_kwargs(out, tmp_path))
    calls: list[str] = []

    def broken(qpos, smooth, arm, ctx):
        if len(calls) == 40:
            raise RuntimeError("synthetic scorer defect on clip 40")
        calls.append(arm)
        return fake_reference_scorer(qpos, smooth, arm, ctx)

    with pytest.raises(exp.CampaignAbort, match="synthetic scorer defect"):
        exp.run_score(**score_kwargs(out, scorer=broken))

    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["blocked"] is False and receipt["resumable"] is True
    assert receipt["status"] == "running"
    assert receipt["stages"]["generate"]["status"] == "complete"
    assert receipt["stages"]["score"]["status"] == "failed"
    assert receipt["stages"]["score"]["failure"]["error_type"] == "RuntimeError"
    assert receipt["last_stage_failure"]["stage"] == "score"
    assert [item["stage"] for item in receipt["stage_failures"]] == ["score"]

    # the archives are untouched, so a fixed scorer finishes the same campaign in place
    stage = exp.run_score(**score_kwargs(out))["stages"]["score"]
    assert stage["status"] == "complete" and stage["scored"] == 128
    assert "failure" not in stage
    assert exp.run_analyze(out=out, **stage_kwargs())["complete"] is True
    assert json.loads((out / "receipt.json").read_text())["stage_failures"][0][
        "error_type"] == "RuntimeError"


def test_a_refused_timing_branch_keeps_every_number_and_stays_resumable(tmp_path):
    """The two committed event-time definitions landing on different branches is a refusal.

    Thirty of the 48 elicited step clips lose their root-crossing time, so root crossing reads
    18/48 (rollout context) while the nominal definition reads 48/48 (generalises).  The
    campaign records the numbers, withholds the branch, and stays unblocked.
    """
    out = tmp_path / "campaign"
    exp.run_generate(**generate_kwargs(out, tmp_path))
    stripped: list[str] = []

    def no_crossing(qpos, smooth, arm, ctx):
        score = fake_reference_scorer(qpos, smooth, arm, ctx)
        if arm == "step" and score["elicitation"]["elicited"] and len(stripped) < 30:
            stripped.append(arm)
            score["timing"] = {**score["timing"], "root_crossing_frame": None,
                               "lift_time_root_crossing_s": None,
                               "within_first_2s_root_crossing": None}
        return score

    exp.run_score(**score_kwargs(out, scorer=no_crossing))
    with pytest.raises(exp.CampaignAbort, match="timing rule refused"):
        exp.run_analyze(out=out, **stage_kwargs())

    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "refused"
    rule = summary["decisions"]["timing_rule"]
    assert rule["definitions"]["root_crossing"]["first_2s"] == {
        "k": 18, "n": 48, "rate": pytest.approx(18 / 48),
        "wilson95": pytest.approx(exp.wilson(18, 48))}
    assert rule["definitions"]["nominal_speed"]["first_2s"]["k"] == 48
    assert rule["refusal"]["reason"] == "definitions_disagree_with_missing_event_times"
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["blocked"] is False and receipt["complete"] is False
    assert receipt["stages"]["analyze"]["refusal"]["reason"] == (
        "definitions_disagree_with_missing_event_times")
    assert receipt["summary"]["arms"]["step"]["elicitation"]["k"] == 48


# ------------------------------------------------------------------------- analyze stage


def test_analyze_produces_the_summary_and_both_decision_rules(tmp_path):
    out = tmp_path / "campaign"
    run_through_score(out, tmp_path)
    receipt = exp.run_analyze(out=out, **stage_kwargs())
    assert receipt["complete"] is True and receipt["status"] == "complete"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_clips"] == 128 and summary["planned_n_per_arm"] == 64
    assert set(summary["arms"]) == {"step", "walk"}
    # the fake scorer elicits on 3 of every 4 step seeds and never on walk
    assert summary["arms"]["step"]["elicitation"]["k"] == 48
    assert summary["arms"]["step"]["elicitation"]["n"] == 64
    assert summary["arms"]["walk"]["elicitation"]["k"] == 0
    decisions = summary["decisions"]
    assert decisions["timing_rule"]["outcome"] == "timing_generalises_to_released_g1_priors"
    assert decisions["screen_rule"]["outcome"] == "screen_generalises_to_released_g1_priors"
    records = [json.loads(line) for line in (out / "clip_records.jsonl").read_text().splitlines()]
    assert len(records) == 128
    anchors = receipt["evidence_anchors"]
    assert anchors["summary"]["file_sha256"] == exp._sha256(out / "summary.json")
    assert anchors["clip_records"]["n_rows"] == 128


def test_analyze_requires_a_complete_score_stage(tmp_path):
    out = tmp_path / "campaign"
    exp.run_generate(**generate_kwargs(out, tmp_path))
    with pytest.raises(exp.CampaignAbort, match="'score' is not complete"):
        exp.run_analyze(out=out, **stage_kwargs())


def test_analyze_is_idempotent_once_complete(tmp_path):
    out = tmp_path / "campaign"
    run_through_score(out, tmp_path)
    exp.run_analyze(out=out, **stage_kwargs())
    digest = exp._sha256(out / "summary.json")
    receipt = exp.run_analyze(out=out, **stage_kwargs())
    assert receipt["stages"]["analyze"]["status"] == "complete"
    assert exp._sha256(out / "summary.json") == digest


# ------------------------------------------------------------------------------- dry run


def test_dry_run_reports_the_plan_the_arms_and_the_gate_without_writing(tmp_path, capsys):
    out = tmp_path / "campaign"
    exp.main(["--dry-run", "--out", str(out)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "dry_run" and payload["writes_performed"] is False
    assert payload["samples_planned"] == 128
    assert payload["batch_plan"]["batch_size"] == 8
    assert len(payload["batch_plan"]["chunks"]) == 16
    assert payload["batch_plan"]["row_plan_sha256"] == exp._json_hash(exp.locked_row_plan())
    assert set(payload["arms"]) == {"step", "walk"}
    for arm in ("step", "walk"):
        assert payload["arms"][arm]["constraint_channel"] == "smooth_root_2d"
        assert payload["arms"][arm]["adapter_channels_written"] == {"smooth_root_2d": 240}
    assert payload["arms"]["walk"]["prompt"] == exp.WALK
    assert "pass" in payload["host_resource_gate"]["generate"]
    assert payload["host_resource_gate"]["generate"]["thresholds"]["min_free_vram_mib"] == 4096
    assert payload["endpoints"]["route_error_measured_against"] == "smooth_root_pos"
    assert payload["endpoints"]["coverage_rule"] == exp.COVERAGE_RULE
    assert payload["endpoints"]["timing_denominator_rule"] == exp.TIMING_DENOMINATOR_RULE
    assert payload["endpoints"]["missing_event_time_rule"] == exp.MISSING_EVENT_TIME_RULE
    assert payload["sonic"] == "none; this campaign is kinematic only"
    assert not out.exists()


def test_stage_all_refuses_an_existing_campaign_without_resume(tmp_path):
    out = tmp_path / "campaign"
    exp.run_generate(**generate_kwargs(out, tmp_path))
    with pytest.raises(SystemExit, match="already holds a campaign"):
        exp.main(["--stage", "all", "--out", str(out)])


# ------------------------------------------------------ conventions shared with the family


def test_local_hashers_match_calibrate_ramp_route_phase():
    payload = {"b": [1, 2, {"c": True}], "a": "x", "n": None}
    assert exp._canonical_json(payload) == cal._canonical_json(payload)
    assert exp._json_hash(payload) == cal._json_hash(payload)
    assert exp._identity("s", payload) == cal._identity("s", payload)
    assert exp._is_sha256("0" * 64) and not exp._is_sha256("0" * 63)
    arrays = {"a": np.arange(6, dtype=np.float32).reshape(2, 3),
              "b": np.zeros((1, 4), dtype=np.float64)}
    assert exp._array_hash(arrays) == cal._array_hash(arrays)
    assert exp._sample_hash(arrays) == cal._sample_hash(arrays)
    assert exp._sha256(ROOT / "env.sh") == cal._sha256(ROOT / "env.sh")


def test_local_writers_match_calibrate_ramp_route_phase(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"b": "y", "a": 2}]
    exp._write_jsonl(tmp_path / "mine.jsonl", rows)
    cal._write_jsonl(tmp_path / "theirs.jsonl", rows)
    assert (tmp_path / "mine.jsonl").read_bytes() == (tmp_path / "theirs.jsonl").read_bytes()
    payload = {"z": 1, "a": [True, None]}
    exp._write_json(tmp_path / "mine.json", payload)
    cal._write_json(tmp_path / "theirs.json", payload)
    assert (tmp_path / "mine.json").read_bytes() == (tmp_path / "theirs.json").read_bytes()


def test_git_state_matches_calibrate_ramp_route_phase():
    assert exp._git_state(ROOT) == cal._git_state(ROOT)


def test_endpoint_constants_match_the_ardy_family():
    from experiments import analyze_e1a_placement as e1a
    from experiments import analyze_trackability_contract as atc
    from experiments import exp022_exact_tracking_bridge as e22

    assert exp.OBSTACLES == tuple(e22.OBSTACLES)
    assert exp.GRADED_HEIGHTS_M == tuple(e22.GRADED_HEIGHTS_M)
    assert exp.OBSTACLE_DEPTH_M == float(e22.OBSTACLE_DEPTH_M)
    assert exp.SCAN_POINTS == 120
    assert exp.OBSTACLE_DEPTH_M == 0.20 and e1a.SCAN_MARGIN_M == 0.30
    assert exp.PRIMARY_GATE_S == atc.PRIMARY_GATE_S
    assert exp.SECONDARY_GATE_S == atc.SECONDARY_GATE_S
    assert exp.THRESHOLD_RECEIPT_SHA256 == atc.THRESHOLD_RECEIPT_SHA256
    assert exp.ELICITATION_MIN_M == 0.03


def test_recovered_runner_contract_is_the_one_the_campaign_assumes():
    assert kr.NOISE_STREAM_VERSION == exp.NOISE_STREAM_VERSION == 2
    spec = exp.campaign_spec()
    data: dict = {k: [] for k in ("smooth_root_2d", "global_root_heading", "root_y_pos",
                                  "global_joints_rots", "global_joints_positions")}
    index: dict = {k: [] for k in data}
    kr.KimodoConstraintSet(spec, root_idx=0, device="cpu").update_constraints(data, index)
    assert [key for key, value in data.items() if value] == ["smooth_root_2d"]
    assert torch.equal(data["smooth_root_2d"][0],
                       torch.as_tensor(spec.root_xz, dtype=torch.float32))


# --------------------------------------------------------- the real scorer, on one clip


def test_real_scorer_produces_a_validated_record(tmp_path, monkeypatch):
    """A single clip through the real mujoco path, with the 120-point scan reduced to 4."""
    monkeypatch.setattr(exp, "SCAN_POINTS", 4)
    support = {"support_height_m": 0.02, "support_speed_mps": 0.3,
               "max_unsupported_run_s": exp.PRIMARY_GATE_S}
    ctx = exp.build_scoring_context(exp.route_xz(), support)
    qpos = synthetic_qpos(4700, "step")
    smooth = synthetic_smooth_root(4700, "step")
    score = exp.validated_reference_score(exp.score_reference_clip(qpos, smooth, "step", ctx))
    assert score["arm"] == "step"
    assert score["elicitation"]["scan_points"] == 4
    assert isinstance(score["elicitation"]["elicited"], bool)
    assert set(score["exact_boxes"]) == {"staged", "unstaged"}
    assert score["exact_boxes"]["staged"]["obstacle_x_m"] == 1.2
    assert score["exact_boxes"]["unstaged"]["obstacle_x_m"] == 3.6
    assert score["screen_predictions"]["primary_threshold_s"] == 0.2
    assert score["contract_features"]["max_unsupported_run_s"] >= 0.0
    assert score["route_fidelity"]["measured_against"] == "smooth_root_pos"
    assert score["route_fidelity"]["smooth_root_path_mae_m"] == pytest.approx(0.0, abs=1e-6)
    assert score["timing"]["fps"] == 30.0


def test_generator_identity_checks_the_pinned_model_not_the_resolved_key(tmp_path, monkeypatch):
    """`load_model` returns a short registry key, so the snapshot check must use the pinned name."""
    cache = tmp_path / "hub"
    snapshot = cache / f"models--nvidia--{exp.MODEL_NAME}" / "snapshots" / "3020ad8c"
    stats = snapshot / "stats" / "motion" / "body"
    stats.mkdir(parents=True)
    for name in ("config.yaml", "model.safetensors"):
        (snapshot / name).write_text("x")
    for block in ("body", "global_root", "local_root"):
        d = snapshot / "stats" / "motion" / block
        d.mkdir(parents=True, exist_ok=True)
        (d / "mean.npy").write_bytes(b"m")
        (d / "std.npy").write_bytes(b"s")

    class FakeRunner:
        model_name = "g1"          # the resolved SHORT KEY, not the repository name
        fps = exp.FPS
        noise_stream_version = 2
        model = type("M", (), {"motion_rep": type("R", (), {
            "body_stats": type("B", (), {"folder": str(stats)})()})()})()

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache), raising=False)
    identity = exp.kimodo_generator_identity(FakeRunner())
    checkpoint = identity["checkpoint"]
    assert checkpoint["model_name"] == exp.MODEL_NAME          # the pinned repository name
    assert checkpoint["resolved_model_key"] == "g1"          # what load_model resolved to
    assert checkpoint["hf_revision"] == "3020ad8c"
