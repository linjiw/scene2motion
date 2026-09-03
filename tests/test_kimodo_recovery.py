"""CPU tests for the recovered Kimodo-G1 runner and reduced audit (EXP-025 prerequisite 1).

`experiments/kimodo_recovered/*` was reconstructed from a session transcript, not from a surviving
file, so nothing about it has ever been exercised by a committed test. These tests pin the
pieces that can be checked without the Kimodo checkpoint or a GPU:

* the per-sample noise contract (NOISE_STREAM_VERSION 2), checked against
  `scene2motion.runner._per_sample_noise` -- the recovered function claims to be a port of
  it, so "port" is asserted, not assumed;
* the prompt-cache key, checked against `scene2motion.runner._key` on the three prompts in
  `outputs/text_cache.npz` -- this is what makes EXP-025's "copy ARDY's cached STEP vector
  into a Kimodo cache under the same key" prerequisite legal;
* the ConstraintSpec -> Kimodo channel adapter, checked against `ArdyConstraintSet`, whose
  only intended difference is the `root_2d` -> `smooth_root_2d` rename;
* the audit's vendored descriptor/validity math and the four counting rows, including the
  `--selftest` output `3.0 / 3.0 / 3.0 / 2.0` that the EXP-025 protocol names as the
  acceptance criterion for the recovery.

Nothing here loads Kimodo-G1-RP-v1, imports the `kimodo` package, or touches the GPU.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
# The recovered audit inserts os.environ["SCENE2MOTION_ROOT"] (default: the ORIGINAL checkout
# path, /home/linjiw/scene2motion) onto sys.path at import time. Point it at this repository
# so a worktree or a second checkout tests its own morphology, not another tree's.
os.environ.setdefault("SCENE2MOTION_ROOT", str(REPO_ROOT))

import scene2motion.morphology as morphology  # noqa: E402  (import order is load-bearing)
from scene2motion.constraints import ArdyConstraintSet, ConstraintSpec  # noqa: E402
from scene2motion.runner import _key as ardy_key  # noqa: E402
from scene2motion.runner import _per_sample_noise as ardy_noise  # noqa: E402

from experiments import audit_delta  # noqa: E402
from experiments.kimodo_recovered import kimodo_reduced_audit as audit  # noqa: E402
from experiments.kimodo_recovered import kimodo_runner as kr  # noqa: E402


# --------------------------------------------------------------- per-sample noise (v2)

def draws(seeds, shape=(5,), device="cpu"):
    with kr._per_sample_noise(list(seeds), device):
        return (torch.randn((len(seeds), *shape)), torch.randn((len(seeds), *shape)))


def test_noise_stream_version_is_two():
    assert kr.NOISE_STREAM_VERSION == 2


def test_per_sample_noise_is_reproducible_from_the_seed_alone():
    first_a, second_a = draws([11, 29])
    first_b, second_b = draws([11, 29])

    torch.testing.assert_close(first_a, first_b)
    torch.testing.assert_close(second_a, second_b)


def test_per_sample_noise_is_independent_of_batch_position():
    together_first, together_second = draws([11, 29])
    alone_first, alone_second = draws([29])
    reversed_first, _ = draws([29, 11])

    torch.testing.assert_close(together_first[1], alone_first[0])
    torch.testing.assert_close(together_second[1], alone_second[0])
    torch.testing.assert_close(together_first[1], reversed_first[0])


def test_per_sample_noise_advances_between_draws_instead_of_repeating():
    # The v1 bug the version tag records: reseeding inside the patch would hand the second
    # draw the identical latent. Kimodo draws once per generate() call, but the streams must
    # still advance if a future version draws more than once.
    first, second = draws([11, 29])
    assert not torch.equal(first, second)


def test_per_sample_noise_passes_through_shapes_whose_leading_dim_is_not_the_batch():
    with kr._per_sample_noise([11, 29], "cpu"):
        torch.manual_seed(7)
        patched = torch.randn((3, 4))
    torch.manual_seed(7)
    expected = torch.randn((3, 4))
    torch.testing.assert_close(patched, expected)


def test_per_sample_noise_restores_torch_randn_even_on_error():
    real = torch.randn
    with pytest.raises(RuntimeError):
        with kr._per_sample_noise([11, 29], "cpu"):
            assert torch.randn is not real
            raise RuntimeError("boom")
    assert torch.randn is real


def test_per_sample_noise_reproduces_the_ardy_runner_stream():
    """The recovered docstring calls this a port of scene2motion/runner.py's v2 helper."""
    seeds = [4700, 4701, 4702]
    with kr._per_sample_noise(seeds, "cpu"):
        kimodo_first = torch.randn((len(seeds), 6, 3))
        kimodo_second = torch.randn((len(seeds), 6, 3))
    with ardy_noise(seeds, "cpu"):
        ardy_first = torch.randn((len(seeds), 6, 3))
        ardy_second = torch.randn((len(seeds), 6, 3))

    torch.testing.assert_close(kimodo_first, ardy_first)
    torch.testing.assert_close(kimodo_second, ardy_second)


# --------------------------------------------------------------- prompt cache keys

CACHED_PROMPTS = (
    "A person walks forward.",                          # WALK
    "A person steps over an obstacle.",                 # STEP
    "A person steps sideways through a narrow gap.",    # SQUEEZE
)


def test_raw_key_is_sha1_of_the_utf8_prompt():
    for text in CACHED_PROMPTS:
        assert kr._raw_key(text) == hashlib.sha1(text.encode("utf-8")).hexdigest()


def test_raw_key_agrees_with_the_ardy_runner_cache_key():
    """EXP-025 copies ARDY's cached STEP vector into a Kimodo cache under the same key."""
    for text in CACHED_PROMPTS:
        assert kr._raw_key(text) == ardy_key(text)


def _cache_key(text, cache, sanitize=lambda s: s):
    stub = SimpleNamespace(_sanitize=sanitize, _text_cache=cache)
    return kr.KimodoRunner._cache_key(stub, text)


def test_cache_key_prefers_the_sanitized_prompt():
    sanitize = str.strip
    raw, clean = "  hello  ", "hello"
    cache = {kr._raw_key(clean): 1, kr._raw_key(raw): 2}
    assert _cache_key(raw, cache, sanitize) == kr._raw_key(clean)


def test_cache_key_falls_back_to_the_raw_prompt_for_ardy_style_caches():
    sanitize = str.strip
    raw = "  hello  "
    cache = {kr._raw_key(raw): 2}
    assert _cache_key(raw, cache, sanitize) == kr._raw_key(raw)


def test_cache_key_is_none_when_the_prompt_is_absent_under_both_schemes():
    assert _cache_key("A person does something new.", {}) is None


# --------------------------------------------------------------- constraint adapter

def dense_spec(T=12):
    t = np.arange(T) / 30.0
    root_xz = np.stack([np.zeros(T), 0.9 * t], -1)
    pos_frames = np.array([4, 6, 8])
    pos_joints = np.array([9, 14])
    pos_targets = np.zeros((len(pos_frames), len(pos_joints), 3)) + 0.5
    rot_frames = np.array([5, 7])
    rot_joints = np.array([3, 4, 5])
    rot_targets = np.tile(np.eye(3), (len(rot_frames), len(rot_joints), 1, 1))
    return ConstraintSpec(root_xz=root_xz, heading=np.linspace(0.0, 0.3, T),
                          root_y=np.full(T, 0.78),
                          pos_frames=pos_frames, pos_joints=pos_joints,
                          pos_targets=pos_targets,
                          rot_frames=rot_frames, rot_joints=rot_joints,
                          rot_targets=rot_targets)


def collect(adapter):
    data, index = defaultdict(list), defaultdict(list)
    adapter.update_constraints(data, index)
    return dict(data), dict(index)


def test_constraint_set_writes_the_kimodo_root_channel_name():
    data, index = collect(kr.KimodoConstraintSet(dense_spec(), 0, "cpu"))
    assert "smooth_root_2d" in data and "smooth_root_2d" in index
    assert "root_2d" not in data


def test_constraint_set_writes_every_kimodo_filler_channel():
    data, _ = collect(kr.KimodoConstraintSet(dense_spec(), 0, "cpu"))
    assert set(data) == {"smooth_root_2d", "root_y_pos", "global_root_heading",
                         "global_joints_rots", "global_joints_positions"}


def test_constraint_set_matches_the_ardy_adapter_apart_from_the_root_rename():
    spec = dense_spec()
    k_data, k_index = collect(kr.KimodoConstraintSet(spec, 0, "cpu"))
    a_data, a_index = collect(ArdyConstraintSet(spec, 0, "cpu"))

    k_data["root_2d"] = k_data.pop("smooth_root_2d")
    k_index["root_2d"] = k_index.pop("smooth_root_2d")
    assert set(k_data) == set(a_data)
    for key in a_data:
        torch.testing.assert_close(k_data[key][0], a_data[key][0])
        torch.testing.assert_close(k_index[key][0], a_index[key][0])


def test_constraint_set_encodes_heading_as_cosine_and_sine():
    spec = dense_spec()
    data, _ = collect(kr.KimodoConstraintSet(spec, 0, "cpu"))
    h = data["global_root_heading"][0].numpy()
    np.testing.assert_allclose(h[:, 0], np.cos(spec.heading), atol=1e-6)
    np.testing.assert_allclose(h[:, 1], np.sin(spec.heading), atol=1e-6)


def test_constraint_set_injects_the_root_joint_at_every_position_frame():
    spec = dense_spec()
    root_idx = 0
    data, index = collect(kr.KimodoConstraintSet(spec, root_idx, "cpu"))
    pairs = index["global_joints_positions"][0].numpy()
    targets = data["global_joints_positions"][0].numpy()
    n_joints = 1 + len(spec.pos_joints)

    assert pairs.shape == (len(spec.pos_frames) * n_joints, 2)
    assert targets.shape == (len(spec.pos_frames) * n_joints, 3)
    for i, frame in enumerate(spec.pos_frames):
        row = i * n_joints
        assert pairs[row].tolist() == [int(frame), root_idx]
        # ... and its target agrees with root_xz / root_y (y-up: x, height, z).
        np.testing.assert_allclose(
            targets[row],
            [spec.root_xz[frame, 0], spec.root_y[frame], spec.root_xz[frame, 1]],
            atol=1e-6)


def test_constraint_set_pairs_every_rotation_frame_with_every_rotation_joint():
    spec = dense_spec()
    data, index = collect(kr.KimodoConstraintSet(spec, 0, "cpu"))
    pairs = index["global_joints_rots"][0].numpy()
    mats = data["global_joints_rots"][0].numpy()

    assert mats.shape == (len(spec.rot_frames) * len(spec.rot_joints), 3, 3)
    expected = [[int(f), int(j)] for f in spec.rot_frames for j in spec.rot_joints]
    assert pairs.tolist() == expected


def test_constraint_set_omits_free_channels():
    spec = ConstraintSpec(root_xz=np.zeros((8, 2)))   # root path only; heading/height free
    data, _ = collect(kr.KimodoConstraintSet(spec, 0, "cpu"))
    assert set(data) == {"smooth_root_2d"}


# --------------------------------------------------------------- small runner helpers

def test_channel_usage_counts_constrained_entries_per_feature_block():
    model = SimpleNamespace(motion_rep=SimpleNamespace(
        slice_dict={"smooth_root_pos": slice(0, 3), "global_root_heading": slice(3, 5),
                    "foot_contacts": slice(5, 9)}))
    mask = torch.zeros(1, 4, 9)
    mask[0, :, 0:2] = 1.0            # 8 entries in smooth_root_pos
    mask[0, 0, 3] = 1.0              # 1 entry in global_root_heading
    assert kr.channel_usage(model, mask) == {"smooth_root_pos": 8,
                                             "global_root_heading": 1,
                                             "foot_contacts": 0}


def test_cache_guard_encoder_refuses_to_encode():
    guard = kr._CacheGuardEncoder()
    assert guard.to("cpu") is guard and guard.eval() is guard
    with pytest.raises(RuntimeError, match="text_encoder"):
        guard(["A person walks forward."])


def test_load_constraint_spec_class_returns_this_repository_dataclass():
    assert kr.load_constraint_spec_class() is ConstraintSpec


def test_generation_defaults_match_the_exp025_protocol():
    import inspect
    sig = inspect.signature(kr.KimodoRunner.generate)
    assert sig.parameters["diffusion_steps"].default == 100
    assert sig.parameters["cfg_weight"].default == (2.0, 2.0)
    assert sig.parameters["cfg_type"].default is None


# --------------------------------------------------------------- audit: counting rows

def test_audit_uses_this_repositorys_morphology():
    assert audit.matched_delta is morphology.matched_delta
    assert audit.CHANNELS is morphology.CHANNELS
    assert audit.Interaction is morphology.Interaction


def test_bits_is_identical_to_the_committed_audit_delta_rule():
    rng = np.random.default_rng(0)
    th = np.full(len(morphology.CHANNELS), 0.01)
    for _ in range(200):
        d = rng.normal(0.0, 0.05, len(morphology.CHANNELS))
        assert audit.bits(d, th) == audit_delta.bits(d, th)


def test_bits_ors_the_two_width_channels_into_one():
    th = np.zeros(8)
    left_only = np.array([-1.0, 1.0, -1.0, -1.0, -1.0, 0, 0, -1.0])
    right_only = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 0, 0, -1.0])
    assert audit.bits(left_only, th)[1] is True
    assert audit.bits(right_only, th)[1] is True
    assert audit.bits(np.full(8, -1.0), th) == (False,) * 5


def calibrated(counts):
    return next(v for k, v in counts.items() if k.startswith("6 seeds"))


def ledger():
    """Neutral (exact zeros) + a real duck + a sub-noise program, as the selftest builds.

    The third program moves a foot channel by 2 cm on every seed: enough to clear both
    naive thresholds and mint a mode of its own, but under the 5 cm noise quantile the
    prior's own seed scatter supplies, so the calibrated row folds it onto "no adaptation".
    """
    duck = np.zeros((6, 8))
    duck[:, 0] = 0.30
    small = np.zeros((6, 8))
    small[:, 3] = 0.02
    return [{"name": "neutral", "deltas": np.zeros((6, 8)).tolist(), "valid_frac": 1.0},
            {"name": "duck", "deltas": duck.tolist(), "valid_frac": 1.0},
            {"name": "small", "deltas": small.tolist(), "valid_frac": 1.0}]


def test_calibrated_counting_silences_a_sub_noise_program():
    counts = audit.count_rows(ledger(), np.full(len(morphology.CHANNELS), 0.05), 0.8)

    assert calibrated(counts) == 2.0                           # neutral + duck only
    assert counts["1 seed, any change > 1 mm"] == 3.0          # the naive rows credit it
    assert counts["1 seed, round 1 cm threshold"] == 3.0


def test_counting_drops_rows_that_never_validate_only_in_the_third_row():
    rows = ledger()
    rows[2]["valid_frac"] = 0.0                                # the small program never validates
    counts = audit.count_rows(rows, np.full(8, 0.05), 0.8)
    assert counts["1 seed, round 1 cm threshold"] == 3.0
    assert counts["1 seed, 1 cm, drop clips that never validate"] == 2.0


def test_calibrated_counting_requires_stability():
    # Half the seeds duck, half go wide: diverse, but not a controllably addressable mode.
    flapping = np.zeros((6, 8))
    flapping[:3, 0] = 0.3
    flapping[3:, 1] = 0.3
    rows = [{"name": "flapping", "deltas": flapping.tolist(), "valid_frac": 1.0}]
    assert calibrated(audit.count_rows(rows, np.full(8, 0.05), 0.8)) == 0.0


def test_calibrated_counting_skips_programs_that_validate_at_most_half_the_time():
    rows = ledger()
    for r in rows:
        r["valid_frac"] = 0.5
    assert calibrated(audit.count_rows(rows, np.full(8, 0.05), 0.8)) == 0.0
    for r in rows:
        r["valid_frac"] = 0.51
    assert calibrated(audit.count_rows(rows, np.full(8, 0.05), 0.8)) == 2.0


def test_selftest_prints_the_counts_the_exp025_protocol_requires(capsys):
    """The protocol's acceptance criterion for the recovery: 3.0 / 3.0 / 3.0 / 2.0."""
    assert audit.selftest() == 0
    printed = capsys.readouterr().out
    assert '"1 seed, any change > 1 mm": 3.0' in printed
    assert '"1 seed, round 1 cm threshold": 3.0' in printed
    assert '"1 seed, 1 cm, drop clips that never validate": 3.0' in printed
    assert '"6 seeds, paired, q99-calibrated, stability >= 0.8": 2.0' in printed


# --------------------------------------------------------------- audit: program authoring

def test_axis_rotation_is_a_proper_rotation_matrix():
    R = audit.axis_rotation([0.0, 0.0, 1.0], np.deg2rad(30.0))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_axis_rotation_turns_by_the_requested_angle_about_the_requested_axis():
    R = audit.axis_rotation([0.0, 0.0, 1.0], np.deg2rad(90.0))
    np.testing.assert_allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(R @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-9)


def test_axis_rotation_normalises_the_axis():
    a = audit.axis_rotation([0.0, 0.0, 3.0], 0.4)
    b = audit.axis_rotation([0.0, 0.0, 1.0], 0.4)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_window_ramp_is_one_inside_the_window_and_zero_far_outside():
    T, fps = 180, 30.0
    w = audit.window_ramp(T, fps)
    lo, hi = int(audit.WINDOW[0] * T), int(audit.WINDOW[1] * T)

    assert w.shape == (T,)
    assert np.all(w[lo:hi] == 1.0)
    assert w[0] == 0.0 and w[-1] == 0.0
    assert w.min() >= 0.0 and w.max() == 1.0


def test_window_ramp_eases_in_over_the_requested_ramp_length():
    T, fps = 180, 30.0
    w = audit.window_ramp(T, fps)
    lo = int(audit.WINDOW[0] * T)
    r = int(audit.RAMP_S * fps)
    rise = w[lo - r: lo]
    assert np.all(np.diff(rise) > 0)
    assert 0.0 < rise[0] < 1.0


def test_window_frames_lie_inside_the_window_at_the_requested_spacing():
    T, fps = 180, 30.0
    f = audit.window_frames(T, fps)
    lo, hi = int(audit.WINDOW[0] * T), int(audit.WINDOW[1] * T)

    assert f.min() >= lo and f.max() < hi
    assert set(np.diff(f).tolist()) == {max(1, int(audit.TARGET_STEP_S * fps))}


# --------------------------------------------------------------- audit: vendored descriptors

def synthetic_clip(T=180, dip=0.0, arm=0.25, heading=0.0, travel=5.4):
    """Kimodo-shaped output dict: y-up joints, route along +z."""
    J = 34
    z = np.linspace(0.0, travel, T)
    pj = np.zeros((T, J, 3))
    pj[..., 1] = 0.8 - dip
    pj[..., 2] = z[:, None]
    pj[:, J // 2:, 0] += arm                       # "arms" out to +x
    pj[:, 1:5, 1] = 0.05                           # feet near the floor
    root = pj[:, 0].copy()
    return {"posed_joints": pj, "root_positions": root, "smooth_root_pos": root,
            "global_root_heading": np.stack([np.full(T, np.cos(heading)),
                                             np.full(T, np.sin(heading))], -1)}


def joint_names(J=34):
    names = [f"j{i}" for i in range(J)]
    names[0] = "pelvis_skel"
    for i, n in enumerate(("left_ankle_roll_skel", "left_toe_base",
                           "right_ankle_roll_skel", "right_toe_base")):
        names[1 + i] = n
    return names


def test_travel_col_is_the_native_z_column_as_a_two_dimensional_array():
    clip = synthetic_clip()
    col = audit._travel_col(clip)
    assert col.shape == (len(clip["posed_joints"]), 1)
    np.testing.assert_allclose(col[:, 0], clip["root_positions"][:, 2])


def test_heading_angle_inverts_the_cosine_sine_encoding():
    for theta in (0.0, 0.4, -1.2):
        got = audit._heading_angle(synthetic_clip(heading=theta))
        np.testing.assert_allclose(got, np.full(len(got), theta), atol=1e-9)


def test_lateral_axis_is_perpendicular_to_the_forward_direction():
    theta = np.array([0.0, 0.4, -1.2])
    n = audit._lateral_axis(theta)
    fwd = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], -1)
    np.testing.assert_allclose(np.einsum("tk,tk->t", n, fwd), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-12)


def test_envelope_series_reports_top_and_side_extents_per_frame():
    clip = synthetic_clip(arm=0.25)
    env = audit._envelope_series(clip)
    assert env.shape == (len(clip["posed_joints"]), 3)
    np.testing.assert_allclose(env[:, 0], 0.8, atol=1e-9)      # top of the joint cloud
    # heading 0 -> lateral axis is -x, so the arms at +x show up as a RIGHT extent.
    np.testing.assert_allclose(env[:, 2], 0.25, atol=1e-9)
    np.testing.assert_allclose(env[:, 1], 0.0, atol=1e-9)


def test_descriptor_registers_a_duck_as_a_lower_top():
    names = joint_names()
    mask = np.ones(180, bool)
    tall = audit._descriptor(synthetic_clip(), names, mask)
    ducked = audit._descriptor(synthetic_clip(dip=0.3), names, mask)
    assert tall["top"] - ducked["top"] == pytest.approx(0.3, abs=1e-9)


def test_descriptor_registers_a_tuck_as_a_narrower_extent():
    names = joint_names()
    mask = np.ones(180, bool)
    wide = audit._descriptor(synthetic_clip(arm=0.25), names, mask)
    tucked = audit._descriptor(synthetic_clip(arm=0.10), names, mask)
    assert wide["w_right"] > tucked["w_right"]


def test_descriptor_falls_back_to_the_whole_clip_when_the_window_is_too_short():
    names = joint_names()
    empty = np.zeros(180, bool)
    np.testing.assert_allclose(audit._descriptor(synthetic_clip(), names, empty)["top"],
                               audit._descriptor(synthetic_clip(), names,
                                                 np.ones(180, bool))["top"])


def test_a_matched_duck_delta_registers_on_the_top_channel():
    names = joint_names()
    inter = morphology.Interaction(2.7)
    control, adapted = synthetic_clip(), synthetic_clip(dip=0.3)
    mask = inter.mask(audit._travel_col(control))
    delta = morphology.matched_delta(
        audit._descriptor(adapted, names, mask), audit._descriptor(control, names, mask),
        audit._travel_col(adapted), audit._travel_col(control), inter, 30.0,
        env_adapted=audit._envelope_series(adapted),
        env_control=audit._envelope_series(control))

    assert delta.shape == (len(morphology.CHANNELS),)
    assert delta[0] > 0.2                     # dh_top: positive means "more adaptation"


# --------------------------------------------------------------- audit: vendored validity

def good_validity_inputs(T=180, speed=0.9, seconds=6.0):
    clip = synthetic_clip(T=T, travel=speed * seconds)
    root = clip["root_positions"]
    return np.zeros((T, 36)), root, root[:, [0, 2]].copy(), speed * seconds


def test_validity_accepts_a_clean_tracked_clip():
    qpos, root, spec_xz, target = good_validity_inputs()
    v = audit._validity(qpos, root, spec_xz, target)
    assert v["valid"] and v["finite"]
    assert v["track_err_mean_m"] == pytest.approx(0.0, abs=1e-12)
    assert v["travel_m"] == pytest.approx(target, abs=1e-9)


def test_validity_rejects_a_non_finite_qpos():
    qpos, root, spec_xz, target = good_validity_inputs()
    qpos[3, 2] = np.nan
    v = audit._validity(qpos, root, spec_xz, target)
    assert not v["valid"] and not v["finite"]


def test_validity_rejects_a_clip_that_leaves_the_requested_path():
    qpos, root, spec_xz, target = good_validity_inputs()
    v = audit._validity(qpos, root, spec_xz + 0.5, target)
    assert not v["valid"]
    assert v["track_err_mean_m"] == pytest.approx(np.sqrt(0.5), abs=1e-9)


def test_validity_rejects_a_clip_that_does_not_travel():
    qpos, root, spec_xz, target = good_validity_inputs()
    stalled = root.copy()
    stalled[:, 2] = root[0, 2]
    v = audit._validity(qpos, stalled, stalled[:, [0, 2]], target)
    assert not v["valid"] and v["travel_m"] == pytest.approx(0.0)


def test_validity_rejects_a_collapsed_pelvis():
    qpos, root, spec_xz, target = good_validity_inputs()
    fallen = root.copy()
    fallen[:, 1] = 0.15
    v = audit._validity(qpos, fallen, spec_xz, target)
    assert not v["valid"] and v["pelvis_max_m"] == pytest.approx(0.15)


# --------------------------------------------------------------- audit: pinned constants

def test_arm_chain_splits_the_arm_joints_by_side():
    names = ["pelvis_skel", "left_shoulder_roll_skel", "left_elbow_skel",
             "right_shoulder_roll_skel", "right_wrist_roll_skel", "left_knee_skel"]
    left, right = audit.arm_chain(names)
    assert left.tolist() == [1, 2]
    assert right.tolist() == [3, 4]


def test_the_replication_constants_match_the_transcript_sourced_receipt():
    """The 2026-08-31 run this script replicates (docs/kimodo-provenance-2026-08-31.md)."""
    assert audit.PROMPT == "A person walks forward at a steady pace."
    assert audit.SEEDS_PAIRED == list(range(100, 106))
    assert audit.SEEDS_NULL == list(range(200, 206))
    assert audit.SECONDS_DEFAULT == 6.0 and audit.SPEED_DEFAULT == 0.9
    assert audit.STAND_PELVIS == 0.78
    assert audit.WINDOW == (0.35, 0.65)
