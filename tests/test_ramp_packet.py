import json
from dataclasses import FrozenInstanceError, asdict, dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from scene2motion.ramp import (
    MAX_PACKET_STRENGTH,
    PACKET_SCHEMA_VERSION,
    PacketControls,
    PhaseMatch,
    TargetPhaseMatch,
    constraint_support_digest,
    extract_absolute_packet,
    extract_packet_pair,
    extract_residual_packet,
    render_packet,
)
from scene2motion.ramp.packet import (
    PHASE_MATCH_SCHEMA_VERSION,
    TARGET_PHASE_MATCH_SCHEMA_VERSION,
)


NAMES = ("pelvis", "hip", "knee")
PARENTS = np.array([-1, 0, 1])
ROOT = 0
FPS = 25.0
PROTOCOL_HASH = "1" * 64
COMMON_PROTOCOL_HASH = "c" * 64
TARGET_PROTOCOL_HASH = "2" * 64


def provenance(representation="residual", *, adapted="a", neutral="b"):
    value = {
        "adapted_clip_sha256": adapted * 64,
        "checkpoint_sha256": "c" * 64,
        "generator_id": "synthetic-test-prior",
        "sampler_seed": 7,
        "noise_stream_version": 2,
        "event_selector": "synthetic-phase-apex-v1",
        "code_revision": "test-revision",
    }
    if representation == "residual":
        value["neutral_clip_sha256"] = neutral * 64
    return value


def phase_match(*, half=2, adapted_offsets=None, neutral_offsets=None,
                phase_delta=0.0, swing_side="left", adapted_stance=None,
                neutral_stance=None):
    offsets = np.arange(-half, half + 1, dtype=float)
    adapted_offsets = offsets if adapted_offsets is None else np.asarray(adapted_offsets)
    neutral_offsets = offsets if neutral_offsets is None else np.asarray(neutral_offsets)
    phases = np.mod(0.45 + np.arange(len(offsets)) * 0.05, 1.0)
    stance = "right" if swing_side == "left" else "left"
    return PhaseMatch(
        method="synthetic-common-phase-v1",
        adapted_query_offsets_frames=tuple(adapted_offsets),
        neutral_query_offsets_frames=tuple(neutral_offsets),
        adapted_phase_knots=tuple(phases),
        neutral_phase_knots=tuple(np.mod(phases + phase_delta, 1.0)),
        max_phase_error=0.02,
        adapted_stance_side=stance if adapted_stance is None else adapted_stance,
        neutral_stance_side=stance if neutral_stance is None else neutral_stance,
        adapted_stance_source="synthetic-contact-sensor-v1",
        neutral_stance_source="synthetic-contact-sensor-v1",
        adapted_stance_support_fraction=0.96,
        neutral_stance_support_fraction=0.97,
        min_stance_support_fraction=0.90,
        support_window_s=0.24,
        measurement_protocol_hash=PROTOCOL_HASH,
        common_physical_protocol_hash=COMMON_PROTOCOL_HASH,
    )


def target_phase_match(packet, event, *, query_offsets=None, packet_knots=None,
                       target_knots=None):
    query_offsets = (
        packet.source_offsets_frames if query_offsets is None else np.asarray(query_offsets))
    packet_knots = packet.phase_knots if packet_knots is None else np.asarray(packet_knots)
    target_knots = packet_knots if target_knots is None else np.asarray(target_knots)
    stance = "right" if event.side == "left" else "left"
    return TargetPhaseMatch(
        method="synthetic-target-common-phase-v1",
        target_center_frame=event.frame,
        swing_side=event.side,
        target_query_offsets_frames=tuple(query_offsets),
        packet_phase_knots=tuple(packet_knots),
        target_phase_knots=tuple(target_knots),
        max_phase_error=0.02,
        target_stance_side=stance,
        target_stance_support_fraction=0.96,
        min_stance_support_fraction=0.90,
        support_window_s=0.24,
        target_stance_source="synthetic-contact-sensor-v1",
        search_half_window_frames=max(1, int(np.ceil(np.max(np.abs(query_offsets))))),
        measurement_protocol_hash=TARGET_PROTOCOL_HASH,
        common_physical_protocol_hash=packet.common_physical_protocol_hash,
    )


@dataclass(frozen=True)
class Event:
    frame: int
    side: str = "left"


def yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation(axis, degrees):
    return Rotation.from_euler(axis, degrees, degrees=True).as_matrix()


def local_stack(T, edits=None):
    local = np.broadcast_to(np.eye(3), (T, len(NAMES), 3, 3)).copy()
    for joint, matrix in (edits or {}).items():
        local[:, joint] = matrix
    return local


def compose(local, headings):
    out = np.empty_like(local)
    for frame, heading in enumerate(headings):
        out[frame, ROOT] = yaw(heading) @ local[frame, ROOT]
        out[frame, 1] = out[frame, 0] @ local[frame, 1]
        out[frame, 2] = out[frame, 1] @ local[frame, 2]
    return out


def sample(T=13, *, height=0.8, headings=0.0, feature_headings=None, local=None):
    headings = np.broadcast_to(np.asarray(headings, dtype=float), (T,)).copy()
    feature_headings = headings if feature_headings is None else np.broadcast_to(
        np.asarray(feature_headings, dtype=float), (T,)).copy()
    local = local_stack(T) if local is None else np.asarray(local, dtype=float)
    heights = np.broadcast_to(np.asarray(height, dtype=float), (T,)).copy()
    root = np.stack([np.zeros(T), heights, np.arange(T) * 0.1], axis=-1)
    return {
        "smooth_root_pos": root,
        "global_rot_mats": compose(local, headings),
        "global_root_heading": np.stack(
            [np.cos(feature_headings), np.sin(feature_headings)], axis=-1),
    }


def extract_pair(adapted, neutral, *, center=6, half=2,
                 adapted_route=0.0, neutral_route=0.0):
    event = Event(center)
    match = phase_match(half=half)
    pair = extract_packet_pair(
        adapted, neutral, event, event, adapted_route_heading=adapted_route,
        neutral_route_heading=neutral_route, phase_match=match,
        joint_names=NAMES, parent_indices=PARENTS, root_idx=ROOT,
        source_fps=FPS, half_window_frames=half,
        absolute_provenance=provenance("absolute"),
        residual_provenance=provenance(),
    )
    return pair.absolute, pair.residual


def render(packet, target, *, event=None, headings=None, nominal_route=None,
           controls=None, first_heading=None, target_fps=FPS,
           target_query_offsets=None, target_knots=None, target_match=None):
    T = len(target["smooth_root_pos"])
    event = event or Event(T // 2)
    headings = np.zeros(T) if headings is None else np.asarray(headings, dtype=float)
    root_xz = np.stack([np.zeros(T), np.arange(T) * 0.1], axis=-1)
    return render_packet(
        packet, target, event, joint_names=NAMES, root_xz=root_xz,
        route_heading=headings,
        target_phase_match=(target_match or target_phase_match(
            packet, event, query_offsets=target_query_offsets,
            target_knots=target_knots)), target_fps=target_fps,
        controls=controls, nominal_route_heading=(
            headings if nominal_route is None else nominal_route),
        first_heading=float(headings[0]) if first_heading is None else first_heading,
    )


def local_from_global(global_rotations, route_heading):
    local = np.empty_like(global_rotations)
    local[0] = yaw(route_heading).T @ global_rotations[0]
    local[1] = global_rotations[0].T @ global_rotations[1]
    local[2] = global_rotations[1].T @ global_rotations[2]
    return local


def test_identity_donor_pair_produces_zero_residual():
    neutral = sample()
    _, packet = extract_pair(neutral, neutral)

    expected = np.broadcast_to(np.eye(3), packet.rotation_payload.shape)
    np.testing.assert_allclose(packet.rotation_payload, expected, atol=1e-12)
    np.testing.assert_allclose(packet.root_height_payload_m, 0.0)
    assert packet.source_offsets_frames.flags.writeable is False
    assert packet.rotation_payload.flags.writeable is False
    assert packet.schema_version == PACKET_SCHEMA_VERSION
    assert packet.phase_match.schema_version == PHASE_MATCH_SCHEMA_VERSION
    assert len(packet.digest()) == 64
    before = packet.digest()
    detached = packet.provenance
    detached["changed"] = True
    assert packet.digest() == before


def test_protocol_identities_are_serialized_and_hash_bound_independently():
    _, packet = extract_pair(sample(height=0.7), sample())
    metadata = packet.metadata()
    assert metadata["measurement_protocol_hash"] == PROTOCOL_HASH
    assert metadata["common_physical_protocol_hash"] == COMMON_PROTOCOL_HASH

    changed_common_match = replace(
        packet.phase_match, common_physical_protocol_hash="d" * 64
    )
    changed_common = replace(
        packet,
        common_physical_protocol_hash="d" * 64,
        phase_match=changed_common_match,
    )
    changed_full_match = replace(
        packet.phase_match, measurement_protocol_hash="e" * 64
    )
    changed_full = replace(
        packet,
        measurement_protocol_hash="e" * 64,
        phase_match=changed_full_match,
    )
    assert len({packet.digest(), changed_common.digest(), changed_full.digest()}) == 3
    with pytest.raises(ValueError, match="different common physical protocols"):
        replace(packet, common_physical_protocol_hash="d" * 64)


def test_legacy_phase_receipt_schemas_fail_closed():
    _, packet = extract_pair(sample(height=0.7), sample())

    class LegacyPhaseMatch(PhaseMatch):
        schema_version = 1

    legacy_source = LegacyPhaseMatch(**packet.phase_match.__dict__)
    with pytest.raises(ValueError, match="unsupported schema version"):
        replace(packet, phase_match=legacy_source)

    target = target_phase_match(packet, Event(6))
    legacy_target = SimpleNamespace(
        **target.__dict__,
        schema_version=1,
        as_dict=target.as_dict,
    )
    with pytest.raises(ValueError, match="unsupported schema version"):
        render(packet, sample(), event=Event(6), target_match=legacy_target)
    unversioned_target = SimpleNamespace(
        **target.__dict__,
        as_dict=target.as_dict,
    )
    with pytest.raises(ValueError, match="unsupported schema version"):
        render(packet, sample(), event=Event(6), target_match=unversioned_target)


def test_residual_composes_with_target_while_absolute_copies_donor():
    neutral_local = local_stack(13)
    adapted_local = local_stack(13, {1: rotation("x", 30), 2: rotation("z", -15)})
    target_local = local_stack(13, {
        0: rotation("x", 5), 1: rotation("y", 20), 2: rotation("x", 10),
    })
    neutral = sample(local=neutral_local, height=0.8)
    adapted = sample(local=adapted_local, height=0.7)
    target = sample(local=target_local, height=0.9)
    absolute, residual = extract_pair(adapted, neutral)

    absolute_spec, absolute_info = render(absolute, target)
    residual_spec, residual_info = render(residual, target)
    a_slot = list(absolute_spec.rot_frames).index(6)
    r_slot = list(residual_spec.rot_frames).index(6)

    expected_residual_local = (adapted_local[6]
                               @ np.swapaxes(neutral_local[6], -1, -2)
                               @ target_local[6])
    expected_residual_global = compose(expected_residual_local[None], [0.0])[0]
    expected_absolute_global = compose(adapted_local[6][None], [0.0])[0]
    np.testing.assert_allclose(residual_spec.rot_targets[r_slot],
                               expected_residual_global, atol=1e-10)
    np.testing.assert_allclose(absolute_spec.rot_targets[a_slot],
                               expected_absolute_global, atol=1e-10)
    assert not np.allclose(residual_spec.rot_targets[r_slot],
                           absolute_spec.rot_targets[a_slot])
    assert residual_spec.root_y[6] == pytest.approx(0.8)
    assert absolute_spec.root_y[6] == pytest.approx(0.7)
    assert residual_info.support_hash == absolute_info.support_hash


def test_absolute_and_residual_use_identical_channel_support():
    neutral = sample()
    adapted = sample(local=local_stack(13, {1: rotation("x", 25)}), height=0.72)
    target = sample(local=local_stack(13, {2: rotation("z", 12)}))
    absolute, residual = extract_pair(adapted, neutral)
    absolute_spec, absolute_info = render(absolute, target)
    residual_spec, residual_info = render(residual, target)

    np.testing.assert_array_equal(absolute_spec.rot_frames, residual_spec.rot_frames)
    np.testing.assert_array_equal(absolute_spec.rot_joints, residual_spec.rot_joints)
    assert absolute_spec.pos_frames is None and residual_spec.pos_frames is None
    assert constraint_support_digest(absolute_spec) == constraint_support_digest(residual_spec)
    assert absolute_info.support_hash == residual_info.support_hash

    # Lock the actual adapter indices too; aggregate channel counts could miss a frame/joint
    # substitution with the same cardinality.
    from scene2motion.constraints import ArdyConstraintSet

    def indices(spec):
        data = {name: [] for name in (
            "root_2d", "global_root_heading", "root_y_pos",
            "global_joints_rots", "global_joints_positions")}
        index = {name: [] for name in data}
        ArdyConstraintSet(spec, ROOT, "cpu").update_constraints(data, index)
        return index

    a_indices, r_indices = indices(absolute_spec), indices(residual_spec)
    for channel in ("root_2d", "root_y_pos", "global_joints_rots"):
        np.testing.assert_array_equal(a_indices[channel][0].numpy(),
                                      r_indices[channel][0].numpy())
    assert not a_indices["global_root_heading"]
    assert not r_indices["global_root_heading"]
    assert not a_indices["global_joints_positions"]
    assert not r_indices["global_joints_positions"]


def test_heading_transport_is_equivariant():
    neutral = sample(headings=0.35)
    adapted = sample(headings=0.35, local=local_stack(13, {1: rotation("x", 20)}))
    target = sample(headings=-0.2, local=local_stack(13, {2: rotation("z", 11)}))
    _, packet = extract_pair(adapted, neutral, adapted_route=0.35, neutral_route=0.35)

    zero_spec, _ = render(packet, target, headings=np.zeros(13), nominal_route=-0.2)
    turn_spec, _ = render(packet, target, headings=np.full(13, np.pi / 2),
                          nominal_route=-0.2)
    expected = np.einsum("ij,fkjl->fkil", yaw(np.pi / 2), zero_spec.rot_targets)
    np.testing.assert_allclose(turn_spec.rot_targets, expected, atol=1e-10)
    np.testing.assert_allclose(turn_spec.root_xz, zero_spec.root_xz)


@pytest.mark.parametrize(("strength", "expected_degrees"),
                         [(0.0, 0.0), (0.5, 30.0), (1.0, 60.0)])
def test_strength_uses_so3_geodesic(strength, expected_degrees):
    neutral = sample()
    adapted = sample(local=local_stack(13, {1: rotation("x", 60)}))
    target = sample()
    _, packet = extract_pair(adapted, neutral)
    spec, _ = render(packet, target, controls=PacketControls(strength=strength))
    slot = list(spec.rot_frames).index(6)
    local = local_from_global(spec.rot_targets[slot], 0.0)
    angle = np.linalg.norm(Rotation.from_matrix(local[1]).as_rotvec())
    assert np.degrees(angle) == pytest.approx(expected_degrees, abs=1e-8)
    flat = spec.rot_targets.reshape(-1, 3, 3)
    gram = np.swapaxes(flat, -1, -2) @ flat
    np.testing.assert_allclose(gram, np.broadcast_to(np.eye(3), gram.shape), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(flat), 1.0, atol=1e-10)


def test_time_warp_is_dense_unique_bounded_and_tapered():
    neutral = sample(T=21, height=0.8)
    adapted = sample(T=21, height=0.7)
    event = Event(10)
    packet = extract_residual_packet(
        adapted, neutral, event, event, adapted_route_heading=0.0,
        neutral_route_heading=0.0, phase_match=phase_match(), joint_names=NAMES,
        parent_indices=PARENTS, root_idx=ROOT, source_fps=FPS,
        half_window_frames=2, provenance=provenance(),
    )
    controls = PacketControls(center_shift_frames=2, duration_scale=2.0)
    spec, info = render(packet, neutral, event=event, controls=controls)

    assert info.target_center_frame == 12
    assert info.target_frames == tuple(range(8, 17))
    assert all(np.diff(info.target_frames) == 1)
    assert info.weights[0] == pytest.approx(0.0)
    assert info.weights[-1] == pytest.approx(0.0)
    delta = spec.root_y[list(info.target_frames)] - 0.8
    assert delta[0] == pytest.approx(0.0)
    assert delta[-1] == pytest.approx(0.0)
    assert delta[4] == pytest.approx(-0.1)
    np.testing.assert_allclose(delta, delta[::-1], atol=1e-12)


def test_cross_fps_warp_and_compression_guard():
    neutral = sample(T=21)
    adapted = sample(T=21, local=local_stack(21, {1: rotation("x", 20)}))
    event = Event(10)
    _, packet = extract_pair(adapted, neutral, center=10)

    _, info = render(
        packet, neutral, event=event, target_fps=50.0,
        target_query_offsets=(-4.0, -2.0, 0.0, 2.0, 4.0),
    )
    assert info.target_frames == tuple(range(6, 15))
    assert info.source_query_frames[0] == pytest.approx(-2.0)
    assert info.source_query_frames[-1] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="collapses"):
        render(packet, neutral, event=event,
               controls=PacketControls(duration_scale=0.1))
    with pytest.raises(ValueError, match="non-finite"):
        render(packet, neutral, event=event,
               controls=PacketControls(duration_scale=1e308))


def test_residual_is_invariant_to_common_source_yaw():
    neutral_local = local_stack(13, {2: rotation("z", 8)})
    adapted_local = local_stack(13, {1: rotation("x", 22), 2: rotation("z", 8)})
    _, packet_zero = extract_pair(sample(local=adapted_local), sample(local=neutral_local))
    _, packet_yaw = extract_pair(sample(local=adapted_local, headings=1.1),
                                 sample(local=neutral_local, headings=1.1),
                                 adapted_route=1.1, neutral_route=1.1)
    np.testing.assert_allclose(packet_zero.rotation_payload,
                               packet_yaw.rotation_payload, atol=1e-10)


def test_side_mismatch_and_out_of_bounds_render_fail_closed():
    neutral = sample()
    adapted = sample(local=local_stack(13, {1: rotation("x", 20)}))
    _, packet = extract_pair(adapted, neutral)
    with pytest.raises(ValueError, match="swing side"):
        render(packet, neutral, event=Event(6, "right"))
    with pytest.raises(ValueError, match="outside"):
        render(packet, neutral, controls=PacketControls(center_shift_frames=100))
    with pytest.raises(ValueError, match="same swing side"):
        extract_residual_packet(
            adapted, neutral, Event(6, "left"), Event(6, "right"),
            adapted_route_heading=0.0, neutral_route_heading=0.0,
            phase_match=phase_match(), joint_names=NAMES,
            parent_indices=PARENTS, root_idx=ROOT, source_fps=FPS,
            half_window_frames=2, provenance=provenance(),
        )


def test_packet_and_program_hashes_track_payload_controls_but_not_support():
    neutral = sample()
    adapted_a = sample(height=0.70, local=local_stack(13, {1: rotation("x", 20)}))
    adapted_b = sample(height=0.71, local=local_stack(13, {1: rotation("x", 20)}))
    _, packet_a = extract_pair(adapted_a, neutral)
    _, packet_b = extract_pair(adapted_b, neutral)
    target = sample()
    spec_a, info_a = render(packet_a, target)
    spec_b, info_b = render(packet_b, target)
    _, info_half = render(packet_a, target, controls=PacketControls(strength=0.5))

    assert packet_a.digest() != packet_b.digest()
    assert info_a.program_hash != info_b.program_hash
    assert info_a.program_hash != info_half.program_hash
    assert info_a.support_hash == info_b.support_hash == info_half.support_hash
    assert constraint_support_digest(spec_a) == constraint_support_digest(spec_b)
    with pytest.raises(FrozenInstanceError):
        info_a.controls.strength = 2.0


def test_program_hash_includes_first_heading_without_changing_support():
    neutral = sample()
    adapted = sample(local=local_stack(13, {1: rotation("x", 20)}))
    _, packet = extract_pair(adapted, neutral)
    _, info_zero = render(packet, neutral, first_heading=0.0)
    _, info_turn = render(packet, neutral, first_heading=0.3)
    assert info_zero.program_hash != info_turn.program_hash
    assert info_zero.support_hash == info_turn.support_hash


def test_adapted_and_neutral_events_are_aligned_independently():
    adapted = sample(T=15, local=local_stack(15, {1: rotation("x", 20)}), height=0.72)
    neutral = sample(T=15)
    packet = extract_residual_packet(
        adapted, neutral, Event(5), Event(8), adapted_route_heading=0.0,
        neutral_route_heading=0.0, phase_match=phase_match(), joint_names=NAMES,
        parent_indices=PARENTS, root_idx=ROOT, source_fps=FPS,
        half_window_frames=2, provenance=provenance(),
    )
    assert packet.adapted_center_frame == 5
    assert packet.neutral_center_frame == 8
    np.testing.assert_array_equal(packet.source_offsets_frames, [-2, -1, 0, 1, 2])


def test_target_equal_neutral_reconstructs_adapted_event_under_new_route_yaw():
    neutral_local = local_stack(13, {1: rotation("y", 7), 2: rotation("z", -4)})
    adapted_local = local_stack(13, {
        0: rotation("x", 3), 1: rotation("x", 25) @ rotation("y", 7),
        2: rotation("z", 13) @ rotation("z", -4),
    })
    neutral = sample(local=neutral_local, headings=0.4, height=0.81)
    adapted = sample(local=adapted_local, headings=0.4, height=0.73)
    _, packet = extract_pair(adapted, neutral, adapted_route=0.4, neutral_route=0.4)
    route_yaw = 0.9
    spec, _ = render(packet, neutral, headings=np.full(13, route_yaw),
                     nominal_route=0.4)
    slot = list(spec.rot_frames).index(6)
    expected = compose(adapted_local[6][None], [route_yaw])[0]

    np.testing.assert_allclose(spec.rot_targets[slot], expected, atol=1e-10)
    assert spec.root_y[6] == pytest.approx(0.73)


def test_body_turn_relative_to_route_is_retained_even_when_heading_feature_disagrees():
    neutral = sample()
    turned_local = local_stack(13, {ROOT: yaw(np.pi / 2)})
    adapted = sample(local=turned_local, headings=0.0, feature_headings=np.pi / 2)
    _, packet = extract_pair(adapted, neutral, adapted_route=0.0, neutral_route=0.0)

    center = list(packet.source_offsets_frames).index(0)
    packet_angle = np.linalg.norm(
        Rotation.from_matrix(packet.rotation_payload[center, ROOT]).as_rotvec())
    assert np.degrees(packet_angle) == pytest.approx(90.0)
    spec, _ = render(packet, neutral, headings=np.zeros(13), nominal_route=0.0)
    slot = list(spec.rot_frames).index(6)
    np.testing.assert_allclose(spec.rot_targets[slot, ROOT], yaw(np.pi / 2), atol=1e-10)
    assert spec.heading is None


def test_common_phase_resampling_removes_different_source_cadences():
    frames = np.arange(21) - 10
    neutral_local = local_stack(21)
    adapted_local = local_stack(21)
    neutral_local[:, 1] = Rotation.from_euler(
        "x", (frames * 10)[:, None], degrees=True).as_matrix()
    adapted_local[:, 1] = Rotation.from_euler(
        "x", (frames * 5)[:, None], degrees=True).as_matrix()
    neutral = sample(T=21, local=neutral_local, height=0.8 + frames * 0.01)
    adapted = sample(T=21, local=adapted_local, height=0.8 + frames * 0.005)
    match = phase_match(
        adapted_offsets=(-4.0, -2.0, 0.0, 2.0, 4.0),
        neutral_offsets=(-2.0, -1.0, 0.0, 1.0, 2.0),
    )
    pair = extract_packet_pair(
        adapted, neutral, Event(10), Event(10), adapted_route_heading=0.0,
        neutral_route_heading=0.0, phase_match=match, joint_names=NAMES,
        parent_indices=PARENTS, root_idx=ROOT, source_fps=FPS,
        half_window_frames=2, absolute_provenance=provenance("absolute"),
        residual_provenance=provenance(),
    )
    packet = pair.residual
    np.testing.assert_array_equal(
        pair.absolute.adapted_query_offsets_frames,
        pair.residual.adapted_query_offsets_frames,
    )
    np.testing.assert_allclose(
        pair.absolute.adapted_query_offsets_frames, (-4, -2, 0, 2, 4))
    expected = np.broadcast_to(np.eye(3), packet.rotation_payload.shape)
    np.testing.assert_allclose(packet.rotation_payload, expected, atol=1e-10)
    np.testing.assert_allclose(packet.root_height_payload_m, 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "match, message",
    [
        (phase_match(phase_delta=0.03), "phase error"),
        (phase_match(adapted_stance="left"), "stance contact"),
        (
            phase_match(adapted_offsets=(-2.0, -1.0, 0.0, 0.0, 2.0)),
            "strictly increasing",
        ),
    ],
)
def test_phase_or_contact_mismatch_fails_closed(match, message):
    with pytest.raises(ValueError, match=message):
        extract_residual_packet(
            sample(), sample(), Event(6), Event(6), adapted_route_heading=0.0,
            neutral_route_heading=0.0, phase_match=match, joint_names=NAMES,
            parent_indices=PARENTS, root_idx=ROOT, source_fps=FPS,
            half_window_frames=2, provenance=provenance(),
        )


def test_strength_extrapolation_has_a_per_program_so3_branch_guard():
    neutral = sample()
    adapted_safe = sample(local=local_stack(13, {1: rotation("x", 60)}))
    _, safe_packet = extract_pair(adapted_safe, neutral)
    safe_spec, _ = render(
        safe_packet, neutral, controls=PacketControls(strength=MAX_PACKET_STRENGTH))
    slot = list(safe_spec.rot_frames).index(6)
    local = local_from_global(safe_spec.rot_targets[slot], 0.0)
    assert np.degrees(np.linalg.norm(
        Rotation.from_matrix(local[1]).as_rotvec())) == pytest.approx(120.0)

    adapted_unsafe = sample(local=local_stack(13, {1: rotation("x", 100)}))
    _, unsafe_packet = extract_pair(adapted_unsafe, neutral)
    with pytest.raises(ValueError, match="principal branch"):
        render(unsafe_packet, neutral,
               controls=PacketControls(strength=MAX_PACKET_STRENGTH))
    with pytest.raises(ValueError, match="strength"):
        PacketControls(strength=MAX_PACKET_STRENGTH + 0.01)


def test_resampled_delta_cannot_cross_the_pi_log_branch_between_safe_knots():
    adapted_local = local_stack(13)
    adapted_local[6, 1] = rotation("z", 170)
    adapted_local[7, 1] = rotation("z", -170)
    _, packet = extract_pair(sample(local=adapted_local), sample())

    with pytest.raises(ValueError, match="resampled packet.*pi branch"):
        render(
            packet,
            sample(),
            controls=PacketControls(strength=0.5, duration_scale=2.0),
        )


def test_source_phase_resampling_occurs_in_hierarchy_local_rotation_space():
    adapted_local = local_stack(13)
    adapted_local[6, 1] = np.eye(3)
    adapted_local[6, 2] = rotation("x", 60)
    adapted_local[7, 1] = rotation("z", 90)
    adapted_local[7, 2] = rotation("y", 60)
    query_offsets = (-2.0, -1.0, 0.0, 0.5, 2.0)
    match = phase_match(
        adapted_offsets=query_offsets,
        neutral_offsets=query_offsets,
    )
    packet = extract_residual_packet(
        sample(local=adapted_local), sample(), Event(6), Event(6),
        adapted_route_heading=0.0, neutral_route_heading=0.0,
        phase_match=match, joint_names=NAMES, parent_indices=PARENTS,
        root_idx=ROOT, source_fps=FPS, half_window_frames=2,
        provenance=provenance(),
    )

    expected_parent = rotation("z", 45)
    expected_child = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix([rotation("x", 60), rotation("y", 60)]),
    )([0.5]).as_matrix()[0]
    np.testing.assert_allclose(packet.rotation_payload[3, 1], expected_parent, atol=1e-10)
    np.testing.assert_allclose(packet.rotation_payload[3, 2], expected_child, atol=1e-10)


def test_near_pi_residual_is_rejected_at_packet_construction():
    adapted = sample(local=local_stack(13, {1: rotation("x", 180)}))
    with pytest.raises(ValueError, match=r"ambiguous SO\(3\) pi branch"):
        extract_pair(adapted, sample())


@pytest.mark.parametrize(
    "bad_event, bad_parents, bad_half, message",
    [
        (Event(6.5), PARENTS, 2, "event frame"),
        (Event(6), np.array([-1.0, 0.0, 1.0]), 2, "parent_indices"),
        (
            Event(6),
            np.array([2**64 - 1, 0, 1], dtype=np.uint64),
            2,
            "outside signed int64",
        ),
        (Event(6), PARENTS, 2.0, "half_window_frames"),
    ],
)
def test_discrete_schema_fields_do_not_silently_truncate(
    bad_event, bad_parents, bad_half, message
):
    with pytest.raises(ValueError, match=message):
        extract_absolute_packet(
            sample(), bad_event, adapted_route_heading=0.0,
            adapted_query_offsets_frames=phase_match().adapted_query_offsets_frames,
            adapted_phase_knots=phase_match().adapted_phase_knots,
            joint_names=NAMES,
            parent_indices=bad_parents, root_idx=ROOT, source_fps=FPS,
            half_window_frames=bad_half, measurement_protocol_hash=PROTOCOL_HASH,
            common_physical_protocol_hash=COMMON_PROTOCOL_HASH,
            provenance=provenance("absolute"),
        )


def test_provenance_is_required_and_metadata_is_json_serializable():
    incomplete = provenance("absolute")
    incomplete.pop("checkpoint_sha256")
    with pytest.raises(ValueError, match="missing required fields"):
        extract_absolute_packet(
            sample(), Event(6), adapted_route_heading=0.0,
            adapted_query_offsets_frames=phase_match().adapted_query_offsets_frames,
            adapted_phase_knots=phase_match().adapted_phase_knots,
            joint_names=NAMES,
            parent_indices=PARENTS, root_idx=ROOT, source_fps=FPS,
            half_window_frames=2, measurement_protocol_hash=PROTOCOL_HASH,
            common_physical_protocol_hash=COMMON_PROTOCOL_HASH,
            provenance=incomplete,
        )

    _, packet = extract_pair(sample(height=0.7), sample())
    _, info = render(packet, sample())
    json.dumps(packet.metadata(), allow_nan=False)
    json.dumps(asdict(info), allow_nan=False)


def test_paired_extractor_rejects_mismatched_generator_provenance():
    match = phase_match()
    absolute_provenance = provenance("absolute")
    absolute_provenance["code_revision"] = "different-revision"
    with pytest.raises(ValueError, match="pair provenance differs in code_revision"):
        extract_packet_pair(
            sample(), sample(), Event(6), Event(6), adapted_route_heading=0.0,
            neutral_route_heading=0.0, phase_match=match, joint_names=NAMES,
            parent_indices=PARENTS, root_idx=ROOT, source_fps=FPS,
            half_window_frames=2, absolute_provenance=absolute_provenance,
            residual_provenance=provenance(),
        )


def test_real_integer_noise_version_and_numpy_first_heading_serialize():
    _, packet = extract_pair(sample(height=0.7), sample())
    assert packet.provenance["noise_stream_version"] == 2
    _, info = render(packet, sample(), first_heading=np.float32(0.25))
    json.dumps(asdict(info), allow_nan=False)


def test_render_consumes_and_hashes_target_phase_mapping_not_only_event_center():
    _, packet = extract_pair(
        sample(local=local_stack(21, {1: rotation("x", 30)}), T=21),
        sample(T=21), center=10,
    )
    target = sample(T=21)
    fast_spec, fast_info = render(packet, target, event=Event(10))
    slow_spec, slow_info = render(
        packet,
        target,
        event=Event(10),
        target_query_offsets=(-4.0, -2.0, 0.0, 2.0, 4.0),
    )

    assert fast_info.target_frames == tuple(range(8, 13))
    assert slow_info.target_frames == tuple(range(6, 15))
    np.testing.assert_allclose(
        slow_info.source_query_frames,
        np.linspace(-2.0, 2.0, 9),
        atol=1e-12,
    )
    assert fast_info.target_phase_match_hash != slow_info.target_phase_match_hash
    assert fast_info.program_hash != slow_info.program_hash
    assert constraint_support_digest(fast_spec) != constraint_support_digest(slow_spec)
    json.loads(slow_info.target_phase_match_json)


def test_timing_controls_warp_nominal_height_with_rotations_and_taper_boundaries():
    frame_height = 0.70 + np.arange(21) * 0.01
    nominal = sample(T=21, height=frame_height)
    _, packet = extract_pair(nominal, nominal, center=10)
    controls = PacketControls(
        strength=0.0, center_shift_frames=2, duration_scale=2.0)
    spec, info = render(packet, nominal, event=Event(10), controls=controls)

    assert info.target_center_frame == 12
    assert info.target_frames == tuple(range(8, 17))
    # The shifted event centre carries nominal phase/frame 10, not output frame 12.
    assert spec.root_y[12] == pytest.approx(frame_height[10])
    # The temporal warp closes smoothly onto the unwarped nominal substrate.
    assert spec.root_y[8] == pytest.approx(frame_height[8])
    assert spec.root_y[16] == pytest.approx(frame_height[16])


def test_target_phase_receipt_from_another_packet_fails_closed():
    _, packet = extract_pair(sample(height=0.7), sample())
    event = Event(6)
    wrong_knots = np.mod(packet.phase_knots + 0.10, 1.0)
    receipt = target_phase_match(packet, event, packet_knots=wrong_knots)
    with pytest.raises(ValueError, match="different packet grid"):
        render(packet, sample(), event=event, target_match=receipt)


def test_target_phase_receipt_preserves_distinct_target_measurement_protocol():
    _, packet = extract_pair(sample(height=0.7), sample())
    event = Event(6)
    baseline_receipt = target_phase_match(packet, event)
    _, baseline_info = render(packet, sample(), event=event, target_match=baseline_receipt)
    receipt = replace(
        target_phase_match(packet, event), measurement_protocol_hash="3" * 64
    )
    _, info = render(packet, sample(), event=event, target_match=receipt)
    assert info.measurement_protocol_hash == packet.measurement_protocol_hash
    assert info.target_measurement_protocol_hash == "3" * 64
    assert info.common_physical_protocol_hash == packet.common_physical_protocol_hash
    archived = json.loads(info.target_phase_match_json)
    assert archived["measurement_protocol_hash"] == "3" * 64
    assert archived["common_physical_protocol_hash"] == COMMON_PROTOCOL_HASH
    assert archived["schema_version"] == TARGET_PHASE_MATCH_SCHEMA_VERSION
    assert info.target_phase_match_hash != baseline_info.target_phase_match_hash
    assert info.program_hash != baseline_info.program_hash


def test_target_phase_receipt_with_another_common_protocol_fails_closed():
    _, packet = extract_pair(sample(height=0.7), sample())
    event = Event(6)
    receipt = replace(
        target_phase_match(packet, event), common_physical_protocol_hash="3" * 64
    )
    with pytest.raises(ValueError, match="different common physical protocols"):
        render(packet, sample(), event=event, target_match=receipt)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"adapted_phase_knots": (0.4,) * 5}, "advance strictly"),
        ({"min_stance_support_fraction": 0.0}, "below the locked threshold"),
        ({"adapted_stance_source": ""}, "evidence source"),
        ({"max_phase_error": 0.5}, r"\[0, 0.5\)"),
        ({"measurement_protocol_hash": "not-a-hash"}, "SHA-256"),
        ({"common_physical_protocol_hash": "not-a-hash"}, "SHA-256"),
    ],
)
def test_manual_phase_receipts_cannot_bypass_alignment_evidence(change, message):
    with pytest.raises(ValueError, match=message):
        replace(phase_match(), **change)
