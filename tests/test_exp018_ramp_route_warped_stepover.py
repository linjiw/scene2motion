import json
from pathlib import Path

import numpy as np
import pytest

from experiments import calibrate_ramp_route_phase as cal
from experiments import exp018_ramp_route_warped_stepover as pilot


REPO = Path(pilot.__file__).resolve().parents[1]


def test_pool_batch_plan_is_six_batches_of_eight_over_locked_seeds():
    batches = pilot.pool_batch_plan()
    assert len(batches) == 6
    assert all(len(batch["seeds"]) == 8 for batch in batches)
    seeds = sorted(seed for batch in batches for seed in batch["seeds"])
    assert seeds == sorted(list(pilot.POOL_SEEDS) * 3)
    labels = [batch["speed_label"] for batch in batches]
    assert labels == ["slow", "reference", "fast"] * 2
    speeds = {batch["speed_label"]: batch["requested_speed_mps"] for batch in batches}
    assert speeds == dict(cal.SPEEDS)


def test_locked_design_constants_are_disjoint_and_consistent():
    assert set(pilot.DONOR_SEEDS).isdisjoint(pilot.POOL_SEEDS)
    prior = set(range(3200, 3216)) | set(range(3300, 3308)) | \
        set(range(3400, 3416)) | set(range(3500, 3508)) | \
        set(range(3600, 3616)) | set(range(3700, 3708)) | set(range(2800, 2808))
    assert set(pilot.POOL_SEEDS).isdisjoint(prior)
    assert pilot.N_SELECT <= len(pilot.POOL_SEEDS)
    # Maximum completed budget: 2D + 3K + N + 2M with M <= N.
    assert 2 * len(pilot.DONOR_SEEDS) + 3 * len(pilot.POOL_SEEDS) \
        + pilot.N_SELECT + 2 * pilot.N_SELECT == 74


def _pilot_program(event_frame=100, event_progress=3.6):
    return cal.reparameterize_route_progress(
        np.asarray([[0.0, 0.0], [0.0, pilot.ROUTE_LENGTH_M]], dtype=float),
        n_frames=pilot.N_FRAMES,
        event_frame=event_frame,
        event_root_progress_m=event_progress,
        timing_bounds=cal.broad_timing_bounds(),
    )


def test_warped_route_materializes_program_progress_on_straight_route():
    program = _pilot_program()
    route = pilot.warped_route_xz(program)
    assert route.shape == (pilot.N_FRAMES, 2)
    assert np.all(route[:, 0] == 0.0)
    assert route[0, 1] == pytest.approx(0.0)
    assert route[-1, 1] == pytest.approx(pilot.ROUTE_LENGTH_M)
    assert route[100, 1] == pytest.approx(3.6, abs=1e-9)
    assert np.all(np.diff(route[:, 1]) > 0.0)


def test_warp_drives_the_exp017_placement_shift_to_zero():
    """The design mechanism: a persisting apex needs no center shift at all.

    exp017's fixed-frame pool died on required shifts of +65/+28/-15 frames against
    a +/-8 bound.  Under the calibrated warp the obstacle progress is placed *at* the
    apex frame, so the same unchanged assignment gate sees a zero shift.
    """
    receipt = json.loads((REPO / pilot.V3_RECEIPT_PATH).read_text())
    bounds = cal.timing_bounds_from_receipt(receipt["route_timing_receipt"])
    selected = receipt["validation"]["pooled"]["selected_programs"]
    assert selected
    for item in selected:
        cycle = item["cycle"]
        placement = float(item["placement_m"])
        offset = float(cycle["nominal_foot_forward_offset_m"])
        program = cal.reparameterize_route_progress(
            np.asarray([[0.0, 0.0], [0.0, pilot.ROUTE_LENGTH_M]], dtype=float),
            n_frames=pilot.N_FRAMES,
            event_frame=cycle["apex_frame"],
            event_root_progress_m=placement - offset,
            timing_bounds=bounds,
        )
        route = pilot.warped_route_xz(program)
        desired_frame = int(np.argmin(np.abs(route[:, 1] - (placement - offset))))
        assert desired_frame == cycle["apex_frame"]
        assert abs(desired_frame - cycle["apex_frame"]) <= \
            pilot.MAX_CENTER_SHIFT_FRAMES
        predicted_foot = float(route[desired_frame, 1]) + offset
        assert predicted_foot == pytest.approx(placement, abs=1e-6)


def test_v3_receipt_loads_with_frozen_quantities_and_rejects_tampering(tmp_path):
    calibration = pilot.load_v3_calibration(REPO / pilot.V3_RECEIPT_PATH)
    assert calibration["target_min_prominence_m"] == pytest.approx(0.042)
    bounds = calibration["route_timing_bounds"]
    assert bounds.max_abs_route_progress_acceleration_mps2 == pytest.approx(0.19)
    assert bounds.max_abs_discrete_route_progress_jerk_mps3 == pytest.approx(1.2)
    assert bounds.max_endpoint_route_progress_speed_deviation_mps == pytest.approx(
        0.17)
    assert calibration["event_placements_m"] == list(cal.EVENT_PLACEMENTS_M)

    tampered = tmp_path / "receipt.json"
    payload = json.loads((REPO / pilot.V3_RECEIPT_PATH).read_text())
    payload["validation"]["passed"] = True
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not the locked v3 artifact"):
        pilot.load_v3_calibration(tampered)


def test_pmin_receipt_digest_is_verified():
    receipt = json.loads((REPO / pilot.V3_RECEIPT_PATH).read_text())
    receipt["target_prominence_receipt"]["fields"][
        "target_min_prominence_m"] = 0.001
    with pytest.raises(ValueError, match="digest mismatch"):
        pilot.pmin_from_receipt(receipt)


def test_archived_donor_bundle_loads_locked_identity():
    archived = pilot.load_archived_donor_bundle(REPO)
    assert archived["archived_pair_hash"] == (
        "5f056ec098213d7885aad0af307811fb88ad073b2550ad62bd5b0d297a675303")
    assert archived["archived_payload_content_sha256"] == (
        pilot.EXPECTED_PACKET_PAYLOAD_CONTENT_SHA256)
    clips = archived["archived_clip_sha256_by_seed"]
    assert sorted(int(seed) for seed in clips) == sorted(pilot.DONOR_SEEDS)
    assert clips[str(pilot.EXPECTED_SELECTED_DONOR_SEED)][0] == (
        pilot.EXPECTED_ADAPTED_CLIP_SHA256)
    assert clips[str(pilot.EXPECTED_SELECTED_DONOR_SEED)][1] == (
        pilot.EXPECTED_NEUTRAL_CLIP_SHA256)


def test_run_pilot_refuses_nonempty_output(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale").write_text("x")
    with pytest.raises(pilot.PilotAbort, match="nonempty"):
        pilot.run_pilot(
            out=out,
            threshold_receipt=REPO / "outputs/exp016_threshold_calibration/receipt.json",
        )
    assert (out / "stale").read_text() == "x"
    assert not (out / "receipt.json").exists()


def test_run_pilot_dirty_worktree_fails_closed_with_receipt(tmp_path):
    def dirty(_repo):
        return {"commit": "c" * 40, "dirty": True, "status": ["?? x"],
                "tracked_diff_sha256": "d" * 64}

    with pytest.raises(pilot.PilotAbort, match="clean git worktree"):
        pilot.run_pilot(
            out=tmp_path / "out",
            threshold_receipt=REPO / "outputs/exp016_threshold_calibration/receipt.json",
            code_state_fn=dirty,
        )
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert receipt["schema"] == pilot.FAILURE_SCHEMA_VERSION
    assert receipt["status"] == "blocked"
    assert receipt["failed_stage"] == "preflight"
    assert receipt["returned_ardy_samples_lower_bound"] == 0
    assert receipt["conservative_charged_ardy_samples"] == 0


def test_run_pilot_rejects_wrong_threshold_dependency(tmp_path):
    def clean(_repo):
        return {"commit": "c" * 40, "dirty": False, "status": [],
                "tracked_diff_sha256": "d" * 64}

    wrong = tmp_path / "thresholds.json"
    wrong.write_text("{}")
    with pytest.raises(pilot.PilotAbort, match="physical threshold"):
        pilot.run_pilot(
            out=tmp_path / "out",
            threshold_receipt=wrong,
            code_state_fn=clean,
        )
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert receipt["failed_stage"] == "preflight"
