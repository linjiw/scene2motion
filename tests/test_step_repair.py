"""Regression tests for the constructive low-obstacle repair operator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scene2motion.robot import G1Body
from scene2motion.step_repair import (
    FootstepRepairConfig,
    SupportRule,
    repair_step_reference,
    support_report,
)
from scene2motion.stepover_eval import step_scene


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs/exp021_elicited_lift_distribution_v2/qpos.npz"
THRESHOLDS = ROOT / "outputs/exp016_threshold_calibration/receipt.json"


def _support_rule() -> SupportRule:
    values = json.loads(THRESHOLDS.read_text())["stepover_thresholds"]
    return SupportRule(
        support_height_m=float(values["support_height_m"]),
        support_speed_mps=float(values["support_speed_mps"]),
        max_unsupported_run_s=float(values["max_unsupported_run_s"]),
    )


def _clip(key: str) -> np.ndarray:
    with np.load(ARCHIVE) as archive:
        return np.asarray(archive[key], dtype=float)


def test_s4434_repairs_the_reference_gap_without_moving_the_root():
    """The known route-completing substrate becomes a supported, exact-centre 5 cm candidate.

    This remains a reference-level result.  The test intentionally does not use the archived
    controller outcome as evidence that the edited reference will track.
    """

    original = _clip("s4434")
    body = G1Body(None)
    obstacle_body = G1Body(step_scene(1.2, 0.05, 0.20))
    result = repair_step_reference(
        original, fps=25.0, obstacle_x_m=1.2, obstacle_height_m=0.05,
        obstacle_depth_m=0.20, support_rule=_support_rule(),
        body=body, obstacle_body=obstacle_body,
    )

    assert result.record["accepted"] is True
    assert result.record["status"] == "accepted"
    assert result.record["before"]["collision"]["collision_free"] is False
    assert result.record["after"]["collision"]["collision_free"] is True
    assert result.record["after"]["support"]["passes"] is True
    assert result.record["after"]["support"]["longest_unsupported_run_s"] == pytest.approx(0.20)
    assert np.array_equal(result.qpos[:, :7], original[:, :7])
    assert np.array_equal(result.qpos[:, 19:], original[:, 19:])
    assert result.record["deformation"]["max_leg_joint_delta_rad"] < 0.50
    assert result.record["deformation"]["max_ik_target_residual_m"] < 0.005
    assert result.record["deformation"]["max_post_smoothing_target_residual_m"] < 0.025
    assert result.record["deformation"]["max_pointwise_joint_speed_increase_rads"] < 2.0


def test_a_support_failing_input_is_refused_without_modification():
    original = _clip("s4402")
    result = repair_step_reference(
        original, fps=25.0, obstacle_x_m=1.2, obstacle_height_m=0.05,
        obstacle_depth_m=0.20, support_rule=_support_rule(),
    )
    assert result.record["status"] == "refused"
    assert result.record["reasons"] == ["input_support_screen_failed"]
    assert result.record["after"] is None
    assert np.array_equal(result.qpos, original)


def test_archived_support_passing_pool_yields_two_preexecution_candidates():
    """Pin the exploratory pool accounting that determines EXP-031's launch list.

    The denominator remains all 64 source references: 56 are refused by the frozen support
    screen, six fail a post-projection admission condition, and two are admitted for the
    obstacle-present engineering pilot.
    """

    rule = _support_rule()
    body = G1Body(None)
    obstacle_body = G1Body(step_scene(1.2, 0.05, 0.20))
    support_passing = []
    accepted = []
    with np.load(ARCHIVE) as archive:
        for key in archive.files:
            qpos = np.asarray(archive[key], dtype=float)
            if not support_report(body, qpos, 25.0, rule)["passes"]:
                continue
            support_passing.append(key)
            result = repair_step_reference(
                qpos, fps=25.0, obstacle_x_m=1.2, obstacle_height_m=0.05,
                obstacle_depth_m=0.20, support_rule=rule,
                body=body, obstacle_body=obstacle_body,
            )
            if result.record["accepted"]:
                accepted.append(key)

    assert support_passing == [
        "s4408", "s4411", "s4418", "s4419",
        "s4434", "s4440", "s4452", "s4459",
    ]
    assert accepted == ["s4408", "s4434"]


def test_invalid_rules_and_budgets_fail_closed():
    with pytest.raises(ValueError, match="support-rule"):
        SupportRule(-0.01, 1.0).validate()
    with pytest.raises(ValueError, match="positive repair"):
        FootstepRepairConfig(max_joint_delta_rad=0.0).validate()


def test_support_report_names_the_measured_construct():
    report = support_report(G1Body(None), _clip("s4434"), 25.0, _support_rule())
    assert report["passes"] is True
    assert report["longest_unsupported_run_frames"] == 4
    assert report["longest_unsupported_run_s"] == pytest.approx(0.16)
    assert set(report["supported_fraction"]) == {"left", "right"}
