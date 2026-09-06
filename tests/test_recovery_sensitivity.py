import copy

import pytest

from scene2motion.recovery_sensitivity import sensitivity


def fixture_rows():
    return [{"unit_id": "u", "split_group_id": "g", "mode": "default", "arm": a, "slot_repeat": i,
             "operational_local_pass": True, "recovery": {"window_complete": True,
             "root_dx_m": -.000313 if a == "fixed2" else .1,
             "first_observed_violation_sample": {"root": None}}}
            for a in ("fixed2", "fixed3") for i in range(4)]


def test_strict_boundary_and_stationary_tolerance():
    result = sensitivity(fixture_rows(), (-.001, 0., .1))
    assert [r["fixed2"] for r in result["curve"]] == [4, 0, 0]
    assert [r["fixed3"] for r in result["curve"]] == [4, 4, 0]
    assert result["recommended_threshold_m"] is None


def test_censoring_and_violations_cannot_be_hidden_by_tolerance():
    rows = fixture_rows()
    before = copy.deepcopy(rows)
    rows[0]["recovery"]["window_complete"] = False
    rows[1]["recovery"]["first_observed_violation_sample"]["root"] = 4
    assert sensitivity(rows, (-1.,))["curve"][0]["fixed2"] == 2
    assert before[0]["recovery"]["window_complete"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "lineage", "nan"])
def test_invalid_source_refused(defect):
    rows = fixture_rows()
    if defect == "missing": rows.pop()
    elif defect == "duplicate": rows.append(rows[0])
    elif defect == "lineage": rows[0]["split_group_id"] = "other"
    else: rows[0]["recovery"]["root_dx_m"] = float("nan")
    with pytest.raises(ValueError):
        sensitivity(rows)
