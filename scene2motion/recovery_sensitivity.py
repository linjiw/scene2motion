"""Post-hoc displacement sensitivity, never an endpoint or quality calibration."""

import math


def sensitivity(rows, thresholds_m=(-.001, 0., .01, .05, .10, .25, .50)):
    selected = [r for r in rows if r["mode"] == "default" and r["arm"] in ("fixed2", "fixed3")]
    index = {(r["unit_id"], r["slot_repeat"], r["arm"]): r for r in selected}
    units = sorted({r["unit_id"] for r in selected})
    expected = {(u, i, a) for u in units for i in range(4) for a in ("fixed2", "fixed3")}
    if not units or len(index) != len(selected) or set(index) != expected:
        raise ValueError("complete paired hold/repeat membership required")
    for u in units:
        if len({r["split_group_id"] for r in selected if r["unit_id"] == u}) != 1:
            raise ValueError("unit lineage differs")
    def passes(row, threshold):
        r = row["recovery"]
        if row["operational_local_pass"] is not True or not r["window_complete"]:
            return False
        dx = r["root_dx_m"]
        if dx is None or not math.isfinite(dx):
            raise ValueError("complete recovery needs finite displacement")
        return dx > threshold and all(v is None for v in r["first_observed_violation_sample"].values())
    curve = []
    for threshold in thresholds_m:
        if not math.isfinite(threshold):
            raise ValueError("finite threshold required")
        by_unit = []
        for u in units:
            arms = {a: [passes(index[u, i, a], threshold) for i in range(4)] for a in ("fixed2", "fixed3")}
            by_unit.append({"unit_id": u, "fixed2": sum(arms["fixed2"]), "fixed3": sum(arms["fixed3"]),
                "unit_fixed_hold_oracle": max(map(sum, arms.values())),
                "slotwise_oracle": sum(a or b for a, b in zip(arms["fixed2"], arms["fixed3"]))})
        curve.append({"strict_net_displacement_gt_m": threshold, "by_unit": by_unit,
                      **{k: sum(r[k] for r in by_unit) for k in ("fixed2", "fixed3", "unit_fixed_hold_oracle", "slotwise_oracle")}})
    return {"schema_version": "astra-recovery-sensitivity-v1", "assigned_units": len(units),
        "assigned_slots": 4 * len(units), "curve": curve, "new_executions": 0,
        "scope": "post-hoc threshold sensitivity on a spent development pool; not prospective qualification, contact or safety",
        "recommended_threshold_m": None, "historical_labels_unchanged": True}
