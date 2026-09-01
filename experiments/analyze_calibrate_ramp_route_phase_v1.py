"""Diagnosis of the 2026-09-01 route-phase calibration v1 refusal.

Reads only the durable refusal evidence in ``outputs/calibrate_ramp_route_phase``
(the v1 campaign, schema ``ramp-route-phase-calibration-v1``) and reports the three
failure sources that sized the v2 preregistration:

1. neutral-substrate attrition: clips with zero physically-complete unilateral
   swing cycles, concentrated in seeds whose gait fails discovery at every speed;
2. timing-envelope washout: cycles whose apex cannot be warped to a fixed event
   progress within the frozen [0.6, 1.2] m/s scalar route-speed envelope;
3. endpoint-stratum brittleness: slow/fast coverage rules sized above the measured
   per-clip washout rate.

It also replays the candidate v2 rule set (reference-gated validation, five-point
event placement set) against the v1 evidence.  Everything printed here is
design-informing analysis of a refusal; it is not confirmatory evidence for v2,
which must run on fresh seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import calibrate_ramp_route_phase as cal  # noqa: E402

V1_CALIBRATION_SEEDS = tuple(range(3200, 3216))
V1_VALIDATION_SEEDS = tuple(range(3300, 3308))
V1_FIXED_OBSTACLE_M = 3.6
CANDIDATE_PLACEMENTS_M = (3.0, 3.3, 3.6, 3.9, 4.2)


def _load_rows(out_dir: Path) -> list[dict]:
    receipt = json.loads((out_dir / "receipt.json").read_text())
    if receipt.get("schema") != "ramp-route-phase-calibration-failure-v1":
        raise SystemExit(
            "this analysis reads the v1 refusal evidence; got schema "
            f"{receipt.get('schema')!r}"
        )
    if receipt.get("failed_stage") != "calibration":
        raise SystemExit(f"unexpected failed stage {receipt.get('failed_stage')!r}")
    accounting = receipt["query_accounting"]
    if accounting["samples_analyzed"] != 72:
        raise SystemExit("v1 evidence does not contain the complete 72 analyzed samples")
    return [json.loads(line) for line in (out_dir / "rows.jsonl").open()]


def _placement_candidates(row, timing, pmin, placements, route):
    out = []
    for cycle in row["cycles"]:
        if not (cycle["prominence_m"] >= pmin and cycle["packet_window_valid"]):
            continue
        for placement in placements:
            s_event = placement - cycle["nominal_foot_forward_offset_m"]
            try:
                program = cal.reparameterize_route_progress(
                    route,
                    n_frames=cal.N_FRAMES,
                    event_frame=cycle["apex_frame"],
                    event_root_progress_m=s_event,
                    timing_bounds=timing,
                )
            except (TypeError, ValueError):
                continue
            out.append((placement, program, cycle))
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/calibrate_ramp_route_phase")
    args = parser.parse_args(argv)
    rows = _load_rows(Path(args.out))
    route = np.asarray([[0.0, 0.0], [0.0, cal.PILOT_ROUTE_LENGTH_M]], dtype=float)
    broad = cal.broad_timing_bounds()

    print("== 1. neutral-substrate attrition ==")
    zero = [(r["split"], r["seed"], r["speed_label"]) for r in rows if r["n_cycles"] == 0]
    print(f"zero-cycle clips: {len(zero)}/72 ({len(zero) / 72:.0%})")
    per_seed = defaultdict(list)
    for split, seed, speed in zero:
        per_seed[(split, seed)].append(speed)
    washouts = sorted(k for k, v in per_seed.items() if len(v) == 3)
    print(f"all-speed washout seeds: {washouts}")
    for speed in ("slow", "reference", "fast"):
        n = sum(1 for r in rows if r["speed_label"] == speed and r["n_cycles"] == 0)
        print(f"  {speed}: {n}/24 zero-cycle clips")

    print("\n== 2. informative Pmin over seeds with background evidence ==")
    seed_bg: dict[int, float] = {}
    for row in rows:
        if row["split"] != "calibration":
            continue
        seen = {}
        for cycle in row["cycles"]:
            for ident, contrast in zip(
                cycle["background_window_identities"], cycle["background_contrasts_m"]
            ):
                seen[json.dumps(ident, sort_keys=True)] = contrast
        if seen:
            seed_bg[row["seed"]] = max(seed_bg.get(row["seed"], 0.0), max(seen.values()))
    print(f"calibration seeds with background evidence: {len(seed_bg)}/16")
    bound = cal.calibrated_upper_bound(
        list(seed_bg.values()), quantum=cal.PROMINENCE_QUANTUM_M
    )
    pmin = bound["value"]
    print(
        f"informative Pmin = {pmin} m "
        f"(nearest-rank Q95 {bound['nearest_rank_value']:.4f} m, x1.25, ceil 1mm)"
    )

    print("\n== 3. timing-envelope washout at the fixed 3.6 m event ==")
    n_valid = n_fixed = n_set = 0
    for row in rows:
        for cycle in row["cycles"]:
            if not (cycle["prominence_m"] >= pmin and cycle["packet_window_valid"]):
                continue
            n_valid += 1
            fixed_ok = False
            set_ok = False
            for placement in CANDIDATE_PLACEMENTS_M:
                s_event = placement - cycle["nominal_foot_forward_offset_m"]
                try:
                    cal.reparameterize_route_progress(
                        route,
                        n_frames=cal.N_FRAMES,
                        event_frame=cycle["apex_frame"],
                        event_root_progress_m=s_event,
                        timing_bounds=broad,
                    )
                except (TypeError, ValueError):
                    continue
                set_ok = True
                if placement == V1_FIXED_OBSTACLE_M:
                    fixed_ok = True
            n_fixed += fixed_ok
            n_set += set_ok
    print(
        f"Pmin-passing packet-valid cycles {n_valid}; warpable to fixed 3.6 m: "
        f"{n_fixed} ({n_fixed / n_valid:.0%}); warpable to any of "
        f"{CANDIDATE_PLACEMENTS_M}: {n_set} ({n_set / n_valid:.0%})"
    )

    print("\n== 4. candidate v2 reference-gated replay ==")
    ref = {(r["split"], r["seed"]): r for r in rows if r["speed_label"] == "reference"}
    per_seed_max: dict[int, dict[str, float]] = {}
    for seed in V1_CALIBRATION_SEEDS:
        agg = {"acc": 0.0, "jerk": 0.0, "end": 0.0}
        contributed = False
        for speed in ("slow", "reference", "fast"):
            row = next(
                r
                for r in rows
                if r["split"] == "calibration"
                and r["seed"] == seed
                and r["speed_label"] == speed
            )
            options = _placement_candidates(row, broad, pmin, CANDIDATE_PLACEMENTS_M, route)
            by_side = defaultdict(list)
            for option in options:
                by_side[option[2]["swing_side"]].append(option)
            for side_options in by_side.values():
                best = min(
                    side_options,
                    key=lambda o: (
                        *o[1].selection_cost,
                        abs(o[0] - V1_FIXED_OBSTACLE_M),
                        o[0],
                        o[2]["evidence_digest"],
                    ),
                )
                program = best[1]
                agg["acc"] = max(
                    agg["acc"],
                    program.max_abs_continuous_route_progress_acceleration_mps2,
                    program.max_abs_discrete_route_progress_acceleration_mps2,
                )
                agg["jerk"] = max(
                    agg["jerk"], float(program.max_abs_discrete_route_progress_jerk_mps3)
                )
                agg["end"] = max(
                    agg["end"],
                    float(program.endpoint_route_progress_speed_deviation_mps),
                )
                contributed = True
        if contributed:
            per_seed_max[seed] = agg
    print(f"all-strata timing-contributing calibration seeds: {len(per_seed_max)}/16")
    acc = cal.calibrated_upper_bound(
        [v["acc"] for v in per_seed_max.values()],
        quantum=cal.ACCELERATION_QUANTUM_MPS2,
        positive_floor=True,
    )
    jerk = cal.calibrated_upper_bound(
        [v["jerk"] for v in per_seed_max.values()],
        quantum=cal.JERK_QUANTUM_MPS3,
        positive_floor=True,
    )
    end = cal.calibrated_upper_bound(
        [v["end"] for v in per_seed_max.values()],
        quantum=cal.ENDPOINT_SPEED_QUANTUM_MPS,
    )
    print(
        f"caps: acc {acc['value']} m/s^2, jerk {jerk['value']} m/s^3, "
        f"endpoint {end['value']} m/s"
    )
    frozen = cal.RouteTimingBounds(
        fps=cal.FPS,
        min_discrete_route_progress_speed_mps=cal.MIN_ROUTE_SPEED_MPS,
        max_discrete_route_progress_speed_mps=cal.MAX_ROUTE_SPEED_MPS,
        max_abs_route_progress_acceleration_mps2=acc["value"],
        max_abs_discrete_route_progress_jerk_mps3=jerk["value"],
        reference_route_progress_speed_mps=cal.REFERENCE_SPEED_MPS,
        max_endpoint_route_progress_speed_deviation_mps=end["value"],
    )

    validation = [ref[("validation", seed)] for seed in V1_VALIDATION_SEEDS]
    attrited = [r["seed"] for r in validation if r["n_cycles"] == 0]
    non_attrited = [r for r in validation if r["n_cycles"] > 0]
    print(f"validation reference attrition {len(attrited)}/8 (candidate gate <=3)")
    full = [
        r["seed"]
        for r in non_attrited
        if _placement_candidates(r, frozen, pmin, CANDIDATE_PLACEMENTS_M, route)
    ]
    need = max(4, math.ceil(0.6 * len(non_attrited)))
    print(
        f"full-frozen feasible {len(full)}/{len(non_attrited)} (candidate gate >={need})"
    )
    usage = Counter()
    for r in non_attrited:
        options = _placement_candidates(r, frozen, pmin, CANDIDATE_PLACEMENTS_M, route)
        by_side = defaultdict(list)
        for option in options:
            by_side[option[2]["swing_side"]].append(option)
        for side_options in by_side.values():
            best = min(
                side_options,
                key=lambda o: (
                    *o[1].selection_cost,
                    abs(o[0] - V1_FIXED_OBSTACLE_M),
                    o[0],
                    o[2]["evidence_digest"],
                ),
            )
            usage[best[0]] += 1
    print(f"validation selected-placement usage: {dict(sorted(usage.items()))}")


if __name__ == "__main__":
    main()
