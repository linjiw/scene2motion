"""Zero-GPU analyses of the completed E1a ledger; these decide what E2 should be.

Reads only `outputs/exp019_gait_matched_stepover_v7/` (24 scored clips, 8 paired seeds)
and answers the five questions that separate "one open-loop correction suffices" from
"placement is search, not control":

A. the **signed** placement error and its spread, not the median magnitude;
B. a placement tolerance curve P(|error| < r), which sizes any best-of-N;
C. whether the error is quantized in strides -- if it is, placement lives on a discrete
   lattice and latent optimization cannot help while selection can;
D. a lead-side x lift-side cross-tab: when a lift came, did it come from the foot the
   packet asked for;
E. the amplitude profile, since a perfectly placed 0.07 m lift still clears nothing on a
   0.30 m box.

The packet-window constraint residual (the positive control -- did the decoder reproduce
the commanded frames at the window) needs the rendered specs and lives in
`analyze_e1a_constraint_residual.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import calibrate_ramp_route_phase as cal  # noqa: E402
from experiments import exp019_ramp_gait_matched_stepover as e19  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.stepover_eval import BoxHeightProbe, foot_kinematics_series  # noqa: E402


DEFAULT_OUT = "outputs/exp019_gait_matched_stepover_v7"
SCAN_MARGIN_M = 0.30
SCAN_POINTS = 400
TOLERANCE_RADII_M = (0.10, 0.25, 0.50, 1.00, 2.00)
GRADED_BOX_HEIGHTS_M = (0.03, 0.05, 0.08, 0.12, 0.20, 0.30)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; the 0/n upper bound is what bounds a hit rate."""
    if n <= 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def box_height_profile(
    qpos: np.ndarray, route_xz: np.ndarray, depth_m: float,
    *, n_points: int = SCAN_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Clearable whole-body box height as a function of route position."""
    route = np.asarray(route_xz, dtype=float)
    low = float(route[0, 1]) + SCAN_MARGIN_M
    high = float(route[-1, 1]) - SCAN_MARGIN_M
    xs = np.linspace(low, high, n_points)
    heights = np.asarray(
        [BoxHeightProbe(float(x), depth_m).probe(qpos) for x in xs], dtype=float)
    return xs, heights


def lift_location(xs: np.ndarray, heights: np.ndarray) -> dict[str, float | None]:
    """Where the clip's best liftable clearance sits, and how peaked it is."""
    if not np.any(heights > 0.0):
        return {"lift_x_m": None, "lift_height_m": 0.0, "n_lift_regions": 0,
                "lift_support_m": 0.0}
    index = int(np.argmax(heights))
    positive = heights > 0.0
    # Count contiguous positive runs: several means the clip lifts more than once.
    padded = np.r_[False, positive, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    step = float(xs[1] - xs[0]) if len(xs) > 1 else 0.0
    return {
        "lift_x_m": float(xs[index]),
        "lift_height_m": float(heights[index]),
        "n_lift_regions": int(len(starts)),
        "lift_support_m": float(np.sum(positive) * step),
    }


def stride_length_m(
    foot_kinematics, thresholds, side: str = "left"
) -> float | None:
    """Median distance between consecutive same-foot footfall clusters."""
    clearance = np.asarray(foot_kinematics[side]["bottom_clearance_m"], dtype=float)
    speed = np.asarray(foot_kinematics[side]["planar_speed_mps"], dtype=float)
    forward = np.asarray(
        foot_kinematics[side]["forward_representative_m"], dtype=float)
    supported = (
        (clearance <= thresholds.support_height_m)
        & (speed <= thresholds.support_speed_mps)
    )
    padded = np.r_[False, supported, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    positions = [float(np.median(forward[a:b])) for a, b in zip(starts, ends) if b > a]
    if len(positions) < 2:
        return None
    gaps = np.diff(np.asarray(positions, dtype=float))
    gaps = gaps[gaps > 1e-3]
    return float(np.median(gaps)) if gaps.size else None


def lift_side(
    qpos: np.ndarray, foot_kinematics, lift_x: float, depth_m: float
) -> str | None:
    """Which foot is high over the lift position when the clearance is achieved."""
    best_side, best_height = None, -np.inf
    for side in ("left", "right"):
        forward = np.asarray(
            foot_kinematics[side]["forward_representative_m"], dtype=float)
        clearance = np.asarray(
            foot_kinematics[side]["bottom_clearance_m"], dtype=float)
        inside = np.abs(forward - lift_x) <= depth_m / 2 + e19.BODY_MARGIN
        if not np.any(inside):
            continue
        height = float(np.max(clearance[inside]))
        if height > best_height:
            best_side, best_height = side, height
    return best_side


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    out = Path(args.out)
    receipt = json.loads((out / "receipt.json").read_text())
    if receipt.get("status") != "complete":
        raise SystemExit("this analysis reads a completed exp019 campaign")
    thresholds, _ = cal._load_physical_threshold_dependency(
        Path("outputs/exp016_threshold_calibration/receipt.json"))
    body = G1Body(None)
    pool = np.load(out / "pool_qpos.npz")
    arms = np.load(out / "arm_qpos.npz")
    per_seed = receipt["placement_selection"]["per_seed"]
    routes = {label: cal.route_xz_for_speed(speed) for label, speed in cal.SPEEDS}
    depth = e19.OBSTACLE_DEPTH_M

    rows: list[dict] = []
    for seed_s, info in sorted(per_seed.items()):
        seed = int(seed_s)
        label = info["speed_label"]
        route = routes[label]
        obstacle_x = float(info["obstacle_x_m"])
        nominal_qpos = pool[f"pilot__{label}__seed{seed}"]
        nominal_feet = foot_kinematics_series(body, nominal_qpos, cal.FPS)
        stride = stride_length_m(nominal_feet, thresholds)
        for arm in ("nominal", "absolute", "residual"):
            qpos = (nominal_qpos if arm == "nominal"
                    else np.asarray(arms[f"s{seed}__{arm}"], dtype=float))
            xs, heights = box_height_profile(qpos, route, depth)
            lift = lift_location(xs, heights)
            feet = (nominal_feet if arm == "nominal"
                    else foot_kinematics_series(body, qpos, cal.FPS))
            row = {
                "seed": seed, "arm": arm, "speed_label": label,
                "obstacle_x_m": obstacle_x,
                "height_at_obstacle_m": float(
                    BoxHeightProbe(obstacle_x, depth).probe(qpos)),
                "stride_length_m": stride,
                **lift,
            }
            if lift["lift_x_m"] is not None:
                signed = lift["lift_x_m"] - obstacle_x
                row["signed_placement_error_m"] = signed
                row["placement_error_strides"] = (
                    signed / stride if stride else None)
                row["lift_side"] = lift_side(
                    qpos, feet, lift["lift_x_m"], depth)
            row["graded_clearance"] = {
                f"{h:g}": bool(row["height_at_obstacle_m"] >= h)
                for h in GRADED_BOX_HEIGHTS_M
            }
            rows.append(row)

    def arm_rows(arm: str) -> list[dict]:
        return [row for row in rows if row["arm"] == arm]

    print("=" * 76)
    print("A. SIGNED placement error (packet arms)")
    print("=" * 76)
    print(f"{'seed':>6} {'arm':>9} {'obst_x':>7} {'lift_x':>7} {'signed':>8} "
          f"{'strides':>8} {'lift_h':>7} {'regions':>7} {'side':>6}")
    for row in rows:
        if row["arm"] == "nominal" or row.get("lift_x_m") is None:
            continue
        print(f"{row['seed']:>6} {row['arm']:>9} {row['obstacle_x_m']:>7.2f} "
              f"{row['lift_x_m']:>7.2f} {row['signed_placement_error_m']:>+8.2f} "
              f"{row['placement_error_strides']:>+8.2f} {row['lift_height_m']:>7.4f} "
              f"{row['n_lift_regions']:>7} {str(row.get('lift_side')):>6}")
    for arm in ("absolute", "residual"):
        signed = np.asarray(
            [r["signed_placement_error_m"] for r in arm_rows(arm)
             if r.get("lift_x_m") is not None], dtype=float)
        if not signed.size:
            continue
        print(f"\n  {arm}: n={signed.size} mean {signed.mean():+.2f} m, "
              f"median {np.median(signed):+.2f} m, sd {signed.std(ddof=1):.2f} m, "
              f"range [{signed.min():+.2f}, {signed.max():+.2f}], "
              f"{int((signed > 0).sum())} after / {int((signed < 0).sum())} before")
        print(f"    |error|: mean {np.abs(signed).mean():.2f} m, "
              f"median {np.median(np.abs(signed)):.2f} m")

    print()
    print("=" * 76)
    print("B. Placement tolerance curve (16 packet clips)")
    print("=" * 76)
    packet = [r for r in rows if r["arm"] != "nominal"]
    errors = np.asarray(
        [abs(r["signed_placement_error_m"]) for r in packet
         if r.get("lift_x_m") is not None], dtype=float)
    n_packet = len(packet)
    print(f"{'radius':>8} {'hits':>6} {'rate':>7}  Wilson95")
    for radius in TOLERANCE_RADII_M:
        hits = int((errors <= radius).sum())
        low, high = wilson_interval(hits, n_packet)
        print(f"{radius:>8.2f} {hits:>6} {hits / n_packet:>7.2f}  "
              f"[{low:.2f}, {high:.2f}]")
    hits_at_box = sum(1 for r in packet if r["height_at_obstacle_m"] > 0)
    low, high = wilson_interval(hits_at_box, n_packet)
    print(f"\n  any clearance AT the obstacle: {hits_at_box}/{n_packet}, "
          f"Wilson95 [{low:.2f}, {high:.2f}]  <- bounds any per-sample hit rate")
    if high > 0:
        print(f"  best-of-N for 90% at the upper bound: "
              f"N >= {math.ceil(math.log(0.10) / math.log(1 - high)):d}")

    print()
    print("=" * 76)
    print("C. Is the error quantized in strides?")
    print("=" * 76)
    strides = np.asarray(
        [r["placement_error_strides"] for r in packet
         if r.get("placement_error_strides") is not None], dtype=float)
    if strides.size:
        print(f"  stride lengths (nominal): "
              f"{np.median([r['stride_length_m'] for r in rows if r['stride_length_m']]):.3f} m median")
        print(f"  error in strides: " + " ".join(f"{v:+.2f}" for v in np.sort(strides)))
        residual = strides - np.round(strides)
        print(f"  distance to nearest integer stride: mean |frac| "
              f"{np.abs(residual).mean():.3f} (0.25 = uniform, 0 = lattice)")
        edges = np.arange(-4.5, 5.5, 1.0)
        counts, _ = np.histogram(strides, bins=edges)
        for centre, count in zip(edges[:-1] + 0.5, counts):
            if count:
                print(f"    {centre:+.0f} strides: {'#' * int(count)} ({count})")

    print()
    print("=" * 76)
    print("D. Lead/lift side (packet asked for the LEFT swing)")
    print("=" * 76)
    for arm in ("nominal", "absolute", "residual"):
        sides = [r.get("lift_side") for r in arm_rows(arm)
                 if r.get("lift_x_m") is not None]
        counts = {side: sides.count(side) for side in ("left", "right")}
        print(f"  {arm:>9}: lifts {len(sides)}/8, by side {counts}")

    print()
    print("=" * 76)
    print("E. Amplitude: graded box heights AT the obstacle, and best anywhere")
    print("=" * 76)
    print(f"{'arm':>9} " + " ".join(f"{h:>6.2f}" for h in GRADED_BOX_HEIGHTS_M)
          + f" {'best_h':>8} {'mean_h':>8}")
    for arm in ("nominal", "absolute", "residual"):
        selected = arm_rows(arm)
        cells = [
            sum(1 for r in selected if r["graded_clearance"][f"{h:g}"])
            for h in GRADED_BOX_HEIGHTS_M
        ]
        best = np.asarray([r["lift_height_m"] for r in selected], dtype=float)
        print(f"{arm:>9} " + " ".join(f"{c:>6d}" for c in cells)
              + f" {best.max():>8.4f} {best.mean():>8.4f}")
    print("\n  (cells are counts of 8 seeds clearing that box height at the obstacle)")

    payload = {
        "source": str(out),
        "n_seeds": len(per_seed),
        "graded_box_heights_m": list(GRADED_BOX_HEIGHTS_M),
        "tolerance_radii_m": list(TOLERANCE_RADII_M),
        "rows": rows,
    }
    destination = out / "placement_analysis.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
