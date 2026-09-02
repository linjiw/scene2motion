"""Fig. 4 — channel addressability and response ("requested vs achieved" per native channel),
regenerated from committed campaign outputs only.

Panels (v2 sampler, one route, one scene, kinematic unless stated):
  (a) text prompt: best whole-body-clearable box height under WALK (exp019 v7 nominal arm) and
      STEP (exp020 text-only arm; exp021), with the 3 cm elicitation line and the exp021 clips
      that exactly clear a 5 cm box at x = 1.2 m ringed;
  (b) position channel (EXP-1C): requested foot lift vs realized swing-foot peak, lift arm
      against its matched control, with the over-response, bilateral-flight and SONIC counts;
  (c) rotation packet (exp019 v7): per-seed compliance of the packet's rotation payload and of
      its root-height payload (1 = command met, 0 = stayed at the nominal, < 0 = moved away);
  (d) rotation packet placement: requested obstacle position vs realized lift position with the
      fitted gain, and the lag-sweep verdict from response_lag.json.
A root-height dose-response panel is NOT drawn by default: the only committed artifact
(``outputs/duck_response/response.json``, fit on exp001d + step_calib) is v1-sampler evidence,
quarantined by house rule 6; ``--v1-root-height`` appends it, labelled as such.  Run with any
interpreter that has numpy + matplotlib, e.g.
``/home/linjiw/isaaclab-install/env_isaaclab/bin/python experiments/fig_channel_snr.py``.
Every number drawn is written to ``fig3_channel_response_numbers.json`` with input paths + sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/figures"
STEM = "fig3_channel_response"

SRC = {
    "exp021_rows": "outputs/exp021_elicited_lift_distribution_v2/rows.jsonl",
    "exp021_receipt": "outputs/exp021_elicited_lift_distribution_v2/receipt.json",
    "contract_rows": "outputs/analysis_trackability_contract/rows.jsonl",
    "exp020_rows": "outputs/exp020_text_only_control/rows.jsonl",
    "exp020_receipt": "outputs/exp020_text_only_control/receipt.json",
    "exp019_placement": "outputs/exp019_gait_matched_stepover_v7/placement_analysis.json",
    "exp019_constraint_residual": "outputs/exp019_gait_matched_stepover_v7/constraint_residual.json",
    "exp019_response_lag": "outputs/exp019_gait_matched_stepover_v7/response_lag.json",
    "exp1c_rows": "outputs/exp1c_stepover/rows.jsonl",
    "exp1c_receipt": "outputs/exp1c_stepover/receipt.json",
}
SRC_V1 = {"duck_response_v1": "outputs/duck_response/response.json"}

# Validated two-slot categorical palette (dataviz reference instance; validator: ALL PASS),
# shared with fig_contract_gate.py / fig_cost_curve.py.
C_TERM = "#eb6834"
C_SURV = "#2a78d6"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e6e5e1"
C_NA = "#b9b7b2"
LIFT_MIN_M = 0.03
BOX_M = 0.05
PLACE_TOL_M = 0.25
RNG = np.random.default_rng(20260902)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=C_INK2, labelsize=6.5, width=0.6, length=3)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def q(a, p):
    return float(np.percentile(np.asarray(a, float), p))


def panel_text(ax, paths, numbers):
    e21 = rows_of(paths["exp021_rows"])
    e21_receipt = json.loads(paths["exp021_receipt"].read_text())
    c21 = {r["seed"]: r for r in rows_of(paths["contract_rows"]) if r["family"] == "exp021_step"}
    e20 = rows_of(paths["exp020_rows"])
    e20_receipt = json.loads(paths["exp020_receipt"].read_text())
    pa = json.loads(paths["exp019_placement"].read_text())
    walk = [r["lift_height_m"] for r in pa["rows"] if r["arm"] == "nominal"]
    step8 = [r["lift_height_m"] for r in e20 if r["arm"] == "text_only"]
    step64 = [r["lift_height_m"] for r in e21]
    ring64 = [bool(c21[r["seed"]]["exact_clear_5cm_x1.2"]) for r in e21]
    assert len(walk) == 8 and len(step8) == 8 and len(step64) == 64
    groups = [("WALK\nexp019 v7 nominal\n(8 seeds)", walk, C_INK2, None),
              ("STEP\nexp020 text-only\n(8 seeds)", step8, C_SURV, None),
              ("STEP\nexp021\n(64 seeds)", step64, C_SURV, ring64)]
    out = {}
    for x, (label, vals, colour, rings) in enumerate(groups):
        v = np.asarray(vals)
        jit = RNG.uniform(-0.2, 0.2, size=len(v))
        ax.scatter(x + jit, v, s=9, color=colour, alpha=0.75, linewidths=0, zorder=3)
        if rings is not None:
            m = np.asarray(rings)
            ax.scatter((x + jit)[m], v[m], s=28, facecolors="none", edgecolors=C_INK, linewidths=0.6, zorder=4)
        k3, k0 = int((v >= LIFT_MIN_M).sum()), int((v > 0).sum())
        txt = f"{k3}/{len(v)} ≥ 3 cm"
        if k0 != k3:
            txt += f"\n{k0}/{len(v)} > 0"
        ax.text(x, 0.475, txt, ha="center", va="top", fontsize=6.2, color=C_INK, linespacing=1.15)
        out[label.replace("\n", " ")] = {"n": len(v), "ge_3cm": k3, "gt_0": k0,
                                         "lift_heights_m": [float(a) for a in v],
                                         "median_m": float(np.median(v)), "max_m": float(v.max())}
        if rings is not None:
            out[label.replace("\n", " ")]["exact_clear_5cm_x1p2"] = int(m.sum())
    ax.axhline(LIFT_MIN_M, color=C_INK, linewidth=0.7, zorder=2)
    ax.axhline(BOX_M, color=C_INK2, linewidth=0.7, linestyle=(0, (3, 2)), zorder=2)
    ax.text(0.5, LIFT_MIN_M - 0.008, "3 cm: elicits", ha="center", va="top", fontsize=5.8, color=C_INK)
    ax.text(0.5, BOX_M + 0.008, "5 cm box", ha="center", va="bottom", fontsize=5.8, color=C_INK2)
    ax.scatter([], [], s=28, facecolors="none", edgecolors=C_INK, linewidths=0.6,
               label=f"exactly clears a 5 cm box\nat x = 1.2 m ({int(np.asarray(ring64).sum())}/64)")
    ax.legend(loc="center left", fontsize=5.8, frameon=False, handletextpad=0.3, borderaxespad=0.2,
              bbox_to_anchor=(0.0, 0.62))
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups], fontsize=6.0, linespacing=1.15)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.015, 0.5)
    ax.set_ylabel("best clearable box height (m)", fontsize=6.5)
    ax.set_title("(a) Text: WALK vs STEP", loc="left")
    style_axis(ax)
    out["cross_checks"] = {"exp021_receipt_n_clearing_0p03": e21_receipt["summary"]["n_clearing"]["0.03"],
                           "exp021_receipt_elicitation_rate_any_lift": e21_receipt["summary"]["elicitation_rate"],
                           "exp020_receipt_text_only": e20_receipt["summary"]["text_only"],
                           "exp020_receipt_nominal": e20_receipt["summary"]["nominal"]}
    assert out["cross_checks"]["exp021_receipt_n_clearing_0p03"] == out["STEP exp021 (64 seeds)"]["ge_3cm"]
    assert e20_receipt["summary"]["text_only"]["n_with_lift"] == out["STEP exp020 text-only (8 seeds)"]["gt_0"]
    assert e20_receipt["summary"]["nominal"]["n_with_lift"] == out["WALK exp019 v7 nominal (8 seeds)"]["gt_0"]
    numbers["panel_a_text"] = out


def panel_position(ax, paths, numbers):
    rows = rows_of(paths["exp1c_rows"])
    receipt = json.loads(paths["exp1c_receipt"].read_text())
    kin = [r for r in rows if r["row_type"] == "kin"]
    son = {(r["label"], r["seed"]): r for r in rows if r["row_type"] == "sonic_clip"}
    amps = sorted({r["lift_req_m"] for r in kin})
    per = {}
    for amp in amps:
        lift = [r for r in kin if r["arm"] == "lift" and r["lift_req_m"] == amp]
        ctrl = [r for r in kin if r["arm"] == "ctrl" and r["lift_req_m"] == amp]
        assert len(lift) == 24 and len(ctrl) == 24
        lp = np.array([r["realized_peak_m"] for r in lift]); cp = np.array([r["realized_peak_m"] for r in ctrl])
        d = np.array([r["d_peak_vs_ctrl_m"] for r in lift])
        rec = receipt["per_amplitude"][str(amp)]
        per[str(amp)] = {
            "lift_peak_median_iqr_m": [float(np.median(lp)), q(lp, 25), q(lp, 75)],
            "ctrl_peak_median_iqr_m": [float(np.median(cp)), q(cp, 25), q(cp, 75)],
            "lift_peaks_m": [float(a) for a in lp],
            "overshoot_recomputed": float(np.median(d) / amp),
            "overshoot_receipt": rec["overshoot_x"],
            "bilateral_flight_frac_lift_recomputed": float(np.mean([r["bilateral_flight_frac"] for r in lift])),
            "bilateral_flight_frac_lift_receipt": rec["bilateral_flight_frac_lift"],
            "sonic_lift_success_rows": int(sum(son[(r["label"], r["seed"])]["success"] for r in lift)),
            "sonic_lift_success_receipt_k": rec["sonic_lift"]["k"],
            "sonic_ctrl_success_receipt_k": rec["sonic_ctrl"]["k"],
        }
        assert abs(per[str(amp)]["overshoot_recomputed"] - rec["overshoot_x"]) < 1e-6
        assert per[str(amp)]["sonic_lift_success_rows"] == rec["sonic_lift"]["k"]
        jit = RNG.uniform(-0.006, 0.006, size=len(lp))
        ax.scatter(amp + jit, lp, s=6, color=C_SURV, alpha=0.35, linewidths=0, zorder=3)
    x = np.array(amps)
    lm = np.array([per[str(a)]["lift_peak_median_iqr_m"][0] for a in amps])
    cm = np.array([per[str(a)]["ctrl_peak_median_iqr_m"][0] for a in amps])
    ax.plot(x, cm + x, color=C_INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=4, label="control + request (unit gain)")
    ax.plot(x, cm, color=C_INK2, linewidth=1.1, marker="o", markersize=3, zorder=4, label="control arm, median")
    ax.plot(x, lm, color=C_SURV, linewidth=1.4, marker="o", markersize=3.2, zorder=5, label="lift arm, median")
    ov = [per[str(a)]["overshoot_receipt"] for a in amps]
    bf = [per[str(a)]["bilateral_flight_frac_lift_receipt"] for a in amps]
    sk = [per[str(a)]["sonic_lift_success_receipt_k"] for a in amps]
    ax.text(0.03, 0.97,
            f"over-response ×{min(ov):.1f}–{max(ov):.1f}\n(median Δpeak ÷ request)\n"
            f"bilateral flight {min(bf):.2f}–{max(bf):.2f}\nof gated frames\n"
            f"SONIC survived {sk[0]}/24 at {amps[0]:.2f} m\n→ {sk[-1]}/24 at {amps[-1]:.2f} m",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color=C_INK, linespacing=1.2)
    ax.legend(loc="upper right", fontsize=5.8, frameon=False, handlelength=1.6, borderaxespad=0.2)
    ax.set_xlim(0.02, 0.31)
    ax.set_xticks(amps)
    ax.set_xticklabels([f"{a:.2f}" for a in amps])
    ax.set_ylim(0, max(1.45, 1.25 * max(max(per[str(a)]["lift_peaks_m"]) for a in amps)))
    ax.set_xlabel("requested foot lift (m)", fontsize=6.5)
    ax.set_ylabel("realized swing-foot peak (m)", fontsize=6.5)
    ax.set_title("(b) Position channel (EXP-1C, 24 seeds/rung)", loc="left")
    style_axis(ax)
    numbers["panel_b_position"] = {"amplitudes_m": amps, "per_amplitude": per, "receipt_verdict": receipt["verdict"],
                                   "window_frames": receipt["window"], "gating": receipt["gating"]}


def panel_compliance(ax, paths, numbers):
    cr = json.loads(paths["exp019_constraint_residual"].read_text())
    cats = [("rotations", "absolute", 0.0, C_SURV), ("rotations", "residual", 1.0, C_SURV),
            ("root height", "absolute", 2.35, C_INK2), ("root height", "residual", 3.35, C_INK2)]
    ymin, ymax = -2.4, 1.3
    out = {}
    for chan, arm, x, colour in cats:
        rows = [r for r in cr["rows"] if r["arm"] == arm]
        assert len(rows) == 8
        key = "rotation_compliance" if chan == "rotations" else "root_height_compliance"
        vals = np.array([r[key] for r in rows])
        cmd = np.array([r["rotation_commanded_change_rad" if chan == "rotations" else "root_height_commanded_change_m"] for r in rows])
        err = np.array([r["rotation_realized_error_rad" if chan == "rotations" else "root_height_realized_error_m"] for r in rows])
        jit = RNG.uniform(-0.13, 0.13, size=len(vals))
        inside = vals >= ymin + 0.05
        ax.scatter((x + jit)[inside], vals[inside], s=11, color=colour, alpha=0.8, linewidths=0, zorder=3)
        for xj, v in zip((x + jit)[~inside], vals[~inside]):
            ax.scatter([xj], [ymin + 0.06], s=16, marker="v", color=colour, linewidths=0, zorder=3)
            ax.text(xj + 0.08, ymin + 0.06, f"{v:.2f}", fontsize=5.4, color=colour, va="center", ha="left")
        mean = float(vals.mean())
        ax.plot([x - 0.3, x + 0.3], [mean, mean], color=C_INK, linewidth=1.1, zorder=4)
        ax.text(x + 0.33, mean, f"{mean:+.2f}", fontsize=6.0, color=C_INK, va="center", ha="left", zorder=5)
        out[f"{chan} {arm}"] = {"compliance_per_seed": [float(a) for a in vals], "mean": mean,
                                "median": float(np.median(vals)), "seeds": [r["seed"] for r in rows],
                                "commanded_change_mean": float(cmd.mean()), "realized_error_mean": float(err.mean()),
                                "unit": "rad" if chan == "rotations" else "m"}
    ax.axhline(1.0, color=C_INK, linewidth=0.7, linestyle=(0, (3, 2)), zorder=2)
    ax.axhline(0.0, color=C_INK2, linewidth=0.7, zorder=2)
    ax.text(1.675, 1.0 + 0.05, "command met", ha="center", va="bottom", fontsize=5.8, color=C_INK)
    ax.text(1.675, 0.0 - 0.06, "stayed at nominal", ha="center", va="top", fontsize=5.8, color=C_INK2)
    ax.text(1.675, -0.95, "moved away\nfrom the command", ha="center", va="center", fontsize=5.8, color=C_INK2, linespacing=1.15)
    ax.text(0.98, 0.03,
            f"request: rotations {out['rotations absolute']['commanded_change_mean']:.2f}–{out['rotations residual']['commanded_change_mean']:.2f} rad;\n"
            f"root height {out['root height residual']['commanded_change_mean'] * 100:.1f}–{out['root height absolute']['commanded_change_mean'] * 100:.1f} cm (noise floor)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=C_INK2, linespacing=1.2)
    ax.set_xticks([c[2] for c in cats])
    ax.set_xticklabels([f"{c[0]}\n{c[1]}" for c in cats], fontsize=6.2, linespacing=1.15)
    ax.set_xlim(-0.5, 4.0)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("compliance (1 met · 0 nominal · < 0 away)", fontsize=6.5)
    ax.set_title("(c) Rotation packet: compliance", loc="left")
    style_axis(ax)
    numbers["panel_c_packet_compliance"] = out


def panel_placement(ax, paths, numbers):
    pa = json.loads(paths["exp019_placement"].read_text())
    rl = json.loads(paths["exp019_response_lag"].read_text())
    out = {}
    xs_all = []
    for arm, marker, face in (("absolute", "o", C_SURV), ("residual", "s", "none")):
        rows = [r for r in pa["rows"] if r["arm"] == arm]
        assert len(rows) == 8 and all(r["lift_x_m"] is not None for r in rows)
        x = np.array([r["obstacle_x_m"] for r in rows]); y = np.array([r["lift_x_m"] for r in rows])
        xs_all.extend(x.tolist())
        slope, intercept = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        err = np.array([r_["signed_placement_error_m"] for r_ in rows])
        ax.scatter(x, y, s=22, marker=marker, facecolors=face, edgecolors=C_SURV, linewidths=0.8, zorder=4,
                   label=f"{arm}: gain {slope:+.2f}")
        xx = np.linspace(x.min(), x.max(), 2)
        ax.plot(xx, slope * xx + intercept, color=C_SURV, linewidth=0.8, linestyle="-" if arm == "absolute" else (0, (2, 2)), zorder=3)
        out[arm] = {"obstacle_x_m": x.tolist(), "lift_x_m": y.tolist(), "gain_slope": float(slope),
                    "intercept_m": float(intercept), "pearson_r": r,
                    "signed_placement_error_m": err.tolist(),
                    "within_0p25m": int((np.abs(err) <= PLACE_TOL_M).sum()),
                    "mean_signed_error_m": float(err.mean()), "sd_signed_error_m": float(err.std(ddof=1))}
    lo, hi = 0.8, 5.6
    ax.plot([lo, hi], [lo, hi], color=C_INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    ax.text(4.85, 4.85 + 0.15, "perfect placement (gain 1)", ha="center", va="bottom", fontsize=5.8, color=C_INK,
            rotation=45, transform_rotates_text=True, rotation_mode="anchor")
    lag_rows = rl["rows"]
    beats = int(sum(bool(r["beats_nominal_at_any_lag"]) for r in lag_rows))
    best_lags = [int(r["best_lag_frames"]) for r in lag_rows]
    within = out["absolute"]["within_0p25m"] + out["residual"]["within_0p25m"]
    ax.text(0.03, 0.97,
            f"within {PLACE_TOL_M:.2f} m of the obstacle: {within}/16\n"
            f"lag sweep {min(rl['lags'])}…+{max(rl['lags'])} frames:\n{beats}/16 beat the nominal at any lag\n"
            f"(best lag median {int(np.median(best_lags)):+d} frame)",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color=C_INK, linespacing=1.2)
    ax.legend(loc="upper right", fontsize=5.8, frameon=False, handletextpad=0.3, borderaxespad=0.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 9.2)
    ax.set_xlabel("requested obstacle position along the route (m)", fontsize=6.5)
    ax.set_ylabel("realized lift position (m)", fontsize=6.5)
    ax.set_title("(d) Rotation packet: placement", loc="left")
    style_axis(ax)
    out["lag_sweep"] = {"lags_frames": [min(rl["lags"]), max(rl["lags"])], "n_clips": len(lag_rows),
                        "beats_nominal_at_any_lag": beats, "best_lag_frames": best_lags,
                        "min_error_over_nominal_gap": [float(min(r["lag_curve"].values()) / r["nominal_gap_rad"]) for r in lag_rows]}
    numbers["panel_d_packet_placement"] = out


def panel_v1_root_height(ax, path, numbers):
    d = json.loads(path.read_text())
    fit = d["fit"]["static_direct"]
    levels = sorted(fit["per_level"], key=float)
    x = np.array([float(l) for l in levels]); m = np.array([fit["per_level"][l]["mean_m"] for l in levels])
    s = np.array([fit["per_level"][l]["sd_m"] for l in levels])
    ax.errorbar(x, m, yerr=s, color=C_NA, ecolor=C_NA, linewidth=1.2, marker="o", markersize=3, capsize=2, zorder=3)
    ax.set_xlabel("requested duck level q (0–1)", fontsize=6.5)
    ax.set_ylabel("achieved overhead height (m)", fontsize=6.5)
    ax.set_title("(e) Root height — v1 sampler (audit only)", loc="left")
    ax.text(0.03, 0.05, f"n = {fit['n']} clips, mean ± sd\nquarantined pre-00d6e3e evidence", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=5.8, color=C_INK2)
    style_axis(ax)
    numbers["panel_e_root_height_v1"] = {"q_levels": x.tolist(), "mean_m": m.tolist(), "sd_m": s.tolist(),
                                         "n": fit["n"], "source_note": fit["source"], "pathway": fit["pathway"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--v1-root-height", action="store_true",
                    help="append the v1-sampler root-height dose-response panel (quarantined evidence; off by default)")
    args = ap.parse_args()
    repo, out = Path(args.repo), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: repo / v for k, v in SRC.items()}
    plt.rcParams.update({"font.size": 7.0, "font.family": "DejaVu Sans", "text.color": C_INK,
                         "axes.labelcolor": C_INK, "axes.titlesize": 7.2, "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    if args.v1_root_height:
        fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.9))
        axes = axes.ravel()
        axes[5].axis("off")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
        axes = axes.ravel()
    numbers: dict = {"figure": STEM, "inputs": {k: {"path": SRC[k], "sha256": sha256(paths[k])} for k in SRC},
                     "thresholds": {"lift_min_m": LIFT_MIN_M, "box_height_m": BOX_M, "placement_tolerance_m": PLACE_TOL_M}}
    panel_text(axes[0], paths, numbers)
    panel_position(axes[1], paths, numbers)
    panel_compliance(axes[2], paths, numbers)
    panel_placement(axes[3], paths, numbers)
    v1_path = repo / SRC_V1["duck_response_v1"]
    numbers["omitted_no_committed_artifact"] = {
        "root_height_dose_response_v2": {
            "reason": "no v2-sampler requested-vs-achieved root-height artifact exists under outputs/; the only dose-response "
                      "(outputs/duck_response/response.json: 8 levels x 6 seeds direct root_y hold + exp001d 220 rows) predates "
                      "commit 00d6e3e and is quarantined v1 evidence (house rule 6); outputs/exp011 (REPORT 4.7) carries only v1 "
                      "SONIC success rates per body mode, no requested-vs-achieved rows; the v2 root-height response that does "
                      "exist is the packet's 2.2-2.3 cm root-height payload compliance drawn in panel (c)",
            "v1_artifact": {"path": SRC_V1["duck_response_v1"], "sha256": sha256(v1_path) if v1_path.exists() else None},
            "drawn": bool(args.v1_root_height),
        }
    }
    if args.v1_root_height:
        panel_v1_root_height(axes[4], v1_path, numbers)
        numbers["inputs"]["duck_response_v1"] = numbers["omitted_no_committed_artifact"]["root_height_dose_response_v2"]["v1_artifact"]
    fig.suptitle("(Fig. 3) Channel addressability and response — requested vs achieved:\n"
                 "text, position and rotation-packet channels" + (" (+ v1 root height, audit only)" if args.v1_root_height else ""),
                 x=0.01, y=0.995, ha="left", va="top", fontsize=8, fontweight="bold", linespacing=1.25)
    fig.text(0.01, 0.005,
             "v2 sampler, one route, one scene, kinematic; (b) SONIC counts are EXP-1C tracker successes at physics seed 0.\n"
             "Root-height dose–response omitted: the only committed artifact (outputs/duck_response) is v1-sampler evidence (house rule 6);\n"
             "the packet's root-height payload compliance is in (c).",
             fontsize=5.6, color=C_INK2, ha="left", va="bottom", linespacing=1.35)
    fig.tight_layout(rect=(0, 0.06, 1, 1.0), w_pad=1.4, h_pad=1.4)
    fig.savefig(out / f"{STEM}.pdf")
    fig.savefig(out / f"{STEM}.png", dpi=220)
    (out / f"{STEM}_numbers.json").write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    a = numbers["panel_a_text"]; c = numbers["panel_c_packet_compliance"]; d = numbers["panel_d_packet_placement"]
    print("text:", {k: (v["ge_3cm"], v["gt_0"], v["n"]) for k, v in a.items() if k != "cross_checks"})
    print("compliance means:", {k: round(v["mean"], 3) for k, v in c.items()})
    print("placement gains:", {k: round(v["gain_slope"], 3) for k, v in d.items() if k != "lag_sweep"}, "lag:", d["lag_sweep"]["beats_nominal_at_any_lag"])


if __name__ == "__main__":
    main()
