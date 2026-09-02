"""Fig. 2 — the calibrated counting funnel (the plan's Table 1 as a figure), regenerated from
committed campaign outputs only.

For every native channel that was asked for a step-over, the number of clips that
(a) *elicit* a whole-body-clearable lift of >= 3 cm anywhere on the route,
(b) are *contact-consistent* under the calibrated gate (longest bilateral no-support run
    <= 0.20 s; the gate flags ``run > 0.20 s``; thresholds frozen in the exp016 receipt),
(c) *exactly clear a 5 cm box at the scene-fixed centre* of that campaign, and
(d) are *retained after SONIC* under EXP-022A's guarded endpoint,
each with a Wilson 95 % interval.  A stage that a campaign never measured is drawn as
"not measured", never as zero.  The dark inner bar / bracketed count is the number of clips
that also passed every earlier *measured* stage, so the funnel's nesting is visible: the
contact-consistent clips are the non-lifting ones, and every clip that clears the box fails
the gate.

Inputs (all committed):
  outputs/exp021_elicited_lift_distribution_v2/{rows.jsonl,receipt.json}   text prompt, frame 0
  outputs/analysis_trackability_contract/{rows.jsonl,receipt.json}          gate + exact clears
  outputs/exp022_exact_tracking_bridge/summary.json                          guarded retention
  outputs/exp1c_stepover/{rows.jsonl,receipt.json}                           position-channel lifts
  outputs/exp019_gait_matched_stepover_v7/{placement_analysis.json,arm_rows.jsonl}  rotation packets
  outputs/exp023_prompt_handoff/{rows.jsonl,receipt.json}                    delayed prompt + WALK
  outputs/exp023b_prompt_switch_control/{rows.jsonl,receipt.json}            delayed prompt + WALK
No model, MuJoCo or GPU dependency; run with any interpreter that has numpy + matplotlib, e.g.
``/home/linjiw/isaaclab-install/env_isaaclab/bin/python experiments/fig_channel_funnel.py``.
Every number drawn is also written to ``fig2_channel_funnel_numbers.json`` beside the figure,
with the path and sha256 of every input file.
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
STEM = "fig2_channel_funnel"

SRC = {
    "exp021_rows": "outputs/exp021_elicited_lift_distribution_v2/rows.jsonl",
    "exp021_receipt": "outputs/exp021_elicited_lift_distribution_v2/receipt.json",
    "contract_rows": "outputs/analysis_trackability_contract/rows.jsonl",
    "contract_receipt": "outputs/analysis_trackability_contract/receipt.json",
    "exp022_summary": "outputs/exp022_exact_tracking_bridge/summary.json",
    "exp1c_rows": "outputs/exp1c_stepover/rows.jsonl",
    "exp1c_receipt": "outputs/exp1c_stepover/receipt.json",
    "exp019_placement": "outputs/exp019_gait_matched_stepover_v7/placement_analysis.json",
    "exp019_arm_rows": "outputs/exp019_gait_matched_stepover_v7/arm_rows.jsonl",
    "exp023_rows": "outputs/exp023_prompt_handoff/rows.jsonl",
    "exp023_receipt": "outputs/exp023_prompt_handoff/receipt.json",
    "exp023b_rows": "outputs/exp023b_prompt_switch_control/rows.jsonl",
    "exp023b_receipt": "outputs/exp023b_prompt_switch_control/receipt.json",
}

# Validated two-slot categorical palette (dataviz reference instance; validator: ALL PASS),
# shared with fig_contract_gate.py / fig_cost_curve.py, plus one neutral grey for "not measured".
C_TERM = "#eb6834"   # SONIC outcome (retained = 0 marker)
C_SURV = "#2a78d6"   # kinematic counts
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e6e5e1"
C_NA = "#b9b7b2"     # stage not measured for that channel

LIFT_MIN_M = 0.03     # "elicits": whole-body-clearable lift >= 3 cm (exp023's event_min_clearance_m)
GATE_S = 0.20         # calibrated gate flags run > 0.20 s; contact-consistent = run <= 0.20 s
BOX_M = 0.05          # graded height used for the fixed-centre stage
BOX_KEY = "0.05"
STAGES = ("elicits", "contact", "clears", "retained")
STAGE_TITLES = {
    "elicits": "(a) elicits\nlift ≥ 3 cm anywhere",
    "contact": "(b) contact-consistent\nrun ≤ 0.20 s (gate)",
    "clears": "(c) clears 5 cm box\nat the fixed centre",
    "retained": "(d) retained, SONIC\nguarded endpoint",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def stage_entry(flags: list[bool] | None, n: int, definition: str, source: str,
                count_only: int | None = None, note: str | None = None) -> dict | None:
    """One funnel cell.  ``flags`` is the per-clip pass vector (None = not measured);
    ``count_only`` is used when a stage is reported by the receipt as a count with no
    per-clip vector (EXP-022A's guarded retention, which is zero)."""
    if flags is None and count_only is None:
        return {"measured": False, "n": n, "definition": definition, "source": source, "note": note}
    k = int(sum(flags)) if flags is not None else int(count_only)
    lo, hi = wilson(k, n)
    return {"measured": True, "k": k, "n": n, "rate": k / n, "wilson95": [lo, hi],
            "definition": definition, "source": source, "note": note}


def nest(stages: dict[str, dict | None], vectors: dict[str, list[bool] | None]) -> None:
    """Write ``nested`` = clips passing this stage and every earlier measured stage."""
    running: np.ndarray | None = None
    for name in STAGES:
        cell = stages[name]
        vec = vectors.get(name)
        if cell is None or not cell["measured"]:
            continue
        if vec is None:               # count-only stage (retained = 0): nested is bounded by k
            cell["nested"] = min(cell["k"], int(running.sum()) if running is not None else cell["k"])
            continue
        v = np.asarray(vec, bool)
        running = v if running is None else (running & v)
        cell["nested"] = int(running.sum())


def style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=C_INK2, labelsize=7, width=0.6, length=3)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def build_channels(paths: dict[str, Path]) -> tuple[list[dict], dict]:
    checks: dict = {}

    # ---- text prompt STEP from frame 0 (exp021, 64 clips) -------------------------------
    e21 = rows_of(paths["exp021_rows"])
    e21_receipt = json.loads(paths["exp021_receipt"].read_text())
    contract = rows_of(paths["contract_rows"])
    c21 = {r["seed"]: r for r in contract if r["family"] == "exp021_step"}
    assert len(e21) == 64 and len(c21) == 64 and all(r["seed"] in c21 for r in e21)
    for r in e21:
        assert abs(c21[r["seed"]]["lift_height_m"] - r["lift_height_m"]) < 1e-12
    seeds21 = [r["seed"] for r in e21]
    v21 = {
        "elicits": [r["lift_height_m"] >= LIFT_MIN_M for r in e21],
        "contact": [c21[s]["max_unsupported_run_s"] <= GATE_S for s in seeds21],
        "clears": [bool(c21[s]["exact_clear_5cm_x1.2"]) for s in seeds21],
    }
    exp022 = json.loads(paths["exp022_summary"].read_text())
    ret = exp022["paired_reference_to_achieved_retention"]["staged"][BOX_KEY]
    assert ret["n_paired"] == 64 and ret["reference_clear"] == sum(v21["clears"]), (ret, sum(v21["clears"]))
    checks["exp021"] = {
        "receipt_n_clearing_0p03": e21_receipt["summary"]["n_clearing"]["0.03"],
        "rows_lift_ge_0p03": int(sum(v21["elicits"])),
        "rows_lift_gt_0": int(sum(r["lift_height_m"] > 0 for r in e21)),
        "receipt_elicitation_rate_any_lift": e21_receipt["summary"]["elicitation_rate"],
        "exp022_reference_clear_0p05": ret["reference_clear"],
        "exp022_achieved_guarded_clear_0p05": ret["achieved_guarded_clear"],
        "exp022_endpoint_guard": ret["endpoint_guard"],
        "contract_terminated": int(sum(bool(c21[s]["terminated"]) for s in seeds21)),
    }
    assert checks["exp021"]["receipt_n_clearing_0p03"] == checks["exp021"]["rows_lift_ge_0p03"]
    st21 = {
        "elicits": stage_entry(v21["elicits"], 64, "exp021 rows: lift_height_m >= 0.03 (best whole-body-clearable box height anywhere on the route)", "exp021_rows"),
        "contact": stage_entry(v21["contact"], 64, "contract rows (family exp021_step): max_unsupported_run_s <= 0.20", "contract_rows"),
        "clears": stage_entry(v21["clears"], 64, "contract rows: exact_clear_5cm_x1.2 (BoxHeightProbe at x = 1.2 m, 0.20 m deep; centre chosen post hoc on these clips)", "contract_rows"),
        "retained": stage_entry(None, 64, "EXP-022A paired guarded retention at 5 cm: achieved-state replay non-terminated, clears the exact box, inside the lateral corridor, finishes beyond the obstacle", "exp022_summary",
                                count_only=ret["achieved_guarded_clear"],
                                note=f"reference_clear {ret['reference_clear']}, lost {ret['lost_reference_clear']}; {checks['exp021']['contract_terminated']}/64 evaluator-terminated"),
    }
    nest(st21, v21)
    ch_text = {"key": "text_step_frame0", "label": "text prompt STEP, frame 0\nexp021 · 64 seeds", "n": 64, "stages": st21}

    # ---- position-channel lifts (EXP-1C lift arms, 144 clips) ---------------------------
    e1c = rows_of(paths["exp1c_rows"])
    e1c_receipt = json.loads(paths["exp1c_receipt"].read_text())
    kin = {(r["label"], r["seed"]): r for r in e1c if r["row_type"] == "kin"}
    son = {(r["label"], r["seed"]): r for r in e1c if r["row_type"] == "sonic_clip"}
    c1c = [r for r in contract if r["family"] == "exp1c_lift"]
    assert len(c1c) == 144 and all((r["arm"], r["seed"]) in kin for r in c1c)
    keys1c = [(r["arm"], r["seed"]) for r in c1c]
    v1c = {
        "elicits": [kin[k]["clearance_at_peak_m"] >= LIFT_MIN_M for k in keys1c],
        "contact": [r["max_unsupported_run_s"] <= GATE_S for r in c1c],
        "clears": [kin[k]["clearance_at_fixed_m"] >= BOX_M for k in keys1c],
    }
    survived = [not r["terminated"] for r in c1c]
    assert all(son[k]["terminated"] == r["terminated"] for k, r in zip(keys1c, c1c))
    checks["exp1c"] = {
        "receipt_verdict": e1c_receipt["verdict"],
        "fixed_probe_x_m": e1c_receipt["fixed_probe_x_m"],
        "probe_geometry": e1c_receipt["probe_geometry"],
        "survived_not_terminated": int(sum(survived)),
        "receipt_sonic_lift_successes_sum": int(sum(a["sonic_lift"]["k"] for a in e1c_receipt["per_amplitude"].values())),
        "fixed_clear_and_survived": int(sum(c and s for c, s in zip(v1c["clears"], survived))),
        "per_amplitude": {amp: {"elicits": int(sum(v1c["elicits"][i] for i, k in enumerate(keys1c) if kin[k]["lift_req_m"] == float(amp))),
                                "contact": int(sum(v1c["contact"][i] for i, k in enumerate(keys1c) if kin[k]["lift_req_m"] == float(amp))),
                                "clears_fixed": int(sum(v1c["clears"][i] for i, k in enumerate(keys1c) if kin[k]["lift_req_m"] == float(amp))),
                                "survived": int(sum(survived[i] for i, k in enumerate(keys1c) if kin[k]["lift_req_m"] == float(amp)))}
                          for amp in e1c_receipt["per_amplitude"]},
    }
    assert checks["exp1c"]["survived_not_terminated"] == checks["exp1c"]["receipt_sonic_lift_successes_sum"]
    st1c = {
        "elicits": stage_entry(v1c["elicits"], 144, "exp1c kin rows (lift arms): clearance_at_peak_m >= 0.03 (whole-body clearance of the exp001c probe box at the clip's own swing peak)", "exp1c_rows"),
        "contact": stage_entry(v1c["contact"], 144, "contract rows (family exp1c_lift): max_unsupported_run_s <= 0.20", "contract_rows"),
        "clears": stage_entry(v1c["clears"], 144, f"exp1c kin rows: clearance_at_fixed_m >= 0.05 at the campaign's fixed probe x = {e1c_receipt['fixed_probe_x_m']} m ({e1c_receipt['probe_geometry']})", "exp1c_rows",
                              note="EXP-1C's own fixed-centre scorer, not BoxHeightProbe(1.2, 0.20)"),
        "retained": stage_entry(None, 144, "guarded endpoint never computed for EXP-1C (achieved states were not replayed against the fixed box)", "exp1c_receipt",
                                note=f"SONIC survival (not terminated) {sum(survived)}/144; {checks['exp1c']['fixed_clear_and_survived']} of the {sum(v1c['clears'])} fixed-clearing clips survived"),
    }
    nest(st1c, v1c)
    ch_pos = {"key": "position_lift_exp1c", "label": "position-channel lift\nEXP-1C lift arms · 144 clips", "n": 144, "stages": st1c}

    # ---- rotation packets (exp019 v7, 8 paired seeds, absolute / residual) --------------
    pa = json.loads(paths["exp019_placement"].read_text())
    arm_rows = rows_of(paths["exp019_arm_rows"])
    ch_packets = []
    checks["exp019"] = {}
    for arm in ("absolute", "residual"):
        prow = [r for r in pa["rows"] if r["arm"] == arm]
        arow = [r for r in arm_rows if r["arm"] == arm]
        assert len(prow) == 8 and len(arow) == 8
        vec = {
            "elicits": [r["lift_height_m"] >= LIFT_MIN_M for r in prow],
            "clears": [bool(r["graded_clearance"][BOX_KEY]) for r in prow],
        }
        checks["exp019"][arm] = {"lift_gt_0": int(sum(r["lift_height_m"] > 0 for r in prow)),
                                 "lift_ge_0p03": int(sum(vec["elicits"])),
                                 "lift_heights_m": [r["lift_height_m"] for r in prow],
                                 "graded_clear_0p05": int(sum(vec["clears"])),
                                 "arm_rows_collision_free_at_8cm_obstacle": int(sum(bool(r["obstacle_collision_free"]) for r in arow))}
        st = {
            "elicits": stage_entry(vec["elicits"], 8, "exp019 v7 placement_analysis rows: lift_height_m >= 0.03", "exp019_placement"),
            "contact": stage_entry(None, 8, "no-support run not computed for exp019 clips (not in the contract analysis)", "contract_receipt"),
            "clears": stage_entry(vec["clears"], 8, "exp019 v7 placement_analysis rows: graded_clearance['0.05'] at the predeclared per-seed obstacle (placed at the nominal's swing apex before generation)", "exp019_placement",
                                  note=f"arm_rows obstacle_collision_free at the scene's 8 cm obstacle: {checks['exp019'][arm]['arm_rows_collision_free_at_8cm_obstacle']}/8"),
            "retained": stage_entry(None, 8, "exp019 packet clips were never tracked", "exp019_arm_rows"),
        }
        nest(st, vec)
        ch_packets.append({"key": f"rotation_packet_{arm}", "label": f"rotation packet, {arm}\nexp019 v7 · 8 seeds", "n": 8, "stages": st})

    # ---- delayed prompt WALK->STEP at frame 52 and the free WALK control ----------------
    e23 = rows_of(paths["exp023_rows"]); e23b = rows_of(paths["exp023b_rows"])
    r23 = json.loads(paths["exp023_receipt"].read_text()); r23b = json.loads(paths["exp023b_receipt"].read_text())
    step52 = [r for r in e23 if r["arm"] == "step_52"] + [r for r in e23b if r["arm"] == "step_52"]
    walk = [r for r in e23 if r["arm"] == "all_walk"] + [r for r in e23b if r["arm"] == "all_walk"]
    assert len(step52) == 16 and len(walk) == 16
    vd = {"elicits": [bool(r["event"]["present"]) for r in step52],
          "clears": [bool(r["fixed_box"]["clears"][BOX_KEY]) for r in step52]}
    vw = {"elicits": [any(bool(w["present"]) for w in r["control_events"].values()) for r in walk],
          "clears": [bool(r["control_fixed_boxes"]["52"]["clears"][BOX_KEY]) for r in walk]}
    c23 = {(r["seed"], r["arm"]): r for r in contract if r["family"] == "exp023_untracked"}
    checks["exp023"] = {
        "receipt_step_52_present_of_8": r23["summary"]["event_rates_missing_retained"]["step_52"]["present"],
        "receipt_fixed_box_step_52_0p05": r23["summary"]["fixed_box_rates"]["step_52"][BOX_KEY]["clears"],
        "receipt_all_walk_seeds_with_any_event": r23["summary"]["all_walk_specificity"]["seeds_with_any_event"],
        "rows_step_52_present": int(sum(vd["elicits"][:8])),
        "rows_all_walk_any_window_event": int(sum(vw["elicits"][:8])),
        "event_definition": {k: step52[0]["event"][k] for k in ("event_min_clearance_m", "analysis_window_frames", "obstacle_depth_m")},
        "predicted_box_centre_m": step52[0]["fixed_box"]["obstacle_x_m"],
        "contract_gate_pass_exp023_half_only": {
            "step_52": int(sum(c23[(r["seed"], "step_52")]["max_unsupported_run_s"] <= GATE_S for r in step52[:8])),
            "all_walk": int(sum(c23[(r["seed"], "all_walk")]["max_unsupported_run_s"] <= GATE_S for r in walk[:8]))},
    }
    checks["exp023b"] = {
        "receipt_step_52_present_of_8": r23b["summary"]["step_event_rates_missing_retained"]["step_52"]["present"],
        "receipt_fixed_box_step_52_0p05": r23b["summary"]["fixed_box_rates"]["step_52"][BOX_KEY]["present"],
        "receipt_pooled_step_52": r23b["decision_rule"]["pooled_step_52_with_exp023"],
        "receipt_all_walk_seeds_with_any_event": r23b["summary"]["all_walk_step_specificity"]["seeds_with_any_event"],
        "rows_step_52_present": int(sum(vd["elicits"][8:])),
        "rows_all_walk_any_window_event": int(sum(vw["elicits"][8:])),
        "refusal_reason": r23b.get("refusal_reason"),
    }
    assert checks["exp023"]["receipt_step_52_present_of_8"] == checks["exp023"]["rows_step_52_present"]
    assert checks["exp023b"]["receipt_step_52_present_of_8"] == checks["exp023b"]["rows_step_52_present"]
    assert checks["exp023b"]["receipt_pooled_step_52"]["pooled_present_of_16"] == sum(vd["elicits"])
    xc = checks["exp023"]["predicted_box_centre_m"]
    std = {
        "elicits": stage_entry(vd["elicits"], 16, "exp023 + exp023b rows (arm step_52): event.present = whole-body clearance >= 0.03 m inside the 96-frame post-switch window", "exp023_rows+exp023b_rows"),
        "contact": stage_entry(None, 16, "no-support run computed only for the EXP-023 half (contract rows, family exp023_untracked); the EXP-023b half is unmeasured, so the pooled stage is not drawn", "contract_rows",
                               note=f"EXP-023 half only: {checks['exp023']['contract_gate_pass_exp023_half_only']['step_52']}/8 pass the gate (0/8 of them elicit)"),
        "clears": stage_entry(vd["clears"], 16, f"exp023 + exp023b rows: fixed_box.clears['0.05'] at the predicted centre x = {xc:.2f} m (onset frame 52 + frozen 34-frame latency at nominal speed)", "exp023_rows+exp023b_rows"),
        "retained": stage_entry(None, 16, "delayed-prompt clips were never tracked", "exp023_receipt"),
    }
    nest(std, vd)
    stw = {
        "elicits": stage_entry(vw["elicits"], 16, "exp023 + exp023b rows (arm all_walk): any control_events[window].present (matched windows 0/52/104 and 0/52)", "exp023_rows+exp023b_rows"),
        "contact": stage_entry(None, 16, "EXP-023 half only in the contract rows; pooled stage not drawn", "contract_rows",
                               note=f"EXP-023 half only: {checks['exp023']['contract_gate_pass_exp023_half_only']['all_walk']}/8 pass the gate"),
        "clears": stage_entry(vw["clears"], 16, f"exp023 + exp023b rows: control_fixed_boxes['52'].clears['0.05'] at x = {xc:.2f} m (the delayed prompt's predicted centre)", "exp023_rows+exp023b_rows"),
        "retained": stage_entry(None, 16, "WALK control clips were never tracked", "exp023_receipt"),
    }
    nest(stw, vw)
    ch_delayed = {"key": "delayed_prompt_walk_to_step_52", "label": "delayed prompt WALK→STEP @ frame 52\nEXP-023 + 023b · 16 seeds", "n": 16, "stages": std}
    ch_walk = {"key": "free_walk_control", "label": "free nominal WALK (floor)\nEXP-023 + 023b all_walk · 16 seeds", "n": 16, "stages": stw}

    return [ch_text, ch_pos, *ch_packets, ch_delayed, ch_walk], checks


def draw(channels: list[dict], out: Path) -> None:
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "text.color": C_INK,
                         "axes.labelcolor": C_INK, "axes.titlesize": 6.5, "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, len(STAGES), figsize=(7.0, 3.2), sharey=True,
                             gridspec_kw={"width_ratios": [1.22, 1, 1, 1]})
    ys = -np.arange(len(channels), dtype=float)
    for ax, stage in zip(axes, STAGES):
        for y, ch in zip(ys, channels):
            cell = ch["stages"][stage]
            if not cell["measured"]:
                ax.barh(y, 1.0, height=0.6, color=C_NA, alpha=0.22, linewidth=0, zorder=2)
                ax.text(0.5, y, "not measured", ha="center", va="center", fontsize=6.2,
                        color=C_INK2, style="italic", zorder=4)
                continue
            k, n, (lo, hi) = cell["k"], cell["n"], cell["wilson95"]
            nested = cell.get("nested", k)
            if stage == "retained":
                ax.scatter([k / n], [y], marker="x", s=34, color=C_TERM, linewidths=1.2, zorder=5)
            else:
                ax.barh(y, k / n, height=0.6, color=C_SURV, alpha=0.9, linewidth=0, zorder=3)
                if 0 < nested < k:
                    ax.barh(y, nested / n, height=0.24, color=C_INK, linewidth=0, zorder=4)
            ax.plot([lo, hi], [y, y], color=C_INK, linewidth=0.7, zorder=5)
            for xcap in (lo, hi):
                ax.plot([xcap, xcap], [y - 0.13, y + 0.13], color=C_INK, linewidth=0.7, zorder=5)
            label = f"{k}/{n}"
            if nested != k:
                label += f"  [{nested}]"
            ax.text(hi + 0.03, y, label, ha="left", va="center", fontsize=6.4, color=C_INK, zorder=6)
        ax.set_xlim(0, 1.45)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
        ax.set_title(STAGE_TITLES[stage], loc="left")
        ax.set_xlabel("fraction of clips", fontsize=6.8)
        style_axis(ax)
        ax.yaxis.grid(False)
        ax.spines["bottom"].set_bounds(0, 1.0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([ch["label"] for ch in channels], fontsize=6.3)
    axes[0].tick_params(axis="y", length=0)
    fig.suptitle("(Fig. 2) What each native channel delivers:\n"
                 "elicits → contact-consistent → clears 5 cm at the fixed centre → retained",
                 x=0.01, y=0.995, ha="left", va="top", fontsize=8, fontweight="bold", linespacing=1.25)
    fig.text(0.01, 0.005,
             "Bars: k/n with Wilson 95 % intervals; dark inner bar and [j]: clips that also passed every earlier measured stage;\n"
             "grey: stage not measured for that channel; ×: SONIC-retained under EXP-022A's guarded endpoint (physics seed 0, one scene, no box in Isaac).\n"
             "Fixed centre: x = 1.2 m BoxHeightProbe (exp021; chosen post hoc on the same clips) · EXP-1C's own probe at x = 3.6 m ·\n"
             "the predeclared per-seed obstacle (packets) · the predicted centre x = 3.11 m (delayed prompt and WALK). Kinematic stages, v2 sampler.",
             fontsize=5.6, color=C_INK2, ha="left", va="bottom", linespacing=1.35)
    fig.tight_layout(rect=(0, 0.115, 0.995, 1.0), w_pad=1.0)
    fig.savefig(out / f"{STEM}.pdf")
    fig.savefig(out / f"{STEM}.png", dpi=220)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(REPO), help="repository root holding outputs/")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    repo, out = Path(args.repo), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: repo / v for k, v in SRC.items()}
    channels, checks = build_channels(paths)
    draw(channels, out)
    numbers = {
        "figure": STEM,
        "inputs": {k: {"path": SRC[k], "sha256": sha256(paths[k])} for k in SRC},
        "thresholds": {"lift_min_m": LIFT_MIN_M, "gate_max_unsupported_run_s": GATE_S, "box_height_m": BOX_M,
                       "wilson_z": 1.959964,
                       "nested_definition": "clips passing this stage and every earlier measured stage of the same channel"},
        "channels": channels,
        "cross_checks_against_receipts": checks,
        "scope": "kinematic stages from archived v2 clips; SONIC stage = EXP-022A achieved-state replay, physics seed 0, one scene, obstacle absent from Isaac",
    }
    (out / f"{STEM}_numbers.json").write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    for ch in channels:
        cells = []
        for s in STAGES:
            c = ch["stages"][s]
            cells.append(f"{s}: {c['k']}/{c['n']} [{c.get('nested', c['k'])}]" if c["measured"] else f"{s}: not measured")
        print(ch["key"], " | ".join(cells))


if __name__ == "__main__":
    main()
