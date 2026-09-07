"""Build the public progress section from compact, hash-linked research receipts.

No model, simulator or motion payload is loaded. Historical pages retain their own
definitions; only the marked current-progress block is regenerated.
"""

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BEGIN = "<!-- BEGIN RECEIPT-DERIVED PROGRESS -->"
END = "<!-- END RECEIPT-DERIVED PROGRESS -->"
SOURCES = {
    "a10": "outputs/astra_a10_analysis_v1/summary.json",
    "a12": "outputs/astra_a12_analysis_v1/summary.json",
    "a13": "outputs/astra_a13_batch_observer_v1/summary.json",
    "a14": "outputs/astra_a14_canonical_context_v2/summary.json",
    "a15": "outputs/astra_a15_controlled_correction_v1/summary.json",
    "a16": "outputs/astra_a16_depth_phase_v1/summary.json",
    "a17": "outputs/astra_a17_remaining_development_v1/summary.json",
    "a17_identity": "outputs/astra_a17_remaining_development_v1/identity.json",
    "a17_audit": "outputs/astra_a17_completion_v1/summary.json",
    "recent_figure": "outputs/astra_a17_figures_v1/receipt.json",
    "a5": "outputs/astra_a5_encoder_d0/summary.json",
    "a4": "outputs/astra_a4_adaptive_d0/summary.json",
    "recovery": "outputs/astra_a5_recovery_audit_v1/summary.json",
    "bank": "outputs/astra_a5_bank_selection_v1/summary.json",
    "ceiling": "outputs/astra_a5_recovery_ceiling_v1/summary.json",
    "sensitivity": "outputs/astra_a5_progress_sensitivity_v1/summary.json",
    "contact": "outputs/astra_contact_replay_v2/summary.json",
    "contact_validation": "outputs/astra_contact_validation_v2/summary.json",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root=ROOT):
    data = {name: json.loads((root / path).read_text()) for name, path in SOURCES.items()}
    a5, recovery = data["a5"], data["recovery"]
    if not a5["complete"] or len(a5["rows"]) != a5["verified_executions"] or a5["verified_executions"] != a5["planned_executions"]:
        raise ValueError("A5 must be complete and all-assigned")
    a5_hash = sha(root / SOURCES["a5"])
    if any(data[name]["source_summary_sha256"] != a5_hash for name in ("recovery", "bank")):
        raise ValueError("audit source mismatch")
    if data["sensitivity"]["input_sha256"] != sha(root / SOURCES["recovery"]):
        raise ValueError("sensitivity source mismatch")
    for name in ("recovery", "bank"):
        matching = [h for p, h in data["ceiling"]["input_hashes"].items() if p.endswith("/" + SOURCES[name])]
        if matching != [sha(root / SOURCES[name])]:
            raise ValueError("ceiling source mismatch")
    # Guard this publication's wording: new contact/quality evidence requires a revision.
    if a5["physical_contact"] is not None or a5["high_quality_demonstrations"] is not None:
        raise ValueError("revise public claim boundary before publishing new evidence tiers")
    cells = []
    labels = {"free_nominal": "Free WALK", "fixed2": "2-second duck", "fixed3": "3-second duck"}
    for arm in labels:
        for mode in ("default", "g1", "teleop"):
            rows = [r for r in a5["rows"] if r["arm"] == arm and r["mode"] == mode and r["obstacle_present"]]
            expected = {(f"a5u{u}", i) for u in range(8) for i in range(4)}
            if len(rows) != 32 or {(r["unit_id"], r["slot_repeat"]) for r in rows} != expected:
                raise ValueError("duplicate or missing A5 assignments")
            passed = sum(r["operational_local_pass"] for r in rows)
            counts = a5["counts"][f"{arm}_{mode}_present"]
            rec = recovery["cells"][f"{arm}/{mode}"]
            if passed != counts["slot_rate_descriptive_only"]["successes"] or passed != rec["local_pass"]:
                raise ValueError("row/count/recovery disagreement")
            cells.append({"arm": arm, "label": labels[arm], "mode": mode,
                "assigned": len(rows), "local_passes": passed,
                "units_passing_3_of_4": counts["unit_repeat_success"]["successes"],
                "unit_wilson95": counts["unit_repeat_success"]["wilson95"],
                "full_route": counts["full_route_completed"],
                "local_and_recovery": rec["local_pass_kinematic_recovery_observed"]})
    if ([c["local_passes"] for c in cells] != [0, 0, 0, 14, 15, 13, 17, 17, 15]
            or any(c["full_route"] for c in cells)
            or data["contact_validation"]["G1_controlled_positive_contact_probe_completed"]):
        raise ValueError("new outcomes require revising the dated narrative")
    recent = recent_snapshot(data, root)
    return {"schema_version": "scene2motion-public-progress-v2", "updated": "2026-09-06",
        "recent": recent,
        "source_receipts": {p: sha(root / p) for p in SOURCES.values()},
        "scope": "simulation development evidence; no hardware, geometry holdout or high-quality dataset claim",
        "references": a5["actual_references"], "executions": a5["verified_executions"],
        "motion_groups": a5["n_motion_seed_groups"], "geometries": a5["n_target_geometries"],
        "physics_seeds": a5["physics_seeds"], "cells": cells,
        "paired_fixed3": {k: v for k, v in a5["paired_present"].items() if k.startswith("fixed3/")},
        "a4": {"counts": data["a4"]["counts"], "paired": data["a4"]["paired_present"]},
        "bank": {k: data["bank"][k] for k in ("selected_passes", "fixed3_passes", "reused_absent_execution_cost", "reused_candidate_references")},
        "ceiling": {k: data["ceiling"][k] for k in ("selected_passes", "fixed3_passes", "unit_fixed_hold_oracle_passes", "slotwise_oracle_passes")},
        "sensitivity": [{k: v for k, v in r.items() if k != "by_unit"} for r in data["sensitivity"]["curve"]],
        "contact": {"operational_replays": data["contact"]["operational_executions"],
            "batches": data["contact"]["batches"],
            "positive_g1_probe_completed": data["contact_validation"]["G1_controlled_positive_contact_probe_completed"],
            "simple_body_controls_passed": data["contact_validation"]["corrected_simple_body_probes_verified"]},
        "resource": {"simulator_wall_s": a5["sonic_wall_s"],
            "peak_child_rss_mib": max(r["child_peak_rss_mib"] for r in a5["costs"])},
        "physical_contact": None, "high_quality_motion_qualified": None}


def recent_snapshot(data, root):
    a15, a16, a17, audit = [data[k] for k in ("a15", "a16", "a17", "a17_audit")]
    for name in ("a15", "a16", "a17"):
        if any(data[name][k] for k in ("new_generations", "new_independent_units", "present_evaluations", "held_out_evaluations")):
            raise ValueError("recent narrative requires unchanged development-only tier")
    for name in ("a15", "a16"):
        if data[name]["assigned_units"] != 12:
            raise ValueError("pilot denominator changed")
    if (a15["promotion_gate"]["pass"] or not a16["development_lead_gate"]["pass"]
            or a17["development_lead_gate"]["pass"] or a17["full_cohort_executed"]
            or a17["arms"] is not None or a17["new_jobs"] != 2
            or any(r["achieved"]["development_geometry_progress_pass"] for r in a17["discriminating_record"].values())
            or a17["unexecuted_planned_jobs"] != 35
            or a17["development_lead_gate"]["selected_net_gain_over36"] != 0
            or not audit["pass"] or audit["independent_nominal_measurements"] != 36
            or audit["new_operational_executions"] != a17["new_operational_executions"]):
        raise ValueError("A17 dated failed-gate narrative requires review")
    expected = (2, 4, 4, 5)
    counts = tuple(a16["arms"][a]["selected_passes"] for a in ("baseline", "fixed_shallow", "correction", "no_phase"))
    if counts != expected or any(a16["arms"][a] != a15["arms"][a] for a in ("baseline", "fixed_shallow", "correction")):
        raise ValueError("pilot comparator counts changed")
    # Compact publications link audits to their exact result; raw archives remain local.
    for name, source in (("a17_audit", "a17"), ("a17", "a17_identity"), ("recent_figure", "a17")):
        links = [h for p, h in data[name]["source_hashes"].items() if p.endswith("/"+SOURCES[source])]
        if links != [sha(root/SOURCES[source])]:
            raise ValueError("recent result/audit source mismatch")
    figure = root/"docs/figures/astra_a17_clearance_progress.png"
    if sha(figure) != data["recent_figure"]["artifact_hashes"]["clearance_progress.png"]:
        raise ValueError("recent figure changed")
    return {"a10": {"nominal_raw": data["a10"]["supply"]["half"]["per_variant"]["baseline"]["development_passes"],
            "shallow_raw": data["a10"]["supply"]["half"]["per_variant"]["depth_minus"]["development_passes"], "assigned": 48},
        "a12": {"arms": data["a12"]["arms"], "gate": data["a12"]["expansion_gate"]},
        "a13": {"exact_replays": data["a13"]["observer_exact_executions"]},
        "a14": {"exact_pairs": data["a14"]["exact_replay_pairs"], "pass": data["a14"]["pass"]},
        "a15": {"arms": a15["arms"], "jobs": a15["unique_jobs"], "rollouts": a15["operational_executions"]},
        "a16": {"arms": a16["arms"], "rollouts": a16["new_operational_executions"], "paired": a16["comparisons"]},
        "a17": {k: a17[k] for k in ("status", "assigned_units", "full_cohort_executed", "arms",
            "new_jobs", "new_operational_executions", "new_unique_scientific_previews",
            "new_context_workload_executions", "development_lead_gate", "unexecuted_planned_jobs")},
        "a17_pair": {a: {k: r["achieved"][k] for k in ("inflated_clearance_m", "continuation_m", "exit_s",
            "development_geometry_progress_pass")} for a, r in a17["discriminating_record"].items()},
        "a17_resource": {k: audit[k] for k in ("simulator_elapsed_sum_s", "peak_child_rss_mib", "resource_refusals")},
        "figure_sha256": sha(figure)}


def render(data):
    rows = []
    for c in data["cells"]:
        rows.append(f'<tr><th scope="row">{c["label"]}</th><td>{c["mode"]}</td>'
            f'<td>{c["local_passes"]} / {c["assigned"]}</td><td>{c["units_passing_3_of_4"]} / {data["motion_groups"]}</td>'
            f'<td>{c["local_and_recovery"]} / {c["assigned"]}</td><td>{c["full_route"]} / {c["assigned"]}</td></tr>')
    curve = [f'<tr><th scope="row">{r["strict_net_displacement_gt_m"] * 100:g} cm</th>'
             f'<td>{r["fixed2"]} / 32</td><td>{r["fixed3"]} / 32</td><td>{r["slotwise_oracle"]} / 32</td></tr>'
             for r in data["sensitivity"]]
    provenance = [f'<li><a href="https://github.com/linjiw/scene2motion/blob/gh-pages/{p}">{p}</a>'
                  f' <code>sha256: {h[:12]}…</code></li>' for p, h in data["source_receipts"].items()]
    replacements = {"A5_TABLE": "\n".join(rows), "SENSITIVITY_TABLE": "\n".join(curve),
        "PROVENANCE": "\n".join(provenance), "REFERENCES": str(data["references"]),
        "EXECUTIONS": str(data["executions"]), "WALL_MIN": f'{data["resource"]["simulator_wall_s"] / 60:.2f}',
        "PEAK_GIB": f'{data["resource"]["peak_child_rss_mib"] / 1024:.2f}',
        "BANK_PASSES": str(data["bank"]["selected_passes"]),
        "BANK_COST": str(data["bank"]["reused_absent_execution_cost"]),
        "A4_FIXED": str(data["a4"]["counts"]["fixed3"]["present_local_pass"]),
        "A4_ADAPTIVE": str(data["a4"]["counts"]["adaptive"]["present_local_pass"])}
    recent = data["recent"]
    labels = {"baseline": "Nominal", "fixed_shallow": "Fixed shallow", "correction": "Full correction", "no_phase": "Depth only"}
    replacements["PILOT_TABLE"] = "\n".join(
        f'<tr><th scope="row">{labels[a]}</th><td>{r["selected_passes"]} / {r["assigned"]}</td>'
        f'<td>{100*r["selected_wilson_95"][0]:.1f}–{100*r["selected_wilson_95"][1]:.1f}%</td></tr>'
        for a in ("baseline", "fixed_shallow", "correction", "no_phase")
        for r in [recent["a16"]["arms"][a]])
    replacements["A17_PAIR"] = "\n".join(
        f'<tr><th scope="row">{labels[a]}</th><td>{r["inflated_clearance_m"]*1000:+.2f} mm</td>'
        f'<td>{r["continuation_m"]:.3f} m</td><td>{r["exit_s"]:.2f} s</td><td>Fail</td></tr>'
        for a, r in recent["a17_pair"].items())
    replacements["RECENT_FIGURE"] = "data:image/png;base64,"+base64.b64encode(
        (ROOT/"docs/figures/astra_a17_clearance_progress.png").read_bytes()).decode()
    replacements["A17_SECONDS"] = f'{recent["a17_resource"]["simulator_elapsed_sum_s"]:.2f}'
    replacements["A10_RAW"] = str(recent["a10"]["shallow_raw"])
    html = (ROOT / "docs/site/_progress.html").read_text()
    for key, value in replacements.items():
        html = html.replace("{{" + key + "}}", value)
    if "{{" in html:
        raise ValueError("unresolved progress placeholder")
    return html


def replace_block(text, block):
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("exactly one progress marker pair required")
    before, rest = text.split(BEGIN)
    old, after = rest.split(END)
    return before + BEGIN + "\n" + block.rstrip() + "\n" + END + after


def build():
    data = snapshot()
    html = render(data)
    # Prepare all edits before any write; a malformed page cannot leave a partial build.
    edits = {ROOT / p: replace_block((ROOT / p).read_text(), html)
             for p in ("docs/index.html", "docs/site/_body.html")}
    edits[ROOT / "docs/progress.json"] = json.dumps(data, indent=2, sort_keys=True) + "\n"
    for path, text in edits.items():
        path.write_text(text)
    print("Built receipt-derived progress in both page sources and docs/progress.json")


if __name__ == "__main__":
    build()
