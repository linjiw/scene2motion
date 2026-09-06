"""Build the public progress section from compact, hash-linked research receipts.

No model, simulator or motion payload is loaded. Historical pages retain their own
definitions; only the marked current-progress block is regenerated.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BEGIN = "<!-- BEGIN RECEIPT-DERIVED PROGRESS -->"
END = "<!-- END RECEIPT-DERIVED PROGRESS -->"
SOURCES = {
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
    return {"schema_version": "scene2motion-public-progress-v1", "updated": "2026-09-05",
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
