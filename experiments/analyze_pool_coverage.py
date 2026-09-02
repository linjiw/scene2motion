"""Pool coverage vs selection for the EXP-021 stepping pool tracked in EXP-022A.

Answers, retrospectively and from committed rows only, the question "is the problem selection
or missing candidates?":

  * how many candidates clear the obstacle at the specified position (reference level),
  * how many complete tracking without the evaluator's cutoff,
  * how many do **both**, and how many satisfy the full traversal endpoint (passage through the
    lateral corridor, finish beyond the obstacle, collision-free at the graded height, no cutoff),
  * the exact coverage-vs-budget curve for a pool of this size, C(n-m, k)/C(n, k), i.e. the
    probability that a random sub-pool of k of these candidates contains at least one success.

An empty intersection means no selector over this pool could succeed, whatever its ranking
function: the limitation is the candidate pool, not the choice among candidates.

Scope: one route, one scene, one obstacle position (chosen post hoc, see the ledger), physics
seed 0, one rollout per reference, obstacle absent from the physics scene (achieved-state
replay).  Descriptive; no new samples are drawn.

Run:  $S2M_PY experiments/analyze_pool_coverage.py
Writes outputs/analysis_pool_coverage/summary.json and prints the table.
"""

from __future__ import annotations

import hashlib
import json
from math import comb, ceil, log
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "outputs/exp022_exact_tracking_bridge"
REFERENCE_ROWS = BRIDGE / "reference_rows.jsonl"
ACHIEVED_ROWS = BRIDGE / "achieved_rows.jsonl"
OUT = ROOT / "outputs/analysis_pool_coverage/summary.json"
HEIGHTS = ("0.03", "0.05", "0.08", "0.12", "0.2", "0.3")
COVERAGE_TARGETS = (0.5, 0.9, 0.95)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coverage_at_budget(n: int, m: int, k: int) -> float:
    """P(at least one success among k candidates drawn without replacement from the pool)."""
    if m <= 0 or k <= 0:
        return 0.0
    if k > n:
        raise ValueError("budget exceeds pool size")
    return 1.0 - (comb(n - m, k) / comb(n, k) if n - m >= k else 0.0)


def smallest_budget_for(n: int, m: int, target: float) -> int | None:
    """Budget for `target` coverage when SUB-SAMPLING this fixed pool (hypergeometric)."""
    for k in range(1, n + 1):
        if coverage_at_budget(n, m, k) >= target:
            return k
    return None


def smallest_fresh_draws_for(n: int, m: int, target: float) -> int | None:
    """Budget for `target` coverage when drawing FRESH seeds at the observed rate m/n.

    This is the decision-relevant number ("how many samples should I generate?") and is the
    convention behind the N90 = 12 quoted in the ledger for the 5 cm box; the sub-sampling
    number above is one smaller because drawing without replacement is more efficient.  The
    two are reported side by side so neither is mistaken for the other.
    """
    if m <= 0:
        return None
    p = m / n
    if p >= 1.0:
        return 1
    return int(ceil(log(1.0 - target) / log(1.0 - p)))


def analyse(label: str, reference: list[dict], achieved: list[dict]) -> dict:
    by_seed_ref = {r["seed"]: r for r in reference}
    by_seed_ach = {r["seed"]: r for r in achieved}
    seeds = sorted(set(by_seed_ref) & set(by_seed_ach))
    n = len(seeds)

    completes = {s for s in seeds if not by_seed_ach[s]["tracker_terminated"]}
    passed_and_finished = {
        s for s in seeds
        if by_seed_ach[s]["passed_obstacle"]
        and by_seed_ach[s]["passed_within_lateral_corridor"]
        and by_seed_ach[s]["finished_beyond_obstacle"]
    }
    heights: dict[str, dict] = {}
    for h in HEIGHTS:
        clears_ref = {s for s in seeds if by_seed_ref[s]["exact_clears"][h]}
        traversal = {s for s in seeds if by_seed_ach[s]["achieved_replay_clear_after_passing"][h]}
        both = clears_ref & completes
        heights[h] = {
            "reference_clears": sorted(clears_ref),
            "n_reference_clears": len(clears_ref),
            "n_reference_clears_and_completes_tracking": len(both),
            "n_traversal_endpoint": len(traversal),
            "coverage_curve_reference": {
                "n90_budget_subsampling_this_pool": smallest_budget_for(n, len(clears_ref), 0.9),
                "n90_budget_fresh_draws": smallest_fresh_draws_for(n, len(clears_ref), 0.9),
                "n50_budget_subsampling_this_pool": smallest_budget_for(n, len(clears_ref), 0.5),
                "coverage_at_full_pool": coverage_at_budget(n, len(clears_ref), n),
            },
            "coverage_curve_joint": {
                "n90_budget_subsampling_this_pool": smallest_budget_for(n, len(both), 0.9),
                "n90_budget_fresh_draws": smallest_fresh_draws_for(n, len(both), 0.9),
                "coverage_at_full_pool": coverage_at_budget(n, len(both), n),
            },
            "coverage_curve_traversal": {
                "n90_budget_subsampling_this_pool": smallest_budget_for(n, len(traversal), 0.9),
                "n90_budget_fresh_draws": smallest_fresh_draws_for(n, len(traversal), 0.9),
                "coverage_at_full_pool": coverage_at_budget(n, len(traversal), n),
            },
        }
    return {
        "obstacle_label": label,
        "obstacle_x_m": reference[0]["obstacle_x_m"],
        "n_candidates": n,
        "n_completes_tracking": len(completes),
        "n_terminated": n - len(completes),
        "n_passed_corridor_and_finished_beyond": len(passed_and_finished),
        "n_passed_and_finished_and_completed": len(passed_and_finished & completes),
        "n_never_reached_obstacle": n - len(passed_and_finished),
        "by_height": heights,
    }


def main() -> None:
    reference = _read(REFERENCE_ROWS)
    achieved = _read(ACHIEVED_ROWS)
    labels = sorted({r["obstacle_label"] for r in reference})
    results = [
        analyse(label,
                [r for r in reference if r["obstacle_label"] == label],
                [r for r in achieved if r["obstacle_label"] == label])
        for label in labels
    ]
    summary = {
        "analysis": "pool coverage vs selection for the tracked stepping pool",
        "question": "is the failure a selection problem or a missing-candidate problem?",
        "descriptive_only": True,
        "scope": ("one route, one scene, physics seed 0, one rollout per reference; the obstacle "
                  "is absent from the physics scene and achieved states are replayed against its "
                  "geometry; the staged position was chosen after inspecting the clips"),
        "sources": {
            "reference_rows": {"path": str(REFERENCE_ROWS.relative_to(ROOT)),
                               "sha256": _sha256(REFERENCE_ROWS)},
            "achieved_rows": {"path": str(ACHIEVED_ROWS.relative_to(ROOT)),
                              "sha256": _sha256(ACHIEVED_ROWS)},
        },
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")

    for entry in results:
        print(f"\n== obstacle '{entry['obstacle_label']}' at x = {entry['obstacle_x_m']} m, "
              f"n = {entry['n_candidates']} candidates")
        print(f"   completes tracking (no cutoff)          {entry['n_completes_tracking']}")
        print(f"   passed corridor and finished beyond     {entry['n_passed_corridor_and_finished_beyond']}"
              f" (of which completed: {entry['n_passed_and_finished_and_completed']})")
        print(f"   never reached the obstacle              {entry['n_never_reached_obstacle']}")
        print(f"   {'height':>7}  {'ref clears':>10}  {'ref & completes':>15}  {'traversal':>9}  "
              f"{'N90 sub':>8}  {'N90 fresh':>9}")
        for h, v in entry["by_height"].items():
            print(f"   {h:>7}  {v['n_reference_clears']:>10}  "
                  f"{v['n_reference_clears_and_completes_tracking']:>15}  "
                  f"{v['n_traversal_endpoint']:>9}  "
                  f"{str(v['coverage_curve_reference']['n90_budget_subsampling_this_pool']):>8}  "
                  f"{str(v['coverage_curve_reference']['n90_budget_fresh_draws']):>9}")
    print("\nwrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
