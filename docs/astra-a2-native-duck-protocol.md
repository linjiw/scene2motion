# ASTRA A2: native duck correction development experiment

**Status: preregistered** — 2026-09-05, before any A2 generation. Version `native-duck-d0`.

## Question and scope

Does a bounded update from measured reference headroom survive frozen SONIC execution,
relative to a fixed extra dip and equal-budget resampling? This is method development,
not a blind generalization study or a qualified high-quality motion bank. A1/A1b qualified
the apparatus, not these new native-height-conditioned references. No new training.

## Assigned units and method

Eight seed–scene units, seeds 60000–60007 in order. Units 0–3 have beam underside 1.05 m;
4–7 have 1.15 m. These are **two geometries**, not eight independent scene families.
Beam center x=1.2 m, y=0, full size (0.24, 2.25, 0.25) m. Route x=0 to 7.488 m.
ARDY released Horizon52, v2 noise, cached WALK prompt, 208 frames at 25 Hz, speed 0.9 m/s,
five diffusion steps and default CFG (2,2). Reuse the pinned checkpoint/runtime/XML checks.

Every arm uses the same root path/heading and native root-height interface. Initial dip
is `clip(1.30 - beam_underside + 0.03, 0, 0.45)` m. Quintic ramp from route 0 to 0.45 m,
hold through 1.8 m, recover through 2.7 m; endpoints have zero first/second derivatives.

- Initial: one proposal.
- Fixed extra: initial dip +0.10 m, capped at 0.45 m; one proposal. This is an untuned
  engineering baseline, not the final strongest tuned heuristic.
- Measured: initial plus two same-seed updates; each adds
  `min(0.10, max(0, 0.02 - measured_min_overhead))`, capped at 0.45 m.
- Best-of-three: same initial schedule, seeds base/base+100/base+200.

The 0.02 m target is additional reference slack after the existing 0.04 m geometry
coverage; it is **not an execution-calibrated safety margin**. Unit gain is a testable
heuristic, not an identified controller inverse. Both multi-candidate arms choose by
reference recovery reach, then maximum whole-body minimum clearance, earliest tie.
No achieved-state input enters updates or selection. Already-satisfied updates are no-ops.
All two update slots are generated and charged, including no-ops; all candidates archived.
Free-nominal controls use the same path/heading/seed with root height unpinned.

## Exact budgets and execution

Seven ordered B=8 generation calls: free nominal, initial, fixed extra, repair1, repair2,
resample1, resample2 = **56 actual references**. Initial is physically shared across
three arms; logical per-arm costs are 1/1/3/3, plus 8 free controls. Reserve additional
seeds 60100–60107 and 60200–60207. Save identities and exact ordered specs before calls.

The selected 8×4=32 references run with **num_envs=32**, fixed unit-major/arm order,
both absent and present, physics seeds 0 and 1: **128 rollouts maximum**. Run eight free
nominal controls absent, seed 0, num_envs=8: **136 total**. Per-motion beam poses carry
the assigned height; verify live pose/extent/neighbor exclusion. No cherry-picked launches:
all finite valid references execute even if reference geometry fails (diagnostic simulation).
If generation or instrumentation fails, stop and preserve the entire attempt; missing
assigned records are not dropped. No retry after a scientific failure; resume only completed
hash-verified work. A harness defect requires a documented amendment and preserved attempt.

SONIC resource gate: 5500 MiB free VRAM, 9500 MiB available RAM, co-tenants recorded.
During a launch, abort below 1200/2500 MiB free VRAM/available RAM or after 900 s.
Generation uses ARDY_GENERATION_GATE (4096/8192 MiB), no text encoder. Release ARDY before
Isaac. One ASTRA lock, clean committed execution clone, outputs outside clone. No other
job is stopped. Do not change batch size after spending seeds absent a harness failure.

## Endpoints, analysis and next decision

Keep A1 whole-body recovery, 0.20 s dwell and 4.0 s local deadline unchanged. Score absent
motion against the raised virtual beam for carrier progress and separately against target
geometry for clearance transmission. Present primary requires local recovery and no nominal
or coverage-envelope violation. Physical contact remains **not measured**; do not call
this contact-free or safe traversal. Preserve cutoff, fall-in-valid-prefix, progress, geometry,
full-route and apparatus flags independently. Full motion quality remains unqualified.

Report all 8 assigned units per arm per physics seed, paired counts and Wilson intervals;
no significance/generalization claim over two geometries. Also report both-seed success,
free-control yield, reference selection, every candidate cost and all failed attempts.
If corrected absent motion fails, address native-control trackability before scaling.
If absent passes but target fails, analyze clearance transmission/contact geometry. If a
method produces repeatable target crossing, freeze its next version and use fresh carrier-
and scene-disjoint evaluation. Never tune this protocol on its own outcomes.
