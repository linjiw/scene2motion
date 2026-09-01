# EXP-019 protocol: gait-matched placement and the paired packet comparison

**Status: preregistered; no exp019 result is reported here.** Third placement design in the
E1 family, after exp017's fixed-frame pool exhausted at 1/8
(`docs/ramp-e1-protocol.md`) and exp018's route warp persisted at 1/6
(`docs/ramp-exp018-route-warp-negative-2026-09-01.md`). Both prior campaigns are preserved
as evidence and all three are reported together. Frozen before any exp019 sample is
generated.

## The change, and its evidential basis

exp017 and exp018 both tried to bring a gait event to a **predeclared** obstacle — by
selecting seeds whose apex happened to land there (1/8), then by warping route timing to
move the root there (1/6). exp018 measured why the second fails: the prior re-plans its
gait when root-path timing changes, and the calibration's selection key made things worse
by scoring deformation against a fixed 7.2 m route while cycles were observed on
4.776/7.200/9.552 m routes (up to 2.4 m of conditioning displacement).

exp019 inverts the lever: **the obstacle is placed at the nominal's own swing apex, on the
nominal's own generating route.** The required center shift is then zero by construction,
no nominal is regenerated, and no route is warped. Replayed on exp018's already-generated
pool this yields **12/16** eligible seeds (3 slow, 3 reference, 6 fast; obstacle x from
1.04 m to 5.03 m) — the analysis is in the exp018 note and is design-informing, not
confirmatory: exp019 runs on fresh seeds.

**Scope, stated up front.** Per-seed placement supports the paired absolute-vs-residual
comparison and nothing else. It is *not* a fixed scene, so exp019 cannot support any
claim about placing behavior at a scene-specified position — exp017 and exp018 are the
evidence on that question, and both are negative. E1's decision rule is about
representation only.

## Design and exact budget

* **Donor bundle:** exp017's archived bank, regenerated deterministically and required to
  reproduce the archived clip hashes, selected seed (2603), swing side (left), center
  frame, and packet payload content hash — same gates as exp018. 2D = 8 samples.
* **Nominal pool:** K = 16 fresh seeds 3900–3915, disjoint from all prior campaigns, each
  at the three calibrated speeds (six batches of eight) on that stratum's own constant
  route (4.776 / 7.200 / 9.552 m). 3K = 48 samples.
* **Outcome-free placement selection.** For each pool clip, enumerate complete left-swing
  cycles under the frozen v3 Pmin (0.042 m) and the ±2-frame packet window. The obstacle
  is placed at `x = route_progress(apex) + foot_offset`, where
  `foot_offset = foot_x(apex) − root_x(apex)` — the swing foot's apex position measured
  against the *prescribed* route rather than the achieved root, so exp017's assignment
  solves `desired_root = route_progress(apex)` and the center shift is **exactly** zero
  rather than zero up to the ~1.4 cm root-tracking residual. A cycle is
  placeable when that obstacle lies strictly inside the clip's route with the simulation
  body margin
  (`route_lo < x − (depth/2 + margin)` and `x + (depth/2 + margin) < route_hi`). Per seed,
  pooling strata, select the placeable cycle minimizing the frozen key:
  1. |x − route midpoint| (margin and autoregressive-history depth);
  2. −prominence;
  3. stratum order (reference, slow, fast);
  4. apex frame;
  5. phase-evidence receipt digest.
  No generated arm response, collision result, or traversal outcome may enter selection.
  **N = 10** evaluation seeds are the first ten eligible in predeclared seed order; fewer
  than ten ⇒ fail-closed pool-exhaustion stop (replay rate predicts ≈ 12/16).
* **Arms.** For each selected seed, exactly one `absolute` and one `residual` arm (STEP
  prompt both, strength 1, duration scale 1, shift 0, identical support hashes and channel
  usage, no joint-position channel, free body heading) on the nominal's own route.
  2N = 20 samples. A third **`nominal`** reference arm costs nothing: the pool clip itself
  is scored against the same obstacle, answering whether either packet adds clearance over
  the gait that was already there.

Completed budget: `2D + 3K + 2N = 8 + 48 + 20 = 76` frozen-prior samples, exact accounting;
every planned arm stays in its denominator.

## Endpoints and analysis

exp017's E1a endpoint vector, unchanged: obstacle-centred whole-body box-height lower
bound and collision-free flag against the simulation collision model with the measured
body margin (**never** swing-foot peak), crossing position/frame error, lead side vs packet
side, stance/contact measures, progress ratio, path error, integrated program deformation
vs the nominal, and generator calls / wall-clock.

Analysis is descriptive and paired on the seed, the only replication unit: per-seed
residual−absolute and packet−nominal differences, their raw counts and medians. **No
confidence intervals, p-values, or population claims** — one donor bundle, one box
geometry, per-seed placements. Report placement, stratum, apex frame, and prominence per
seed so any placement-difficulty confound is visible.

## Decision rule (inherited from exp017, unchanged)

The residual representation advances to the local response optimizer only if it shows a
descriptive whole-body box-clearance advantage over the exact-support absolute arm
**without** degrading swing-side, stance/contact, crossing, and progress gates, or
produces a valid kinematic local step where absolute does not. That is permission to
implement the optimizer — not evidence for execution, RepairNet, transfer, routing,
duck/squeeze, or the full E1 claim. If neither packet arm beats the nominal reference on
clearance, the packet path itself is in question and the pivot is to output-space residual
adaptation, per exp017 §Staged decision.

Provenance: clean worktree, full identity binding with post-generation revalidation, the
hash-locked exp016 threshold and v3 calibration dependencies, atomic incremental evidence,
exact accounting, hash-anchored refusals, no post-hoc gate relaxation. Harness-defect
aborts may be fixed and rerun into a fresh output directory with every attempt preserved;
gate outcomes are never rerun. Execution remains out of scope (kinematic stage only).
