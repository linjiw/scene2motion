# ASTRA A1: Historical WALK apparatus qualification

**Status: preregistered**. 2026-09-05. Protocol version 1; active plan: [astra.md](../astra.md).
The immutable clean execution commit and this document's SHA-256 must be recorded before launch.
This is a new development pilot, not EXP-029 validation or a fresh method comparison.

## Question and assignment

Can archived neutral WALK carriers traverse a reachable local region in matched absent/raised-beam
runs, and does the spawned beam retain its intended dimensions and pose?

Use every EXP-023 `s4500_all_walk`–`s4507_all_walk` in order. If apparatus readback passes but
fewer than seven paired controls pass, use the declared EXP-023b reserve `s4640_all_walk`–
`s4647_all_walk` in order. Preserve the latter campaign's refusal metadata. No new generation,
quality-based preselection, or seed replacement. The preparation manifest binds arrays/receipts.

Each cohort runs eight environments, absent then raised, physics seed 0. At most four launches,
32 assigned rollouts. Stop after a cohort with at least 7/8 **paired** local passes and all
apparatus checks passing; stop immediately on apparatus or harness failure. If both cohorts fail,
return to carrier/controller compatibility. Do not launch a blocking beam or duck comparison yet:
those require a separately frozen follow-on protocol. These controls are development data forever.

## Pinned apparatus and resource limits

Use the clean patched SONIC checkout at `7c63c539a17008f5efb1e768034c0fb434ae1f65`, released
motion-tracking checkpoint and current G1 XML with existing expected hashes. Record all core and
evaluator source hashes, checkpoint/config identities, commands and resolved termination terms.
Both arms use `tracking/eval`, eight environments, spacing 12 m and episode limit 20 s.

Beam full extents: `(0.24, 2.25, 0.25)` m; center `(1.2, 0, 1.725)` m; underside 1.60 m.
Store pose in every motion's `table_pos/table_quat`, including the absent-arm input; disable
`add_object` and enable `add_table` only in the raised arm. Identical motion pickle for both arms.

Before every launch: at least 12 GiB free VRAM, 18 GiB available RAM, no concurrent Isaac or
campaign poller. One scoped advisory lock. Abort only this launch below 1200 MiB free VRAM or
1500 MiB available RAM, or after 900 s. Preserve failures; never interrupt a co-tenant. A resource
refusal before launch may resume without spending an attempt. A spent failed/partial attempt
cannot automatically rerun; a demonstrated harness defect requires a new versioned attempt.

## Evidence and endpoints

The read-only callback retains live rigid-body poses at every valid achieved sample, first/last
valid snapshots, all environment origins/table centers, and first/batch-end USD world extents.
No reset padding is scored (achieved archive schema 2). Require pose and extent error ≤2 mm,
identity quaternion component error ≤0.001, and no neighboring beam intersecting the conservative
local-frame route envelope `x=[-1.5,10]`, `|y|≤2.0` m (plus beam half-width). The absent arm must
contain no table. This checks the sampled valid prefix; it is not a contact measurement.

Operational local completion: root reaches x=1.82 m (beam far edge plus 0.50 m), **all robot
collision primitives including feet**, inflated by the existing 4 cm coverage margin, are beyond
x=1.32 m, and the robot remains upright (pelvis ≥0.50 m; up-axis z≥0.70) and root within
`|y|≤1.20` m for 0.20 s, completed by 4.0 s. All pre-completion nominal/margin collisions, falls
and corridor violations invalidate it. Rescore the whole-body completion prefix using evaluator
v2; later route failure is reported separately. In the absent arm the raised geometry is explicitly
a virtual compatibility check, never a physical exposure. Preserve root-only v2 and 7.2 m
full-route outcomes. Contact is **not measured**, never false. This is not high-quality bank
qualification: floor/slip/dynamics quality and repeated physics seeds remain future requirements.

## Reporting and integrity

Report every assigned carrier/arm, paired pass count, readback deviations, local and full-route
outcomes, cutoff flags, hashes, resource/timeout failures and costs. The 7/8 threshold is an
engineering progression rule, not a generalization claim. No significance or causal-method
claim from these reused eight-carrier cohorts. Retain old EXP-029/031 definitions and outputs.
Execute from an isolated clean checkout, leaving unrelated worktree changes untouched. Resume
only complete hash-verified arms; any source/input/analysis mismatch fails closed.

## Pre-execution harness amendment 1 (2026-09-05)

Commit `401acc1` stopped at its first resource preflight: 14112 MiB free VRAM, 17744 MiB
available RAM, and two purported Isaac processes. No SONIC process was started. Host process
inspection identified both as the installed `isaacsim_mcp` documentation servers (`isaacsim-mcp`
and `nat mcp serve`), not simulation engines. The successor recognizes only these specific
entrypoints as documentation services, retains them in the resource receipt, and still refuses
other Isaac processes. **The 12/18 GiB thresholds, inputs, endpoints and budget are unchanged.**
Preserve `outputs/astra_a1_apparatus_v1` (identity and prepared input only); use a new output
`outputs/astra_a1_apparatus_v1_1` and clean committed source for the corrected harness. Resource
refusals now write their measured report before returning. There are still zero spent rollouts.

## User-authorized resource amendment 2 (2026-09-05)

The user explicitly authorized using less than 18 GiB RAM and adapting execution parallelism.
Before any A1 rollout, adopt the repository's measured `SONIC_LAUNCH_GATE`: **5500 MiB free
VRAM and 9500 MiB available RAM**. Its source measurement is
`outputs/probe_sonic_vram/report_envs32.json`: 32 environments used 3769 MiB peak launch VRAM
and approximately 6810 MiB host RAM. Record simulator co-tenants rather than prohibiting them;
never stop their processes. Keep one owner of this campaign and forbid competing legacy pollers.
Increase the running host-RAM abort floor from 1500 to **2500 MiB**; retain the 1200 MiB VRAM
floor and 900 s timeout. Abort only this launch if the shared host becomes constrained.

Use fresh output `outputs/astra_a1_apparatus_v1_2` and a new clean source commit. Preserve v1
and v1_1 preflight artifacts. Scientific endpoints, all 16 source candidates, eight environments
per cohort, arm order and at-most-32-rollout budget are unchanged. Eight is the current useful
parallelism: duplicating the eight assigned references to fill extra environments would add no
independent evidence. Larger new cohorts or a different batch layout require a recorded new
plan before their outcomes, not a change midway through paired arms. This amendment supersedes
only the earlier resource limits/co-tenant prohibition, with explicit user authority; no spent
rollout or outcome has been reinterpreted.
