# EXP-022A result — prompt-step clearance does not survive the current SONIC bridge

`outputs/exp022_exact_tracking_bridge/`, schema
`exp022a-exact-tracking-bridge-v1`, status `complete`, **64/64 SONIC rollouts**, zero new
ARDY samples. Scene2Motion commit `291f2ec`; two launches of 32 environments, physics seed 0,
both subprocess return codes 0. Result independently re-scored from the source and achieved
qpos archives with zero row or endpoint mismatches.

## Evidentiary boundary

This is a post-hoc bridge over the complete EXP-021 pool. SONIC tracked each reference on its
standard evaluation terrain; the box was **not present in Isaac**. Achieved qpos was then replayed
against Scene2Motion's fixed-box collision geometry. The result is achieved-state replay, not
contact-rich obstacle execution, and the already-inspected x=1.2 m centre is not fresh evidence.

## Primary paired result

At the staged centre x=1.2 m, every kinematically clear reference was lost after tracking:

| box height | reference clears | achieved replay clears after passage | paired retained | paired lost |
|---:|---:|---:|---:|---:|
| 3 cm | 13/64 | 0/64 | 0/13 | 13/13 |
| **5 cm** | **12/64** | **0/64** | **0/12** | **12/12** |
| **8 cm** | **11/64** | **0/64** | **0/11** | **11/11** |
| 12 cm | 7/64 | 0/64 | 0/7 | 7/7 |
| 20 cm | 6/64 | 0/64 | 0/6 | 6/6 |
| 30 cm | 2/64 | 0/64 | 0/2 | 2/2 |

The achieved endpoint requires a non-terminated rollout, passage through the obstacle's lateral
corridor, finishing beyond the obstacle, and exact whole-body collision freedom at the graded
height. The zero is not caused by the non-termination guard: even if that guard is removed, no
achieved trajectory both passes the staged box and clears it.

The paired mid-route control at x=3.6 m is also zero after passage. Its reference exact counts are
2/64 at 3, 5, and 8 cm; 1/64 at 12 cm; and zero at 20 and 30 cm.

The historical contiguous N=8 partitions make the same point. At x=1.2 m, 6/8 blocks contain a
reference 5 cm success and 5/8 contain a reference 8 cm success; **0/8** selected references retain
the achieved-state endpoint at either height. These blocks remain descriptive, not eight fresh
scenes.

## Why the unguarded numbers are misleading

SONIC terminated 53/64 rollouts. Only 14 achieved trajectories passed and finished beyond the
staged box; 11 of those were non-terminated, and all 14 failed exact geometry. Nevertheless, an
unguarded collision query says 43/64 achieved clips clear 5 cm and 42/64 clear 8 cm at x=1.2 m.
Those are mostly robots that stopped before encountering the hypothetical box. At x=3.6 m, all
53 apparent raw clears are nonarrivals.

Therefore never report `achieved.exact_clears` as execution clearance. The scientific field is
`achieved_replay_clear_after_passing`, and it is zero at every height and both centres.

## Tracker-envelope caveat

This result is specific to EXP-021's root-XZ-only references and the pinned SONIC evaluation
configuration. All 12 references that clear the staged 5 cm box are among the terminated set; the
11 SONIC survivors are all non-clearing references. The campaign therefore demonstrates a real
failure of the proposed frozen-prior→tracker pipeline, but it does not isolate whether the cause is
step amplitude alone, the unconstrained root height/heading and speed distribution of this pool,
or the tracker. It does not show that every prompt-generated step is untrackable under every
reference contract.

That caveat does not rescue the current staging result. Selection was defined on these exact
references, and none retains scene-relative clearance after the tracker. A fresh staged-selection
campaign (EXP-022B) is therefore not justified under this pipeline; adding samples cannot repair a
zero reference-to-achieved retention rate.

## Decision

1. Close the current "stage, then select, then SONIC" method claim.
2. Keep the kinematic addressability map and sampling cost as a diagnostic result, not a traversal
   method.
3. Run EXP-023 because prompt handoff still decides the native-interface timing claim.
4. Do not start another controller or tune SONIC on these outcomes for the ICRA deadline. A future
   tracker-envelope study must preregister a reference-quality contract and separate it from the
   scene-placement question.

The complete receipt, paired rows, summaries, achieved state archives, per-launch metrics, return
codes, and hashes are preserved under `outputs/exp022_exact_tracking_bridge/`. Raw SONIC logs and
deterministically regenerable motion pickles remain machine-local and are named by hash in the
receipt.
