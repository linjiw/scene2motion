# EXP-017 protocol: paired absolute-vs-residual step-over packets

**Status: harness landed; first GPU preflight failed before arms, 2026-08-31.** The
representation/phase implementation and CPU-tested exp017 orchestration exist. A `D=1`
preflight spent two donor samples and stopped before nominal/final generation because seed
2600 had no phase-alignable adapted/neutral cycle. Therefore no absolute-vs-residual outcome
or SONIC comparison exists. This document locks the next diagnostic and the first
scientifically fair arm test. It does not claim that either arm works.

## Question and deliberately narrow scope

On a straight prescribed route, does subtracting a phase-aligned neutral gait from a coherent
step donor and transporting that residual onto a held-out nominal gait improve a spatially
placed step-over relative to copying the same adapted donor as an absolute packet?

This experiment tests **representation**, not response repair. It has two final arms:

1. `absolute`: the adapted donor packet from `CoherentPacketPair.absolute`;
2. `residual`: the hierarchy-local adapted-minus-neutral packet from
   `CoherentPacketPair.residual`.

Both use `strength=1`, `duration_scale=1`, and the same geometry-derived integer event shift.
No local optimizer, RepairNet, best-of-N selection, or post-outcome adjustment is allowed.
Version 1 is one unilateral step event only. It cannot support claims about duck,
squeeze/turn, multi-event traversal, route choice, or cross-prior transfer.

## Units, locked placements, and splits

Let:

* `D` be the number of donor seeds;
* `N` be the number of held-out evaluation seeds;
* `P` be the number of route-placement strata.

Donor and evaluation seed sets are disjoint. Exp017 uses fixed, seed-independent obstacle
positions; the descriptive pilot defaults to `x = (2.4, 3.6, 4.8) m`. The old exp016 frame
anchors `(56, 78, 100, 108, 130, 152)` remain an autoregressive-history diagnostic, not the
scene definition for this representation test. For each held-out nominal clip, exp017 locates
complete qpos-derived physical swing cycles, makes a deterministic one-to-one assignment to
the fixed scenes, constructs one `TargetPhaseMatch` per scene, and derives a bounded integer
path-placement shift using only the prescribed route and nominal physical-foot offset. No
final-arm outcome enters this assignment.

Every nominal seed must admit all `P` assignments before the manifest is frozen. A missing
cycle, phase extrapolation, protocol mismatch, or out-of-bound shift aborts the campaign before
any final-arm generation and writes a partial-spend failure receipt; exp017 never substitutes a
nearest frame or silently reduces `N` or `P`. The later confirmatory design must add genuinely
different scene geometries/topologies rather than treating three positions of one box as a
broad scene distribution.

The pilot is conditional on the selected donor bundle. Donor-bank uncertainty is not folded
into seed uncertainty. A later confirmatory experiment must repeat the result across donor
bundles before making a donor-general claim.

## Exact frozen-prior sample budget

Every **completed** campaign spends exactly

$$
B_{\mathrm{physical}} = 2D + N + 2NP.
$$

Cached clips still count toward this budget. A failed preflight reports the actual partial
spend and produces no outcome claim. Kinematic analysis, qpos phase extraction, packet
construction, and SONIC replay do not count as frozen-prior samples, but their wall time is
reported separately. Python `generate(...)` invocation count is reported separately from
samples.

| Stage | Samples | Locked purpose |
|---|---:|---|
| Donor bank | `2D` | For every donor seed/noise stream, generate one adapted clip with the exact step-over prompt and one neutral clip with the exact walk prompt. |
| Held-out nominal | `N` | Generate one walk-prompt nominal per evaluation seed. Reuse that one immutable clip to measure all `P` target cycles; do not regenerate a nominal per placement or arm. |
| Final paired arms | `2NP` | For every evaluation seed and placement, make exactly one absolute call and one residual call. |

The adapted source clip is byte-identically shared by the absolute and residual packets; the
residual arm alone uses the paired neutral clip for subtraction. Donor eligibility and the
deterministic selection order are locked from physical qpos evidence only—complete swing,
minimum relative lift, contralateral stance support, penetration bound, and common phase
support—before any held-out final generation. Evaluation outcomes cannot affect donor choice.

## Pairing and provenance invariants

The only intended difference between final calls is the packet **value representation**.
Before launching either call, the harness must assert and archive all of the following.

### Source-packet lock

`extract_packet_pair` must construct one `CoherentPacketPair`. Its two arms share the same:

* adapted clip and physical adapted event;
* adapted fractional phase queries, packet phase knots, source window, and taper;
* source route-heading frame, skeleton ordering, hierarchy, and root index;
* frozen generator ID, checkpoint SHA-256, adapted sampler seed, integer noise-stream
  version, event-selector string, and code revision.

The residual provenance additionally pins the neutral clip SHA-256. Archive the pair hash,
both packet hashes, both source `StepPhaseCycle` receipts, and the source `PhaseMatch` receipt.
Never build the absolute arm through the legacy position scaffold in
`scene2motion/semantic_scaffold.py`; that would change channels as well as representation.

### Held-out target lock

For every `(evaluation seed, placement)` pair, both final arms must share the exact same:

* target nominal clip SHA-256 and qpos-derived `StepPhaseCycle`;
* `TargetPhaseMatch` JSON and SHA-256, swing side, contralateral stance side, measured stance
  fraction, locked support threshold/window, and physical evidence source;
* prescribed `root_xz`, route heading, target FPS, first heading, obstacle geometry, event
  center, and geometry-derived `center_shift_frames`;
* final prompt string, sampler seed, DDIM settings, and per-sample noise stream v2.

Use the exact step-over prompt for both final arms; prompt wording is a locked constant, not an
experimental factor. The matched nominal call uses the exact walk prompt. A prompt factorial
belongs to exp016 and must not be smuggled into this representation comparison.

### Exact conditioning-support lock

Render both packets before generation and require equal `constraint_support_digest` values.
The v1 renderer must give both arms dense root XZ, dense nominal-root-height plus packet
payload, and full-body global rotations at identical frames and joints. It uses no joint
position channel. The ARDY body-heading feature remains free in both arms because it is not
the route tangent; route heading is used for hierarchy-local canonicalization and global
rotation composition. `program_hash` values are expected to differ because target values
differ, while `support_hash` values must be equal. A support mismatch is a harness error and
invalidates the pair before generation.

Exp017 is a single-shot, non-resumable pilot and refuses a non-empty output directory. A
canonical experiment identity hashes repository/checkpoint/calibration/generation settings,
prompts, splits, fixed scenes, route, source evidence, packet pair, and measurement protocol.
Every program and final-output row carries a schema-tagged canonical identity whose payload
includes the experiment identity plus its own scene/seed/arm, clip, packet/program/support,
phase/control/channel, and output hashes. Final rows additionally carry the frozen manifest
SHA-256 as a foreign key; the manifest is deliberately absent from identity payloads to avoid
a hash cycle. The final receipt anchors the complete rows file, output-identity set, qpos
content, and qpos archive by SHA-256. Noise-stream version 2 is asserted before source
generation, not merely logged. A progressive `run_provenance` identity is complete before the
first source call and is copied into success or failure receipts; it binds code/checkpoint,
calibration, prompts/settings, exact budget, splits, fixed scenes, and route even when no
manifest can be created.

## Physical phase and contact receipts

The generic phase arrays are not labels inferred from seed or frame number. Exp017 must use
`scene2motion/ramp/step_phase.py` to derive them from
`foot_kinematics_series` and locked `StepOverThresholds`:

* takeoff is the first unsupported swing-foot frame after a stable support dwell;
* apex is the maximum physical relative foot lift within that unsupported run;
* landing is the first subsequently supported frame followed by a stable support dwell;
* phase is monotone and fixed at `0.25 / 0.50 / 0.75` for takeoff/apex/landing;
* contralateral stance support is measured over the locked event-centred window;
* a canonical measurement-protocol hash binds support height/speed, support duration/window,
  stance threshold, penetration/lift gates, and the phase convention across source and target
  while allowing their FPS-derived sample counts to differ.

The source adapted and neutral cycles must have the same swing side and one locked phase/contact
protocol. The held-out nominal target must have the packet's swing side. A donor-side failure
is an eligibility rejection recorded in `donor_candidates.jsonl`; a target-side failure aborts
the campaign before its manifest/final arms and is recorded in the partial-spend
`receipt.json`. Truncated takeoff or landing, bilateral flight, insufficient lift, excessive
floor penetration, weak stance support, ambiguous phase progression, incomplete common phase
support, or target extrapolation is never repaired by a nearest-frame fallback.

Support/contact thresholds are locked from tracker-successful neutral walks before final-arm
outcomes, using the existing threshold-calibration receipt machinery. The current kinematic
stage uses the same receipt for donor, target, and reference-output evaluation; the future
SONIC stage must reuse that exact receipt for achieved-state evaluation.

## Launch sequence

Exp017 refuses a dirty worktree. The first real-model preflight was locked as:

```bash
source env.sh
$S2M_PY experiments/exp017_ramp_residual_stepover.py \
  --out outputs/exp017_ramp_preflight_d1_n1_p1 \
  --n_donors 1 --n_seeds 1 --obstacle_x 3.6 \
  --threshold_calibration_receipt outputs/exp016_threshold_calibration/receipt.json
```

It failed closed in source discovery after exactly two donor samples: both adapted and neutral
cycle lists were empty for seed 2600, with no nominal or final-arm samples. This is a
schema/eligibility failure, not a result. The v1 failure ledger is archived at
`outputs/exp017_ramp_preflight_d1_n1_p1/`. It preserved hashes and rejection text but not the
rejected qpos, so it cannot distinguish clearance from speed as the failed support sub-gate.
The harness now archives and hashes every candidate donor qpos before eligibility and emits
diagnostic-only support fractions, clearance, and speed summaries without changing thresholds
or selection.

Before inspecting any additional donor outcome, the second attempt is locked to the next
wider **sequential** bank, not a cherry-picked seed:

```bash
source env.sh
$S2M_PY experiments/exp017_ramp_residual_stepover.py \
  --out outputs/exp017_ramp_preflight_d4_n1_p1 \
  --n_donors 4 --n_seeds 1 --obstacle_x 3.6 \
  --threshold_calibration_receipt outputs/exp016_threshold_calibration/receipt.json
```

It used donor seeds 2600–2603, evaluation seed 2800, the unchanged calibration receipt and
gates, and one fixed placement. Three donor pairs were eligible and the deterministic selector
chose seed 2603: a coherent left-swing source with 0.232 m adapted relative lift. The run then
failed target assignment for nominal seed 2800 after nine samples (`8 + 1`), before its
manifest or either final arm. The ledger is archived at
`outputs/exp017_ramp_preflight_d4_n1_p1/`. This establishes source eligibility for the fixed
design, not packet performance.

The second failure exposed the same evidence-order issue on the target side: nominal qpos and
per-cycle assignment checks were not written before the exception. The harness now archives
all returned nominal qpos before any gate, records phase/shift/window rejections per cycle,
and hash-anchors the finalized nominal ledger. Before inspecting any new nominal outcome, the
third attempt is locked as an exact-design replay whose only intended difference is logging:

```bash
source env.sh
$S2M_PY experiments/exp017_ramp_residual_stepover.py \
  --out outputs/exp017_ramp_preflight_d4_n1_p1_v2 \
  --n_donors 4 --n_seeds 1 --obstacle_x 3.6 \
  --threshold_calibration_receipt outputs/exp016_threshold_calibration/receipt.json
```

The replay keeps donor seeds 2600–2603, evaluation seed 2800, placement, thresholds, shift
bound, prompts, and sampler fixed. All regenerated samples count again. If assignment repeats
its failure, the run stops after nine samples and the archived checks determine the binding
gate; if it reaches both arms, it spends eleven.

A completed preflight remains a schema/eligibility check, not a result. Inspect
`receipt.json`, `manifest.json`, both packet/program support hashes, physical cycle receipts,
and all row counts before scheduling the larger kinematic pilot. The current harness
intentionally stops at kinematics; an equal-budget `2NP` SONIC achieved-qpos stage must be
added and run before reporting execution success.

## Endpoints

The primary kinematic quantity is the obstacle-centred whole-body
`max_box_height_lower_bound_m` from `BoxHeightProbe`, evaluated against the robot's simulation
collision primitives with the separately measured body margin. It is not swing-foot peak.
For a fixed predeclared obstacle, also report `obstacle_collision_free` and the full
`evaluate_local_step` gate bundle.

The required E1a kinematic endpoint vector for every arm is:

* maximum whole-body box-height lower bound and fixed-obstacle minimum clearance;
* local-step success, root progress ratio, and path error;
* crossing-position error (`phase_error_m`) and crossing-frame error;
* selected lead/swing side and whether it matches the packet side;
* contralateral support fractions, maximum unsupported duration, bilateral-flight fraction,
  landing dwell, and maximum foot-floor penetration;
* integrated rotation/root-height deformation and frozen-prior calls/wall-clock time.

E1b adds the same crossing, contact, progress, clearance, and local-step readouts after equal-
budget SONIC achieved-qpos replay.

Until an obstacle is instantiated in Isaac, “SONIC execution success” means replaying achieved
qpos against the Scene2Motion simulation collision model and the physical support/progress
gates. It is **not** an Isaac obstacle-contact result. Report reference and achieved-state
endpoints separately; never substitute tracking success or foot peak for traversal success.

## Statistical analysis

E1a crosses the same held-out nominal seeds with fixed path positions of one box geometry and
uses one selected donor bundle. Those positions are **placement strata, not independent
scenes**. The pilot therefore reports raw paired residual-minus-absolute effects per placement
after averaging repeated seeds, the descriptive mean over placements, raw paired counts, and
no confidence interval or p-value. It cannot support population inference.

The attempted-seed ledger separately reports source eligibility and pre-manifest target
attrition from partial-spend receipts. These cases are not assigned to an arm that was never
generated. After the manifest is frozen, every planned absolute/residual pair remains in the
arm denominator, including generation/scoring failures; no outcome-dependent complete-case
filtering is allowed.

E1b must introduce independently sampled scene geometries/topologies and repeat donor bundles.
For each true scene, collapse repeated seeds before a cluster bootstrap over scenes, or use a
predeclared crossed/mixed-effects model that represents seed and scene reuse. Seeds are never
pooled as independent observations. Primary interpretation at both stages uses the joint
endpoint vector: better clearance without stance, crossing, and progress is not a successful
kinematic step; execution claims additionally require achieved-state retention.

## Staged decision and kill condition

E1a advances the packet representation to a **local-response-optimizer implementation** only
if the residual arm shows a descriptive whole-body box-clearance advantage over the exact-
support absolute arm without degrading swing-side, stance/contact, crossing, and progress
gates, or produces a valid kinematic local step where the absolute arm does not. This is only
permission to implement the optimizer; it is not evidence for execution, RepairNet,
cross-prior transfer, route planning, duck/squeeze, or the full E1 claim.

E1b, not the current harness, adjudicates the public-interface path. If independently varied
scenes/donors with equal-budget SONIC replay show neither a residual advantage over same-
support absolute packets nor any physically valid achieved step off the floor, stop expanding
this packet path. Record the negative result and pivot to an output-space residual adapter
`x_adapt = x_nominal ⊕ A_psi(S, x_nominal)` or per-window latent optimization. Do not retreat
to an audit-paper claim.
