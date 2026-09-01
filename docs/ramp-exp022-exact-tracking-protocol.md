# EXP-022A protocol: exact-obstacle tracking bridge

**Status: implementation and protocol complete; no SONIC launch has occurred.** The first
non-dry invocation must happen from a clean, committed Scene2Motion worktree. This is an
archived-pool bridge used to establish whether the natural STEP motions in EXP-021 retain
their clearance under SONIC. It is not the fresh-seed E2 confirmation.

## Question and evidentiary boundary

Does SONIC retain the exact, obstacle-centred clearance of prompt-elicited ARDY-G1 STEP
motions? Every one of the 64 clips from `outputs/exp021_elicited_lift_distribution_v2/`
is tracked. No clip is selected using a SONIC outcome, and no new ARDY generation occurs.

The obstacle is **not instantiated in Isaac**. SONIC tracks each reference on its configured
evaluation terrain, the achieved qpos is archived, and Scene2Motion replays that qpos against
the fixed box geometry. Every dynamic result is therefore named **achieved-state replay**.
It is not contact-rich execution and cannot support a claim about collision response,
stepping on the obstacle, or recovery from obstacle contact.

The pool and the staging point were already inspected. All inferential summaries in this
campaign are descriptive and post hoc. The fresh-seed campaign remains necessary.

## Locked source

Only the complete EXP-021-v2 archive is admissible:

| artifact | locked SHA-256 |
|---|---|
| `receipt.json` | `0c53d8c5dc2bdfa587f8c0b35d069fcd677f1cdc30221b5ce1afa70d1a5ccf7e` |
| `rows.jsonl` | `1d8cc57df2494bd7179940bfe57325ac922f3f41e2581fcc7cb789b5e0c28f71` |
| `qpos.npz` | `2a4b34479aa24894b854301d91bafe1ad870dc530b70eed5b6703eb02c284687` |

The harness additionally requires:

- source commit `cde0246fc89d45cb71f23fa6b8b3d8729ec00e6b`;
- the EXP-021 source at that commit to hash to
  `c68dd1c50f5e8d60eb12f17f0909c9b08cbdf4a11157f17d21c0b6224c603258`;
- the historical calibration and runner sources at that commit to hash to
  `ccbf5a13a78578c31579121650ab9378468c5b73e8c61c758a7c9cc9b503c26a` and
  `29139739ebce4891ebbe694010ff367e281a8f12cb3440a4db5410220b372aa0`;
- exactly seeds 4400–4463, 64 launched, 64 returned, and 64 archived arrays;
- exactly one finite `(200, 36)` qpos per seed;
- every qpos to match its per-row content hash and the complete archive content hash;
- the locked STEP prompt, graded-height vector, clean source receipt, and completed status.

EXP-021's receipt predates an explicit top-level `noise_stream_version` field. Its v2 identity
is bound by the clean source commit: that source refuses generation unless
`runner.noise_stream_version == cal.NOISE_STREAM_VERSION`, and the locked constant is 2.
This chain is recorded explicitly rather than silently inferring v2 from the output name.

## Endpoint correction that motivates this bridge

EXP-021's original best-of-N helper asks whether a clearable box centre exists **anywhere
within ±0.25 m** of a target. That is useful for mapping an addressable region, but it is not
the endpoint of a fixed scene obstacle.

EXP-022A uses `BoxHeightProbe(x, 0.20).clears(qpos, h)` at exactly one centre. On the archived
pool, the staged centre `x = 1.2 m` gives these pre-SONIC counts:

| height | exact fixed-height clear | 5 mm probe lower bound ≥ height |
|---|---:|---:|
| 0.05 m | 12/64 | 12/64 |
| 0.08 m | 11/64 | 10/64 |
| 0.20 m | 6/64 | 5/64 |

The difference is expected: `probe()` is a conservative, 5 mm-resolution lower bound.
The binary graded endpoint and any budget calculation use the exact fixed-height collision
query. The lower bound is retained as a continuous diagnostic and is always labelled as such.

At the unstaged `x = 3.6 m` centre, the exact 5 cm and 8 cm counts are 2/64, not zero.
If the exact staged rates are used as descriptive independent-resampling plug-ins, N=8 is
81.0% at 5 cm, 77.9% at 8 cm, and 54.5% at 20 cm. These are post-hoc sizing quantities, not
fresh-scene success estimates; the probe lower-bound counts are never substituted into them.

## Locked design

### Geometry

- `staged`: box centre `x = 1.2 m`;
- `unstaged`: box centre `x = 3.6 m`;
- box depth: `0.20 m`;
- graded heights: `0.03, 0.05, 0.08, 0.12, 0.20, 0.30 m`;
- Scene2Motion's existing 4 cm robot-body margin remains in the collision model.

Both centres are scored against every reference and every achieved trajectory. The same
motion is therefore the paired unit for the staged-versus-unstaged descriptive comparison.

### SONIC launches

The 64 motion keys are sorted by seed and split before execution:

1. `chunk00_seed0`: seeds 4400–4431, 32 motions;
2. `chunk01_seed0`: seeds 4432–4463, 32 motions.

Each launch uses physics seed 0, 32 environments, the local achieved-state callback, and the
standard tracking/eval termination set. A new archive must use schema v2, contain exactly the
32 requested keys, and report `sample_dt_s = 0.02`. The reference pickle, log, metrics,
archive, and launch receipt are hashed. The per-motion keys, termination flags, progress
values, failed-key list, and aggregate success/progress rates must agree across the log,
metrics JSON, and achieved archive; coexistence of canonical and rank-sharded archives is an
ambiguous mixed-artifact failure. Archive motion IDs must be exactly `0..31`, and ID `i` must
carry the key at position `i` in the input pickle. A matching key set alone is insufficient:
it would not detect two achieved trajectories being relabelled onto the wrong references.

Two chunks are used instead of relying on an unvalidated >36-motion internal batching path.
Historical wall time is 52–94 seconds per launch, so the expected total is 3–6 minutes
including reference and achieved-state geometry scoring.

## Per-motion endpoints

For each tier (`reference`, `achieved`) and each exact obstacle centre, record:

- exact clear/no-clear at every graded height;
- maximum box-height lower bound, with its 5 mm resolution understood;
- valid state count, maximum and final local root x, and maximum absolute local root y;
- actual route-progress ratio from maximum root displacement over the 7.2 m route;
- whether maximum root x passes the expanded obstacle rear edge
  (`x + depth/2 + body margin`);
- the first pass frame, root y at that frame, and whether it remains inside the box's raw
  corridor half-width (`|root y| <= 1.4 m`), so walking around the box cannot count as a step;
- whether the final root remains beyond that edge;
- route completion at actual progress ≥ 0.80.

For achieved rows also record SONIC's termination flag and reported progress. A **stall** is
a non-terminated rollout whose actual route-progress ratio is below 0.80. A
`stalled_before_obstacle` rollout is stalled and never passes the expanded obstacle edge.
This avoids treating SONIC survival/progress as proof of spatial traversal, the defect caught
by EXP-1B.

`achieved_replay_clear_after_passing[h]` requires all five:

1. SONIC did not terminate the rollout;
2. the achieved root passed the exact obstacle;
3. the first passage occurred inside the obstacle's lateral corridor rather than around it;
4. the final root remained beyond the obstacle rear edge;
5. achieved qpos clears the fixed-height box in Scene2Motion replay.

This name does not imply the obstacle was present in Isaac.

For every centre and height, the primary retention table is paired by motion key. It reports
the reference-clear denominator, retained reference clears, lost reference clears,
achieved-only gains, neither-clear pairs, and the retained subset that also completes at least
80% of the route. Marginal reference and achieved counts are diagnostics only: equal marginal
counts can conceal complete turnover in which none of the reference-clear motions survived.

## Descriptive N=8 bridge

For each centre and height, sorted seeds are partitioned into the eight predeclared contiguous
blocks 4400–4407, …, 4456–4463. Within each block, the first reference that exactly clears the
fixed-height box is the selected motion; if none clears, the block fails after eight calls.
The selected key is then looked up in the already-complete achieved archive.

These blocks are a deterministic description of one previously inspected scene, not eight
independent scenes and not a fresh validation. They may size the fresh campaign but must not
be reported as its success estimate.

## Fail-closed and resume rules

Before a real launch, the harness requires:

- an exactly clean Scene2Motion worktree;
- the locked source archive and per-array hashes;
- one SONIC checkout used by both conversion and execution;
- no tracked modifications in SONIC;
- hashes of the SONIC checkpoint, release config, evaluator, callback base, environment
  builder, termination config, callback config, and converter;
- Scene2Motion's Python/NumPy/Torch/MuJoCo and ARDY source identity, plus the separate SONIC
  Python/Torch/CUDA/package identity, Python executable hash, and clean Isaac Lab commit;
- an empty output or a receipt with the identical campaign identity.

Reference rows, their summary, and the running receipt are atomically persisted before SONIC.
Each chunk has numbered attempt directories. A complete key-matching attempt is adopted on
resume only when a durable process-result sidecar records an observed zero return code and
binds the log hash, physics seed, key order, and motion-pickle hash. A complete archive with an
unknown return code is indeterminate and blocks rather than being silently adopted. Motion
pickles are regenerated deterministically and byte-compared on resume. Every numbered attempt
is audited before any earlier result is adopted, so a later failed attempt cannot be hidden by
an earlier valid one. Incomplete interrupted attempts remain preserved. A recorded failed attempt, a
completed attempt whose artifacts no longer validate, a nonzero return code, missing log
metrics, wrong keys, wrong schema, or wrong sample period blocks that output permanently.
Any successor must use a new output directory.

The source artifacts, Scene2Motion numerical runtime, SONIC and Isaac Lab git states, SONIC
Python runtime, checkpoint, core sources, and configs are revalidated after each external
launch. Partial achieved rows are persisted after
each completed chunk. The initial invocation requires a fully clean Scene2Motion tree; a
resume permits only files beneath its own campaign output to account for later git-status
changes. Any change elsewhere still blocks the campaign.

## Commands

Read-only preflight; allowed on a dirty Scene2Motion tree and writes nothing:

```bash
source env.sh
$S2M_PY experiments/exp022_exact_tracking_bridge.py --dry-run
```

Real run, only after committing the protocol, driver, tests, and SONIC-root unification:

```bash
source env.sh
$S2M_PY experiments/exp022_exact_tracking_bridge.py \
  --out outputs/exp022_exact_tracking_bridge
```

The real run spends **zero new ARDY samples** and requests exactly 64 SONIC rollouts.
