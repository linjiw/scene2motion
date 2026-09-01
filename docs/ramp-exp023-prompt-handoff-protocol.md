# EXP-023 protocol — delayed prompt handoff

**Status:** preregistered before generation. This protocol spends seeds 4500–4507 once.
No result-dependent arm, threshold, seed, history policy, chunk, or observation-window change
is permitted.

## Question and scope

For the released frozen ARDY-G1 Horizon52 checkpoint, does a STEP event remain tied to the
start of a rollout, or can it follow a later WALK-to-STEP prompt change made through ARDY's
released interactive autoregressive interface?

This is a kinematic interface experiment. It makes no SONIC, contact-rich execution, or
cross-prior claim. It is specifically scoped to the interactive GUI's released default
**minimum History Crop Length of one token (four frames)**. ARDY documents this setting as
facilitating faster adaptation to new prompts and constraints. The campaign retains the full
accepted output transcript immutably, but each continuation sees only its last four frames.
It does not establish what would happen under longer-history settings or a latent-state API.

## Locked generation design

The eight fresh paired seeds are **4500–4507**. Each seed produces four schedule trajectories:

| arm | frames 0–51 | 52–103 | 104–155 | 156–207 | prompt onset |
|---|---|---|---|---|---:|
| `all_walk` | WALK | WALK | WALK | WALK | none |
| `step_0` | STEP | STEP | STEP | STEP | 0 |
| `step_52` | WALK | STEP | STEP | STEP | 52 |
| `step_104` | WALK | WALK | STEP | STEP | 104 |

The exact cached strings are `A person walks forward.` and
`A person steps over an obstacle.`. There are 32 frozen-prior samples: one complete schedule
trajectory per (seed, arm). Generation is locked into four calls of two seeds × four arms
(B=8): `(4500,4501)`, `(4502,4503)`, `(4504,4505)`, and `(4506,4507)`. This bounds separated-CFG
expansion while keeping every same-seed causal comparison in one call. Each call makes four
internal Horizon52 autoregressive window calls, for 16 internal calls in total; those are not
128 independent samples.

All four same-seed rows receive identical corresponding-window initial noise. One persistent
noise-stream-v2 context encloses all four windows within a chunk, so later windows receive
advancing draws rather than replaying window 0. End-to-end replay must preserve this B=8 chunk
plan; per-row noise construction alone does not promise batch-shape-invariant GPU inference.

The continuation contract is locked window by window:

| window | global history start | accepted transcript before | visible input | model `num_frames` | accepted transcript after |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 208 | 52 |
| 1 | 48 | 52 | 4 | 160 | 104 |
| 2 | 100 | 104 | 4 | 108 | 156 |
| 3 | 152 | 156 | 4 | 56 | 208 |

For every continuation, the dense constraint mask and observed motion are sliced from the
global history start. `autoregressive_step` receives only the visible four-frame suffix and
`num_frames=208-global_history_start`. Its returned visible prefix is audited, but cannot
replace already accepted frames: exactly the returned suffix after the four visible frames is
appended to the immutable full transcript. This follows the interactive demo's append
semantics and prevents a prompt change from retroactively rewriting a causal prefix.

The following are hard gates, first within every chunk and then after exact global row-order
reconstruction:

1. exactly four audited latent draws per chunk and 16 completed window calls overall;
2. within each (seed, window), all four arm noise hashes are equal;
3. within each (seed, arm), all four window hashes differ, and different seeds do not collide;
4. the history starts, accepted lengths, visible lengths, model lengths, and output lengths
   exactly match the table above;
5. every visible-input hash matches the archived transcript slice `[48:52]`, `[100:104]`, or
   `[152:156]`, as applicable;
6. every stable full-transcript hash matches the archived normalized features through 52, 104,
   156, or 208 frames;
7. normalized feature prefixes are byte-identical for `all_walk`, `step_52`, and `step_104`
   through frame 51, and for `all_walk` and `step_104` through frame 103;
8. the same causal-prefix equalities hold after deterministic decoding to qpos at native
   precision.

The model generates 208 frames. All 208 normalized features and qpos frames are archived, but
scientific endpoints use only frames 0–199. A frame-156 switch is excluded because only 44
scored frames remain, shorter than exp021's approximately 54-frame 90th-percentile latency.

Generation otherwise reproduces exp021: 25 fps, five deterministic DDIM steps,
`cfg_weight=(2,2)`, and the same 200-frame, 7.2 m straight reference-speed route. The only
constraint is dense root XZ. Frames 200–207 are unconditioned padding, not an extended route.
The actual normalized mask must contain exactly 400 nonzero `root_pos` entries and no other
channel.

## Event and obstacle endpoints

Each STEP arm is evaluated in an equal 96-frame post-onset window `[s,s+96)` for
`s in {0,52,104}`. The `all_walk` trajectory is evaluated in all three matching pseudo-onset
windows. Missing events remain in the planned denominator.

The event detector is frozen as follows:

1. sweep 120 obstacle centres between 0.30 m inside the beginning and end of the window's
   prescribed route segment;
2. at each centre, compute whole-body clearable height with `BoxHeightProbe`, 0.20 m box depth,
   5 mm resolution, 0.40 m cap, and the released margin-inflated G1 collision model;
3. reject centres not traversed before-over-after by both physical-foot envelopes inside the
   analysis window;
4. call an event present only if the largest remaining whole-body lower bound is at least
   **0.03 m**;
5. choose maximum whole-body height, then earlier route centre; identify the side/frame by
   maximum physical-foot bottom clearance while overlapping that centre, breaking ties by
   earlier frame and then left side.

The row records event presence, missing reason, global frame, prompt-relative latency, side,
profile centre, physical-foot position and clearance, and whole-body clearable height.

The secondary fixed-placement endpoint freezes the historical, provisional exp021 34-frame
anchor as a design constant; it is not presented as a separately validated timing estimate.
For each onset `s`,
the obstacle centre is the prescribed route position at frame `s+34` (approximately 1.23, 3.11,
and 4.99 m). The exact 0.20 m-deep box is scored over the complete 200-frame trajectory at
heights 0.03, 0.05, 0.08, 0.12, 0.20, and 0.30 m. A success requires both physical feet to
cross the margin-expanded obstacle slab in before-over-after order **and** the collision probe
to clear that height. Collision-free nonarrivals are not successes: their endpoint lower bound
is zero. The raw collision-free lower bound and booleans, traversal result for both feet, and
traversal-gated success booleans are all retained separately. Route MAE, progress ratio,
physical-foot penetration, event occurrence, generator calls, and wall-clock are supporting
endpoints.

## Gates and inference

Before generation: clean worktree, empty output directory, frozen source/checkpoint/runtime/G1
identities, ARDY revision `059b8007df0ba194a006a877b59a563955ac7b70`, denoiser SHA-256
`0c16ac26c1ab75e511cd24bb25bd9ad92078a2460b4ce78529788b5da22647a2`, ARDY runtime commit
`693f74d13b3d04a0a22ce127ee79c929dd89756b`, G1 XML SHA-256
`5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1`, noise stream v2,
Horizon52, four-frame token/history, 25 fps, cached prompt bytes, exact route/mask, exact 32-row
plan, and exact four-chunk plan. The receipt binds a direct SHA-256 of this protocol as well as
all other source hashes. Identities and hashes are revalidated after every completed chunk,
after generation, and after analysis.

After scoring, two measurement gates apply without deleting rows:

- **substrate:** at least 4/8 `step_0` rows contain the frozen event in frames 0–95;
- **specificity:** at most 1/8 `all_walk` seeds contains an event in any of its three matched
  windows.

Failure produces a durable refusal with all returned trajectories and rows preserved. It does
not license retuning the 3 cm threshold or reusing spent seeds. Delayed-arm absence is not a
gate failure; it is a substantive outcome.

At n=8, inference is descriptive and paired. The primary timing result is the event rate for
each delayed arm using all eight planned seeds, with missing events retained. Report exact-box
graded rates on the same planned denominators. Present-event slopes, complete-case per-seed
slopes, and paired differences are secondary diagnostics only and must state their observed
complete-case counts. They cannot override missing delayed events. No confidence interval or
binary “timed prompting supported” verdict is assigned at this sample size.

## Evidence and accounting

Before model construction or generation, write a running receipt, empty `rows.jsonl`, empty
`qpos.npz`, empty `features.npz`, and empty `noise_audit.json`. Before each B=8 call, mark the
chunk running, its two seeds spent, and its eight trajectories launched. Immediately after a
return, archive that chunk's normalized features and raw runner audit before any shape or
causal validation. Only then may the chunk be marked complete. Completed chunks, their exact
eight returned trajectories, and their four audited window calls remain durable if a later
chunk fails.

A generation exception makes the current chunk's completed-window/sample count unknown; it
does not erase the exact accounting of prior completed chunks. The receipt lists spent and
unlaunched locked seeds. Successful completion requires four completed schedule invocations,
16 completed window calls, and exactly 32 returned, converted, and analyzed trajectories.

The production path constructs `ArdyRunner` only after the empty evidence bundle is durable. A
runner factory and injected body/scorers exist solely for CPU tests; any run using one is
explicitly marked non-evidentiary. Even injected event and box records must pass the frozen
schema, window, threshold, traversal, obstacle, and monotonic-height checks before reaching a
gate. Decoded qpos is archived and fork-compared at native precision so a lossy cast cannot
make two prefixes appear byte-identical.
