# Adversarial review of the 2026-08-30 changeset (noise v2 · phase4e · exp016 · sonic_state_export)

Six area reviewers + two-lens verification (evidence / impact) over the uncommitted working
tree, before running it. **What is confirmed sound:** the v2 sampler semantics (persistent
per-sample streams verified against ARDY's one-`randn`-per-window loop, batch-position
independence and call-to-call reproducibility pinned by tests and by a live GPU run); budget
accounting (one attempt = one generation, consistently across loop/resample/demo/ledgers);
SONIC state-export joint mapping (by name, duplicate/missing rejection, Isaac↔MuJoCo order and
wxyz quaternions verified against both checkouts); exp016 donor/eval seed disjointness
(enforced in code); scaffold heading-frame math (bit-identical to ARDY's own
`corrective_mat_Y`); "117 tests pass" (reproduced twice, no hidden skips).

## Defects to fix, by stage they poison

### Blocks 1B (execution replay) — fix before any achieved-state run

1. **[critical] `sonic_state_export.py:426` — final frame of every *successful* rollout is
   post-reset teleport garbage.** SONIC's `motion_time_out` fires during iteration
   `ref_len−1` and Isaac resets the env inside that same `step()`; the snapshot at index
   `ref_len−1` is the frame-0 pose at the env origin, kept inside `valid = ref_len`. Finite
   values, so no guard fires; end-pose clearance reads the start pose and last-interval
   velocities spike ~350 m/s. SONIC's own metrics slice `[: i−1]` for exactly this reason.
   *Fix:* `valid = ref_len − 1` for non-terminated envs (or drop the frame whose
   `extras['time_outs']` is true).
2. **[low] `sonic_state_export.py:448` — no frame-0 sample; achieved[i] ↔ 50 Hz reference
   frame i.** Consumers must subsample `achieved[1::2]` against 25 fps references; encode the
   convention in the archive.

### Blocks exp016 (2A) — fix before the pilot

3. **[high] `exp016:387` — default support thresholds classify a tracked neutral walk as
   ~80 % airborne**, so `kinematic_step_success` would be ~0 in every arm and the factorial
   reads "all null". The promised calibration from tracker-successful neutral walks must run
   first (`--threshold_calibration_receipt` path exists; the tool does not).
4. **[high] `exp016:702` — frame-count gates (`max_unsupported_run_frames=2`,
   `landing_dwell_frames=3`) are applied at 25 fps for ARDY rows and 50 Hz for SONIC replay**,
   silently doubling stringency at replay. Express both in seconds, convert per fps.
5. **[medium] `exp016:706` — `_resample_plan` maps the full route onto terminated rollouts**,
   corrupting `path_error_m` for early falls. Resample to `ref_len`, truncate to
   `valid_length`.
6. **[medium] `stepover_eval.py:38` — same frame-count/fps coupling inside
   `evaluate_local_step`** (root cause of 4).
7. **[low] `exp015b:126` — `phase_error_m` is slab-bounded (±~0.4 m)**; report the global
   swing-peak offset alongside and rename the bounded quantity.
8. **[low] `exp016:837` — rows.jsonl written only after SONIC replay**; dump the kinematic
   ledger right after `aggregate()` so a replay crash cannot lose the screen.
9. **[low] `exp016:403` — `--confirmatory` does not lock duration/speed/obstacle/steps**;
   pin them in the confirmatory branch. Receipt also omits duration/speed/T/route.

### Latent footguns (no planned run triggers them)

10. **[low, verified real] `loop.py:254` — resample ladder shares the flat seed space**; two
    top-level seeds 10 000 apart silently duplicate clips across arms. Guard: assert no two
    requested seeds differ by a multiple of the stride, or hash-derive samples i>0.
11. **[low] `fit_response.collect()` globs the cache without checking `cache_version`** — v1
    clips would leak into any future response refit. (Evidence lens rated the *current* fit
    unaffected; guard before refitting.)
12. **[low] `runner.py:73` — the `randn` interceptor matches on leading dim only.** Sole
    generation-time `randn` today is `ardy_model.py:481` (verified), but pin the shape or
    count interceptions to survive an ARDY upgrade.
13. **[low] `cache.py:27` — `NOISE_STREAM_VERSION` is not part of the key preimage**; the
    v1→v2 invalidation worked only because both constants were bumped together. Fold it in.
14. **[low] `demo/ardy_runner.py:104` — the demo's learned/optimized layers still generate
    the unused path-reference clip** (the fix landed only in `verify/loop.py`).
15. **[low] `loop.py:101` — `LoopResult.ardy_calls` (executed) vs `to_dict()['ardy_calls']`
    (logical) share a name**; rename the field.
16. **[low] `loop.py:102` — `legacy_ardy_calls` is fabricated for resample rows** (no v1 ever
    ran them); emit only for repair mode.
17. **[low] `experiment_architecture.py:272` — repair_stats include steps whose product was
    never generated under `--no-generate`.**
18. **[low] `loop.py:190` — unverified path rebuilds `q_final` from 4-decimal-rounded dips.**
19. **[low] `semantic_scaffold.py:94` — contact split hardcodes 4 columns but accepts ≥4**
    (SOMA's 6-channel layout would silently corrupt L/R).
20. **[low] `semantic_scaffold.py:208` — transplanted Y is re-based on target root_y**, mixing
    donor foot height with pelvis delta; use donor absolute Y for feet.
21. **[low] docs link untracked files; commit the changeset atomically.**

### Refuted by the verification pass (no action)

- "v1 clips leak into the τ fit **as run**" (the committed response.json predates v2 but was
  fit from exp001d rows, not the glob) — the *guard* in 11 is still worth adding.
- "Box-probe geometry incommensurable with exp001c" (depth/cap differ by design and are
  recorded in metadata).
- "48-motion/24-env two-loop SONIC path untested ⇒ broken" (bookkeeping verified correct by
  reading the batch-boundary code; still worth the cheap 2-motion validation run).

**Bottom line:** phase4e is safe to run as-is with ordinary seed lists; 1B needs fix 1 (one
line) first; exp016 needs fixes 3–5 plus the calibration tool before its pilot means anything.
