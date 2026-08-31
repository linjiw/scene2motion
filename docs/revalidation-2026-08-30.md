# Sampling revalidation ledger — 2026-08-30

## Finding

`ArdyRunner.generate(seeds=[...])` was introduced to make a sample's noise independent of its
position in a batch. The legacy implementation created and seeded a new `torch.Generator` on
every intercepted `torch.randn` call. ARDY draws a new initial latent for each 52-frame
autoregressive window, so a long sample received the same latent row again at every window.

The seed itself was honored and results are reproducible. The error is that its stream did not
advance. This is not the intended ARDY sampling process.

## Scope

- Long clips generated with the `seeds=` argument under noise stream v1 are affected.
- Calls using ARDY's ordinary `seed=` path advanced the global stream and are not affected by
  this specific defect.
- Clips no longer than one generation horizon have only one latent draw and are not affected
  by cross-window repetition.
- A within-v1 comparison can still diagnose the behavior of that controlled sampler, but it
  cannot establish an effect for the intended ARDY sampler.
- v1 and v2 clips must never be pooled in one estimate.

## Correction

`scene2motion.runner.NOISE_STREAM_VERSION = 2` keeps one persistent generator per sample for
the entire generation context. Each sample remains invariant to batch position, while later
autoregressive windows receive later draws from its own stream. Tests establish both
properties.

`scene2motion.demo.cache.CACHE_VERSION = 2` invalidates every legacy cached clip. New receipts
record the noise-stream version; source and generation identity are also frozen by the Phase 4E
architecture experiment.

The Phase 4 generator also no longer creates its unused path-reference clip. Each attempt now
costs one candidate-producing ARDY invocation; receipts keep `legacy_ardy_calls = 2 * attempts`
only so old tables remain interpretable.

Historical artifacts are not deleted or silently rewritten. EXP-015b and cache inventories
label their v1 source explicitly.

## Locked rerun order

1. Run the proposer × feedback matrix under v2: heuristic/QP/TCN × 0/1/2 repairs, plus
   unchanged-proposal resampling controls at equal generation budgets.
2. Run EXP-016 under v2 with disjoint donor/evaluation seeds, seam-stratified obstacle
   placements, exact foot crossing/support gates, and achieved-state SONIC replay.
3. Re-run the minimal capability-audit cells needed to verify the calibrated 6× counting
   result; expand only if a key classification changes.
4. Promote numerical statements back to headline status only after receipts identify v2 and
   no legacy cache entry was used.

## Current claim status

- The software/methodological finding that output must be verified remains valid.
- The old Phase 4 effect size (0.750 → 1.000 on 36 scenes) is a v1 result awaiting replication.
- The EXP-015 text-versus-position result is exploratory v1 evidence and now motivates, but
  does not answer, EXP-016.
- The 85.6 mm / 23.5% execution-margin proxy is withdrawn as a certificate: no achieved qpos
  exist, its body subset was misidentified, and it double-counted `BODY_MARGIN`. See
  [`exec-gate-audit.md`](exec-gate-audit.md).
- Geometry-only findings and the discovery of the sampling defect itself are unaffected.
