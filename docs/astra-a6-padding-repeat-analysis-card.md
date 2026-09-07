# A6 duplicate-reference execution diagnostic

September 6, 2026. **Post hoc development analysis**, specified after reading A8's
failed readiness gate and midpoint-departure diagnostic. This is not preregistered
confirmation, a new campaign, or an amendment of A6/A8.

Question: how much do existing executions of an identical reference differ across
simulator slots, even when their realized default encoder modes match? A6 group1
contains 16 scientific references and their 16 explicitly labelled padding duplicates
for each of eight variants. Inspect all 128 pairs. Keep the 112 native pairs and
16 free-WALK pairs separate. Padding never becomes new independent training data.

Verify archived assignment, input pickle, process, apparatus, achieved-state and
runtime-mode receipts before measurement. Compare slots i and i+16 only if they bind
the same source reference, scene, seed and command. Classify every pair by exact
runtime encoder-mask equality; different-mode pairs are a separate descriptive group.
Do not select pairs on outcomes or discard failures.

Use the existing A6 measurement without changing geometry, thresholds or censoring.
Report status/termination/descriptor disagreement over all pairs, and absolute numeric
differences only where both values are observed, with unavailable counts. Report each
scene separately and the mean of scene means, alongside pair medians and maxima.
Measure root-position difference in metres on the shared valid time prefix; report
both prefix lengths and never compare reset/padded states or fill a missing suffix.
No hypothesis test or calibrated error allowance is assigned to this diagnostic.

These are **different-slot** repeats: encoder identity does not fix simulator origin,
randomized initial state, observation noise or other slot-dependent effects. Their
differences cannot isolate numerical nondeterminism, same-slot repeatability, or the
cause of A8's response error. They can motivate a separate same-slot identical-input
replay before spending generation budget on smaller perturbations. No optimizer
expansion, contact-free label, present execution, or held-out seed is authorized by
this analysis. Save to a fresh output and preserve all source artifacts.
