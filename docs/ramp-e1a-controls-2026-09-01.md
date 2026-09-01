# E1a controls: the packet is a style cue, and it is worse than the prompt alone

Four analyses on top of the completed E1a campaign
(`docs/ramp-e1a-result-2026-09-01.md`). Three are replays of the archived 24 clips with
no new evidence (arm clips verified byte-identical, 16/16); the fourth spends 8 samples on
the matched control the first three imply. Together they turn E1a's null into a mechanism.

## 1. Placement does not respond to the request

`experiments/analyze_e1a_placement.py` →
`outputs/exp019_gait_matched_stepover_v7/placement_analysis.json`.

* **The signed error is not consistently signed.** Absolute: 4 lifts after the obstacle,
  4 before, mean −0.61 m, sd **2.43 m**. Residual: 2 after, 6 before, mean −1.42 m, sd
  **1.66 m**. No open-loop lead correction exists to be found.
* **Gain ≈ 0.** Regressing realized lift position on requested obstacle position gives
  slope **−0.44** (absolute, r = −0.32) and **+0.02** (residual, r = +0.02) where perfect
  placement is 1.0.
* **The lift belongs to the seed, not the request.** Absolute and residual land within
  0.25 m of each other in **5/8** seeds (median difference 0.07 m), despite carrying
  different payloads.
* **Tolerance curve** over 16 packet clips: 0/16 within 0.25 m, 1/16 within 0.5 m, 3/16
  within 1.0 m. Any clearance at the obstacle: **0/16**, Wilson95 upper bound 0.19 —
  implying best-of-N ≥ 11 even at that optimistic bound.
* **Not stride-quantized.** Mean distance to the nearest integer stride is 0.277 against
  0.25 for a uniform distribution, so placement is not living on a discrete lattice
  either — selection is not being rescued by structure.

## 2. The command is never satisfied — not at the window, not at any lag

`analyze_e1a_constraint_residual.py` and `analyze_e1a_response_lag.py`.

Compliance is `1 − ‖realized − commanded‖ / ‖nominal − commanded‖`, measured at exactly
the constrained (frame, joint) pairs: 1.0 means the command was met, 0.0 means the clip
stayed at the nominal, negative means it moved away.

| channel | commanded change | residual error | compliance |
|---|---:|---:|---:|
| rotations, absolute arm | 0.285 rad | 0.572 rad | **−0.26** |
| rotations, residual arm | 0.267 rad | 0.609 rad | **−0.44** |
| root height, absolute | 0.023 m | 0.024 m | +0.20 |
| root height, residual | 0.022 m | 0.024 m | +0.14 |

The realized clips end **further from the commanded rotations than the nominal was**. Root
height — the project's one historically clean channel — is weakly positive on a request
(2.3 cm) at the tracking noise floor, so it carries almost no information here.

The obvious rescue is that the decoder honors the packet *late*. It does not: sweeping the
comparison window over lags −40…+80 frames, **0/16 clips match the command better than
the nominal did at any lag**. Best lag is median +1 frame, and it correlates *negatively*
with the observed placement error (r = −0.48) where a single delayed-execution mechanism
predicts +1.

So the packet is not late, and not partially honored. It is read as a soft cue the
decoder re-plans around.

## 3. The world cannot be anchored by shifting the start

If the prior were translation-equivariant along a straight route, a lift measured at `s`
could be made to coincide with an obstacle at `s*` by starting `Δ = s* − s` earlier. It is
not: offsetting the route by 0.5 m and 1.3 m gives max |qpos − shifted(qpos)| of 1.4–3.5,
and the realized start stays near the origin while the endpoint moves. **A route offset is
absorbed as a speed change, not a translation** — which is the same mechanism that made
exp018's route warp re-plan the gait.

## 4. The control that matters: the packet is worse than the prompt alone

`experiments/exp020_text_only_control.py` (8 samples) generates the same eight seeds,
strata and routes under the same STEP prompt with **route conditioning only** — no packet,
no rotation channel, no root-height channel — and scores the identical endpoint vector.

| arm | elicits a lift | Wilson95 | mean lift | max lift | clears the obstacle |
|---|---:|---|---:|---:|---:|
| nominal (WALK) | 3/8 | [0.14, 0.69] | 0.0019 m | 0.0050 m | 0/8 |
| **text only (STEP)** | **7/8** | [0.53, 0.98] | **0.1213 m** | **0.2118 m** | **3/8** |
| STEP + absolute packet | 8/8 | [0.68, 1.00] | 0.0721 m | 0.1377 m | 0/8 |
| STEP + residual packet | 8/8 | [0.68, 1.00] | 0.0609 m | 0.1439 m | 0/8 |

Paired per seed, the text-only arm beats the packet arms on amplitude by a median
**+6.0 cm** (absolute, 6/8 seeds) and **+6.8 cm** (residual, 6/8). Its three clearances
are 0.067, 0.067 and 0.073 m — it clears a **5 cm** box in 3/8 scenes where both packet
arms clear 0/8, though none clears the 8 cm obstacle those scenes actually specified.

**The coherent packet is not neutral; it is harmful.** It adds elicitation the prompt
already provides (7/8 → 8/8, well inside each other's intervals), halves the amplitude,
and removes the prompt's only successes. This is consistent with the compliance
measurement: the packet's rotation targets sit off the prior's own coordinated manifold,
and the decoder's compromise between them and its learned step-over is an attenuated,
displaced motion.

Placement stays bad for text too (mean |error| 2.30 m), so this does not rescue placement
— it relocates the whole question. The prompt is the better elicitor; nothing tried
places.

**Scope of this control.** It isolates *packet versus no packet*, not *rotations versus
root height*: the packet arms add both a full-body rotation channel and a root-height
channel over the text-only arm's route conditioning, so the attenuation cannot be
attributed to one of them from these eight seeds alone. The compliance measurement points
at the rotations (negative, on a 0.27–0.29 rad request) rather than the root height
(weakly positive, on a 0.02 m request), but a channel-ablation arm would be needed to
close that. The text-only arm's progress ratio is 0.997, so it is not winning by
degenerating into a different motion. Eight paired seeds, one donor bundle, one box
geometry, kinematic stage only — descriptive, no interval on the difference itself.

## What this settles for E2

The guidance's E2 ordering asked for a placement-response gain measurement first and a
re-anchoring fixed-point loop second. Both are answered here at zero additional GPU cost,
and both are negative: the gain is zero, and re-anchoring presumes a command the decoder
honors somewhere, which the lag sweep rules out. Latent optimization is not indicated
either — the objective is not smooth in a channel whose command is never satisfied.

What survives is **measure-then-select**: generate under the prompt, measure where the
lift lands and how high it is, and choose. exp021 measures the joint distribution of
(lift position, lift height) over 64 text-conditioned clips to size that method — the
per-clip hit rate against a scene-specified obstacle, and the best-of-N curve it implies,
at graded box heights. The operating envelope this work should target is 5–7 cm, not the
8 cm the earlier scenes assumed.

Two harness consequences, both adopted: the box-height endpoint is now reported **graded**
(0.03–0.30 m) rather than as pass/fail against one demanding obstacle, and the tracker
belongs in the endpoint before any 5–7 cm claim is made, because the kinematic endpoint is
ambiguous at that amplitude.
