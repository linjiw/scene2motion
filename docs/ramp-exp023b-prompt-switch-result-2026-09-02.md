# EXP-023b result — prompt-switch positive control (refused at the substrate gate; delayed STEP replicates at 3/8)

**Run:** 2026-09-02 07:35–07:39 EDT, `outputs/exp023b_prompt_switch_control/` (receipt status
`refused`, `refusal_reason = squeeze0_substrate_gate_failed`; 32/32 samples returned, converted
and analysed; wall clock 268 s; protocol sha `6fffe214…` bound; code commit `9fd70a8`, clean;
host gate at launch 7,878 MiB free VRAM / 13,624 MiB available RAM beside two co-tenant Isaac
processes, recorded not gated). Seeds **4640–4647 are spent**. Protocol:
`docs/ramp-exp023b-prompt-switch-positive-control-protocol.md` (preregistered before the first
sample). Driver `experiments/exp023b_prompt_switch_control.py`.

## 1. Gates

| gate | rule | observed | pass |
|---|---|---|---|
| substrate | `squeeze_0` sidestep composite ≥ 4/8 | **0/8** (Wilson 0–0.32) | **no → refused** |
| all_walk sidestep specificity | ≤ 1/8 seeds with any signature in either window | 0/8 | yes |
| all_walk step specificity | ≤ 1/8 seeds with any step event | 0/8 | yes |

The three sidestep signatures never fired in any arm: maximum heading deviation from the route
6–28°, maximum lateral root excursion ≤ 0.07 m, feet never crossed (minimum pelvis-aligned
separation ≥ 0.002 m, positive). **Under the dense `root_xz` route constraint the SQUEEZE prompt
does not produce a sidle, a lateral excursion or a foot crossing on this route, so it is not a
usable sidestep positive control here; the campaign is inconclusive for the question it was
built to answer** (protocol, first decision rule). The decision-rule verdict in the receipt is
therefore marked invalid.

## 2. Outcomes (planned denominators; all arms traversed the route, progress ratio 0.996–1.014, route MAE 0.008–0.048 m)

| arm | step event in the 96-frame post-onset window (≥ 3 cm whole-body-clearable lift, two-foot traversal) | latencies (frames) and clearances | exact box at the predicted centre (onset + 34 frames) |
|---|---|---|---|
| `all_walk` | 0/8 in both control windows | — | 0/8 at every height (both windows) |
| `squeeze_0` | **5/8** (Wilson 0.30–0.86) | 16, 19, 37, 42, 81; 0.036–0.187 m | 2/8 at 3 cm and 5 cm, 0/8 at 8 cm (centre 1.23 m) |
| `squeeze_52` | 1/8 | 56 (frame 108); 0.082 m | 0/8 (centre 3.11 m) |
| `step_52` | **3/8** (Wilson 0.14–0.69) | 21, 45, 75 (frames 73, 97, 127); 0.039, 0.067, 0.104 m | 0/8 at every height (centre 3.11 m) |

Handoff-transmission statistic (RMS joint-angle difference over frames 52–147, descriptive):
`squeeze_0` vs `all_walk` median 0.34 rad (0.23–0.44); `squeeze_52` vs `all_walk` 0.14 rad
(0.11–0.23); `step_52` vs `all_walk` 0.20 rad (0.13–0.30); `step_52` vs `squeeze_52` 0.15 rad;
frames 0–51 of every delayed arm differ from `all_walk` by exactly 0 (fork audit passed, all
32 rows; corresponding-window noise shared across the four arms of each seed).

## 3. What this changes

1. **The SQUEEZE prompt is read but does not sidestep.** It elicits a whole-body-clearable lift
   in 5/8 from frame 0 (two of them clear a 5 cm box at the predicted centre) and moves the
   joints by 0.34 rad RMS relative to walking, with the root pinned to the route. "Steps sideways
   through a narrow gap" is executed by this prior as a lifting gait, not a sidle. The positive
   control has to be re-designed around a behaviour the prior can express with a pinned root
   (candidates: a crouch prompt, or SQUEEZE with the heading channel free), and needs the CPU
   encoder for any new phrasing.
2. **EXP-023's delayed-arm zero did not replicate.** `step_52` produced 3/8 events on fresh seeds
   (EXP-023: 0/8), latencies 0.84–3.0 s after the switch. Pooled over the two campaigns the
   delayed STEP prompt elicits the step in **3/16** (Wilson 0.07–0.43) against **6/8** from frame 0
   (0.41–0.93, EXP-023) and 5/8 for SQUEEZE from frame 0. Under the preregistered replication
   rule (`step_52 ≥ 2/8`) the abstract's "a prompt delivered late does not elicit the step" is
   **withdrawn**; the statement that survives is "a prompt delivered at frame 52 through the
   minimum-history handoff elicits the step less often (3/16 vs 6/8) and, when it does, 0.8–3 s
   after the switch; none of those clips cleared a box at the predicted centre (0/8 at any
   height), against 1/8 at 3 cm for `step_0` in EXP-023". "Tied to the rollout origin" is
   withdrawn as a mechanism claim.
3. **The handoff transmits prompts.** Both delayed prompts change the post-switch motion (0.14–
   0.20 rad RMS) and both produce events; attenuation is no longer the leading alternative for
   the (now smaller) delayed-prompt deficit.

Scope: Horizon52 at the released minimum history, one route, one scene, n = 8 per arm,
descriptive paired differences; kinematic only (no tracking). The three delayed-STEP events have
not yet been scored for the contract features; the A0 analyser will be run on this archive.

## 4. Ledger

- `outputs/exp023b_prompt_switch_control/{receipt.json, rows.jsonl, qpos.npz, features.npz, noise_audit.json}`, committed `90b6c5b`.
- Seeds 4640–4647 spent; no rerun on these seeds; a re-designed positive control needs fresh
  seeds and a new preregistration.
