# EXP-026 result — the reference screen ranks duck-family cutoffs too, weakly; speed does not

**Run:** 2026-09-03, CPU only, 17.4 s (`outputs/analysis_duck_contract/`, receipt status
`complete`, 526/526 clips scored, protocol sha bound, clean worktree at `43019b5`). Protocol
`docs/ramp-exp026-duck-contract-protocol.md` (preregistered `f52ba5c`, amended `8291721` before
any feature was computed). Driver `experiments/analyze_duck_contract.py`. No new samples, no GPU.

## 1. What was asked

The reference screen — reject a reference whose longest bilateral no-support run exceeds 0.20 s —
predicts the controller evaluator's stopping rule on the **step family** through two actuation
channels (prompt-elicited, AUC 0.997; position-channel ladder, 0.92). Both are motions that leave
the ground, so the screen could be an artefact of stepping. EXP-026 asks whether it also ranks
cutoffs in the **duck family**, whose references are crouches produced by a different pipeline
(planner → schedule → repair loop), against the confound the plan named in advance: EXP-1B's
14 s clip cap forced reference speeds up to ~2.3 m/s, and speed alone might explain the cutoffs.

Corpus: the 526 unique EXP-1B duck references, first rollout per clip, 344 terminated, 36 scenes.

## 2. Result

**Pooled AUC** (cluster bootstrap over the 36 scenes; higher value predicts a cutoff, direction
fixed in advance and never flipped):

| group | primary | AUC | 95 % cluster bootstrap |
|---|---|---|---|
| **contact** | longest no-support run | **0.674** | 0.625 – 0.721 |
| crouch depth | requested peak dip | 0.707 | 0.632 – 0.770 |
| **speed** (the confound) | max root planar speed | **0.441** | 0.328 – 0.552 |

**Within scene** (route length, beam geometry and gap constant; 13 of 36 scenes have ≥ 5 of each
outcome; 1,144 terminated × surviving pairs): contact **0.694**, speed 0.601, crouch **0.565**.

**The preregistered rule is satisfied on both measures.** Contact − speed is **+0.233
(CI 0.120 – 0.362)** pooled and **+0.092 (CI −0.018 – 0.210)** within scene, so the verdict is
`contract_transfers_to_the_duck_family`; the within-scene margin is positive but its interval
includes zero, so the *ordering* is established pooled and only suggested once the between-scene
confound is removed.

**Every stratum agrees about contact, and only about contact.** Contact is the one primary above
0.5 in all seven preregistered strata:

| stratum | n | terminated | contact | crouch | speed |
|---|---|---|---|---|---|
| dip [0.25, 0.35) | 152 | 66 | 0.651 | 0.484 | 0.448 |
| dip [0.35, 0.45) | 138 | 84 | 0.578 | 0.454 | 0.350 |
| dip [0.45, 0.50] | 236 | 194 | 0.693 | 0.624 | 0.408 |
| 3 beams | 88 | 68 | 0.664 | 0.792 | 0.583 |
| 4 beams | 124 | 84 | 0.663 | 0.629 | 0.501 |
| 5 beams | 154 | 97 | 0.670 | 0.685 | 0.437 |
| 6 beams | 160 | 95 | 0.707 | 0.785 | 0.548 |

**The confound the plan feared is not there.** Speed does not predict duck cutoffs: its pooled
AUC is 0.441 with an interval spanning chance, it is at or below 0.5 in five of the seven strata,
and the surviving references are *faster* than the terminated ones (median max root planar speed
1.87 vs 1.75 m/s). The 14 s cap made every reference fast; it did not decide which were cut off.

**Crouch depth is mostly a between-scene effect.** It is the best pooled predictor (0.707) but
falls to 0.565 within scene and to 0.45–0.62 inside two of the three dip bins — deeper crouches
were requested in harder scenes, and the pooled number partly reads the scene, not the reference.

**Contact is not a repackaged speed.** Pearson correlation between the primaries: contact vs
speed **−0.18**, contact vs crouch +0.20, speed vs crouch +0.19.

**The screen as a duck accept/reject rule: no.** At its calibrated 0.20 s threshold it flags
313/344 cutoffs (sensitivity 0.910, CI 0.875–0.936) but also 128/182 survivors (specificity
0.297, CI 0.235–0.367); at the step family's post hoc 0.28 s cut, 0.860 / 0.346. It ranks, but at
this threshold it would reject most of the corpus: **441 of 526 duck references (84 %) have a
no-support run longer than 0.20 s**, and every one of the 526 has some no-support run. The
crouch pipeline produces floats too.

Median longest no-support run: 0.82 s for terminated references (q10–q90 0.24–2.51), 0.44 s for
survivors (0.08–1.60).

## 3. What this licenses, and what it does not

- **Licensed.** The screen is not an artefact of stepping. In a second behaviour family, produced
  by a different pipeline and a different actuation channel, the longest no-support run ranks the
  controller's cutoffs above chance in every stratum, beats the speed confound pooled, and is
  uncorrelated with it. Together with the two step-family channels this makes the screen a
  property of *this controller's evaluator applied to references*, not of one behaviour.
- **Not licensed.** It is far weaker here than on the step family (0.674 vs 0.997) and cannot be
  used as a duck-family accept/reject rule at its calibrated threshold. Nothing here says a duck
  reference that passes the screen is trackable: EXP-1B's endpoint was 0/859 traversals.
- **Scope.** Post hoc on a completed campaign (the outcome labels were public in REPORT §25; the
  groups, primaries, direction and rule were fixed beforehand). One physics seed, one rollout per
  clip, schema-v1 archives, one tracker checkpoint, 36 scenes of one beam family. EXP-1B's 14 s
  clip cap is a property of that campaign, not of ducking, so a weak transfer here bounds this
  corpus rather than the behaviour.

## 4. Ledger

`outputs/analysis_duck_contract/{receipt.json, rows.jsonl}` — one row per clip with its
provenance hashes, committed row fields and the full feature vector. Inputs bound by hash: the
EXP-1B rows, the exp016 threshold receipt, the protocol, and every source file.
