# EXP-024 — kinematic stage landed (generation, scoring, predictions committed); SONIC stage pending

**Run:** 2026-09-02, generation 08:49 EDT (128/128 samples, seeds 4600–4631 spent, 16 B=8 calls,
ARDY-only host gate 7.9 GB free), scoring 08:55–09:31 (128 clips at ≈ 8 s/clip), predictions
09:47. Protocol `docs/ramp-exp024-reference-contract-protocol.md` (preregistered `976cd91`).
Ledger `outputs/exp024_reference_contract/` (commit `6f4d1ca`); **`predictions.jsonl`
(sha256 `18a2fb14…`, 128 rows, primary rule `run > 0.20 s`, secondary `run > 0.28 s`) is
committed at HEAD blob `51e1a5a` before any SONIC launch** (protocol item 3). The score stage
was blocked twice by provenance refusals unrelated to the data (an untracked agent-worktree
directory, then the driver's own resume path) and resumed through the documented path: both
blocked attempts are preserved beside the ledger and eight re-scored rows reproduced
byte-for-byte (`stages.score.resume_rescoring_verification.identical = true`).

## Reference endpoints per arm (planned denominators, 32 seeds each; Wilson 95 % in the receipt)

| arm | constructible (preregistered criterion) | elicits (lift ≥ 3 cm) | exact 5 cm at x = 1.2 m | flagged by the calibrated gate (`run > 0.20 s`) | flagged by the post hoc rule (`run > 0.28 s`) | contact-consistent (13-gate local step ∧ run ≤ 0.20 s) | median root peak (m) | median longest no-support run (s) |
|---|---|---|---|---|---|---|---|---|
| `free` | yes (no manipulation) | 23/32 (0.72) | 7/32 (0.22) | 26/32 | 24/32 | **0/32** | 0.955 | 0.48 |
| `pin_h` (heading 0) | yes (median heading range 7.8°) | 27/32 (0.84) | 8/32 (0.25) | 29/32 | 26/32 | **0/32** | 0.943 | 0.74 |
| `pin_y` (root 0.78 m) | yes (median root-height range 0.036 m) | 19/32 (0.59) | 1/32 (0.03) | 24/32 | 19/32 | **0/32** | 0.776 | 0.36 |
| `pin_yh` | yes (0.027 m; 7.6°) | 11/32 (0.34) | 2/32 (0.06) | 16/32 | 12/32 | **0/32** | 0.770 | 0.22 |

## What is already decided kinematically

- **P2 (replication of exp021 under `free`) passes**: elicitation 0.72 is inside the
  preregistered [0.55, 0.95] (exp021: 44/64 = 0.69 at the same ≥ 3 cm definition) and exact 5 cm
  clearance at the now-preregistered centre 0.22 is inside [0.06, 0.35] (exp021: 12/64 = 0.19).
- **P4 (prescriptive contract) is NO-GO in every arm before tracking**: no clip in any arm is
  contact-consistent, so none can be "contact-consistent ∧ exact-clear ∧ retained". Under all
  four native root contracts the STEP prompt elicits its lift inside a bilateral no-support run;
  pinning the root height shortens the run (median 0.48 → 0.36 / 0.22 s) and lowers the peak
  (0.955 → 0.78 m) and elicitation (0.72 → 0.59 / 0.34), but 16–24 of 32 clips are still flagged.
  Pinning the heading alone lengthens the run (0.74 s) and raises elicitation (0.84).
- **P3** needs the paired per-seed comparison (analysis stage) for the "shorter run in ≥ 20/32
  paired seeds" clause; the median-root-peak clause (≤ 0.85 m) holds for both pinned arms.
- **P1 (prospective gate test)** waits for SONIC: 95 flagged and 33 passed references across
  the four arms give the 2×2 its denominators. The predictions are frozen.

## Pending

Four SONIC launches of 32 (release evaluator, physics seed 0) and EXP-028's termination-free
pass on the same 128 clips; both require the SONIC host gate (≥ 12 GiB free VRAM, ≥ 18 GiB RAM,
no Isaac co-tenant), which the co-tenant lucid jobs have held all day.
