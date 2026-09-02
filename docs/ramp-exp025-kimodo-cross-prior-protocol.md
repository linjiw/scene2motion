# EXP-025 protocol — cross-prior timing and contract on Kimodo-G1 (DRAFT, not yet preregistered)

**Status:** draft written 2026-09-01 (evening) from the plan of record
(`docs/plan-2026-09-01-icra2027.md` §3). Becomes preregistered when committed with
`Status: preregistered` and its sha256 bound into the receipt before the first sample. Seeds
**4700–4763** (Kimodo per-sample stream) are reserved; the reduced-audit rerun keeps its original
seeds 100–105 / 200–205 because it is a replication.

## Question and scope

exp021 and exp023 established, for the autoregressive ARDY-G1 prior, that the STEP prompt's
behaviour is emitted once, early, tied to the rollout origin, and that it is a non-ballistic float
(both feet outside the calibrated support envelope for far longer than a ballistic flight of the
pelvis rise). Kimodo-G1-RP-v1
is a released **offline (non-autoregressive)** diffusion prior for the same skeleton. Running the
same measurement on it separates two readings of the ARDY result:

- if Kimodo's elicited event is also early-concentrated and also a float, the timing and contract
  findings are properties of released text-conditioned G1 priors more broadly;
- if Kimodo's event-time distribution is broad (or prompt-driven), "tied to the rollout origin"
  is an autoregressive-rollout property and the paper scopes it as such.

Either outcome is reportable; no prediction is made about which. A second, cheaper part reruns
the 84-clip reduced capability audit whose 4.5× overcount row is currently transcript-sourced
(`docs/kimodo-provenance-2026-08-31.md`), so that row can be cited from a committed receipt.

## Prerequisites (engineering, before preregistration)

1. Recover `kimodo_runner.py`, `smoke_kimodo.py`, `kimodo_reduced_audit.py` and `NOTES.md` from
   the session transcript `~/.claude/projects/-home-linjiw-ardy/f4440d67-ed27-4331-be07-dc169754a80c.jsonl`
   (subagent `agent-ac06b53618f8a2379`: Writes L93/L95/L102/L181, Edits L98/L100/L110/L112/L122/
   L190/L192/L194, sed edits L118/L200/L203, applied in transcript order); commit them under
   `experiments/kimodo/`; `--selftest` must print `3.0/3.0/3.0/2.0`. (~2 h)
2. **No CPU encoder is needed.** ARDY and Kimodo use the identical LLM2Vec preset
   (McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp + -mntp-supervised, llm_dim 4096) through
   wrappers that differ only in docstring and device helper, and both caches key entries by
   `sha1(prompt text)`. Copy ARDY's cached STEP vector (`outputs/text_cache.npz`, (1, 4096)
   float32) into a campaign-local Kimodo cache under the same key; bind both content hashes and
   assert the wrapper-source equality in the receipt. (~15 min) The CPU encoder is reserved for
   EXP-027 only and must never run beside the co-tenant GPU job (it needs ≥ 18 GB available RAM).
3. Verify the G1Body path on exported Kimodo qpos (the donor screen on 2026-08-31 already ran
   `G1Body` on this corpus under `$S2M_PY`; `g1.xml` md5 identical in both checkouts). (~30 min)

## Locked generation design (part A: timing and contract)

Kimodo-G1-RP-v1 (HF snapshot `3020ad8c…`), 30 fps, 100 DDIM steps, cfg (2.0, 2.0) separated,
first heading 0, `post_processing=False`, per-sample noise (the recovered runner's
`_per_sample_noise` port; NOISE_STREAM_VERSION 2). Prompt STEP. Route: straight, 0.9 m/s, 8.0 s
→ 240 frames, 7.2 m; `root2d` dense path constraint only (Kimodo's analogue of ARDY's `root_xz`),
root height and heading free, matching the exp021 `free` contract. Seeds 4700–4763 (64) in
batches of 8 (Kimodo throughput ≈ 2 s/clip at B = 6–8). Export to MuJoCo qpos (Z-up, g1.xml
joint order, the indoor_nav CSV convention) and archive every clip.

Control arm: WALK prompt (`A person walks forward at a steady pace.`, already cached) on the
same 64 seeds — the free nominal arm (elicitation floor).

## Endpoints (kinematic only; no SONIC in this campaign)

Per clip, planned denominators: elicitation (whole-body-clearable lift ≥ 0.03 m anywhere,
`box_height_profile` at 120 points through `G1Body`); lift position in metres and time in
seconds (30 fps; report seconds, never frames, when comparing to ARDY's 25 fps); exact
clearance at x = 1.2 m and 3.6 m, graded heights; contract features from
`analyze_trackability_contract.features` (fps 30; support thresholds in seconds are fps-free);
route fidelity.

Summary endpoints: elicitation rate with Wilson 95 %; lift-time distribution (median, q10–q90;
fraction inside the first 2.0 s — the ARDY comparison window is "80–86 % inside 2.0 s");
per-target exact hit rates at 1.2 m and 3.6 m; fraction of elicited clips with
`max_unsupported_run_s > 0.20 s` (the calibrated gate; float fraction; ARDY: 44/44 lifting clips
above it), with the post hoc 0.32 s cut reported alongside.

## Part B: reduced capability audit rerun — OUT OF SCOPE for ICRA 2027

The 84-clip reduced audit (transcript-sourced 8/9/9/2, 4.5×) is **not used** by the paper and
is marked "not used" in the claims ledger; the 6× / 4.5× counting rows are demoted to a
motivating remark at most. If time remains after the freeze, the recovered
`kimodo_reduced_audit.py --steps 100` may be rerun (≈ 3 GPU-min) for the technical report, with
the joint-point-descriptor / scene-free-validity caveat. It is not a prerequisite for Part A.

## Decision rules

- The timing statement in the paper is written from the measured distribution: "early and
  rollout-tied" is claimed only for ARDY unless Kimodo's first-2 s fraction is also ≥ 0.7; if
  Kimodo's fraction is ≤ 0.4 the paper attributes the ARDY window to autoregressive rollout
  context; in between, the paper reports both distributions without a mechanism claim.
- The contract statement generalises to "released G1 priors" only if ≥ 80 % of Kimodo's
  elicited clips are also floats by the calibrated 0.20 s gate; otherwise it stays ARDY-scoped.
- No arm expansion after seeing outcomes.

## Gates

Clean worktree; empty output directory; both prompt embeddings' content hashes bound (STEP copied
from ARDY's cache, WALK from Kimodo's); Kimodo snapshot and runner source hashes bound and
revalidated; 128/128 exact accounting; every clip archived before scoring; rows written before
summary; host-resource gate (≥ 12 GB free VRAM, ≥ 18 GB available RAM, no concurrent Isaac job)
recorded in the receipt.

## Budget

Part A: 128 clips ≈ 5 GPU-min at ~2 s/clip plus model load; scoring ≈ 128 × 4 s ≈ 9 min CPU.
No encoder. Total < 10 GPU-minutes. Becomes a **must** if the runner recovery lands by Sep 4
noon, because it is the only experiment that answers the "characterisation of one released
model" objection.

## Risks

- Recovery effort is the real cost (~half a day including tests); if it slips past Sep 4, drop
  EXP-025 and state cross-prior generality as a limitation.
- Kimodo's `root2d` constraint following (mean 0.09 m on the indoor_nav corpus) is looser than
  ARDY's 1.4 cm; report route MAE and do not over-interpret a shifted lift position.
- RAM: the CPU encoder cannot run beside the ARDY GPU job's host memory; schedule it alone.
