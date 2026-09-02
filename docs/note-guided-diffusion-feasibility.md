# Note — test-time guided diffusion on the frozen ARDY-G1 prior: feasibility (Discussion / future-work bridge)

**Written:** 2026-09-02, read-only audit of `/home/linjiw/ardy` at `693f74d` (snapshot `059b8007…`)
and `scene2motion/runner.py`; no GPU run. Numbers are from committed receipts, configs or
safetensors headers unless labelled *estimate*; unfound code facts are *not verified*.

## 1. What the sampler actually does

- **Steps and evaluations.** `num_base_steps: 10` (`config.yaml`); `diffusion_steps=5` runs indices
  `[4..0]` (`ardy/model/ardy_model.py:622, 737`), one `denoising_step` each (`:492-516`). Each step is
  **one** denoiser call on a **3B** batch: `cfg_type: separated` stacks text-only / constraint-only /
  unconditional rows and mixes them with weights (2, 2) (`ardy/model/cfg.py:74-78, 129, 148-149`;
  `ardy_model.py:272-277`). One Horizon52 window = 5 network evaluations = 15 B-row forwards.
- **What is predicted.** The clean sample **x0** (`token_seq_pred_clean`, `ardy_model.py:280`); DDIM
  recovers ε from it (`ardy/model/diffusion.py:128-132`), η = 0, so the chain is **deterministic
  given the initial noise** `torch.randn(B, 13, 148)` (`ardy_model.py:476-481`; 13 tokens × 4 frames;
  148 = 20 explicit root values + 128 body latents).
- **Two stages.** Stage 1 predicts the explicit root (position + heading, 5/frame), stage 2 the body
  latent given the local root (`auto_latent_twostage_denoiser.py:41-42, 270-279, 366-375`); NVIDIA's
  comment at `:300-310` reads "At test-time want to allow gradient through for guidance". Denoiser
  157.9 M fp32 params (root and body blocks 73.5 M each, 8 layers, d = 1024), tokenizer 35.8 M
  (safetensors headers); ≤ 52 motion + 3 prefix tokens per call.
- **Decoder.** hybrid → explicit: `global_root_to_local_root` (torch, `motion_rep/reps/base.py:103-146`)
  → `detokenize` (`autoencoder/fsq.py:215-255`) → 414-D frames (3+2+99+204+102+4,
  `reps/ardy_motionrep.py:57-64`; exp023 `features.npz` rows are (208, 414)) → `inverse`
  (`ardy_motionrep.py:238-282`): cont6d → matrices → local rotations → **torch FK**
  (`skeleton/kinematics.py:120-171`, jit-scripted, 34 joints) → `to_qpos` in torch `atan2`
  (`exports/mujoco.py:186-251`), converted to numpy by the runner (`runner.py:509-516`,
  `exports/mujoco.py:164-166`). **After qpos everything is MuJoCo/numpy**: contact distances
  (`scene2motion/robot.py:252-280`), `BoxHeightProbe.clears` (`stepover_eval.py:131`), foot bottom
  clearance and planar speed (`stepover_eval.py:173-222`).
- **Blockers.** The denoiser call is under `torch.inference_mode()` (`ardy_model.py:279`) inside
  `no_grad` loops (`:494`; `runner.py:256, 377`); the body latent is rounded to a 64-level FSQ grid
  with straight-through gradient at decode (`fsq.py:23-26, 236-237`) and re-quantised at each
  window end (`ardy_model.py:789-790, 658-660`). Guidance means overriding `denoising_step`
  (~80 lines, weights untouched); the decoded body is **piecewise constant** in its latent
  (sub-cell steps change nothing) while the root channel is continuous.

## 2. Measured latencies (RTX 5080, 16 GB)

| quantity | value | source |
|---|---|---|
| 64 clips generated + scored (v2) | 462.8 s (7.2 s/clip; scoring dominates) | `outputs/exp021_elicited_lift_distribution_v2/receipt.json` |
| 32 schedule samples (4 B=8 calls × 4 windows) incl. load, decode, first rows | 60.4 s | exp023 receipt; result doc l.65-66 |
| one B=8 call: peak CUDA reserved / host RSS | 1,076 MiB / 2,297 MiB | EXP-024 protocol l.176-178 |
| model load / one B=8 four-window schedule call / decode | 2.2 s / 0.7 s / 0.1 s | 2026-09-02 session probe beside a co-tenant; **uncommitted** |
| SONIC, two launches for 64 clips | 38.7 s each; 139.6 s total | `outputs/exp022_exact_tracking_bridge/receipt.json` |

From the probe, one window ≈ 0.175 s at B=8, ≈ 35 ms per denoiser call (*estimate*). One window is
52 frames = **2.08 s of motion**. The "0.4 s" figure is in no receipt or README we hold (the ARDY
README names TensorRT / torch.compile modes without numbers); read as B=1 per-window compute it is
0.19× real time (*not verified*) — a compute latency, not the 0.8–3 s prompt-to-behaviour latency
EXP-023/023b measure.

## 3. Q1 — is analytic collision guidance tractable interactively?

**Cost (estimates).** (a) *x0-guidance*: per step decode x0 → torch FK → box SDF on a
sphere/capsule set, backward through decoder + FK only: +20–40 % per window. (b) *Full guidance*
∇ₓₜ through the denoiser: backward ≈ 2× forward, ≈ 3× per step → ≈ 0.5 s/window at B=8, ≈ 2.1 s
per four-window call; activations for 24 rows × 55 tokens × 16 layers are small beside 1,076 MiB.
(c) *Noise optimisation* (DNO-style, K passes of the 5-step chain): K = 50 → 25 s/window, offline
only. Under (b) a window still costs less than the 2.08 s it produces, so one-window-lookahead
streaming stays plausible; (c) does not.

**What must be re-implemented.** Not FK — ARDY's `fk` is torch. Missing is a torch surrogate of the
MuJoCo collision model: Unitree's primitive geoms (offsets from `g1.xml`, sha `5d76cf92…`) on the FK
frames, the 4 cm `BODY_MARGIN`, a box SDF, foot bottom clearance and planar speed. Whether ARDY
joint frames match MuJoCo body frames up to `_rot_offsets` (`exports/mujoco.py:131-147`) must be
tested against `G1Body.body_points` (*not verified*); autograd through the jit FK and the STE
decoder is expected but untested.

## 4. Q2 — gradient guidance vs the external verify→repair loop

| | gradient guidance | verify→repair (Δq = e/\|g′\|, 3τ·v lead, bounded, refuses) |
|---|---|---|
| information | analytic gradient of a surrogate (SDF on primitives) at 5 steps | measured deficit from the exact MuJoCo model, one scalar per attempt |
| can fix | reference geometry inside the prior's support, per window | overhead deficit on a native channel; 36 scenes (framing note Table 2) |
| cannot fix | trackability — it moves x_t, not the tracker; the float is a support-phase artefact | anything without a native scalar channel (the float: exp020, EXP-1C) |
| support term | soft only: sigmoid-relaxed foot height/speed; a run-length count has zero gradient | the 0.2 s gate applied after sampling, exact |
| guarantee | none; returns a clip silently | quantified refusal (26/300 pilot) |
| naturalness | weight vs prior fidelity; 5 steps give few correction points | monotone repairs; loses on margin for an already-right proposer |
| paper scope | outside the native interface: Discussion only, bounded by DNO, PRESTO, CLoSD, PhysDiff, BRIC/SafeFlow | inside; kinematic, multi-scene |

Two mechanics matter. A collision-only term has a cheaper continuous path than the quantised body:
raise the root — clearing clips already peak at 0.97 m vs 0.78 m for WALK — so it deepens the float.
A support term is definable (per foot s_t = σ((h_sup−h_t)/β)·σ((v_sup−v_t)/β) with the exp016
thresholds 0.0465 m / 1.18 m/s; penalise Π over 6-frame windows of (1−s_L)(1−s_R)) but pulls against
the swing foot: the joint objective is single-support stepping, which 0/44 lifting clips show.
Guidance would have to leave the prior's mode for this prompt, which a weight sweep can hide.

## 5. Q3 — a fair minimal test inside the audit (proposed EXP-029)

Fresh seeds 4950–4981 (32; disjoint from all spent/reserved), preregistered, v2 sampler, clean
worktree, STEP prompt, exp021 route contract. Paired-noise arms, one B=8 call per two seeds:
`free`; `guide_col` (x0-guidance, box SDF at the fixed centre x = 1.2 m; one weight and β fixed a
priori); `guide_col_sup` (adds the soft support term); probes: weight 0 must be byte-identical to
`free`, a box behind the start must leave clearance unchanged. Endpoints: (1) exact
`BoxHeightProbe(1.2, 0.20).clears` at 3/5/8 cm; (2) the calibrated 0.2 s gate, longest no-support
run, ballistic ratio, predictions committed before SONIC; (3) SONIC guarded retention under the
release evaluator plus EXP-028's termination-free class; `motion_metrics` skating/jerk for
naturalness. Budget: 96 clips, 12 calls (*estimate* ≤ 1 GPU-min), 3 SONIC launches (≈ 2–5 GPU-min),
scoring ≈ 11 CPU-min. **Kill conditions:** probe failure → harness defect, fresh seeds; `guide_col`
5 cm clearance ≤ `free` (paired, Wilson) → stop; clearance up but median run longer or gate 0/N →
"clearance bought with float", stop; `guide_col_sup` ≥ 3/32 gate-pass ∧ exact-clear → SONIC, where
0/N retention falsifies the gate in that regime and is reported. No re-tuning on the seeds.

**Not verified:** B=1 per-window time; origin of 0.4 s; TRT/compile gains; frame coincidence for
primitive attachment; backward memory; informativeness of STE gradients; the probe timings.
