# Kimodo-G1 runner adapter — audit notes

Draft adapter giving Scene2Motion the same `ArdyRunner` interface on top of the
Kimodo-G1-RP-v1 prior. Nothing generated yet (GPU busy); import checks pass under the
kimodo venv. All paths absolute; all signatures verified against source, cited file:line.

Files here:
- `kimodo_runner.py` — `KimodoRunner` (generate / to_qpos / encode / fps / model_name),
  `KimodoConstraintSet` (ConstraintSpec -> kimodo channels), `_per_sample_noise` v2 port,
  `build_conditions`, `channel_usage`, `load_constraint_spec_class`.
- `smoke_kimodo.py` — 2 cached prompts x straight-path spec x seeds [1234, 5678],
  ~4 s @ 30 fps, qpos export + sanity prints + npz dump. Heavy work is inside `main()`.

## Exact call signatures found (kimodo checkout /home/linjiw/kimodo)

- `load_model(modelname=None, device=None, eval_mode=True, default_family="Kimodo",
  return_resolved_name=False, text_encoder=None, text_encoder_fp32=False)` —
  kimodo/model/load_model.py:108. Passing a pre-built `text_encoder` OBJECT skips encoder
  selection entirely (load_model.py:194-215); `text_encoder=None` triggers
  `TEXT_ENCODER_MODE=auto` (API probe -> silent local Llama fallback, load_model.py:86-105)
  — the exact failure mode ArdyRunner guards against, so the runner always installs a
  guard object. `LOCAL_CACHE=true` makes checkpoint resolution try the local HF cache
  before going online (load_model.py:44-59). Name resolution:
  registry `resolve_model_name` kimodo/model/registry.py:359; `"Kimodo-G1-RP-v1"`
  resolves to short key `kimodo-g1-rp` (single version, registry.py:15-23).

- `Kimodo.__call__(prompts, num_frames, num_denoising_steps, multi_prompt=False,
  constraint_lst=[], cfg_weight=[2.0, 2.0], num_samples=None, cfg_type=None,
  return_numpy=False, first_heading_angle=None, num_transition_frames=5,
  post_processing=False, root_margin=0.04, progress_bar=tqdm)` —
  kimodo/model/kimodo_model.py:380. Does NOT accept precomputed text features.

- `Kimodo._generate(texts, max_frames, num_denoising_steps, pad_mask,
  first_heading_angle, motion_mask, observed_motion, cfg_weight=2.0, text_feat=None,
  text_pad_mask=None, guide_masks=None, cfg_type=None, progress_bar=tqdm)` —
  kimodo_model.py:562. DOES accept `text_feat`/`text_pad_mask`, so the runner calls it
  directly (then `motion_rep.inverse(motion, is_normalized=True)`, exactly what
  `__call__` does at kimodo_model.py:534-538). `motion_mask=None`/`observed_motion=None`
  is the unconstrained path `__call__` itself uses (kimodo_model.py:509).
  Single noise draw: `cur_mot = torch.randn(shape)` with shape
  `(B, max_frames, motion_rep_dim)` (motion_rep_dim = 417 for G1Skeleton34) at
  kimodo_model.py:610 — the ONLY stochastic input;
  `DDIMSampler` is deterministic eta=0 (kimodo/model/diffusion.py:113).

- Diffusion steps: `num_denoising_steps` is a DDIM subsampling of `num_base_steps=1000`
  (checkpoint config.yaml; `Diffusion.__init__` diffusion.py:29,
  `space_timesteps` diffusion.py:50). No scheduler object to configure — just pass the
  int, same shape as ARDY's `diffusion_steps`. The indoor_nav dataset used **100**
  (spec.jsonl `"diffusion_steps": 100`); `KimodoRunner.generate` defaults to 100, not
  ArdyRunner's 10 — quality at 10 steps on this model is unvalidated.

- Conditions: `MotionRepBase.create_conditions_from_constraints_batched(constraints_lst,
  lengths, to_normalize, device)` — kimodo/motion_rep/reps/base.py:262. Accepts a
  per-sample list-of-lists (same as ARDY) or one shared flat list.
  `KimodoMotionRep.create_conditions` — kimodo/motion_rep/reps/kimodo_motionrep.py:222.
  Filler channel names (what `update_constraints` may append to data/index dicts):
  `smooth_root_2d` (:242), `root_y_pos` (:253), `global_root_heading` (:262, (cos,sin)),
  `global_joints_rots` (:271, takes 3x3 matrices, converts with matrix_to_cont6d itself),
  `global_joints_positions` (:285, arbitrary (frame, joint) pairs — full passthrough like
  ARDY's).

- `KimodoMotionRep.inverse(features, is_normalized, posed_joints_from="rotations",
  return_numpy=False)` — kimodo_motionrep.py:167. Output keys (:209-217):
  `local_rot_mats, global_rot_mats, posed_joints, root_positions, smooth_root_pos,
  foot_contacts, global_root_heading`. Pose is decoded by FK FROM THE ROTATION CHANNEL by
  default — same reframing caveat as ARDY (scene2motion/constraints.py header applies
  verbatim: position constraints steer only indirectly).

- qpos: `MujocoQposConverter(skeleton)` — kimodo/exports/mujoco.py:27;
  `dict_to_qpos(output, device=None, root_quat_w_first=True, numpy=True,
  mujoco_rest_zero=False)` — mujoco.py:215; reads exactly `local_rot_mats` +
  `root_positions`, returns (B, T, 7+29=36): pelvis xyz (Z-up) + wxyz quat + 29 joint
  angles. `save_csv` at mujoco.py:329. Identical contract to the ARDY converter.

- Text encoder: `LLM2VecEncoder.__call__(texts)` -> `(feat (B, 1, 4096), lengths=[1]*B)`
  — kimodo/model/llm2vec/llm2vec_wrapper.py:65-88 (always one pooled token; internal
  batch_size pinned to 1 for repeatability). Honors `TEXT_ENCODER_DEVICE`
  (llm2vec_wrapper.py:40). Prompt sanitization: `sanitize_text` kimodo/sanitize.py:6.

- Working example this mirrors: `/home/linjiw/kimodo/indoor_nav_dataset/generate.py`
  (model call :171-190 — `post_processing=False` with the comment "unreliable on the G1
  skeleton"; cache-backed encoder class :28) and `encode_prompts.py` (CPU encode,
  `TEXT_ENCODER_DEVICE=cpu`, cache keyed by sha1 of the SANITIZED prompt).

## Interface mismatches vs ArdyRunner (audit scripts must know)

| aspect | ARDY (scene2motion) | Kimodo-G1-RP-v1 |
|---|---|---|
| fps | 25 | **30** (`motion_rep fps: 30` in checkpoint config.yaml; `model.fps` kimodo_model.py:49). 4 s = **120** frames, not 100. Never reuse ARDY frame counts. |
| architecture | autoregressive windows, `crop_history_length` | non-autoregressive TwostageDenoiser, one pass; no history args exist. Trained-length ceiling unknown from source (PE max_len=5000 is no constraint, backbone.py:242); indoor_nav clips went up to ~8 s + multi-prompt stitching — treat >10 s as unvalidated. |
| root path channel | `root_2d` -> raw root_pos x/z | `smooth_root_2d` -> `smooth_root_pos` x/z (kimodo_motionrep.py:242-251). Targets the ADMM-SMOOTHED trajectory (0.06 m margin, smooth_root.py:202): the raw pelvis legitimately sways ~6 cm around a constrained path, so path-tracking metrics should compare against `smooth_root_pos`, or widen tolerances vs ARDY's 1.4 cm mean. |
| root height | `root_y_pos` | same name, writes smooth_root_pos y (:253); height stays ABSOLUTE (duck channel intact). |
| heading | `global_root_heading` (cos, sin), 0 = +Z | identical convention: `atan2(hip_dz, -hip_dx)` feature_utils.py:112; `first_heading_angle` "Defaults to 0 (facing +Z)" (kimodo_model.py:430). Sparse-program first-heading rule ported unchanged. |
| joint position constraints | root joint REQUIRED at every constrained frame (ARDY asserts) | NOT required; instead smooth_root_2d must be constrained at those frames or ValueError (kimodo_motionrep.py:291). `KimodoConstraintSet` still injects the root target for parity — note this also pins pelvis onto the smooth root there. `local_joints_positions` = global − smooth_root(xz, y=0): ground-plane-relative, absolute height, not heading-rotated — same expressiveness as ARDY. |
| joint rotation constraints | 3x3 matrices | identical (filler converts to cont6d itself, :277). Joint indices refer to **G1Skeleton34** (34 joints: 32 articulated + 2 toe endpoints, definitions.py:286; root_idx 0 = pelvis_skel). Joint ORDER may differ from ARDY's skeleton — remap by name via `runner.joint_names`, never by index. |
| output dict | posed_joints, global_rot_mats, smooth_root_pos, foot_contacts, ... | superset-compatible: adds `root_positions`, `global_root_heading` (T,2 cos/sin), `local_rot_mats`. posed_joints is (T, 34, 3) not ARDY's joint count. |
| diffusion_steps default | 10 | **100** (what the working example used; 1000 base steps). |
| cfg | `cfg_weight=(2.0, 2.0)` | same [text, constraint] pair; checkpoint `cfg_type: separated`; extra `cfg_type` kwarg exposed. indoor_nav used [1.5, 2.0]. |
| text cache key | sha1(raw prompt) | sha1(**sanitized** prompt) to match kimodo's existing cache; raw-key fallback kept, so ArdyRunner-style caches also resolve. The 300-entry cache `/home/linjiw/kimodo/data/indoor_nav_1k/text_cache.npz` verified: key = sha1(sanitized), value (1, 4096) float32. |
| seed mechanics | `_per_sample_noise` patches many per-window randn draws | same patch, but exactly ONE (B, T, D) draw per generate call — so a given seed produces a completely different latent than under ARDY (different shape/stream). Seeds are comparable WITHIN Kimodo runs only; matched-pair independence from batch position holds identically (NOISE_STREAM_VERSION = 2). |

## VRAM / runtime expectations

- Checkpoint: ~1.1 GB on disk (`~/.cache/huggingface/hub/models--nvidia--Kimodo-G1-RP-v1`);
  16-layer, latent 1024 transformer — expect ~1.5-3 GB VRAM resident incl. activations
  for B=2, T=120 (indoor_nav ran B=1 fine on this 16 GB RTX 5080 alongside desktop use).
- GPU currently shows ~9.1 GB used by other work — the smoke needs roughly 2-3 GB free.
- Throughput reference: generate.log shows **3.1 s/clip** at 100 steps, ~5-6 s clips,
  B=1 incl. qpos+npz IO. The 2-clip smoke should take well under a minute after load.
- Text encoder: NOT loaded (cache-only). If ever needed: CPU, ~14 GB RAM
  (`TEXT_ENCODER_DEVICE=cpu`), ~seconds per prompt.

## Commands to run once the GPU frees

```bash
# 1) cheap import check (already verified to pass; no GPU, no model load)
/home/linjiw/kimodo/.venv/bin/python -c "import sys; sys.path.insert(0, '/tmp/claude-1000/-home-linjiw-ardy/f4440d67-ed27-4331-be07-dc169754a80c/scratchpad/kimodo'); import smoke_kimodo, kimodo_runner; print('imports ok')"

# 2) the smoke itself (loads Kimodo-G1-RP-v1 on cuda:0, 2 clips, ~4 s each, saves npz)
cd /tmp/claude-1000/-home-linjiw-ardy/f4440d67-ed27-4331-be07-dc169754a80c/scratchpad/kimodo \
  && /home/linjiw/kimodo/.venv/bin/python smoke_kimodo.py --device cuda:0 --steps 100
```

## Open risks

1. `_generate` is a private method; a kimodo upgrade could change its signature. Pinned
   to the checkout at /home/linjiw/kimodo (verified 2026-08-31).
2. Smooth-root semantics: quantitative parity with ARDY's root_2d tracking numbers
   (1.4 cm mean) is NOT expected; re-baseline capability-envelope numbers on kimodo.
3. `first_heading_angle` is a real denoiser input here (`input_first_heading_angle: true`
   in config.yaml), likely stronger than ARDY's — worth a quick A/B before trusting
   sparse-heading programs.
4. Position-constraint route decodes from rotations (same indirectness as ARDY); tuck /
   step-over effect sizes must be re-measured, not carried over.
5. `torch.randn` patch assumes the leading dim of a (tuple-shaped) randn equals B; true
   for kimodo_model.py:610 and harmless elsewhere (falls through to real randn), same
   assumption ArdyRunner v2 already ships.

## Addendum: kimodo_reduced_audit.py (naive-vs-calibrated counting on the second prior)

- One straight 6 s route @ 0.9 m/s (T=180 @ 30 fps); 12 body programs (duck 0.10/0.20/
  0.30/0.40, tuck 0.30/0.60 both arms, left-leg lift 0.08/0.20, arm-chain rot +/-30 deg,
  duck+tuck and duck+lift combos) x 6 paired seeds (100..105) + 6 matched controls + 6
  null-arm controls (200..205) = **84 clips**, ~5-9 min at 100 steps after load.
- Counting rows use audit_delta.py's exact rules (bits() copied verbatim); q99 comes from
  the null arm (per instruction), with exp005f-style within-program residual q99 reported
  alongside in the receipt for comparability.
- VENDORED (and why): raw_descriptor/envelope_series need G1Body -> mujoco (absent from
  the kimodo venv) + ARDY's g1.xml. Replaced with joint-point descriptors on Kimodo's
  native posed_joints; constant geometry offsets cancel in paired deltas, absolute
  extents are NOT comparable to the ARDY MuJoCo numbers. Validity is also vendored
  (tracking/progress/height instead of goal+collision-free). Everything pure-numpy is
  imported from scene2motion.morphology unchanged (matched_delta, active_set,
  seed_statistics, stability, whitener, Interaction, CHANNELS, PHYSICAL_FLOOR).
- Run: `cd <scratchpad>/kimodo && /home/linjiw/kimodo/.venv/bin/python
  kimodo_reduced_audit.py --steps 100`; GPU-free checks: `--selftest` and plain import.
