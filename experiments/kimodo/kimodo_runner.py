# Scene2Motion-G1: the KIMODO generation runner (drop-in sibling of scene2motion ArdyRunner).
#
# Public interface is identical to scene2motion/runner.py ArdyRunner:
#     KimodoRunner(model_name, device, cache_path, text_encoder)
#       .generate(prompts, specs, num_frames, diffusion_steps, seeds=[...]) -> list[dict]
#       .to_qpos(sample) -> (T, 36) MuJoCo qpos          .encode(texts)
#       .fps (== 30.0, NOT ARDY's 25)                    .model_name
#
# Differences from ARDY that callers must know (full detail in NOTES.md next to this file):
#   * fps is 30, so num_frames for a given duration is duration*30, not *25.
#   * Kimodo is NON-autoregressive (TwostageDenoiser): one denoising pass over the whole
#     clip, no history window, no crop_history_length. The single stochastic draw is
#     torch.randn(shape) at kimodo/model/kimodo_model.py:610 and the DDIM sampler is
#     deterministic (eta = 0, kimodo/model/diffusion.py:113), so _per_sample_noise below
#     makes sample i reproducible from seeds[i] alone -- same contract as ArdyRunner v2.
#   * The root-path channel is named "smooth_root_2d" (not "root_2d") and targets the
#     SMOOTHED root trajectory (ADMM smoother, 0.06 m margin,
#     kimodo/motion_rep/smooth_root.py:202), not the raw pelvis. root_y_pos /
#     global_root_heading / global_joints_rots / global_joints_positions carry over.
#   * The prompt-embedding cache is keyed by sha1(sanitize_text(prompt)) to be
#     byte-compatible with kimodo's existing 300-prompt cache
#     (/home/linjiw/kimodo/data/indoor_nav_1k/text_cache.npz). sha1(raw prompt) is
#     accepted as a fallback key so ArdyRunner-style caches still resolve.
#
# The text encoder is the same LLM2Vec Llama-3-8B lineage as ARDY's (~16 GB). As in
# ArdyRunner, the cache is authoritative: the 8B encoder is NEVER loaded implicitly. A
# guard object is installed as model.text_encoder so kimodo's load_model cannot silently
# probe the API service or fall back to loading Llama on CUDA (TEXT_ENCODER_MODE=auto
# would otherwise do exactly that, kimodo/model/load_model.py:86-105).

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from contextlib import nullcontext as contextlib_nullcontext
from pathlib import Path

import numpy as np
import torch

# Same meaning as scene2motion.runner.NOISE_STREAM_VERSION: v2 = persistent per-sample
# generator streams. Kimodo draws initial noise once (no autoregressive windows), so v1's
# repeat-the-latent bug cannot occur here, but keep the version tag so receipts stay
# comparable across the two runners.
NOISE_STREAM_VERSION = 2

SCENE2MOTION_ROOT = os.environ.get("SCENE2MOTION_ROOT", "/home/linjiw/scene2motion")


def load_constraint_spec_class():
    """Import scene2motion's ConstraintSpec without requiring the package be installed.

    scene2motion/constraints.py only needs numpy+torch, but the kimodo venv does not have
    scene2motion on its path; loading by file avoids importing anything ARDY-flavoured.
    """
    import sys
    try:
        from scene2motion.constraints import ConstraintSpec  # installed / on PYTHONPATH
        return ConstraintSpec
    except ImportError:
        pass
    # scene2motion/__init__.py is empty, so importing the subpackage pulls in nothing
    # ARDY-flavoured (constraints.py needs only numpy + torch + dataclasses).
    # Note: exec-ing constraints.py via importlib.spec_from_file_location without
    # registering it in sys.modules breaks @dataclass (it resolves cls.__module__
    # through sys.modules), so a plain path-based package import is the robust route.
    sys.path.insert(0, SCENE2MOTION_ROOT)
    try:
        from scene2motion.constraints import ConstraintSpec
        return ConstraintSpec
    finally:
        sys.path.remove(SCENE2MOTION_ROOT)


@contextmanager
def _per_sample_noise(seeds: list[int], device: str):
    """Make each batch element's initial noise depend only on its own seed.

    Port of scene2motion/runner.py:37 (v2). Kimodo's only stochastic input is the single
    initial latent `torch.randn(shape)` with shape (B, T, motion_rep_dim) at
    kimodo/model/kimodo_model.py:610; the DDIM sampler is deterministic. Intercepting
    randn calls whose leading dimension equals the batch size and filling each row from a
    persistent per-sample generator makes sample i reproducible from seeds[i] alone,
    independent of batch composition. The generators stay alive for the whole context so
    that if a future kimodo version draws more than once, streams advance instead of
    repeating (the exact v1 bug the ARDY runner fixed).
    """
    real = torch.randn
    B = len(seeds)
    generators: dict[tuple[int, str], torch.Generator] = {}

    def generator(i: int, dev) -> torch.Generator:
        key = (i, str(dev))
        if key not in generators:
            g = torch.Generator(device=dev)
            g.manual_seed(int(seeds[i]))
            generators[key] = g
        return generators[key]

    def patched(*args, **kwargs):
        shape = args[0] if len(args) == 1 and not isinstance(args[0], int) else args
        try:
            shape = tuple(int(x) for x in shape)
        except TypeError:
            return real(*args, **kwargs)
        if not shape or shape[0] != B:
            return real(*args, **kwargs)
        dev = kwargs.get("device", device)
        rows = []
        for i in range(B):
            rows.append(real(shape[1:], device=dev, generator=generator(i, dev),
                             dtype=kwargs.get("dtype")))
        return torch.stack(rows, 0)

    torch.randn = patched
    try:
        yield
    finally:
        torch.randn = real


def _raw_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class _CacheGuardEncoder:
    """Installed as model.text_encoder so load_model never instantiates/probes a real one.

    kimodo/model/load_model.py:194-215: passing a pre-built `text_encoder` object skips
    encoder selection entirely (no API probe, no silent Llama-on-CUDA fallback). Nothing
    in KimodoRunner ever calls model.text_encoder -- generate() passes text_feat straight
    to Kimodo._generate -- so calling this is always a bug worth loud failure.
    """

    def to(self, *a, **k):
        return self

    def eval(self):
        return self

    def __call__(self, texts):
        raise RuntimeError(
            "model.text_encoder was invoked, but KimodoRunner supplies cached embeddings "
            "via text_feat. This path must not be reached; if you need new prompts "
            "encoded, use KimodoRunner.encode with text_encoder=True.")


class KimodoConstraintSet:
    """Adapter from a scene2motion ConstraintSpec to Kimodo's ``update_constraints`` protocol.

    Direct port of scene2motion.constraints.ArdyConstraintSet with one rename: the root
    ground-path channel is ``smooth_root_2d`` (kimodo/motion_rep/reps/kimodo_motionrep.py:242),
    not ``root_2d``. Everything else -- root_y_pos, global_root_heading (cos, sin),
    global_joints_rots as 3x3 matrices (kimodo converts with matrix_to_cont6d itself,
    kimodo_motionrep.py:277), and arbitrary (frame, joint) pairs for
    global_joints_positions -- carries over unchanged.

    Kimodo's position filler (kimodo_motionrep.py:285-303) requires smooth_root_2d to be
    constrained at every position-constrained frame (ValueError at :291) but, unlike
    ARDY's, does NOT require the root joint among the constrained joints. We keep
    injecting the root target anyway for behavioural parity with ArdyConstraintSet; note
    that in kimodo this additionally pins the pelvis onto the smooth root in the ground
    plane at those frames (the smoother allows the true pelvis to sway ~6 cm around it).

    Height semantics match ARDY: local_joints_positions are smooth-root-relative in the
    ground plane and ABSOLUTE in height (local_reference[..., 1] = 0.0 at
    kimodo_motionrep.py:294-295), so world-height targets pass through directly.
    """

    name = "scene2motion-kimodo"

    def __init__(self, spec, root_idx: int, device: str):
        self.spec, self.root_idx, self.device = spec, root_idx, device

    def _t(self, x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        s = self.spec
        fi = (self._t(s.root_frames, torch.long) if s.root_frames is not None
              else torch.arange(s.T, device=self.device))

        data_dict["smooth_root_2d"].append(self._t(s.root_xz))
        index_dict["smooth_root_2d"].append(fi)

        if s.heading is not None:
            h = self._t(s.heading)
            data_dict["global_root_heading"].append(torch.stack([torch.cos(h), torch.sin(h)], -1))
            index_dict["global_root_heading"].append(fi)

        if s.root_y is not None:
            data_dict["root_y_pos"].append(self._t(s.root_y))
            index_dict["root_y_pos"].append(fi)

        if s.rot_frames is not None:
            rf = self._t(s.rot_frames, torch.long)
            rj = self._t(s.rot_joints, torch.long)
            pairs = torch.stack([
                rf[:, None].expand(-1, len(rj)).reshape(-1),
                rj[None].expand(len(rf), -1).reshape(-1),
            ], -1)
            data_dict["global_joints_rots"].append(
                self._t(s.rot_targets).reshape(-1, 3, 3))
            index_dict["global_joints_rots"].append(pairs)

        if s.pos_frames is not None:
            pf = self._t(s.pos_frames, torch.long)
            # Root injected at every position-constrained frame with a target consistent
            # with root_xz / root_y, exactly as ArdyConstraintSet does. Kimodo requires
            # only that smooth_root_2d is constrained at these frames -- guaranteed when
            # pos_frames is a subset of the root-constrained frames (all frames for a
            # dense spec); otherwise create_conditions raises (kimodo_motionrep.py:291).
            joints = self._t(np.concatenate([[self.root_idx], s.pos_joints]), torch.long)
            root_tgt = torch.stack([
                self._t(s.root_xz[s.pos_frames, 0]),
                self._t(s.root_y[s.pos_frames]),
                self._t(s.root_xz[s.pos_frames, 1]),
            ], -1)                                             # (K, 3)
            targets = torch.cat([root_tgt[:, None], self._t(s.pos_targets)], 1)  # (K, 1+J, 3)
            pairs = torch.stack([
                pf[:, None].expand(-1, len(joints)).reshape(-1),
                joints[None].expand(len(pf), -1).reshape(-1),
            ], -1)
            data_dict["global_joints_positions"].append(targets.reshape(-1, 3))
            index_dict["global_joints_positions"].append(pairs)


def build_conditions(model, spec, device: str):
    """(observed_motion, motion_mask) ready to pass to ``Kimodo._generate``."""
    cs = KimodoConstraintSet(spec, model.skeleton.root_idx, device)
    return model.motion_rep.create_conditions_from_constraints_batched(
        [cs], torch.tensor([spec.T], device=device), to_normalize=True, device=device
    )


def channel_usage(model, mask: torch.Tensor) -> dict[str, int]:
    """Constrained (frame, channel) entries per feature block; authoring-bug guard.

    Kimodo block names: smooth_root_pos, global_root_heading, local_joints_positions,
    global_rot_data, velocities, foot_contacts (kimodo_motionrep.py:33-41).
    """
    return {k: int(mask[0][:, sl].sum().item()) for k, sl in model.motion_rep.slice_dict.items()}


class KimodoRunner:
    """Frozen Kimodo-G1 prior + prompt-embedding cache + MuJoCo qpos export."""

    def __init__(self, model_name: str = "Kimodo-G1-RP-v1", device: str = "cuda:0",
                 cache_path: str | os.PathLike | None = None,
                 text_encoder: bool | None = None):
        """`text_encoder=None` (default) allows one ONLY if the prompt cache is empty.

        Same policy as ArdyRunner: the encoder is Llama-3-8B (~14 GB RAM on CPU, ~16 GB
        VRAM on GPU) and is never loaded implicitly. With text_encoder falsy, a cache
        miss in encode() raises with instructions instead of grabbing the GPU. Unlike
        ArdyRunner (which loads the encoder at init), the encoder here is built lazily on
        the first actual miss, on TEXT_ENCODER_DEVICE (default cpu).
        """
        # Resolve the checkpoint from the local HF cache first instead of pinging the
        # hub on every construction (kimodo/model/load_model.py:44-59).
        os.environ.setdefault("LOCAL_CACHE", "true")

        from kimodo.model import load_model
        from kimodo.sanitize import sanitize_text

        self.device = device
        self._sanitize = sanitize_text
        self.noise_stream_version = NOISE_STREAM_VERSION

        self.cache_path = Path(cache_path) if cache_path else None
        self._text_cache: dict[str, np.ndarray] = {}
        if self.cache_path and self.cache_path.exists():
            with np.load(self.cache_path) as z:
                self._text_cache = {k: z[k] for k in z.files}

        if text_encoder is None:
            text_encoder = not self._text_cache
        self._want_encoder = bool(text_encoder)
        self._encoder = None  # lazy LLM2VecEncoder, built only on a cache miss

        self.model, self.model_name = load_model(
            model_name, device=device, text_encoder=_CacheGuardEncoder(),
            return_resolved_name=True)
        self.fps = float(self.model.fps)                 # 30.0 for Kimodo-G1-RP-v1
        self.skeleton = self.model.skeleton              # G1Skeleton34, root_idx 0
        self.joint_names = list(self.skeleton.bone_index.keys())

        from kimodo.exports.mujoco import MujocoQposConverter
        self._qpos_conv = MujocoQposConverter(self.skeleton)

    # -- text ---------------------------------------------------------------------------

    def _cache_key(self, text: str) -> str | None:
        """Cache key for a prompt, or None if absent under both key schemes.

        Canonical scheme: sha1(sanitize_text(text)) -- matches kimodo's
        indoor_nav_dataset/encode_prompts.py cache. sha1(raw) accepted for
        ArdyRunner-style caches.
        """
        for key in (_raw_key(self._sanitize(text)), _raw_key(text)):
            if key in self._text_cache:
                return key
        return None

    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached (text_feat, text_pad_mask) for a batch of prompts.

        LLM2Vec pools each prompt to a single 4096-d token
        (kimodo/model/llm2vec/llm2vec_wrapper.py:65-88), so L is 1 in practice; the
        padding logic still handles variable lengths for cache compatibility.
        """
        missing = [t for t in dict.fromkeys(texts) if self._cache_key(t) is None]
        if missing and not self._want_encoder:
            raise RuntimeError(
                f"{len(missing)} prompt(s) are not in the embedding cache and no text "
                f"encoder is allowed: {missing[:3]}. Either add them to "
                f"{self.cache_path} by constructing KimodoRunner(text_encoder=True) "
                f"once (encoder runs on TEXT_ENCODER_DEVICE, default cpu, ~14 GB RAM), "
                f"or reuse a cached prompt (e.g. kimodo's "
                f"data/indoor_nav_1k/text_cache.npz has 300). Refusing to load "
                f"Llama-3-8B implicitly.")
        if missing:
            if self._encoder is None:
                from kimodo.model import LLM2VecEncoder
                self._encoder = LLM2VecEncoder(
                    base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
                    dtype="bfloat16", llm_dim=4096,
                    device=os.environ.get("TEXT_ENCODER_DEVICE", "cpu"))
            sanitized = [self._sanitize(t) for t in missing]
            feat, lens = self._encoder(sanitized)        # (B, 1, 4096), lengths
            for i, s in enumerate(sanitized):
                self._text_cache[_raw_key(s)] = feat[i, : lens[i]].float().cpu().numpy()
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(self.cache_path, **self._text_cache)
        embs = [self._text_cache[self._cache_key(t)] for t in texts]
        L = max(len(e) for e in embs)
        out = np.zeros((len(embs), L, embs[0].shape[-1]), dtype=np.float32)
        pad = np.zeros((len(embs), L), dtype=bool)
        for i, e in enumerate(embs):
            out[i, : len(e)] = e
            pad[i, : len(e)] = True
        return (torch.from_numpy(out).to(self.device),
                torch.from_numpy(pad).to(self.device))

    # -- generation ---------------------------------------------------------------------

    def generate(self, prompts: list[str], specs: list | None,
                 num_frames: int, diffusion_steps: int = 100,
                 cfg_weight: tuple[float, float] = (2.0, 2.0),
                 seed: int | None = None,
                 seeds: list[int] | None = None,
                 cfg_type: str | None = None) -> list[dict]:
        """Generate one motion per (prompt, spec). Returns per-sample numpy output dicts.

        Same contract as ArdyRunner.generate. Notes specific to Kimodo:
        * `diffusion_steps` maps to Kimodo's ``num_denoising_steps`` -- a DDIM
          subsampling of the 1000 base steps (kimodo/model/diffusion.py:50
          space_timesteps). The indoor_nav dataset used 100; the default here follows
          that, NOT ArdyRunner's 10 (quality at 10 steps is unvalidated on this model).
        * `cfg_weight` is [text_cfg, constraint_cfg]; the checkpoint's cfg_type is
          "separated". `cfg_type=None` keeps the model default.
        * Output dicts carry kimodo's inverse() keys (kimodo_motionrep.py:209-217):
          local_rot_mats, global_rot_mats, posed_joints, root_positions,
          smooth_root_pos, foot_contacts, global_root_heading. Superset-compatible with
          the ARDY keys scene2motion reads (posed_joints / global_rot_mats /
          smooth_root_pos / foot_contacts).
        """
        from kimodo.motion_rep.feature_utils import length_to_mask

        if len(prompts) != len(specs):
            raise ValueError("prompts and specs must have the same length")
        B = len(prompts)
        dev = self.device
        texts = [self._sanitize(p) for p in prompts]
        lengths = torch.tensor([num_frames] * B, device=dev)
        pad_mask = length_to_mask(lengths)
        text_feat, text_pad = self.encode(prompts)

        obs = mask = None
        if any(s is not None for s in specs):
            per_sample = []
            for s in specs:
                if s is None:
                    per_sample.append([])
                else:
                    if s.T != num_frames:
                        raise ValueError(f"spec length {s.T} != num_frames {num_frames}")
                    per_sample.append([KimodoConstraintSet(s, self.skeleton.root_idx, dev)])
            obs, mask = self.model.motion_rep.create_conditions_from_constraints_batched(
                per_sample, lengths, to_normalize=True, device=dev)

        # Same first-heading rule as ArdyRunner: for a sparse program whose first
        # constrained frame is late in the clip, importing heading[0] would rotate the
        # whole motion to match a pose the robot should only reach at the end.
        # Convention check: kimodo's heading angle is atan2(hip_dz, -hip_dx)
        # (kimodo/motion_rep/feature_utils.py:112) with 0 == facing +Z, matching ARDY.
        def _first_heading(s) -> float:
            if s is None:
                return 0.0
            if s.first_heading is not None:
                return float(s.first_heading)
            if s.heading is None:
                return 0.0
            if s.root_frames is not None and int(np.min(s.root_frames)) > 0:
                return 0.0
            return float(s.heading[0])

        h0 = torch.tensor([_first_heading(s) for s in specs], device=dev)

        if seeds is not None and len(seeds) != B:
            raise ValueError(f"seeds has length {len(seeds)}, expected {B}")
        if seed is not None:
            torch.manual_seed(seed)
        ctx = (_per_sample_noise(seeds, dev) if seeds is not None
               else contextlib_nullcontext())
        # Kimodo.__call__ (kimodo_model.py:380) cannot take precomputed text features, so
        # we call _generate (kimodo_model.py:562) directly -- it accepts
        # text_feat/text_pad_mask -- and then run the same inverse() step __call__ would.
        # Skipped __call__ extras, deliberately: post_processing (documented unreliable
        # on the G1 skeleton, see indoor_nav_dataset/generate.py:185) and the
        # SOMASkeleton30 output conversion (G1 skeleton, not applicable).
        with ctx, torch.no_grad():
            motion = self.model._generate(
                texts, int(num_frames), num_denoising_steps=int(diffusion_steps),
                pad_mask=pad_mask, first_heading_angle=h0,
                motion_mask=mask, observed_motion=obs, cfg_weight=list(cfg_weight),
                text_feat=text_feat, text_pad_mask=text_pad,
                cfg_type=cfg_type,
                progress_bar=lambda x, **kw: x,
            )
            out = self.model.motion_rep.inverse(motion, is_normalized=True)
        return [self._split(out, i) for i in range(B)]

    def _split(self, out: dict, i: int) -> dict:
        return {k: (v[i].detach().cpu().numpy() if torch.is_tensor(v) and v.dim() > 0 else v)
                for k, v in out.items()}

    # -- export -------------------------------------------------------------------------

    def to_qpos(self, sample: dict) -> np.ndarray:
        """(T, 36) MuJoCo qpos: pelvis xyz (Z-up) + wxyz quaternion + 29 joint angles.

        kimodo/exports/mujoco.py:215 dict_to_qpos reads exactly local_rot_mats and
        root_positions; returns numpy (B, T, 36) with numpy=True (the default).
        """
        batched = {k: torch.as_tensor(np.asarray(sample[k]), dtype=torch.float32,
                                      device=self.device)[None]
                   for k in ("local_rot_mats", "root_positions")}
        qpos = self._qpos_conv.dict_to_qpos(batched, self.device)
        return np.asarray(qpos)[0]
