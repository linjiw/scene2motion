# Scene2Motion-G1: the ARDY generation runner.
#
# One process holds the diffusion model on the GPU for the whole experiment. The diffusion
# model is small (~930 MB of VRAM); the Llama-3-8B text encoder is not, and it is the part
# that makes this fragile on a shared box: ~14 GB of RAM as a CPU service, ~16 GB of VRAM
# locally, and ARDY's default `auto` mode falls back from one to the other SILENTLY.
#
# So the prompt-embedding cache is treated as authoritative. Every experiment here reuses a
# handful of prompts, so the encoder is not loaded at all unless a genuinely new prompt
# appears. Populate the cache once, with the CPU service running:
#     ardy/.venv/bin/python ardy/scripts/run_text_encoder_server.py --device cpu
#     ArdyRunner(cache_path=..., text_encoder=True).encode([...new prompts...])
# Thereafter runs need neither the service nor the 8B model, and start in ~3 s.

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from .constraints import ConstraintSpec, build_conditions


def _key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


class ArdyRunner:
    """Frozen ARDY-G1 prior + prompt-embedding cache + MuJoCo qpos export."""

    def __init__(self, model_name: str = "g1", device: str = "cuda:0",
                 cache_path: str | os.PathLike | None = None,
                 text_encoder: bool | None = None):
        """`text_encoder=None` (default) loads one ONLY if the prompt cache is empty.

        The encoder is Llama-3-8B: ~14 GB of RAM as a CPU service, ~16 GB of VRAM locally.
        ARDY's default `TEXT_ENCODER_MODE=auto` probes the CPU service and, if it is
        unreachable, silently falls back to loading the 8B model on CUDA. On a shared box
        that turns a dead service into a CUDA OOM several minutes into a run, which is how
        two experiments were lost. Since every experiment here reuses a handful of prompts,
        the cache is authoritative and the encoder is not loaded at all unless a genuinely
        new prompt appears -- in which case `encode` raises with instructions rather than
        quietly grabbing the GPU.
        """
        from ardy.model import load_model
        from ardy.model.registry import resolve_model_name

        self.device = device
        self.model_name = resolve_model_name(model_name)

        self.cache_path = Path(cache_path) if cache_path else None
        self._text_cache: dict[str, np.ndarray] = {}
        if self.cache_path and self.cache_path.exists():
            with np.load(self.cache_path) as z:
                self._text_cache = {k: z[k] for k in z.files}

        if text_encoder is None:
            text_encoder = not self._text_cache
        self._want_encoder = bool(text_encoder)
        self.model = load_model(model_name, device=device,
                                text_encoder=None if self._want_encoder else False)
        self.fps = float(self.model.motion_rep.fps)
        self.skeleton = self.model.skeleton
        self.joint_names = list(self.skeleton.bone_index.keys())
        # Longest history that keeps each autoregressive step inside ARDY's trained 10 s
        # window; matches scripts/generate.py. Unbounded history degrades into jitter.
        patch = self.model.num_frames_per_token
        win = (int(10 * self.fps) // patch) * patch
        self.history_frames = ((win - self.model.gen_horizon_len) // patch) * patch

        from ardy.exports.mujoco import MujocoQposConverter
        self._qpos_conv = MujocoQposConverter(self.skeleton)

    # -- text ---------------------------------------------------------------------------

    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached (text_feat, text_pad_mask) for a batch of prompts."""
        missing = [t for t in dict.fromkeys(texts) if _key(t) not in self._text_cache]
        if missing and self.model.text_encoder is None:
            raise RuntimeError(
                f"{len(missing)} prompt(s) are not in the embedding cache and no text "
                f"encoder is loaded: {missing[:3]}. Either add them to "
                f"{self.cache_path} by constructing ArdyRunner(text_encoder=True) once "
                f"with the CPU encoder service running "
                f"(ardy/scripts/run_text_encoder_server.py --device cpu), or reuse a "
                f"cached prompt. Refusing to load Llama-3-8B implicitly.")
        if missing:
            feat, lens = self.model.text_encoder(missing)
            for i, t in enumerate(missing):
                self._text_cache[_key(t)] = feat[i, : lens[i]].float().cpu().numpy()
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(self.cache_path, **self._text_cache)
        embs = [self._text_cache[_key(t)] for t in texts]
        L = max(len(e) for e in embs)
        out = np.zeros((len(embs), L, embs[0].shape[-1]), dtype=np.float32)
        pad = np.zeros((len(embs), L), dtype=bool)
        for i, e in enumerate(embs):
            out[i, : len(e)] = e
            pad[i, : len(e)] = True
        return (torch.from_numpy(out).to(self.device),
                torch.from_numpy(pad).to(self.device))

    # -- generation ---------------------------------------------------------------------

    def generate(self, prompts: list[str], specs: list[ConstraintSpec | None],
                 num_frames: int, diffusion_steps: int = 10,
                 cfg_weight: tuple[float, float] = (2.0, 2.0),
                 seed: int | None = None) -> list[dict]:
        """Generate one motion per (prompt, spec). Returns per-sample numpy output dicts.

        Batched: every sample shares `num_frames` but may carry a different constraint spec,
        which ARDY supports through the per-sample constraint-list form of
        ``create_conditions_from_constraints_batched``.
        """
        from ardy.motion_rep.tools import length_to_mask

        if len(prompts) != len(specs):
            raise ValueError("prompts and specs must have the same length")
        B = len(prompts)
        dev = self.device
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
                    from .constraints import ArdyConstraintSet
                    per_sample.append([ArdyConstraintSet(s, self.skeleton.root_idx, dev)])
            obs, mask = self.model.motion_rep.create_conditions_from_constraints_batched(
                per_sample, lengths, to_normalize=True, device=dev)

        # The first frame's heading must agree with the requested heading, or the model
        # spends the opening of the clip unwinding a contradiction it was handed. For a
        # SPARSE program this is not `heading[0]`: that is the heading at the first
        # *constrained* frame, which may be seconds in, and importing it would rotate the
        # whole clip to match a pose the robot should only reach at the end.
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

        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            motion = self.model(
                prompts, num_frames, num_denoising_steps=diffusion_steps,
                pad_mask=pad_mask, first_heading_angle=h0,
                motion_mask=mask, observed_motion=obs, cfg_weight=cfg_weight,
                text_feat=text_feat, text_pad_mask=text_pad,
                crop_history_length=self.history_frames,
                progress_bar=lambda x, **kw: x,
            )
            out = self.model.motion_rep.inverse(motion, is_normalized=True)
        return [self._split(out, i) for i in range(B)]

    def _split(self, out: dict, i: int) -> dict:
        return {k: (v[i].detach().cpu().numpy() if torch.is_tensor(v) and v.dim() > 0 else v)
                for k, v in out.items()}

    # -- export -------------------------------------------------------------------------

    def to_qpos(self, sample: dict) -> np.ndarray:
        """(T, 36) MuJoCo qpos: pelvis xyz (Z-up) + wxyz quaternion + 29 joint angles."""
        batched = {k: torch.as_tensor(v, device=self.device)[None]
                   for k, v in sample.items() if isinstance(v, np.ndarray)}
        qpos = self._qpos_conv.dict_to_qpos(batched, self.device)
        if torch.is_tensor(qpos):
            qpos = qpos.detach().cpu().numpy()
        return np.asarray(qpos)[0]
