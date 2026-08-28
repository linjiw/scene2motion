# Scene2Motion-G1: the ARDY generation runner.
#
# One process holds the diffusion model on the GPU for the whole experiment. The Llama-3-8B
# text encoder does NOT fit alongside it on a 16 GB card, so it runs as a CPU gradio service
# (ardy/scripts/run_text_encoder_server.py --device cpu) and ARDY's TextEncoderAPI client
# talks to it. Prompt embeddings are cached here as well, because in these experiments the
# same handful of prompts is reused across thousands of scenes and a CPU Llama round-trip
# costs seconds while a generation costs ~0.3 s.

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
                 cache_path: str | os.PathLike | None = None):
        from ardy.model import load_model
        from ardy.model.registry import resolve_model_name

        self.device = device
        self.model_name = resolve_model_name(model_name)
        self.model = load_model(model_name, device=device)
        self.fps = float(self.model.motion_rep.fps)
        self.skeleton = self.model.skeleton
        self.joint_names = list(self.skeleton.bone_index.keys())
        # Longest history that keeps each autoregressive step inside ARDY's trained 10 s
        # window; matches scripts/generate.py. Unbounded history degrades into jitter.
        patch = self.model.num_frames_per_token
        win = (int(10 * self.fps) // patch) * patch
        self.history_frames = ((win - self.model.gen_horizon_len) // patch) * patch

        self.cache_path = Path(cache_path) if cache_path else None
        self._text_cache: dict[str, np.ndarray] = {}
        if self.cache_path and self.cache_path.exists():
            with np.load(self.cache_path) as z:
                self._text_cache = {k: z[k] for k in z.files}

        from ardy.exports.mujoco import MujocoQposConverter
        self._qpos_conv = MujocoQposConverter(self.skeleton)

    # -- text ---------------------------------------------------------------------------

    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached (text_feat, text_pad_mask) for a batch of prompts."""
        missing = [t for t in dict.fromkeys(texts) if _key(t) not in self._text_cache]
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
        # spends the opening of the clip unwinding a contradiction it was handed.
        h0 = torch.tensor(
            [float(s.heading[0]) if (s is not None and s.heading is not None) else 0.0
             for s in specs], device=dev)

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
