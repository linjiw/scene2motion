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
from contextlib import nullcontext as contextlib_nullcontext
from pathlib import Path

import numpy as np
import torch

from contextlib import contextmanager

from .constraints import ConstraintSpec, build_conditions


# Version 1 recreated and reseeded each sample's generator at every autoregressive window,
# repeating the same latent row through a long clip. Version 2 keeps one advancing stream per
# sample. Record this in new receipts and cache identities; artifacts generated under v1 remain
# reproducible historical evidence but must not be mixed with v2 reruns.
NOISE_STREAM_VERSION = 2


@contextmanager
def _per_sample_noise(seeds: list[int], device: str,
                      audit: list[dict] | None = None):
    """Make each batch element's initial noise depend only on its own seed.

    ARDY seeds once per generation call, so a sample's noise depends on its POSITION in the
    batch. That silently breaks every matched-pair measurement in this repo: two clips that
    differ only in scene geometry also differ in their noise draw unless they happen to land
    in the same slot, and three separate analyses measured the resulting floor at a scale
    comparable to the effect being studied (a 0.085 m null against a 0.137 m signal; the same
    nominal condition yielding worst-of-3 half-widths of 0.281 m and 0.380 m in two
    experiments).

    Its sampler is deterministic DDIM (eta = 0), so each autoregressive window's stochastic
    input is the `torch.randn(shape)` initial latent (ardy/model/ardy_model.py:481).
    Intercepting those calls — whose leading dimension is the batch size — and filling each
    row from a persistent per-sample generator makes the *latent draws* for sample i
    reproducible from seed i alone, independent of its batch position, while still advancing
    to fresh noise at the next window.  It does not make end-to-end GPU inference invariant to
    batch shape or row order: exact trajectory replay must preserve the original batch plan.
    """
    real = torch.randn
    B = len(seeds)
    # One advancing stream per (sample seed, device). ARDY draws fresh initial noise for
    # every autoregressive window. Recreating and reseeding a Generator inside `patched`
    # would repeat the identical latent row at every window, producing a deterministic but
    # statistically wrong clip. Keeping the generators alive for the whole context gives
    # each window the next draw while preserving batch-position independence.
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
        drawn = torch.stack(rows, 0)
        if audit is not None:
            # Hash every row rather than persisting latent tensors. Prompt-schedule
            # experiments use this to prove equal corresponding-window noise across arms
            # and fresh draws across windows without turning random inputs into a large
            # campaign artifact.
            audit.append({
                "shape": list(shape),
                "row_sha256": [
                    hashlib.sha256(
                        row.detach().contiguous().cpu().numpy().tobytes()
                    ).hexdigest()
                    for row in drawn
                ],
            })
        return drawn

    torch.randn = patched
    try:
        yield
    finally:
        torch.randn = real


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
        self.noise_stream_version = NOISE_STREAM_VERSION

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
                 seed: int | None = None,
                 seeds: list[int] | None = None) -> list[dict]:
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

        # `seeds` gives each sample its own noise, independent of batch position; `seed`
        # keeps the old whole-batch behaviour for code that does not need pairing.
        if seeds is not None and len(seeds) != B:
            raise ValueError(f"seeds has length {len(seeds)}, expected {B}")
        if seed is not None:
            torch.manual_seed(seed)
        ctx = (_per_sample_noise(seeds, dev) if seeds is not None
               else contextlib_nullcontext())
        with ctx, torch.no_grad():
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

    def generate_prompt_schedule(
        self,
        prompt_schedules: list[list[str] | tuple[str, ...]],
        specs: list[ConstraintSpec | None],
        num_frames: int,
        diffusion_steps: int = 10,
        cfg_weight: tuple[float, float] = (2.0, 2.0),
        *,
        seeds: list[int],
        history_frames: int | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """Generate a window-by-window prompt schedule through ARDY's demo interface.

        One persistent per-row RNG context surrounds the complete autoregressive loop.
        Re-entering that context for every window would reset each generator and recreate
        the repeated-latent defect quarantined as noise-stream v1. Duplicate seed values
        are intentional when paired arms need identical corresponding-window noise: the
        stream key includes the batch row, so equal seeds give independent but identical
        advancing streams.

        The accepted output transcript is always immutable and complete.  Only its last
        ``history_frames`` are shown to each continuation; the default is one token patch,
        matching ARDY's interactive GUI default for fastest prompt adaptation.  Constraint
        tensors and the model-visible time axis are sliced to the same global history start.

        Returns the complete padded normalized feature sequence plus a hash-only audit of
        every intercepted initial latent draw and history splice. Callers should score only
        ``:num_frames``; the final Horizon window may extend beyond it to a token boundary.
        """
        B = len(prompt_schedules)
        if B < 1 or len(specs) != B or len(seeds) != B:
            raise ValueError(
                "prompt_schedules, specs, and seeds must have the same positive length"
            )
        horizon = int(self.model.gen_horizon_len)
        patch = int(self.model.num_frames_per_token)
        if horizon <= 0 or horizon % patch:
            raise ValueError(
                f"invalid checkpoint horizon/token sizes: {horizon}/{patch}"
            )
        visible_history_limit = patch if history_frames is None else int(history_frames)
        if visible_history_limit <= 0 or visible_history_limit % patch:
            raise ValueError(
                f"history_frames must be a positive multiple of token size {patch}"
            )
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        n_windows = int(np.ceil(num_frames / horizon))
        padded_frames = n_windows * horizon
        schedules = [tuple(schedule) for schedule in prompt_schedules]
        if any(len(schedule) != n_windows for schedule in schedules):
            raise ValueError(
                f"every prompt schedule must contain {n_windows} Horizon{horizon} windows"
            )

        dev = self.device
        lengths = torch.tensor([num_frames] * B, device=dev)
        # Build the caller's exact num_frames-long program first. Padding is deliberately
        # unconstrained rather than extending the endpoint route by eight hidden frames.
        obs = mask = None
        if any(spec is not None for spec in specs):
            per_sample = []
            for spec in specs:
                if spec is None:
                    per_sample.append([])
                    continue
                if spec.T != num_frames:
                    raise ValueError(
                        f"spec length {spec.T} != scored num_frames {num_frames}"
                    )
                from .constraints import ArdyConstraintSet
                per_sample.append([
                    ArdyConstraintSet(spec, self.skeleton.root_idx, dev)
                ])
            obs, mask = self.model.motion_rep.create_conditions_from_constraints_batched(
                per_sample, lengths, to_normalize=True, device=dev)
            if padded_frames > num_frames:
                pad = padded_frames - num_frames
                obs = torch.nn.functional.pad(obs, (0, 0, 0, pad))
                mask = torch.nn.functional.pad(mask, (0, 0, 0, pad))

        def first_heading(spec: ConstraintSpec | None) -> float:
            if spec is None or spec.first_heading is None:
                return 0.0
            return float(spec.first_heading)

        initial_heading = torch.tensor(
            [first_heading(spec) for spec in specs], dtype=torch.float32, device=dev)
        initial_translation = torch.zeros(
            (B, self.model.motion_rep.nfeats_dict["root_pos"]),
            dtype=torch.float32, device=dev)
        # Keep an explicit, immutable transcript of already accepted frames.  ARDY's
        # ``autoregressive_step`` re-encodes and decodes the supplied history before it
        # returns history+suffix, so replacing the transcript with that return can subtly
        # rewrite frames that precede a prompt change.  The official interactive demo
        # instead retains its existing timeline and appends only ``samples[history_len:]``.
        # Matching that contract is essential for a causal prompt-handoff comparison:
        # arms that fork at frame 52 or 104 must remain byte-identical before the fork.
        transcript = None
        noise_audit: list[dict] = []

        def row_sha256(tensor: torch.Tensor) -> list[str]:
            return [
                hashlib.sha256(
                    row.detach().contiguous().cpu().numpy().tobytes()
                ).hexdigest()
                for row in tensor
            ]

        with _per_sample_noise(seeds, dev, audit=noise_audit), torch.no_grad():
            for window in range(n_windows):
                prompts = [schedule[window] for schedule in schedules]
                text_feat, text_pad = self.encode(prompts)
                # Never expose the accepted transcript itself to model code.  A future
                # implementation could mutate its input in place; passing a clone and
                # checking it afterwards makes that corruption fail closed.
                accepted_before = None if transcript is None else transcript.clone()
                accepted_frames = (
                    0 if accepted_before is None else int(accepted_before.shape[1])
                )
                visible_frames = min(visible_history_limit, accepted_frames)
                global_history_start = accepted_frames - visible_frames
                visible_history = (
                    None if accepted_before is None
                    else accepted_before[:, global_history_start:].clone()
                )
                model_input = (
                    None if visible_history is None else visible_history.clone()
                )
                model_num_frames = padded_frames - global_history_start
                window_mask = (
                    None if mask is None else mask[:, global_history_start:]
                )
                window_obs = (
                    None if obs is None else obs[:, global_history_start:]
                )
                returned = self.model.autoregressive_step(
                    num_frames=model_num_frames,
                    num_denoising_steps=diffusion_steps,
                    motion_mask=window_mask,
                    observed_motion=window_obs,
                    cfg_weight=cfg_weight,
                    text_feat=text_feat,
                    text_pad_mask=text_pad,
                    init_history_sequence=model_input,
                    init_global_translation=(initial_translation if window == 0 else None),
                    init_first_heading_angle=(initial_heading if window == 0 else None),
                )
                if (
                    visible_history is not None
                    and not torch.equal(model_input, visible_history)
                ):
                    raise RuntimeError(
                        f"window {window} mutated its supplied history in place"
                    )
                expected_returned = visible_frames + horizon
                if (
                    returned.ndim != 3
                    or tuple(returned.shape[:2]) != (B, expected_returned)
                ):
                    raise RuntimeError(
                        f"window {window} returned features {tuple(returned.shape)}, "
                        f"expected ({B}, {expected_returned}, features)"
                    )

                prior = accepted_before
                if prior is None:
                    transcript = returned[:, :horizon].clone()
                    reconstruction_exact = [True] * B
                    reconstruction_max_abs = [0.0] * B
                    input_hashes: list[str] = []
                    returned_input_hashes: list[str] = []
                else:
                    reconstructed = returned[:, :visible_frames]
                    delta = (reconstructed - visible_history).abs().reshape(B, -1)
                    reconstruction_exact = [
                        bool(torch.equal(reconstructed[row], visible_history[row]))
                        for row in range(B)
                    ]
                    reconstruction_max_abs = [
                        float(delta[row].max().item()) if delta.shape[1] else 0.0
                        for row in range(B)
                    ]
                    input_hashes = row_sha256(visible_history)
                    returned_input_hashes = row_sha256(reconstructed)
                    transcript = torch.cat(
                        [prior, returned[:, visible_frames:expected_returned]], dim=1
                    )

                expected_transcript = horizon * (window + 1)
                if tuple(transcript.shape[:2]) != (B, expected_transcript):
                    raise RuntimeError(
                        f"window {window} assembled transcript {tuple(transcript.shape)}, "
                        f"expected ({B}, {expected_transcript}, features)"
                    )
                if len(noise_audit) != window + 1:
                    raise RuntimeError(
                        f"window {window} expected one initial latent draw, observed "
                        f"{len(noise_audit) - window}"
                    )
                noise_audit[window].update({
                    "window_index": window,
                    "global_history_start_frame": global_history_start,
                    "accepted_transcript_frames_before": accepted_frames,
                    "input_history_frames": visible_frames,
                    "model_num_frames": model_num_frames,
                    "transcript_frames": expected_transcript,
                    "input_history_row_sha256": input_hashes,
                    "returned_input_history_row_sha256": returned_input_hashes,
                    "returned_history_reconstruction_exact": reconstruction_exact,
                    "returned_history_reconstruction_max_abs": reconstruction_max_abs,
                    "stable_transcript_row_sha256": row_sha256(transcript),
                })

        if transcript is None or len(noise_audit) != n_windows:
            raise RuntimeError(
                f"prompt schedule expected {n_windows} initial latent draws, "
                f"observed {len(noise_audit)}"
            )
        return transcript.detach().cpu().numpy(), noise_audit

    def decode_features(self, features: np.ndarray | torch.Tensor) -> list[dict]:
        """Decode normalized explicit features into the same dictionaries as ``generate``."""
        motion = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if motion.ndim != 3:
            raise ValueError(
                f"normalized features must be (batch, frames, features), got {motion.shape}"
            )
        expected = int(self.model.motion_rep.motion_rep_dim)
        if int(motion.shape[2]) != expected:
            raise ValueError(f"feature width {motion.shape[2]} != {expected}")
        with torch.no_grad():
            out = self.model.motion_rep.inverse(motion, is_normalized=True)
        return [self._split(out, i) for i in range(int(motion.shape[0]))]

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
