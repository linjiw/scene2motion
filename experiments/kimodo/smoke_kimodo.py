#!/home/linjiw/kimodo/.venv/bin/python
"""Smoke test for KimodoRunner: 2 prompts x 2 seeds, straight-path spec, ~4 s clips.

DO NOT run while the GPU is occupied by another experiment. Run with the KIMODO venv:

    cd /tmp/claude-1000/-home-linjiw-ardy/f4440d67-ed27-4331-be07-dc169754a80c/scratchpad/kimodo
    /home/linjiw/kimodo/.venv/bin/python smoke_kimodo.py

Cheap import-only check (no model load, no GPU):

    /home/linjiw/kimodo/.venv/bin/python -c "import smoke_kimodo, kimodo_runner; print('imports ok')"

Text embeddings: reuses kimodo's existing 300-prompt cache at
/home/linjiw/kimodo/data/indoor_nav_1k/text_cache.npz (keys are sha1 of the sanitized
prompt -- verified byte-identical to KimodoRunner's key scheme). Both smoke prompts below
are confirmed present in that cache, so neither the CPU Llama service nor the local 8B
encoder is ever touched. If you change PROMPTS to uncached text, point --cache at a
writable path and construct the runner with text_encoder=True (CPU encode, ~14 GB RAM).

What it checks (soft, printed as OK/WARN -- generation is stochastic):
  * output shapes: posed_joints (T,34,3), qpos (T,36), fps == 30
  * pelvis height stays in a standing band (native y-up root_positions[:,1] and MuJoCo
    z-up qpos[:,2] should agree)
  * forward progress along +Z close to the requested straight path (speed * duration)
  * mean tracking error of the root xz against the constrained path
  * seed independence: the two samples differ (different prompts AND different seeds)

Outputs one npz per sample under ./smoke_out/.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Both prompts verified present in data/indoor_nav_1k/text_cache.npz (already in
# sanitized form, so sha1(sanitize_text(p)) == sha1(p) for these exact strings).
PROMPTS = [
    "A person walks forward at a steady pace.",
    "A happy person walks forward at a steady pace.",
]
SEEDS = [1234, 5678]
KIMODO_CACHE = "/home/linjiw/kimodo/data/indoor_nav_1k/text_cache.npz"


def straight_spec(ConstraintSpec, T: int, fps: float, speed: float):
    """Dense straight-line root path along +Z at `speed`, heading locked to 0 (+Z)."""
    t = np.arange(T, dtype=np.float64) / fps
    root_xz = np.stack([np.zeros(T), speed * t], axis=-1)   # (T, 2) = (x, z), y-up frame
    heading = np.zeros(T)                                   # 0 rad == facing +Z
    return ConstraintSpec(root_xz=root_xz, heading=heading)


def check(label, ok, detail):
    print(f"  [{'OK ' if ok else 'WARN'}] {label}: {detail}")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=100,
                    help="DDIM denoising steps (indoor_nav dataset used 100)")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--speed", type=float, default=0.9, help="root path speed, m/s")
    ap.add_argument("--cache", default=KIMODO_CACHE)
    ap.add_argument("--out", default=str(HERE / "smoke_out"))
    args = ap.parse_args()

    # Resolve the checkpoint from the local HF cache; never probe the text-encoder API.
    os.environ.setdefault("LOCAL_CACHE", "true")

    import torch

    from kimodo_runner import (KimodoConstraintSet, KimodoRunner, channel_usage,
                               load_constraint_spec_class)

    ConstraintSpec = load_constraint_spec_class()

    print(f"Loading KimodoRunner on {args.device} (cache: {args.cache}) ...")
    runner = KimodoRunner("Kimodo-G1-RP-v1", device=args.device,
                          cache_path=args.cache, text_encoder=False)
    print(f"model={runner.model_name} fps={runner.fps} joints={len(runner.joint_names)} "
          f"noise_stream_v{runner.noise_stream_version}")
    assert runner.fps == 30.0, f"expected 30 fps, got {runner.fps}"

    T = int(round(args.seconds * runner.fps))
    spec = straight_spec(ConstraintSpec, T, runner.fps, args.speed)
    specs = [spec, spec]

    # Cheap authoring guard before spending GPU time: the mask must actually cover the
    # smooth_root_pos and global_root_heading blocks.
    obs, mask = runner.model.motion_rep.create_conditions_from_constraints_batched(
        [[KimodoConstraintSet(spec, runner.skeleton.root_idx, args.device)]],
        torch.tensor([T], device=args.device),
        to_normalize=True, device=args.device)
    usage = channel_usage(runner.model, mask)
    print(f"channel usage: { {k: v for k, v in usage.items() if v} }")
    assert usage["smooth_root_pos"] == 2 * T, "smooth_root_2d channel not filled as expected"
    assert usage["global_root_heading"] == 2 * T, "heading channel not filled as expected"

    print(f"Generating {len(PROMPTS)} clips: T={T} frames ({args.seconds:.1f} s), "
          f"steps={args.steps}, seeds={SEEDS}")
    samples = runner.generate(PROMPTS, specs, num_frames=T,
                              diffusion_steps=args.steps, seeds=SEEDS)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_z = args.speed * args.seconds
    roots = []
    all_ok = True

    for i, (prompt, seed, sample) in enumerate(zip(PROMPTS, SEEDS, samples)):
        print(f"\n--- sample {i}: seed={seed} prompt={prompt!r}")
        qpos = runner.to_qpos(sample)
        pj = np.asarray(sample["posed_joints"])          # (T, 34, 3), y-up
        root = np.asarray(sample["root_positions"])      # (T, 3), y-up
        roots.append(root)
        print(f"  shapes: posed_joints={pj.shape} qpos={qpos.shape} "
              f"foot_contacts={np.asarray(sample['foot_contacts']).shape}")
        all_ok &= check("shapes", pj.shape == (T, 34, 3) and qpos.shape == (T, 36),
                        f"expected ({T},34,3) and ({T},36)")
        all_ok &= check("finite qpos", np.isfinite(qpos).all(), "all entries finite")

        h_native = root[:, 1]                            # y-up pelvis height
        h_qpos = qpos[:, 2]                              # MuJoCo z-up pelvis height
        all_ok &= check("pelvis height",
                        0.45 < h_native.min() and h_native.max() < 1.05,
                        f"native y [{h_native.min():.3f}, {h_native.max():.3f}] m, "
                        f"qpos z [{h_qpos.min():.3f}, {h_qpos.max():.3f}] m")
        all_ok &= check("height frame agreement",
                        abs(h_native.mean() - h_qpos.mean()) < 0.02,
                        f"means {h_native.mean():.3f} vs {h_qpos.mean():.3f}")

        dz = root[-1, 2] - root[0, 2]
        dx = root[-1, 0] - root[0, 0]
        all_ok &= check("forward progress",
                        abs(dz - target_z) < 0.5 and abs(dx) < 0.4,
                        f"dz={dz:.2f} m (target {target_z:.2f}), |dx|={abs(dx):.2f} m")

        # Tracking against the requested SMOOTH root path. Note the constraint targets
        # the smoothed root; the raw pelvis may legitimately sway a few cm around it.
        err = np.linalg.norm(root[:, [0, 2]] - spec.root_xz, axis=1)
        all_ok &= check("path tracking", err.mean() < 0.15,
                        f"mean {err.mean()*100:.1f} cm, max {err.max()*100:.1f} cm "
                        f"(smooth-root margin is ~6 cm)")

        path = out_dir / f"smoke_{i}_seed{seed}.npz"
        np.savez(path, qpos=qpos, prompt=prompt, seed=seed, fps=runner.fps,
                 **{k: np.asarray(v) for k, v in sample.items()})
        print(f"  saved {path}")

    diff = float(np.abs(roots[0] - roots[1]).max())
    all_ok &= check("samples differ (seeds/prompts)", diff > 1e-4,
                    f"max |root delta| = {diff:.4f} m")

    print(f"\n{'SMOKE PASSED' if all_ok else 'SMOKE FINISHED WITH WARNINGS'} "
          f"-> {out_dir}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
