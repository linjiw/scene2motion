# Setting up Scene2Motion-G1 on a new machine

Written 2026-09-02 from the working install on the RTX 5080 box. Follow it top to bottom; every
pin below is the one the committed receipts bind, so a different commit or checkpoint will be
refused by the drivers' provenance checks rather than silently accepted.

## 0. What you need

| item | used here | notes |
|---|---|---|
| OS | Ubuntu 22.04, kernel 6.8 HWE | Isaac Sim 5.1 needs a glibc ≥ 2.35 distro |
| GPU | RTX 5080 16 GB, driver 595.84 (CUDA 12.8-capable) | ARDY alone needs ~2 GB; one SONIC eval launch needs ≥ 12 GiB free |
| RAM | 32 GB | the CPU text encoder needs ~14 GB free; the SONIC host gate needs ≥ 18 GiB available |
| disk | ~60 GB | Llama-3-8B 15 GB, ARDY checkpoints 1.5 GB, SONIC checkpoint 0.9 GB, Isaac Sim ~15 GB, this repo 0.4 GB |
| accounts | Hugging Face token with access to `meta-llama/Meta-Llama-3-8B-Instruct`; NVIDIA Omniverse EULA acceptance for Isaac Sim |

## 1. Directory layout (keep it exactly)

Eleven source files and `env.sh` carry absolute paths under `/home/linjiw`
(`scene2motion/robot.py`, `scene2motion/sonic_export.py`, `experiments/exp1b_execution_clearance.py`,
`experiments/exp022_exact_tracking_bridge.py`, `experiments/exp011_tracked_addressability.py`,
`experiments/fig_*.py`, `experiments/exp000_*.py`, `experiments/exp001_*.py`). Reproduce the
layout instead of editing them; if your username differs, make `/home/linjiw` a symlink to your
home directory (`sudo ln -s "$HOME" /home/linjiw`).

```
/home/linjiw/
├── scene2motion/                      this repo
├── ardy/                              nv-tlabs/ardy at 693f74d, with .venv (Python 3.11)
├── kimodo/                            nv-tlabs/kimodo at 1aece8c, with .venv (optional, EXP-025)
├── lucid/GR00T-WholeBodyControl/      SONIC tracker fork, branch research/practice-utility
├── isaaclab-install/
│   ├── env_isaaclab/                  Python 3.11 venv with isaacsim 5.1.0 + isaaclab 0.54.2 + matplotlib
│   ├── IsaacLab/                      isaac-sim/IsaacLab at tag v2.3.2
│   └── env.sh                         (contents in §5)
└── .cache/huggingface/hub/            model snapshots (§3)
```

## 2. ARDY (the frozen motion prior)

```bash
sudo apt install -y cmake build-essential ffmpeg git python3.11 python3.11-venv
git clone https://github.com/nv-tlabs/ardy.git /home/linjiw/ardy
cd /home/linjiw/ardy && git checkout 693f74d
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # torch 2.11.0+cu128 here
pip install -e ".[demo]"          # core + gradio text-encoder server; add [trt] only if you want TensorRT
pip install mujoco==3.12.0 scipy pytest
pip install -U huggingface_hub && hf auth login   # token with Llama-3 access
```

Versions in the working venv: torch 2.11.0+cu128, mujoco 3.12.0, numpy 1.26.4, transformers 5.8.1,
gradio 6.26.0, scipy 1.17.1, pytest 9.1.1. The `ardy` worktree must stay clean at `693f74d`:
every receipt binds the ARDY runtime commit and a source manifest, and the drivers refuse to
resume a campaign whose generator source changed.

## 3. Model checkpoints

Download once; afterwards the repo runs with `HF_HUB_OFFLINE=1` if you like.

```bash
source /home/linjiw/ardy/.venv/bin/activate
hf download nvidia/ARDY-G1-RP-25FPS-Horizon52 --revision 059b8007df0ba194a006a877b59a563955ac7b70
hf download nvidia/ARDY-G1-RP-25FPS-Horizon8          # untested contract, listed in the plan
hf download meta-llama/Meta-Llama-3-8B-Instruct       # gated; text encoder (CPU)
```

Verify the denoiser bytes; the drivers pin this hash (`PINNED_DENOISER_SHA256` in
`experiments/exp023_prompt_handoff.py`):

```bash
sha256sum ~/.cache/huggingface/hub/models--nvidia--ARDY-G1-RP-25FPS-Horizon52/snapshots/059b8007*/denoiser.safetensors
# expect 0c16ac26…
```

## 4. This repository

```bash
git clone git@github.com:linjiw/scene2motion.git /home/linjiw/scene2motion
cd /home/linjiw/scene2motion
source env.sh                         # S2M_ROOT, ARDY_ROOT, S2M_PY, PYTHONPATH, text-encoder env
$S2M_PY -m pytest tests -q            # CPU only, ~400 tests, a few minutes
```

**Prompt-embedding cache.** `outputs/text_cache.npz` is gitignored and holds exactly three
prompts (WALK, STEP, SQUEEZE). Every campaign binds the STEP/WALK embedding bytes in its receipt,
so the safest path is to copy the file from the old machine:

```bash
scp old-box:/home/linjiw/scene2motion/outputs/text_cache.npz outputs/   # 50 KB, sha256 330c6996…
```

If you must regenerate it, start the CPU encoder (needs ~14 GB free RAM; it does not fit next to
the diffusion model on a 16 GB GPU) and encode the three prompts; then compare the sha256 with the
one above before resuming any campaign, and treat a mismatch as a new generator identity:

```bash
$S2M_PY /home/linjiw/ardy/scripts/run_text_encoder_server.py --device cpu &     # serves :9550
$S2M_PY - <<'EOF'
from scene2motion.runner import ArdyRunner
r = ArdyRunner(cache_path="outputs/text_cache.npz", text_encoder=True)
r.encode(["A person walks forward.", "A person steps over an obstacle.",
          "A person steps sideways through a narrow gap."])
EOF
```

**Smoke test of generation + scoring (GPU, ~1 min):**

```bash
$S2M_PY - <<'EOF'
from scene2motion.runner import ArdyRunner
r = ArdyRunner(cache_path="outputs/text_cache.npz")
assert r.noise_stream_version == 2
out = r.generate(["A person walks forward."], [None], num_frames=104, seeds=[0])
print(len(out), sorted(out[0].keys()))                             # one clip, 25 fps
EOF
$S2M_PY -c "from scene2motion.host_gate import host_resource_report; print(host_resource_report())"
```

## 5. Isaac Lab + SONIC (the tracker)

Isaac Sim is installed from pip into its own Python 3.11 venv; Isaac Lab is a source checkout
at v2.3.2. Accept the Omniverse EULA in the environment or the first launch blocks on a prompt.

```bash
mkdir -p /home/linjiw/isaaclab-install && cd /home/linjiw/isaaclab-install
python3.11 -m venv env_isaaclab && source env_isaaclab/bin/activate
echo "setuptools<81" > build-constraints.txt
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab && git checkout v2.3.2
./isaaclab.sh --install            # isaaclab 0.54.2 and friends into the active venv
pip install matplotlib             # the only interpreter here with matplotlib; figure scripts use it
cat > /home/linjiw/isaaclab-install/env.sh <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
source /home/linjiw/isaaclab-install/env_isaaclab/bin/activate
export UV_BUILD_CONSTRAINT=/home/linjiw/isaaclab-install/build-constraints.txt
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACLAB_PATH=/home/linjiw/isaaclab-install/IsaacLab
EOF
```

SONIC lives in a fork with the research branch the receipts pin:

```bash
mkdir -p /home/linjiw/lucid && cd /home/linjiw/lucid
git clone -b research/practice-utility https://github.com/linjiw/GR00T-WholeBodyControl-lucid.git GR00T-WholeBodyControl
cd GR00T-WholeBodyControl && git checkout ca86b5e        # commit observed 2026-09-02; re-pin in every receipt
source /home/linjiw/isaaclab-install/env.sh
pip install -e .
python download_from_hf.py            # fetches sonic_release/last.pt + config.yaml from nvidia/GEAR-SONIC
sha256sum sonic_release/last.pt       # expect e6bdab3f…
```

The branch on the old machine was 66 commits ahead of GitHub on 2026-09-02; push it before you
rely on the clone (`git push origin research/practice-utility` from the old box).

Launch smoke test (headless, 2 envs, ~1 min after Isaac warm-up):

```bash
source /home/linjiw/isaaclab-install/env.sh && cd /home/linjiw/lucid/GR00T-WholeBodyControl
python -m gear_sonic.eval_agent_trl +checkpoint=sonic_release/last.pt +headless=True ++num_envs=2 ++run_eval_loop=False
```

Rendering the findings-page videos needs `MUJOCO_GL=glfw` and a display; EGL and OSMesa failed
on the reference box.

## 6. Kimodo (optional, EXP-025 only)

```bash
git clone https://github.com/nv-tlabs/kimodo.git /home/linjiw/kimodo && cd /home/linjiw/kimodo && git checkout 1aece8c
python3.11 -m venv .venv && .venv/bin/pip install -e .
hf download nvidia/Kimodo-G1-RP-v1        # snapshot 3020ad8c…
```

No text encoder is needed: EXP-025 copies ARDY's cached STEP embedding. See
`docs/kimodo-provenance-2026-08-31.md` and `docs/ramp-exp025-kimodo-cross-prior-protocol.md`.

## 7. Where the research stands and how to continue it

Read, in this order: `CLAUDE.md` (orientation and house rules), `docs/plan-2026-09-01-icra2027.md`
(plan of record, Sep 6 number freeze, Sep 15 ICRA deadline), `docs/REPORT.md` §40–42 (latest
ledger), `docs/ramp-exp024-kinematic-stage-2026-09-02.md` (last landed result).

State on 2026-09-02: EXP-024's kinematic stage is complete and committed (128 references, seeds
4600–4631, per-clip gate predictions frozen at HEAD blob `51e1a5a`). The next step is the SONIC
chain, which needs the SONIC host gate (≥ 12 GiB free VRAM, ≥ 18 GiB available RAM, no other
Isaac process):

```bash
cd /home/linjiw/scene2motion && source env.sh
git status --short              # must be empty: the drivers refuse a dirty worktree
# 1. EXP-028: termination-free SONIC rollouts on the exp021 clips (protocol docs/ramp-exp028-*.md)
$S2M_PY experiments/exp028_termination_free_rollouts.py --stage all --out outputs/exp028_termination_free_rollouts
# 2. EXP-024 SONIC stage on the committed predictions, then the analysis
$S2M_PY experiments/exp024_reference_contract.py --stage sonic --require-committed-predictions --out outputs/exp024_reference_contract
$S2M_PY experiments/exp024_reference_contract.py --stage analyze --out outputs/exp024_reference_contract
```

Or let the poller fire the chain when the host frees up (it re-checks every 20 s for up to
`MAX_S` seconds and never relaunches a non-empty output directory):

```bash
REQUIRE_NO_ISAAC=1 MIN_VRAM=12300 MIN_RAM=18500 MAX_S=43200 \
  bash experiments/launch_when_host_free.sh outputs/exp028_termination_free_rollouts /path/to/poll.log -- \
  bash -c 'source env.sh; $S2M_PY experiments/exp028_termination_free_rollouts.py --stage all --out outputs/exp028_termination_free_rollouts && $S2M_PY experiments/exp024_reference_contract.py --stage sonic --require-committed-predictions --out outputs/exp024_reference_contract && $S2M_PY experiments/exp024_reference_contract.py --stage analyze --out outputs/exp024_reference_contract'
```

After the chain lands: write `docs/ramp-exp028-result-*.md` and `docs/ramp-exp024-result-*.md`,
add REPORT §43, fill the `[EXP-024-*]` and `[EXP-028-*]` macros in the paper draft, regenerate
Fig. 4/5 (`experiments/fig_cost_curve.py`, `experiments/fig_contract_gate.py` under the Isaac
venv python), and add a row per artifact to `docs/ai-disclosure-and-review-log.md`. Then, in
plan order: EXP-024b (jump control), EXP-025 (Kimodo), EXP-027 (prompt battery; needs the CPU
encoder), EXP-026 (duck contract; CPU; no protocol yet).

Rules that bite on a fresh machine:

- Commit before launching; the harnesses raise on a dirty tree, a non-empty `--out`, or a
  sampler that is not noise-stream v2.
- Never rerun a refused gate on the same seeds. Fresh seeds go in a fresh output directory;
  the first free block is 4700+ (EXP-025 reserves 4700–4763, EXP-027 4800–4927).
- Never regenerate to finish a killed campaign; re-score the archives through byte-identical
  sources (`experiments/exp023_prompt_handoff_resume_analysis.py` is the pattern).
- Committed `outputs/**/qpos.npz` archives are the evidence; `run/`, `*.log`, `motions.pkl`
  and `text_cache.npz` are machine-local and gitignored.

## 8. Rebuilding figures and the findings page

```bash
IPY=/home/linjiw/isaaclab-install/env_isaaclab/bin/python
$IPY experiments/fig_contract_gate.py && $IPY experiments/fig_cost_curve.py
$IPY experiments/fig_channel_funnel.py && $IPY experiments/fig_channel_response.py
$S2M_PY experiments/export_demo_motions.py
MUJOCO_GL=glfw $S2M_PY experiments/render_demo_videos.py
$S2M_PY docs/site/build.py        # -> docs/site/index.html (standalone) and artifact.html
```

The findings page's staged numbers, timing absolutes and dataset totals are flagged stale in
`CLAUDE.md`; regenerate before publishing.
