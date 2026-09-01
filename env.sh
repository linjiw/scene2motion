# Scene2Motion-G1 environment. Source before any python.
export S2M_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARDY_ROOT=/home/linjiw/ardy
export KIMODO_ROOT=/home/linjiw/kimodo
export SONIC_ROOT=/home/linjiw/lucid/GR00T-WholeBodyControl
export S2M_PY="$ARDY_ROOT/.venv/bin/python"
export PYTHONPATH="$S2M_ROOT:$ARDY_ROOT:$PYTHONPATH"
# Text encoder runs on CPU as a gradio service so the 16GB GPU stays free for the diffusion model.
export TEXT_ENCODER_MODE=auto
export TEXT_ENCODER_URL=http://127.0.0.1:9550/
export TEXT_ENCODER_DEVICE=cpu
export HF_HUB_OFFLINE=0
