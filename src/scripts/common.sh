#!/usr/bin/env bash
# Shared setup for every PRIMO-R1 training launcher.
# Sourced by run_sft_{3b,7b}.sh and run_rl_{3b,7b}.sh — not meant to be run directly.

set -euo pipefail

# Resolve the repository layout from this script's location so the launchers work
# from any working directory.
_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_SCRIPTS_DIR}/../.." && pwd)"
R1V_DIR="${REPO_ROOT}/src/r1-v"

# ---------------------------------------------------------------------------
# Paths — override by exporting these before invoking a launcher.
# ---------------------------------------------------------------------------
export VIDEO_DATA_ROOT="${VIDEO_DATA_ROOT:-${REPO_ROOT}/data}"
export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"

# PYTHONPATH must contain `src` and `src/open_r1` (DatasetLoader is imported as a
# top-level module by the training entry points) plus the repo's own `src`, which
# holds the shared `primo_prompts` / `primo_video_utils` modules.
export PYTHONPATH="${REPO_ROOT}/src:${R1V_DIR}/src:${R1V_DIR}/src/open_r1${PYTHONPATH:+:${PYTHONPATH}}"

# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------
# Defaults to every visible GPU. Override with CUDA_VISIBLE_DEVICES=0,1 ...
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        _GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
    else
        _GPU_COUNT=1
    fi
else
    _GPU_COUNT="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
    # Re-export so the selection reaches torchrun even if it was set as a plain
    # shell variable. When unset we leave it alone: an empty CUDA_VISIBLE_DEVICES
    # hides every device, which would give the workers no GPU at all.
    export CUDA_VISIBLE_DEVICES
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-${_GPU_COUNT}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-12345}"

# NCCL: long video batches can stall collectives past the default timeout.
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------
# WANDB_API_KEY is read from the environment and never hardcoded here.
# Without it, training falls back to `--report_to none`.
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
    export WANDB_ENTITY="${WANDB_ENTITY:-}"
    REPORT_TO="wandb"
else
    echo "[primo] WANDB_API_KEY is not set - disabling W&B reporting." >&2
    REPORT_TO="none"
fi

# ---------------------------------------------------------------------------
# Dataset mixtures (registered in src/r1-v/src/open_r1/DatasetLoader.py)
# ---------------------------------------------------------------------------
SFT_DATASET="${SFT_DATASET:-primo-sft-agibot,primo-sft-behavior-1k,primo-sft-robovqa,primo-sft-robotwin-clean,primo-sft-robotwin-randomized,primo-sft-nextqa,primo-sft-perceptiontest,primo-sft-seed-bench-r1,primo-sft-star,primo-sft-sharerobot}"
RL_DATASET="${RL_DATASET:-primo-rl-agibot,primo-rl-behavior-1k,primo-rl-robovqa,primo-rl-robotwin-clean,primo-rl-robotwin-randomized,primo-rl-sharerobot}"

primo_banner() {
    echo "[primo] repo root       : ${REPO_ROOT}"
    echo "[primo] VIDEO_DATA_ROOT : ${VIDEO_DATA_ROOT}"
    echo "[primo] MODEL_ROOT      : ${MODEL_ROOT}"
    echo "[primo] GPUs            : ${CUDA_VISIBLE_DEVICES:-all} (nproc_per_node=${NPROC_PER_NODE})"
    echo "[primo] report_to       : ${REPORT_TO}"
}
