#!/usr/bin/env bash
# PRIMO-R1 baseline: video-only GRPO RL on Qwen2.5-VL-7B.
#
# This is the ablation baseline for the interleaved input format. It uses
# src/open_r1/grpo.py, which builds a bare video conversation, versus
# run_rl_7b.sh / grpo_interleave.py which anchor the clip between the initial
# and current frames (I_init + V_seq + I_curr).
#
# --temporal    selects T-GRPO (true) vs plain GRPO (false)
# --len_control toggles the length-control reward
# --num_generations is GRPO's group size G
#
# NOTE: this baseline configuration has NOT been re-validated after the repo
# cleanup. It is kept so the video-only setting can be reproduced.
#
# Usage:
#   export SFT_CKPT=/path/to/primo-sft-baseline-7b/checkpoint-XXXX
#   bash src/scripts/run_rl_baseline_7b.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODEL_PATH="${SFT_CKPT:-${MODEL_ROOT}/primo-sft-baseline-7b}"
RUN_NAME="${RUN_NAME:-primo-rl-baseline-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-${R1V_DIR}/log/${RUN_NAME}}"

# Matches run_rl_7b.sh so the two differ only in input format.
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"

export DEBUG_MODE="${DEBUG_MODE:-true}"
export LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/debug_log_rl_baseline.txt}"
export WANDB_PROJECT="${WANDB_PROJECT:-PRIMO-RL}"

mkdir -p "${OUTPUT_DIR}"
primo_banner
echo "[primo] stage           : RL 7B baseline (GRPO, video-only)"
echo "[primo] SFT checkpoint  : ${MODEL_PATH}"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "[primo] ERROR: SFT checkpoint not found at ${MODEL_PATH}" >&2
    echo "[primo] Run src/scripts/run_sft_baseline_7b.sh first, then set SFT_CKPT." >&2
    exit 1
fi

cd "${R1V_DIR}"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/open_r1/grpo.py \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "${RL_DATASET}" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --report_to "${REPORT_TO}" \
    --gradient_checkpointing true \
    --temporal false \
    --len_control true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name "${RUN_NAME}" \
    --save_steps 100 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8
