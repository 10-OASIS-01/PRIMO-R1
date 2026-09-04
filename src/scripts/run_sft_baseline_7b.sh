#!/usr/bin/env bash
# PRIMO-R1 baseline: video-only SFT on Qwen2.5-VL-7B-Instruct.
#
# This is the ablation baseline for the interleaved input format. It uses
# src/open_r1/sft_video.py, which feeds the model a bare video clip, versus
# run_sft_7b.sh / sft_interleave.py which anchor the clip between the initial
# and current frames (I_init + V_seq + I_curr).
#
# Paper Table 4 is the corresponding comparison: dropping the anchor frames
# costs accuracy on progress estimation.
#
# NOTE: this baseline configuration has NOT been re-validated after the repo
# cleanup. It is kept so the video-only setting can be reproduced.
#
# Usage:
#   export MODEL_ROOT=/path/to/models VIDEO_DATA_ROOT=/path/to/PRIMO-Data
#   bash src/scripts/run_sft_baseline_7b.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Qwen2.5-VL-7B-Instruct}"
RUN_NAME="${RUN_NAME:-primo-sft-baseline-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-${R1V_DIR}/log/${RUN_NAME}}"

# Matches run_sft_7b.sh so the two differ only in input format.
LEARNING_RATE="${LEARNING_RATE:-1.0e-6}"

export DEBUG_MODE="${DEBUG_MODE:-true}"
export LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/debug_log_sft_baseline.txt}"
export WANDB_PROJECT="${WANDB_PROJECT:-PRIMO-SFT}"

mkdir -p "${OUTPUT_DIR}"
primo_banner
echo "[primo] stage           : SFT 7B baseline (video-only, no interleaved frames)"
echo "[primo] base model      : ${MODEL_PATH}"

cd "${R1V_DIR}"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/open_r1/sft_video.py \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "${SFT_DATASET}" \
    --seed 42 \
    --deepspeed local_scripts/zero3.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate "${LEARNING_RATE}" \
    --logging_steps 1 \
    --bf16 \
    --report_to "${REPORT_TO}" \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name "${RUN_NAME}" \
    --save_steps 1000 \
    --max_grad_norm 4 \
    --save_only_model true
