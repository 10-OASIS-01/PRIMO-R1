#!/usr/bin/env bash
# PRIMO-R1 stage 1: SFT cold start on Qwen2.5-VL-7B-Instruct.
#
# This is the configuration used for the released 7B checkpoint.
#
# Usage:
#   export MODEL_ROOT=/path/to/models VIDEO_DATA_ROOT=/path/to/PRIMO-Data
#   bash src/scripts/run_sft_7b.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Qwen2.5-VL-7B-Instruct}"
RUN_NAME="${RUN_NAME:-primo-sft-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-${R1V_DIR}/log/${RUN_NAME}}"

# The paper (Table 9) reports lr 1.0e-6 for SFT; the original launcher used 2e-6.
# The paper value is the default here.
LEARNING_RATE="${LEARNING_RATE:-1.0e-6}"

export DEBUG_MODE="${DEBUG_MODE:-true}"
export LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/debug_log_sft.txt}"
export WANDB_PROJECT="${WANDB_PROJECT:-PRIMO-SFT}"

mkdir -p "${OUTPUT_DIR}"
primo_banner
echo "[primo] stage           : SFT 7B"
echo "[primo] base model      : ${MODEL_PATH}"

cd "${R1V_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/open_r1/sft_interleave.py \
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
