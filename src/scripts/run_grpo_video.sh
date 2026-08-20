

cd src/r1-v

export DEBUG_MODE="true"
export LOG_PATH="./llm_run——all.txt"
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE="online"
export WANDB_PROJECT="RL" 
export WANDB_ENTITY="physical-agentic"
export WANDB_API_KEY="4616b49d5c4f6a63ad27a480ea39bc33712b8790"
export VIDEO_DATA_ROOT="/mnt/pfs/pg4hw0/yibin_workspace/PRIMO-Data"
 # 设置CUDA内存分配策略以避免OOM
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# QWEN_PATH="/mnt/pfs/pg4hw0/yibin_workspace/model/Qwen2.5-VL-7B-Instruct"
QWEN_PATH="/mnt/pfs/pg4hw0/yibin_workspace/yaxing/Video-R1/src/r1-v/log/Qwen2.5-VL-7B-Video-sft-interleave-Ablation-All/checkpoint-1476"
HF_DATASET="primo-rl-agibot,primo-rl-behavior-1k,primo-rl-robovqa,primo-rl-robotwin-clean,primo-rl-robotwin-randomized,primo-rl-sharerobot"
# HF_DATASET="agibot,behavior-1k"
OUTPUT_DIR="./log/Qwen2.5-VL-7B-Video-GRPO-20260303-primo-only-progress"

if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
RUN_NAME="Qwen2.5-VL-7B-Video-GRPO-20260303-primo-only-progress"
DS_CONFIG="local_scripts/zero3.json"

# For resume training:  --resume_from_checkpoint Model_Path \
# Set temporal to choose between T-GRPO and GRPO, and len_control to enable or disable the length control reward.

# Qwen/Qwen2.5-VL-7B-Instruct

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    src/open_r1/grpo_interleave.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --temporal false \
    --len_control true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance
