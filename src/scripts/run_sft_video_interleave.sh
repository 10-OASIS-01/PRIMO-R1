# 回到正确的工作目录
cd /mnt/pfs/pg4hw0/yibin_workspace/yaxing/Video-R1/src/r1-v

# 直接运行训练命令
export DEBUG_MODE="true"
export LOG_PATH="./debug_log_sft_interleave.txt"
export WANDB_MODE="online"
export WANDB_PROJECT="SFT" 
export WANDB_ENTITY="physical-agentic"
export WANDB_API_KEY="4616b49d5c4f6a63ad27a480ea39bc33712b8790"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VIDEO_DATA_ROOT="/mnt/pfs/pg4hw0/yibin_workspace/PRIMO-Data"

SFT_DATASET="primo-sft-agibot,primo-sft-behavior-1k,primo-sft-robovqa,primo-sft-robotwin-clean,primo-sft-robotwin-randomized,primo-sft-nextqa,primo-sft-perceptiontest,primo-sft-seed-bench-r1,primo-sft-star,primo-sft-sharerobot"

# 避免 NCCL 超时
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/sft_interleave.py \
    --output_dir "./log/Qwen2.5-VL-7B-Video-sft-interleave-primo" \
    --model_name_or_path "/mnt/pfs/pg4hw0/yibin_workspace/model/Qwen2.5-VL-7B-Instruct" \
    --dataset_name "${SFT_DATASET}" \
    --seed 42 \
    --deepspeed local_scripts/zero3.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name PRIMO_SFT_NEW \
    --save_steps 1000 \
    --max_grad_norm 4 \
    --save_only_model true