cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_sft_action.txt"
export WANDB_MODE=offline
 
#  # 设置CUDA内存分配策略以避免OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# 可选：在此处设置统一随机种子（影响数据打乱复现）。
SEED=42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.2" \
    --master_port="12345" \
    src/open_r1/sft_video.py \
    --output_dir "./log/Qwen2.5-VL-7B-Video-sft-Ablation-All" \
    --model_name_or_path "/mnt/pfs/pg4hw0/yibin_workspace/model/Qwen2.5-VL-7B-Instruct" \
    --dataset_name "perceptiontest_cot,nextqa_cot,star_cot,seed-bench-r1_cot,behavior-1k_train,spaciallogic_split_train,robotwin_subtask_clean_train,robotwin_subtask_randomized_train" \
    --seed ${SEED} \
    --deepspeed local_scripts/zero3.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-7B-Video-sft-Ablation-All \
    --save_steps 1000 \
    --max_grad_norm 4 \
    --save_only_model true \