#!/usr/bin/env bash
# Ablation eval. Set data path before running.

set -e

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
PRIMO_ROOT=$(cd $SCRIPT_DIR/../../.. && pwd)

if [ -z "$VIDEO_DATA_ROOT" ]; then
    VIDEO_DATA_ROOT=$PRIMO_ROOT/data
fi

if [ -z "$MODEL_ROOT" ]; then
    MODEL_ROOT=$PRIMO_ROOT/models
fi

EVAL_OUTPUT_ROOT=$PRIMO_ROOT/src/r1-v/eval_outputs

GPUS=4,5,6,7
DECORD_EOF_RETRY_MAX=20480
PYTHONPATH=$PRIMO_ROOT/src:$PRIMO_ROOT/src/r1-v/src/open_r1:$PYTHONPATH

export VIDEO_DATA_ROOT
export DECORD_EOF_RETRY_MAX
export PYTHONPATH

# export ABLATION_DATA_FILE=$VIDEO_DATA_ROOT/primo-bench/agibot/ood.json
# export ABLATION_DATASET_NAME=primo-bench-ood-agibot
# export ABLATION_MODEL_PATH=$MODEL_ROOT/Qwen2.5-VL-7B-Instruct

if [ -z "$ABLATION_DATA_FILE" ]; then
    echo "ERROR: please set ABLATION_DATA_FILE=/path/to/data.json"
    exit 1
fi

if [ -z "$ABLATION_DATASET_NAME" ]; then
    echo "ERROR: please set ABLATION_DATASET_NAME=dataset_name"
    exit 1
fi

if [ -z "$ABLATION_MODEL_PATH" ]; then
    ABLATION_MODEL_PATH=$MODEL_ROOT/Qwen2.5-VL-7B-Instruct
fi

ABLATION_BATCH_SIZE=64
ABLATION_NFRAMES=32
ABLATION_GPUS=$GPUS
model_name=$(basename $ABLATION_MODEL_PATH)
OUTPUT_DIR=$EVAL_OUTPUT_ROOT/ablation/$model_name/$ABLATION_DATASET_NAME

modalities=$(cat <<EOF | grep -v '^#' | grep -v '^$'
current_only
init_current
video_only
video_current
init_video
init_video_current
EOF
)

cd $PRIMO_ROOT
mkdir -p $OUTPUT_DIR

for modality in $modalities; do
    echo "========================================================"
    echo "Time:       $(date)"
    echo "Eval:       ablation"
    echo "Model:      $ABLATION_MODEL_PATH"
    echo "Dataset:    $ABLATION_DATASET_NAME"
    echo "Modality:   $modality"
    echo "Output:     $OUTPUT_DIR/results_$modality.json"
    echo "Using GPUs: $ABLATION_GPUS"
    echo "========================================================"

    CUDA_VISIBLE_DEVICES=$ABLATION_GPUS python src/eval/eval_ablation_modality.py         --model_path $ABLATION_MODEL_PATH         --data_file $ABLATION_DATA_FILE         --dataset_name $ABLATION_DATASET_NAME         --modality $modality         --output_dir $OUTPUT_DIR         --test_batch_size $ABLATION_BATCH_SIZE         --nframes $ABLATION_NFRAMES

done
