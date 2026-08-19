#!/usr/bin/env bash
# Interleave eval: initial frame + video + current frame.

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

GPUS=2,3
DECORD_EOF_RETRY_MAX=20480
PYTHONPATH=$PRIMO_ROOT/src:$PRIMO_ROOT/src/r1-v/src/open_r1:$PYTHONPATH

export VIDEO_DATA_ROOT
export DECORD_EOF_RETRY_MAX
export PYTHONPATH

INTERLEAVE_EVAL_PY=src/eval/eval_interleave.py

BATCH_SIZE=64
NFRAMES=32
SAMPLE_SIZE=0
# SAMPLE_SIZE=50

model_paths=$(cat <<EOF | grep -v '^#' | grep -v '^$'
$MODEL_ROOT/Qwen2.5-VL-7B-Instruct
# $MODEL_ROOT/Video-R1-7B
# $MODEL_ROOT/Cosmos-Reason1-7B
# $MODEL_ROOT/RoboBrain2.0-7B
EOF
)

file_names=$(cat <<EOF | grep -v '^#' | grep -v '^$'
primo-bench-ood-agibot
# primo-bench-ood-behavior-1k
# primo-bench-ood-real-humanoid
# primo-bench-ood-robotwin
# primo-bench-id-robotwin
EOF
)

cd $PRIMO_ROOT

for model in $model_paths; do
    model_name=$(basename $model)

    for file_name in $file_names; do
        output_path=$EVAL_OUTPUT_ROOT/interleave/$model_name/$file_name.json
        mkdir -p $(dirname $output_path)

        echo "========================================================"
        echo "Time:       $(date)"
        echo "Eval:       interleave"
        echo "Model:      $model"
        echo "File:       $file_name"
        echo "Output:     $output_path"
        echo "Using GPUs: $GPUS"
        echo "========================================================"

        CUDA_VISIBLE_DEVICES=$GPUS python $INTERLEAVE_EVAL_PY             --model_path $model             --file_name $file_name             --output_path $output_path             --batch_size $BATCH_SIZE             --nframes $NFRAMES             --sample_size $SAMPLE_SIZE
    done
done
