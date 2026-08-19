#!/usr/bin/env bash
# API eval.

set -e

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
PRIMO_ROOT=$(cd $SCRIPT_DIR/../../.. && pwd)

if [ -z "$VIDEO_DATA_ROOT" ]; then
    VIDEO_DATA_ROOT=$PRIMO_ROOT/data
fi

if [ -z "$API_URL" ]; then
    API_URL=http://35.220.164.252:3888/v1/chat/completions
fi

if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME=claude-haiku-4-5-20251001
fi

EVAL_OUTPUT_ROOT=$PRIMO_ROOT/src/r1-v/eval_outputs
PYTHONPATH=$PRIMO_ROOT/src:$PRIMO_ROOT/src/r1-v/src/open_r1:$PYTHONPATH

export VIDEO_DATA_ROOT
export PYTHONPATH

API_EVAL_PY=src/eval/eval_api.py
WORKERS=4
SAMPLE_SIZE=0
# SAMPLE_SIZE=50

# export API_KEY=sk-xxx

if [ -z "$API_KEY" ]; then
    echo "ERROR: please set API_KEY=sk-xxx"
    exit 1
fi

file_names=$(cat <<EOF | grep -v '^#' | grep -v '^$'
primo-bench-ood-agibot
# primo-bench-ood-behavior-1k
# primo-bench-ood-real-humanoid
# primo-bench-ood-robotwin
# primo-bench-id-robotwin
EOF
)

cd $PRIMO_ROOT

for file_name in $file_names; do
    output_dir=$EVAL_OUTPUT_ROOT/api/$MODEL_NAME
    output_path=$output_dir/$file_name.json
    mkdir -p $output_dir

    echo "========================================================"
    echo "Time:    $(date)"
    echo "Eval:    api"
    echo "Model:   $MODEL_NAME"
    echo "File:    $file_name"
    echo "Output:  $output_path"
    echo "Workers: $WORKERS"
    echo "========================================================"

    python $API_EVAL_PY         --file_name $file_name         --api_url $API_URL         --api_key $API_KEY         --model_name $MODEL_NAME         --workers $WORKERS         --output_path $output_path         --sample_size $SAMPLE_SIZE

    sleep 10

done
