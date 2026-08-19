# PRIMO-R1 Evaluation

Evaluation launch scripts are located in `src/eval/src/`. Run the appropriate script from the project root.

## 1. Environment Setup

```bash
cd /path/to/PRIMO-R1
conda activate daqi

export VIDEO_DATA_ROOT=/path/to/PRIMO-Data
export MODEL_ROOT=/path/to/models
```

If `VIDEO_DATA_ROOT` and `MODEL_ROOT` are not set, the scripts use the project's `data/` and `models/` directories by default.

## 2. Choose an Evaluation

| Evaluation | Launch Script | Use Case |
| --- | --- | --- |
| Standard local baseline | `eval_baseline_local.sh` | Compare local models such as Qwen2.5-VL, Cosmos-Reason1, RoboBrain, and ProgressLM |
| PRIMO-R1 interleave | `eval_interleave_local.sh` | Evaluate PRIMO-R1 models |
| Modality ablation | `eval_ablation.sh` | Run modality ablation experiments |
| API evaluation | `eval_api.sh` | Evaluate models through an API |

Before running a local baseline, select the Python entry point in `eval_baseline_local.sh`:

```bash
# Qwen2.5-VL, Cosmos-Reason1, RoboBrain, ProgressLM
BASELINE_EVAL_PY=src/eval/eval_local.py

# InternVL3.5
# BASELINE_EVAL_PY=src/eval/eval_internvl.py
```

## 3. Select Models and Datasets

For standard local baseline and interleave evaluation, edit `model_paths` and `file_names` in the corresponding script. Keep the entries to run and comment out the others with `#`.

```bash
$MODEL_ROOT/Qwen2.5-VL-7B-Instruct
# $MODEL_ROOT/Cosmos-Reason1-7B

primo-bench-ood-agibot
# primo-bench-ood-behavior-1k
```

Available dataset names:

```text
primo-bench-ood-agibot
primo-bench-ood-behavior-1k
primo-bench-ood-real-humanoid
primo-bench-ood-robotwin
primo-bench-id-robotwin
```

## 4. Run the Baseline

Use this for the standard local-model baseline.

```bash
bash src/eval/src/eval_baseline_local.sh
```

Results are saved to:

```text
src/r1-v/eval_outputs/baseline/<model_name>/<dataset>.json
```

## 5. Run PRIMO-R1 Interleave Evaluation

Use this to evaluate PRIMO-R1 models.

```bash
bash src/eval/src/eval_interleave_local.sh
```

Results are saved to:

```text
src/r1-v/eval_outputs/interleave/<model_name>/<dataset>.json
```

## 6. Run Modality Ablation

Configure the data file, dataset name, and model path first:

```bash
export ABLATION_DATA_FILE=$VIDEO_DATA_ROOT/primo-bench/agibot/ood.json
export ABLATION_DATASET_NAME=primo-bench-ood-agibot
export ABLATION_MODEL_PATH=$MODEL_ROOT/Qwen2.5-VL-7B-Instruct
```

Then run:

```bash
bash src/eval/src/eval_ablation.sh
```

To run only selected modalities, comment out the unneeded entries in `modalities` within `eval_ablation.sh`.

Results are saved to:

```text
src/r1-v/eval_outputs/ablation/<model_name>/<dataset>/results_<modality>.json
```

## 7. Run API Evaluation

Configure the API settings:

```bash
export API_KEY=sk-xxx
export API_URL=http://your-api-endpoint/v1/chat/completions
export MODEL_NAME=your-model-name
```

Run:

```bash
bash src/eval/src/eval_api.sh
```

Select datasets in `file_names` within `eval_api.sh`. `WORKERS` controls request concurrency. If `MODEL_NAME` is not set, the default model is `claude-haiku-4-5-20251001`; the default worker count is 4, and the script waits 10 seconds between datasets.

Results are saved to:

```text
src/r1-v/eval_outputs/api/<model_name>/<dataset>.json
```
