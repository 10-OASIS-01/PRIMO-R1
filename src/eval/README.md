# PRIMO R1 Evaluation

Four launchers live in `src/eval/src/`. Run them from the project root.

```bash
bash src/eval/src/eval_baseline_local.sh     # local baselines, vLLM  -> eval_outputs/baseline/
bash src/eval/src/eval_interleave_local.sh   # PRIMO R1 models        -> eval_outputs/interleave/
bash src/eval/src/eval_ablation.sh           # input-modality sweep   -> eval_outputs/ablation/
bash src/eval/src/eval_api.sh                # remote API models      -> eval_outputs/api/
```

Results land in `src/r1-v/eval_outputs/<kind>/<model_name>/<dataset>.json`, with
per-sample records under `results` and a summary under `final_acc`. Every
harness resumes from an existing output file, so an interrupted run continues
where it stopped.

The benchmark itself — split sizes, record format, and which video group each split needs — is documented on [🤗 primo-bench-json](https://huggingface.co/datasets/LeonOverload/primo-bench-json).

Decoding follows the Qwen2.5-VL demo: `top_p=0.001`, `temperature=0.01`. Larger
`top_p` produces garbled output. Training caps videos at 16 frames; eval samples
more at higher resolution.

## 1. Environment Setup

```bash
cd /path/to/PRIMO-R1
conda activate primo-r1

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
primo-bench-id-agibot
primo-bench-ood-agibot
primo-bench-id-behavior-1k
primo-bench-ood-behavior-1k
primo-bench-id-robotwin
primo-bench-ood-robotwin
primo-bench-ood-real-humanoid
```

The authoritative list is the `cfg` dict in
`src/r1-v/src/open_r1/DatasetLoader.py`; add new splits there rather than
passing raw paths through the scripts.

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

## 8. Metrics, and one divergence worth knowing

Answers are scored per `problem_type` by `reward_fn`: `multiple choice` and
`boolean` are exact match, `free-form` is mean ROUGE-1/2/L F-measure, and
`numerical` / `regression` are relative-accuracy scores.

**The harnesses score `regression` with three different formulas.** They are not
comparable, so never place their numbers in the same column. Each harness records
which one it used in the `regression_metric` field of its output JSON:

| Harness | `regression_metric` | Formula |
| --- | --- | --- |
| `eval_interleave.py` | `linear_relative_accuracy` | `1 - abs(pred-gt)/abs(gt)`, clipped to [0,1] |
| `eval_local.py`, `eval_api.py`, `eval_internvl.py` | `threshold_relative_accuracy` | fraction of thresholds t ∈ [0.5, 0.95] step 0.05 where relative error < 1-t |
| `eval_ablation_modality.py` | `absolute_range_accuracy` | `1 - abs(pred-gt)/100`, against a fixed 0–100 progress range |

The published PRIMO R1 numbers come from `linear_relative_accuracy`. The
threshold variant is stricter and takes only 10 discrete values per sample. The
math is preserved exactly as it was when the paper's numbers were produced; only
the names were unified.

The ablation harness additionally reports MAE, RMSE, Acc@5 and Acc@10 via
`compute_metrics`.

## 9. Input format

"Interleave" is the PRIMO input format — `I_init + V_seq + I_curr` — versus the
baseline's video-only input. `eval_ablation_modality.py` sweeps the six
combinations to isolate each component's contribution: `current_only`,
`init_current`, `video_only`, `video_current`, `init_video`,
`init_video_current`.

Evaluation ignores the `init_frame_path` / `current_frame_path` fields in the
JSON and re-derives both anchor frames with OpenCV at run time, so no
preprocessing step is needed.

Prompts come from `src/primo_prompts.py`, frame helpers from
`src/primo_video_utils.py`. The answer extractors `extract_think` and
`extract_answer` are coupled to the prompt format — editing one without the
other silently collapses scores rather than erroring. See
[`src/README.md`](../README.md).
