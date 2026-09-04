# PRIMO R1: From Passive Observer to Active Critic

**Reinforcement Learning Elicits Process Reasoning for Robotic Manipulation**

[[📖 Paper](https://arxiv.org/abs/2603.15600)] [[🌐 Project Page](https://10-oasis-01.github.io/primo-r1-website/)] [[🤗 Model/Dataset](https://huggingface.co/collections/LeonOverload/primo-r1)]

Yibin Liu, Yaxing Lyu, Daqi Gao, Zhixuan Liang, Weiliang Tang, Shilong Mu, Xiaokang Yang, Yao Mu

---

## About

<p align="center">
  <img src="assets/mainfigure.png" alt="PRIMO R1 overview: in-domain and OOD environments, the interleaved I_init + V_seq + I_curr input, and the planning/observation/reasoning trace" width="100%">
</p>

Accurate process supervision is a bottleneck for long-horizon robotic manipulation. Video MLLMs trained under a pure SFT paradigm behave as passive **Observers**: they recognise what is happening rather than judging the current state against the final goal.

PRIMO R1 (**P**rocess **R**easoning **I**nduced **MO**nitoring) is a 7B framework that turns a video MLLM into an active **Critic**. Two ideas carry the work:

1. **Outcome-based RL for explicit reasoning.** GRPO on a verifiable progress reward incentivises chain-of-thought generation for progress estimation, instead of imitating CoT token-by-token.
2. **Structured temporal input.** The video sequence is anchored between an initial-state and a current-state image — the *interleaved* format `I_init + V_seq + I_curr` — rather than fed as a bare clip.

Training is two staged: an SFT cold start, then GRPO RL on top of that checkpoint.

## Results

Progress estimation, Mean Relative Accuracy (MRA↑) and Mean Absolute Error (MAE↓), averaged over four environments (paper Table 1):

| Model | Avg MRA ↑ | Avg MAE ↓ |
| --- | --- | --- |
| GPT-4o | 79.33 | 20.67 |
| GPT-5 mini | 75.38 | 23.96 |
| Qwen2.5-VL-72B | 73.80 | 23.80 |
| Qwen2.5-VL-7B | 67.79 | 29.99 |
| InternVL 3.5 8B | 71.74 | 28.09 |
| ProgressLM | 78.32 | 20.87 |
| VLAC | 74.90 | 25.10 |
| **PRIMO R1 (7B)** | **82.90** | **15.52** |

The 7B model roughly halves the MAE of specialised progress baselines (27–29 → 15.52) and beats 72B-scale general MLLMs.

Stage ablation, MRA↑ (paper Table 2). ID and OOD columns are averaged from the paper's
per-environment numbers; Overall is the paper's own 7-split average.

| Model | ID avg | OOD avg | Overall |
| --- | --- | --- | --- |
| Qwen2.5-VL-7B (base) | 70.38 | 65.26 | 67.46 |
| SFT only | 81.46 | 77.77 | 79.35 |
| RL only | 81.71 | 72.97 | 76.72 |
| **PRIMO R1 (SFT+RL)** | **88.47** | **82.90** | **85.28** |

RL without SFT underperforms — the model struggles to discover the output format and reasoning structure from scratch. SFT alone generalises poorly out of domain (67.30 MRA on the real-humanoid cross-environment split).

Zero-shot failure detection on RoboFail (paper Table 3): PRIMO R1 reaches **67.0%**, matching Gemini 2.0 Flash and above GPT-4o (63.0). Note SFT alone *regresses* to 51.0 from the 57.6 base — format overfitting that RL corrects.

Input-modality ablation (paper Table 4) confirms the interleaved design: a current-state image alone gives 59.50 avg MAE, adding the video sequence drops it to ~31–37, and the full `I_init + V_seq + I_curr` gives the best Acc@10 (28.89).

## Models and data

Everything is released on Hugging Face, collected under [🤗 PRIMO R1](https://huggingface.co/collections/LeonOverload/primo-r1). Each repo's card is self-contained; the sizes below are what you actually download.

| Repo | What it is | Size |
| --- | --- | --- |
| [PRIMO-R1-7B](https://huggingface.co/LeonOverload/PRIMO-R1-7B) | final SFT+RL checkpoint — use this one | 16.6 GB |
| [PRIMO-COT-SFT-7B](https://huggingface.co/LeonOverload/PRIMO-COT-SFT-7B) | stage-1 cold start; the SFT-only ablation, and the RL starting point | 16.6 GB |
| [primo-bench-json](https://huggingface.co/datasets/LeonOverload/primo-bench-json) | benchmark annotations, 7 splits / 23,704 samples | 71 MB |
| [primo-sft-json](https://huggingface.co/datasets/LeonOverload/primo-sft-json) | stage-1 CoT annotations, 10 subsets / 116,755 records | 856 MB |
| [primo-rl-json](https://huggingface.co/datasets/LeonOverload/primo-rl-json) | stage-2 RL annotations, 6 subsets / 328,454 records | 1.7 GB |
| [primo-video-media](https://huggingface.co/datasets/LeonOverload/primo-video-media) | videos + pre-extracted anchor frames, multipart zip | **6.58 TB** |

<p align="center">
  <img src="assets/dataset_dist.png" alt="Dataset distribution for SFT (left), RL (middle), and PRIMO Bench (right)" width="100%">
</p>

<p align="center"><em>Dataset distribution for SFT (left), RL (middle), and PRIMO Bench (right).</em></p>

The RL panel shows the 182k mixture the paper trained on; `primo-rl-json` ships 328,454 records because it releases all available `behavior-1k` annotations (206,029) rather than the 60,000 sampled for training. Every other RL subset matches the figure exactly.

**The annotations are small; the videos are not, and they are extremely skewed.** `behavior-1k` alone is 5,981 GB — 91% of the media release. Skip it unless you specifically need the long-horizon progress-estimation results. `robotwin` is 12.5 GB and covers both benchmark robotwin splits plus both robotwin training subsets, so it is the cheapest way to get a working run.

A minimal end-to-end setup — the model plus one benchmark environment, roughly 29 GB of download:

```bash
export VIDEO_DATA_ROOT=/path/to/PRIMO-Data
export MODEL_ROOT=/path/to/models

# Model
hf download LeonOverload/PRIMO-R1-7B --local-dir "$MODEL_ROOT/PRIMO-R1-7B"

# Benchmark annotations -> $VIDEO_DATA_ROOT/primo-bench/
hf download LeonOverload/primo-bench-json --repo-type dataset \
    --include "raw_json/*" --local-dir /tmp/primo-bench
cp -r /tmp/primo-bench/raw_json/primo-bench "$VIDEO_DATA_ROOT/"

# Videos for the robotwin splits -> $VIDEO_DATA_ROOT/primo-video/
hf download LeonOverload/primo-video-media --repo-type dataset \
    --include "robotwin.z*" --local-dir /tmp/primo-video-zips
7z x /tmp/primo-video-zips/robotwin.zip -o"$VIDEO_DATA_ROOT/primo-video/"

bash src/eval/src/eval_interleave_local.sh
```

Training data follows the same shape — see the [SFT](https://huggingface.co/datasets/LeonOverload/primo-sft-json) and [RL](https://huggingface.co/datasets/LeonOverload/primo-rl-json) cards for the per-subset download commands and the `SFT_DATASET` / `RL_DATASET` mixtures that drop `behavior-1k`.

`PRIMO-R1-7B`'s DeepSpeed resume state from RL step 2500 lives on a separate `training-state` branch, so a default download is 16.6 GB rather than ~116 GB:

```python
from huggingface_hub import snapshot_download

snapshot_download("LeonOverload/PRIMO-R1-7B", local_dir="models/PRIMO-R1-7B")   # inference
snapshot_download("LeonOverload/PRIMO-R1-7B", revision="training-state")        # resume RL
```

Note that `hf download`'s `--include` / `--exclude` are silently ignored if you also pass filenames positionally.

## Install

Requires Linux with CUDA GPUs. Local macOS work is limited to reading and editing code.

```bash
conda create -n primo-r1 python=3.11 && conda activate primo-r1
cd PRIMO-R1
bash setup.sh
```

`setup.sh` installs, in this order:

1. `src/r1-v` editable, which pulls in `torch`, `vllm==0.7.2`, `trl==0.16.0`, DeepSpeed and the eval dependencies.
2. `tensorboardx` and `flash-attn` (the latter with `--no-build-isolation`, since it needs an existing torch).
3. The vendored `src/qwen-vl-utils` editable with the `decord` extra — **after** step 1, so this local copy shadows any `qwen_vl_utils` a transitive dependency pulled from PyPI.
4. The vendored `transformers-main/` tree — **last**, so nothing replaces it.

The pins are not cosmetic. Qwen2.5-VL support shifts between transformers releases, and installing a PyPI `transformers` over the vendored tree is the usual cause of shape and processor errors. `vllm==0.7.2` and `trl==0.16.0` are likewise pinned by upstream R1-V. `transformers` is deliberately absent from `setup.py`'s dependency lists so pip cannot silently replace the vendored install.

Verify:

```bash
python -c "import transformers, trl, vllm, qwen_vl_utils; print(transformers.__version__, trl.__version__, vllm.__version__)"
```

## Data layout

The published repos are packaged to drop straight into this tree — annotations go to `$VIDEO_DATA_ROOT/`, archives extract to `$VIDEO_DATA_ROOT/primo-video/`:

```
$VIDEO_DATA_ROOT/
├── primo-bench/<source>/{id,ood}.json       <- primo-bench-json  raw_json/
├── primo-sft/<source>/train_cot.json        <- primo-sft-json    raw_json/
├── primo-rl/<source>/train.json             <- primo-rl-json     raw_json/
└── primo-video/<group>/{videos,frames}/     <- primo-video-media archives
```

`raw_json/` is what the entry points read. The `jsonl/` shards in the same repos exist only to drive the Dataset Viewer, so `load_dataset(...)` is for browsing, not for training.

Two environment variables locate everything:

| Variable | Meaning | Default |
| --- | --- | --- |
| `VIDEO_DATA_ROOT` | root of the PRIMO data tree | `PRIMO-R1/data` |
| `MODEL_ROOT` | checkpoint directory | `PRIMO-R1/models` |

`src/r1-v/src/open_r1/DatasetLoader.py` is the **single dataset registry**. It maps a logical name to a JSON manifest plus a video root under `VIDEO_DATA_ROOT` and returns `(json_list, video_paths)`. Add new datasets to its `cfg` dict rather than threading raw paths through scripts.

23 registered datasets, named `primo-{sft,rl,bench}-<source>`:

- **SFT (10)**: `agibot`, `behavior-1k`, `robovqa`, `robotwin-clean`, `robotwin-randomized`, `nextqa`, `perceptiontest`, `seed-bench-r1`, `star`, `sharerobot`
- **RL (6)**: `agibot`, `behavior-1k`, `robovqa`, `robotwin-clean`, `robotwin-randomized`, `sharerobot`
- **Bench (7)**: `primo-bench-{id,ood}-agibot`, `primo-bench-{id,ood}-behavior-1k`, `primo-bench-{id,ood}-robotwin`, `primo-bench-ood-real-humanoid`

A `-cot` suffix selects the CoT variant; SFT datasets force CoT filtering (`select == true` plus the four required fields `process`, `planning`, `observation`, `reasoning`).

The interleaved input format needs a first and last frame per video. **The published archives already ship them**, and the published JSON already carries `init_frame_path` / `current_frame_path`, so there is no preprocessing step for the released data. For datasets you add yourself:

```bash
python src/preprocess_video_frames.py                       # all 23 datasets
python src/preprocess_video_frames.py --only-behavior       # one subset
```

This writes `<VIDEO_DATA_ROOT>/primo-video/<source>/frames/` and adds `init_frame_path` / `current_frame_path` to each JSON record in place, backing up the original to `.json.backup`. Records whose extraction fails are left untouched rather than dropped.

Training reads those two fields from the JSON; evaluation ignores them and re-derives both frames with OpenCV at run time.

## Training

Six launchers, all reading paths from the environment. `src/scripts/common.sh` holds the shared setup and is sourced, not run.

```bash
# Stage 1: SFT cold start
bash src/scripts/run_sft_7b.sh
bash src/scripts/run_sft_3b.sh

# Stage 2: GRPO RL, starting from the stage-1 checkpoint
export SFT_CKPT=/path/to/primo-sft-7b/checkpoint-XXXX
bash src/scripts/run_rl_7b.sh
bash src/scripts/run_rl_3b.sh
```

To skip stage 1 entirely, point `SFT_CKPT` at the published cold start:

```bash
hf download LeonOverload/PRIMO-COT-SFT-7B --local-dir "$MODEL_ROOT/PRIMO-COT-SFT-7B"
export SFT_CKPT="$MODEL_ROOT/PRIMO-COT-SFT-7B"
bash src/scripts/run_rl_7b.sh
```

| | SFT 7B | SFT 3B | RL 7B | RL 3B |
| --- | --- | --- | --- | --- |
| base | Qwen2.5-VL-7B-Instruct | Qwen2.5-VL-3B-Instruct | SFT ckpt | SFT ckpt |
| entry point | `sft_interleave.py` | same | `grpo_interleave.py` | same |
| learning rate | `1.0e-6` | `1.0e-6` | `1e-6` | `1e-6` |
| per-device batch | 1 | 1 | 1 | 1 |
| grad accum | 8 | 8 | 1 | 1 |
| DeepSpeed | `zero3.json` | `zero2.json` | `zero3.json` | `zero2.json` |
| `num_generations` (G) | — | — | 8 | 8 |
| `max_pixels` | — | — | 401408 | 401408 |

**The 3B settings have not been validated.** They are scaled from the 7B recipe (ZeRO-2 instead of ZeRO-3) and are marked as untested in each script. The released checkpoint is 7B.

### Video-only baselines

Two extra launchers keep the pre-interleave recipe reproducible. They are the input-format ablation: same data, same hyper-parameters, but the model sees a bare clip instead of `I_init + V_seq + I_curr`.

```bash
bash src/scripts/run_sft_baseline_7b.sh    # src/open_r1/sft_video.py
export SFT_CKPT=/path/to/primo-sft-baseline-7b/checkpoint-XXXX
bash src/scripts/run_rl_baseline_7b.sh     # src/open_r1/grpo.py
```

`sft_video.py` and `grpo.py` are the video-only entry points; `sft_interleave.py` and `grpo_interleave.py` are the ones that produced the released checkpoint. Paper Table 4 is the corresponding comparison — dropping the anchor frames raises avg MAE from ~29 to 31–37, and the current frame alone gives 59.50. These baseline configs have not been re-validated since the cleanup.

Anything can be overridden from the environment, e.g. `RUN_NAME`, `OUTPUT_DIR`, `MAX_COMPLETION_LENGTH`, `SFT_DATASET`, `RL_DATASET`, `CUDA_VISIBLE_DEVICES`. GPU count is auto-detected via `nvidia-smi`.

Two defaults follow the paper rather than the original launcher, which differed:

- SFT learning rate is `1.0e-6` (paper Table 9); the original script used `2e-6`.
- RL `max_completion_length` is `4096` (paper Table 10); the original script used `768`. Lower it if you hit OOM.

Invariants inherited from R1-V: keep `per_device_train_batch_size=1`. `--temporal` selects T-GRPO vs plain GRPO, `--len_control` toggles the length-control reward, and `--num_generations` is GRPO's group size G — lowering it is cheaper but raises gradient variance. Reward functions live in `reward_funcs_registry` in `grpo_interleave.py` (`accuracy_reward`, `format_reward`); add new ones there so `--reward_funcs` can select them.

W&B is read from the environment only. Set `WANDB_API_KEY` to enable logging; the scripts warn and fall back to `report_to=none` when it is unset.

## Evaluation

Four launchers, run from the project root. `src/eval/README.md` is the detailed reference.

```bash
bash src/eval/src/eval_baseline_local.sh     # local baselines, vLLM  -> eval_outputs/baseline/
bash src/eval/src/eval_interleave_local.sh   # PRIMO R1 models        -> eval_outputs/interleave/
bash src/eval/src/eval_ablation.sh           # input-modality sweep   -> eval_outputs/ablation/
bash src/eval/src/eval_api.sh                # remote API models      -> eval_outputs/api/
```

Results land in `src/r1-v/eval_outputs/<kind>/<model_name>/<dataset>.json`, with per-sample records under `results` and a summary under `final_acc`. All harnesses resume from an existing output file.

**Models and datasets are selected by a comment-toggled heredoc**, not CLI flags. Edit `model_paths` / `file_names` in the launcher and comment out lines with `#`:

```bash
# src/eval/src/eval_interleave_local.sh — the two released models
model_paths=$(cat <<EOF | grep -v '^#' | grep -v '^$'
$MODEL_ROOT/PRIMO-R1-7B
# $MODEL_ROOT/PRIMO-COT-SFT-7B
EOF
)

file_names=$(cat <<EOF | grep -v '^#' | grep -v '^$'
primo-bench-ood-robotwin
# primo-bench-id-robotwin
EOF
)
```

Reproduction details, including which video group each split needs, are on the [primo-bench-json](https://huggingface.co/datasets/LeonOverload/primo-bench-json) card.

`eval_baseline_local.sh` picks its Python entry point by variable: `src/eval/eval_local.py` for Qwen2.5-VL / Cosmos-Reason1 / RoboBrain / ProgressLM, or `src/eval/eval_internvl.py` for InternVL 3.5 (which also needs `MAX_MODEL_LEN=40960`).

`eval_api.sh` requires both `API_KEY` and `API_URL` (an OpenAI-compatible `/v1/chat/completions` endpoint). Neither has a default.

Decoding follows the Qwen2.5-VL demo: `top_p=0.001`, `temperature=0.01`. Larger `top_p` produces garbled output. Training caps videos at 16 frames; eval samples more at higher resolution.

`eval_ablation_modality.py` sweeps the six input combinations — `current_only`, `init_current`, `video_only`, `video_current`, `init_video`, `init_video_current` — to isolate each component's contribution.

### Metrics, and one divergence worth knowing

Answers are scored per `problem_type` by `reward_fn`: `multiple choice` and `boolean` are exact match, `free-form` is mean ROUGE-1/2/L F-measure, and `numerical` / `regression` are relative-accuracy scores.

**The harnesses score `regression` with three different formulas.** They are not comparable, so never place their numbers in the same column. Each harness records which one it used in the `regression_metric` field of its output JSON:

| Harness | `regression_metric` | Formula |
| --- | --- | --- |
| `eval_interleave.py` | `linear_relative_accuracy` | `1 - abs(pred-gt)/abs(gt)`, clipped to [0,1] |
| `eval_local.py`, `eval_api.py`, `eval_internvl.py` | `threshold_relative_accuracy` | fraction of thresholds t ∈ [0.5, 0.95] step 0.05 where relative error < 1-t |
| `eval_ablation_modality.py` | `absolute_range_accuracy` | `1 - abs(pred-gt)/100`, against a fixed 0–100 progress range |

The published PRIMO R1 numbers come from `linear_relative_accuracy`. The threshold variant is stricter and takes only 10 discrete values per sample. The math is preserved exactly as it was when the paper's numbers were produced; only the names were unified.

The ablation harness additionally reports MAE, RMSE, Acc@5 and Acc@10 via `compute_metrics`.

### Shared modules

Two modules under `src/` are the single source of truth for everything the entry points used to define locally. Import from them rather than pasting a copy — that is how the prompts drifted in the first place (`SYSTEM_PROMPT` appeared in five files, `TYPE_TEMPLATE` in two silently different variants).

`src/primo_prompts.py` — every prompt:

```python
from primo_prompts import SYSTEM_PROMPT, QUESTION_TEMPLATE, TYPE_TEMPLATE, build_question

text = build_question(question, problem_type)   # QUESTION_TEMPLATE.format(...) + the type hint
```

| Name | Used by |
| --- | --- |
| `SYSTEM_PROMPT`, `QUESTION_TEMPLATE`, `TYPE_TEMPLATE` | PRIMO R1 — the released checkpoint's format |
| `QUESTION_TEMPLATE_BASELINE`, `TYPE_TEMPLATE_BASELINE` | video-only baselines, the CoT generator, `inference_example.py` |
| `QUESTION_TEMPLATE_SFT_VIDEO` | `sft_video.py`, the video-only SFT baseline |
| `QUESTION_TEMPLATE_PARSER_ONLY` | `eval_api.py` — remote models wrap answers in prose |
| `QUESTION_TEMPLATE_QUESTION_ONLY` | `eval_internvl.py` — InternVL supplies its own framing |

The variants are not redundancy. `TYPE_TEMPLATE_BASELINE` is `TYPE_TEMPLATE` minus the `boolean` key, and the baseline numbers in the paper were produced without it, so adding the key would change which records get a type hint. Each is preserved verbatim.

`src/primo_video_utils.py` — the frame helpers behind the interleaved format:

```python
from primo_video_utils import (
    extract_frames_on_demand,       # (init, current) as PIL, LRU-cached
    extract_first_and_last_frame,   # the same two frames written out as JPEGs
    choose_nframes,                 # clamp the requested frame count
    init_frame_placeholder,         # defer extraction until collation
    current_frame_placeholder,
    resolve_placeholders_in_messages,
)
```

`MAX_NFRAMES` (env `INTERLEAVE_MAX_NFRAMES`) defaults to 22, which is the effective cap the published numbers were produced with even though the launchers pass `--nframes 32`. The video-only baseline harness never had an upper bound and passes `max_nframes=None` to keep its own numbers reproducible.

Both modules are importable because `src/scripts/common.sh` puts the repo's `src` on `PYTHONPATH`; each entry point also inserts it itself, so running a script directly with `python` works.

### Prompt coupling

`QUESTION_TEMPLATE` + `TYPE_TEMPLATE` ask the model to ground observations in a procedural plan and emit a strictly formatted final answer. The answer extractors `extract_think` and `extract_answer` are coupled to that format — editing a prompt without updating them will silently collapse scores rather than error out. Treat `primo_prompts.py` and the extractors as one unit.

## Single example

```bash
python ./src/inference_example.py
```

## Lint

`src/r1-v` carries the upstream open-r1 tooling. There is no test suite.

```bash
cd src/r1-v
make style      # black --line-length 119 + isort
make quality    # check-only + flake8
```

## Citation

```bibtex
@misc{liu2026passiveobserveractivecritic,
      title={From Passive Observer to Active Critic: Reinforcement Learning Elicits Process Reasoning for Robotic Manipulation},
      author={Yibin Liu and Yaxing Lyu and Daqi Gao and Zhixuan Liang and Weiliang Tang and Shilong Mu and Xiaokang Yang and Yao Mu},
      year={2026},
      eprint={2603.15600},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.15600},
}
```

## Acknowledgement

Built on [Video-R1](https://github.com/tulerfeng/Video-R1), which is itself built on [R1-V](https://github.com/Deep-Agent/R1-V) and [open-r1](https://github.com/huggingface/open-r1). Thanks to those authors for releasing their code.
