# PRIMO R1 Training

Six launchers live here. `common.sh` holds the shared setup and is *sourced* by
each launcher, not run directly.

Training is two staged: an SFT cold start, then GRPO RL on top of that
checkpoint. `sft_interleave.py` and `grpo_interleave.py` are the entry points
that produced the released checkpoint.

## 1. Environment

```bash
cd /path/to/PRIMO-R1
conda activate primo-r1

export VIDEO_DATA_ROOT=/path/to/PRIMO-Data
export MODEL_ROOT=/path/to/models
```

Both default to the project's `data/` and `models/` directories. `common.sh`
puts `src`, `src/r1-v/src`, and `src/r1-v/src/open_r1` on `PYTHONPATH`, so the
shared `primo_prompts` / `primo_video_utils` modules and `DatasetLoader` all
resolve.

GPU count is auto-detected via `nvidia-smi`. Set `CUDA_VISIBLE_DEVICES` to
restrict the selection; `NPROC_PER_NODE` follows from it.

W&B is read from the environment only. Set `WANDB_API_KEY` to enable logging;
without it the launchers warn and fall back to `--report_to none`.

## 2. Stage 1 — SFT cold start

```bash
bash src/scripts/run_sft_7b.sh
bash src/scripts/run_sft_3b.sh
```

## 3. Stage 2 — GRPO RL

```bash
export SFT_CKPT=/path/to/primo-sft-7b/checkpoint-XXXX
bash src/scripts/run_rl_7b.sh
bash src/scripts/run_rl_3b.sh
```

The launcher fails fast with a clear message if `SFT_CKPT` does not point at an
existing directory.

To skip stage 1 entirely, point `SFT_CKPT` at the published cold start:

```bash
hf download LeonOverload/PRIMO-COT-SFT-7B --local-dir "$MODEL_ROOT/PRIMO-COT-SFT-7B"
export SFT_CKPT="$MODEL_ROOT/PRIMO-COT-SFT-7B"
bash src/scripts/run_rl_7b.sh
```

## 4. Configurations

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
| `max_completion_length` | — | — | 4096 | 4096 |

**The 3B settings have not been validated.** They are scaled from the 7B recipe
(ZeRO-2 instead of ZeRO-3) and are marked as untested in each script. The
released checkpoint is 7B.

Two defaults follow the paper rather than the original launcher, which differed:

- SFT learning rate is `1.0e-6` (paper Table 9); the original script used `2e-6`.
- RL `max_completion_length` is `4096` (paper Table 10); the original script
  used `768`. Lower it if you hit OOM.

DeepSpeed configs are in `src/r1-v/local_scripts/`.

## 5. Overrides

Anything can be overridden from the environment:

| Variable | Meaning |
| --- | --- |
| `MODEL_PATH` | base model for SFT |
| `SFT_CKPT` | stage-1 checkpoint RL starts from |
| `RUN_NAME`, `OUTPUT_DIR` | run naming and output location |
| `LEARNING_RATE` | SFT learning rate |
| `MAX_COMPLETION_LENGTH` | RL generation budget |
| `SFT_DATASET`, `RL_DATASET` | comma-separated dataset mixtures |
| `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE` | GPU selection |
| `MASTER_ADDR`, `MASTER_PORT` | torchrun rendezvous |
| `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY` | logging |

The dataset mixtures default to the full set registered in
`src/r1-v/src/open_r1/DatasetLoader.py` — 10 SFT subsets and 6 RL subsets. To
drop `behavior-1k`, whose video media is 5,981 GB of the 6.58 TB release:

```bash
export RL_DATASET=primo-rl-agibot,primo-rl-robovqa,primo-rl-robotwin-clean,primo-rl-robotwin-randomized,primo-rl-sharerobot
```

## 6. Video-only baselines

Two extra launchers keep the pre-interleave recipe reproducible. They are the
input-format ablation: same data, same hyper-parameters, but the model sees a
bare clip instead of `I_init + V_seq + I_curr`.

```bash
bash src/scripts/run_sft_baseline_7b.sh    # src/open_r1/sft_video.py
export SFT_CKPT=/path/to/primo-sft-baseline-7b/checkpoint-XXXX
bash src/scripts/run_rl_baseline_7b.sh     # src/open_r1/grpo.py
```

`sft_video.py` and `grpo.py` are the video-only entry points. Paper Table 4 is
the corresponding comparison. These baseline configs have not been re-validated
since the code cleanup.

## 7. Invariants

Inherited from R1-V, and worth not changing casually:

- Keep `per_device_train_batch_size=1`.
- `--temporal` selects T-GRPO vs plain GRPO.
- `--len_control` toggles the length-control reward.
- `--num_generations` is GRPO's group size G — lowering it is cheaper but raises
  gradient variance.

Reward functions live in `reward_funcs_registry` in `grpo_interleave.py`
(`accuracy_reward`, `format_reward`); add new ones there so `--reward_funcs` can
select them.

Training reads `init_frame_path` / `current_frame_path` from the JSON records.
The published archives already ship those frames, so there is no preprocessing
step for the released data — see the top-level README's data layout section for
datasets you add yourself.

Prompts come from `src/primo_prompts.py` and frame helpers from
`src/primo_video_utils.py`. The answer extractors are coupled to the prompt
format, so treat the prompt module and the extractors as one unit — details in
[`src/README.md`](../README.md).
