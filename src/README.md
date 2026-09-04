# Shared modules

Two modules in this directory are the single source of truth for everything the
entry points used to define locally. Import from them rather than pasting a copy
— that is how the prompts drifted in the first place (`SYSTEM_PROMPT` appeared
in five files, `TYPE_TEMPLATE` in two silently different variants).

Both are importable because `src/scripts/common.sh` puts this directory on
`PYTHONPATH`; each entry point also inserts it itself, so running a script
directly with `python` works too.

## `primo_prompts.py`

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

The variants are not redundancy. `TYPE_TEMPLATE_BASELINE` is `TYPE_TEMPLATE`
minus the `boolean` key, and the baseline numbers in the paper were produced
without it, so adding the key would change which records get a type hint. Each
is preserved verbatim.

`QUESTION_TEMPLATE_SFT_VIDEO` relies on Markdown hard breaks — 32 lines ending
in two spaces. A whitespace-stripping editor will silently change the prompt
without changing anything visible.

## `primo_video_utils.py`

The frame helpers behind the interleaved format:

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

`MAX_NFRAMES` (env `INTERLEAVE_MAX_NFRAMES`) defaults to 22, which is the
effective cap the published numbers were produced with even though the launchers
pass `--nframes 32`. The video-only baseline harness never had an upper bound and
passes `max_nframes=None` to keep its own numbers reproducible.

Two placeholder protocols are both supported, because the trainer and the eval
harness deferred frame extraction differently: dicts for eval,
`__INIT_FRAME__:` / `__CURRENT_FRAME__:` string prefixes for the trainer.

## Prompt coupling

`QUESTION_TEMPLATE` + `TYPE_TEMPLATE` ask the model to ground observations in a
procedural plan and emit a strictly formatted final answer. The answer
extractors `extract_think` and `extract_answer` are coupled to that format —
editing a prompt without updating them will silently collapse scores rather than
error out. Treat `primo_prompts.py` and the extractors as one unit.

## Other files here

| Path | What it is |
| --- | --- |
| `scripts/` | training launchers — see [`scripts/README.md`](scripts/README.md) |
| `eval/` | evaluation harnesses — see [`eval/README.md`](eval/README.md) |
| `r1-v/src/open_r1/` | training entry points and `DatasetLoader.py`, the dataset registry |
| `preprocess_video_frames.py` | anchor-frame extraction for datasets you add yourself |
| `generate_cot_vllm.py` | CoT teacher used to build the stage-1 annotations |
| `inference_example.py` | minimal single-video example |
| `qwen-vl-utils/` | vendored, pinned; frame sampling lives here |
