# Copyright 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example usage:
accelerate launch \
    --config_file=deepspeed_zero2.yaml \
    train_video_llm.py \
    --dataset_name mfarre/simplevideoshorts \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir video-llm-output \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
"""

import os
import json
import random
import requests
import torch
import numpy as np
from typing import List, Dict, Any
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info

from datasets import Dataset, DatasetDict

import wandb
import time
import torch.distributed as dist
from DatasetLoader import dataset_loader

# Shared prompts live in the repo's own `src/`. src/scripts/common.sh puts it on
# PYTHONPATH; this makes a direct `python sft_video.py` work as well. This is the
# video-only baseline, so it uses the pre-interleave QUESTION_TEMPLATE variant.
# The imports below have to follow the sys.path insert, hence the E402 waivers.
import sys
from pathlib import Path

_PRIMO_SRC = Path(__file__).resolve().parents[3]
if str(_PRIMO_SRC) not in sys.path:
    sys.path.insert(0, str(_PRIMO_SRC))

from primo_prompts import QUESTION_TEMPLATE_SFT_VIDEO as QUESTION_TEMPLATE  # noqa: E402
from primo_prompts import TYPE_TEMPLATE  # noqa: E402

DATA_BASE_PATH_ENV = os.environ.get("VIDEO_DATA_ROOT")
_DATA_BASE_PATH = DATA_BASE_PATH_ENV


def get_data_base_path() -> str:
    """Return the active media root directory."""
    if _DATA_BASE_PATH is None:
        raise ValueError("Relative media path provided but data base path is not set.")
    return _DATA_BASE_PATH


def set_data_base_path(path: str) -> None:
    """Update the active media root directory."""
    global _DATA_BASE_PATH
    _DATA_BASE_PATH = path


def set_global_seed(seed: int | None) -> None:
    """Set global random seed for reproducibility across random/numpy/torch.
    Also configures cuDNN for deterministic behavior when possible.
    """
    if seed is None:
        return
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        import torch.backends.cudnn as cudnn  # type: ignore
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass
    # Encourage deterministic cublas when possible
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolve_media_path(resource_path: str) -> str:
    """Resolve media paths relative to the configured data root."""
    if not resource_path:
        raise ValueError("Resource path is empty.")

    if os.path.isabs(resource_path):
        return resource_path

    base_path = get_data_base_path()

    if resource_path.startswith("./"):
        relative_path = resource_path[2:]
    elif resource_path.startswith("/"):
        relative_path = resource_path[1:]
    else:
        relative_path = resource_path

    return os.path.join(base_path, relative_path)


def infer_data_base_path(dataset_path: str, dataset: DatasetDict) -> str:
    """Infer the media root directory when relative paths are provided."""
    candidate_roots = []

    if dataset_path and os.path.isabs(dataset_path):
        current_dir = os.path.dirname(dataset_path)
        while current_dir and current_dir not in candidate_roots:
            candidate_roots.append(current_dir)
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                break
            current_dir = parent_dir

    cwd = os.getcwd()
    if cwd not in candidate_roots:
        candidate_roots.append(cwd)

    sample_path = None
    if "train" in dataset and len(dataset["train"]) > 0:
        sample_path = dataset["train"][0].get("path")

    if sample_path:
        sanitized_sample = sample_path[2:] if sample_path.startswith("./") else sample_path.lstrip("/")
        for root in candidate_roots:
            if os.path.exists(os.path.join(root, sanitized_sample)):
                return root

    return candidate_roots[0] if candidate_roots else cwd

# print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

def get_current_device():
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def download_video(url: str, folder: str = '/tmp/videos/') -> str:
    """Download video if not already present locally."""
    filename = url.split("/")[-1]
    local_path = os.path.join(folder, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    except requests.RequestException as e:
        raise Exception(f"Failed to download video: {e}")

def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare dataset example for training."""

    

    system_message = "You are a helpful assistant"
    
    


    
    if example["problem_type"] == 'multiple choice':
        question = example['problem'] + "Options:\n"
        for op in example["options"]:
            question += op + "\n"
    else:
        question = example['problem']


    media_path = resolve_media_path(example['path'])

    # If the media path is a local file, skip the example when the file is missing.
    # Allow remote URLs (http/https) to pass through.
    if not (str(media_path).startswith("http://") or str(media_path).startswith("https://")):
        if not os.path.exists(media_path):
            print(f"[prepare_dataset] Warning: media not found, skipping example: {media_path}")
            return None

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": example['data_type'],
                    example['data_type']: media_path
                    # "max_pixels": 360*420,
                    # "fps": 1.0
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question, question_type=example['problem_type']) + TYPE_TEMPLATE[example['problem_type']]
                }
                # {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question, question_type=TYPE_TEMPLATE.get(example['problem_type'], ""))}
            ]
        },
        {
            "role": "assistant",
            # "content": [{"type": "text", "text": example['solution']}]
            # "content": [{"type": "text", "text": example['planning'] + "\n" + example['solution']}]
            # "content": [{"type": "text", "text": example['observation'] + "\n" + example['solution']}]
            # "content": [{"type": "text", "text": example['reasoning'] + "\n" + example['solution']}]
            "content": [{"type": "text", "text": example['process'] + "\n" + example['solution']}]

        }
    ]
    

    return {"messages": messages}

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    # video_inputs = []
    # image_inputs = []

    for i, example in enumerate(examples):
        try:

            texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
            image_inputs, video_inputs, video_kwargs = process_vision_info(example["messages"], return_video_kwargs=True)
            
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handle visual tokens based on processor type
    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else [
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ]

    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels
    return inputs

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    # Set seed early for reproducibility (affects shuffling and any random ops)
    set_global_seed(getattr(training_args, "seed", 42))
    print(f"[sft_video] Using global seed: {getattr(training_args, 'seed', 42)}")
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset(s) using DatasetLoader.
    # --dataset_name takes one or more registered dataset names separated by
    # commas; a single JSON path is also accepted for backward compatibility.
    dataset_entries: List[Dict[str, Any]] = []
    ds_arg = str(script_args.dataset_name).strip()
    if ds_arg.endswith('.json') or ds_arg.endswith('.jsonl') or os.path.sep in ds_arg:
        # Compatibility path: a JSON file given directly (single dataset).
        print(f"[sft_video] compatibility mode: loading from JSON file -> {ds_arg}")
        dataset = DatasetDict({"train": Dataset.from_json(ds_arg)})
        for ex in dataset['train']:
            dataset_entries.append(dict(ex))
    else:
        # Registry path: load several datasets through DatasetLoader.
        ds_names = [x.strip() for x in ds_arg.split(',') if x.strip()]
        print(f"[sft_video] loading datasets via DatasetLoader: {ds_names}")
        total = 0
        for name in ds_names:
            json_list, video_paths = dataset_loader(name)
            # Write the absolute video path back into entry['path'] so the rest
            # of the pipeline does not depend on environment variables.
            for entry, vpath in zip(json_list, video_paths):
                entry = dict(entry)
                entry['path'] = vpath
                dataset_entries.append(entry)
            print(f"[sft_video] dataset {name} -> loaded {len(json_list)} records")
            total += len(json_list)
        print(f"[sft_video] total samples loaded: {total}")

    # In DatasetLoader mode, infer the data root. Not needed when every path is
    # already absolute.
    if _DATA_BASE_PATH is None:
        try:
            # Infer from any relative-path sample; absolute-only sets skip this.
            if dataset_entries:
                tmp_ds = DatasetDict({"train": Dataset.from_list(dataset_entries)})
                _DATA_BASE_PATH = infer_data_base_path(ds_arg, tmp_ds)
        except Exception:
            pass

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # # Quantization configuration for 4-bit training
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # Model initialization
    # Check if DeepSpeed is being used to avoid device_map conflict
    is_deepspeed = training_args.deepspeed is not None
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        # DeepSpeed Zero-3 is not compatible with device_map
        device_map=None if is_deepspeed else get_kbit_device_map(),
        # quantization_config=bnb_config,
    )
    
    
    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )

    # Prepare dataset and filter out examples with missing local media
    prepared_dataset = []
    skipped_examples = 0
    source_iter = dataset_entries if 'dataset' not in locals() else dataset['train']
    for example in source_iter:
        prepared = prepare_dataset(example)
        if prepared is None:
            skipped_examples += 1
            continue
        prepared_dataset.append(prepared)

    print(f"[sft_video] Prepared {len(prepared_dataset)} examples, skipped {skipped_examples} missing media files.")

    # Shuffle training data before feeding into Trainer to avoid ordered ingestion
    if len(prepared_dataset) > 1:
        random.shuffle(prepared_dataset)
        print(f"[sft_video] Shuffled training dataset: {len(prepared_dataset)} examples")

    # # Initialize wandb if specified
    # is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    if training_args.report_to == "wandb": # and is_main:
        os.environ.setdefault("WANDB_MODE", "offline")           # default to offline
        os.environ.setdefault("WANDB_DIR", "./src/r1-v/wandb")
        os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
        run_name = getattr(training_args, "run_name", f"run-{time.strftime('%Y%m%d-%H%M%S')}")
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "video-llm-training-sft"),
            name=run_name,
            config={
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "model_name_or_path": model_config.model_name_or_path,
            },
            save_code=True
        )
        wandb.watch(model, log="all", log_freq=1)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
        # tokenizer=processor.tokenizer
    )

    # Train model
    trainer.train()

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
