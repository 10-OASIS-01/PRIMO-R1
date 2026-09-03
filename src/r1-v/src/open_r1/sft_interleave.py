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
SFT training with Initial State + Video + Current State input modality.
Matches the GRPO training input format for consistency.

Example usage:
accelerate launch \
    --config_file=deepspeed_zero2.yaml \
    sft_interleave.py \
    --dataset_name your_dataset.json \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir sft-interleave-output \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
"""

import os
import json
import random
import torch
import numpy as np
import time
from PIL import Image
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

from typing import List, Dict, Any

# DatasetLoader support
try:
    from DatasetLoader import dataset_loader
except ImportError:
    dataset_loader = None

# ==================== Data path helpers ====================

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
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass
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

# ============ Pre-extracted frame mode (no on-the-fly extraction, avoids dataloader deadlocks) ============


def get_current_device():
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The Assistant is an expert AI specializing in embodied procedure and event reasoning based on visual input (video or images). "
    "The assistant must strictly follow a specific thought process and output format. "
    "The reasoning process is enclosed within <think> </think> tags, and the final answer is within <answer> </answer> tags. "
    "The <think> block must contain three ordered subsections: <planning>, <observation>, and <reasoning>. "
    "The <answer> block must contain only the final output required by the question type and no other commentary."
)

QUESTION_TEMPLATE = (
    "QUESTION:\n{Question}\n\n"
    "QUESTION TYPE:\n{question_type}\n\n"
    "Analyze the provided visual data and reason about the ongoing task.\n\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Provide your detailed reasoning between the <think> and </think> tags, following the subsections <planning>, <observation>, and <reasoning>. "
    "Then give your final answer between the <answer> and </answer> tags.\n\n"
    "Below is the required template:\n\n"
    "<think>\n"
    "<planning>\n"
    "Identify the high-level goal of the agent, what is the initial state? What does successful completion look like?\n"
    "Break down the high-level goal into a logical sequence of canonical steps. This serves as your mental plan for interpreting the task.\n"
    "Use this plan to interpret actions, map observed behaviors to steps, assess progress, detect anomalies, and predict what happens next.\n"
    "</planning>\n"
    "<observation>\n"
    "View the video as a temporal sequence of actions contributing to the procedure.\n"
    "Objectively describe what is occurring in the current moment, noting evidence of progress or state changes.\n"
    "Identify fine-grained actions and explain how they move the task forward.\n"
    "List relevant objects, tools, and environmental context, emphasizing functional states and transformations.\n"
    "Note cues—repetition, transitions, or completion indicators—that situate the action in the procedural script.\n"
    "</observation>\n"
    "<reasoning>\n"
    "Think through the question as a human would, Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'hmm', 'oh, I see', 'let's break it down', etc.\n"
    "Connect observations to the procedural plan to determine which step is being executed, progress, correctness, or anomalies.\n"
    "Reflect on assumptions, verify interpretations, and, if appropriate, predict the agent's next likely action.\n"
    "Synthesize understanding of what the agent is doing, how it fits into the broader task, and whether the process seems successful.\n"
    "You are encouraged to include self-reflection or verification in your reasoning process.\n"
    "</reasoning>\n"
    "</think>\n"
    "<answer>\n"
    "[Final answer here — strictly follow the `{question_type}` output format and include no extra commentary.]\n"
    "</answer>"
)

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "boolean": " Please provide only 'Yes' or 'No' as your answer within the <answer> </answer> tags."
}


def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Prepare dataset example for training with Initial State + Video + Current State modality.
    Matches GRPO input format.
    """
    
    if example["problem_type"] == 'multiple choice':
        question = example['problem'] + "\nOptions:\n"
        for op in example["options"]:
            question += op + "\n"
    else:
        question = example['problem']

    # Resolve the video/image path
    media_path = example.get('path') or example.get('video') or example.get('video_path')
    
    # Try to resolve a relative path (when DATA_BASE_PATH is configured)
    if media_path and _DATA_BASE_PATH is not None:
        try:
            if not (str(media_path).startswith('http://') or str(media_path).startswith('https://') or os.path.isabs(media_path)):
                media_path = resolve_media_path(media_path)
        except Exception:
            pass  # keep the original path
    
    # Skip the sample if it is a local file that does not exist
    if media_path and not (str(media_path).startswith("http://") or str(media_path).startswith("https://")):
        if not os.path.exists(media_path):
            print(f"[sft_interleave] Warning: media not found, skipping example: {media_path}")
            return None
    
    data_type = example.get('data_type', 'video')
    
    # Build the user message content
    content = []
    
    # Only video samples get the Initial State + Video + Current State treatment
    if data_type == 'video' and media_path:
        # Use the pre-extracted first/last frames (all frames are prepared offline)
        if 'init_frame_path' not in example or 'current_frame_path' not in example:
            print(f"[sft_interleave] Warning: preprocessed frame paths missing, skipping example: {media_path}")
            return None
        
        init_path = example['init_frame_path']
        current_path = example['current_frame_path']

        # Same handling as media_path: relative frame paths from the JSON are supported.
        if _DATA_BASE_PATH is not None:
            try:
                if not (str(init_path).startswith('http://') or str(init_path).startswith('https://') or os.path.isabs(init_path)):
                    init_path = resolve_media_path(init_path)
                if not (str(current_path).startswith('http://') or str(current_path).startswith('https://') or os.path.isabs(current_path)):
                    current_path = resolve_media_path(current_path)
            except Exception:
                pass
        
        # Verify the pre-extracted frame files exist
        if not os.path.exists(init_path):
            print(f"[sft_interleave] Warning: init frame not found, skipping example: {init_path}")
            return None
        if not os.path.exists(current_path):
            print(f"[sft_interleave] Warning: current frame not found, skipping example: {current_path}")
            return None
        
        # Use the pre-extracted JPG paths directly
        content.append({
            "type": "image",
            "image": init_path
        })
        
        content.append({
            "type": "video",
            "video": media_path
        })
        
        content.append({
            "type": "image",
            "image": current_path
        })
    else:
        # Non-video sample: keep the original data type
        content.append({
            "type": data_type,
            data_type: media_path
        })
    
    # Append the question text
    content.append({
        "type": "text",
        "text": QUESTION_TEMPLATE.format(
            Question=question, 
            question_type=example["problem_type"]
        ) + TYPE_TEMPLATE.get(example['problem_type'], "")
    })

    # ================= =================
    if 'process' not in example or 'solution' not in example:
        print("\n" + "="*60)
        print("🚨 Found a malformed sample missing 'process' or 'solution'!")
        print(f"👉 Keys present: {list(example.keys())}")
        print(f"👉 Media path (path/video): {example.get('path', example.get('video', 'N/A'))}")
        print(f"👉 Problem: {example.get('problem', 'N/A')}")
        
        # Pretty-print the full dict so the log stays readable
        import json
        try:
            print(f"👉 Full sample:\n{json.dumps(example, ensure_ascii=False, indent=2)}")
        except Exception:
            print(f"👉 Full sample:\n{example}")
            
        print("="*60 + "\n")
        
        # Raise explicitly so the run stops immediately and the log above is easy to find
        raise KeyError("Data format error: Missing 'process' or 'solution' key.")
    # ====================================================
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": content
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example['process'] + "\n" + example['solution']}]
        }
    ]

    return {"messages": messages}


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate batch of examples for training.
    Uses the same approach as sft_egoplan.py to avoid decord deadlock.
    """
    
    texts = []
    image_inputs = []  # list of lists, not flattened
    video_inputs = []  # list of lists, not flattened

    for i, example in enumerate(examples):
        try:
            # Pre-extracted mode: use messages as-is, no placeholder resolution
            messages = example["messages"]
            
            texts.append(processor.apply_chat_template(messages, tokenize=False))
            
            # Important: follow the egoplan handling and do not pass return_video_kwargs
            imgs, vids = process_vision_info(messages)
            image_inputs.append(imgs)  # append, not extend
            video_inputs.append(vids)
            
        except Exception as e:
            print(f"❌ FAILED at example {i}: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to process example {i}: {e}")

    # Run every input through the processor
    inputs = processor(
        text=texts,
        images=image_inputs,  # pass the nested list directly
        videos=video_inputs,  # pass the nested list directly
        return_tensors="pt",
        padding=True
    )
    
    # Build labels
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handle vision tokens
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
    print(f"[sft_interleave] Using global seed: {getattr(training_args, 'seed', 42)}")
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset(s) using DatasetLoader - accepts a comma-separated list
    dataset_entries: List[Dict[str, Any]] = []
    ds_arg = str(script_args.dataset_name).strip()
    
    if ds_arg.endswith('.json') or ds_arg.endswith('.jsonl') or os.path.sep in ds_arg:
        # Backward compatible: a raw JSON file path (single dataset)
        print(f"[sft_interleave] Compatibility mode: loading from JSON file -> {ds_arg}")
        if ds_arg.endswith('.jsonl'):
            with open(ds_arg, 'r', encoding='utf-8') as f:
                for line in f:
                    dataset_entries.append(json.loads(line.strip()))
        else:
            with open(ds_arg, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    dataset_entries.extend(data)
                elif isinstance(data, dict) and 'data' in data:
                    dataset_entries.extend(data['data'])
                else:
                    dataset_entries.append(data)
    else:
        # Preferred mode: load one or more datasets through DatasetLoader
        if dataset_loader is None:
            raise ImportError("DatasetLoader not available. Please check sys.path or use JSON file input.")
        
        ds_names = [x.strip() for x in ds_arg.split(',') if x.strip()]
        print(f"[sft_interleave] Loading datasets via DatasetLoader: {ds_names}")
        total = 0
        for name in ds_names:
            json_list, video_paths = dataset_loader(name)
            # Write the absolute video path back into entry['path'] so no env var is needed later
            for entry, vpath in zip(json_list, video_paths):
                entry = dict(entry)
                entry['path'] = vpath
                dataset_entries.append(entry)
            print(f"[sft_interleave] Dataset {name} -> loaded {len(json_list)} samples")
            total += len(json_list)
        print(f"[sft_interleave] Total samples loaded: {total}")
    
    # In DatasetLoader mode, infer or set the data root; not needed when every path is absolute
    if _DATA_BASE_PATH is None and dataset_entries:
        try:
            tmp_ds = DatasetDict({"train": Dataset.from_list(dataset_entries)})
            set_data_base_path(infer_data_base_path(ds_arg, tmp_ds))
            print(f"[sft_interleave] Inferred data root: {_DATA_BASE_PATH}")
        except Exception:
            pass

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # Model initialization
    # Check if DeepSpeed is being used to avoid device_map conflict
    is_deepspeed = training_args.deepspeed is not None
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        # DeepSpeed Zero-3 is not compatible with device_map
        device_map=None if is_deepspeed else get_kbit_device_map(),
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

    # Prepare dataset with Initial State + Video + Current State modality
    print("[sft_interleave] Preparing dataset with Initial State + Video + Current State modality...")
    prepared_dataset = []
    skipped_examples = 0
    
    for example in dataset_entries:
        prepared = prepare_dataset(example)
        if prepared is None:
            skipped_examples += 1
            continue
        prepared_dataset.append(prepared)
    
    print(f"[sft_interleave] Prepared {len(prepared_dataset)} examples, skipped {skipped_examples} missing media files.")
    
    # Guard against an empty dataset
    if len(prepared_dataset) == 0:
        raise ValueError(f"❌ Dataset is empty! All {len(dataset_entries)} samples were skipped. Check the data paths and the frame pre-extraction state.")
    
    # Shuffle training data before feeding into Trainer to avoid ordered ingestion
    if len(prepared_dataset) > 1:
        random.shuffle(prepared_dataset)
        print(f"[sft_interleave] Shuffled training dataset: {len(prepared_dataset)} examples")

    # Initialize wandb if specified. `report_to` is a list in HF TrainingArguments.
    # WANDB_MODE is intentionally left to the environment so runs stay online by default.
    if "wandb" in (training_args.report_to or []):
        os.environ.setdefault("WANDB_DIR", "./wandb")
        os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
        run_name = getattr(training_args, "run_name", f"run-{time.strftime('%Y%m%d-%H%M%S')}")
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "sft-interleave-training"),
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
    
    if training_args.report_to == "wandb":
        wandb.finish()
