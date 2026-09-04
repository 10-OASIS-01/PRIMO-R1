# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import sys
from pathlib import Path

import av
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

import wandb
import os
from DatasetLoader import dataset_loader

# Shared prompts live in the repo's own `src/`. src/scripts/common.sh puts it on
# PYTHONPATH; this makes a direct `python grpo.py` work as well. The import below
# has to follow the sys.path insert, hence the E402 waiver.
_PRIMO_SRC = Path(__file__).resolve().parents[3]
if str(_PRIMO_SRC) not in sys.path:
    sys.path.insert(0, str(_PRIMO_SRC))

from primo_prompts import SYSTEM_PROMPT, QUESTION_TEMPLATE, TYPE_TEMPLATE  # noqa: E402

# Compatibility patch: torchvision's video reader probes for av.AVError, which
# newer versions of PyAV renamed to FFmpegError. Alias it back so the probe
# succeeds instead of raising AttributeError at import time.
try:
    av.AVError
except AttributeError:
    av.AVError = av.FFmpegError


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )



def accuracy_reward(completions, solution, **kwargs):
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            s = (num_str or "").strip()
            s = s.replace('≈', '').replace('~', '').replace('％', '%')
            if s.endswith('%'):
                s = s[:-1]
                s = s.replace(',', '')
                return float(s) / 100.0
            s = s.replace(',', '')
            return float(s)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def normalized_relative_score(pred, target, max_range=100.0):
        try:
            p = float(pred)
            t = float(target)
            if max_range <= 0:
                max_range = 100.0
            score = 1.0 - abs(p - t) / max_range
            if score < 0:
                score = 0.0
            if score > 1:
                score = 1.0
            return score
        except Exception:
            return 0.0

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    

    question_type = kwargs['problem_type'][0]
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(contents, solution):
    
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if question_type == "multiple choice":
                reward = 1.0 if output_ans.strip().lower() == gt_ans.strip().lower() else 0.0
            elif question_type == "boolean": return 1.0 if output_ans.strip().lower() == gt_ans.strip().lower() else 0.0
            elif question_type == "numerical":
                max_range_env = os.environ.get("NUMERIC_MAX_RANGE")
                try:
                    max_range = float(max_range_env) if max_range_env else 100.0
                except Exception:
                    max_range = 100.0
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                else:
                    reward = normalized_relative_score(out_number, gt_number, max_range=max_range)
            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                rel_diff = min(1.0, max(0.0, rel_diff))
                reward = 1 - rel_diff
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    # return [1.0 if match else 0.0 for match in matches]
    pattern = (
        r"<think>\s*"
        r"<planning>.*?</planning>\s*"
        r"<observation>.*?</observation>\s*"
        r"<reasoning>.*?</reasoning>\s*"
        r"</think>\s*"
        r"<answer>.*?</answer>"
    )  
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents] 
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset_entries = []
    ds_arg = str(script_args.dataset_name).strip()
    ds_names = [x.strip() for x in ds_arg.split(",") if x.strip()]
    use_dataset_loader = False
    for name in ds_names:
        try:
            json_list, video_paths = dataset_loader(name)
        except ValueError:
            dataset_entries = []
            use_dataset_loader = False
            break
        else:
            use_dataset_loader = True
            for entry, vpath in zip(json_list, video_paths):
                cur = dict(entry)
                cur["path"] = vpath
                dataset_entries.append(cur)

    if use_dataset_loader:
        dataset = DatasetDict({"train": Dataset.from_list(dataset_entries)})
    elif ds_arg.endswith(".json") or ds_arg.endswith(".jsonl") or os.path.sep in ds_arg:
        dataset = DatasetDict({"train": Dataset.from_json(ds_arg)})
    else:
        dataset = load_dataset(ds_arg, name=script_args.dataset_config)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
        
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
    }
        
    def make_conversation_image_and_video(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        
        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                            # example['data_type']: os.getcwd() + "/Video-R1-data" + example['path'][1:]    
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question, question_type=example['problem_type']) + TYPE_TEMPLATE[example['problem_type']]
                        }
                        ]
                }]
            }
        return msg

    
    dataset = dataset.map(make_conversation_image_and_video)

    eval_dataset = None
    if training_args.eval_strategy != "no" and script_args.dataset_test_split in dataset:
        eval_dataset = dataset[script_args.dataset_test_split]

    if training_args.use_vllm:
        raise NotImplementedError(
            "vLLM-backed GRPO is not supported in this repository. "
            "Run with --use_vllm false (the default)."
        )
    trainer_cls = Qwen2VLGRPOTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
