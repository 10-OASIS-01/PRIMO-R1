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

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

import wandb
import os
from DatasetLoader import dataset_loader

#运行报错补丁
import av

# --- 添加这段补丁代码 ---
try:
    # 检查是否存在 AVError
    av.AVError
except AttributeError:
    # 如果不存在（新版 av），则将其指向 FFmpegError，骗过 torchvision
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
            # elif question_type == "numerical":
            #     gt_val = normalize_number(gt_ans)
            #     out_val = normalize_number(output_ans)
                
            #     if gt_val is None or out_val is None:
            #         reward = 0.0
            #     else:
            #         # --- Core Parameter Configuration ---
            #         # delta: Tolerance margin, set to 0.02 to resolve up to 20 subtasks (0.05 step size)
            #         delta = 0.02          
            #         # lambda_succ: Success bonus to provide a strong convergence signal in GRPO groups
            #         lambda_succ = 3.0     
            #         # lambda_err: Distance penalty to provide continuous gradient guidance during exploration
            #         lambda_err = 1.0      
                    
            #         # Calculate absolute error
            #         diff = abs(out_val - gt_val)
                    
            #         # --- Accuracy Reward Implementation ---
            #         # Success Bonus: Grant a significant positive reward if within the acceptable margin
            #         success_bonus = lambda_succ if diff < delta else 0.0
                    
            #         # Distance Penalty: Apply a linear penalty based on the distance from the target
            #         distance_penalty = lambda_err * diff
                    
            #         reward = max(0.0, min(1.0, success_bonus - distance_penalty))
                    
            #         # Note: Since GRPO performs group-level z-score normalization, the absolute 
            #         # scale of these values will be automatically adjusted for the policy gradient.
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


# 与 SFT 一致：直接使用预处理好的首尾帧路径，不在训练时实时抽帧。


DATA_BASE_PATH_ENV = os.environ.get("VIDEO_DATA_ROOT")


def _is_remote_path(path: str) -> bool:
    return isinstance(path, str) and (path.startswith("http://") or path.startswith("https://"))


def _resolve_media_path(path: Optional[str]) -> Optional[str]:
    """Resolve relative media path to absolute path using VIDEO_DATA_ROOT when available."""
    if not path or not isinstance(path, str):
        return path

    if _is_remote_path(path) or os.path.isabs(path):
        return path

    rel = path[2:] if path.startswith("./") else path.lstrip("/")
    if DATA_BASE_PATH_ENV:
        return os.path.join(DATA_BASE_PATH_ENV, rel)
    return path


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The Assistant is an expert AI specializing in embodied procedure and event reasoning based on visual input (video or images). "
    "The assistant must strictly follow a specific thought process and output format. "
    "The reasoning process is enclosed within <think> </think> tags, and the final answer is within <answer> </answer> tags. "
    "The <think> block must contain three ordered subsections: <planning>, <observation>, and <reasoning>. "
    "The <answer> block must contain only the final output required by the question type and no other commentary."
)


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

    def make_conversation_image(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']
        
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question, question_type=example["problem_type"]) + TYPE_TEMPLATE.get(example['problem_type'], "")},
                    ],
                },
            ],
        }
    
        
    def make_conversation_video(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']
        
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question, question_type=example["problem_type"]) + TYPE_TEMPLATE.get(example['problem_type'], "")},
                    ],
                },
            ],
    }
        
        
    def make_conversation_image_and_video(example):
        """
        构建 Initial State + Video + Current State 的多模态输入
        使用预提取帧路径：直接加载 init_frame_path / current_frame_path
        
        输入顺序：
        1. Initial State（预提取首帧）
        2. Video (完整视频)
        3. Current State（预提取尾帧）
        4. Text (问题文本)
        """
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']
        
        # 获取视频/图片路径
        media_path = example.get('path') or example.get('video') or example.get('video_path')
        media_path = _resolve_media_path(media_path)
        data_type = example.get('data_type', 'video')
        init_frame_path = _resolve_media_path(example.get('init_frame_path'))
        current_frame_path = _resolve_media_path(example.get('current_frame_path'))
        
        # 构建 content 列表
        content = []
        skip_sample = False
        
        # 只对视频类型数据进行 Initial State + Video + Current State 处理
        if data_type == 'video' and media_path:
            if not init_frame_path or not current_frame_path:
                print(f"[grpo_interleave] Warning: missing frame paths, skipping sample: {media_path}")
                skip_sample = True
            elif (
                (not _is_remote_path(media_path) and not os.path.exists(media_path))
                or (not _is_remote_path(init_frame_path) and not os.path.exists(init_frame_path))
                or (not _is_remote_path(current_frame_path) and not os.path.exists(current_frame_path))
            ):
                print(f"[grpo_interleave] Warning: media/frame not found, skipping sample: {media_path}")
                skip_sample = True

            # 1. Initial State
            content.append({
                "type": "image", 
                "image": init_frame_path
            })
            
            # 2. Video
            content.append({
                "type": "video",
                "video": media_path,
            })
            
            # 3. Current State
            content.append({
                "type": "image", 
                "image": current_frame_path
            })
            
            # 4. 添加问题文本，加入模态特定的提示
            prompt_prefix = "Given the initial state in the first image, the progress shown in the video, and the current state in the final image, "
            full_question = prompt_prefix + QUESTION_TEMPLATE.format(Question=question, question_type=example["problem_type"]) + TYPE_TEMPLATE.get(example['problem_type'], "")
            content.append({
                "type": "text",
                "text": full_question
            })
        else:
            # 对于图片类型或路径无效的情况，保持原有逻辑
            content.append({
                "type": data_type,
            })
            content.append({
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=question, question_type=example["problem_type"]) + TYPE_TEMPLATE.get(example['problem_type'], "")
            })
        
        msg = {
            "prompt": [{
                "role": "user",
                "content": content,
            }],
            "_skip_sample": skip_sample,
        }
        
        return msg


    
    dataset = dataset.map(make_conversation_image_and_video)
    if "_skip_sample" in dataset[script_args.dataset_train_split].column_names:
        before_train = len(dataset[script_args.dataset_train_split])
        dataset[script_args.dataset_train_split] = dataset[script_args.dataset_train_split].filter(
            lambda x: not x.get("_skip_sample", False)
        )
        after_train = len(dataset[script_args.dataset_train_split])
        print(f"[grpo_interleave] Filtered skipped train samples: {before_train - after_train}")

        if script_args.dataset_test_split in dataset:
            before_test = len(dataset[script_args.dataset_test_split])
            dataset[script_args.dataset_test_split] = dataset[script_args.dataset_test_split].filter(
                lambda x: not x.get("_skip_sample", False)
            )
            after_test = len(dataset[script_args.dataset_test_split])
            print(f"[grpo_interleave] Filtered skipped eval samples: {before_test - after_test}")

    eval_dataset = None
    if training_args.eval_strategy != "no" and script_args.dataset_test_split in dataset:
        eval_dataset = dataset[script_args.dataset_test_split]

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
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
