from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import argparse
import json
import re
import os
import sys
import copy
import cv2
import torch
import time
import random
from functools import lru_cache
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


parser = argparse.ArgumentParser(description="Evaluation benchmark with Initial State + Video + Current State")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
parser.add_argument('--file_name', type=str, required=True, help="Dataset name, json file, or jsonl file")
parser.add_argument('--output_path', type=str, default=None, help="Path to save the result json")
parser.add_argument('--batch_size', type=int, default=64, help="Evaluation batch size")
parser.add_argument('--nframes', type=int, default=32, help="Number of frames sampled per video")
parser.add_argument('--sample_size', type=int, default=0, help="Random sample size; 0 means full dataset")
parser.add_argument('--seed', type=int, default=42, help="Random seed used when sample_size > 0")
parser.add_argument('--tensor_parallel_size', type=int, default=None, help="vLLM tensor parallel size; default uses all visible GPUs")
args = parser.parse_args()

MODEL_PATH = args.model_path
file_name = args.file_name
BSZ = args.batch_size


# ============ Lazy frame extraction helpers (matches SFT/GRPO) ============

@lru_cache(maxsize=256)
def extract_frames_on_demand(video_path: str):
    """
    Extract a video's first and last frame on demand, LRU-cached to avoid re-decoding.
    
    Args:
        video_path: path to the video file
        
    Returns:
        (init_img, current_img): a pair of PIL.Image objects, or (None, None) on failure
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open video: {video_path}")
            return None, None
        
        # Read the first frame (Initial State)
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            print(f"Warning: Cannot read first frame from: {video_path}")
            return None, None
        
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        init_img = Image.fromarray(first_frame_rgb)
        
        # Read the last frame (Current State)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
        ret, last_frame = cap.read()
        
        if not ret:
            # Fall back to the second-to-last frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 2))
            ret, last_frame = cap.read()
        
        cap.release()
        
        if not ret:
            print(f"Warning: Cannot read last frame from: {video_path}, using first frame")
            return init_img, init_img
        
        last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        current_img = Image.fromarray(last_frame_rgb)
        
        return init_img, current_img
        
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return None, None

def get_total_frames_cv2(video_path: str) -> int | None:
    """Return total frame count if local video readable, else None."""
    try:
        # Local paths only; http/https returns None
        if str(video_path).startswith("http://") or str(video_path).startswith("https://"):
            return None
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total if total > 0 else None
    except Exception:
        return None


# Upper bound on frames per video. NOTE: the published results were produced with
# this capped at 22, so --nframes 32 in the launchers effectively sampled 22 frames.
# Kept as the default for reproducibility; raise INTERLEAVE_MAX_NFRAMES to lift it.
MAX_NFRAMES = int(os.environ.get("INTERLEAVE_MAX_NFRAMES", 22))


def choose_nframes(requested: int, video_path: str) -> int:
    """
    Clamp the requested frame count:
    - use total_frames when the video is shorter than `requested`
    - keep the result within [2, MAX_NFRAMES]
    """
    req = int(requested)
    total = get_total_frames_cv2(video_path)
    if total is not None:
        req = min(req, total)
    req = max(2, min(MAX_NFRAMES, req))
    return req


def resolve_frame_placeholders(content_list: list) -> list:
    """
    Resolve frame placeholders in a content list, extracting frames on the fly.

    Args:
        content_list: content list containing placeholders

    Returns:
        The resolved content list with placeholders replaced by PIL.Image objects
    """
    resolved = []
    frame_cache = {}  # extract each video at most once per call
    
    for item in content_list:
        if not isinstance(item, dict):
            resolved.append(item)
            continue
            
        if item.get("type") == "image":
            image_value = item.get("image")
            
            if isinstance(image_value, dict):
                # This entry is a placeholder
                placeholder_type = image_value.get("_placeholder_type")
                video_path = image_value.get("video_path")
                
                if placeholder_type and video_path:
                    # Take from cache, or extract
                    if video_path not in frame_cache:
                        init_img, current_img = extract_frames_on_demand(video_path)
                        frame_cache[video_path] = (init_img, current_img)
                    else:
                        init_img, current_img = frame_cache[video_path]
                    
                    # Pick the frame matching the placeholder type
                    if placeholder_type == "initial_state":
                        if init_img:
                            resolved.append({"type": "image", "image": init_img})
                        else:
                            print(f"Warning: Failed to extract initial frame, skipping")
                    elif placeholder_type == "current_state":
                        if current_img:
                            resolved.append({"type": "image", "image": current_img})
                        else:
                            print(f"Warning: Failed to extract current frame, skipping")
                    else:
                        resolved.append(item)
                else:
                    resolved.append(item)
            else:
                # A regular image entry
                resolved.append(item)
        else:
            resolved.append(item)
            
    return resolved


def resolve_placeholders_in_messages(messages: list) -> list:
    """
    Resolve every frame placeholder in a message list.

    Args:
        messages: the message list

    Returns:
        The resolved message list
    """
    for message in messages:
        if "content" in message and isinstance(message["content"], list):
            message["content"] = resolve_frame_placeholders(message["content"])
    return messages


# ============ End of lazy frame extraction helpers ============


llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=args.tensor_parallel_size or torch.cuda.device_count(),
    max_model_len = 8192 * 2,
    gpu_memory_utilization=0.8,
    limit_mm_per_prompt={"image": 3, "video": 1},  # 2 images (initial + current) + 1 video
)


sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    max_tokens=4096,
    stop_token_ids=[],
)


processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer


# Media path resolution (copied from sft_video.py behavior)
DATA_BASE_PATH_ENV = os.environ.get("VIDEO_DATA_ROOT")
_DATA_BASE_PATH = DATA_BASE_PATH_ENV


def get_data_base_path() -> str:
    if _DATA_BASE_PATH is None:
        raise ValueError("Relative media path provided but data base path is not set.")
    return _DATA_BASE_PATH


def resolve_media_path(resource_path: str) -> str:
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


"""
Single-file / dataset evaluation:
- If --file_name is an existing .json/.jsonl path, read the samples straight from it.
- Otherwise treat it as a dataset name and load it through DatasetLoader.py
  (which returns json_list plus absolute video paths).
The output filename is derived from file_name (or the dataset name).
"""

# Resolve the input: a path or a dataset name
data = []
dataset_name_or_path = file_name

# Add the bundled loader directory to sys.path so DatasetLoader can be imported directly.
from pathlib import Path
_CANONICAL_LOADER_DIR = Path(__file__).resolve().parents[1] / "r1-v" / "src" / "open_r1"
sys.path.insert(0, str(_CANONICAL_LOADER_DIR))
try:
    from DatasetLoader import dataset_loader
except Exception:
    dataset_loader = None

is_file = os.path.exists(dataset_name_or_path)

if is_file and (dataset_name_or_path.endswith('.jsonl') or dataset_name_or_path.endswith('.json')):
    # Read the user-supplied JSON/JSONL directly
    if dataset_name_or_path.endswith('.jsonl'):
        with open(dataset_name_or_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        with open(dataset_name_or_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    base_tag = os.path.splitext(os.path.basename(dataset_name_or_path))[0]
elif (not is_file) and dataset_loader is not None:
    # Treat the argument as a dataset name and load it with DatasetLoader
    json_list, video_paths = dataset_loader(dataset_name_or_path)
    # Write absolute video paths back onto the samples to keep later resolution consistent
    for entry, vpath in zip(json_list, video_paths):
        entry = dict(entry)
        entry['path'] = vpath
        data.append(entry)
    base_tag = dataset_name_or_path
else:
    raise ValueError(f"Unable to read data: pass an existing .json/.jsonl file or a dataset name known to DatasetLoader. Got: {dataset_name_or_path}")

if args.sample_size > 0 and len(data) > args.sample_size:
    random.seed(args.seed)
    data = random.sample(data, args.sample_size)
    print(f"Sampled {args.sample_size} examples for evaluation with seed {args.seed}.")
else:
    print(f"Loaded {len(data)} examples for evaluation.")

if args.output_path:
    OUTPUT_PATH = args.output_path
else:
    OUTPUT_PATH = f"./src/r1-v/eval_outputs/interleave/{base_tag}.json"


# Create the output directory
output_dir = os.path.dirname(OUTPUT_PATH)
os.makedirs(output_dir, exist_ok=True)


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


# Build (sample, messages) pairs using interleave-style message structure
pairs = []
skipped = 0
for x in data:
    if x.get("problem_type") == 'multiple choice':
        question = x['problem'] + "\nOptions:\n"
        for op in x.get("options", []):
            question += op + "\n"
    else:
        question = x.get('problem', '')

    raw_path = x.get('path') or x.get('video') or x.get('video_path')
    if not raw_path:
        skipped += 1
        continue

    # Resolve the media path: prefer http/https or absolute paths, then VIDEO_DATA_ROOT,
    # then fall back to the JSON's directory and its parents
    if str(raw_path).startswith('http://') or str(raw_path).startswith('https://') or os.path.isabs(raw_path):
        media_path = raw_path
    else:
        try:
            media_path = resolve_media_path(raw_path)
        except Exception:
            # VIDEO_DATA_ROOT unset: fall back to the JSON file's directory as the root
            if is_file:
                json_dir = os.path.dirname(os.path.abspath(dataset_name_or_path))
                candidate_paths = [
                    os.path.join(json_dir, raw_path),
                    os.path.join(os.path.dirname(json_dir), raw_path.lstrip("./")),
                ]
                resolved = False
                for cp in candidate_paths:
                    if os.path.exists(cp):
                        media_path = cp
                        resolved = True
                        break
                if not resolved:
                    media_path = raw_path
            else:
                media_path = raw_path

    # skip missing local files
    if not (str(media_path).startswith("http://") or str(media_path).startswith("https://")) and not os.path.exists(media_path):
        print(f"[eval_bench_interleave] Warning: media not found, skipping sample: {media_path}")
        skipped += 1
        continue

    dtype = x.get('data_type', 'video')

    system_msg = {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}]
    }

    # Build the Initial State + Video + Current State input
    content = []
    
    # Only video samples get the three-part treatment
    if dtype == 'video' and media_path:
        # Initial State (placeholder)
        content.append({
            "type": "image",
            "image": {
                "_placeholder_type": "initial_state",
                "video_path": media_path
            }
        })
        
        # Video
        nf = choose_nframes(args.nframes, media_path)
        content.append({
            "type": "video",
            "video": media_path,
            "nframes": nf
        })
        
        # Current State (placeholder)
        content.append({
            "type": "image",
            "image": {
                "_placeholder_type": "current_state",
                "video_path": media_path
            }
        })
    else:
        nf = choose_nframes(args.nframes, media_path)
        # Non-video sample: keep the original data type
        content.append({
            "type": dtype,
            dtype: media_path,
            "nframes": nf
        })
    
    # Append the question text
    content.append({
        "type": "text",
        "text": QUESTION_TEMPLATE.format(
            Question=question,
            question_type=x.get('problem_type')
        ) + TYPE_TEMPLATE.get(x.get('problem_type'), "")
    })

    user_msg = {
        "role": "user",
        "content": content
    }

    # keep assistant empty for evaluation (no ground-truth prompt injection)
    messages = [system_msg, user_msg]
    
    pairs.append((x, messages))

print(f"[eval_bench_interleave] Loaded {len(pairs)} samples, skipped {skipped} missing media files.")


final_output = []
start_idx = 0
if os.path.exists(OUTPUT_PATH):
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
            final_output = existing.get("results", [])
            start_idx = len(final_output)
            print(f"Resuming from sample index {start_idx}")
    except Exception as e:
        print(f"Error reading existing output file: {e}")


def extract_think(output_str):
    pattern = r'<think>\s*(.*?)\s*</think>'
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def normalize_number(num_str):
    try:
        s = (num_str or "").strip()
        # remove common approximate symbols and spaces
        s = s.replace('≈', '').replace('~', '').replace('％', '%')
        # handle percentage
        if s.endswith('%'):
            s = s[:-1]
            s = s.replace(',', '')
            return float(s) / 100.0
        s = s.replace(',', '')
        return float(s)
    except Exception as e:
        return None
        
# Name of the formula this harness uses to score `regression`, recorded in the
# output file so a reader can tell which of the two formulas produced a number.
# See the metrics section of the top-level README.
REGRESSION_METRIC = "linear_relative_accuracy"


def linear_relative_accuracy(pred, target):
    """Score a regression answer as 1 - |pred-target|/|target|, clipped to [0,1].

    This is the formula behind every published PRIMO-R1 number. The baseline
    harnesses (eval_local / eval_api / eval_internvl) instead use
    `threshold_relative_accuracy`, which is stricter; the two are not comparable.
    """
    rel_diff = (abs(pred - target) + 1e-9) / (abs(target) + 1e-9)
    rel_diff = min(1.0, max(0.0, rel_diff))
    return 1 - rel_diff


# Normalized relative score (adapted from eval_bench_aaa but scaled 0-1 instead of 0-100)
def normalized_relative_score(pred, target, max_range=100.0):
    """Return 1 - |pred-target|/max_range clipped to [0,1]. If inputs invalid returns 0."""
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


def reward_fn(sample, model_output, question_type):
    try:
        # try to get predicted answer
        output_ans = extract_answer(model_output)
        if output_ans == '':
            output_ans = model_output
        gt_text = sample.get("solution", "")
        gt_ans = extract_answer(gt_text)
        if question_type == "multiple choice":
            return 1.0 if output_ans.strip().lower() == gt_ans.strip().lower() else 0.0
        elif question_type == "boolean": 
            return 1.0 if output_ans.strip().lower() == gt_ans.strip().lower() else 0.0
        elif question_type == "free-form":
            # Score free-form answers with ROUGE
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(gt_ans, output_ans)
            average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
            return average_fmeasure
        elif question_type == "numerical":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            else:
                max_range = float(os.environ.get("NUMERIC_MAX_RANGE", 100.0))
                return normalized_relative_score(out_number, gt_number, max_range=max_range)
        elif question_type == "regression":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return linear_relative_accuracy(out_number, gt_number)
        else:
            return 0.0
    except Exception as e:
        return 0.0


def recompute_cached_rewards(entries):
    cached_acc = []
    cached_mra = []
    updated = 0
    for entry in entries:
        q_type = (entry.get("problem_type", "") or "").strip().lower()
        cached_output = entry.get("output") or entry.get("prediction") or ""
        new_reward = reward_fn(entry, cached_output, q_type)
        if entry.get("reward") != new_reward:
            updated += 1
        entry["reward"] = new_reward
        entry["correct"] = True if new_reward == 1.0 else False
        if q_type in ("numerical", "regression"):
            cached_mra.append(new_reward)
        else:
            cached_acc.append(new_reward)
    if updated:
        print(f"[eval_bench_interleave] Recomputed cached rewards for {updated} samples.")
    return cached_acc, cached_mra

mean_acc = []
mean_mra = []
if final_output:
    cached_acc, cached_mra = recompute_cached_rewards(final_output)
    mean_acc.extend(cached_acc)
    mean_mra.extend(cached_mra)

# Process pairs in batches
for i in tqdm(range(start_idx, len(pairs), BSZ), desc="Processing batches"):
    batch_pairs = pairs[i:i + BSZ]
    batch_messages = [msgs for (_, msgs) in batch_pairs]

    # Deep-copy, then resolve placeholders
    batch_messages_resolved = []
    for msgs in batch_messages:
        msgs_copy = copy.deepcopy(msgs)
        msgs_resolved = resolve_placeholders_in_messages(msgs_copy)
        batch_messages_resolved.append(msgs_resolved)

    prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages_resolved]

    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages_resolved, return_video_kwargs=True)

        image_idx = 0
        video_idx = 0

        llm_inputs = []
        for idx, prompt in enumerate(prompts):
            llm_input = {"prompt": prompt, "multi_modal_data": {}}
            
            # Count how many images and videos this sample carries
            user_content = None
            for msg in batch_messages_resolved[idx]:
                if msg.get("role") == "user":
                    user_content = msg.get("content", [])
                    break
            
            if user_content:
                num_images = sum(1 for item in user_content if isinstance(item, dict) and item.get("type") == "image")
                num_videos = sum(1 for item in user_content if isinstance(item, dict) and item.get("type") == "video")
                
                if num_images > 0 and image_inputs:
                    llm_input["multi_modal_data"]["image"] = image_inputs[image_idx:image_idx + num_images]
                    image_idx += num_images
                
                if num_videos > 0 and video_inputs:
                    llm_input["multi_modal_data"]["video"] = video_inputs[video_idx:video_idx + num_videos]
                    video_idx += num_videos
            
            llm_inputs.append(llm_input)

        # Record generation time and output token count
        start_time = time.time()
        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
        end_time = time.time()
        
        batch_duration = end_time - start_time
        time_per_sample = batch_duration / len(llm_inputs) if llm_inputs else 0

        batch_output_text = []
        batch_output_tokens = []
        for out in outputs:
            try:
                text = out.outputs[0].text
                token_count = len(out.outputs[0].token_ids)
            except Exception:
                try:
                    text = str(out)
                    token_count = 0
                except Exception:
                    text = ""
                    token_count = 0
            batch_output_text.append(text)
            batch_output_tokens.append(token_count)

            
    except Exception as e:
        # report which original sample failed if possible
        try:
            failed_sample_id = batch_pairs[0][0].get('problem_id', 'unknown')
            print(f'Batch failed at sample starting with problem_id: {failed_sample_id}')
        except Exception:
            print('Batch failed but could not identify sample.')
        print('Exception:', e)
        batch_output_text = ['<answer>error</answer>'] * len(batch_pairs)
        batch_output_tokens = [0] * len(batch_pairs)
        time_per_sample = 0.0
        

    # zip with batch_pairs to keep alignment
    for offset, (sample, sample_messages) in enumerate(batch_pairs):
        model_output = batch_output_text[offset]
        token_count = batch_output_tokens[offset]
        
        think_chain = extract_think(model_output)
        final_ans = extract_answer(model_output)
        if final_ans == "":
            final_ans = model_output
            
        sample["prompt"] = sample_messages
        sample["output"] = model_output
        sample["prediction"] = final_ans
        
        # Record timing and token counts
        sample["generation_time"] = round(time_per_sample, 4)
        sample["output_tokens"] = token_count
        # ===============================
        
        q_type = (sample.get("problem_type", "") or "").strip().lower()
        sample["reward"] = reward_fn(sample, model_output, q_type)
        sample['correct'] = True if sample["reward"]==1.0 else False

        # Aggregation: numerical & regression go to mean_mra; everything else (including free-form) to mean_acc
        if q_type in ("numerical", "regression"):
            mean_mra.append(sample["reward"])
        else:
            mean_acc.append(sample["reward"])
        if think_chain:
            sample["think"] = think_chain
        final_output.append(sample)
    

    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
        print(f"Processed batch {(i - start_idx)//BSZ + 1}, saved {len(final_output)} samples.")
    except Exception as e:
        print(f"Error writing to output file: {e}")

final_acc={'mean_acc': 0.0, 'mean_mra': 0.0, 'regression_metric': REGRESSION_METRIC}
if mean_acc != []:
    final_acc['mean_acc'] = torch.tensor(mean_acc).mean().item()
if mean_mra != []:
    final_acc['mean_mra'] = torch.tensor(mean_mra).mean().item()

# Aggregate mean generation time and output tokens
valid_samples = len(final_output)
if valid_samples > 0:
    total_time = sum(item.get("generation_time", 0.0) for item in final_output)
    total_tokens = sum(item.get("output_tokens", 0) for item in final_output)
    
    final_acc['mean_generation_time'] = round(total_time / valid_samples, 4)
    final_acc['mean_output_tokens'] = round(total_tokens / valid_samples, 2)
else:
    final_acc['mean_generation_time'] = 0.0
    final_acc['mean_output_tokens'] = 0.0
# ==================================
 
try:
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": final_output, "final_acc": [final_acc]}, f, indent=2, ensure_ascii=False)
    print(f"Final accuracy saved to {OUTPUT_PATH}")
    print(f"Final Stats: {json.dumps(final_acc, indent=2)}")
except Exception as e:
    print(f"Error writing final accuracy to output file: {e}")
 
print(f"Results saved to {OUTPUT_PATH}")