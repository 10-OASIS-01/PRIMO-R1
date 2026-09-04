import os
import json
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse
import sys
from pathlib import Path

# The shared prompt and frame-extraction modules live in `src/`. The launchers put
# that on PYTHONPATH; this makes `python src/eval/eval_local.py ...` work too.
_PRIMO_SRC = Path(__file__).resolve().parents[1]
if str(_PRIMO_SRC) not in sys.path:
    sys.path.insert(0, str(_PRIMO_SRC))

from primo_prompts import SYSTEM_PROMPT, TYPE_TEMPLATE  # noqa: E402
from primo_prompts import QUESTION_TEMPLATE_BASELINE as QUESTION_TEMPLATE  # noqa: E402
from primo_video_utils import choose_nframes as _choose_nframes  # noqa: E402


parser = argparse.ArgumentParser(description="Evaluation benchmark")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
parser.add_argument('--file_name', type=str, required=True, help="Dataset name, json file, or jsonl file")
parser.add_argument('--output_path', type=str, default=None, help="Path to save the result json")
parser.add_argument('--batch_size', type=int, default=64, help="Evaluation batch size")
parser.add_argument('--nframes', type=int, default=32, help="Number of frames sampled per video")
parser.add_argument('--tensor_parallel_size', type=int, default=None, help="vLLM tensor parallel size; default uses all visible GPUs")
parser.add_argument('--gpu_memory_utilization', type=float, default=0.85, help="vLLM GPU memory utilization")
parser.add_argument('--max_model_len', type=int, default=8192 * 2, help="vLLM max model length")
parser.add_argument('--sample_size', type=int, default=0, help="Random sample size; 0 means full dataset")
parser.add_argument('--seed', type=int, default=42, help="Random seed used when sample_size > 0")
args = parser.parse_args()

MODEL_PATH = args.model_path
file_name = args.file_name
BSZ = args.batch_size



llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=args.tensor_parallel_size or torch.cuda.device_count(),
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    limit_mm_per_prompt={"image": 1, "video": 1},
)


sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    max_tokens=1024,
    stop_token_ids=[],
)


processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer


# Media path resolution (copied from sft_video.py behavior)
DATA_BASE_PATH_ENV = os.environ.get("VIDEO_DATA_ROOT")
_DATA_BASE_PATH = DATA_BASE_PATH_ENV


def choose_nframes(requested: int, video_path: str) -> int:
    """Matches the original baseline harness: no upper bound, unlike the
    interleaved harness which caps at `primo_video_utils.MAX_NFRAMES`."""
    return _choose_nframes(requested, video_path, max_nframes=None)


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


# Note: We intentionally avoid saving token-level details; only keep text output and extracted answer.


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
                data.append(json.loads(line))
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

if args.output_path:
    OUTPUT_PATH = args.output_path
else:
    OUTPUT_PATH = f"./src/r1-v/eval_outputs/local/{base_tag}.json"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
print(f"Results will be saved to: {OUTPUT_PATH}")

# Build (sample, messages) pairs using the sft-style message structure, taking video paths from the JSON
pairs = []
skipped = 0
for x in data:
    if x.get("problem_type") == 'multiple choice':
        question = x['problem'] + "Options:\n"
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
                # Walk up the parent directories
                trial_paths = [
                    os.path.join(json_dir, raw_path.lstrip('./').lstrip('/')),
                    os.path.join(os.path.dirname(json_dir), raw_path.lstrip('./').lstrip('/')),
                    os.path.join(os.path.dirname(os.path.dirname(json_dir)), raw_path.lstrip('./').lstrip('/')),
                ]
                media_path = None
                for tp in trial_paths:
                    if os.path.exists(tp):
                        media_path = tp
                        break
                if media_path is None:
                    media_path = os.path.join(json_dir, raw_path.lstrip('./').lstrip('/'))
            else:
                # Unreachable for dataset names: DatasetLoader already returns absolute paths
                media_path = raw_path

    # skip missing local files
    if not (str(media_path).startswith("http://") or str(media_path).startswith("https://")) and not os.path.exists(media_path):
        print(f"[eval_bench] Warning: media not found, skipping sample: {media_path}")
        skipped += 1
        continue

    dtype = x.get('data_type', 'video')

    system_msg = {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}]
    }

    nframes = choose_nframes(args.nframes, media_path)
    user_msg = {
        "role": "user",
        "content": [
            {
                "type": dtype,
                dtype: media_path,
                "nframes": nframes,
                "max_pixels": 256 * 28 * 28,
            },
            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[x.get('problem_type')]}
            # {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question, question_type=x['problem_type']) + TYPE_TEMPLATE[x['problem_type']]}
        ]
    }

    # keep assistant empty for evaluation (no ground-truth prompt injection)
    messages = [system_msg, user_msg]
    
    pairs.append((x, messages))

print(f"[eval_bench] Loaded {len(pairs)} samples, skipped {skipped} missing media files.")

if args.sample_size > 0 and len(pairs) > args.sample_size:
    import random
    random.seed(args.seed)
    pairs = random.sample(pairs, args.sample_size)
    print(f"[eval_bench] Sampled {args.sample_size} examples with seed {args.seed}.")

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
REGRESSION_METRIC = "threshold_relative_accuracy"


def threshold_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    """Fraction of thresholds in [start, end] where the relative error clears 1-t.

    This is the stricter, staircase-valued metric used by the baseline harnesses.
    eval_interleave.py instead uses `linear_relative_accuracy`; the two are not
    comparable, so never put their numbers in the same column.
    """

    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)
    
    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
    
    thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
    
    conditions = rel_error < (1 - thresholds)  
    mra = conditions.float().mean()  
    return mra.item()

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
            return max(0.0, min(1.0, average_fmeasure))
        elif question_type == "numerical":
            # Numerical: use normalized relative score (0-1). Treated as part of mean_mra aggregate.
            max_range_env = os.environ.get("NUMERIC_MAX_RANGE")
            try:
                max_range = float(max_range_env) if max_range_env else 100.0
            except Exception:
                max_range = 100.0
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return normalized_relative_score(out_number, gt_number, max_range=max_range)
        elif question_type == "regression":
            # Regression: use threshold-based MRA.
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return threshold_relative_accuracy(out_number, gt_number)
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
        print(f"[eval_bench] Recomputed cached rewards for {updated} samples.")
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

    prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]

    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)

        image_idx = 0
        video_idx = 0

        llm_inputs = []
        for idx, prompt in enumerate(prompts):
            # in our messages structure, user message is at index 1
            mm_type = batch_messages[idx][1]['content'][0]['type']
            sample_mm_data = {}
            sample_video_kw = {}
            if mm_type == 'image':
                sample_mm_data["image"] = image_inputs[image_idx]
                image_idx += 1
            elif mm_type == 'video':
                sample_mm_data["video"] = video_inputs[video_idx]
                for key, value in video_kwargs.items():
                    sample_video_kw[key] = value[video_idx]
                video_idx += 1

            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": sample_mm_data,
                "mm_processor_kwargs": sample_video_kw,
            })

        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

        # collect plain text outputs only
        batch_output_text = []
        for out in outputs:
            try:
                text = out.outputs[0].text
            except Exception:
                try:
                    text = str(out)
                except Exception:
                    text = ""
            batch_output_text.append(text)
    except Exception as e:
        # report which original sample failed if possible
        try:
            print('error:', pairs[i][0]['path'])
        except Exception:
            pass
        print('Exception:', e)
        batch_output_text = ['<answer>error</answer>'] * len(batch_pairs)
        

    # zip with batch_pairs to keep alignment
    for offset, (sample, sample_messages) in enumerate(batch_pairs):
        model_output = batch_output_text[offset]
        think_chain = extract_think(model_output)
        final_ans = extract_answer(model_output)
        if final_ans == "":
            final_ans = model_output
            
        sample["prompt"] = sample_messages
        sample["output"] = model_output
        sample["prediction"] = final_ans
        q_type = (sample.get("problem_type", "") or "").strip().lower()
        sample["reward"] = reward_fn(sample, model_output, q_type)
        sample['correct'] = True if sample["reward"]==1.0 else False

        # Aggregation: numerical & regression go to mean_mra; everything else (including free-form) to mean_acc
        if q_type in ("numerical", "regression"):
            mean_mra.append(sample["reward"])
        else:
            mean_acc.append(sample["reward"])
        if think_chain:
            sample["process"] = f"<think>{think_chain}</think>"
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
 
try:
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": final_output, "final_acc": [final_acc]}, f, indent=2, ensure_ascii=False)
    print(f"Final accuracy saved to {OUTPUT_PATH}")
except Exception as e:
    print(f"Error writing final accuracy to output file: {e}")
 
print(f"Results saved to {OUTPUT_PATH}")
