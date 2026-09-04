import os
import json
import re
import argparse
import sys
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import cv2
import math

# Imports needed for manual data loading
import decord 
from transformers import AutoProcessor, AutoTokenizer, Qwen2TokenizerFast
from vllm import LLM, SamplingParams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pathlib import Path

# The shared prompt module lives in `src/`. The launchers put that on
# PYTHONPATH; this makes `python src/eval/eval_internvl.py ...` work too.
_PRIMO_SRC = Path(__file__).resolve().parents[1]
if str(_PRIMO_SRC) not in sys.path:
    sys.path.insert(0, str(_PRIMO_SRC))

# InternVL supplies its own instruction framing through its chat template, so
# this harness passes the bare question rather than the PRIMO prompt.
from primo_prompts import TYPE_TEMPLATE  # noqa: E402
from primo_prompts import QUESTION_TEMPLATE_QUESTION_ONLY as QUESTION_TEMPLATE  # noqa: E402


parser = argparse.ArgumentParser(description="Evaluation benchmark for InternVL")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
parser.add_argument('--file_name', type=str, required=True, help="Dataset name, json file, or jsonl file")
parser.add_argument('--output_path', type=str, default=None, help="Path to save the result json")
parser.add_argument('--batch_size', type=int, default=64, help="Evaluation batch size")
parser.add_argument('--nframes', type=int, default=32, help="Number of frames sampled per video")
parser.add_argument('--tensor_parallel_size', type=int, default=None, help="vLLM tensor parallel size; default uses all visible GPUs")
parser.add_argument('--gpu_memory_utilization', type=float, default=0.85, help="vLLM GPU memory utilization")
parser.add_argument('--max_model_len', type=int, default=40960, help="vLLM max model length")
args = parser.parse_args()

MODEL_PATH = args.model_path
file_name = args.file_name
BSZ = args.batch_size
NUM_FRAMES = args.nframes

# ==============================================================================
# 0. Tokenizer patch & data loading helpers
# ==============================================================================
def patch_tokenizer_classes():
    """Patch in the tokenizer attributes InternVL's remote code expects."""
    print("Applying Patch to Qwen2TokenizerFast...")
    image_pad = "<|image_pad|>"
    vision_start = "<|vision_start|>"
    vision_end = "<|vision_end|>"
    
    token_map = {
        "start_image_token": vision_start,
        "end_image_token":   vision_end,
        "context_image_token": image_pad,
        "video_token": image_pad,
        "start_video_token": vision_start,
        "end_video_token": vision_end,
    }

    for attr_name, token_str in token_map.items():
        if not hasattr(Qwen2TokenizerFast, attr_name):
            setattr(Qwen2TokenizerFast, attr_name, token_str)

    for attr_name, token_str in token_map.items():
        id_attr_name = attr_name + "_id"
        if not hasattr(Qwen2TokenizerFast, id_attr_name):
            def make_getter(t_str):
                return property(lambda self: self.convert_tokens_to_ids(t_str))
            setattr(Qwen2TokenizerFast, id_attr_name, make_getter(token_str))

# Apply the patch
patch_tokenizer_classes()

def load_video_frames(video_path, num_frames=32):
    """Read a video with decord and return a uint8 numpy array."""
    if not os.path.exists(video_path):
        if os.path.exists(os.path.join("./", video_path)):
             video_path = os.path.join("./", video_path)
        else:
            print(f"Warning: Video not found: {video_path}")
            return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)
    
    try:
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        
        # ==============================================================
        # Enforce the pixel budget here, matching the paper: 256 * 28 * 28 = 200,704
        # ==============================================================
        max_pixels = 256 * 28 * 28 
        resized_frames = []
        
        for frame in frames:
            h, w = frame.shape[:2]
            # Downscale proportionally when the frame exceeds the budget
            if h * w > max_pixels:
                beta = math.sqrt((h * w) / max_pixels)
                new_h = int(h / beta)
                new_w = int(w / beta)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            resized_frames.append(frame)
            
        # Repack into a numpy array
        frames = np.array(resized_frames, dtype=np.uint8)
        # ==============================================================

        return frames
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)

def load_image(image_path):
    """Read an image with PIL."""
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return Image.new('RGB', (224, 224), (0, 0, 0))

# ==============================================================================
# 1. Model initialisation
# ==============================================================================
print(f"Initializing model from {MODEL_PATH}...")

# trust_remote_code=True is required for InternVL
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=args.tensor_parallel_size or torch.cuda.device_count(),
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    limit_mm_per_prompt={"image": 1, "video": 1},
    trust_remote_code=True, 
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    max_tokens=1024,
    stop_token_ids=[],
)

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer

# ==============================================================================
# 2. Data handling and path resolution
# ==============================================================================
DATA_BASE_PATH_ENV = os.environ.get("VIDEO_DATA_ROOT")
_DATA_BASE_PATH = DATA_BASE_PATH_ENV

def get_data_base_path() -> str:
    return _DATA_BASE_PATH if _DATA_BASE_PATH is not None else "./"

def resolve_media_path(resource_path: str) -> str:
    if not resource_path: return ""
    if str(resource_path).startswith(('http', 'https')) or os.path.isabs(resource_path):
        return resource_path
    
    base_path = get_data_base_path()
    # Simple path joining
    clean_path = resource_path.lstrip("./").lstrip("/")
    return os.path.join(base_path, clean_path)

# Load the data
data = []
dataset_name_or_path = file_name
_CANONICAL_LOADER_DIR = Path(__file__).resolve().parents[1] / "r1-v" / "src" / "open_r1"
sys.path.insert(0, str(_CANONICAL_LOADER_DIR))
try:
    from DatasetLoader import dataset_loader
except Exception:
    dataset_loader = None

is_file = os.path.exists(dataset_name_or_path)

if is_file and dataset_name_or_path.endswith(('.jsonl', '.json')):
    if dataset_name_or_path.endswith('.jsonl'):
        with open(dataset_name_or_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
    else:
        with open(dataset_name_or_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    base_tag = os.path.splitext(os.path.basename(dataset_name_or_path))[0]
elif (not is_file) and dataset_loader is not None:
    json_list, video_paths = dataset_loader(dataset_name_or_path)
    for entry, vpath in zip(json_list, video_paths):
        entry = dict(entry)
        entry['path'] = vpath
        data.append(entry)
    base_tag = dataset_name_or_path
else:
    raise ValueError(f"Unable to read dataset: {dataset_name_or_path}")

if args.output_path:
    OUTPUT_PATH = args.output_path
else:
    model_name = os.path.basename(os.path.normpath(MODEL_PATH))
    OUTPUT_PATH = f"./src/r1-v/eval_outputs/baseline/{model_name}/{base_tag}.json"

output_dir = os.path.dirname(OUTPUT_PATH)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved to: {OUTPUT_PATH}")

# Pre-build the data pairs
pairs = []
skipped = 0
for x in data:
    if x.get("problem_type") == 'multiple choice':
        question = x['problem'] + "Options:\n"
        for op in x.get("options", []):
            question += op + "\n"
    else:
        question = x.get('problem', '')

    raw_path = x.get('path') or x.get('video') or x.get('video_path') or x.get('image') or x.get('image_path')
    if not raw_path:
        skipped += 1
        continue

    # Path resolution
    media_path = raw_path
    if not (str(raw_path).startswith(('http', 'https')) or os.path.isabs(str(raw_path))):
        # Also look next to the JSON manifest
        if is_file:
            json_dir = os.path.dirname(os.path.abspath(dataset_name_or_path))
            potential_path = os.path.join(json_dir, str(raw_path).lstrip('/'))
            if os.path.exists(potential_path):
                media_path = potential_path
            else:
                media_path = resolve_media_path(raw_path)
        else:
            media_path = resolve_media_path(raw_path)

    # Determine the data type (video / image)
    dtype = x.get('data_type', 'video')
    if str(media_path).lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.webp')):
        dtype = 'image'
    
    # Build a bare message without kwargs since media is loaded manually
    user_msg = {
        "role": "user",
        "content": [
            {"type": dtype},  # placeholder
            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE.get(x.get('problem_type', 'free-form'), "")}
        ]
    }
    
    # Keep the resolved path on the sample dict for the manual loader below
    x['resolved_media_path'] = media_path
    x['resolved_dtype'] = dtype
    
    pairs.append((x, [user_msg]))

print(f"[eval_bench] Loaded {len(pairs)} samples, skipped {skipped} empty paths.")

# ==============================================================================
# 3. Scoring helpers
# ==============================================================================
def extract_think(output_str):
    pattern = r'<think>\s*(.*?)\s*</think>'
    match = re.search(pattern, output_str, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def normalize_number(num_str):
    try:
        s = (num_str or "").strip().replace('≈', '').replace('~', '').replace('％', '%')
        if s.endswith('%'):
            return float(s[:-1].replace(',', '')) / 100.0
        return float(s.replace(',', ''))
    except:
        return None

def normalized_relative_score(pred, target, max_range=100.0):
    try:
        score = 1.0 - abs(float(pred) - float(target)) / (max_range if max_range > 0 else 100.0)
        return max(0.0, min(1.0, score))
    except:
        return 0.0

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
    if not torch.is_tensor(pred): pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target): target = torch.tensor(target, dtype=torch.float32)
    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
    thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
    return (rel_error < (1 - thresholds)).float().mean().item()

def reward_fn(sample, model_output, question_type):
    try:
        output_ans = extract_answer(model_output)
        if output_ans == '': output_ans = model_output
        gt_ans = extract_answer(sample.get("solution", ""))
        
        if question_type in ["multiple choice", "boolean"]:
            return 1.0 if output_ans.strip().lower() == gt_ans.strip().lower() else 0.0
        elif question_type == "free-form":
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(gt_ans, output_ans)
            return (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        elif question_type in ("numerical", "regression"):
            out_number = normalize_number(output_ans)
            gt_number = normalize_number(gt_ans)
            if out_number is None or gt_number is None:
                return 0.0
            if question_type == "numerical":
                return normalized_relative_score(out_number, gt_number)
            return threshold_relative_accuracy(out_number, gt_number)
        return 0.0
    except:
        return 0.0

# ==============================================================================
# 4. Batched inference loop
# ==============================================================================
final_output = []
start_idx = 0
mean_acc = []
mean_mra = []

# Resume from a previous run if possible
if os.path.exists(OUTPUT_PATH):
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
            final_output = existing.get("results", [])
            start_idx = len(final_output)
            print(f"Resuming from sample index {start_idx}")
    except:
        pass

for i in tqdm(range(start_idx, len(pairs), BSZ), desc="Processing batches"):
    batch_pairs = pairs[i:i + BSZ]
    batch_messages = [msgs for (_, msgs) in batch_pairs]
    batch_samples = [sample for (sample, _) in batch_pairs]

    # Build the prompt
    prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]

    llm_inputs = []
    
    # Load every video/image in the batch manually
    for idx, (sample, prompt) in enumerate(zip(batch_samples, prompts)):
        media_path = sample['resolved_media_path']
        dtype = sample['resolved_dtype']
        
        multi_modal_data = {}
        
        if dtype == 'video':
            # Load video frames with decord (uint8)
            frames = load_video_frames(media_path, num_frames=NUM_FRAMES)
            multi_modal_data['video'] = frames
        elif dtype == 'image':
            # Load the image with PIL
            image = load_image(media_path)
            multi_modal_data['image'] = image
            
        llm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": multi_modal_data,
            # InternVL's vLLM interface needs no extra fps arguments
        })

    # Run inference
    try:
        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
        batch_output_text = [out.outputs[0].text for out in outputs]
    except Exception as e:
        print(f"Error in batch: {e}")
        batch_output_text = ["<answer>error</answer>"] * len(batch_pairs)

    # Collect the results
    for offset, sample in enumerate(batch_samples):
        model_output = batch_output_text[offset]
        sample["output"] = model_output
        sample["prediction"] = extract_answer(model_output)
        
        # Drop scratch fields to keep the output clean
        if 'resolved_media_path' in sample: del sample['resolved_media_path']
        if 'resolved_dtype' in sample: del sample['resolved_dtype']
        if 'prompt' in sample: del sample['prompt']  # optional: keeps the output file smaller

        q_type = (sample.get("problem_type", "") or "").strip().lower()
        sample["reward"] = reward_fn(sample, model_output, q_type)
        sample['correct'] = True if sample["reward"] == 1.0 else False

        if q_type in ("numerical", "regression"):
            mean_mra.append(sample["reward"])
        else:
            mean_acc.append(sample["reward"])
        
        final_output.append(sample)

    # Save incrementally
    if (i // BSZ) % 5 == 0:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)

# ==============================================================================
# 5. Final aggregation
# ==============================================================================
final_stats = {'mean_acc': 0.0, 'mean_mra': 0.0, 'regression_metric': REGRESSION_METRIC}
if mean_acc:
    final_stats['mean_acc'] = torch.tensor(mean_acc).mean().item()
if mean_mra:
    final_stats['mean_mra'] = torch.tensor(mean_mra).mean().item()

try:
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": final_output, "final_acc": [final_stats]}, f, indent=2, ensure_ascii=False)
    print(f"Final results saved to {OUTPUT_PATH}")
    print(f"Final Stats: {final_stats}")
except Exception as e:
    print(f"Error writing final results: {e}")
