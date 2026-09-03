"""
Ablation script: measure how each input modality affects task progress estimation.

The six supported input modalities:
1. Current State Only (single image) - current state image only
2. Initial State + Current State (Image Pair) - initial + current state images
3. Video Only (single video) - video only
4. Video + Current State (video + image) - video + current state image
5. Initial State + Video (image + video) - initial state image + video
6. Initial State + Video + Current State (image + video + image) - the full input

Usage:
python eval_bench_ablation_modality.py \
    --model_path /path/to/model \
    --data_file /path/to/data.json \
    --modality "video_current" \
    --output_dir ./ablation_results \
    --test_batch_size 64
"""

import os
import json
import re
from tqdm import tqdm
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse
import sys
from pathlib import Path

_CANONICAL_LOADER_DIR = Path(__file__).resolve().parents[1] / "r1-v" / "src" / "open_r1"
sys.path.insert(0, str(_CANONICAL_LOADER_DIR))
from DatasetLoader import dataset_loader


# The six input modalities
MODALITIES = {
    "current_only": {
        "name": "Current State Only",
        "description": "current state image only",
        "prompt_prefix": "Based on the current state shown in the image, "
    },
    "init_current": {
        "name": "Initial State + Current State",
        "description": "initial + current state image pair",
        "prompt_prefix": "Comparing the initial state in the first image and the current state in the second image, "
    },
    "video_only": {
        "name": "Video Only",
        "description": "video only",
        "prompt_prefix": "Based on the progress shown in the video, "
    },
    "video_current": {
        "name": "Video + Current State",
        "description": "video + current state image",
        "prompt_prefix": "Considering the progress shown in the video and my current observation shown in the image, "
    },
    "init_video": {
        "name": "Initial State + Video",
        "description": "initial state image + video",
        "prompt_prefix": "Starting from the initial state shown in the image and observing the progress in the video, "
    },
    "init_video_current": {
        "name": "Initial State + Video + Current State",
        "description": "initial state + video + current state image (full input)",
        "prompt_prefix": "Given the initial state in the first image, the progress shown in the video, and the current state in the final image, "
    }
}


parser = argparse.ArgumentParser(description="Ablation study for task progress estimation")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
parser.add_argument('--data_file', type=str, required=True, help="Path to the data JSON/JSONL file")
parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset for DatasetLoader")
parser.add_argument('--modality', type=str, required=True, choices=list(MODALITIES.keys()), 
                    help="Input modality to test")
parser.add_argument('--output_dir', type=str, default='./ablation_results', help="Output directory")
parser.add_argument('--test_batch_size', type=int, default=16, help="Batch size for testing")
parser.add_argument('--nframes', type=int, default=32, help="Number of frames to extract from video")
parser.add_argument('--tensor_parallel_size', type=int, default=None, help="vLLM tensor parallel size; default uses all visible GPUs")
args = parser.parse_args()

MODEL_PATH = args.model_path
BSZ = args.test_batch_size
MODALITY = args.modality

# Create the output directory
os.makedirs(args.output_dir, exist_ok=True)
OUTPUT_PATH = os.path.join(args.output_dir, f"results_{MODALITY}.json")

print(f"\n{'='*80}")
print(f"Running Ablation Experiment")
print(f"Modality: {MODALITIES[MODALITY]['name']}")
print(f"Description: {MODALITIES[MODALITY]['description']}")
print(f"{'='*80}\n")


# Initialise the model
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=args.tensor_parallel_size or torch.cuda.device_count(),
    max_model_len=8192 * 2,  # raised to fit long sequences
    gpu_memory_utilization=0.8,
    limit_mm_per_prompt={"image": 4, "video": 2},
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

processor.image_processor.num_frames = 256



def extract_first_and_last_frame(video_path, temp_dir):
    """Extract the first frame (initial state) and last frame (current state) of a video."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None
        
        # Read the first frame
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            return None, None
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        first_img = Image.fromarray(first_frame_rgb)
        
        # Seek to the last frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, last_frame = cap.read()
        if not ret:
            # Fall back to the second-to-last frame if the last one cannot be read
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 2))
            ret, last_frame = cap.read()
        
        cap.release()
        
        if not ret:
            return first_img, None
        
        last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        last_img = Image.fromarray(last_frame_rgb)
        
        # Write to a temporary file
        init_path = os.path.join(temp_dir, f"init_{os.path.basename(video_path)}.jpg")
        current_path = os.path.join(temp_dir, f"current_{os.path.basename(video_path)}.jpg")
        
        first_img.save(init_path)
        last_img.save(current_path)
        
        return init_path, current_path
    
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return None, None


def build_content_for_modality(video_path, modality, question_text, temp_dir):
    """Build the content array for the requested modality."""
    content = []
    
    # Extract the initial and current frames
    init_frame_path, current_frame_path = extract_first_and_last_frame(video_path, temp_dir)
    
    if modality == "current_only":
        # 1. Current State Only (single image)
        if current_frame_path and os.path.exists(current_frame_path):
            content.append({"type": "image", "image": current_frame_path})
    
    elif modality == "init_current":
        # 2. Initial State + Current State (Image Pair)
        if init_frame_path and os.path.exists(init_frame_path):
            content.append({"type": "image", "image": init_frame_path})
        if current_frame_path and os.path.exists(current_frame_path):
            content.append({"type": "image", "image": current_frame_path})
    
    elif modality == "video_only":
        # 3. Video Only (single video)
        if os.path.exists(video_path):
            content.append({"type": "video", "video": video_path, "nframes": args.nframes})
    
    elif modality == "video_current":
        # 4. Video + Current State (video + image)
        if os.path.exists(video_path):
            content.append({"type": "video", "video": video_path, "nframes": args.nframes})
        if current_frame_path and os.path.exists(current_frame_path):
            content.append({"type": "image", "image": current_frame_path})
    
    elif modality == "init_video":
        # 5. Initial State + Video (image + video)
        if init_frame_path and os.path.exists(init_frame_path):
            content.append({"type": "image", "image": init_frame_path})
        if os.path.exists(video_path):
            content.append({"type": "video", "video": video_path, "nframes": args.nframes})
    
    elif modality == "init_video_current":
        # 6. Initial State + Video + Current State (image + video + image)
        if init_frame_path and os.path.exists(init_frame_path):
            content.append({"type": "image", "image": init_frame_path})
        if os.path.exists(video_path):
            content.append({"type": "video", "video": video_path, "nframes": args.nframes})
        if current_frame_path and os.path.exists(current_frame_path):
            content.append({"type": "image", "image": current_frame_path})
    
    # Append the rewritten question (modality hint inserted after 'Question:')
    modified_question = inject_prompt_to_question(question_text, MODALITIES[modality]['prompt_prefix'])
    content.append({"type": "text", "text": modified_question})
    
    return content


def inject_prompt_to_question(question_text, prompt_prefix):
    """Insert the modality-specific hint right after 'Question:' in the prompt."""
    # Locate "Question:"
    if "Question:" in question_text:
        parts = question_text.split("Question:", 1)
        modified = parts[0] + "Question:\n" + prompt_prefix + parts[1]
        return modified
    else:
        # If "Question:" is absent, prepend the hint instead
        return prompt_prefix + question_text


# Load data through DatasetLoader
data, video_paths = dataset_loader(args.dataset_name, data_file=args.data_file)
print(f"Loaded {len(data)} samples for dataset {args.dataset_name} from {args.data_file}")


# Temporary directory for extracted frames
temp_dir = tempfile.mkdtemp(prefix="ablation_frames_")
print(f"Temporary directory for frames: {temp_dir}")


# Build the (sample, messages) pairs
pairs = []
skipped = 0

for i, x in enumerate(data):
    video_path = video_paths[i]
    
    if not video_path or not os.path.exists(video_path):
        print(f"Warning: Video not found, skipping sample: {video_path}")
        skipped += 1
        continue
    
    question = x.get('problem', '')
    
    # Build the content for this modality
    content = build_content_for_modality(video_path, MODALITY, question, temp_dir)
    
    if not content or len(content) <= 1:  # text only, no media attached
        print(f"Warning: No valid media content for modality {MODALITY}, skipping")
        skipped += 1
        continue
    
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    
    pairs.append((x, messages))

print(f"Built {len(pairs)} message pairs, skipped {skipped} samples\n")


# Helper functions
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
    except Exception:
        return None


# Name of the formula this harness uses to score progress predictions, recorded in
# the output file so a reader can tell which formula produced a number. The
# ablation sweep scores against a fixed 0-100 progress range, which is a third
# formula distinct from eval_interleave's `linear_relative_accuracy` and the
# baseline harnesses' `threshold_relative_accuracy`. Do not mix the three.
REGRESSION_METRIC = "absolute_range_accuracy"


def normalized_relative_score(pred, target, max_range=100.0):
    """Score a progress prediction as 1 - |pred-target|/max_range, clipped to [0, 1]."""
    try:
        p = float(pred)
        t = float(target)
        if max_range <= 0:
            max_range = 100.0
        score = 1.0 - abs(p - t) / max_range
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def compute_metrics(predictions, ground_truths):
    """Compute evaluation metrics: MAE, RMSE, Accuracy@5, Accuracy@10."""
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    mae = np.mean(np.abs(predictions - ground_truths))
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    acc_5 = np.mean(np.abs(predictions - ground_truths) <= 5) * 100
    acc_10 = np.mean(np.abs(predictions - ground_truths) <= 10) * 100
    
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Acc@5": float(acc_5),
        "Acc@10": float(acc_10)
    }


def recompute_cached_metrics(entries):
    """Rebuild the metric accumulators from cached results.

    Without this, a resumed run reports metrics over only the newly generated
    tail of the dataset instead of every sample.
    """
    predictions = []
    ground_truths = []
    scores = []
    for entry in entries:
        pred_number = entry.get("prediction_numeric")
        gt_number = entry.get("ground_truth_numeric")
        if pred_number is None or gt_number is None:
            # Recompute from the raw text for entries written by older revisions.
            pred_ans = entry.get("prediction") or entry.get("output") or ""
            gt_ans = extract_answer(entry.get("solution", ""))
            pred_number = normalize_number(pred_ans)
            gt_number = normalize_number(gt_ans)
        if pred_number is None or gt_number is None:
            continue
        predictions.append(pred_number)
        ground_truths.append(gt_number)
        scores.append(normalized_relative_score(pred_number, gt_number, max_range=100.0))
    return predictions, ground_truths, scores


# Batch processing
final_output = []
start_idx = 0

# Resume from an existing result file when present
if os.path.exists(OUTPUT_PATH):
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
            final_output = existing.get("results", [])
            start_idx = len(final_output)
            print(f"Resuming from sample index {start_idx}")
    except Exception as e:
        print(f"Error reading existing output file: {e}")

all_predictions = []
all_ground_truths = []
all_scores = []

if final_output:
    all_predictions, all_ground_truths, all_scores = recompute_cached_metrics(final_output)
    print(f"Restored metrics for {len(all_predictions)} cached samples.")

for i in tqdm(range(start_idx, len(pairs), BSZ), desc=f"Processing {MODALITY}"):
    batch_pairs = pairs[i:i + BSZ]
    batch_messages = [msgs for (_, msgs) in batch_pairs]

    prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
               for msg in batch_messages]

    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)

        image_idx = 0
        video_idx = 0

        llm_inputs = []
        for idx, prompt in enumerate(prompts):
            user_content = batch_messages[idx][0]['content']
            
            sample_mm_data = {}
            sample_video_kw = {}
            
            # Count how many images and videos this sample carries
            num_images = sum(1 for item in user_content if item['type'] == 'image')
            num_videos = sum(1 for item in user_content if item['type'] == 'video')
            
            # Collect the images
            if num_images > 0 and image_inputs:
                # vLLM may merge multiple images, so slice them apart correctly
                if image_idx < len(image_inputs):
                    if num_images == 1:
                        sample_mm_data["image"] = image_inputs[image_idx]
                        image_idx += 1
                    else:
                        # Multi-image case
                        sample_mm_data["image"] = image_inputs[image_idx:image_idx + num_images]
                        image_idx += num_images
            
            # Collect the video
            if num_videos > 0 and video_inputs:
                if video_idx < len(video_inputs):
                    sample_mm_data["video"] = video_inputs[video_idx]
                    for key, value in video_kwargs.items():
                        if video_idx < len(value):
                            sample_video_kw[key] = value[video_idx]
                    video_idx += 1

            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": sample_mm_data,
                "mm_processor_kwargs": sample_video_kw,
            })

        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

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
        print(f'Exception during inference: {e}')
        batch_output_text = ['<answer>error</answer>'] * len(batch_pairs)

    # Handle the outputs
    for offset, (sample, sample_messages) in enumerate(batch_pairs):
        model_output = batch_output_text[offset]
        final_ans = extract_answer(model_output)
        
        if final_ans == "":
            final_ans = model_output
        
        # Extract the ground truth
        gt_text = sample.get("solution", "")
        gt_ans = extract_answer(gt_text)
        
        # Convert to a number
        pred_number = normalize_number(final_ans)
        gt_number = normalize_number(gt_ans)
        
        # Score the sample
        if pred_number is not None and gt_number is not None:
            score = normalized_relative_score(pred_number, gt_number, max_range=100.0)
            all_predictions.append(pred_number)
            all_ground_truths.append(gt_number)
            all_scores.append(score)
        else:
            score = 0.0
        
        sample["modality"] = MODALITY
        sample["modality_description"] = MODALITIES[MODALITY]['description']
        sample["prompt"] = sample_messages
        sample["output"] = model_output
        sample["prediction"] = final_ans
        sample["prediction_numeric"] = pred_number
        sample["ground_truth_numeric"] = gt_number
        sample["score"] = score
        
        final_output.append(sample)
    
    # Persist intermediate results
    try:
        intermediate_metrics = {}
        if all_predictions:
            intermediate_metrics = compute_metrics(all_predictions, all_ground_truths)
            intermediate_metrics["mean_score"] = float(np.mean(all_scores))
            intermediate_metrics["regression_metric"] = REGRESSION_METRIC
        
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "modality": MODALITY,
                "modality_info": MODALITIES[MODALITY],
                "results": final_output,
                "metrics": intermediate_metrics,
                "num_samples": len(final_output)
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Processed batch {(i - start_idx)//BSZ + 1}, saved {len(final_output)} samples.")
    except Exception as e:
        print(f"Error writing to output file: {e}")


# Compute the final metrics
if all_predictions:
    final_metrics = compute_metrics(all_predictions, all_ground_truths)
    final_metrics["mean_score"] = float(np.mean(all_scores))
    final_metrics["regression_metric"] = REGRESSION_METRIC
    
    print(f"\n{'='*80}")
    print(f"Final Results for {MODALITIES[MODALITY]['name']}")
    print(f"{'='*80}")
    print(f"Total Samples: {len(final_output)}")
    print(f"MAE: {final_metrics['MAE']:.2f}")
    print(f"RMSE: {final_metrics['RMSE']:.2f}")
    print(f"Accuracy@5: {final_metrics['Acc@5']:.2f}%")
    print(f"Accuracy@10: {final_metrics['Acc@10']:.2f}%")
    print(f"Mean Score: {final_metrics['mean_score']:.4f}")
    print(f"{'='*80}\n")
else:
    final_metrics = {}
    print("Warning: No valid predictions to compute metrics.")

# Write the final results
try:
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "modality": MODALITY,
            "modality_info": MODALITIES[MODALITY],
            "results": final_output,
            "metrics": final_metrics,
            "num_samples": len(final_output)
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {OUTPUT_PATH}")
except Exception as e:
    print(f"Error writing final results: {e}")

# Clean up the temporary directory
try:
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory: {temp_dir}")
except Exception as e:
    print(f"Warning: Could not clean up temporary directory: {e}")
