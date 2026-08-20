import os
import json
import re
import cv2
import base64
import requests
import concurrent.futures
from threading import Lock
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
import argparse
import sys
import math
import random
import time


parser = argparse.ArgumentParser(description="Evaluation benchmark via API")
parser.add_argument('--file_name', type=str, required=True, help="Dataset name, json file, or jsonl file")
parser.add_argument('--api_url', type=str, default=os.environ.get('API_URL', 'http://35.220.164.252:3888/v1/chat/completions'), help="API endpoint URL")
parser.add_argument('--api_key', type=str, default=os.environ.get('API_KEY'), help="API key; defaults to API_KEY env var")
parser.add_argument('--model_name', type=str, default=os.environ.get('MODEL_NAME', 'claude-haiku-4-5-20251001'), help="Model name to call")
parser.add_argument('--workers', type=int, default=4, help="Concurrent API workers")
parser.add_argument('--output_path', type=str, default=None, help="Path to save the result json")
parser.add_argument('--output_dir', type=str, default=None, help="Output directory used when output_path is not set")
parser.add_argument('--sample_size', type=int, default=0, help="Random sample size; 0 means full dataset")
parser.add_argument('--seed', type=int, default=42, help="Random seed used when sample_size > 0")
args = parser.parse_args()

API_URL = args.api_url
API_KEY = args.api_key
MODEL_NAME = args.model_name
file_name = args.file_name
MAX_WORKERS = args.workers

if not API_KEY:
    raise ValueError('Please provide --api_key or set API_KEY in the environment.')



DATA_BASE_PATH_ENV = os.environ.get("VIDEO_DATA_ROOT")
_DATA_BASE_PATH = DATA_BASE_PATH_ENV

def get_data_base_path() -> str:
    if _DATA_BASE_PATH is None:
        return "./"
    return _DATA_BASE_PATH

def resolve_media_path(resource_path: str) -> str:
    if not resource_path:
        raise ValueError("Resource path is empty.")
    if str(resource_path).startswith('http') or os.path.isabs(resource_path):
        return resource_path
    base_path = get_data_base_path()
    if resource_path.startswith("./"):
        relative_path = resource_path[2:]
    elif resource_path.startswith("/"):
        relative_path = resource_path[1:]
    else:
        relative_path = resource_path
    return os.path.join(base_path, relative_path)




data = []
dataset_name_or_path = file_name

from pathlib import Path
_CANONICAL_LOADER_DIR = Path(__file__).resolve().parents[1] / "r1-v" / "src" / "open_r1"
sys.path.insert(0, str(_CANONICAL_LOADER_DIR))
try:
    from DatasetLoader import dataset_loader
except Exception:
    dataset_loader = None

is_file = os.path.exists(dataset_name_or_path)

if is_file and (dataset_name_or_path.endswith('.jsonl') or dataset_name_or_path.endswith('.json')):
    if dataset_name_or_path.endswith('.jsonl'):
        with open(dataset_name_or_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
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
    if not is_file:
         print(f"File not found: {dataset_name_or_path}, assuming it is a dataset name but DatasetLoader not found.")
         raise ValueError(f"无法读取数据：文件不存在且无法加载数据集 loader。Path: {dataset_name_or_path}")

if args.output_path:
    OUTPUT_PATH = args.output_path
else:
    output_dir = args.output_dir or os.path.join("src", "r1-v", "eval_outputs", "api", MODEL_NAME)
    OUTPUT_PATH = os.path.join(output_dir, f"{base_tag}.json")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
print(f"Results will be saved to: {OUTPUT_PATH}")

QUESTION_TEMPLATE = "{Question}\n\nYou are a rigid evaluation parser. You must strictly output ONLY the requested tags and absolutely nothing else.Output format: <answer>number</answer>\n"
TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "boolean": " Please provide only 'Yes' or 'No' as your answer within the <answer> </answer> tags."
}

# 预处理数据对
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

    try:
        if str(raw_path).startswith(('http://', 'https://')) or os.path.isabs(str(raw_path)):
            media_path = raw_path
        else:
            if is_file:
                json_dir = os.path.dirname(os.path.abspath(dataset_name_or_path))
                potential_path = os.path.join(json_dir, str(raw_path).lstrip('/'))
                if os.path.exists(potential_path):
                    media_path = potential_path
                else:
                    media_path = resolve_media_path(raw_path)
            else:
                media_path = resolve_media_path(raw_path)
    except Exception:
        media_path = raw_path 

    if not (str(media_path).startswith("http") or os.path.exists(media_path)):
        print(f"[eval_bench] Warning: media not found, skipping sample: {media_path}")
        skipped += 1
        continue

    dtype = x.get('data_type', 'video')

    # 原版帧数控制逻辑
    target_nframes = 16
    if dtype == 'video' and not str(media_path).startswith("http"):
        try:
            cap = cv2.VideoCapture(media_path)
            actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if actual_frames > 0:
                target_nframes = max(2, min(16, actual_frames))
        except Exception as e:
            pass 

    full_question = QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE.get(x.get('problem_type', 'free-form'), "")
    
    sample_info = {
        "media_path": media_path,
        "dtype": dtype,
        "target_nframes": target_nframes,
        "text_prompt": full_question
    }
    pairs.append((x, sample_info))

print(f"[eval_bench] Loaded {len(pairs)} samples, skipped {skipped} missing media files.")

if args.sample_size > 0 and len(pairs) > args.sample_size:
    random.seed(args.seed)
    pairs = random.sample(pairs, args.sample_size)
    print(f"Sampled {args.sample_size} examples for API evaluation with seed {args.seed}.")
else:
    print(f"Loaded {len(pairs)} examples for API evaluation.")

# 断点续传逻辑
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
        s = s.replace('≈', '').replace('~', '').replace('％', '%')
        if s.endswith('%'):
            s = s[:-1]
            s = s.replace(',', '')
            return float(s) / 100.0
        s = s.replace(',', '')
        return float(s)
    except Exception as e:
        return None
        
def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
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

def reward_fn(sample, model_output, question_type):
    try:
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
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(gt_ans, output_ans)
            average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
            return max(0.0, min(1.0, average_fmeasure))
        elif question_type == "numerical":
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
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return mean_relative_accuracy(out_number, gt_number)
        else:
            return 0.0
    except Exception as e:
        return 0.0

# 收集历史断点的指标
mean_acc = []
mean_mra = []
def recompute_cached_rewards(entries):
    cached_acc, cached_mra = [], []
    for entry in entries:
        q_type = (entry.get("problem_type", "") or "").strip().lower()
        cached_output = entry.get("output") or entry.get("prediction") or ""
        new_reward = reward_fn(entry, cached_output, q_type)
        entry["reward"] = new_reward
        entry["correct"] = True if new_reward == 1.0 else False
        if q_type in ("numerical", "regression"):
            cached_mra.append(new_reward)
        else:
            cached_acc.append(new_reward)
    return cached_acc, cached_mra

if final_output:
    cached_acc, cached_mra = recompute_cached_rewards(final_output)
    mean_acc.extend(cached_acc)
    mean_mra.extend(cached_mra)


def extract_frames_to_base64(media_path, target_nframes):
    cap = cv2.VideoCapture(media_path)
    actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if actual_frames <= 0 or target_nframes <= 0:
        cap.release()
        return []

    frame_indices = [int(i * (actual_frames - 1) / (target_nframes - 1)) for i in range(target_nframes)]
    base64_frames = []
    
    # 论文要求的最大像素上限
    max_pixels = 256 * 28 * 28 
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 【完美对齐】：计算等比例缩放，绝不破坏长宽比
            h, w = frame.shape[:2]
            if h * w > max_pixels:
                beta = math.sqrt((h * w) / max_pixels)
                new_h = int(h / beta)
                new_w = int(w / beta)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            _, buffer = cv2.imencode(".jpg", frame)
            base64_str = base64.b64encode(buffer).decode("utf-8")
            base64_frames.append(base64_str)
            
    cap.release()
    return base64_frames

def process_single_sample_api(sample_idx, sample, sample_info):
    media_path = sample_info["media_path"]
    dtype = sample_info["dtype"]
    target_nframes = sample_info["target_nframes"]
    text_prompt = sample_info["text_prompt"]

    content_list = [{"type": "text", "text": text_prompt}]

    try:
        max_pixels = 256 * 28 * 28
        if dtype == "video" and not str(media_path).startswith("http"):
            b64_frames = extract_frames_to_base64(media_path, target_nframes)
            for b64 in b64_frames:
                content_list.append({
                    "type": "image_url",
                    # 【完美对齐】：改为 high，获取 255 tokens 的视力
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
                })
        elif dtype == "image" and not str(media_path).startswith("http"):
            frame = cv2.imread(media_path)
            if frame is not None:
                # 单图同样使用等比例缩放
                h, w = frame.shape[:2]
                if h * w > max_pixels:
                    beta = math.sqrt((h * w) / max_pixels)
                    frame = cv2.resize(frame, (int(w / beta), int(h / beta)), interpolation=cv2.INTER_CUBIC)
                    
                _, buffer = cv2.imencode(".jpg", frame)
                b64 = base64.b64encode(buffer).decode("utf-8")
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
                })
    except Exception as e:
        # ===== 修改：新增返回耗时 0.0 和 token 0 =====
        return sample_idx, sample, f"<answer>error: {str(e)}</answer>", 0.0, 0

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content_list}],
        "max_tokens": 1024,
        "temperature": 0.1,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    max_retries = 5
    for attempt in range(max_retries):
        start_time = time.time()  # ===== 新增：请求开始计秒表 =====
        try:
            # 1. 调大 timeout 到 300，以适应 32 帧高清图的极长处理时间
            response = requests.post(API_URL, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            
            # ===== 新增：请求成功，计算耗时 =====
            end_time = time.time()
            generation_time = end_time - start_time
            
            resp_json = response.json()
            output_text = resp_json["choices"][0]["message"]["content"]
            
            # ===== 新增：尝试从 API 响应中提取 Token 数量 =====
            token_count = resp_json.get("usage", {}).get("completion_tokens", 0)
            
            # 2. 成功获取结果后，强制休眠 2 秒，给中转站喘息空间，防止并发超限
            time.sleep(2)
            
            # 返回包含了时间和 token
            return sample_idx, sample, output_text, generation_time, token_count

        except Exception as e:
            if attempt < max_retries - 1:
                # 3. 如果请求失败（如 Read timed out），等待一段时间后重试
                wait_time = (attempt + 1) * 5 
                print(f"⚠️ 样本请求失败，{wait_time}秒后进行第 {attempt+2} 次尝试... 错误: {e}")
                time.sleep(wait_time)
                continue
            else:
                # 4. 彻底失败时，必须将错误信息包装在 <answer> 标签中，防止后续代码解析提取为空
                output_text = f"<answer>error API after {max_retries} retries: {str(e)}</answer>"
                return sample_idx, sample, output_text, 0.0, 0

write_lock = Lock()
print(f"🚀 准备开启多线程评测，并发数量: {MAX_WORKERS}...")

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(process_single_sample_api, idx, sample, sample_info): idx
        for idx, (sample, sample_info) in enumerate(pairs[start_idx:], start=start_idx)
    }

    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="API Evaluation Progress"):
        # ===== 修改：解包接收增加的时间和 token =====
        idx, sample, model_output, gen_time, token_count = future.result()

        think_chain = extract_think(model_output)
        final_ans = extract_answer(model_output)
        if final_ans == "":
            final_ans = model_output
            
        sample["output"] = model_output
        sample["prediction"] = final_ans
        
        # ===== 新增：写入样本字典 =====
        sample["generation_time"] = round(gen_time, 4)
        sample["output_tokens"] = token_count
        # ===========================
        
        q_type = (sample.get("problem_type", "") or "").strip().lower()
        sample["reward"] = reward_fn(sample, model_output, q_type)
        sample['correct'] = True if sample["reward"] == 1.0 else False

        if q_type in ("numerical", "regression"):
            mean_mra.append(sample["reward"])
        else:
            mean_acc.append(sample["reward"])
            
        if think_chain:
            sample["process"] = f"<think>{think_chain}</think>"
            
        # 实时存入文件以防止意外中断丢失进度
        with write_lock:
            final_output.append(sample)
            try:
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
            except Exception as e:
                pass


final_acc={'mean_acc': 0.0, 'mean_mra': 0.0}
if mean_acc != []:
    final_acc['mean_acc'] = torch.tensor(mean_acc).mean().item()
if mean_mra != []:
    final_acc['mean_mra'] = torch.tensor(mean_mra).mean().item()

# ===== 新增：汇总平均时间和 Token =====
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
    # 打印最终面板
    print(f"Final Stats: {json.dumps(final_acc, indent=2)}")
except Exception as e:
    print(f"Error writing final accuracy to output file: {e}")
