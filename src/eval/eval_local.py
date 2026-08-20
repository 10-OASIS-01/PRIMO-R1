import os
import json
import re
import cv2
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse
import sys


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
    """
    与原评测一致：视频不足 requested 帧时使用实际帧数，至少保留 2 帧。
    """
    req = int(requested)
    total = get_total_frames_cv2(video_path)
    if total is not None:
        req = min(req, total)
    req = max(2, req)
    return req


def get_total_frames_cv2(video_path: str) -> int | None:
    """Return total frame count if local video readable, else None."""
    try:
        if str(video_path).startswith("http://") or str(video_path).startswith("https://"):
            return None  # 不支持 HTTP/HTTPS 视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total if total > 0 else None
    except Exception:
        return None

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
单文件/数据集评测：
- 如果 --file_name 是存在的 .json/.jsonl 路径：直接读取该文件中的样本。
- 如果 --file_name 不是文件路径：按数据集名称，参考 DatasetLoader.py 加载（返回 json_list + 绝对视频路径）。
输出文件名根据 file_name（或数据集名）自动生成。
"""

# 解析输入：路径或数据集名
data = []
dataset_name_or_path = file_name

# 为了支持 DatasetLoader 直接加载，加入项目自带 loader 所在目录到 sys.path。
from pathlib import Path
_CANONICAL_LOADER_DIR = Path(__file__).resolve().parents[1] / "r1-v" / "src" / "open_r1"
sys.path.insert(0, str(_CANONICAL_LOADER_DIR))
try:
    from DatasetLoader import dataset_loader
except Exception:
    dataset_loader = None

is_file = os.path.exists(dataset_name_or_path)

if is_file and (dataset_name_or_path.endswith('.jsonl') or dataset_name_or_path.endswith('.json')):
    # 直接读取用户提供的 JSON/JSONL
    if dataset_name_or_path.endswith('.jsonl'):
        with open(dataset_name_or_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(dataset_name_or_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    base_tag = os.path.splitext(os.path.basename(dataset_name_or_path))[0]
elif (not is_file) and dataset_loader is not None:
    # 作为数据集名，使用 DatasetLoader 读取
    json_list, video_paths = dataset_loader(dataset_name_or_path)
    # 将绝对视频路径写回样本，避免后续路径解析不一致
    for entry, vpath in zip(json_list, video_paths):
        entry = dict(entry)
        entry['path'] = vpath
        data.append(entry)
    base_tag = dataset_name_or_path
else:
    raise ValueError(f"无法读取数据：请提供存在的 .json/.jsonl 文件，或可由 DatasetLoader 识别的数据集名。当前: {dataset_name_or_path}")

if args.output_path:
    OUTPUT_PATH = args.output_path
else:
    OUTPUT_PATH = f"./src/r1-v/eval_outputs/local/{base_tag}.json"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
print(f"Results will be saved to: {OUTPUT_PATH}")

QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
)

# QUESTION_TEMPLATE = """You are an expert AI assistant specializing in embodied procedure and event reasoning. Your task is to analyze visual input (video or images) and reason about the ongoing task. Follow the internal thought process outlined below to structure your analysis before providing a direct answer to the user's query.

# **QUESTION:**
# {Question}

# **QUESTION TYPE:**
# {question_type}

# Analyze the provided visual data and the user's question.

# **OUTPUT FORMAT:**
# Provide your detailed reasoning between the `<think>` and `</think>` tags, and then give your final answer between the `<answer>` and `</answer>` tags.
# - The `<think>` block **must** include these three ordered subsections: `<planning>`, `<observation>`, and `<reasoning>`.
# - The `<answer>` block **must** contain only the final output required by {question_type} and must not include any additional commentary, explanation, or metadata.

# Below is the required `<think>` / `<answer>` template you must follow.

# <think>
# <planning>
# **Identify the High-Level Goal:**  
# Determine the most likely overall objective of the agent. What does successful task completion look like?  
# (e.g., Goal: Assemble a chair. Success state: A fully constructed, stable chair).  
# **Decompose into Key Steps:**  
# Break down the high-level goal into a logical sequence of canonical steps. This serves as your mental plan for interpreting the task.  
# (e.g., 1. Unpack parts. 2. Attach legs to base. 3. Fix backrest. 4. Tighten all screws. 5. Final inspection).  
# **Plan-Guided Observation and Reasoning:**  
# When observing, focus on **what evidence** (objects, actions, tools, states, spatial relations) can support identifying which step the agent is performing.  
# When reasoning, use the plan to **map observed actions** to steps, **assess progress**, **detect anomalies**, and **predict next actions**.  
# The plan thus serves as a structured reference framework for both visual grounding and logical inference.  
# </planning>
# <observation>
# **Procedural Focus:**  
# Observe the video as a temporal sequence of steps toward a goal, rather than as a static scene.  
# Identify where the agent currently is in the broader procedure and how their current actions transition the task from one state to another.  
# **Objective Scene Description:**  
# Provide a concise but accurate account of what is happening at this moment in the visual input, noting any visible progress or change relative to earlier states.  
# **Agent & Action Dynamics:**  
# Identify who/what the agent is and describe their fine-grained, temporally grounded actions (e.g., grasping, pouring, aligning, tightening).  
# Emphasize how these actions contribute to the procedural goal rather than only what motion occurs.  
# **Objects, Artifacts & State Changes:**  
# List the key objects involved and describe their functional states and transformations over time (e.g., “the screw that was loose is now tightened,” “the liquid level has decreased”).  
# Focus on state transitions as evidence of task progress.  
# **Environment, Tools & Spatial Context:**  
# Summarize the operational context, including tools, supporting surfaces, and spatial arrangements that constrain or facilitate the task.  
# Note any tool-object interactions or environmental affordances relevant to the ongoing step.  
# **Temporal Cues (if applicable):**  
# Pay attention to cues like step boundaries, repetitive motion patterns, or completion indicators that signal progression within the procedural script.  
# </observation>
# <reasoning>
# Please think about this question as if you were a human pondering deeply.  
# Engage in an internal dialogue using expressions such as *"let me think," "wait," "hmm," "oh, I see,"* or *"let's break it down,"* to express a natural thought process.  
# As you reason, connect your **observations** to your **goal decomposition**, assessing which step the agent is currently performing, how far along they are, and whether their actions appear correct or anomalous.  
# It's encouraged to include **self-reflection or verification** — for instance, questioning your assumptions, revisiting earlier observations, or confirming that your interpretation aligns with the goal.  
# You may also predict the agent's next likely action or the upcoming state transition.  
# Conclude this reasoning section by synthesizing your overall understanding of what the agent is doing, how it fits within the broader task plan, and whether the process seems successful or not.  
# </reasoning>
# </think>
# <answer>
# [Final answer here — must strictly follow the `{question_type}` output format and include no extra commentary.]
# </answer>"""


#     # --- 完整的 QUESTION_TEMPLATE (Reasoning) ---
# QUESTION_TEMPLATE = """You are an expert AI assistant specializing in embodied procedure and event reasoning. Your task is to analyze visual input (video or images) and reason about the ongoing task. Follow the internal thought process outlined below to structure your analysis before providing a direct answer to the user's query.
# **QUESTION:**
# {Question}

# **QUESTION TYPE:**
# {question_type}

# Analyze the provided visual data and the user's question.

# **OUTPUT FORMAT:**
# Provide your detailed reasoning between the `<think>` and `</think>` tags, and then give your final answer between the `<answer>` and `</answer>` tags.
# - The `<think>` block **must** include one subsection: `<reasoning>`.
# - The `<answer>` block **must** contain only the final output required by {question_type} and must not include any additional commentary, explanation, or metadata.

# Below is the required `<think>` / `<answer>` template you must follow.

# <think>
# <reasoning>
# Please think about this question as if you were a human pondering deeply.  
# Engage in an internal dialogue using expressions such as *"let me think," "wait," "hmm," "oh, I see,"* or *"let's break it down,"* to express a natural thought process.  
# It's encouraged to include **self-reflection or verification** — for instance, questioning your assumptions, revisiting earlier observations, or confirming that your interpretation aligns with the goal.  
# You may also predict the agent's next likely action or the upcoming state transition.  
# </reasoning>
# </think>
# <answer>
# [Final answer here — must strictly follow the `{question_type}` output format and include no extra commentary.]
# </answer>"""

#     # --- 完整的 QUESTION_TEMPLATE (Planning + Reasoning) ---
# QUESTION_TEMPLATE = """You are an expert AI assistant specializing in embodied procedure and event reasoning. Your task is to analyze visual input (video or images) and reason about the ongoing task. Follow the internal thought process outlined below to structure your analysis before providing a direct answer to the user's query.
# **QUESTION:**
# {Question}

# **QUESTION TYPE:**
# {question_type}

# Analyze the provided visual data and the user's question.

# **OUTPUT FORMAT:**
# Provide your detailed reasoning between the `<think>` and `</think>` tags, and then give your final answer between the `<answer>` and `</answer>` tags.
# - The `<think>` block **must** include these two ordered subsections: `<planning>`, `<reasoning>`.
# - The `<answer>` block **must** contain only the final output required by {question_type} and must not include any additional commentary, explanation, or metadata.

# Below is the required `<think>` / `<answer>` template you must follow.

# <think>
# <planning>
# **Identify the High-Level Goal:**  
# Determine the most likely overall objective of the agent. What does successful task completion look like?  
# (e.g., Goal: Assemble a chair. Success state: A fully constructed, stable chair).  
# **Decompose into Key Steps:**  
# Break down the high-level goal into a logical sequence of canonical steps. This serves as your mental plan for interpreting the task.  
# (e.g., 1. Unpack parts. 2. Attach legs to base. 3. Fix backrest. 4. Tighten all screws. 5. Final inspection).  
# **Plan-Guided Reasoning:**  
# When reasoning, use the plan to **map observed actions** to steps, **assess progress**, **detect anomalies**, and **predict next actions**.  
# The plan thus serves as a structured reference framework for both visual grounding and logical inference.  
# </planning>
# <reasoning>
# Please think about this question as if you were a human pondering deeply.  
# Engage in an internal dialogue using expressions such as *"let me think," "wait," "hmm," "oh, I see,"* or *"let's break it down,"* to express a natural thought process.  
# As you reason, connect your **planning** to your **reasoning**, assessing which step the agent is currently performing, how far along they are, and whether their actions appear correct or anomalous.  
# It's encouraged to include **self-reflection or verification** — for instance, questioning your assumptions, revisiting earlier observations, or confirming that your interpretation aligns with the goal.  
# You may also predict the agent's next likely action or the upcoming state transition.  
# Conclude this reasoning section by synthesizing your overall understanding of what the agent is doing, how it fits within the broader task plan, and whether the process seems successful or not.  
# </reasoning>
# </think>
# <answer>
# [Final answer here — must strictly follow the `{question_type}` output format and include no extra commentary.]
# </answer>"""

#     # --- 完整的 QUESTION_TEMPLATE (Observation+Reasoning) ---
# QUESTION_TEMPLATE = """You are an expert AI assistant specializing in embodied procedure and event reasoning. Your task is to analyze visual input (video or images) and reason about the ongoing task. Follow the internal thought process outlined below to structure your analysis before providing a direct answer to the user's query.
# **QUESTION:**
# {Question}

# **QUESTION TYPE:**
# {question_type}

# Analyze the provided visual data and the user's question.

# **OUTPUT FORMAT:**
# Provide your detailed reasoning between the `<think>` and `</think>` tags, and then give your final answer between the `<answer>` and `</answer>` tags.
# - The `<think>` block **must** include these two ordered subsections: `<observation>`, and `<reasoning>`.
# - The `<answer>` block **must** contain only the final output required by {question_type} and must not include any additional commentary, explanation, or metadata.

# Below is the required `<think>` / `<answer>` template you must follow.

# <think>
# <observation>
# **Procedural Focus:**  
# Observe the video as a temporal sequence of steps toward a goal, rather than as a static scene.  
# Identify where the agent currently is in the broader procedure and how their current actions transition the task from one state to another.  
# **Objective Scene Description:**  
# Provide a concise but accurate account of what is happening at this moment in the visual input, noting any visible progress or change relative to earlier states.  
# **Agent & Action Dynamics:**  
# Identify who/what the agent is and describe their fine-grained, temporally grounded actions (e.g., grasping, pouring, aligning, tightening).  
# Emphasize how these actions contribute to the procedural goal rather than only what motion occurs.  
# **Objects, Artifacts & State Changes:**  
# List the key objects involved and describe their functional states and transformations over time (e.g., “the screw that was loose is now tightened,” “the liquid level has decreased”).  
# Focus on state transitions as evidence of task progress.  
# **Environment, Tools & Spatial Context:**  
# Summarize the operational context, including tools, supporting surfaces, and spatial arrangements that constrain or facilitate the task.  
# Note any tool-object interactions or environmental affordances relevant to the ongoing step.  
# **Temporal Cues (if applicable):**  
# Pay attention to cues like step boundaries, repetitive motion patterns, or completion indicators that signal progression within the procedural script.  
# </observation>
# <reasoning>
# Please think about this question as if you were a human pondering deeply.  
# Engage in an internal dialogue using expressions such as *"let me think," "wait," "hmm," "oh, I see,"* or *"let's break it down,"* to express a natural thought process.  
# As you reason, connect your observations to your reasoning, assessing which step the agent is currently performing, how far along they are, and whether their actions appear correct or anomalous.  
# It's encouraged to include **self-reflection or verification** — for instance, questioning your assumptions, revisiting earlier observations, or confirming that your interpretation aligns with the goal.  
# You may also predict the agent's next likely action or the upcoming state transition.  
# Conclude this reasoning section by synthesizing your overall understanding of what the agent is doing, how it fits within the broader task plan, and whether the process seems successful or not.  
# </reasoning>
# </think>
# <answer>
# [Final answer here — must strictly follow the `{question_type}` output format and include no extra commentary.]
# </answer>"""

#     # --- 完整的 QUESTION_TEMPLATE (All) ---
# QUESTION_TEMPLATE = """You are an expert AI assistant specializing in embodied procedure and event reasoning. Your task is to analyze visual input (video or images) and reason about the ongoing task. Follow the internal thought process outlined below to structure your analysis before providing a direct answer to the user's query.
# **QUESTION:**
# {Question}

# **QUESTION TYPE:**
# {question_type}

# Analyze the provided visual data and the user's question.

# **OUTPUT FORMAT:**
# Provide your detailed reasoning between the `<think>` and `</think>` tags, and then give your final answer between the `<answer>` and `</answer>` tags.
# - The `<think>` block **must** include these three ordered subsections: `<planning>`, `<observation>`, and `<reasoning>`.
# - The `<answer>` block **must** contain only the final output required by {question_type} and must not include any additional commentary, explanation, or metadata.

# Below is the required `<think>` / `<answer>` template you must follow.

# <think>
# <planning>
# **Identify the High-Level Goal:**  
# Determine the most likely overall objective of the agent. What does successful task completion look like?  
# (e.g., Goal: Assemble a chair. Success state: A fully constructed, stable chair).  
# **Decompose into Key Steps:**  
# Break down the high-level goal into a logical sequence of canonical steps. This serves as your mental plan for interpreting the task.  
# (e.g., 1. Unpack parts. 2. Attach legs to base. 3. Fix backrest. 4. Tighten all screws. 5. Final inspection).  
# **Plan-Guided Observation and Reasoning:**  
# When observing, focus on **what evidence** (objects, actions, tools, states, spatial relations) can support identifying which step the agent is performing.  
# When reasoning, use the plan to **map observed actions** to steps, **assess progress**, **detect anomalies**, and **predict next actions**.  
# The plan thus serves as a structured reference framework for both visual grounding and logical inference.  
# </planning>
# <observation>
# **Procedural Focus:**  
# Observe the video as a temporal sequence of steps toward a goal, rather than as a static scene.  
# Identify where the agent currently is in the broader procedure and how their current actions transition the task from one state to another.  
# **Objective Scene Description:**  
# Provide a concise but accurate account of what is happening at this moment in the visual input, noting any visible progress or change relative to earlier states.  
# **Agent & Action Dynamics:**  
# Identify who/what the agent is and describe their fine-grained, temporally grounded actions (e.g., grasping, pouring, aligning, tightening).  
# Emphasize how these actions contribute to the procedural goal rather than only what motion occurs.  
# **Objects, Artifacts & State Changes:**  
# List the key objects involved and describe their functional states and transformations over time (e.g., “the screw that was loose is now tightened,” “the liquid level has decreased”).  
# Focus on state transitions as evidence of task progress.  
# **Environment, Tools & Spatial Context:**  
# Summarize the operational context, including tools, supporting surfaces, and spatial arrangements that constrain or facilitate the task.  
# Note any tool-object interactions or environmental affordances relevant to the ongoing step.  
# **Temporal Cues (if applicable):**  
# Pay attention to cues like step boundaries, repetitive motion patterns, or completion indicators that signal progression within the procedural script.  
# </observation>
# <reasoning>
# Please think about this question as if you were a human pondering deeply.  
# Engage in an internal dialogue using expressions such as *"let me think," "wait," "hmm," "oh, I see,"* or *"let's break it down,"* to express a natural thought process.  
# As you reason, connect your **observations** to your **goal decomposition**, assessing which step the agent is currently performing, how far along they are, and whether their actions appear correct or anomalous.  
# It's encouraged to include **self-reflection or verification** — for instance, questioning your assumptions, revisiting earlier observations, or confirming that your interpretation aligns with the goal.  
# You may also predict the agent's next likely action or the upcoming state transition.  
# Conclude this reasoning section by synthesizing your overall understanding of what the agent is doing, how it fits within the broader task plan, and whether the process seems successful or not.  
# </reasoning>
# </think>
# <answer>
# [Final answer here — must strictly follow the `{question_type}` output format and include no extra commentary.]
# </answer>"""


#     # --- 完整的 QUESTION_TEMPLATE (All_easy) ---
# QUESTION_TEMPLATE = """You are an expert AI assistant specializing in embodied procedure and event reasoning. Your task is to analyze visual input (video or images) and reason about the ongoing task. Follow the internal thought process outlined below to structure your analysis before providing a direct answer to the user's query.
# **QUESTION:**
# {Question}

# **QUESTION TYPE:**
# {question_type}

# Analyze the provided visual data and the user's question.

# **OUTPUT FORMAT:**
# Provide your detailed reasoning between the `<think>` and `</think>` tags, and then give your final answer between the `<answer>` and `</answer>` tags.
# - The `<think>` block **must** include these three ordered subsections: `<planning>`, `<observation>`, and `<reasoning>`.
# - The `<answer>` block **must** contain only the final output required by {question_type} and must not include any additional commentary, explanation, or metadata.

# Below is the required `<think>` / `<answer>` template you must follow.

# <think>
# <planning>
# Identify the high-level goal of the agent — what does successful completion look like?
# Decompose the overall goal into a sequence of canonical steps that represent the procedural structure of the task.
# Use this plan as a reference for interpreting the agent’s actions, mapping observed behaviors to steps, assessing progress, detecting anomalies, and predicting what should happen next. 
# </planning>
# <observation>
# View the video as a temporal sequence of actions contributing to the overarching procedure.
# Objectively describe what is occurring in the current moment, focusing on evidence of progress or state change.
# Identify the agent's fine-grained actions and explain how they move the task forward.
# List relevant objects, tools, and environmental context, emphasizing functional states and transformations.
# Note any cues—repetition, transitions, or completion indicators—that help situate the action within the procedural script. 
# </observation>
# <reasoning>
# Think through the question as a human would, with natural internal dialogue (e.g., “let me think,” “hmm,” “wait,” “oh, I see”).
# Connect observations to the procedural plan, determining which step is being executed, how far along the agent is, and whether the action appears correct or anomalous.
# Reflect on your assumptions, verify interpretations, and, if appropriate, predict the agent’s next likely action.
# Synthesize an understanding of what the agent is doing, how it fits into the broader task, and whether the process seems successful. 
# </reasoning>
# </think>
# <answer>
# [Final answer here — must strictly follow the `{question_type}` output format and include no extra commentary.] 
# </answer>"""

# test
# QUESTION_TEMPLATE = (
#     "QUESTION:\n{Question}\n\n"
#     "QUESTION TYPE:\n{question_type}\n\n"
#     "Analyze the provided visual data and reason about the ongoing task.\n\n"
#     "Please think about this question as if you were a human pondering deeply. "
#     "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'hmm', 'oh, I see', 'let's break it down', etc. "
#     "You are encouraged to include self-reflection or verification in your reasoning process.\n\n"
#     "Provide your detailed reasoning between the <think> and </think> tags, following the subsections <planning>, <observation>, and <reasoning>. "
#     "Then give your final answer between the <answer> and </answer> tags.\n\n"
#     "Below is the required template:\n\n"
#     "<think>\n"
#     "<planning>\n"
#     "Identify the high-level goal of the agent — what does successful completion look like?\n"
#     "Decompose the overall goal into a sequence of canonical steps representing the procedural structure of the task.\n"
#     "Use this plan to interpret actions, map observed behaviors to steps, assess progress, detect anomalies, and predict what happens next.\n"
#     "</planning>\n"
#     "<observation>\n"
#     "View the video as a temporal sequence of actions contributing to the procedure.\n"
#     "Objectively describe what is occurring in the current moment, noting evidence of progress or state changes.\n"
#     "Identify fine-grained actions and explain how they move the task forward.\n"
#     "List relevant objects, tools, and environmental context, emphasizing functional states and transformations.\n"
#     "Note cues—repetition, transitions, or completion indicators—that situate the action in the procedural script.\n"
#     "</observation>\n"
#     "<reasoning>\n"
#     "Think through the question as a human would, with natural internal dialogue.\n"
#     "Connect observations to the procedural plan to determine which step is being executed, progress, correctness, or anomalies.\n"
#     "Reflect on assumptions, verify interpretations, and, if appropriate, predict the agent’s next likely action.\n"
#     "Synthesize understanding of what the agent is doing, how it fits into the broader task, and whether the process seems successful.\n"
#     "</reasoning>\n"
#     "</think>\n"
#     "<answer>\n"
#     "[Final answer here — strictly follow the `{question_type}` output format and include no extra commentary.]\n"
#     "</answer>"
# )


TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "boolean": " Please provide only 'Yes' or 'No' as your answer within the <answer> </answer> tags."
}


# Build (sample, messages) pairs using sft-style message structure，直接用 JSON 中的视频路径
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

    # resolve media path: 优先支持 http/https 或绝对路径；否则用 VIDEO_DATA_ROOT；再次回退到 JSON 所在目录/其父级
    if str(raw_path).startswith('http://') or str(raw_path).startswith('https://') or os.path.isabs(raw_path):
        media_path = raw_path
    else:
        try:
            media_path = resolve_media_path(raw_path)
        except Exception:
            # VIDEO_DATA_ROOT 未设置时回退：尝试以 JSON 文件目录为根
            if is_file:
                json_dir = os.path.dirname(os.path.abspath(dataset_name_or_path))
                # 尝试逐级父目录拼接
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
                # 数据集名场景，DatasetLoader 已返回绝对路径，这里不应到达
                media_path = raw_path

    # skip missing local files
    if not (str(media_path).startswith("http://") or str(media_path).startswith("https://")) and not os.path.exists(media_path):
        print(f"[eval_bench] Warning: media not found, skipping sample: {media_path}")
        skipped += 1
        continue

    dtype = x.get('data_type', 'video')

    SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The Assistant is an expert AI specializing in embodied procedure and event reasoning based on visual input (video or images). "
    "The assistant must strictly follow a specific thought process and output format. "
    "The reasoning process is enclosed within <think> </think> tags, and the final answer is within <answer> </answer> tags. "
    "The <think> block must contain three ordered subsections: <planning>, <observation>, and <reasoning>. "
    "The <answer> block must contain only the final output required by the question type and no other commentary."
    )
    
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
            # 使用 ROUGE 分数计算 free-form 类型的 reward
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
            return mean_relative_accuracy(out_number, gt_number)
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

        # Aggregation: numerical & regression 归入 mean_mra; 其他类型(包括 free-form)归入 mean_acc
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

final_acc={'mean_acc': 0.0, 'mean_mra': 0.0}
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
