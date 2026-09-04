"""Single source of truth for every prompt PRIMO R1 uses.

The training entry points, the eval harnesses, and the inference example all
build their messages from the constants below. They used to be copy-pasted into
ten files, which is how they drifted: `SYSTEM_PROMPT` appeared five times,
`TYPE_TEMPLATE` in two variants, and `QUESTION_TEMPLATE` in five.

The answer extractors (`extract_think` / `extract_answer`) are coupled to this
format. Editing `QUESTION_TEMPLATE` or `TYPE_TEMPLATE` without updating them
silently collapses scores instead of raising, so treat them as one unit.

Naming:

- `SYSTEM_PROMPT` / `QUESTION_TEMPLATE` / `TYPE_TEMPLATE` are the PRIMO R1
  format used by the released checkpoint. Import these unless you have a reason
  not to.
- The `*_BASELINE` / `*_PARSER_ONLY` / `*_QUESTION_ONLY` names are the baseline
  and ablation prompts. They are preserved verbatim because the paper's baseline
  numbers were produced with them; do not "fix" them into the PRIMO format.
"""

# ---------------------------------------------------------------------------
# PRIMO R1 — the format the released checkpoint was trained and evaluated on
# ---------------------------------------------------------------------------

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

# Appended to QUESTION_TEMPLATE, keyed by the record's `problem_type`.
TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "boolean": " Please provide only 'Yes' or 'No' as your answer within the <answer> </answer> tags.",
}

# ---------------------------------------------------------------------------
# Baseline and ablation prompts — preserved verbatim
# ---------------------------------------------------------------------------

# Video-R1's original prompt. Used by the local-baseline harness, the CoT
# generator, and the inference example, none of which assume the PRIMO
# planning/observation/reasoning structure.
QUESTION_TEMPLATE_BASELINE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
)

# Same as TYPE_TEMPLATE minus the "boolean" key. Kept separate because the
# baseline harnesses were run without it; adding the key would change which
# records get a type hint.
TYPE_TEMPLATE_BASELINE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
}

# API harness: remote models are prone to wrapping the answer in prose, so the
# prompt is reduced to a parser instruction.
QUESTION_TEMPLATE_PARSER_ONLY = (
    "{Question}\n\n"
    "You are a rigid evaluation parser. You must strictly output ONLY the requested tags and absolutely nothing else."
    "Output format: <answer>number</answer>\n"
)

# InternVL 3.5 harness: the question alone, since the model's own chat template
# supplies the instruction framing.
QUESTION_TEMPLATE_QUESTION_ONLY = "{Question}\n"

# The video-only SFT baseline (`sft_video.py`). A longer, more prescriptive
# variant of the PRIMO prompt that predates it.
QUESTION_TEMPLATE_SFT_VIDEO = (
    "You are an expert AI assistant specializing in embodied procedure and event reasoning. Your task is to analyze visual input (video or images) and reason about the ongoing task. Follow the internal thought process outlined below to structure your analysis before providing a direct answer to the user's query.\n"
    "\n"
    "**QUESTION:**\n"
    "{Question}\n"
    "\n"
    "**QUESTION TYPE:**\n"
    "{question_type}\n"
    "\n"
    "Analyze the provided visual data and the user's question.\n"
    "\n"
    "**OUTPUT FORMAT:**\n"
    "Provide your detailed reasoning between the `<think>` and `</think>` tags, and then give your final answer between the `<answer>` and `</answer>` tags.\n"
    "- The `<think>` block **must** include these three ordered subsections: `<planning>`, `<observation>`, and `<reasoning>`.\n"
    "- The `<answer>` block **must** contain only the final output required by {question_type} and must not include any additional commentary, explanation, or metadata.\n"
    "\n"
    "Below is the required `<think>` / `<answer>` template you must follow.\n"
    "\n"
    "<think>\n"
    "<planning>\n"
    "**Identify the High-Level Goal:**  \n"
    "Determine the most likely overall objective of the agent. What does successful task completion look like?  \n"
    "(e.g., Goal: Assemble a chair. Success state: A fully constructed, stable chair).  \n"
    "**Decompose into Key Steps:**  \n"
    "Break down the high-level goal into a logical sequence of canonical steps. This serves as your mental plan for interpreting the task.  \n"
    "(e.g., 1. Unpack parts. 2. Attach legs to base. 3. Fix backrest. 4. Tighten all screws. 5. Final inspection).  \n"
    "**Plan-Guided Observation and Reasoning:**  \n"
    "When observing, focus on **what evidence** (objects, actions, tools, states, spatial relations) can support identifying which step the agent is performing.  \n"
    "When reasoning, use the plan to **map observed actions** to steps, **assess progress**, **detect anomalies**, and **predict next actions**.  \n"
    "The plan thus serves as a structured reference framework for both visual grounding and logical inference.  \n"
    "</planning>\n"
    "<observation>\n"
    "**Procedural Focus:**  \n"
    "Observe the video as a temporal sequence of steps toward a goal, rather than as a static scene.  \n"
    "Identify where the agent currently is in the broader procedure and how their current actions transition the task from one state to another.  \n"
    "**Objective Scene Description:**  \n"
    "Provide a concise but accurate account of what is happening at this moment in the visual input, noting any visible progress or change relative to earlier states.  \n"
    "**Agent & Action Dynamics:**  \n"
    "Identify who/what the agent is and describe their fine-grained, temporally grounded actions (e.g., grasping, pouring, aligning, tightening).  \n"
    "Emphasize how these actions contribute to the procedural goal rather than only what motion occurs.  \n"
    "**Objects, Artifacts & State Changes:**  \n"
    "List the key objects involved and describe their functional states and transformations over time (e.g., “the screw that was loose is now tightened,” “the liquid level has decreased”).  \n"
    "Focus on state transitions as evidence of task progress.  \n"
    "**Environment, Tools & Spatial Context:**  \n"
    "Summarize the operational context, including tools, supporting surfaces, and spatial arrangements that constrain or facilitate the task.  \n"
    "Note any tool-object interactions or environmental affordances relevant to the ongoing step.  \n"
    "**Temporal Cues (if applicable):**  \n"
    "Pay attention to cues like step boundaries, repetitive motion patterns, or completion indicators that signal progression within the procedural script.  \n"
    "</observation>\n"
    "<reasoning>\n"
    "Please think about this question as if you were a human pondering deeply.  \n"
    'Engage in an internal dialogue using expressions such as *"let me think," "wait," "hmm," "oh, I see,"* or *"let\'s break it down,"* to express a natural thought process.  \n'
    "As you reason, connect your **observations** to your **goal decomposition**, assessing which step the agent is currently performing, how far along they are, and whether their actions appear correct or anomalous.  \n"
    "It's encouraged to include **self-reflection or verification** — for instance, questioning your assumptions, revisiting earlier observations, or confirming that your interpretation aligns with the goal.  \n"
    "You may also predict the agent's next likely action or the upcoming state transition.  \n"
    "Conclude this reasoning section by synthesizing your overall understanding of what the agent is doing, how it fits within the broader task plan, and whether the process seems successful or not.  \n"
    "</reasoning>\n"
    "</think>\n"
    "<answer>\n"
    "[Final answer here — must strictly follow the `{question_type}` output format and include no extra commentary.]\n"
    "</answer>"
)


def build_question(
    question: str, question_type: str, template: str = QUESTION_TEMPLATE, type_template: dict = None
) -> str:
    """Format a question with its type hint appended.

    Mirrors what every call site does by hand:
    ``QUESTION_TEMPLATE.format(...) + TYPE_TEMPLATE[problem_type]``

    Args:
        question: the question text, already including options for MCQ records
        question_type: the record's `problem_type`
        template: which QUESTION_TEMPLATE to use
        type_template: which TYPE_TEMPLATE to use; defaults to the PRIMO one

    Returns:
        The formatted prompt. An unknown `question_type` contributes no hint
        rather than raising, matching the previous per-file behaviour.
    """
    if type_template is None:
        type_template = TYPE_TEMPLATE
    return template.format(Question=question, question_type=question_type) + type_template.get(question_type, "")


__all__ = [
    "SYSTEM_PROMPT",
    "QUESTION_TEMPLATE",
    "TYPE_TEMPLATE",
    "QUESTION_TEMPLATE_BASELINE",
    "TYPE_TEMPLATE_BASELINE",
    "QUESTION_TEMPLATE_PARSER_ONLY",
    "QUESTION_TEMPLATE_QUESTION_ONLY",
    "QUESTION_TEMPLATE_SFT_VIDEO",
    "build_question",
]
