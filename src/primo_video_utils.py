"""Shared video/frame helpers for the interleaved `I_init + V_seq + I_curr` format.

PRIMO R1 anchors every video between its first frame (initial state) and its
last frame (current state). Extracting those two frames was implemented three
times — once in `eval_interleave.py`, once in `trainer/grpo_trainer.py`, and
once in `eval_ablation_modality.py` — with the same cv2 logic and slightly
different fallbacks. This module is the single implementation.

Two placeholder protocols existed for deferring extraction until collation, and
both are still accepted by `resolve_frame_placeholders` so neither call site
changes behaviour:

- dict form, used by the eval harness:
  ``{"type": "image", "image": {"_placeholder_type": "initial_state",
                                "video_path": "..."}}``
- string-prefix form, used by the GRPO trainer:
  ``{"type": "image", "image": "__INIT_FRAME__:/path/to.mp4"}``

Prefer the dict form (`init_frame_placeholder` / `current_frame_placeholder`)
for new code: it cannot collide with a real path that happens to start with the
magic prefix.
"""

import os
from functools import lru_cache

import cv2
from PIL import Image


# String-prefix placeholder protocol (the GRPO trainer's form).
INIT_FRAME_PREFIX = "__INIT_FRAME__:"
CURRENT_FRAME_PREFIX = "__CURRENT_FRAME__:"

# Dict placeholder protocol (the eval harness's form).
PLACEHOLDER_KEY = "_placeholder_type"
INITIAL_STATE = "initial_state"
CURRENT_STATE = "current_state"

# Upper bound on frames sampled per video. NOTE: the published results were
# produced with this capped at 22, so `--nframes 32` in the launchers
# effectively sampled 22 frames. Kept as the default for reproducibility; raise
# INTERLEAVE_MAX_NFRAMES to lift it.
MAX_NFRAMES = int(os.environ.get("INTERLEAVE_MAX_NFRAMES", 22))


@lru_cache(maxsize=256)
def extract_frames_on_demand(video_path: str):
    """Extract a video's first and last frame, LRU-cached to avoid re-decoding.

    Args:
        video_path: path to the video file

    Returns:
        ``(init_img, current_img)`` as PIL.Image objects, or ``(None, None)`` on
        failure. When the last frame cannot be decoded the first frame is
        returned for both, which keeps the message shape valid.
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


def extract_first_and_last_frame(video_path: str, temp_dir: str):
    """Extract the two anchor frames and write them to `temp_dir` as JPEGs.

    The path-returning counterpart of `extract_frames_on_demand`, for call sites
    that hand the processor file paths rather than PIL objects.

    Returns:
        ``(init_path, current_path)``, or ``(None, None)`` if the video cannot be
        read. ``(init_path, None)`` is *not* returned: when only the first frame
        decodes, `extract_frames_on_demand` reuses it as the current state, and
        both paths are written.
    """
    init_img, current_img = extract_frames_on_demand(video_path)
    if init_img is None or current_img is None:
        return None, None

    base = os.path.basename(video_path)
    init_path = os.path.join(temp_dir, f"init_{base}.jpg")
    current_path = os.path.join(temp_dir, f"current_{base}.jpg")

    try:
        init_img.save(init_path)
        current_img.save(current_path)
    except Exception as e:
        print(f"Error saving frames for {video_path}: {e}")
        return None, None

    return init_path, current_path


def get_total_frames_cv2(video_path: str):
    """Return a local video's total frame count, or None if it is unreadable."""
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


def choose_nframes(requested: int, video_path: str, max_nframes=MAX_NFRAMES) -> int:
    """Clamp the requested frame count.

    Uses `total_frames` when the video is shorter than `requested`, then keeps
    the result within ``[2, max_nframes]``.

    Args:
        max_nframes: upper bound, or None for no upper bound. The interleaved
            harness caps at `MAX_NFRAMES`; the video-only baseline harness never
            did, and passes None to keep its published numbers reproducible.
    """
    req = int(requested)
    total = get_total_frames_cv2(video_path)
    if total is not None:
        req = min(req, total)
    if max_nframes is not None:
        req = min(int(max_nframes), req)
    return max(2, req)


def init_frame_placeholder(video_path: str) -> dict:
    """Build an initial-state image entry whose extraction is deferred."""
    return {"type": "image", "image": {PLACEHOLDER_KEY: INITIAL_STATE, "video_path": video_path}}


def current_frame_placeholder(video_path: str) -> dict:
    """Build a current-state image entry whose extraction is deferred."""
    return {"type": "image", "image": {PLACEHOLDER_KEY: CURRENT_STATE, "video_path": video_path}}


def _classify_placeholder(image_value):
    """Map an `image` value to ``(placeholder_type, video_path)``.

    Returns ``(None, None)`` for a plain image, which the caller passes through
    untouched. Accepts both the dict and the string-prefix protocol.
    """
    if isinstance(image_value, dict):
        return image_value.get(PLACEHOLDER_KEY), image_value.get("video_path")
    if isinstance(image_value, str):
        if image_value.startswith(INIT_FRAME_PREFIX):
            return INITIAL_STATE, image_value[len(INIT_FRAME_PREFIX) :]
        if image_value.startswith(CURRENT_FRAME_PREFIX):
            return CURRENT_STATE, image_value[len(CURRENT_FRAME_PREFIX) :]
    return None, None


def resolve_frame_placeholders(content_list: list) -> list:
    """Replace every frame placeholder in a content list with a PIL.Image.

    Entries that are not placeholders pass through unchanged. A placeholder
    whose extraction fails is dropped rather than left in place, since an
    unresolved dict would crash the processor further down.
    """
    resolved = []
    frame_cache = {}  # extract each video at most once per call

    for item in content_list:
        if not isinstance(item, dict) or item.get("type") != "image":
            resolved.append(item)
            continue

        placeholder_type, video_path = _classify_placeholder(item.get("image"))
        if not placeholder_type or not video_path:
            # A regular image entry, or a malformed placeholder: leave it alone
            resolved.append(item)
            continue

        if video_path not in frame_cache:
            frame_cache[video_path] = extract_frames_on_demand(video_path)
        init_img, current_img = frame_cache[video_path]

        if placeholder_type == INITIAL_STATE:
            if init_img is not None:
                resolved.append({"type": "image", "image": init_img})
            else:
                print(f"Warning: Failed to extract initial frame from {video_path}, skipping")
        elif placeholder_type == CURRENT_STATE:
            if current_img is not None:
                resolved.append({"type": "image", "image": current_img})
            else:
                print(f"Warning: Failed to extract current frame from {video_path}, skipping")
        else:
            resolved.append(item)

    return resolved


def resolve_placeholders_in_messages(messages: list) -> list:
    """Resolve every frame placeholder in a message list, in place.

    No deepcopy: callers that need to keep the original already copy it.
    """
    for message in messages:
        if "content" in message and isinstance(message["content"], list):
            message["content"] = resolve_frame_placeholders(message["content"])
    return messages


__all__ = [
    "INIT_FRAME_PREFIX",
    "CURRENT_FRAME_PREFIX",
    "PLACEHOLDER_KEY",
    "INITIAL_STATE",
    "CURRENT_STATE",
    "MAX_NFRAMES",
    "extract_frames_on_demand",
    "extract_first_and_last_frame",
    "get_total_frames_cv2",
    "choose_nframes",
    "init_frame_placeholder",
    "current_frame_placeholder",
    "resolve_frame_placeholders",
    "resolve_placeholders_in_messages",
]
