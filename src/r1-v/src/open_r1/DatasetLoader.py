import os
import json
from typing import Optional


DEFAULT_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data"))
PRIMO_VIDEO_DIR = "primo-video"


def _resolve_base_dir(base_dir: Optional[str]) -> str:
    if base_dir:
        return base_dir
    return os.environ.get("VIDEO_DATA_ROOT") or DEFAULT_BASE_DIR


def dataset_loader(
    dataset_name: str,
    base_dir: Optional[str] = None,
    data_file: Optional[str] = None,
):
    """
    Load a PRIMO dataset, returning each sample alongside its absolute video path.

    Args:
        dataset_name (str): dataset key, e.g. "primo-sft-agibot",
            "primo-rl-robotwin-clean", "primo-bench-ood-real-humanoid".
        base_dir (str | None): dataset root; when empty, VIDEO_DATA_ROOT is used.
        data_file (str | None): optional explicit JSON path, for a custom location.

    Returns:
        json_list: List[dict], the JSON record of each entity.
        video_paths: List[str], the absolute video path of each entity.
    """
    if data_file is None and base_dir and os.path.isfile(base_dir):
        data_file, base_dir = base_dir, None

    base_dir = _resolve_base_dir(base_dir)
    ds_in = dataset_name.strip()
    ds_lower = ds_in.lower()
    is_cot = ds_lower.endswith("-cot") or ds_lower.endswith("_cot")
    clean_name = ds_lower.replace("-cot", "").replace("_cot", "")

    cfg = {
        # PRIMO SFT
        "primo-sft-agibot": {
            "json_rel": "primo-sft/agibot/train_cot.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
            "force_cot": True,
        },
        "primo-sft-behavior-1k": {
            "json_rel": "primo-sft/behavior-1k/train_cot.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
            "force_cot": True,
        },
        "primo-sft-robovqa": {
            "json_rel": "primo-sft/robovqa/train_cot.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
            "force_cot": True,
        },
        "primo-sft-robotwin-clean": {
            "json_rel": "primo-sft/robotwin-clean/train_cot.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
            "force_cot": True,
        },
        "primo-sft-robotwin-randomized": {
            "json_rel": "primo-sft/robotwin-randomized/train_cot.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
            "force_cot": True,
        },
        "primo-sft-nextqa": {
            "json_rel": "primo-sft/nextqa/train_cot.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
            "force_cot": True,
        },
        "primo-sft-perceptiontest": {
            "json_rel": "primo-sft/perceptiontest/train_cot.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
            "force_cot": True,
        },
        "primo-sft-seed-bench-r1": {
            "json_rel": "primo-sft/seed-bench-r1/train_cot.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
            "force_cot": True,
        },
        "primo-sft-star": {
            "json_rel": "primo-sft/star/train_cot.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
            "force_cot": True,
        },
        "primo-sft-sharerobot": {
            "json_rel": "primo-sft/sharerobot/train_cot.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
            "force_cot": True,
        },
        # PRIMO RL
        "primo-rl-agibot": {
            "json_rel": "primo-rl/agibot/train.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
        },
        "primo-rl-behavior-1k": {
            "json_rel": "primo-rl/behavior-1k/train.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
        },
        "primo-rl-robovqa": {
            "json_rel": "primo-rl/robovqa/train.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
        },
        "primo-rl-robotwin-clean": {
            "json_rel": "primo-rl/robotwin-clean/train.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
        },
        "primo-rl-robotwin-randomized": {
            "json_rel": "primo-rl/robotwin-randomized/train.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
        },
        "primo-rl-sharerobot": {
            "json_rel": "primo-rl/sharerobot/train.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
        },
        # PRIMO Bench
        "primo-bench-id-agibot": {
            "json_rel": "primo-bench/agibot/id.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
        },
        "primo-bench-ood-agibot": {
            "json_rel": "primo-bench/agibot/ood.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
        },
        "primo-bench-id-behavior-1k": {
            "json_rel": "primo-bench/behavior-1k/id.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
        },
        "primo-bench-ood-behavior-1k": {
            "json_rel": "primo-bench/behavior-1k/ood.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
        },
        "primo-bench-id-robotwin": {
            "json_rel": "primo-bench/robotwin/id.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
        },
        "primo-bench-ood-robotwin": {
            "json_rel": "primo-bench/robotwin/ood.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
        },
        "primo-bench-ood-real-humanoid": {
            "json_rel": "primo-bench/real-humanoid/ood.json",
            "video_root_rel": PRIMO_VIDEO_DIR,
        },
    }

    seed_bench_variants = {"seed-bench-r1"}

    if clean_name not in cfg:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    meta = cfg[clean_name]
    json_file = data_file or os.path.join(base_dir, meta["json_rel"])

    if meta.get("video_root_rel"):
        video_root_path = os.path.join(base_dir, meta["video_root_rel"])
    else:
        video_root_path = base_dir

    use_cot_filter = bool(meta.get("force_cot")) or is_cot

    if not os.path.exists(json_file):
        print(f"[warning] JSON file does not exist: {json_file}")
        return [], []

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception as e:
        print(f"[error] Failed to parse JSON: {json_file} -> {e}")
        return [], []

    # Accept several JSON layouts
    if isinstance(loaded, list):
        data_list = loaded
    elif isinstance(loaded, dict):
        for key in ["results", "data", "annotations", "train", "items"]:
            if key in loaded and isinstance(loaded[key], list):
                data_list = loaded[key]
                break
        else:
            # If the dict holds exactly one list value, fall back to that
            lists = [v for v in loaded.values() if isinstance(v, list)]
            data_list = lists[0] if lists else []
    else:
        data_list = []

    print(f"[info] Loaded dataset: {ds_in} | JSON: {json_file} | entries: {len(data_list)}")

    # For CoT data, keep only the samples flagged by the `select` field
    if use_cot_filter:
        before = len(data_list)
        data_list = [e for e in data_list if isinstance(e, dict) and e.get("select") is True]
        after = len(data_list)
        if after != before:
            print(f"[info] CoT filter: select=True kept {after}/{before}")

        required_fields = ("process", "planning", "observation", "reasoning")
        before_fields = len(data_list)
        filtered = []
        for entry in data_list:
            if not all(entry.get(field) for field in required_fields):
                continue
            filtered.append(entry)
        skipped = before_fields - len(filtered)
        data_list = filtered
        if skipped:
            missing = ",".join(required_fields)
            print(f"[info] CoT filter: skipped {skipped} entries missing required fields ({missing})")

    json_list = []
    video_paths = []
    for entry in data_list:
        video_rel_path = entry.get("path") or entry.get("video") or entry.get("video_path")
        if not video_rel_path:
            continue
        video_rel_path = str(video_rel_path).replace("\\", "/")
        if video_rel_path.startswith("./"):
            video_rel_path = video_rel_path[2:]

        # Rule 1 (seed-bench-r1): its JSON stores pseudo-absolute paths starting with
        # "/videos" or "/images" that are really relative to the dataset root, so
        # strip the leading slash when that pattern shows up.
        if video_rel_path.startswith("/videos/") or video_rel_path.startswith("/images/"):
            video_rel_path = video_rel_path.lstrip("/")

        # Rule 2: seed-bench-r1 video paths omit the "merged_videos" directory, so insert it.
        # e.g. videos/EpicKitchens/P01_01-xxx.mp4 -> videos/EpicKitchens/merged_videos/P01_01-xxx.mp4
        if any(name in clean_name for name in seed_bench_variants) and video_rel_path.startswith("videos/EpicKitchens/") and "/merged_videos/" not in video_rel_path:
            parts = video_rel_path.split("/")
            # Expected layout: [videos, EpicKitchens, <filename or subdirs...>]
            # Insert merged_videos as the third path component.
            if len(parts) >= 3:
                # Insert right after EpicKitchens even when the third part is itself a directory.
                video_rel_path = "/".join([parts[0], parts[1], "merged_videos"] + parts[2:])

        # Everything is now a plain relative path (no pseudo-absolute paths left), so join it.
        video_abs_path = os.path.abspath(os.path.join(video_root_path, video_rel_path.lstrip("/")))

        json_list.append(entry)
        video_paths.append(video_abs_path)

    return json_list, video_paths


if __name__ == "__main__":
    # Minimal self-test
    for name in [
        "primo-bench-ood-real-humanoid",
    ]:
        jl, vp = dataset_loader(name)
        print(f"{name}: entries={len(jl)}, videos={len(vp)}")