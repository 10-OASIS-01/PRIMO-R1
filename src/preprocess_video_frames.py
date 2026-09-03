#!/usr/bin/env python3
"""
Batch pre-extraction of first/last video frames for the whole PRIMO data tree.

What it does:
1. Covers all 23 PRIMO SFT/RL/Bench datasets.
2. Extracts the first and last frame of every video.
3. Writes the frames per data source under <VIDEO_DATA_ROOT>/primo-video/<dataset>/frames.
4. Updates the JSON in place, adding init_frame_path / current_frame_path to each record.
   Entries whose extraction fails are kept unchanged, so no data is dropped.
"""

import json
import os
import sys
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from tqdm import tqdm

# Import the shared DatasetLoader
CURRENT_DIR = Path(__file__).resolve().parent
LOADER_DIR = CURRENT_DIR / "r1-v" / "src" / "open_r1"
sys.path.insert(0, str(LOADER_DIR))

try:
    from DatasetLoader import dataset_loader
except ImportError:
    print("Unable to import DatasetLoader; check the path")
    sys.exit(1)


# ==================== Configuration ====================

DEFAULT_DATA_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DATA_BASE_DIR = os.environ.get("VIDEO_DATA_ROOT", DEFAULT_DATA_BASE_DIR)
VIDEO_ROOT = os.path.join(DATA_BASE_DIR, "primo-video")
FRAMES_BASE_DIR = os.path.join(DATA_BASE_DIR, "primo-video")
NUM_WORKERS = 32
JPEG_QUALITY = 95

# All 23 PRIMO datasets
DATASET_JSON_MAP: Dict[str, str] = {
    # SFT
    "primo-sft-agibot": "primo-sft/agibot/train_cot.json",
    "primo-sft-behavior-1k": "primo-sft/behavior-1k/train_cot.json",
    "primo-sft-robovqa": "primo-sft/robovqa/train_cot.json",
    "primo-sft-robotwin-clean": "primo-sft/robotwin-clean/train_cot.json",
    "primo-sft-robotwin-randomized": "primo-sft/robotwin-randomized/train_cot.json",
    "primo-sft-nextqa": "primo-sft/nextqa/train_cot.json",
    "primo-sft-perceptiontest": "primo-sft/perceptiontest/train_cot.json",
    "primo-sft-seed-bench-r1": "primo-sft/seed-bench-r1/train_cot.json",
    "primo-sft-star": "primo-sft/star/train_cot.json",
    "primo-sft-sharerobot": "primo-sft/sharerobot/train_cot.json",
    # RL
    "primo-rl-agibot": "primo-rl/agibot/train.json",
    "primo-rl-behavior-1k": "primo-rl/behavior-1k/train.json",
    "primo-rl-robovqa": "primo-rl/robovqa/train.json",
    "primo-rl-robotwin-clean": "primo-rl/robotwin-clean/train.json",
    "primo-rl-robotwin-randomized": "primo-rl/robotwin-randomized/train.json",
    "primo-rl-sharerobot": "primo-rl/sharerobot/train.json",
    # Bench
    "primo-bench-id-agibot": "primo-bench/agibot/id.json",
    "primo-bench-ood-agibot": "primo-bench/agibot/ood.json",
    "primo-bench-id-behavior-1k": "primo-bench/behavior-1k/id.json",
    "primo-bench-ood-behavior-1k": "primo-bench/behavior-1k/ood.json",
    "primo-bench-id-robotwin": "primo-bench/robotwin/id.json",
    "primo-bench-ood-robotwin": "primo-bench/robotwin/ood.json",
    "primo-bench-ood-real-humanoid": "primo-bench/real-humanoid/ood.json",
}

# Dataset name -> shared video-source directory name
DATASET_SOURCE_GROUP: Dict[str, str] = {
    # SFT
    "primo-sft-agibot": "agibot",
    "primo-sft-behavior-1k": "behavior-1k",
    "primo-sft-robovqa": "robovqa",
    "primo-sft-robotwin-clean": "robotwin",
    "primo-sft-robotwin-randomized": "robotwin",
    "primo-sft-nextqa": "nextqa",
    "primo-sft-perceptiontest": "perceptiontest",
    "primo-sft-seed-bench-r1": "seed-bench-r1",
    "primo-sft-star": "star",
    "primo-sft-sharerobot": "sharerobot",
    # RL
    "primo-rl-agibot": "agibot",
    "primo-rl-behavior-1k": "behavior-1k",
    "primo-rl-robovqa": "robovqa",
    "primo-rl-robotwin-clean": "robotwin",
    "primo-rl-robotwin-randomized": "robotwin",
    "primo-rl-sharerobot": "sharerobot",
    # Bench
    "primo-bench-id-agibot": "agibot",
    "primo-bench-ood-agibot": "agibot",
    "primo-bench-id-behavior-1k": "behavior-1k",
    "primo-bench-ood-behavior-1k": "behavior-1k",
    "primo-bench-id-robotwin": "robotwin",
    "primo-bench-ood-robotwin": "robotwin",
    "primo-bench-ood-real-humanoid": "real-humanoid",
}

ALL_DATASETS: List[str] = list(DATASET_JSON_MAP.keys())


def _select_datasets_from_args() -> List[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--only-datasets",
        type=str,
        default="",
        help="Comma-separated dataset names; process only these",
    )
    parser.add_argument(
        "--only-behavior",
        action="store_true",
        help="Process only the datasets containing behavior-1k",
    )
    args, _ = parser.parse_known_args()

    env_only = os.environ.get("PRIMO_ONLY_DATASETS", "").strip()
    cli_only = args.only_datasets.strip()

    if args.only_behavior:
        selected = [d for d in ALL_DATASETS if "behavior-1k" in d]
        print(f"Processing the behavior subset only: {selected}")
        return selected

    raw = cli_only or env_only
    if not raw:
        return list(ALL_DATASETS)

    requested = [x.strip() for x in raw.split(",") if x.strip()]
    valid = [x for x in requested if x in ALL_DATASETS]
    invalid = [x for x in requested if x not in ALL_DATASETS]

    if invalid:
        print(f"Ignoring unknown datasets: {invalid}")

    if not valid:
        print("No valid dataset selected; check --only-datasets or PRIMO_ONLY_DATASETS")
        sys.exit(1)

    print(f"Processing only the requested datasets: {valid}")
    return valid


# ==================== Helpers ====================

def _resolve_json_list_container(data: Any) -> Tuple[str, Optional[str], List[Any]]:
    """
    Return (container_type, key, list_obj).

    container_type: list | dict | unknown
    key: meaningful only when container_type is dict
    """
    if isinstance(data, list):
        return "list", None, data

    if isinstance(data, dict):
        for key in ["results", "data", "annotations", "train", "items"]:
            if key in data and isinstance(data[key], list):
                return "dict", key, data[key]

        for key, value in data.items():
            if isinstance(value, list):
                return "dict", key, value

    return "unknown", None, []


def _extract_frames(
    video_abs_path: str,
    video_rel_to_root: str,
    source_group: str,
) -> Optional[Tuple[str, str]]:
    """
    Extract the first and last frame.

    Returns paths relative to DATA_BASE_DIR, ready to be written into the JSON:
        ./primo-video/<dataset>/frames/<video_rel_without_dataset_and_ext>_init.jpg
        ./primo-video/<dataset>/frames/<video_rel_without_dataset_and_ext>_current.jpg
    """
    if not os.path.exists(video_abs_path):
        return None

    try:
        rel_norm = video_rel_to_root.replace("\\", "/")
        rel_no_ext = os.path.splitext(rel_norm)[0]
        parts = rel_no_ext.split("/") if rel_no_ext else []
        if not parts:
            return None

        # Datasets that share a video source (e.g. agibot across sft/rl/bench) all land
        # in the same directory, even though their dataset_name differs.
        if parts[0] == source_group:
            rel_inside_dataset = "/".join(parts[1:])
        else:
            rel_inside_dataset = rel_no_ext

        if not rel_inside_dataset:
            return None

        init_abs = os.path.join(FRAMES_BASE_DIR, source_group, "frames", f"{rel_inside_dataset}_init.jpg")
        curr_abs = os.path.join(FRAMES_BASE_DIR, source_group, "frames", f"{rel_inside_dataset}_current.jpg")

        os.makedirs(os.path.dirname(init_abs), exist_ok=True)

        if not (os.path.exists(init_abs) and os.path.exists(curr_abs)):
            cap = cv2.VideoCapture(video_abs_path)
            if not cap.isOpened():
                return None

            ok, first_frame = cap.read()
            if not ok or first_frame is None:
                cap.release()
                return None

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
            ok, last_frame = cap.read()
            if not ok or last_frame is None:
                last_frame = first_frame

            cap.release()

            cv2.imwrite(init_abs, first_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            cv2.imwrite(curr_abs, last_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        init_rel = "./" + os.path.relpath(init_abs, DATA_BASE_DIR).replace("\\", "/")
        curr_rel = "./" + os.path.relpath(curr_abs, DATA_BASE_DIR).replace("\\", "/")
        return init_rel, curr_rel
    except Exception:
        return None


def _process_one(task: Tuple[int, Dict[str, Any], str, str]) -> Tuple[int, Dict[str, Any]]:
    """
    Process a single record in a worker. Always returns (index, updated_entry),
    falling back to the original entry on failure.
    """
    idx, entry, video_abs, source_group = task

    try:
        rel = os.path.relpath(video_abs, VIDEO_ROOT).replace("\\", "/")
    except Exception:
        return idx, entry

    result = _extract_frames(video_abs, rel, source_group)
    if result is None:
        return idx, entry

    init_rel, curr_rel = result
    updated = dict(entry)
    updated["init_frame_path"] = init_rel
    updated["current_frame_path"] = curr_rel
    return idx, updated


def _json_path_for_dataset(dataset_name: str) -> str:
    return os.path.join(DATA_BASE_DIR, DATASET_JSON_MAP[dataset_name])


def _process_dataset(dataset_name: str) -> Tuple[int, int]:
    """
    Return (total_count, success_count).
    """
    json_path = _json_path_for_dataset(dataset_name)
    if not os.path.exists(json_path):
        print(f"JSON does not exist, skipping: {json_path}")
        return 0, 0

    try:
        # Use the shared loader to get records plus absolute video paths
        entries, video_paths = dataset_loader(dataset_name, base_dir=DATA_BASE_DIR)
    except Exception as e:
        print(f"Failed to load: {dataset_name} -> {e}")
        return 0, 0

    if not entries or len(entries) != len(video_paths):
        print(f"No entries, or count mismatch: {dataset_name} ({len(entries)} vs {len(video_paths)})")
        return len(entries), 0

    print(f"\nProcessing {dataset_name}: {len(entries)} records")

    source_group = DATASET_SOURCE_GROUP.get(dataset_name)
    if not source_group:
        print(f"No source_group configured, skipping: {dataset_name}")
        return len(entries), 0

    tasks = [(i, entries[i], video_paths[i], source_group) for i in range(len(entries))]
    updated_entries: List[Dict[str, Any]] = [dict(e) for e in entries]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for idx, updated in tqdm(
            executor.map(_process_one, tasks),
            total=len(tasks),
            desc=f"{dataset_name}",
            ncols=100,
        ):
            updated_entries[idx] = updated

    success_count = 0
    for old, new in zip(entries, updated_entries):
        if "init_frame_path" in new and "current_frame_path" in new:
            # Count only results produced now, regardless of any pre-existing value
            if (
                new.get("init_frame_path") != old.get("init_frame_path")
                or new.get("current_frame_path") != old.get("current_frame_path")
                or ("init_frame_path" not in old)
                or ("current_frame_path" not in old)
            ):
                success_count += 1

    # Write the JSON back, preserving its original container shape
    try:
        backup_path = json_path + ".backup"
        if not os.path.exists(backup_path):
            import shutil

            shutil.copy2(json_path, backup_path)

        with open(json_path, "r", encoding="utf-8") as f:
            original = json.load(f)

        container_type, key, _ = _resolve_json_list_container(original)
        if container_type == "list":
            to_write = updated_entries
        elif container_type == "dict" and key is not None:
            original[key] = updated_entries
            to_write = original
        else:
            # Fallback
            to_write = updated_entries

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(to_write, f, ensure_ascii=False, indent=2)

        print(f"Write-back complete: {json_path}")
    except Exception as e:
        print(f"Write-back failed: {json_path} -> {e}")
        traceback.print_exc()

    return len(entries), success_count


def main() -> None:
    datasets = _select_datasets_from_args()

    print("\n" + "=" * 80)
    print("PRIMO first/last frame batch extraction")
    print("=" * 80)
    print(f"DATA_BASE_DIR : {DATA_BASE_DIR}")
    print(f"VIDEO_ROOT    : {VIDEO_ROOT}")
    print(f"FRAMES_ROOT   : {FRAMES_BASE_DIR}")
    print(f"NUM_WORKERS  : {NUM_WORKERS}")
    print(f"DATASETS      : {len(datasets)}")
    print("=" * 80)

    os.makedirs(FRAMES_BASE_DIR, exist_ok=True)

    total_entries = 0
    total_success = 0

    for idx, ds in enumerate(datasets, start=1):
        print(f"\n[{idx}/{len(datasets)}] {ds}")
        n_all, n_ok = _process_dataset(ds)
        total_entries += n_all
        total_success += n_ok

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total records: {total_entries}")
    print(f"Records with added/updated frame fields: {total_success}")
    print(f"Frame output root: {FRAMES_BASE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
