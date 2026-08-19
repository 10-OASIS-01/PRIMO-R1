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
    PRIMO 数据集加载：返回对应视频的绝对路径。

    Args:
        dataset_name (str): 数据集名，例如 "primo-sft-agibot", "primo-rl-robotwin-clean", "primo-bench-ood-real-humanoid"。
        base_dir (str | None): 数据集根目录；为空时优先使用 VIDEO_DATA_ROOT 环境变量。
        data_file (str | None): 可选的 JSON 文件路径（用于特定数据集或自定义路径）。

    Returns:
        json_list: List[dict] 每个 entity 的 JSON 数据。
        video_paths: List[str] 每个 entity 对应的视频绝对路径。
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
        raise ValueError(f"未知数据集名称: {dataset_name}")

    meta = cfg[clean_name]
    json_file = data_file or os.path.join(base_dir, meta["json_rel"])

    if meta.get("video_root_rel"):
        video_root_path = os.path.join(base_dir, meta["video_root_rel"])
    else:
        video_root_path = base_dir

    use_cot_filter = bool(meta.get("force_cot")) or is_cot

    if not os.path.exists(json_file):
        print(f"[警告] JSON 文件不存在: {json_file}")
        return [], []

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception as e:
        print(f"[错误] 解析 JSON 失败: {json_file} -> {e}")
        return [], []

    # 兼容多种 JSON 结构
    if isinstance(loaded, list):
        data_list = loaded
    elif isinstance(loaded, dict):
        for key in ["results", "data", "annotations", "train", "items"]:
            if key in loaded and isinstance(loaded[key], list):
                data_list = loaded[key]
                break
        else:
            # 若 dict 仅包含一个 list 值，也尝试取出
            lists = [v for v in loaded.values() if isinstance(v, list)]
            data_list = lists[0] if lists else []
    else:
        data_list = []

    print(f"[信息] 加载数据集: {ds_in} | JSON: {json_file} | 条目数: {len(data_list)}")

    # 若为 COT 数据，则按 select 字段筛选
    if use_cot_filter:
        before = len(data_list)
        data_list = [e for e in data_list if isinstance(e, dict) and e.get("select") is True]
        after = len(data_list)
        if after != before:
            print(f"[信息] COT 过滤：select=True 保留 {after}/{before}")

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
            print(f"[信息] COT 过滤：缺少必需字段({','.join(required_fields)}) 跳过 {skipped} 条")

    json_list = []
    video_paths = []
    for entry in data_list:
        video_rel_path = entry.get("path") or entry.get("video") or entry.get("video_path")
        if not video_rel_path:
            continue
        video_rel_path = str(video_rel_path).replace("\\", "/")
        if video_rel_path.startswith("./"):
            video_rel_path = video_rel_path[2:]

        # 规则一：seed-bench-r1）JSON 中使用以 "/videos" 或 "/images" 开头的伪绝对路径，
        # 实际应视为数据集根下的相对路径。若检测到该模式，则去掉前导斜杠。
        if video_rel_path.startswith("/videos/") or video_rel_path.startswith("/images/"):
            video_rel_path = video_rel_path.lstrip("/")

        # 规则二：若为 seed-bench-r1 数据集的视频路径缺少 "merged_videos" 目录，则插入。
        # 示例原始可能为: videos/EpicKitchens/P01_01-xxx.mp4 -> 需改为 videos/EpicKitchens/merged_videos/P01_01-xxx.mp4
        if any(name in clean_name for name in seed_bench_variants) and video_rel_path.startswith("videos/EpicKitchens/") and "/merged_videos/" not in video_rel_path:
            parts = video_rel_path.split("/")
            # 结构期望: [videos, EpicKitchens, <filename or subdirs...>]
            # 插入 merged_videos 作为第三层目录
            if len(parts) >= 3:
                # 若第三个部分已经是目录而不是文件，也仍统一插入在 EpicKitchens 后面
                video_rel_path = "/".join([parts[0], parts[1], "merged_videos"] + parts[2:])

        # 修正后统一视为相对路径（不再保留上述伪绝对路径），进行拼接
        video_abs_path = os.path.abspath(os.path.join(video_root_path, video_rel_path.lstrip("/")))

        json_list.append(entry)
        video_paths.append(video_abs_path)

    return json_list, video_paths


if __name__ == "__main__":
    # 简单自测
    for name in [
        "primo-bench-ood-real-humanoid",
    ]:
        jl, vp = dataset_loader(name)
        print(f"{name}: entries={len(jl)}, videos={len(vp)}")