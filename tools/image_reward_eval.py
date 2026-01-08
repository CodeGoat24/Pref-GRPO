#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tqdm import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastvideo.rewards.dispatcher import RewardDispatcher, parse_reward_spec


DEFAULT_REWARD_SPEC = ",".join(
    [
        "clip",
        "hpsv2",
        "hpsv3",
        "pickscore",
        "unifiedreward_alignment",
        "unifiedreward_style",
        "unifiedreward_coherence",
    ]
)


def normalize_index(value: str) -> str:
    text = str(value).strip()
    if not text:
        return text
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def load_caption_map(csv_path: str, index_col: str, caption_col: str) -> dict[str, str]:
    caption_map: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if index_col not in reader.fieldnames or caption_col not in reader.fieldnames:
            raise ValueError(
                f"CSV must include columns {index_col!r} and {caption_col!r}."
            )
        for row in reader:
            idx = normalize_index(row.get(index_col, ""))
            caption = (row.get(caption_col, "") or "").strip()
            if not idx or not caption:
                continue
            caption_map[idx] = caption
    if not caption_map:
        raise ValueError(f"No captions found in CSV: {csv_path}")
    return caption_map


def list_images(image_dir: str, recursive: bool, exts: set[str]) -> list[str]:
    images: list[str] = []
    if recursive:
        for root, _, files in os.walk(image_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in exts:
                    images.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(image_dir):
            path = os.path.join(image_dir, filename)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in exts:
                images.append(path)
    return sorted(images)


def parse_image_index(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    prefix = stem.split("_", 1)[0]
    return normalize_index(prefix)


def build_reward_inputs(dispatcher: RewardDispatcher, items: list[dict]) -> dict[str, list[dict]]:
    reward_inputs = dispatcher.build_reward_inputs()
    for entry in items:
        for key in reward_inputs:
            reward_inputs[key].append(entry)
    return reward_inputs


def save_results_json(
    output_path: str,
    image_paths: list[str],
    captions: list[str],
    reward_tensors: dict[str, torch.Tensor],
    reward_means: dict[str, float],
) -> None:
    records = []
    for i, (path, caption) in enumerate(zip(image_paths, captions)):
        scores = {name: float(tensor[i].item()) for name, tensor in reward_tensors.items()}
        records.append({"path": path, "caption": caption, "scores": scores})
    payload = {"mean_scores": reward_means, "records": records}
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def get_dist_info() -> tuple[bool, int, int, int | None]:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = os.getenv("LOCAL_RANK")
    local_rank_val = int(local_rank) if local_rank is not None else None
    return world_size > 1, rank, world_size, local_rank_val


def setup_distributed(device: str) -> None:
    if not dist.is_available() or dist.is_initialized():
        return
    backend = "nccl" if "cuda" in device else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")


def split_items(items: list[dict], rank: int, world_size: int) -> list[dict]:
    if world_size <= 1:
        return items
    return items[rank::world_size]


def compute_rewards_in_chunks(
    dispatcher: RewardDispatcher,
    items: list[dict],
    chunk_size: int,
    *,
    show_progress: bool,
) -> dict[str, torch.Tensor]:
    reward_tensors_by_name: dict[str, list[torch.Tensor]] = {}
    total = len(items)
    for start in tqdm(
        range(0, total, chunk_size),
        desc="Scoring images",
        unit="batch",
        disable=not show_progress,
    ):
        chunk = items[start : start + chunk_size]
        reward_inputs = build_reward_inputs(dispatcher, chunk)
        with torch.inference_mode():
            reward_tensors, _ = dispatcher.compute_rewards(reward_inputs)
        for name, tensor in reward_tensors.items():
            reward_tensors_by_name.setdefault(name, []).append(tensor)
    merged = {
        name: torch.cat(tensors, dim=0)
        for name, tensors in reward_tensors_by_name.items()
    }
    return merged


def reduce_reward_means(
    reward_tensors: dict[str, torch.Tensor],
    reward_weights: dict[str, float],
    *,
    device: str,
    is_distributed: bool,
) -> dict[str, float]:
    reward_means: dict[str, float] = {}
    for name in reward_weights:
        tensor = reward_tensors.get(name)
        if tensor is None:
            local_sum = torch.tensor(0.0, device=device)
            local_count = torch.tensor(0.0, device=device)
        else:
            local_sum = tensor.sum()
            local_count = torch.tensor(float(tensor.numel()), device=device)
        if is_distributed:
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        count = float(local_count.item())
        reward_means[name] = float(local_sum.item()) / count if count > 0 else 0.0
    return reward_means


def merge_rank_outputs(output_path: str, world_size: int) -> None:
    all_records = []
    mean_scores: dict[str, float] = {}
    for rank in range(world_size):
        shard_path = f"{output_path}.rank{rank}"
        if not os.path.exists(shard_path):
            continue
        with open(shard_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not mean_scores and "mean_scores" in payload:
            mean_scores = payload["mean_scores"]
        all_records.extend(payload.get("records", []))
    merged = {"mean_scores": mean_scores, "records": all_records}
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=True, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate image rewards with CLIP/HPS/PickScore/UnifiedReward."
    )
    parser.add_argument("--image_dir", required=True, help="Folder with generated images.")
    parser.add_argument("--prompt_csv", required=True, help="CSV with prompts.")
    parser.add_argument(
        "--reward_spec",
        default=DEFAULT_REWARD_SPEC,
        help="Reward list (comma or JSON). Default uses all supported image rewards.",
    )
    parser.add_argument(
        "--caption_col",
        default="prompt_en",
        help="CSV column containing captions (default: prompt_en).",
    )
    parser.add_argument(
        "--index_col",
        default="index",
        help="CSV column containing prompt indices (default: index).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan image_dir for images.",
    )
    parser.add_argument(
        "--image_exts",
        default=".png,.jpg,.jpeg,.webp",
        help="Comma-separated image extensions to include.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for reward models.",
    )
    parser.add_argument(
        "--api_url",
        default=os.getenv("REWARD_API_URL", None),
        help="API URL for UnifiedReward (if needed).",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional JSON output path with per-image scores.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=32,
        help="Images per batch for scoring (enables progress bar).",
    )
    args = parser.parse_args()

    reward_weights = parse_reward_spec(args.reward_spec)
    if not reward_weights:
        raise ValueError("reward_spec is empty; specify at least one reward.")

    caption_map = load_caption_map(args.prompt_csv, args.index_col, args.caption_col)
    exts = {ext.strip().lower() for ext in args.image_exts.split(",") if ext.strip()}
    image_paths = list_images(args.image_dir, args.recursive, exts)
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")

    selected_paths: list[str] = []
    selected_captions: list[str] = []
    missing = 0
    for path in image_paths:
        idx = parse_image_index(path)
        caption = caption_map.get(idx)
        if caption is None:
            missing += 1
            continue
        selected_paths.append(path)
        selected_captions.append(caption)

    if not selected_paths:
        raise ValueError("No images matched captions; check filename indices.")
    if missing:
        print(f"[warn] Skipped {missing} images without matching caption indices.")

    is_distributed, rank, world_size, local_rank = get_dist_info()
    device = args.device
    if device.startswith("cuda") and local_rank is not None:
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    if is_distributed:
        setup_distributed(device)

    dispatcher_args = SimpleNamespace(api_url=args.api_url)
    dispatcher = RewardDispatcher(
        args=dispatcher_args,
        device=device,
        reward_weights=reward_weights,
        modality="image",
    )

    items = [
        {"path": path, "prompt": caption}
        for path, caption in zip(selected_paths, selected_captions)
    ]
    local_items = split_items(items, rank, world_size)
    if args.chunk_size and args.chunk_size > 0:
        reward_tensors = compute_rewards_in_chunks(
            dispatcher,
            local_items,
            args.chunk_size,
            show_progress=rank == 0,
        )
    else:
        if local_items:
            reward_inputs = build_reward_inputs(dispatcher, local_items)
            with torch.inference_mode():
                reward_tensors, _ = dispatcher.compute_rewards(reward_inputs)
        else:
            reward_tensors = {}

    reward_means = reduce_reward_means(
        reward_tensors,
        reward_weights,
        device=device,
        is_distributed=is_distributed,
    )
    if rank == 0:
        print("Mean reward scores:")
        for name, value in sorted(reward_means.items()):
            print(f"  {name}: {value:.6f}")

    if args.output_json:
        local_paths = [item["path"] for item in local_items]
        local_captions = [item["prompt"] for item in local_items]
        shard_path = (
            f"{args.output_json}.rank{rank}" if is_distributed else args.output_json
        )
        save_results_json(
            shard_path,
            local_paths,
            local_captions,
            reward_tensors,
            reward_means,
        )
        if is_distributed:
            dist.barrier()
            if rank == 0:
                merge_rank_outputs(args.output_json, world_size)


if __name__ == "__main__":
    main()
