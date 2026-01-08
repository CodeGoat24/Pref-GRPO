import base64
import itertools
import json
import os
import re
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from vllm_utils.vllm_request import evaluate_batch


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_float_list(name: str) -> Optional[List[float]]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    items = [item.strip() for item in value.split(",")]
    weights = []
    for item in items:
        if not item:
            continue
        try:
            weights.append(float(item))
        except ValueError:
            continue
    return weights or None


def _encode_image(image: Image.Image) -> str:
    image = image.convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _extract_frames(video_path: str, *, num_frames: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return []

    num_frames = min(num_frames, frame_count)
    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()
    return frames


def _parse_json_payload(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _normalize_winner(text: str) -> Optional[int]:
    if not text:
        return None
    lowered = text.lower()
    first = re.search(r"\b(?:video|image)\s*1\b", lowered)
    second = re.search(r"\b(?:video|image)\s*2\b", lowered)
    if first and not second:
        return 1
    if second and not first:
        return 2
    return None


def _iter_category_winners(data: dict) -> List[Tuple[str, Optional[int]]]:
    cat_winners = []
    categories = data.get("categories", [])
    for cat in categories[:3]:
        cat_name = str(cat.get("name", "")).strip()
        winner = _normalize_winner(str(cat.get("cat_winner", "")))
        key = cat_name or "category"
        cat_winners.append((key, winner))
    return cat_winners


def _pairwise_win_rate(
    *,
    num_items: int,
    responses: List[dict],
    overall_weight: float,
    dim_weight: float,
    category_weights: Optional[List[float]],
    device: str | torch.device = "cuda",
) -> Tuple[List[torch.Tensor], Dict[str, List[float]]]:
    win_count = defaultdict(lambda: defaultdict(float))
    compare_count = defaultdict(lambda: defaultdict(int))
    dim_key_order: List[str] = []

    
    for result in responses:
        idx1 = result["first_index"]
        idx2 = result["second_index"]

        compare_count["overall"][idx1] += 1
        compare_count["overall"][idx2] += 1

        parsed = _parse_json_payload(result.get("model_output", ""))
        
        overall_winner = _normalize_winner(
            str(parsed.get("winner", "")) if parsed else ""
        )
        if overall_winner == 1:
            win_count["overall"][idx1] += 1
        elif overall_winner == 2:
            win_count["overall"][idx2] += 1
        else:
            win_count["overall"][idx1] += 0.5
            win_count["overall"][idx2] += 0.5

        if not parsed:
            continue

        for dim_key, winner in _iter_category_winners(parsed):
            if dim_key not in dim_key_order:
                dim_key_order.append(dim_key)
            compare_count[dim_key][idx1] += 1
            compare_count[dim_key][idx2] += 1
            if winner == 1:
                win_count[dim_key][idx1] += 1
            elif winner == 2:
                win_count[dim_key][idx2] += 1
            else:
                win_count[dim_key][idx1] += 0.5
                win_count[dim_key][idx2] += 0.5

    overall_win_rate = [
        round(
            win_count["overall"][idx]
            / max(compare_count["overall"][idx], 1),
            3,
        )
        for idx in range(num_items)
    ]

    dim_keys = [k for k in dim_key_order if k in compare_count and k != "overall"]
    dim_reward: Dict[str, List[float]] = {"overall_reward": overall_win_rate}

    dim_mean_rates = []
    dim_rate_flags = []
    if category_weights and len(category_weights) >= len(dim_keys):
        weights = category_weights[: len(dim_keys)]
    elif category_weights:
        weights = category_weights + [1.0] * (len(dim_keys) - len(category_weights))
    else:
        weights = [1.0] * len(dim_keys)
    weights = [float(w) for w in weights]
    for idx in range(num_items):
        per_dim_rates = []
        per_dim_weights = []
        for dim_key, dim_weight in zip(dim_keys, weights):
            if compare_count[dim_key][idx] > 0:
                per_dim_rates.append(
                    win_count[dim_key][idx] / compare_count[dim_key][idx]
                )
                per_dim_weights.append(dim_weight)
        dim_rate_flags.append(bool(per_dim_rates))
        if per_dim_rates:
            if per_dim_weights and sum(per_dim_weights) > 0:
                weighted = sum(
                    rate * w for rate, w in zip(per_dim_rates, per_dim_weights)
                ) / sum(per_dim_weights)
                dim_mean_rates.append(round(float(weighted), 3))
            else:
                dim_mean_rates.append(round(float(np.mean(per_dim_rates)), 3))
        else:
            dim_mean_rates.append(0.0)

    dim_reward["dim_mean_reward"] = dim_mean_rates

    for dim_key in dim_keys:
        dim_reward[dim_key] = [
            round(
                win_count[dim_key][idx] / max(compare_count[dim_key][idx], 1),
                3,
            )
            for idx in range(num_items)
        ]

    rewards = []
    for idx in range(num_items):
        overall = overall_win_rate[idx]
        dim_mean = dim_mean_rates[idx]
        if dim_weight > 0 and dim_rate_flags[idx]:
            reward = (overall_weight * overall + dim_weight * dim_mean) / (
                overall_weight + dim_weight
            )
        else:
            reward = overall
        rewards.append(torch.tensor(reward, device=device).unsqueeze(0))

    return rewards, dim_reward


def cal_win_rate_videos(
    all_input_data: List[dict],
    *,
    api_url: str,
    num_frames: int = 8,
    overall_weight: float = 1.0,
    dim_weight: float = 1.0,
    category_weights: Optional[List[float]] = None,
    device: str | torch.device = "cuda",
) -> Tuple[List[torch.Tensor], Dict[str, List[float]]]:
    overall_weight = _get_env_float("OVERALL_WEIGHT", overall_weight)
    dim_weight = _get_env_float("DIM_WEIGHT", dim_weight)
    if category_weights is None:
        category_weights = _get_env_float_list("CATEGORY_WEIGHTS")
    videos = [
        _extract_frames(data["videos"], num_frames=num_frames) for data in all_input_data
    ]

    pairs = list(itertools.combinations(enumerate(videos), 2))
    problem = all_input_data[0]["problem"]
    payload = [
        {
            "images": [_encode_image(frame) for frame in frames1 + frames2],
            "problem": problem,
            "first_index": idx1,
            "second_index": idx2,
        }
        for (idx1, frames1), (idx2, frames2) in pairs
    ]

    responses = evaluate_batch(payload, api_url=api_url)
    return _pairwise_win_rate(
        num_items=len(videos),
        responses=responses,
        overall_weight=overall_weight,
        dim_weight=dim_weight,
        category_weights=category_weights,
        device=device,
    )


def cal_win_rate_images(
    all_input_data: List[dict],
    *,
    api_url: str,
    overall_weight: float = 1.0,
    dim_weight: float = 1.0,
    category_weights: Optional[List[float]] = None,
    device: str | torch.device = "cuda",
) -> Tuple[List[torch.Tensor], Dict[str, List[float]]]:
    overall_weight = _get_env_float("OVERALL_WEIGHT", overall_weight)
    dim_weight = _get_env_float("DIM_WEIGHT", dim_weight)
    if category_weights is None:
        category_weights = _get_env_float_list("CATEGORY_WEIGHTS")
    images = [data["images"] for data in all_input_data]
    pairs = list(itertools.combinations(enumerate(images), 2))
    problem = all_input_data[0]["problem"]
    payload = [
        {
            "images": [
                _encode_image(_load_image(img1)),
                _encode_image(_load_image(img2)),
            ],
            "problem": problem,
            "first_index": idx1,
            "second_index": idx2,
        }
        for (idx1, img1), (idx2, img2) in pairs
    ]

    responses = evaluate_batch(payload, api_url=api_url)
    return _pairwise_win_rate(
        num_items=len(images),
        responses=responses,
        overall_weight=overall_weight,
        dim_weight=dim_weight,
        category_weights=category_weights,
        device=device,
    )
